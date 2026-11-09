"""Per-sample physical evaluation. A final test always needs a frozen lock."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import BASELINE_KINDS,predict_baseline
from .data import MorphologyDataset,verify_prepared_dataset,parameter_names
from .experiments import check_test_lock,fingerprint,read_json,source_hash,write_json
from .metrics import geometry_metrics,summarize_metrics,node_geometry_metrics
from .models import build_model
from .splits import validate_split


def load_neural_model(run_dir,device="cpu"):
    run=Path(run_dir);m=read_json(run/"manifest.json")
    shape=(len(m["grid"]["theta_rad"]),len(m["grid"]["r_mm"]))
    names=m.get('parameter_names',m['normalization'].get('parameter_names',['preload_N','pressure_MPa']))
    model=build_model(m["config"]["model"],input_channels=len(names)+2,parameter_dim=len(names),output_size=shape)
    checkpoint=torch.load(run/m["checkpoint_file"],map_location="cpu",weights_only=True)
    if checkpoint["model_config"]!=m["config"]["model"]:
        raise ValueError("Checkpoint model configuration differs from the run manifest")
    if checkpoint["normalization"]!=m["normalization"]:
        raise ValueError("Checkpoint normalization differs from the run manifest")
    model.load_state_dict(checkpoint["model"],strict=True)
    return model.to(device).eval()


def evaluate_run(run_dir,*,stage="val",lock_path=None,processed_dir=None,device=None):
    if stage not in {"val","test"}:raise ValueError("Evaluation stage must be val or test")
    run=Path(run_dir).resolve();m=read_json(run/"manifest.json")
    if m["status"]!="completed":raise ValueError("Only completed runs can be evaluated")
    if stage=="test":check_test_lock(lock_path,run)
    if source_hash()!=m["source_hash"]:raise ValueError("Research source changed after the run")
    if fingerprint(m["config"])!=m["config_hash"] or fingerprint(m["split"])!=m["split_hash"]:
        raise ValueError("Run settings or split changed after training")
    data=Path(processed_dir).resolve() if processed_dir else (run/m["data_dir"]).resolve()
    meta=verify_prepared_dataset(data)
    if meta["dataset_fingerprint"]!=m["dataset_fingerprint"]:
        raise ValueError("Dataset differs from the trained run")
    if meta["prepared_metadata_sha256"]!=m.get("prepared_metadata_sha256"):
        raise ValueError("Prepared dataset metadata differs from the trained run")
    if meta["grid"]!=m["grid"] or meta["build_options"]!=m["label_spec"]:
        raise ValueError("Run grid or label specification differs from the dataset")
    validate_split(m["split"],[r["id"] for r in meta["samples"]])
    out=run/"evaluation"/stage
    if out.exists():raise FileExistsError("Evaluation already exists; do not overwrite scientific records")
    ids=m["split"][stage];lookup={v["id"]:v for v in meta["samples"]}
    model_kind=m["config"]["model"]["kind"]
    if model_kind in BASELINE_KINDS:
        parameters=[[lookup[k][name] for name in parameter_names(meta)] for k in ids]
        predictions,outside=predict_baseline(run,parameters)
    else:
        dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model=load_neural_model(run,dev)
        ds=MorphologyDataset(data,ids,m["normalization"],input_kind="parameters" if model_kind=="parameter_mlp" else "field")
        predictions=[]
        with torch.inference_mode():
            for batch in DataLoader(ds,batch_size=1,shuffle=False):
                pred=model(batch["input"].to(dev)).float().cpu().numpy()[0,0]
                predictions.append(pred*m["normalization"]["target_scale_um"]+m["normalization"]["target_offset_um"])
        predictions=np.stack(predictions);outside=np.zeros(len(ids),dtype=bool)
    rows=[]
    nodes_xy=np.load(data/meta['nodes_xy_mm'],allow_pickle=False)
    for i,sample_id in enumerate(ids):
        with np.load(data/lookup[sample_id]["npz"],allow_pickle=False) as z:
            metrics=geometry_metrics(predictions[i],z["target_um"],meta["grid"]["r_mm"],meta["grid"]["theta_rad"],**m["config"]["metrics"])
            metrics.update(node_geometry_metrics(predictions[i],z['target_um'],z['node_target_um'],
                           meta['grid']['r_mm'],meta['grid']['theta_rad'],nodes_xy))
        rows.append({"id":sample_id,"stage":stage,"run_id":m['run_id'],
                     **{name:lookup[sample_id][name] for name in parameter_names(meta)},
                     "selected_column_1based":lookup[sample_id]['selected_column_1based'],
                     **metrics,"interpolation_nearest_fallback":bool(outside[i])})
    summary=summarize_metrics(rows)
    summary.update(stage=stage,metric_version=m["metric_version"],dataset_fingerprint=meta["dataset_fingerprint"],
                   run_id=m['run_id'],split_hash=m['split_hash'],prepared_metadata_sha256=meta['prepared_metadata_sha256'],
                   nearest_fallback_count=int(outside.sum()),excluded_samples=0)
    out.mkdir(parents=True,exist_ok=False)
    with (out/"scores.csv").open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/"predictions.npz",ids=np.asarray(ids,dtype=str),prediction_um=predictions,
                        dataset_fingerprint=np.asarray(meta["dataset_fingerprint"]),
                        prepared_metadata_sha256=np.asarray(meta["prepared_metadata_sha256"]))
    write_json(out/"summary.json",summary,exclusive=True)
    return summary


def benchmark_run(run_dir,*,device=None,warmup=10,repeats=100):
    if warmup<1 or repeats<2:raise ValueError("Benchmark needs warmup>=1 and repeats>=2")
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model=load_neural_model(run_dir,dev);m=read_json(Path(run_dir)/"manifest.json")
    count=len(m.get('parameter_names',m['normalization'].get('parameter_names',['preload_N','pressure_MPa'])))
    shape=(1,count) if m["config"]["model"]["kind"]=="parameter_mlp" else (1,count+2,len(m["grid"]["theta_rad"]),len(m["grid"]["r_mm"]))
    x=torch.zeros(shape,device=dev);times=[]
    if dev.type=="cuda":torch.cuda.reset_peak_memory_stats(dev)
    with torch.inference_mode():
        for i in range(warmup+repeats):
            if dev.type=="cuda":torch.cuda.synchronize(dev)
            start=time.perf_counter();model(x)
            if dev.type=="cuda":torch.cuda.synchronize(dev)
            if i>=warmup:times.append((time.perf_counter()-start)*1000)
    return {"device":str(dev),"batch_size":1,"precision":"float32","warmup":warmup,"repeats":repeats,
            "latency_ms_median":float(np.median(times)),"latency_ms_p95":float(np.quantile(times,.95)),
            "scope":"model forward only; excludes preprocessing and nodal projection",
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "peak_cuda_memory_MiB":None if dev.type!="cuda" else torch.cuda.max_memory_allocated(dev)/1024**2}
