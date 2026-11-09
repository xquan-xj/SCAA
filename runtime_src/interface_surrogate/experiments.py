"""Experiment identity and final-test protocol guards. No model selection on test."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path,value,*,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x" if exclusive else "w",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def source_hash():
    root=Path(__file__).parent
    return fingerprint({p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob("*.py"))})


def environment_info():
    result={"python":platform.python_version(),"platform":platform.platform(),
            "torch":str(torch.__version__),"numpy":np.__version__,"scipy":scipy.__version__,
            "scikit_learn":sklearn.__version__,"cuda_runtime":torch.version.cuda,
            "cuda_available":torch.cuda.is_available()}
    if torch.cuda.is_available():
        prop=torch.cuda.get_device_properties(0)
        result["gpu"]={"name":prop.name,"memory_GiB":prop.total_memory/1024**3}
    return result


def create_run(runs_root,config,dataset_fingerprint,split,*,run_id=None):
    name=str(config.get("name",config.get("model",{}).get("kind","experiment")))
    name=re.sub(r"[^A-Za-z0-9_-]+","_",name)[:48]
    if run_id is None:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id=f"{stamp}_{name}_s{config.get('seed',0)}_{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):
        raise ValueError("Run ID must use letters, numbers, underscores and hyphens")
    run=Path(runs_root)/run_id
    run.mkdir(parents=True,exist_ok=False)
    manifest={"run_id":run_id,"config":config,"config_hash":fingerprint(config),
              "dataset_fingerprint":dataset_fingerprint,"split":split,"split_hash":fingerprint(split),
              "source_hash":source_hash(),"environment":environment_info(),
              "status":"created","checkpoint_file":None,
              "metric_version":"polar_local_score_v2_with_original_nodes"}
    write_json(run/"manifest.json",manifest,exclusive=True)
    return run


def freeze_protocol(run_dirs,output):
    if not run_dirs:
        raise ValueError("At least one completed run is required")
    entries=[];seen=set()
    for value in run_dirs:
        run=Path(value);m=read_json(run/"manifest.json")
        if m.get('status')!='completed':
            raise ValueError('Only completed runs may enter a final-test protocol')
        checkpoint=m.get("checkpoint_file")
        if not checkpoint or not (run/checkpoint).is_file():
            raise ValueError(f"Run {run.name} has no fitted checkpoint")
        if m["run_id"] in seen:
            raise ValueError("Duplicate run in final-test lock")
        seen.add(m["run_id"])
        entries.append({k:m[k] for k in ["run_id","config_hash","dataset_fingerprint","split_hash","source_hash","metric_version"]})
        entries[-1]["checkpoint_sha256"]=file_hash(run/checkpoint)
        entries[-1]["manifest_sha256"]=file_hash(run/"manifest.json")
    lock={"protocol_version":1,"purpose":"Predeclared final test comparison; do not select models using test outcomes",
          "created_utc":datetime.now(timezone.utc).isoformat(),"runs":entries}
    write_json(output,lock,exclusive=True)
    return lock


def check_test_lock(lock_path,run_dir):
    if lock_path is None or not Path(lock_path).is_file():
        raise ValueError("Final test evaluation requires a frozen protocol lock")
    run=Path(run_dir);m=read_json(run/"manifest.json")
    matches=[e for e in read_json(lock_path)["runs"] if e["run_id"]==m["run_id"]]
    if len(matches)!=1:
        raise ValueError("Run is not listed exactly once in the test lock")
    entry=matches[0]
    if not m.get("checkpoint_file") or file_hash(run/m["checkpoint_file"])!=entry["checkpoint_sha256"]:
        raise ValueError("The checkpoint differs from the final-test lock")
    if file_hash(run/"manifest.json")!=entry.get("manifest_sha256"):
        raise ValueError("The run manifest metadata differs from the final-test lock")
    for key in ["config_hash","dataset_fingerprint","split_hash","source_hash","metric_version"]:
        if entry[key]!=m[key]:
            raise ValueError(f"Final-test lock mismatch: {key}")
    if fingerprint(m["config"])!=m["config_hash"] or fingerprint(m["split"])!=m["split_hash"]:
        raise ValueError("Run configuration or split was changed after training")
    if source_hash()!=m["source_hash"]:
        raise ValueError("Research source changed after the run; revalidate and create a new run")
    return entry
