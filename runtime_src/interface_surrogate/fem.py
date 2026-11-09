"""Interfaces for real FEM evidence; no contact solver or synthetic results."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr,spearmanr

from .data import verify_prepared_dataset,parameter_names
from .experiments import write_json


def compare_pressures(prediction,reference,*,effective_threshold_MPa):
    """Compare matching FE nodes using their supplied tributary surface areas."""
    for table in [prediction,reference]:
        if any(k not in table for k in ["node_id","pressure_MPa","area_mm2"]):
            raise ValueError("Pressure tables require node_id, pressure_MPa and area_mm2")
        ids=np.asarray(table["node_id"])
        if len(np.unique(ids))!=len(ids):
            raise ValueError("Duplicate FE node IDs")
    ids=np.asarray(reference["node_id"]);pred_ids=np.asarray(prediction["node_id"])
    if set(ids)!=set(pred_ids):
        raise ValueError("Predicted/reference FE node sets differ")
    order={v:i for i,v in enumerate(pred_ids)}
    indices=[order[v] for v in ids]
    p=np.asarray(prediction["pressure_MPa"],dtype=float)[indices]
    t=np.asarray(reference["pressure_MPa"],dtype=float)
    area=np.asarray(reference["area_mm2"],dtype=float)
    pa=np.asarray(prediction["area_mm2"],dtype=float)[indices]
    if not all(np.isfinite(x).all() for x in [p,t,area,pa]) or (area<=0).any():
        raise ValueError("Pressure and positive FE surface areas must be finite")
    if not np.allclose(area,pa,rtol=1e-6,atol=1e-12):
        raise ValueError("FE nodal surface areas differ; use a documented conservative projection")
    w=area/area.sum();mp=float((w*p).sum());mt=float((w*t).sum())
    vp=float((w*(p-mp)**2).sum());vt=float((w*(t-mt)**2).sum())
    return {"n_nodes":len(ids),"pressure_rmse_MPa":float(np.sqrt((w*(p-t)**2).sum())),
            "peak_pressure_error_MPa":float(p.max()-t.max()),
            "pressure_variance_error_MPa2":vp-vt,
            "effective_area_error_mm2":float(area[p>effective_threshold_MPa].sum()-area[t>effective_threshold_MPa].sum()),
            "effective_threshold_MPa":float(effective_threshold_MPa)}


def read_pressure_csv(path):
    rows=list(csv.DictReader(Path(path).open(encoding="utf-8-sig",newline="")))
    if not rows:
        raise ValueError("Pressure table is empty")
    return {"node_id":np.array([int(r["node_id"]) for r in rows]),
            "pressure_MPa":np.array([float(r["pressure_MPa"]) for r in rows]),
            "area_mm2":np.array([float(r["area_mm2"]) for r in rows])}


def export_case(processed_dir,prediction_npz,output_dir):
    """Validate a bound prediction batch, then export every original node row."""
    data=Path(processed_dir);output=Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("FEM case output is not empty")
    metadata=verify_prepared_dataset(data)
    with np.load(prediction_npz,allow_pickle=False) as p:
        for field in ["dataset_fingerprint","prepared_metadata_sha256"]:
            if field not in p.files:
                raise ValueError(f"Prediction source is missing {field}")
            value=p[field]
            if value.shape!=() or value.dtype.kind!="U":
                raise ValueError(f"Prediction source {field} must be a scalar Unicode field")
            if value.item()!=metadata[field]:
                raise ValueError(f"Prediction source {field} does not match the verified dataset")
        if "ids" not in p.files or "prediction_um" not in p.files:
            raise ValueError("Predictions require ids and prediction_um")
        id_array=p["ids"]
        if id_array.ndim!=1 or id_array.dtype.kind!="U" or not len(id_array):
            raise ValueError("Prediction IDs must be a nonempty one-dimensional Unicode array")
        ids=id_array.tolist();preds=p["prediction_um"]
    if any(not value for value in ids) or len(ids)!=len(set(ids)):
        raise ValueError("Prediction IDs must be nonempty and unique")
    samples={str(v["id"]):v for v in metadata["samples"]}
    if any(sample_id not in samples for sample_id in ids):
        raise ValueError("Prediction has an unknown sample ID")
    theta=np.asarray(metadata["grid"]["theta_rad"]);r=np.asarray(metadata["grid"]["r_mm"])
    shape=(len(theta),len(r))
    if (preds.shape!=(len(ids),*shape) or not np.issubdtype(preds.dtype,np.number)
            or np.iscomplexobj(preds) or not np.isfinite(preds).all()):
        raise ValueError(f"Prediction shape must be exactly {(len(ids),*shape)} with finite real values")
    xy=np.load(data/metadata["nodes_xy_mm"],allow_pickle=False)
    if (xy.ndim!=2 or xy.shape[1]!=2 or not len(xy) or not np.issubdtype(xy.dtype,np.number)
            or np.iscomplexobj(xy) or not np.isfinite(xy).all()):
        raise ValueError("Original node coordinates must have finite real shape (n_nodes,2)")
    coords=np.column_stack([np.arctan2(xy[:,1],xy[:,0]),np.sqrt((xy**2).sum(1))])
    limits=np.array([[theta[0],theta[-1]],[r[0],r[-1]]])
    tolerance=1e-10*np.maximum(1.,np.max(np.abs(limits),axis=1))
    if ((coords<limits[:,0]-tolerance).any() or (coords>limits[:,1]+tolerance).any()
            or not np.isfinite(coords).all()):
        raise ValueError("Original nodes lie outside the prepared grid")
    # Floating roundoff at extrema is clipped, not physical extrapolation.
    coords[:,0]=np.clip(coords[:,0],theta[0],theta[-1]);coords[:,1]=np.clip(coords[:,1],r[0],r[-1])
    base=float(metadata.get("build_options",{}).get("base_z_mm",-2.))
    if not np.isfinite(base):
        raise ValueError("Base node elevation must be finite")
    pending=[]
    for index,sample_id in enumerate(ids):
        sample=samples[sample_id]
        with np.load(data/sample["npz"],allow_pickle=False) as z:
            if "node_target_um" not in z.files or "target_um" not in z.files:
                raise ValueError("Reference labels require node_target_um and target_um")
            target=z["node_target_um"];grid_target=z["target_um"]
        for label,expected in [(target,(len(xy),)),(grid_target,shape)]:
            if (label.shape!=expected or not np.issubdtype(label.dtype,np.number)
                    or np.iscomplexobj(label) or not np.isfinite(label).all()):
                raise ValueError(f"Reference labels must have finite real shape {expected}")
        if not np.isfinite([float(sample["preload_N"]),float(sample["pressure_MPa"])]).all():
            raise ValueError("Sample conditions must be finite")
        interpolator=RegularGridInterpolator((theta,r),preds[index],bounds_error=True)
        z_pred=interpolator(coords)
        if z_pred.shape!=(len(xy),) or not np.isfinite(z_pred).all():
            raise ValueError("Interpolated node predictions must be finite with one value per node")
        fname=f"sample_{sample_id}_nodes.csv"
        if Path(fname).name!=fname or ":" in fname:
            raise ValueError("Sample ID cannot be represented by a safe node export filename")
        pending.append((sample,fname,z_pred,target))
    # All dataset, prediction, node, and label validation precedes any output.
    output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for sample,fname,z_pred,target in pending:
        with (output/fname).open("x",encoding="utf-8",newline="") as f:
            w=csv.writer(f);w.writerow(["node_row","x_mm","y_mm","predicted_z_mm","reference_z_mm","predicted_adjustment_um","reference_adjustment_um"])
            for index in range(len(xy)):
                x,y=xy[index];zp=z_pred[index];zt=target[index]
                w.writerow([index+1,x,y,base+zp/1000,base+zt/1000,zp,zt])
        cases.append({"id":sample["id"],**{name:sample[name] for name in parameter_names(metadata)},
                      "nodes_file":fname,"status":"awaiting_real_FE_solve"})
    write_json(output/"cases.json",{"dataset_fingerprint":metadata["dataset_fingerprint"],
               "prepared_metadata_sha256":metadata["prepared_metadata_sha256"],
               "label_policy":metadata['build_options']['label_policy'],"node_identity":"row order; map to actual FE node IDs before solving", "cases":cases})
    return cases


def calibrate_local_threshold(scores,mechanics_pass,*,stage,min_samples=10,max_false_accept_fraction=0.):
    if stage!="val":
        raise ValueError("Threshold calibration requires validation-stage mechanics results")
    x=np.asarray(scores,dtype=float);passed=np.asarray(mechanics_pass,dtype=bool)
    if len(x)!=len(passed) or len(x)<min_samples or not np.isfinite(x).all() or (x<0).any():
        raise ValueError("Insufficient or invalid paired validation samples")
    if not 0<=max_false_accept_fraction<=1:
        raise ValueError("Invalid false acceptance bound")
    result={"threshold":None,"accepted_count":0,"false_accept_count":0,"validation_count":len(x),
            "max_false_accept_fraction":max_false_accept_fraction,
            "evidence":"empirical validation calibration; no out-of-sample guarantee"}
    for threshold in sorted(np.unique(x)):
        accept=x<=threshold;n=int(accept.sum());fp=int((accept&~passed).sum())
        if fp/n<=max_false_accept_fraction:
            result.update(threshold=float(threshold),accepted_count=n,false_accept_count=fp)
    return result


def metric_mechanics_correlations(rows,mechanics_key="pressure_rmse_MPa"):
    """One paired row per case; positive, negative or absent associations are allowed."""
    if len(rows)<3:
        raise ValueError("At least three paired cases are required")
    case_ids=[]
    for row in rows:
        identity=row.get("case_id",row.get("id"))
        if identity is None or not str(identity).strip():
            raise ValueError("Every correlation row requires a nonempty case id")
        case_ids.append(str(identity).strip())
    if len(case_ids)!=len(set(case_ids)):
        raise ValueError("Duplicate case ids cannot be pooled in correlations")
    group={}
    for key in ["run_id","method","method_id","model_kind"]:
        if any(key in row for row in rows):
            values=[row.get(key) for row in rows]
            if any(value is None or not str(value).strip() for value in values) or len({str(value) for value in values})!=1:
                raise ValueError(f"Correlations require one complete {key} group; split mixed runs/methods first")
            group[key]=str(values[0])
    result={"n_cases":len(rows),"mechanics_key":mechanics_key,"metrics":{}}
    result.update(case_ids=case_ids,group=group)
    y=np.array([float(r[mechanics_key]) for r in rows])
    for key in ["rmse_um","max_abs_um","r2","local_score_v2"]:
        if any(r.get(key) is None for r in rows):
            continue
        x=np.array([float(r[key]) for r in rows])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Correlation inputs must be finite paired observations")
        if np.std(x)==0 or np.std(y)==0:
            result["metrics"][key]={"pearson_r":None,"spearman_rho":None}
        else:
            pr=pearsonr(x,y);sr=spearmanr(x,y)
            result["metrics"][key]={"pearson_r":float(pr.statistic),"pearson_p":float(pr.pvalue),
                                      "spearman_rho":float(sr.statistic),"spearman_p":float(sr.pvalue)}
    return result
