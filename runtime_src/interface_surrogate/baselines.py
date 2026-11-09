"""Parameter-space interpolation and train-only POD/regression baselines."""
from __future__ import annotations

import csv
from itertools import combinations
import os
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator,RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,WhiteKernel
import torch
from torch import nn

from .data import fit_normalization,verify_prepared_dataset,parameter_names,validate_parameter_domain
from .experiments import create_run,read_json,write_json
from .settings import resolve_config
from .splits import validate_split


BASELINE_KINDS={"linear_interpolation","rbf","pod_ridge","bilinear","pod_gpr","pod_mlp"}


def _polynomial(x):
    return np.column_stack([np.ones(len(x)),x[:,0],x[:,1],x[:,0]**2,x[:,0]*x[:,1],x[:,1]**2])


def _load_rows(data,lookup,ids,norm,shape):
    parameters=[];targets=[]
    for sample_id in ids:
        with np.load(data/lookup[sample_id]["npz"],allow_pickle=False) as sample:
            x=np.asarray(sample["parameters"],dtype=float)
            y=np.asarray(sample["target_um"],dtype=float)
        if x.shape!=(len(norm['parameter_mean']),) or y.shape!=shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid baseline training/validation arrays for {sample_id}")
        validate_parameter_domain(x,norm)
        parameters.append(x);targets.append(y.ravel())
    x=(np.stack(parameters)-np.array(norm["parameter_mean"]))/np.array(norm["parameter_scale"])
    x=x[:,norm['active_parameter_indices']]
    y=(np.stack(targets)-norm["target_offset_um"])/norm["target_scale_um"]
    return x,y


def _fit_pod(y,count,seed):
    if not 1<=count<=min(y.shape[0]-1,y.shape[1]):
        raise ValueError("POD component count exceeds the training-data rank bound")
    pca=PCA(n_components=count,svd_solver="randomized",random_state=seed)
    coefficients=pca.fit_transform(y)
    coefficient_mean=coefficients.mean(axis=0)
    coefficient_scale=coefficients.std(axis=0)
    threshold=np.finfo(float).eps*max(1.,float(np.max(np.abs(coefficients))))
    coefficient_scale[np.ptp(coefficients,axis=0)<=threshold]=1.
    arrays={"pca_mean":pca.mean_,"pca_components":pca.components_,
            "explained_variance_ratio":np.nan_to_num(pca.explained_variance_ratio_,nan=0.),
            "train_coefficients":coefficients,"coefficient_mean":coefficient_mean,
            "coefficient_scale":coefficient_scale,
            "train_coefficients_scaled":(coefficients-coefficient_mean)/coefficient_scale}
    return arrays


def _fit_coefficient_mlp(x,y,val_x,val_y,config,run,manifest):
    """CPU coefficient MSE; validation selects weights without gradient updates."""
    settings=config["training"];hidden=config["model"].get("hidden",[128,128])
    if (not isinstance(hidden,(list,tuple)) or not hidden
            or any(isinstance(width,bool) or not isinstance(width,int) or width<1 for width in hidden)):
        raise ValueError("POD-MLP hidden widths must be positive integers")
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(settings["deterministic"])
    layers=[];widths=[x.shape[1],*hidden,y.shape[1]]
    for index,(left,right) in enumerate(zip(widths[:-1],widths[1:])):
        layers.append(nn.Linear(left,right))
        if index<len(widths)-2:layers.append(nn.ReLU())
    model=nn.Sequential(*layers)
    optimizer=getattr(torch.optim,settings["optimizer"])(model.parameters(),lr=settings["learning_rate"],
                                                         weight_decay=settings["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,
                                                        patience=settings["scheduler_patience"])
    tx=torch.tensor(x,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32)
    vx=torch.tensor(val_x,dtype=torch.float32);vy=torch.tensor(val_y,dtype=torch.float32)
    generator=torch.Generator().manual_seed(config["seed"])
    best=float("inf");bad_epochs=0;best_arrays=None;history=[];start=time.perf_counter()
    for epoch in range(1,settings["epochs"]+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        order=torch.randperm(len(tx),generator=generator);total=0.;accum_samples=0;accum_batches=0
        batches=list(order.split(settings["batch_size"]))
        for step,indices in enumerate(batches):
            loss=nn.functional.mse_loss(model(tx[indices]),ty[indices])
            if not torch.isfinite(loss):raise ValueError("POD-MLP training loss became non-finite")
            count=len(indices);total+=float(loss.detach())*count
            (loss*count).backward();accum_samples+=count;accum_batches+=1
            if accum_batches==settings["gradient_accumulation"] or step+1==len(batches):
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad.div_(accum_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip"])
                optimizer.step();optimizer.zero_grad(set_to_none=True);accum_samples=0;accum_batches=0
        model.eval()
        with torch.inference_mode():
            val_loss=float(nn.functional.mse_loss(model(vx),vy))
        if not np.isfinite(val_loss):raise ValueError("POD-MLP validation loss became non-finite")
        if val_loss<best-settings["min_delta"]:
            best=val_loss;bad_epochs=0;manifest["best_epoch"]=epoch
            linear=[layer for layer in model if isinstance(layer,nn.Linear)]
            best_arrays={"mlp_layer_count":np.array(len(linear),dtype=np.int64)}
            for index,layer in enumerate(linear):
                best_arrays[f"mlp_weight_{index}"]=layer.weight.detach().numpy().copy()
                best_arrays[f"mlp_bias_{index}"]=layer.bias.detach().numpy().copy()
        else:bad_epochs+=1
        scheduler.step(val_loss)
        row={"epoch":epoch,"train_loss":total/len(tx),"val_loss":val_loss,
             "learning_rate":optimizer.param_groups[0]["lr"],"best_epoch":manifest["best_epoch"]}
        history.append(row)
        with (run/"training.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=row);writer.writeheader();writer.writerows(history)
        if bad_epochs>=settings["early_stopping_patience"]:break
    manifest.update(best_val_loss=best,epochs_completed=len(history),elapsed_seconds=time.perf_counter()-start,
                    parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                    trainable_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                    optimizer_class=type(optimizer).__name__,device="cpu",actual_amp=False,actual_num_workers=0)
    return best_arrays


def fit_baseline(processed_dir,split,config,runs_root):
    data=Path(processed_dir).resolve();meta=verify_prepared_dataset(data)
    validate_split(split,[r["id"] for r in meta["samples"]]);config=resolve_config(config)
    if split.get("dataset_fingerprint",meta["dataset_fingerprint"])!=meta["dataset_fingerprint"]:
        raise ValueError("Split belongs to a different dataset")
    if split.get("prepared_metadata_sha256",meta["prepared_metadata_sha256"])!=meta["prepared_metadata_sha256"]:
        raise ValueError("Split belongs to different prepared metadata")
    spec=config["model"];kind=spec["kind"]
    if kind not in BASELINE_KINDS:raise ValueError("Unknown simple baseline")
    norm=fit_normalization(data,split["train"]);lookup={r["id"]:r for r in meta["samples"]}
    names=parameter_names(meta);active=norm['active_parameter_indices']
    if not active:
        raise ValueError('Numerical baselines require at least one varying training parameter')
    if kind in {'bilinear','linear_interpolation','pod_ridge'} and len(active)!=2:
        raise ValueError('This baseline requires exactly two varying training parameter dimensions')
    shape=(len(meta["grid"]["theta_rad"]),len(meta["grid"]["r_mm"]))
    x,y=_load_rows(data,lookup,split["train"],norm,shape)
    arrays={"train_parameters":x,"train_targets":y}
    if kind in {"pod_ridge","pod_gpr","pod_mlp"}:
        arrays.update(_fit_pod(y,int(spec.get("pod_components",16)),config["seed"]))
    if kind=="pod_ridge":
        coeff=arrays["train_coefficients"];phi=_polynomial(x)
        alpha=float(spec.get("ridge_alpha",.001))
        if alpha<=0:raise ValueError("ridge_alpha must be positive")
        penalty=np.eye(phi.shape[1])*alpha;penalty[0,0]=0
        regression=np.linalg.solve(phi.T@phi+penalty,phi.T@coeff)
        arrays["regression"]=regression
    elif kind=="pod_gpr":
        length=float(spec.get("gpr_length_scale",1.));noise=float(spec.get("gpr_noise",1e-6))
        if not np.isfinite([length,noise]).all() or min(length,noise)<=0:
            raise ValueError("GPR length scale and noise must be finite and positive")
        kernel=ConstantKernel(1.,"fixed")*RBF(length,"fixed")+WhiteKernel(noise,"fixed")
        gpr=GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=False,alpha=0.)
        gpr.fit(x,arrays["train_coefficients_scaled"])
        arrays.update(gpr_alpha=gpr.alpha_,gpr_cholesky=gpr.L_,
                      gpr_length_scale=np.array(length),gpr_noise=np.array(noise),gpr_constant=np.array(1.))
    elif kind in {"linear_interpolation","rbf"} and (len(x)<len(active)+1 or np.linalg.matrix_rank(x-x.mean(0))<len(active)):
        raise ValueError("Interpolation needs a full-rank training parameter domain")
    elif kind=="bilinear" and len(np.unique(x,axis=0))!=len(x):
        raise ValueError("Bilinear interpolation requires unique training parameter pairs")
    run=create_run(runs_root,config,meta["dataset_fingerprint"],split)
    manifest=read_json(run/"manifest.json")
    manifest.update(status="fitting",checkpoint_file="baseline.npz",normalization=norm,
                    parameter_names=names,active_parameter_indices=active,
                    constant_parameter_names=[name for i,name in enumerate(names) if i not in active],
                    prepared_metadata_sha256=meta["prepared_metadata_sha256"],
                    data_dir=os.path.relpath(data,run),grid=meta["grid"],label_spec=meta["build_options"],
                    fitted_train_ids=split["train"])
    if kind=="bilinear":
        manifest["interpolation_policy"]=("Smallest-area enclosing rectangle with four training-only corners; "
            "exact training hits returned directly; missing rectangle/out-of-domain uses nearest training "
            "point in standardized parameter coordinates and reports nearest_fallback=True")
    elif kind=="pod_gpr":
        manifest.update(gpr_optimizer=None,gpr_kernel="Constant(1)*RBF+fixed WhiteKernel",
                        objective="fixed_kernel_training_POD_coefficient_GPR",
                        coefficient_normalization="training-only per-component population mean/std")
    elif kind=="pod_mlp":
        manifest.update(status="training",selection_ids=split["val"],
                        objective="standardized_POD_coefficient_MSE",
                        objective_note="Training and validation use standardized POD coefficient MSE; "
                        "this differs from the configured physical field loss, which is not used.",
                        configured_field_loss_not_used=config["loss"],
                        coefficient_normalization="training-only per-component population mean/std")
    write_json(run/"manifest.json",manifest)
    try:
        if kind=="pod_mlp":
            val_x,val_y=_load_rows(data,lookup,split["val"],norm,shape)
            val_coeff=(val_y-arrays["pca_mean"])@arrays["pca_components"].T
            val_coeff=(val_coeff-arrays["coefficient_mean"])/arrays["coefficient_scale"]
            arrays.update(_fit_coefficient_mlp(x,arrays["train_coefficients_scaled"],val_x,val_coeff,
                                              config,run,manifest))
        np.savez_compressed(run/"baseline.npz",**arrays)
        manifest["status"]="completed"
    except Exception as exc:
        manifest.update(status="failed",error=repr(exc));raise
    finally:write_json(run/"manifest.json",manifest)
    return run


def _predict_bilinear(train_x,train_y,queries):
    """Enumerate only complete training rectangles; never triangulate or fill holes."""
    lookup={tuple(point):index for index,point in enumerate(train_x)}
    rows={force:set(train_x[train_x[:,0]==force,1]) for force in np.unique(train_x[:,0])}
    rectangles=[];corners=[];areas=[]
    for lower_f,upper_f in combinations(sorted(rows),2):
        for lower_p,upper_p in combinations(sorted(rows[lower_f]&rows[upper_f]),2):
            rectangles.append((lower_f,upper_f,lower_p,upper_p))
            corners.append([lookup[(lower_f,lower_p)],lookup[(upper_f,lower_p)],
                            lookup[(lower_f,upper_p)],lookup[(upper_f,upper_p)]])
            areas.append((upper_f-lower_f)*(upper_p-lower_p))
    if rectangles:
        order=np.argsort(areas,kind="stable")
        bounds=np.asarray(rectangles)[order];corner_indices=np.asarray(corners)[order]
    else:bounds=np.empty((0,4));corner_indices=np.empty((0,4),dtype=int)
    values=[];fallback=[];cache={}
    for point in queries:
        key=tuple(point)
        if key not in cache:
            if key in lookup:
                cache[key]=(np.array([lookup[key]]),np.ones(1),False)
            else:
                enclosed=(bounds[:,0]<=point[0])&(point[0]<=bounds[:,1])&(bounds[:,2]<=point[1])&(point[1]<=bounds[:,3])
                found=np.flatnonzero(enclosed)
                if len(found):
                    rectangle=found[0];f0,f1,p0,p1=bounds[rectangle]
                    u=(point[0]-f0)/(f1-f0);v=(point[1]-p0)/(p1-p0)
                    weights=np.array([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v])
                    cache[key]=(corner_indices[rectangle],weights,False)
                else:
                    nearest=int(np.argmin(((train_x-point)**2).sum(axis=1)))
                    cache[key]=(np.array([nearest]),np.ones(1),True)
        indices,weights,used_nearest=cache[key]
        values.append(weights@train_y[indices]);fallback.append(used_nearest)
    return np.stack(values),np.asarray(fallback,dtype=bool)


def predict_baseline(run_dir,parameters):
    run=Path(run_dir);manifest=read_json(run/"manifest.json")
    spec=manifest["config"]["model"];norm=manifest["normalization"]
    if manifest.get("status")!="completed":raise ValueError("Only completed baseline runs can predict")
    raw=np.asarray(parameters,dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=len(norm['parameter_mean']) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Prediction parameters must match the declared finite (N,parameters) array")
    validate_parameter_domain(raw,norm)
    x=(raw-np.asarray(norm["parameter_mean"]))/np.asarray(norm["parameter_scale"])
    x=x[:,manifest.get('active_parameter_indices',list(range(raw.shape[1])))]
    with np.load(run/manifest["checkpoint_file"],allow_pickle=False) as z:
        if spec["kind"]=="pod_ridge":
            values=(_polynomial(x)@z["regression"])@z["pca_components"]+z["pca_mean"]
            outside=np.zeros(len(x),dtype=bool)
        elif spec["kind"] in {"pod_gpr","pod_mlp"}:
            if spec["kind"]=="pod_gpr":
                squared=((x[:,None,:]-z["train_parameters"][None,:,:])**2).sum(axis=2)
                kernel=float(z["gpr_constant"])*np.exp(-.5*squared/float(z["gpr_length_scale"])**2)
                coefficients=kernel@z["gpr_alpha"]
            else:
                coefficients=np.asarray(x,dtype=np.float32)
                count=int(z["mlp_layer_count"])
                for index in range(count):
                    coefficients=coefficients@z[f"mlp_weight_{index}"].T+z[f"mlp_bias_{index}"]
                    if index+1<count:coefficients=np.maximum(coefficients,0.)
            coefficients=coefficients*z["coefficient_scale"]+z["coefficient_mean"]
            values=coefficients@z["pca_components"]+z["pca_mean"]
            outside=np.zeros(len(x),dtype=bool)
        elif spec["kind"]=="bilinear":
            values,outside=_predict_bilinear(z["train_parameters"],z["train_targets"],x)
        elif spec["kind"]=="rbf":
            neighbors=min(int(spec.get("rbf_neighbors",24)),len(z["train_parameters"]))
            rbf=RBFInterpolator(z["train_parameters"],z["train_targets"],kernel="thin_plate_spline",
                                neighbors=neighbors,smoothing=float(spec.get("rbf_smoothing",0.)))
            values=rbf(x);outside=np.zeros(len(x),dtype=bool)
        elif spec["kind"]=="linear_interpolation":
            values=LinearNDInterpolator(z["train_parameters"],z["train_targets"])(x)
            outside=~np.isfinite(values).all(axis=1)
            if outside.any():
                values[outside]=NearestNDInterpolator(z["train_parameters"],z["train_targets"])(x[outside])
        else:raise ValueError("Unknown fitted baseline kind")
    if not np.isfinite(values).all():raise ValueError("Baseline produced non-finite predictions")
    shape=(len(manifest["grid"]["theta_rad"]),len(manifest["grid"]["r_mm"]))
    physical=values.reshape(len(x),*shape)*norm["target_scale_um"]+norm["target_offset_um"]
    return physical,outside
