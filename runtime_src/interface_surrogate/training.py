"""Fresh supervised runs: train updates, validation selects, test is untouched."""
from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MorphologyDataset,fit_normalization,verify_prepared_dataset,parameter_names
from .experiments import create_run,read_json,write_json
from .metrics import MorphologyLoss
from .models import build_model
from .settings import resolve_config
from .splits import validate_split


def seed_everything(seed,deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def train_experiment(processed_dir,split,config,runs_root,*,device=None,run_id=None):
    data=Path(processed_dir).resolve();meta=verify_prepared_dataset(data)
    validate_split(split,[v["id"] for v in meta["samples"]])
    if split.get("dataset_fingerprint",meta["dataset_fingerprint"])!=meta["dataset_fingerprint"]:
        raise ValueError("Split belongs to a different dataset")
    if split.get("prepared_metadata_sha256",meta["prepared_metadata_sha256"])!=meta["prepared_metadata_sha256"]:
        raise ValueError("Split belongs to different prepared metadata")
    config=resolve_config(config);t=config["training"]
    seed_everything(config["seed"],t["deterministic"])
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    norm=fit_normalization(data,split["train"])
    shape=(len(meta["grid"]["theta_rad"]),len(meta["grid"]["r_mm"]))
    names=parameter_names(meta)
    model=build_model(config["model"],input_channels=len(names)+2,
                      parameter_dim=len(names),output_size=shape).to(dev)
    kind="parameters" if config["model"]["kind"]=="parameter_mlp" else "field"
    train=MorphologyDataset(data,split["train"],norm,input_kind=kind)
    val=MorphologyDataset(data,split["val"],norm,input_kind=kind)
    generator=torch.Generator().manual_seed(config["seed"])
    train_loader=DataLoader(train,batch_size=t["batch_size"],shuffle=True,generator=generator,num_workers=t["num_workers"])
    val_loader=DataLoader(val,batch_size=t["batch_size"],shuffle=False,num_workers=t["num_workers"])
    criterion=MorphologyLoss(norm,meta["grid"]["r_mm"],meta["grid"]["theta_rad"],**config["loss"],**config["metrics"]).to(dev)
    opt_class=getattr(torch.optim,t["optimizer"])
    optimizer=opt_class(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=t["scheduler_patience"])
    amp=bool(t["amp"] and dev.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    run=create_run(runs_root,config,meta["dataset_fingerprint"],split,run_id=run_id)
    manifest=read_json(run/"manifest.json")
    manifest.update(data_dir=os.path.relpath(data,run),normalization=norm,status="training",parameter_names=names,
                    prepared_metadata_sha256=meta["prepared_metadata_sha256"],
                    grid=meta["grid"],checkpoint_file="best.pth",parameter_count=sum(p.numel() for p in model.parameters()),
                    trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    device=str(dev),actual_amp=amp,optimizer_class=type(optimizer).__name__,
                    loss_class=type(criterion).__name__,label_spec=meta["build_options"])
    write_json(run/"manifest.json",manifest)
    best=float("inf");bad_epochs=0;start=time.perf_counter();history=[]
    if dev.type=="cuda":torch.cuda.reset_peak_memory_stats(dev)
    try:
        for epoch in range(1,t["epochs"]+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            train_sum=0.;accum_samples=0;accum_batches=0
            for step,batch in enumerate(train_loader):
                x=batch["input"].to(dev);y=batch["target"].to(dev);n=len(x)
                with torch.autocast(device_type=dev.type,enabled=amp):
                    loss=criterion(model(x),y)
                if not torch.isfinite(loss):raise ValueError("Training loss became non-finite")
                train_sum+=float(loss.detach())*n
                scaler.scale(loss*n).backward();accum_samples+=n;accum_batches+=1
                if accum_batches==t["gradient_accumulation"] or step+1==len(train_loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:parameter.grad.div_(accum_samples)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),t["gradient_clip"])
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    accum_samples=0;accum_batches=0
            model.eval();val_sum=0.
            with torch.inference_mode():
                for batch in val_loader:
                    x=batch["input"].to(dev);y=batch["target"].to(dev)
                    with torch.autocast(device_type=dev.type,enabled=amp):
                        vl=criterion(model(x),y)
                    val_sum+=float(vl)*len(x)
            val_loss=val_sum/len(val)
            if not np.isfinite(val_loss):raise ValueError("Validation loss became non-finite")
            improved=val_loss<best-t["min_delta"]
            if improved:
                best=val_loss;bad_epochs=0;manifest["best_epoch"]=epoch
                torch.save({"model":model.state_dict(),"model_config":config["model"],"epoch":epoch,
                            "val_loss":best,"normalization":norm},run/"best.pth")
            else:bad_epochs+=1
            scheduler.step(val_loss)
            row={"epoch":epoch,"train_loss":train_sum/len(train),"val_loss":val_loss,
                 "learning_rate":optimizer.param_groups[0]["lr"],"best_epoch":manifest["best_epoch"]}
            history.append(row)
            with (run/"training.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerows(history)
            print(f"{run.name}: epoch {epoch}, train={row['train_loss']:.6g}, val={val_loss:.6g}",flush=True)
            if bad_epochs>=t["early_stopping_patience"]:break
        manifest.update(status="completed",best_val_loss=best,epochs_completed=len(history),
                        elapsed_seconds=time.perf_counter()-start,
                        peak_cuda_memory_MiB=None if dev.type!="cuda" else torch.cuda.max_memory_allocated(dev)/1024**2)
    except Exception as exc:
        manifest.update(status="failed",error=repr(exc));raise
    finally:write_json(run/"manifest.json",manifest)
    return run
