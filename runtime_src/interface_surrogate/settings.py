"""Resolved, serialized settings; never infer a loss from a folder name."""
from copy import deepcopy
import math

from .models.common import validate_controls

DEFAULTS={"name":"experiment","seed":8012,"model":{"kind":"unet"},
          "training":{"epochs":300,"batch_size":2,"gradient_accumulation":8,
                       "learning_rate":.0005,"weight_decay":.0001,"optimizer":"AdamW",
                       "early_stopping_patience":30,"scheduler_patience":10,
                       "min_delta":0.,"amp":True,"deterministic":False,"gradient_clip":1.,"num_workers":0},
          "loss":{"kind":"huber","beta_local":0.},
          "metrics":{"tolerance_um":.1,"length_mm":.1,"mape_epsilon_um":.001}}

MODEL_KEYS={"kind","width","depth","padding","fusion","axes","gated","content_attention",
            "cross_attention","query_source","key_source","attention_placement",
            "smoothing","smoothing_kernels","pool_factor","attention_heads","key_dim","hidden",
            "transformer_hidden","transformer_layers","transformer_heads","resnet_units",
            "transformer_mlp_ratio","transformer_dropout","transformer_attention_dropout",
            "decoder_channels","pod_components","ridge_alpha","rbf_neighbors","rbf_smoothing",
            "gpr_length_scale","gpr_noise"}

MODEL_DEFAULTS={
    "unet":{"width":32,"depth":4},
    "hybrid_transunet":{"transformer_hidden":768,"transformer_layers":12,"transformer_heads":12,
                        "resnet_units":[3,4,9],"transformer_mlp_ratio":4.,"transformer_dropout":.1,
                        "transformer_attention_dropout":0.,"decoder_channels":[256,128,64,16]},
    "parameter_mlp":{"hidden":[128,128]},
    "bilinear":{},
    "pod_gpr":{"pod_components":16,"gpr_length_scale":1.,"gpr_noise":1e-6},
    "pod_mlp":{"pod_components":16,"hidden":[128,128]},
    "linear_interpolation":{},"rbf":{"rbf_neighbors":24,"rbf_smoothing":0.},
    "pod_ridge":{"pod_components":16,"ridge_alpha":.001}}


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def resolve_config(config):
    if set(config)-set(DEFAULTS):
        raise ValueError("Unknown top-level experiment setting: "+str(set(config)-set(DEFAULTS)))
    result=deepcopy(DEFAULTS)
    for key,value in config.items():
        if key in {"training","loss","metrics"}:
            if set(value)-set(DEFAULTS[key]):
                raise ValueError(f"Unknown {key} setting: {set(value)-set(DEFAULTS[key])}")
            result[key].update(value)
        elif key=="model":
            if set(value)-MODEL_KEYS:
                raise ValueError("Unknown model setting: "+str(set(value)-MODEL_KEYS))
            result[key]=deepcopy(value)
        else:result[key]=value
    seed=result["seed"]
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    kind=result["model"].get("kind")
    if kind not in MODEL_DEFAULTS:raise ValueError("Unknown model kind")
    model=deepcopy(MODEL_DEFAULTS[kind]);model.update(result["model"])
    if kind in {"unet","hybrid_transunet","parameter_mlp"}:
        controls=validate_controls(model);controls['smoothing_kernels']=list(controls['smoothing_kernels'])
        model.update(controls)
    elif kind=='rbf':
        n=model['rbf_neighbors']
        if isinstance(n,bool) or not isinstance(n,int) or n<3:
            raise ValueError('rbf_neighbors must be an integer at least3')
        if not _number(model['rbf_smoothing']) or model['rbf_smoothing']<0:
            raise ValueError('rbf_smoothing must be finite and nonnegative')
    elif kind in {'pod_ridge','pod_gpr','pod_mlp'}:
        n=model['pod_components']
        if isinstance(n,bool) or not isinstance(n,int) or n<1:
            raise ValueError('pod_components must be positive')
        if kind=='pod_ridge' and (not _number(model['ridge_alpha']) or model['ridge_alpha']<=0):
            raise ValueError('ridge_alpha must be finite and positive')
        if kind=='pod_gpr' and any(not _number(model[k]) or model[k]<=0 for k in ['gpr_length_scale','gpr_noise']):
            raise ValueError('GPR scales must be finite and positive')
        if kind=='pod_mlp' and (not isinstance(model['hidden'],(list,tuple)) or not model['hidden'] or
                               any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in model['hidden'])):
            raise ValueError('POD MLP hidden widths must be positive integers')
    result['model']=model
    t=result["training"]
    for key in ["epochs","batch_size","gradient_accumulation","early_stopping_patience","scheduler_patience"]:
        if isinstance(t[key],bool) or not isinstance(t[key],int) or t[key]<1:
            raise ValueError(f"training.{key} must be a positive integer")
    if not all(_number(t[k]) for k in ['learning_rate','weight_decay','gradient_clip','min_delta']):
        raise ValueError('Training coefficients must be finite numbers')
    if t["learning_rate"]<=0 or t["weight_decay"]<0 or t["gradient_clip"]<=0 or t["min_delta"]<0:
        raise ValueError("Invalid optimizer/early stopping settings")
    if isinstance(t['num_workers'],bool) or not isinstance(t['num_workers'],int) or t['num_workers']<0:
        raise ValueError('num_workers must be a nonnegative integer')
    if any(not isinstance(t[k],bool) for k in ['amp','deterministic']):
        raise ValueError('amp and deterministic must be booleans')
    if t["optimizer"] not in {"Adam","AdamW"}:
        raise ValueError("optimizer must be Adam or AdamW")
    if result['loss']['kind'] not in {'huber','mse','mae','mape'}:
        raise ValueError('Unknown morphology loss')
    if not all(_number(x) for x in [result['loss']['beta_local'],*result['metrics'].values()]):
        raise ValueError('Loss and metric scales must be finite numbers')
    if result["loss"]["beta_local"]<0 or min(result["metrics"].values())<=0:
        raise ValueError("Invalid loss or metric scales")
    return result
