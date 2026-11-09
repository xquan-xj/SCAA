"""Physical morphology errors on a polar grid with axes [theta, radius].

local_score_v2 = area_mean(sqrt(max(|e|-1,0)) * ell_mm**2 * |Delta_polar(e)|),
where e=(prediction_um-target_um)/tolerance_um. It is dimensionless, uses
physical grid spacing and excludes one boundary ring from curvature evaluation.
It is not the historical GWRE and has no inherited 0.01 acceptance threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch
from torch import nn
from torch.nn import functional as F


def _grid(r_mm, theta_rad, *, device=None, dtype=torch.float64):
    r = torch.as_tensor(r_mm, dtype=dtype, device=device)
    theta = torch.as_tensor(theta_rad, dtype=dtype, device=device)
    if r.ndim != 1 or theta.ndim != 1 or len(r)<3 or len(theta)<3:
        raise ValueError("At least three one-dimensional radial/angular coordinates are required")
    if not torch.isfinite(r).all() or not torch.isfinite(theta).all() or (r<=0).any():
        raise ValueError("Grid coordinates must be finite and radius positive")
    for axis in (r, theta):
        delta=axis[1:]-axis[:-1]
        if (delta<=0).any() or not torch.allclose(delta,delta.mean().expand_as(delta),rtol=2e-4,atol=1e-7):
            raise ValueError("Coordinates must be increasing and regularly spaced")
    return r,theta


def area_weights(r_mm, theta_rad, *, device=None, dtype=torch.float64):
    """Trapezoidal polar area weights; absolute normalization cancels in means."""
    r,theta=_grid(r_mm,theta_rad,device=device,dtype=dtype)
    wr=r.clone();wt=torch.ones_like(theta)
    wr[[0,-1]]*=.5;wt[[0,-1]]*=.5
    return wt[:,None]*wr[None,:]


def polar_laplacian(error, r_mm, theta_rad):
    """Centered physical polar Laplace operator on the interior, in mm^-2."""
    if error.ndim != 4 or not error.is_floating_point():
        raise ValueError("error must be a floating B,C,H,W tensor")
    r,theta=_grid(r_mm,theta_rad,device=error.device,dtype=error.dtype)
    if error.shape[-2:] != (len(theta),len(r)):
        raise ValueError("Tensor shape must match [theta,radius] coordinates")
    dr=(r[-1]-r[0])/(len(r)-1)
    dt=(theta[-1]-theta[0])/(len(theta)-1)
    center=error[...,1:-1,1:-1]
    drr=(error[...,1:-1,2:]-2*center+error[...,1:-1,:-2])/dr.square()
    dr1=(error[...,1:-1,2:]-error[...,1:-1,:-2])/(2*dr)
    dtt=(error[...,2:,1:-1]-2*center+error[...,:-2,1:-1])/dt.square()
    radius=r[1:-1].reshape(1,1,1,-1)
    return drr+dr1/radius+dtt/radius.square()


def _weighted_mean(value, weight):
    return (value*weight).sum(dim=(-2,-1))/weight.sum()


def geometry_metrics(pred_um, true_um, r_mm, theta_rad, *, tolerance_um=.1,
                     length_mm=.1, mape_epsilon_um=.001):
    if tolerance_um<=0 or length_mm<=0 or mape_epsilon_um<=0:
        raise ValueError("Physical reference scales must be positive")
    pred=np.asarray(pred_um,dtype=np.float64);true=np.asarray(true_um,dtype=np.float64)
    if pred.shape!=true.shape or pred.ndim!=2:
        raise ValueError("Predicted and true morphology must have matching H,W shapes")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Morphologies must contain only finite values")
    w=area_weights(r_mm,theta_rad).numpy()
    if w.shape!=pred.shape:
        raise ValueError("Morphology shape differs from the physical grid")
    w=w/w.sum();error=pred-true;a=np.abs(error)
    mean_true=float((w*true).sum())
    sst=float((w*(true-mean_true)**2).sum())
    mse=float((w*error**2).sum())
    e=torch.from_numpy(error/tolerance_um)[None,None]
    curvature=polar_laplacian(e,r_mm,theta_rad).abs().numpy()[0,0]*length_mm**2
    exceed=np.sqrt(np.maximum(a[1:-1,1:-1]/tolerance_um-1,0))
    wi=np.asarray(r_mm,dtype=np.float64)[None,1:-1]*np.ones((len(theta_rad)-2,1))
    score=float((wi*exceed*curvature).sum()/wi.sum())
    order=np.argsort(a.ravel());sorted_errors=a.ravel()[order]
    cumulative=np.cumsum(w.ravel()[order])
    p95=float(np.interp(.95,cumulative,sorted_errors))
    return {"mae_um":float((w*a).sum()),"rmse_um":float(np.sqrt(mse)),
            "bias_um":float((w*error).sum()),
            "max_abs_um":float(a.max()),"abs_error_p95_um":p95,
            "r2":None if sst<=1e-20 else float(1-mse/sst),
            "constant_target":bool(sst<=1e-20),
            "mape_percent":float(100*(w*a/(np.abs(true)+mape_epsilon_um)).sum()),
            "exceed_area_fraction":float((w*(a>tolerance_um)).sum()),
            "pointwise_pass":bool(a.max()<=tolerance_um),"local_score_v2":score}


def project_grid_to_nodes(pred_um, r_mm, theta_rad, nodes_xy_mm):
    """Linear grid projection with roundoff-only clipping at domain extrema."""
    r,theta=_grid(r_mm,theta_rad)
    r=r.numpy();theta=theta.numpy()
    grid=np.asarray(pred_um,dtype=float);xy=np.asarray(nodes_xy_mm,dtype=float)
    if grid.shape!=(len(theta),len(r)) or not np.isfinite(grid).all():
        raise ValueError("Projection grid must have finite [theta,radius] shape")
    if xy.ndim!=2 or xy.shape[1]!=2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Node coordinates must have finite (N,2) shape")
    coords=np.column_stack((np.arctan2(xy[:,1],xy[:,0]),np.linalg.norm(xy,axis=1)))
    lower=np.array([theta[0],r[0]]);upper=np.array([theta[-1],r[-1]])
    rounding=1e-10*np.maximum(1,np.maximum(np.abs(lower),np.abs(upper)))
    if (coords<lower-rounding).any() or (coords>upper+rounding).any():
        raise ValueError("Original nodes lie outside the projection grid")
    coords=np.clip(coords,lower,upper)
    return RegularGridInterpolator((theta,r),grid,bounds_error=True)(coords)


def node_geometry_metrics(pred_um,true_grid_um,true_nodes_um,r_mm,theta_rad,nodes_xy_mm):
    """Unweighted original-node errors and the label representation floor.

    FE tributary areas are unavailable here; these diagnostics are explicitly
    unweighted and do not replace the polar-area grid or real FEM metrics.
    """
    true=np.asarray(true_nodes_um,dtype=float)
    pred=project_grid_to_nodes(pred_um,r_mm,theta_rad,nodes_xy_mm)
    represented=project_grid_to_nodes(true_grid_um,r_mm,theta_rad,nodes_xy_mm)
    if true.shape!=pred.shape or not np.isfinite(true).all():
        raise ValueError("Original node labels must match the finite node count")
    error=pred-true;representation=represented-true
    return {"node_mae_um":float(np.abs(error).mean()),
            "node_rmse_um":float(np.sqrt(np.mean(error**2))),
            "node_max_abs_um":float(np.abs(error).max()),"node_bias_um":float(error.mean()),
            "node_weighting":"unweighted_original_nodes",
            "representation_node_rmse_um":float(np.sqrt(np.mean(representation**2))),
            "representation_node_max_abs_um":float(np.abs(representation).max())}


class MorphologyLoss(nn.Module):
    """Train in physical tolerance units regardless of target standardization.

    The local term is a smooth surrogate of local_score_v2. The base error term
    penalizes uniform offsets which a curvature-only indicator cannot detect.
    """
    def __init__(self,normalization,r_mm,theta_rad,*,kind="huber",beta_local=0.,
                 tolerance_um=.1,length_mm=.1,mape_epsilon_um=.001):
        super().__init__()
        if kind not in {"huber","mae","mse","mape"}:
            raise ValueError("Unknown morphology loss")
        if beta_local<0 or min(tolerance_um,length_mm,mape_epsilon_um)<=0:
            raise ValueError("Invalid physical loss coefficients")
        self.kind=kind;self.beta_local=float(beta_local)
        self.offset=float(normalization["target_offset_um"])
        self.scale=float(normalization["target_scale_um"])
        if self.scale<=0:
            raise ValueError("Target scale must be positive")
        self.tolerance=float(tolerance_um);self.length=float(length_mm)
        self.mape_eps=float(mape_epsilon_um)
        self.register_buffer("r_mm",torch.as_tensor(r_mm,dtype=torch.float32))
        self.register_buffer("theta_rad",torch.as_tensor(theta_rad,dtype=torch.float32))
        self.register_buffer("area",area_weights(r_mm,theta_rad,dtype=torch.float32))

    def forward(self,prediction,target):
        # Curvature stencils need float32 even when the network uses AMP.
        pred=prediction.float();truth=target.float()
        error=(pred-truth)*self.scale/self.tolerance
        if self.kind=="huber":
            base=F.smooth_l1_loss(error,torch.zeros_like(error),reduction="none")
        elif self.kind=="mae":base=error.abs()
        elif self.kind=="mse":base=error.square()
        else:
            true_um=truth*self.scale+self.offset
            base=error.abs()*self.tolerance/(true_um.abs()+self.mape_eps)
        result=_weighted_mean(base,self.area).mean()
        if self.beta_local:
            g=polar_laplacian(error,self.r_mm,self.theta_rad)*self.length**2
            a=torch.sqrt(error[...,1:-1,1:-1].square()+1e-8)
            # Smooth tolerance exceedance and curvature for stable gradients.
            exceed=F.softplus((a-1)/.1)*.1
            local=torch.sqrt(exceed+1e-8)*(torch.sqrt(g.square()+1e-8)-1e-4)
            wi=self.r_mm[None,1:-1].expand(len(self.theta_rad)-2,-1)
            result=result+self.beta_local*_weighted_mean(local,wi).mean()
        return result


def summarize_metrics(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    n=len(rows);count=sum(bool(r["pointwise_pass"]) for r in rows)
    result={"n_samples":n,"pointwise_pass_count":count,"pointwise_pass_rate":count/n}
    for key in rows[0]:
        if key in {"id","pointwise_pass","constant_target","preload_N","temperature_C","pressure_MPa",
                   "nominal_contact_area_mm2","selected_column_1based"}:
            continue
        values=[r[key] for r in rows if isinstance(r.get(key),(int,float)) and not isinstance(r[key],bool)]
        if values:
            result[key+"_mean"]=float(np.mean(values))
            result[key+"_defined_count"]=len(values)
    return result
