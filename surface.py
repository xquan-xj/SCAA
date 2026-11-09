"""Prediction-only, polar-area weighted continuous bicubic surface projection.

No target, validation score, boundary padding or fitted hyperparameter enters
the projection. Nominal knot spacing is an experimental representation scale,
not a laser spot size or a manufacturing acceptance limit.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import qr, solve_triangular


def _axis(value):
    axis = np.asarray(value, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 4 or not np.isfinite(axis).all():
        raise ValueError('Axis must have at least four finite coordinates')
    if np.any(np.diff(axis) <= 0):
        raise ValueError('Axis must be strictly increasing')
    return axis


def _knots(axis, intervals):
    edges = np.linspace(axis[0], axis[-1], intervals+1)
    return np.r_[np.repeat(edges[0], 4), edges[1:-1], np.repeat(edges[-1], 4)]


def _basis(knots, points, derivative=0):
    n = len(knots)-4
    return BSpline(knots, np.eye(n), 3, extrapolate=False)(points, nu=derivative)


def _weighted_inverse(basis, weights):
    weighted = basis*np.sqrt(weights)[:, None]
    q, r = qr(weighted, mode='economic')
    if r.shape[0] != r.shape[1] or np.linalg.cond(r) > 1e10:
        raise ValueError('Spline basis is underdetermined or ill-conditioned')
    return solve_triangular(r, q.T*np.sqrt(weights)[None, :])


class SurfaceProjector:
    def __init__(self, r_mm, theta_rad, nominal_spacing_mm):
        self.r, self.theta = _axis(r_mm), _axis(theta_rad)
        if np.any(self.r <= 0) or not np.isfinite(nominal_spacing_mm) or nominal_spacing_mm <= 0:
            raise ValueError('Positive radii and finite positive knot spacing required')
        self.h = float(nominal_spacing_mm)
        self.r_mean = float((self.r[0]+self.r[-1])/2)
        nr = max(1, int(np.ceil(np.ptp(self.r)/self.h)))
        nt = max(1, int(np.ceil(self.r_mean*np.ptp(self.theta)/self.h)))
        self.knots_r, self.knots_theta = _knots(self.r, nr), _knots(self.theta, nt)
        self.br, self.bt = _basis(self.knots_r, self.r), _basis(self.knots_theta, self.theta)
        if self.br.shape[1] > len(self.r) or self.bt.shape[1] > len(self.theta):
            raise ValueError('Requested surface resolution exceeds the input grid')
        wr, wt = self.r.copy(), np.ones(len(self.theta))
        wr[[0, -1]] *= .5
        wt[[0, -1]] *= .5
        self.area = wt[:, None]*wr[None, :]
        self.left = _weighted_inverse(self.bt, wt)
        self.right = _weighted_inverse(self.br, wr)
        self.description = dict(degree=[3, 3], nominal_spacing_mm=self.h,
            radial_interval_mm=float(np.ptp(self.r)/nr),
            angular_interval_rad=float(np.ptp(self.theta)/nt),
            reference_arc_interval_mm=float(self.r_mean*np.ptp(self.theta)/nt),
            reference_radius_mm=self.r_mean, coefficient_shape=[self.bt.shape[1], self.br.shape[1]],
            coefficient_count=self.bt.shape[1]*self.br.shape[1],
            interior_continuity='C2', axes=['theta', 'radius'],
            boundary_policy='Full-domain weighted fit; no padding, no fixed endpoint height/slope',
            scope='Mathematical representation; not a laser manufacturing limit')

    def fit(self, prediction_um):
        value = np.asarray(prediction_um, dtype=np.float64)
        if value.ndim < 2 or value.shape[-2:] != self.area.shape or not np.isfinite(value).all():
            raise ValueError('Finite prediction fields must match [theta,radius] grid')
        return self.left @ value @ self.right.T

    def evaluate(self, coefficients, *, r_mm=None, theta_rad=None):
        c = self._coefficients(coefficients)
        r = self.r if r_mm is None else np.asarray(r_mm, dtype=np.float64)
        theta = self.theta if theta_rad is None else np.asarray(theta_rad, dtype=np.float64)
        r = self._points(r, self.r)
        theta = self._points(theta, self.theta)
        return _basis(self.knots_theta, theta) @ c @ _basis(self.knots_r, r).T

    def at_points(self, coefficients, r_mm, theta_rad, derivative=(0, 0)):
        c = self._coefficients(coefficients)
        r, theta = np.broadcast_arrays(np.asarray(r_mm, dtype=np.float64),
                                       np.asarray(theta_rad, dtype=np.float64))
        shape = r.shape
        r, theta = self._points(r.ravel(), self.r), self._points(theta.ravel(), self.theta)
        if len(derivative) != 2 or any(x not in (0, 1, 2) for x in derivative):
            raise ValueError('Derivative orders must be 0,1 or2 for theta,radius')
        bt = _basis(self.knots_theta, theta, derivative[0])
        br = _basis(self.knots_r, r, derivative[1])
        result = np.einsum('na,...ab,nb->...n', bt, c, br, optimize=True)
        return result.reshape(c.shape[:-2]+shape)

    def _coefficients(self, value):
        c = np.asarray(value, dtype=np.float64)
        if c.ndim < 2 or c.shape[-2:] != (self.bt.shape[1], self.br.shape[1]) or not np.isfinite(c).all():
            raise ValueError('Finite coefficients must match the declared spline basis')
        return c

    @staticmethod
    def _points(value, domain):
        eps = 1e-10*max(1., abs(domain[0]), abs(domain[-1]))
        if not np.isfinite(value).all() or np.any(value < domain[0]-eps) or np.any(value > domain[-1]+eps):
            raise ValueError('Surface evaluation outside the declared physical domain is forbidden')
        return np.clip(value, domain[0], domain[-1])


def physical_roughness(height_um, r_mm, theta_rad):
    """Same centered physical gradient/Hessian diagnostics for raw and fitted.

    Hessian components in the orthonormal polar frame:
      Hrr=zrr, Hrt=zrtheta/r-ztheta/r², Htt=zr/r+zthetatheta/r².
    Endpoint ring is excluded from derivatives only, never from geometry scores.
    """
    r, theta = _axis(r_mm), _axis(theta_rad)
    z = np.asarray(height_um, dtype=np.float64)
    if z.shape != (len(theta), len(r)) or not np.isfinite(z).all():
        raise ValueError('Finite height field must match physical coordinates')
    for axis in (r, theta):
        if not np.allclose(np.diff(axis), np.diff(axis).mean(), rtol=2e-4, atol=1e-10):
            raise ValueError('Centered diagnostics require uniform physical axes')
    dr, dt = float(np.diff(r).mean()), float(np.diff(theta).mean())
    center = z[1:-1, 1:-1]
    zr = (z[1:-1, 2:]-z[1:-1, :-2])/(2*dr)
    zt = (z[2:, 1:-1]-z[:-2, 1:-1])/(2*dt)
    zrr = (z[1:-1, 2:]-2*center+z[1:-1, :-2])/dr**2
    ztt = (z[2:, 1:-1]-2*center+z[:-2, 1:-1])/dt**2
    zrt = (z[2:, 2:]-z[2:, :-2]-z[:-2, 2:]+z[:-2, :-2])/(4*dr*dt)
    radius = r[None, 1:-1]
    gradient2 = zr*zr+(zt/radius)**2
    hessian2 = zrr*zrr+2*(zrt/radius-zt/radius**2)**2+(zr/radius+ztt/radius**2)**2
    weights = np.broadcast_to(radius, center.shape)
    return dict(gradient_rms_um_mm=float(np.sqrt(np.sum(weights*gradient2)/weights.sum())),
        gradient_max_um_mm=float(np.sqrt(gradient2.max())),
        hessian_rms_um_mm2=float(np.sqrt(np.sum(weights*hessian2)/weights.sum())))
