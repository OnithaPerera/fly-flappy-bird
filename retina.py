"""
Compound eye: the only view of the world the controller gets.

The arena is 2-D, so the eye is a 1-D vertical panorama of ommatidia looking
forward, from dorsal (+100 deg, just past vertical) to ventral (-100 deg). Each ommatidium reports the
nearness of the first surface along its line of sight (ceiling, ground or pipe),
1.0 at the eye and 0.0 at EYE_RANGE or beyond. The eye does not label what it sees:
there is no "gap position", "pipe distance" or "altitude" signal, only this image.
"""
import numpy as np
from config import N_OMMATIDIA, EYE_ELEVATION_DEG, ACUTE_ZONE, EYE_RANGE, FLY_X, GROUND_Y, CEILING_Y

# Acute zone: ommatidia are packed more densely around the horizon, where the gap
# has to be resolved, and spread out towards the zenith and nadir.
_u = np.linspace(1.0, -1.0, N_OMMATIDIA)
ELEVATIONS = np.radians(EYE_ELEVATION_DEG * ((1 - ACUTE_ZONE) * _u + ACUTE_ZONE * _u ** 3))
RAY_DX = np.cos(ELEVATIONS)            # < 0 for the few rays that look slightly backwards
RAY_DY = -np.sin(ELEVATIONS)           # screen y grows downwards, so dorsal rays have dy < 0
_EPS = 1e-9


def ray_distances(y, obstacles):
    """
    y: (N,) fly heights. obstacles: (P, 4) rectangles [x0, x1, y0, y1].
    Returns (N, N_OMMATIDIA) distance to the first surface along each ray.
    """
    fy = np.asarray(y, dtype=np.float64)[:, None]
    dx, dy = RAY_DX[None, :], RAY_DY[None, :]
    safe_dy = np.where(np.abs(dy) < _EPS, _EPS, dy)

    # Ceiling and ground are horizontal lines across the whole arena
    t_ceiling = np.where(dy < -_EPS, (CEILING_Y - fy) / safe_dy, np.inf)
    t_ground = np.where(dy > _EPS, (GROUND_Y - fy) / safe_dy, np.inf)
    t = np.minimum(t_ceiling, t_ground)

    # Pipes: slab test against each rectangle
    for x0, x1, y0, y1 in obstacles:
        tx0, tx1 = (x0 - FLY_X) / dx, (x1 - FLY_X) / dx
        ty_a, ty_b = (y0 - fy) / safe_dy, (y1 - fy) / safe_dy
        flat = np.abs(dy) < _EPS
        ty_lo = np.where(flat, np.where((fy >= y0) & (fy <= y1), -np.inf, np.inf), np.minimum(ty_a, ty_b))
        ty_hi = np.where(flat, np.where((fy >= y0) & (fy <= y1), np.inf, -np.inf), np.maximum(ty_a, ty_b))
        t_in = np.maximum(np.minimum(tx0, tx1), ty_lo)
        t_out = np.minimum(np.maximum(tx0, tx1), ty_hi)
        hit = (t_out >= np.maximum(t_in, 0.0))
        t = np.where(hit, np.minimum(t, np.maximum(t_in, 0.0)), t)
    return t


def sample(y, obstacles):
    """Nearness image (N, N_OMMATIDIA) in [0, 1]."""
    return np.clip(1.0 - ray_distances(y, obstacles) / EYE_RANGE, 0.0, 1.0)


def ray_endpoints(y, distances):
    """Where each of one fly's rays stops (for drawing). distances: (N_OMMATIDIA,)."""
    d = np.minimum(distances, EYE_RANGE)
    return FLY_X + RAY_DX * d, y + RAY_DY * d
