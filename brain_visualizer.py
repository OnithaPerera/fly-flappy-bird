"""
Procedural Drosophila brain renderer showing the leader fly's live circuit.

The anatomy (optic lobes, central brain, mushroom bodies, antennal lobes, central
complex, neck connective and ventral nerve cord) is generated procedurally to
resemble frontal FlyWire renders. It is a stylised drawing, NOT FlyWire morphology.

The highlighted neurons are the simulated circuit itself (circuit.py), and every
flash is real model activity of the leader fly:

    MEDULLA  one column per ommatidium, brightness = its nearness and looming (graded)
    LPLC2    the 8 retinotopic LIF units, flash = a spike
    LPi      the 8 inhibitory LIF units, flash = a spike
    PROP     the climb / sink proprioceptors, flash = a spike
    GF       the Giant Fiber, flash = a spike; white dots are that spike travelling
    MN       flight motor neurons, flash = a wing stroke actually starting

The simulated world is 2-D, so both eyes see the same panorama and each unit is
drawn once in each hemisphere with identical activity.
"""
import colorsys
import numpy as np
import pygame
import cv2
from config import V_REST, N_LPLC2, N_LPI
from retina import ELEVATIONS

VIEW_W, VIEW_H = 500, 400
ORIGIN = np.array([250.0, 120.0])  # brain centre inside the view
RASTER_LEN = 240                   # frames of history (4 s at 1x)

# key, legend label, colour, raster rows
GROUPS = [
    ("lplc2", "LPLC2", (255, 168, 40), N_LPLC2),
    ("lpi", "LPi", (255, 70, 175), N_LPI),
    ("prop", "PROP", (165, 125, 255), 2),
    ("gf", "GF", (255, 236, 120), 1),
    ("mn", "MN", (70, 255, 140), 1),
]
GROUP_COLOR = {g[0]: g[2] for g in GROUPS}
GROUP_COLOR["medulla"] = (0, 205, 255)


# ---------------------------------------------------------------- geometry ---

def _ellipse(cx, cy, rx, ry):
    return lambda x, y: ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def _union(*fns):
    def inside(x, y):
        m = fns[0](x, y)
        for f in fns[1:]:
            m = m | f(x, y)
        return m
    return inside


def _intersect(a, b):
    return lambda x, y: a(x, y) & b(x, y)


def _minus(a, b):
    return lambda x, y: a(x, y) & ~b(x, y)


def _toward(cx, cy):
    return lambda p: (np.full(len(p), cx), np.full(len(p), cy))


def _sample_inside(rng, inside, bbox, n):
    x0, y0, x1, y1 = bbox
    out = np.empty((0, 2))
    while len(out) < n:
        pts = rng.uniform((x0, y0), (x1, y1), size=(n * 3, 2))
        out = np.vstack([out, pts[inside(pts[:, 0], pts[:, 1])]])
    return out[:n]


def _wander(rng, starts, steps, step_len, inside, target, turn=0.45):
    """Vectorised random-walk neurites that steer back whenever they leave their neuropil."""
    n = len(starts)
    pos = np.asarray(starts, dtype=np.float64).copy()
    theta = rng.uniform(0, 2 * np.pi, n)
    paths = np.empty((n, steps + 1, 2))
    paths[:, 0] = pos
    for s in range(1, steps + 1):
        theta += rng.normal(0, turn, n)
        cand = pos + step_len * np.column_stack((np.cos(theta), np.sin(theta)))
        out = ~inside(cand[:, 0], cand[:, 1])
        if out.any():
            tx, ty = target(pos[out])
            theta[out] = np.arctan2(ty - pos[out, 1], tx - pos[out, 0]) + rng.normal(0, 0.6, out.sum())
            cand[out] = pos[out] + step_len * np.column_stack((np.cos(theta[out]), np.sin(theta[out])))
        pos = cand
        paths[:, s] = pos
    return paths


def _smooth(pts, iters=3):
    """Chaikin corner cutting: turns a few control points into a smooth neurite."""
    pts = np.asarray(pts, dtype=np.float64)
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((len(q) * 2, 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def _neurite(rng, ctrl, jitter=1.5, iters=3):
    ctrl = np.asarray(ctrl, dtype=np.float64).copy()
    ctrl[1:-1] += rng.normal(0, jitter, ctrl[1:-1].shape)
    return _smooth(ctrl, iters)


def _tuft(rng, center, radius, branches):
    """Short dendritic branches radiating from a point."""
    paths = []
    for a in rng.uniform(0, 2 * np.pi, branches):
        length = radius * rng.uniform(0.5, 1.0)
        bend = a + rng.normal(0, 0.5)
        mid = center + 0.55 * length * np.array([np.cos(a), np.sin(a)])
        end = mid + 0.45 * length * np.array([np.cos(bend), np.sin(bend)])
        paths.append(_smooth([center, mid, end], 2))
    return paths


def _ray_to_ellipse(f, d, cx, cy, rx, ry):
    """Distance from interior point f along unit direction d to the ellipse boundary."""
    fx, fy = f[0] - cx, f[1] - cy
    a = (d[0] / rx) ** 2 + (d[1] / ry) ** 2
    b = 2 * (fx * d[0] / rx ** 2 + fy * d[1] / ry ** 2)
    c = (fx / rx) ** 2 + (fy / ry) ** 2 - 1
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


def _rainbow(rng, n, sat=(0.7, 1.0)):
    hues = rng.uniform(0, 1, n)
    sats = rng.uniform(*sat, n)
    return np.array([colorsys.hsv_to_rgb(h, s, 1.0) for h, s in zip(hues, sats)])


def _to_view(path):
    return [(int(x), int(y)) for x, y in path + ORIGIN]


def _hot(color, level):
    """Group colour scaled by activity, bleaching towards white at the spike peak."""
    white = 255 * max(0.0, level - 0.55) * 1.1
    return tuple(int(min(255, c * level + white)) for c in color)


class _Splat:
    """
    Bakes the static neuropil once. Fibres are painted opaquely in random order (like
    an opaque FlyWire mesh render, so every neuron keeps its own saturated colour)
    and then shaded by local fibre density, plus a soft glow.
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.paint = pygame.Surface((w, h))
        self.density = np.zeros(w * h)
        self.haze = np.zeros((h, w), dtype=np.float32)

    def paths(self, paths, colors, weight=1.0):
        idx_all = []
        for p, c in zip(paths, colors):
            p = np.asarray(p, dtype=np.float64) + ORIGIN
            pygame.draw.lines(self.paint, [int(255 * v * weight) for v in c], False, p.tolist(), 1)
            seg = np.hypot(*np.diff(p, axis=0).T)
            s = np.concatenate(([0.0], np.cumsum(seg)))
            t = np.arange(0.0, s[-1], 0.7)
            xi = np.floor(np.interp(t, s, p[:, 0])).astype(np.int64)
            yi = np.floor(np.interp(t, s, p[:, 1])).astype(np.int64)
            ok = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h)
            idx_all.append(yi[ok] * self.w + xi[ok])
        if idx_all:
            self.density += np.bincount(np.concatenate(idx_all), minlength=self.w * self.h)

    def region(self, inside, strength):
        ys, xs = np.mgrid[0:self.h, 0:self.w]
        self.haze += inside(xs - ORIGIN[0] + 0.5, ys - ORIGIN[1] + 0.5) * np.float32(strength)

    def render(self):
        color = pygame.surfarray.array3d(self.paint).transpose(1, 0, 2).astype(np.float32) / 255.0
        dens = cv2.GaussianBlur(self.density.reshape(self.h, self.w).astype(np.float32), (0, 0), 1.2)
        shade = 0.3 + 0.7 * (1.0 - np.exp(-dens * 0.25))
        img = color * shade[..., None]
        haze = cv2.GaussianBlur(self.haze, (0, 0), 5)
        img += haze[..., None] * np.array([0.05, 0.07, 0.11], dtype=np.float32)
        glow = cv2.GaussianBlur(img, (0, 0), 3.0)
        img = np.clip(img * 0.8 + glow * 0.3, 0, 1)
        arr = np.ascontiguousarray((img * 255).astype(np.uint8).transpose(1, 0, 2))
        return pygame.surfarray.make_surface(arr)



# -------------------------------------------------------------- visualizer ---

class BrainVisualizer:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.font = pygame.font.SysFont("Consolas", 11)

        self.neurons = []   # highlighted circuit neurons (dicts)
        splat = _Splat(VIEW_W, VIEW_H)
        self._build_anatomy(splat)
        self._build_functional(splat)
        self.base = splat.render()
        self.overlay = pygame.Surface((VIEW_W, VIEW_H))

        groups = [g[0] for g in GROUPS] + ["medulla"]
        self.idx = {k: np.array([i for i, n in enumerate(self.neurons) if n["group"] == k]) for k in groups}
        self.unit = np.array([n["unit"] for n in self.neurons])

        self.raster_colors = [g[2] for g in GROUPS for _ in range(g[3])]
        self.raster = np.zeros((len(self.raster_colors), RASTER_LEN), dtype=bool)
        self.head = 0
        self.flash = np.zeros(len(self.neurons))
        self.rate = {g[0]: 0.0 for g in GROUPS}
        self.gf_level = 0.0
        self.pulses = []

    def reset(self):
        self.raster[:] = False
        self.flash[:] = 0.0
        self.pulses = []

    # ---- anatomy ----------------------------------------------------------

    def _add_bg(self, splat, neurons, colors=None):
        """neurons: list of neurons, each a list of paths. Splats them with one colour per neuron."""
        if colors is None:
            colors = _rainbow(self.rng, len(neurons))
        paths, cols = [], []
        for n, c in zip(neurons, colors):
            paths.extend(n)
            cols.extend([c] * len(n))
        splat.paths(paths, cols)

    def _arbors(self, starts, steps, step_len, inside, target, branches=2, turn=0.45):
        rng = self.rng
        trunks = _wander(rng, starts, steps, step_len, inside, target, turn)
        neurons = [[t] for t in trunks]
        for _ in range(branches):
            k = rng.integers(steps // 4, steps, len(trunks))
            br = _wander(rng, trunks[np.arange(len(trunks)), k], steps // 3, step_len * 0.8,
                         inside, target, turn * 1.3)
            for n, b in zip(neurons, br):
                n.append(b)
        return neurons

    def _build_anatomy(self, splat):
        rng = self.rng

        # Two hemispheres with a dorsal midline notch, minus the oesophageal foramen
        central = _minus(_union(_ellipse(-46, -4, 62, 86), _ellipse(46, -4, 62, 86), _ellipse(0, 14, 56, 76)),
                         _ellipse(0, 32, 13, 16))
        vnc = _union(_ellipse(0, 160, 34, 20), _ellipse(0, 198, 42, 24),
                     _ellipse(0, 234, 34, 18), _ellipse(0, 262, 16, 14))
        neck = _ellipse(0, 118, 10, 30)
        lobes = {s: (s * 172.0, 4.0, 64.0, 100.0) for s in (-1, 1)}
        for region in (central, vnc, neck, _ellipse(*lobes[-1]), _ellipse(*lobes[1])):
            splat.region(region, 1.0)

        # Central brain: local arbors clustered into sub-neuropils so it has texture
        for c in _sample_inside(rng, central, (-105, -90, 105, 90), 30):
            sub = _minus(_intersect(_ellipse(c[0], c[1], 26, 22), central), _ellipse(0, 32, 13, 16))
            starts = _sample_inside(rng, sub, (c[0] - 26, c[1] - 22, c[0] + 26, c[1] + 22), 22)
            self._add_bg(splat, self._arbors(starts, 26, 2.4, sub, _toward(*c)))
        starts = _sample_inside(rng, central, (-105, -90, 105, 90), 520)
        self._add_bg(splat, self._arbors(starts, 40, 2.6, central, _toward(0, 0), 1))

        # Optic lobes
        for s in (-1, 1):
            cx, cy, rx, ry = lobes[s]
            focus = np.array([s * 118.0, 4.0])

            def column(a_deg, f0=0.36, f1=0.97, jitter=0.9):
                a = np.radians(a_deg)
                d = np.array([s * np.cos(a), np.sin(a)])
                t = _ray_to_ellipse(focus, d, cx, cy, rx, ry)
                ctrl = [focus + f * t * d for f in np.linspace(f0, f1, 4)]
                return _neurite(rng, ctrl, jitter, 2)

            # Medulla columns + lamina cartridges give the radial striation
            cols = [[column(a)] for a in rng.uniform(-86, 86, 260)]
            self._add_bg(splat, cols)
            lamina = [[column(a, 0.9, 1.0, 0.3)] for a in rng.uniform(-88, 88, 160)]
            self._add_bg(splat, lamina)

            # Tangential strata across the columns
            strata = []
            for frac in (0.48, 0.58, 0.68, 0.78, 0.88):
                for _ in range(5):
                    a0, a1 = sorted(rng.uniform(-88, 88, 2))
                    if a1 - a0 < 40:
                        a1 = min(88.0, a0 + 40)
                    pts = []
                    for a in np.linspace(a0, a1, 28):
                        ar = np.radians(a)
                        d = np.array([s * np.cos(ar), np.sin(ar)])
                        pts.append(focus + frac * _ray_to_ellipse(focus, d, cx, cy, rx, ry) * d)
                    pts = np.array(pts) + rng.normal(0, 0.6, (len(pts), 2))
                    strata.append([pts])
            self._add_bg(splat, strata)

            # Lobula / lobula plate and the chiasm linking it to the medulla
            lob = _ellipse(s * 130, 10, 22, 48)
            starts = _sample_inside(rng, lob, (s * 130 - 22, -38, s * 130 + 22, 58), 160)
            self._add_bg(splat, self._arbors(starts, 18, 2.0, lob, _toward(s * 130, 10), 1))
            chiasm = []
            for _ in range(70):
                a = np.radians(rng.uniform(-70, 70))
                d = np.array([s * np.cos(a), np.sin(a)])
                end = focus + 0.36 * _ray_to_ellipse(focus, d, cx, cy, rx, ry) * d
                start = np.array([s * 130 + rng.normal(0, 8), 10 + rng.normal(0, 25)])
                chiasm.append([_neurite(rng, [start, (start + end) / 2 + rng.normal(0, 4, 2), end], 1.0)])
            self._add_bg(splat, chiasm)

        # Mushroom bodies: calyx -> peduncle -> vertical + medial lobes
        for s in (-1, 1):
            calyx = np.array([s * 50.0, -60.0])
            kcs = []
            for i in range(170):
                c0 = calyx + rng.normal(0, 6, 2)
                ped = np.array([s * 30.0, 2.0]) + rng.normal(0, 1.5, 2)
                if i % 2:
                    end = np.array([s * 30.0, -58.0]) + rng.normal(0, 3, 2)
                    ctrl = [c0, (c0 + ped) / 2, ped, (ped + end) / 2 + (s * 2, 0), end]
                else:
                    end = np.array([s * 5.0, 5.0]) + rng.normal(0, 2, 2)
                    ctrl = [c0, (c0 + ped) / 2, ped, (ped + end) / 2, end]
                kcs.append([_neurite(rng, ctrl, 1.2)])
            self._add_bg(splat, kcs)
            cal = _ellipse(calyx[0], calyx[1], 14, 12)
            starts = _sample_inside(rng, cal, (calyx[0] - 14, -72, calyx[0] + 14, -48), 90)
            self._add_bg(splat, self._arbors(starts, 10, 1.6, cal, _toward(*calyx), 1))

        # Antennal lobes: glomeruli + projection neurons to calyx / lateral horn
        for s in (-1, 1):
            al = np.array([s * 30.0, 58.0])
            glom = []
            for g in range(36):
                gc = al + rng.normal(0, 6.5, 2)
                inside = _ellipse(gc[0], gc[1], 4, 4)
                glom.extend(_wander(rng, np.tile(gc, (4, 1)), 10, 1.3, inside, _toward(*gc), 0.9))
            colors = np.repeat(_rainbow(rng, 36), 4, axis=0)
            self._add_bg(splat, [[g] for g in glom], colors=colors)
            pns = []
            for _ in range(26):
                start = al + rng.normal(0, 5, 2)
                end = np.array([s * 80.0, -38.0]) if rng.random() < 0.5 else np.array([s * 50.0, -58.0])
                pns.append([_neurite(rng, [start, (s * 40, 20), (s * 55, -25), end + rng.normal(0, 4, 2)], 2.0)])
            self._add_bg(splat, pns)

        # Central complex: protocerebral bridge, fan-shaped body, ellipsoid body
        fb = _ellipse(0, -40, 38, 15)
        starts = _sample_inside(rng, fb, (-38, -55, 38, -25), 150)
        self._add_bg(splat, self._arbors(starts, 14, 2.0, fb, _toward(0, -40), 1))
        ring = []
        for _ in range(60):
            a0 = rng.uniform(0, 2 * np.pi)
            a = np.linspace(a0, a0 + rng.uniform(1.0, 2.4), 16)
            r = rng.uniform(8, 12)
            ring.append([np.column_stack((np.cos(a) * r, -16 + np.sin(a) * r * 0.7))])
        self._add_bg(splat, ring)
        columnar = []
        for _ in range(50):
            x = rng.uniform(-32, 32)
            columnar.append([_neurite(rng, [(x, -66 + 0.006 * x * x), (x * 1.05, -50), (x * 1.1, -32)], 0.8)])
        self._add_bg(splat, columnar)

        # Neck connective
        neck_fibers = []
        for _ in range(60):
            x0 = rng.uniform(-8, 8)
            neck_fibers.append([_neurite(rng, [(x0, 70), (x0 * 0.8, 100), (x0 * 0.7, 125), (x0 * 1.6, 150)], 0.8)])
        self._add_bg(splat, neck_fibers)

        # Ventral nerve cord: neuromere tangles + leg nerves
        target = lambda p: (np.zeros(len(p)), np.clip(p[:, 1], 160, 262))
        starts = _sample_inside(rng, vnc, (-42, 140, 42, 276), 520)
        self._add_bg(splat, self._arbors(starts, 28, 2.3, vnc, target))
        nerves = []
        for y, rx, dy in ((160, 34, -14), (198, 42, 0), (234, 34, 14)):
            for s in (-1, 1):
                for _ in range(8):
                    start = (s * rx * 0.55, y + rng.normal(0, 3))
                    end = (s * (rx + 32), y + dy + rng.normal(0, 3))
                    nerves.append([_neurite(rng, [start, (s * rx, y + dy / 2), end], 1.2)])
        self._add_bg(splat, nerves)


    # ---- circuit neurons --------------------------------------------------

    def _add_neuron(self, splat, group, unit, paths, width=1, soma=None):
        color = GROUP_COLOR[group]
        splat.paths(paths, [np.array(color) / 255.0] * len(paths), 0.6)
        self.neurons.append({
            "group": group, "unit": unit, "color": color, "width": width,
            "paths": [_to_view(p) for p in paths],
            "soma": tuple(int(v) for v in np.asarray(soma) + ORIGIN) if soma is not None else None,
        })

    def _build_functional(self, splat):
        rng = self.rng

        # Medulla columns, one per ommatidium; dorsal ommatidia sit at the top of the lobe
        for s in (-1, 1):
            cx, cy, rx, ry = s * 172.0, 4.0, 64.0, 100.0
            focus = np.array([s * 118.0, 4.0])
            for k, a_deg in enumerate(-0.8 * np.degrees(ELEVATIONS)):
                a = np.radians(a_deg)
                d = np.array([s * np.cos(a), np.sin(a)])
                t = _ray_to_ellipse(focus, d, cx, cy, rx, ry)
                path = _neurite(rng, [focus + f * t * d for f in (0.4, 0.6, 0.8, 0.98)], 0.6, 2)
                self._add_neuron(splat, "medulla", k, [path])

        # LPLC2: dendrites in the lobula, axons converge on the GF dendrite glomerulus
        for s in (-1, 1):
            for k, y0 in enumerate(np.linspace(-40, 50, N_LPLC2)):
                den = np.array([s * 132.0, y0])
                axon = _neurite(rng, [den, (s * 112, y0 * 0.6 + 8), (s * 84, 20), (s * 64, 24)], 2.0)
                paths = [axon] + _tuft(rng, den, 11, 5) + _tuft(rng, np.array([s * 64.0, 24.0]), 6, 3)
                self._add_neuron(splat, "lplc2", k, paths, soma=den)

        # LPi: inhibitory interneurons in the lobula plate
        for s in (-1, 1):
            for k, y0 in enumerate(np.linspace(-40, 40, N_LPI)):
                path = _neurite(rng, [(s * 140, y0 - 9), (s * 147, y0), (s * 140, y0 + 9)], 1.0)
                self._add_neuron(splat, "lpi", k, [path] + _tuft(rng, np.array([s * 145.0, y0]), 7, 3),
                                 soma=(s * 147, y0))

        # Proprioceptive afferents: enter the metathoracic neuromere and ascend to the GF
        for s in (-1, 1):
            for k in range(2):
                o = k * 4
                path = _neurite(rng, [(s * 70, 246 + o), (s * 30, 236 + o), (s * (10 + o), 200),
                                      (s * (6 + o / 2), 140), (s * 9, 100), (s * 30, 70), (s * 52, 34)], 1.2)
                self._add_neuron(splat, "prop", k, [path] + _tuft(rng, np.array([s * 52.0, 34.0]), 6, 3))

        # Giant Fibers: large dendrite at the LPLC2 glomerulus, axon down the neck to the TTMn
        self.gf_axons = []
        for s in (-1, 1):
            den = np.array([s * 62.0, 22.0])
            axon = _neurite(rng, [den, (s * 36, 44), (s * 12, 70), (s * 5, 100), (s * 5, 150),
                                  (s * 7, 182), (s * 12, 196), (s * 26, 202)], 0.8)
            soma_path = _neurite(rng, [(s * 56, -34), (s * 62, -8), den], 1.0)
            self._add_neuron(splat, "gf", 0, [axon, soma_path] + _tuft(rng, den, 16, 8),
                             width=3, soma=(s * 56, -34))
            pts = axon + ORIGIN
            seg = np.hypot(*np.diff(pts, axis=0).T)
            self.gf_axons.append((pts, np.concatenate(([0.0], np.cumsum(seg)))))

        # Flight motor neurons in the mesothoracic neuromere
        for s in (-1, 1):
            for j in range(3):
                soma = np.array([s * (20.0 + j * 3), 188.0 + j * 7])
                path = _neurite(rng, [soma, (s * 34, 198 + j * 4), (s * 56, 202 + j * 3), (s * 86, 206 + j * 6)], 1.0)
                self._add_neuron(splat, "mn", 0, [path] + _tuft(rng, soma, 7, 3), width=2, soma=soma)

    # ---- simulation -------------------------------------------------------

    def step(self, circuit, i, stroke):
        """Copy one frame of fly i's circuit activity. stroke: a wing stroke started this frame."""
        spikes = {
            "lplc2": circuit.spk_lplc2[i],
            "lpi": circuit.spk_lpi[i],
            "prop": circuit.spk_prop[i],
            "gf": circuit.spk_gf[i:i + 1],
            "mn": np.array([stroke]),
        }
        self.flash *= 0.75
        for key, s in spikes.items():
            idx = self.idx[key]
            self.flash[idx[s[self.unit[idx]]]] = 1.0
            self.rate[key] += 0.05 * (s.mean() * 60.0 - self.rate[key])   # Hz per unit, smoothed

        med = self.idx["medulla"]
        k = self.unit[med]
        level = 0.1 + 0.5 * circuit.near[i, k] + 0.6 * np.clip(circuit.loom[i, k], 0, 1)
        self.flash[med] = np.maximum(self.flash[med] * 0.6, np.clip(level, 0, 1))

        self.raster[:, self.head] = np.concatenate([spikes[g[0]] for g in GROUPS])
        self.head = (self.head + 1) % RASTER_LEN

        thr = circuit.gf_vth[i]
        self.gf_level = float(np.clip((circuit.gf_v[i] - V_REST) / (thr - V_REST), 0.0, 1.0))
        if spikes["gf"][0]:
            self.pulses.append(0.0)
        self.pulses = [t + 0.07 for t in self.pulses if t + 0.07 < 1.0]

    # ---- rendering --------------------------------------------------------

    def draw(self, surface, pos):
        surface.blit(self.base, pos)
        ov = self.overlay
        ov.fill((0, 0, 0))
        for i, n in enumerate(self.neurons):
            level = self.flash[i]
            if n["group"] == "gf":
                level = max(level, 0.15 + 0.35 * self.gf_level)  # subthreshold depolarisation
            if level < 0.05:
                continue
            color = _hot(n["color"], level)
            for path in n["paths"]:
                pygame.draw.lines(ov, color, False, path, n["width"])
            if n["soma"]:
                pygame.draw.circle(ov, color, n["soma"], n["width"] + 1)

        # Action potentials travelling down both GF axons
        for t in self.pulses:
            for pts, cum in self.gf_axons:
                d = t * cum[-1]
                x, y = np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])
                pygame.draw.circle(ov, (255, 255, 255), (int(x), int(y)), 3)

        bloom = pygame.transform.smoothscale(
            pygame.transform.smoothscale(ov, (VIEW_W // 4, VIEW_H // 4)), (VIEW_W, VIEW_H))
        surface.blit(bloom, pos, special_flags=pygame.BLEND_ADD)
        surface.blit(ov, pos, special_flags=pygame.BLEND_ADD)

        ox, oy = pos[0] + ORIGIN[0], pos[1] + ORIGIN[1]
        for text, center in (("L OPTIC LOBE", (ox - 172, oy + 116)), ("R OPTIC LOBE", (ox + 172, oy + 116)),
                             ("VNC", (ox + 40, oy + 266))):
            img = self.font.render(text, True, (122, 138, 163))
            surface.blit(img, img.get_rect(center=center))

    def draw_raster(self, surface, plot, label_right):
        order = (np.arange(RASTER_LEN) + self.head) % RASTER_LEN  # oldest -> newest
        data = self.raster[:, order]
        row_h = plot.h / len(self.raster_colors)
        dx = plot.w / RASTER_LEN
        row = 0
        for key, label, color, rows in GROUPS:
            top = plot.y + row * row_h
            if row:
                pygame.draw.line(surface, (30, 41, 59), (plot.x, top), (plot.right, top))
            img = self.font.render(label, True, color)
            surface.blit(img, img.get_rect(midright=(label_right, top + rows * row_h / 2)))
            row += rows
        for r in range(len(self.raster_colors)):
            y0 = plot.y + r * row_h + 1
            y1 = max(y0, plot.y + (r + 1) * row_h - 1)
            for c in np.flatnonzero(data[r]):
                x = plot.x + c * dx
                pygame.draw.line(surface, self.raster_colors[r], (x, y0), (x, y1), 2)
