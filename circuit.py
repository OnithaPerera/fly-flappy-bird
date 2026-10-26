"""
Looming-escape circuit of the fruit fly, as a vectorised spiking network.

    ommatidia (nearness image, N_OMMATIDIA)
        │  temporal contrast: loom = d(nearness)/dt, clipped at 0   (T4/T5-like, graded)
        ▼
    LPLC2  x N_LPLC2   retinotopic LIF units, cholinergic  → GF   (weights >= 0)
    LPi    x N_LPI     retinotopic LIF units, glutamatergic → GF  (weights <= 0)
    PROP   x 2         proprioceptors for climbing / sinking  → GF (sign evolved)
        ▼
    Giant Fiber        one LIF neuron; each spike commands a wing stroke

What is anatomy and what is learned:
  * Fixed: the cell types, the sign of every synapse (Dale's law), the retinotopic
    tiling of LPLC2/LPi receptive fields, and the GF as the sole trigger of a stroke.
  * Evolved: how strongly each unit responds to nearness vs. looming, every
    unit -> GF synaptic weight, and the GF's tonic drive, time constant and threshold.

Simplifications, stated plainly: the 2-D world gives a 1-D eye; real LPi neurons
inhibit LPLC2 dendrites rather than the GF directly; real halteres sense rotation,
so PROP is a generic body-velocity sense. The synaptic weights are NOT taken from
the FlyWire connectome; see connectome.py for the published reference used instead.
"""
import numpy as np
from config import (N_OMMATIDIA, N_LPLC2, N_LPI, RF_SIGMA, LOOM_MAX, PROP_GAIN, FPS,
                    SUBSTEPS, DT_MS, V_REST, V_RESET, V_THRESH, TAU_UNIT_MS,
                    REFRACTORY_MS, TAU_SYN_MS)

# name, low, high
GENE_SPEC = (
    [("lplc2_near", 0.0, 60.0), ("lplc2_loom", 0.0, 60.0),
     ("lpi_near", 0.0, 60.0), ("lpi_loom", 0.0, 60.0)]
    + [(f"w_lplc2_{k}", 0.0, 40.0) for k in range(N_LPLC2)]
    + [(f"w_lpi_{k}", -40.0, 0.0) for k in range(N_LPI)]
    + [("w_prop_climb", -40.0, 40.0), ("w_prop_sink", -40.0, 40.0),
       ("gf_tonic", -10.0, 25.0), ("gf_tau_ms", 5.0, 40.0), ("gf_vth", -56.0, -44.0)]
)
GENE_NAMES = [g[0] for g in GENE_SPEC]
GENE_BOUNDS = np.array([g[1:] for g in GENE_SPEC], dtype=np.float64)
GENOME_SIZE = len(GENE_SPEC)
GENOME_VERSION = 2   # bump whenever GENE_SPEC changes so stale checkpoints are rejected

IDX = {name: i for i, name in enumerate(GENE_NAMES)}
SL_LPLC2 = slice(IDX["w_lplc2_0"], IDX["w_lplc2_0"] + N_LPLC2)
SL_LPI = slice(IDX["w_lpi_0"], IDX["w_lpi_0"] + N_LPI)

_REFRACTORY_STEPS = int(round(REFRACTORY_MS / DT_MS))


def receptive_fields(n_units, n_omm=N_OMMATIDIA, sigma=RF_SIGMA):
    """Gaussian receptive fields tiling the eye dorsal -> ventral; rows sum to 1."""
    centers = np.linspace(0, n_omm - 1, n_units)
    spacing = (n_omm - 1) / max(1, n_units - 1)
    idx = np.arange(n_omm)
    rf = np.exp(-0.5 * ((idx[None, :] - centers[:, None]) / (sigma * spacing)) ** 2)
    return rf / rf.sum(axis=1, keepdims=True)


RF_LPLC2 = receptive_fields(N_LPLC2)
RF_LPI = receptive_fields(N_LPI)


def random_genomes(n, rng):
    return rng.uniform(GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1], size=(n, GENOME_SIZE))


def clip_genomes(g):
    return np.clip(g, GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1])


class _LIF:
    """A population of leaky integrate-and-fire units, shape (N, K)."""

    def __init__(self, shape, tau_ms):
        self.v = np.full(shape, V_REST)
        self.refractory = np.zeros(shape, dtype=np.int32)
        self.decay = np.exp(-DT_MS / np.asarray(tau_ms, dtype=np.float64))

    def step(self, drive, v_thresh=V_THRESH):
        """drive: steady-state depolarisation in mV. Returns the spikes of this substep."""
        active = self.refractory <= 0
        target = V_REST + drive
        self.v = np.where(active, target + (self.v - target) * self.decay, V_RESET)
        self.refractory -= 1
        spikes = active & (self.v >= v_thresh)
        self.v[spikes] = V_RESET
        self.refractory[spikes] = _REFRACTORY_STEPS
        return spikes


class FlyCircuit:
    def __init__(self, genomes):
        g = np.asarray(genomes, dtype=np.float64)
        self.genomes = g
        self.n = len(g)
        col = lambda name: g[:, IDX[name]][:, None]
        self.lplc2_near, self.lplc2_loom = col("lplc2_near"), col("lplc2_loom")
        self.lpi_near, self.lpi_loom = col("lpi_near"), col("lpi_loom")
        self.w_lplc2 = g[:, SL_LPLC2]
        self.w_lpi = g[:, SL_LPI]
        self.w_prop = np.column_stack((g[:, IDX["w_prop_climb"]], g[:, IDX["w_prop_sink"]]))
        self.gf_tonic = g[:, IDX["gf_tonic"]]
        self.gf_vth = g[:, IDX["gf_vth"]]
        self.gf_tau = g[:, IDX["gf_tau_ms"]]
        self.reset()

    def reset(self):
        n = self.n
        self.lplc2 = _LIF((n, N_LPLC2), TAU_UNIT_MS)
        self.lpi = _LIF((n, N_LPI), TAU_UNIT_MS)
        self.prop = _LIF((n, 2), TAU_UNIT_MS)
        self.gf = _LIF((n,), self.gf_tau)
        self.syn_decay = np.exp(-DT_MS / TAU_SYN_MS)
        # Synaptic current into the GF, kept per pathway so it can be displayed
        self.i_exc = np.zeros(n)
        self.i_inh = np.zeros(n)
        self.i_prop = np.zeros(n)
        self.prev_near = None
        # Per-frame observables (any spike within the frame's substeps)
        self.loom = np.zeros((n, N_OMMATIDIA))
        self.near = np.zeros((n, N_OMMATIDIA))
        self.spk_lplc2 = np.zeros((n, N_LPLC2), dtype=bool)
        self.spk_lpi = np.zeros((n, N_LPI), dtype=bool)
        self.spk_prop = np.zeros((n, 2), dtype=bool)
        self.spk_gf = np.zeros(n, dtype=bool)

    def step(self, nearness, vy):
        """
        nearness: (N, N_OMMATIDIA) image from the eye. vy: (N,) the fly's own vertical
        velocity (px/frame, + is down). Returns (N,) bool: did the GF fire this frame.
        """
        near = np.array(nearness, dtype=np.float64)   # a copy: callers may reuse their buffer
        if self.prev_near is None:
            self.prev_near = near
        loom = np.clip((near - self.prev_near) * FPS, 0.0, LOOM_MAX)
        self.prev_near = near
        self.near, self.loom = near, loom

        near_rf, loom_rf = near @ RF_LPLC2.T, loom @ RF_LPLC2.T
        drive_lplc2 = self.lplc2_near * near_rf + self.lplc2_loom * loom_rf
        near_rf, loom_rf = near @ RF_LPI.T, loom @ RF_LPI.T
        drive_lpi = self.lpi_near * near_rf + self.lpi_loom * loom_rf
        v = np.asarray(vy, dtype=np.float64) / 10.0
        drive_prop = PROP_GAIN * np.column_stack((np.clip(-v, 0, 1.5), np.clip(v, 0, 1.5)))

        self.spk_lplc2[:] = False
        self.spk_lpi[:] = False
        self.spk_prop[:] = False
        self.spk_gf[:] = False
        for _ in range(SUBSTEPS):
            s_lplc2 = self.lplc2.step(drive_lplc2)
            s_lpi = self.lpi.step(drive_lpi)
            s_prop = self.prop.step(drive_prop)
            self.i_exc = self.i_exc * self.syn_decay + (s_lplc2 * self.w_lplc2).sum(axis=1)
            self.i_inh = self.i_inh * self.syn_decay + (s_lpi * self.w_lpi).sum(axis=1)
            self.i_prop = self.i_prop * self.syn_decay + (s_prop * self.w_prop).sum(axis=1)
            s_gf = self.gf.step(self.gf_tonic + self.i_exc + self.i_inh + self.i_prop, self.gf_vth)
            self.spk_lplc2 |= s_lplc2
            self.spk_lpi |= s_lpi
            self.spk_prop |= s_prop
            self.spk_gf |= s_gf
        return self.spk_gf.copy()

    @property
    def gf_v(self):
        return self.gf.v
