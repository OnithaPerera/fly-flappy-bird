"""
What the circuit borrows from the real fly, with sources, and how the evolved
weights compare.

Cell types and signs
  LPLC2    lobula plate / lobula columnar neurons tuned to looming (radial expansion),
           cholinergic, the main visual input to the Giant Fiber.
           Klapoetke et al. 2017, Nature 551:237; Ache et al. 2019, Curr Biol 29:1073.
  LPi      lobula plate intrinsic neurons, glutamatergic and inhibitory in the lobula plate.
           Mauss et al. 2015, Cell 162:351.
  GF       Giant Fiber (DNp01), descending neuron whose spike triggers the escape takeoff.
           von Reyn et al. 2014, Nat Neurosci 17:962.

Dorsoventral gradient
  LPLC2 neurons with dorsal receptive fields drive the GF more strongly than ventral
  ones: GF responses to activating dorsal LPLC2 were significantly larger than to
  ventral (Dombrovski et al. 2023, Nature 613:534, "Synaptic gradients transform
  object location to action"). A threat from above makes biological sense to escape.

  This is a qualitative result. The simulator does NOT load FlyWire synapse counts;
  it only checks whether evolution arrives at the same or the opposite gradient.
  In Flappy Bird a flap moves the fly up, so an obstacle *below* is the one to flee,
  and the task is expected to favour the reverse (ventral > dorsal) gradient.
"""
import numpy as np
from circuit import SL_LPLC2, N_LPLC2

BIOLOGICAL_GRADIENT = "dorsal > ventral"


def gradient_slope(weights):
    """Rank correlation between unit position (0 = dorsal) and weight, in [-1, 1]."""
    w = np.asarray(weights, dtype=np.float64)
    if np.ptp(w) == 0:
        return 0.0
    pos = np.arange(len(w), dtype=np.float64)
    rank = np.argsort(np.argsort(w)).astype(np.float64)
    return float(np.corrcoef(pos, rank)[0, 1])


def evolved_gradient(genome):
    rho = gradient_slope(genome[SL_LPLC2])
    if rho > 0.3:
        return "ventral > dorsal", rho
    if rho < -0.3:
        return "dorsal > ventral", rho
    return "flat", rho


def compare_to_biology(genome):
    label, rho = evolved_gradient(genome)
    w = genome[SL_LPLC2]
    half = N_LPLC2 // 2
    verdict = ("matches" if label == BIOLOGICAL_GRADIENT else
               "is the reverse of" if label != "flat" else "shows no clear gradient vs.")
    return (f"LPLC2 -> GF gradient: {label} (Spearman rho vs. ventral position = {rho:+.2f};"
            f" dorsal half sum {w[:half].sum():.1f}, ventral half sum {w[half:].sum():.1f}).\n"
            f"This {verdict} the biological fly ({BIOLOGICAL_GRADIENT}, Dombrovski et al. 2023).")
