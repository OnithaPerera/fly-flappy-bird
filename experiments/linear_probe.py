"""
Upper-bound check: how well can a plain linear readout of the compound eye fly?

    python experiments/linear_probe.py

A policy flaps when w . [nearness, loom, vy/10, 1] > 0. The weights are found with
the cross-entropy method on training courses and scored on the same held-out test
courses as the spiking circuit. If this baseline can't fly, the eye can't support the
task and tuning the circuit or the GA won't help.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import N_OMMATIDIA, FPS, LOOM_MAX                      # noqa: E402
from evolution import harmonic_mean, episode_fitness, test_seeds   # noqa: E402
from world import World                                            # noqa: E402
import retina                                                      # noqa: E402

DIM = 2 * N_OMMATIDIA + 2


def fly(weights, seed, max_frames):
    w = World(len(weights), seed, max_frames)
    prev = None
    while not w.done:
        near = retina.sample(w.y, w.obstacles())
        loom = np.zeros_like(near) if prev is None else np.clip((near - prev) * FPS, 0, LOOM_MAX)
        prev = near
        features = np.concatenate([near, loom, (w.vy / 10)[:, None], np.ones((len(weights), 1))], axis=1)
        w.step((features * weights).sum(axis=1) > 0)
    return episode_fitness(w), w.pipes_passed


def main(iterations=30, samples=100, elite=20, courses=4, max_frames=3000):
    rng = np.random.default_rng(0)
    mean, std = np.zeros(DIM), np.ones(DIM)
    for it in range(iterations):
        weights = mean + std * rng.standard_normal((samples, DIM))
        fit = harmonic_mean(np.column_stack([fly(weights, it * 10 + j, max_frames)[0] for j in range(courses)]))
        best = np.argsort(fit)[::-1][:elite]
        mean, std = weights[best].mean(axis=0), weights[best].std(axis=0) + 0.02
        print(f"iter {it:2d} | best {fit.max():7.0f} | mean {fit.mean():7.0f}")
    pipes = [int(fly(mean[None, :], s, 6000)[1][0]) for s in test_seeds(20)]
    print(f"\nLinear readout on 20 held-out courses: mean {np.mean(pipes):.1f} pipes, "
          f"median {np.median(pipes):.0f}, max {max(pipes)}")


if __name__ == "__main__":
    main()
