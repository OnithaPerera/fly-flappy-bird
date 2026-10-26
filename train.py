"""
Headless training and evaluation.

    python train.py --generations 60            # evolve, save champion_genome.npz
    python train.py --eval champion_genome.npz  # score a saved genome on held-out courses
"""
import argparse
import csv
import time
import numpy as np
from config import PIPE_SPEED, MAX_FRAMES
from circuit import SL_LPLC2, SL_LPI
from evolution import Evolution, evaluate, test_seeds, pick_champion, save_champion, load_champion
from connectome import compare_to_biology


def report(genome, n_test, max_frames):
    fit, pipes = evaluate(genome[None, :], test_seeds(n_test), max_frames)
    capped = int((fit[0] >= max_frames * PIPE_SPEED - 1e-6).sum())
    print(f"\nHeld-out test ({n_test} unseen courses, cap {max_frames} frames):")
    print(f"  pipes passed  mean {pipes.mean():.1f}  median {np.median(pipes):.0f}"
          f"  min {pipes.min()}  max {pipes.max()}")
    print(f"  courses flown to the cap: {capped}/{n_test}")
    print("\nEvolved LPLC2 -> GF weights, dorsal to ventral:",
          " ".join(f"{w:5.1f}" for w in genome[SL_LPLC2]))
    print("Evolved LPi   -> GF weights, dorsal to ventral:",
          " ".join(f"{w:5.1f}" for w in genome[SL_LPI]))
    print(compare_to_biology(genome))
    return pipes[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0, help="GA random seed")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--test-courses", type=int, default=50)
    ap.add_argument("--val-courses", type=int, default=20)
    ap.add_argument("--out", default="champion_genome.npz")
    ap.add_argument("--log", default="training_log.csv")
    ap.add_argument("--resume", help="seed the population with a saved genome")
    ap.add_argument("--eval", metavar="GENOME", help="only evaluate a saved genome")
    args = ap.parse_args()

    if args.eval:
        genome = load_champion(args.eval)
        if genome is None:
            raise SystemExit(f"Could not load {args.eval}")
        report(genome, args.test_courses, args.max_frames)
        return

    init = load_champion(args.resume) if args.resume else None
    evo = Evolution(seed=args.seed, init=init)
    with open(args.log, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["generation", "best_fitness", "mean_fitness", "best_pipes_mean", "record", "seconds"])
        for _ in range(args.generations):
            t0 = time.perf_counter()
            gen = evo.generation
            scores, pipes = evaluate(evo.population, evo.seeds(), args.max_frames)
            fit = evo.tell(scores)
            best = int(np.argmax(fit))
            secs = time.perf_counter() - t0
            log.writerow([gen, f"{fit[best]:.1f}", f"{fit.mean():.1f}", f"{pipes[best].mean():.2f}",
                          f"{evo.record:.1f}", f"{secs:.2f}"])
            f.flush()
            print(f"gen {gen:4d} | best {fit[best]:8.0f} px ({pipes[best].mean():5.1f} pipes)"
                  f" | mean {fit.mean():7.0f} | record {evo.record:8.0f} | {secs:5.1f}s")

    candidates = evo.hall_of_fame[-8:]
    champion, val = pick_champion(candidates, args.val_courses, args.max_frames)
    print(f"\nValidation ({args.val_courses} courses), last {len(candidates)} record holders:",
          " ".join(f"{v:.1f}" for v in val), "pipes")
    save_champion(args.out, champion, generations=evo.generation - 1, ga_seed=args.seed,
                  record=evo.record)
    print(f"Saved champion to {args.out}")
    report(champion, args.test_courses, args.max_frames)


if __name__ == "__main__":
    main()
