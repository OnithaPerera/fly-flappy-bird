"""
Live view: watch the swarm evolve, or replay the champion on courses it has never seen.

    python main.py                         # evolve (seeds the swarm with champion_genome.npz if present)
    python main.py --fresh                 # evolve from random weights, ignoring the saved champion
    python main.py --replay                # fly the saved champion on held-out courses
"""
import argparse
from collections import deque
from types import SimpleNamespace

import numpy as np
import pygame

from diagnostics import run_preflight_checks
from config import (WINDOW_WIDTH, WINDOW_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, FPS, COLOR_BG,
                    SEEDS_PER_GENERATION, MAX_FRAMES)
from world import World
from circuit import FlyCircuit
from evolution import Evolution, episode_fitness, test_seeds, save_champion, load_champion
import retina
from game import ArenaView
from brain_visualizer import BrainVisualizer, RASTER_LEN
from dashboard import Dashboard
from assets_loader import load_or_fetch_assets
import sound_fx

CHECKPOINT = "champion_genome.npz"
SPEEDS = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_5: 5, pygame.K_0: 15}


class App:
    def __init__(self, args):
        self.saved = None if getattr(args, "fresh", False) else load_champion(CHECKPOINT)
        self.evo = Evolution(seed=args.seed, init=self.saved)
        self.replay = False
        self.paused = False
        self.finished = False      # replay course finished, waiting for N / R
        self.speed = 1
        self.course = 0            # index into the generation's seeds, or into test courses
        self.scores = np.zeros((self.evo.pop_size, SEEDS_PER_GENERATION))
        self.best_pipes = 0
        self.replay_results = []   # pipes passed on each finished unseen course
        self.trace = deque(maxlen=RASTER_LEN)
        self.brain = BrainVisualizer()
        if args.replay:
            self.toggle_replay()
        else:
            self.start_course()

    # ---- episodes -----------------------------------------------------------

    def champion(self):
        return self.evo.champion if self.evo.champion is not None else self.saved

    def start_course(self):
        if self.replay:
            seed = test_seeds(self.course + 1)[-1]
            genomes = self.champion()[None, :]
        else:
            seed = self.evo.seeds()[self.course]
            genomes = self.evo.population
        self.seed = seed
        self.world = World(len(genomes), seed, MAX_FRAMES)
        self.circuit = FlyCircuit(genomes)
        self.distances = None
        self.trace.clear()
        self.brain.reset()
        self.finished = False

    def end_course(self):
        if self.replay:
            self.best_pipes = max(self.best_pipes, int(self.world.pipes_passed[0]))
            self.replay_results.append(int(self.world.pipes_passed[0]))
            self.finished = True
            return
        self.scores[:, self.course] = episode_fitness(self.world)
        self.best_pipes = max(self.best_pipes, int(self.world.pipes_passed.max()))
        self.course += 1
        if self.course == SEEDS_PER_GENERATION:
            fit = self.evo.tell(self.scores)
            print(f"gen {self.evo.generation - 1:4d} | best {fit.max():8.0f} | mean {fit.mean():7.0f}"
                  f" | record {self.evo.record:8.0f}")
            self.course = 0
        self.start_course()

    def toggle_replay(self):
        if not self.replay and self.champion() is None:
            print("No champion yet: let a generation finish, or train one with train.py.")
            return
        self.replay = not self.replay
        self.course = 0
        self.best_pipes = 0
        self.replay_results = []
        self.start_course()

    def leader(self):
        """Lowest-index survivor. Elites come first in the population, so this is the
        best-ranked genome from the previous generation that is still flying."""
        alive = np.flatnonzero(self.world.alive)
        return int(alive[0]) if len(alive) else -1

    def step(self):
        w = self.world
        if self.finished:          # replay course over; wait for N or R
            return
        if w.done:
            self.end_course()
            return
        obstacles = w.obstacles()
        dist = retina.ray_distances(w.y, obstacles)
        near = np.clip(1.0 - dist / retina.EYE_RANGE, 0.0, 1.0)
        lead = self.leader()
        w.step(self.circuit.step(near, w.vy))
        if lead >= 0:
            self.distances = dist[lead]
            self.brain.step(self.circuit, lead, bool(w.flapped[lead]))
            spiked = bool(self.circuit.spk_gf[lead])
            self.trace.append(20.0 if spiked else float(self.circuit.gf_v[lead]))
            if spiked and self.speed <= 2:
                sound_fx.play_spike_click()

    # ---- input --------------------------------------------------------------

    def handle(self, event):
        if event.type != pygame.KEYDOWN:
            return
        k = event.key
        if k in SPEEDS:
            self.speed = SPEEDS[k]
        elif k == pygame.K_p:
            self.paused = not self.paused
        elif k == pygame.K_r:
            self.toggle_replay()
        elif k in (pygame.K_n, pygame.K_SPACE) and self.replay:
            self.course += 1           # a skipped course isn't counted in the results
            self.start_course()
        elif k == pygame.K_s:
            champ = self.champion()
            if champ is not None:
                save_champion(CHECKPOINT, champ, record=self.evo.record)
                self.saved = champ
                print(f"Saved champion to {CHECKPOINT}")

    # ---- telemetry ----------------------------------------------------------

    def hud(self):
        w, c, lead = self.world, self.circuit, self.leader()
        show = lead if lead >= 0 else 0
        if self.finished:
            banner = f"COURSE FINISHED: {w.pipes_passed[0]} pipes  ·  [N] next unseen course  ·  [R] back to evolution"
        elif self.paused:
            banner = "PAUSED  ·  [P] resume"
        else:
            banner = None
        mode = "REPLAY" if self.replay else ("PAUSED" if self.paused else "EVOLVING")
        course = (f"UNSEEN COURSE {self.course + 1}" if self.replay
                  else f"COURSE {self.course + 1}/{SEEDS_PER_GENERATION}")
        return SimpleNamespace(
            mode=mode, generation=self.evo.generation, course=course, speed=self.speed, banner=banner,
            leader=show, pipes=int(w.pipes_passed[show]),
            best_pipes=max(self.best_pipes, int(w.pipes_passed.max())),
            best_label="BEST (UNSEEN)" if self.replay else "BEST THIS RUN",
            alive=int(w.alive.sum()), population=w.n,
            near=c.near[show], loom=c.loom[show],
            currents={"exc": c.i_exc[show], "inh": c.i_inh[show], "prop": c.i_prop[show],
                      "tonic": c.gf_tonic[show],
                      "net": c.i_exc[show] + c.i_inh[show] + c.i_prop[show] + c.gf_tonic[show]},
            genome=c.genomes[show], vth=float(c.gf_vth[show]), trace=list(self.trace),
            best_hist=self.evo.best_history, mean_hist=self.evo.mean_history,
            record=self.evo.record, variance=self.evo.variance, pulses=self.evo.pulses,
            replay_results=self.replay_results if self.replay else None,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true", help=f"start by replaying {CHECKPOINT}")
    ap.add_argument("--fresh", action="store_true", help="start from random weights, ignoring the saved champion")
    ap.add_argument("--seed", type=int, default=0, help="GA random seed")
    args = ap.parse_args()

    run_preflight_checks()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Drosophila looming-escape circuit")
    canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    clock = pygame.time.Clock()

    assets = load_or_fetch_assets()
    if args.replay and args.fresh:
        raise SystemExit("--replay and --fresh can't be combined")
    if args.replay and load_champion(CHECKPOINT) is None:
        raise SystemExit(f"--replay needs {CHECKPOINT}; train one with: python train.py")
    app = App(args)
    arena = ArenaView(assets)
    dashboard = Dashboard(app.brain)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                if screen.get_flags() & pygame.FULLSCREEN:
                    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                else:
                    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
            else:
                app.handle(event)

        if not app.paused and not app.finished:
            for _ in range(app.speed):
                app.step()
                if app.finished:
                    break

        canvas.fill(COLOR_BG)
        canvas.blit(arena.draw(app.world, app.leader(), app.distances), (0, 0))
        dashboard.draw(canvas, app.hud())

        # Letterbox the fixed-size canvas into the window
        win_w, win_h = screen.get_size()
        scale = min(win_w / CANVAS_WIDTH, win_h / CANVAS_HEIGHT)
        size = (int(CANVAS_WIDTH * scale), int(CANVAS_HEIGHT * scale))
        screen.fill((7, 10, 19))
        screen.blit(pygame.transform.smoothscale(canvas, size), ((win_w - size[0]) // 2, (win_h - size[1]) // 2))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
