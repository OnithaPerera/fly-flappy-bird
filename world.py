"""
Headless, vectorised Flappy Bird physics for a swarm of flies.

The world knows nothing about neurons. It moves the flies, scrolls the pipes and
decides who dies. Pygame is not needed here, so training can run without a window.
"""
import random
import numpy as np
from config import (ARENA_WIDTH, GROUND_Y, CEILING_Y, FLY_X, FLY_RADIUS, GRAVITY,
                    FLAP_STRENGTH, FLAP_COOLDOWN, PIPE_SPEED, PIPE_SPACING, PIPE_GAP,
                    PIPE_WIDTH, FIRST_PIPE_X, GAP_MARGIN, MAX_GAP_SHIFT, MAX_FRAMES)

GAP_MIN = CEILING_Y + GAP_MARGIN + PIPE_GAP / 2
GAP_MAX = GROUND_Y - GAP_MARGIN - PIPE_GAP / 2


class World:
    def __init__(self, n_agents, seed, max_frames=MAX_FRAMES):
        self.n = n_agents
        self.seed = seed
        self.max_frames = max_frames
        # Private RNG so nothing else can perturb the seeded course
        self.rng = random.Random(seed)

        self.y = np.full(n_agents, (CEILING_Y + GROUND_Y) / 2.0)
        self.vy = np.zeros(n_agents)
        self.alive = np.ones(n_agents, dtype=bool)
        self.cooldown = np.zeros(n_agents, dtype=np.int32)
        self.flapped = np.zeros(n_agents, dtype=bool)   # stroke started this frame
        self.distance = np.zeros(n_agents)              # px flown before dying
        self.pipes_passed = np.zeros(n_agents, dtype=np.int32)
        self.death_cause = np.full(n_agents, "", dtype=object)
        self.gap_error = np.zeros(n_agents)             # |y - next gap centre| at death

        self.frame = 0
        self.scroll = 0.0
        self.pipes = []      # [x, gap_y, counted] for each pipe on screen
        self._spawn(FIRST_PIPE_X)

    # ---- course -------------------------------------------------------------

    def _spawn(self, x):
        lo, hi = GAP_MIN, GAP_MAX
        if self.pipes:
            prev = self.pipes[-1][1]
            lo, hi = max(lo, prev - MAX_GAP_SHIFT), min(hi, prev + MAX_GAP_SHIFT)
        self.pipes.append([float(x), self.rng.uniform(lo, hi), False])

    def obstacles(self):
        """Pipe rectangles as an (P, 4) array of [x0, x1, y0, y1], top and bottom halves."""
        rects = []
        for x, gap_y, _ in self.pipes:
            rects.append((x, x + PIPE_WIDTH, -1e6, gap_y - PIPE_GAP / 2))
            rects.append((x, x + PIPE_WIDTH, gap_y + PIPE_GAP / 2, 1e6))
        return np.array(rects, dtype=np.float64).reshape(-1, 4)

    def next_pipe(self):
        """The first pipe whose trailing edge is still ahead of the flies (for telemetry only)."""
        return next((p for p in self.pipes if p[0] + PIPE_WIDTH > FLY_X - FLY_RADIUS), None)

    @property
    def done(self):
        return not self.alive.any() or self.frame >= self.max_frames

    # ---- dynamics -----------------------------------------------------------

    def step(self, flap_commands):
        """Advance one frame. flap_commands: (N,) bool, a motor command from each fly."""
        if self.done:
            return
        live = self.alive
        self.flapped = live & np.asarray(flap_commands, dtype=bool) & (self.cooldown <= 0)
        self.vy[self.flapped] = FLAP_STRENGTH
        self.cooldown[self.flapped] = FLAP_COOLDOWN
        self.cooldown[live] -= 1

        self.vy[live] += GRAVITY
        self.y[live] += self.vy[live]
        self.distance[live] += PIPE_SPEED

        for p in self.pipes:
            p[0] -= PIPE_SPEED
        self.scroll += PIPE_SPEED
        if ARENA_WIDTH - self.pipes[-1][0] >= PIPE_SPACING:
            self._spawn(ARENA_WIDTH)
        self.pipes = [p for p in self.pipes if p[0] + PIPE_WIDTH > 0]

        self._collide()
        for p in self.pipes:
            if not p[2] and p[0] + PIPE_WIDTH < FLY_X - FLY_RADIUS:
                p[2] = True
                self.pipes_passed[self.alive] += 1
        self.frame += 1

    def _collide(self):
        live = self.alive.copy()
        top = self.y - FLY_RADIUS
        bottom = self.y + FLY_RADIUS
        self._kill(live & (top <= CEILING_Y), "ceiling")
        self._kill(live & (bottom >= GROUND_Y), "ground")
        for x, gap_y, _ in self.pipes:
            if x < FLY_X + FLY_RADIUS and x + PIPE_WIDTH > FLY_X - FLY_RADIUS:
                hit = (top < gap_y - PIPE_GAP / 2) | (bottom > gap_y + PIPE_GAP / 2)
                self._kill(self.alive & hit, "pipe")

    def _kill(self, mask, cause):
        if not mask.any():
            return
        self.alive[mask] = False
        self.death_cause[mask] = cause
        pipe = self.next_pipe()
        if pipe is not None:
            self.gap_error[mask] = np.abs(self.y[mask] - pipe[1])
