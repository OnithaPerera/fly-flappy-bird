"""
Pygame drawing of the arena: background, pipes, ground, the swarm and the leader's
lines of sight. All physics lives in world.py; this module only looks at it.
"""
import math
import numpy as np
import pygame
from config import (ARENA_WIDTH, ARENA_HEIGHT, GROUND_Y, FLY_X, PIPE_GAP, PIPE_WIDTH,
                    COLOR_LEADER, COLOR_ACCENT)
import retina


class ArenaView:
    def __init__(self, assets):
        self.assets = assets
        self.surface = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT))
        self.cap_h = assets["pipe_cap"].get_height()
        self.top_cap = pygame.transform.flip(assets["pipe_cap"], False, True)
        self._body_cache = {}
        self.rays = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT), pygame.SRCALPHA)

    def _body(self, height):
        h = max(1, int(height))
        if h not in self._body_cache:
            self._body_cache[h] = pygame.transform.scale(self.assets["pipe_body"], (PIPE_WIDTH, h))
        return self._body_cache[h]

    def draw(self, world, leader, distances=None):
        s = self.surface
        s.blit(self.assets["background"], (0, 0))
        for x, gap_y, _ in world.pipes:
            x = int(x)
            top = int(gap_y - PIPE_GAP / 2)
            bottom = int(gap_y + PIPE_GAP / 2)
            if top - self.cap_h > 0:
                s.blit(self._body(top - self.cap_h), (x, 0))
            s.blit(self.top_cap, (x, top - self.cap_h))
            s.blit(self.assets["pipe_cap"], (x, bottom))
            if GROUND_Y - bottom - self.cap_h > 0:
                s.blit(self._body(GROUND_Y - bottom - self.cap_h), (x, bottom + self.cap_h))
        gx = -(world.scroll % ARENA_WIDTH)
        s.blit(self.assets["ground"], (gx, GROUND_Y))
        s.blit(self.assets["ground"], (gx + ARENA_WIDTH, GROUND_Y))

        if leader >= 0 and distances is not None:
            self._draw_rays(world.y[leader], distances)
        for i in np.flatnonzero(world.alive):
            if i != leader:
                self._draw_fly(world, i, alpha=80)
        if leader >= 0:
            self._draw_fly(world, leader, alpha=255)
            pygame.draw.circle(s, COLOR_LEADER, (FLY_X, int(world.y[leader])), 28, 2)
        return s

    def _draw_rays(self, y, distances):
        """The leader's ommatidia: each ray stops where it meets a surface, brighter when near."""
        self.rays.fill((0, 0, 0, 0))
        xs, ys = retina.ray_endpoints(y, distances)
        for ex, ey, d in zip(xs, ys, distances):
            n = max(0.0, 1.0 - d / retina.EYE_RANGE)
            pygame.draw.line(self.rays, (*COLOR_ACCENT, int(60 + 150 * n)), (FLY_X, y), (ex, ey), 2 if n > 0 else 1)
            if n > 0:
                pygame.draw.circle(self.rays, (255, 255, 255, int(120 + 135 * n)), (int(ex), int(ey)), 3)
        self.surface.blit(self.rays, (0, 0))

    def _draw_fly(self, world, i, alpha):
        if world.flapped[i] or world.cooldown[i] > 1:
            sprite = self.assets["fly_0"]
        elif world.vy[i] > 0:
            sprite = self.assets["fly_2"]
        else:
            sprite = self.assets["fly_1"]
        angle = max(-45.0, min(30.0, -math.degrees(math.atan2(world.vy[i], 10))))
        img = pygame.transform.rotate(sprite, angle)
        img.set_alpha(alpha)
        self.surface.blit(img, img.get_rect(center=(FLY_X, int(world.y[i]))))
