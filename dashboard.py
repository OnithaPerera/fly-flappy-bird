"""
Telemetry panels to the right of the arena.

Layout (1600 x 900 canvas):
    arena | header bar ------------------------------------------------
          | brain view (500 x 400)          | score
          | population legend               | compound eye
          | activity: spike raster + GF Vm  | GF input currents
          |                                 | synaptic gradient
          |                                 | evolution
"""
import math
import numpy as np
import pygame
from config import (ARENA_WIDTH, CANVAS_WIDTH, CANVAS_HEIGHT, COLOR_BG, COLOR_TEXT, COLOR_ACCENT,
                    COLOR_PHOSPHOR, COLOR_LEADER, N_LPLC2, N_LPI, V_REST)
from retina import ELEVATIONS
from brain_visualizer import VIEW_W, VIEW_H, RASTER_LEN, GROUPS
from circuit import SL_LPLC2, SL_LPI
from connectome import BIOLOGICAL_GRADIENT, evolved_gradient

CARD = (15, 21, 34)
CARD_BORDER = (32, 44, 64)
MUTED = (122, 138, 163)
DIM = (44, 56, 78)
PLOT_BG = (6, 9, 15)
GOLD = (255, 204, 0)
EXC = (70, 230, 160)
INH = (255, 80, 140)

PAD = 12
HEADER_H = 44
LAB_X = ARENA_WIDTH + PAD
HUD_X = LAB_X + VIEW_W + PAD
HUD_W = CANVAS_WIDTH - HUD_X - PAD
GUTTER = 52        # left label gutter shared by the raster and the Vm trace
V_MIN, V_MAX = -80.0, 25.0


class Dashboard:
    def __init__(self, brain):
        self.brain = brain
        self.f_small = pygame.font.SysFont("Consolas", 11)
        self.f_title = pygame.font.SysFont("Consolas", 12, bold=True)
        self.f_body = pygame.font.SysFont("Consolas", 13)
        self.f_big = pygame.font.SysFont("Consolas", 30, bold=True)
        self.f_head = pygame.font.SysFont("Consolas", 15, bold=True)
        self.f_banner = pygame.font.SysFont("Consolas", 18, bold=True)

    # ---- primitives ---------------------------------------------------------

    def text(self, surf, s, font, color, pos, anchor="topleft"):
        img = font.render(s, True, color)
        rect = img.get_rect(**{anchor: pos})
        surf.blit(img, rect)
        return rect

    def card(self, surf, rect, title=None, note=None):
        """Draws a card and returns the y where its content starts."""
        pygame.draw.rect(surf, CARD, rect, border_radius=6)
        pygame.draw.rect(surf, CARD_BORDER, rect, 1, border_radius=6)
        if title:
            self.text(surf, title, self.f_title, COLOR_ACCENT, (rect.x + PAD, rect.y + 10))
        if note:
            self.text(surf, note, self.f_small, MUTED, (rect.right - PAD, rect.y + 11), "topright")
        return rect.y + (30 if title or note else PAD)

    def pill(self, surf, right, y, label, color, filled=False):
        """Right-aligned pill; returns the x where the next pill (to its left) should end."""
        img = self.f_title.render(label, True, COLOR_BG if filled else color)
        r = pygame.Rect(0, y, img.get_width() + 16, 22)
        r.right = right
        pygame.draw.rect(surf, color, r, 0 if filled else 1, border_radius=11)
        surf.blit(img, img.get_rect(center=r.center))
        return r.x - 6

    # ---- frame --------------------------------------------------------------

    def draw(self, surf, s):
        pygame.draw.line(surf, CARD_BORDER, (ARENA_WIDTH, 0), (ARENA_WIDTH, CANVAS_HEIGHT), 1)
        self.draw_header(surf, s)
        self.draw_lab(surf, s)
        self.draw_hud(surf, s)
        if s.banner:
            self.draw_banner(surf, s.banner)

    def draw_header(self, surf, s):
        y = (HEADER_H - 22) // 2
        title = self.text(surf, "DROSOPHILA LOOMING-ESCAPE CIRCUIT", self.f_head, COLOR_TEXT, (LAB_X, y + 2))
        self.text(surf, "vision-driven spiking controller", self.f_small, MUTED, (title.right + 10, y + 5))
        mode_color = {"REPLAY": COLOR_LEADER, "PAUSED": MUTED}.get(s.mode, COLOR_PHOSPHOR)
        x = self.pill(surf, CANVAS_WIDTH - PAD, y, s.mode, mode_color, filled=True)
        x = self.pill(surf, x, y, f"{s.speed}x", COLOR_TEXT)
        x = self.pill(surf, x, y, s.course, COLOR_TEXT)
        if s.replay_results is None:
            self.pill(surf, x, y, f"GEN {s.generation}", COLOR_TEXT)

    # ---- centre column ------------------------------------------------------

    def draw_lab(self, surf, s):
        view = pygame.Rect(LAB_X, HEADER_H + 4, VIEW_W, VIEW_H)
        self.brain.draw(surf, view.topleft)
        pygame.draw.rect(surf, CARD_BORDER, view, 1, border_radius=4)
        self.text(surf, f"leader fly #{s.leader}", self.f_small, MUTED, (view.x + 8, view.y + 6))
        self.text(surf, "anatomy procedural, activity simulated", self.f_small, DIM,
                  (view.right - 8, view.y + 6), "topright")

        # Population legend with live per-unit firing rates
        x, y = view.x, view.bottom + 8
        for key, label, color, _ in GROUPS:
            pygame.draw.rect(surf, color, (x, y + 3, 8, 8), border_radius=2)
            r = self.text(surf, label, self.f_small, MUTED, (x + 12, y))
            r = self.text(surf, f"{self.brain.rate[key]:4.0f} Hz", self.f_small, COLOR_TEXT, (r.right + 5, y))
            x = r.right + 16

        act = pygame.Rect(LAB_X, y + 22, VIEW_W, CANVAS_HEIGHT - (y + 22) - PAD)
        top = self.card(surf, act, "SPIKES", f"last {RASTER_LEN // 60} s  ·  newest on the right")
        plot_x, plot_w = act.x + GUTTER, act.w - GUTTER - PAD
        raster = pygame.Rect(plot_x, top + 2, plot_w, int((act.bottom - top) * 0.58))
        pygame.draw.rect(surf, PLOT_BG, raster)
        self.brain.draw_raster(surf, raster, plot_x - 8)
        self.text(surf, "GIANT FIBER Vm", self.f_title, COLOR_ACCENT, (act.x + PAD, raster.bottom + 10))
        self.text(surf, "mV", self.f_small, MUTED, (act.right - PAD, raster.bottom + 11), "topright")
        scope = pygame.Rect(plot_x, raster.bottom + 30, plot_w, act.bottom - raster.bottom - 42)
        self.draw_scope(surf, scope, s.trace, s.vth)

    def draw_scope(self, surf, plot, trace, vth):
        pygame.draw.rect(surf, PLOT_BG, plot)

        def to_y(v):
            return plot.bottom - (min(max(v, V_MIN), V_MAX) - V_MIN) / (V_MAX - V_MIN) * plot.h

        y = to_y(V_REST)
        pygame.draw.line(surf, (24, 33, 49), (plot.x, y), (plot.right, y))
        self.text(surf, "-70", self.f_small, MUTED, (plot.x - 8, y), "midright")
        y = to_y(vth)
        for x in range(plot.x, plot.right, 8):
            pygame.draw.line(surf, COLOR_LEADER, (x, y), (min(x + 4, plot.right), y))
        self.text(surf, "Vth", self.f_small, COLOR_LEADER, (plot.x - 8, y), "midright")
        if len(trace) > 1:
            dx = plot.w / RASTER_LEN
            offset = RASTER_LEN - len(trace)   # right-align with the raster's newest column
            pts = [(plot.x + (offset + i) * dx, to_y(v)) for i, v in enumerate(trace)]
            pygame.draw.lines(surf, (10, 110, 40), False, pts, 3)
            pygame.draw.lines(surf, COLOR_PHOSPHOR, False, pts, 1)

    # ---- right column -------------------------------------------------------

    def draw_hud(self, surf, s):
        x, w = HUD_X, HUD_W
        y = HEADER_H + 4

        # Score
        r = pygame.Rect(x, y, w, 96)
        self.card(surf, r)
        half = r.x + w // 2
        self.text(surf, "PIPES", self.f_small, MUTED, (r.x + PAD, r.y + 10))
        self.text(surf, f"{s.pipes}", self.f_big, COLOR_ACCENT, (r.x + PAD, r.y + 24))
        self.text(surf, s.best_label, self.f_small, MUTED, (half, r.y + 10))
        self.text(surf, f"{s.best_pipes}", self.f_big, GOLD, (half, r.y + 24))
        bar = pygame.Rect(r.x + PAD, r.y + 72, w - 130, 8)
        pygame.draw.rect(surf, DIM, bar, border_radius=4)
        fill = bar.copy()
        fill.w = int(bar.w * s.alive / max(1, s.population))
        if fill.w:
            pygame.draw.rect(surf, COLOR_PHOSPHOR, fill, border_radius=4)
        self.text(surf, f"{s.alive}/{s.population} alive", self.f_small, COLOR_TEXT,
                  (r.right - PAD, r.y + 69), "topright")

        r = pygame.Rect(x, r.bottom + 10, w, 168)
        top = self.card(surf, r, "COMPOUND EYE", "nearness · loom")
        self.draw_eye(surf, pygame.Rect(r.x, top, w, r.bottom - top - 6), s.near, s.loom)

        r = pygame.Rect(x, r.bottom + 10, w, 132)
        top = self.card(surf, r, "GF INPUT", "mV")
        self.draw_currents(surf, r, top, s.currents)

        r = pygame.Rect(x, r.bottom + 10, w, 150)
        top = self.card(surf, r, "SYNAPSES ONTO GF")
        k = self.text(surf, "LPi −", self.f_small, (255, 70, 175), (r.right - PAD, r.y + 11), "topright")
        self.text(surf, "LPLC2 +", self.f_small, (255, 168, 40), (k.x - 10, r.y + 11), "topright")
        self.draw_weights(surf, r, top, s.genome)

        r = pygame.Rect(x, r.bottom + 10, w, CANVAS_HEIGHT - (r.bottom + 10) - 40)
        if s.replay_results is not None:
            top = self.card(surf, r, "CHAMPION ON UNSEEN COURSES", "pipes per course")
            chart = pygame.Rect(r.x + PAD, top, w - 2 * PAD, r.bottom - top - 26)
            self.draw_results(surf, chart, s.replay_results)
            done = s.replay_results
            mean = f"mean {sum(done) / len(done):.1f} over {len(done)}" if done else "first course in progress"
            self.text(surf, mean, self.f_small, MUTED, (r.x + PAD, chart.bottom + 8))
        else:
            top = self.card(surf, r, "EVOLUTION", f"record {s.record:,.0f} px")
            chart = pygame.Rect(r.x + PAD, top, w - 2 * PAD, r.bottom - top - 26)
            self.draw_fitness(surf, chart, s.best_hist, s.mean_hist)
            self.text(surf, f"diversity σ² {s.variance:.3f}   ·   pulses {s.pulses}", self.f_small, MUTED,
                      (r.x + PAD, chart.bottom + 8))

        kx, ky = x, CANVAS_HEIGHT - 30
        for key, desc in (("P", "pause"), ("R", "replay"), ("N", "next"), ("S", "save"),
                          ("1/2/5/0", "speed"), ("F11", "full")):
            k = self.text(surf, key, self.f_title, COLOR_ACCENT, (kx, ky))
            d = self.text(surf, desc, self.f_small, MUTED, (k.right + 4, ky + 1))
            kx = d.right + 10

    def draw_eye(self, surf, area, near, loom):
        """The ommatidia as a fan of wedges, the same geometry as the rays in the arena."""
        n = len(near)
        cx, cy = area.x + 58, area.centery
        r0, r1, r2 = 14, area.h // 2 - 8, area.h // 2
        # Wedge edges halfway between neighbouring ommatidia (they're denser at the horizon)
        el = ELEVATIONS
        edges = np.concatenate(([el[0] + (el[0] - el[1]) / 2], (el[:-1] + el[1:]) / 2,
                                [el[-1] - (el[-2] - el[-1]) / 2]))
        for k in range(n):
            a0, a1 = edges[k], edges[k + 1]
            v = float(near[k])
            col = (int(20 + 10 * v), int(40 + 190 * v), int(70 + 185 * v))
            wedge = [(cx + r0 * math.cos(a0), cy - r0 * math.sin(a0)),
                     (cx + r1 * math.cos(a0), cy - r1 * math.sin(a0)),
                     (cx + r1 * math.cos(a1), cy - r1 * math.sin(a1)),
                     (cx + r0 * math.cos(a1), cy - r0 * math.sin(a1))]
            pygame.draw.polygon(surf, col, wedge)
            lv = float(min(1.0, loom[k]))
            if lv > 0.02:
                ring = [(cx + r1 * math.cos(a0), cy - r1 * math.sin(a0)),
                        (cx + r2 * math.cos(a0), cy - r2 * math.sin(a0)),
                        (cx + r2 * math.cos(a1), cy - r2 * math.sin(a1)),
                        (cx + r1 * math.cos(a1), cy - r1 * math.sin(a1))]
                pygame.draw.polygon(surf, (255, int(120 + 100 * lv), 40), ring)
        pygame.draw.circle(surf, COLOR_LEADER, (cx, cy), 4)
        tx = cx + r2 + 24
        self.text(surf, "D", self.f_title, MUTED, (area.x + PAD, area.y + 4))
        self.text(surf, "V", self.f_title, MUTED, (area.x + PAD, area.bottom - 4), "bottomleft")
        lines = ["Each wedge is one ommatidium.",
                 "Brighter = surface is nearer.",
                 "Orange rim = surface approaching",
                 "(looming), which drives LPLC2.",
                 "",
                 "This image is the fly's only",
                 "view of the course."]
        for i, line in enumerate(lines):
            self.text(surf, line, self.f_small, MUTED, (tx, area.y + 14 + i * 15))

    def draw_currents(self, surf, card, y, currents):
        bar_x = card.x + 74
        bar_w = card.right - PAD - bar_x - 46
        mid = bar_x + bar_w // 2
        for label, key in (("LPLC2", "exc"), ("LPi", "inh"), ("PROP", "prop"), ("tonic", "tonic"), ("NET", "net")):
            val = currents.get(key, 0.0)
            self.text(surf, label, self.f_body, COLOR_TEXT if key == "net" else MUTED, (card.x + PAD, y))
            w = int(np.clip(val / 40.0, -1, 1) * (bar_w // 2))
            bar = pygame.Rect(min(mid, mid + w), y + 4, abs(w), 9)
            pygame.draw.rect(surf, EXC if val >= 0 else INH, bar, border_radius=2)
            self.text(surf, f"{val:+5.1f}", self.f_small, COLOR_TEXT, (card.right - PAD, y + 1), "topright")
            y += 19
        pygame.draw.line(surf, (70, 84, 108), (mid, card.y + 30), (mid, y - 4), 1)

    def draw_weights(self, surf, card, y, genome):
        area = pygame.Rect(card.x + PAD, y, card.w - 2 * PAD, card.bottom - y - 30)
        mid = area.centery
        pygame.draw.line(surf, DIM, (area.x + 14, mid), (area.right - 14, mid))
        self.text(surf, "D", self.f_title, MUTED, (area.x, mid), "midleft")
        self.text(surf, "V", self.f_title, MUTED, (area.right, mid), "midright")
        area = area.inflate(-28, 0)
        if genome is None:
            return
        n = max(N_LPLC2, N_LPI)
        slot = area.w / n
        bw = max(4, int(slot * 0.55))
        for k in range(n):
            bx = int(area.x + (k + 0.5) * slot - bw / 2)
            if k < N_LPLC2:
                h = int(genome[SL_LPLC2][k] / 40.0 * (area.h / 2 - 2))
                pygame.draw.rect(surf, (255, 168, 40), (bx, mid - h, bw, h), border_radius=2)
            if k < N_LPI:
                h = int(-genome[SL_LPI][k] / 40.0 * (area.h / 2 - 2))
                pygame.draw.rect(surf, (255, 70, 175), (bx, mid + 1, bw, h), border_radius=2)
        label, rho = evolved_gradient(genome)
        self.text(surf, f"evolved: {label} (ρ {rho:+.2f})", self.f_small, COLOR_TEXT, (card.x + PAD, card.bottom - 26))
        self.text(surf, f"real fly: {BIOLOGICAL_GRADIENT} (Dombrovski 2023)", self.f_small, MUTED,
                  (card.x + PAD, card.bottom - 13))

    def draw_fitness(self, surf, rect, best, mean):
        pygame.draw.rect(surf, PLOT_BG, rect, border_radius=3)
        if len(best) < 2:
            self.text(surf, "awaiting 2nd generation…", self.f_small, MUTED, rect.center, "center")
            return
        top = max(1.0, max(best))
        for series, color, width in ((mean, MUTED, 1), (best, COLOR_ACCENT, 2)):
            pts = [(rect.x + i / (len(series) - 1) * rect.w,
                    rect.bottom - 4 - max(0.0, v) / top * (rect.h - 8)) for i, v in enumerate(series)]
            pygame.draw.lines(surf, color, False, pts, width)
        legend = self.text(surf, "best", self.f_small, COLOR_ACCENT, (rect.x + 6, rect.y + 4))
        self.text(surf, "mean", self.f_small, MUTED, (legend.right + 8, rect.y + 4))

    def draw_results(self, surf, rect, results):
        pygame.draw.rect(surf, PLOT_BG, rect, border_radius=3)
        if not results:
            return
        shown = results[-24:]
        top = max(max(shown), 1)
        slot = rect.w / 24
        for i, v in enumerate(shown):
            h = int(v / top * (rect.h - 22))
            bar = pygame.Rect(int(rect.x + i * slot + 2), rect.bottom - 4 - h, max(3, int(slot) - 4), h)
            pygame.draw.rect(surf, GOLD, bar, border_radius=2)
        self.text(surf, f"{top}", self.f_small, MUTED, (rect.x + 6, rect.y + 4))

    def draw_banner(self, surf, msg):
        overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        surf.blit(overlay, (0, 0))
        img = self.f_banner.render(msg, True, COLOR_TEXT)
        box = img.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2)).inflate(40, 24)
        pygame.draw.rect(surf, CARD, box, border_radius=8)
        pygame.draw.rect(surf, COLOR_ACCENT, box, 1, border_radius=8)
        surf.blit(img, img.get_rect(center=box.center))
