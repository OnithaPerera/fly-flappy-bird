import pygame
import math
import random
import numpy as np
from config import COLOR_PANEL, COLOR_GRID, COLOR_TEXT, COLOR_ACCENT

# Helper to convert hex to RGB
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

class Particle:
    def __init__(self, start_pos, end_pos, color, speed=3.0, size=2, tail=False):
        self.x, self.y = start_pos
        self.target_x, self.target_y = end_pos
        self.color = color
        self.speed = speed
        self.size = size
        self.tail = tail
        self.active = True
        
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.vx = (dx / dist) * speed
            self.vy = (dy / dist) * speed
        else:
            self.vx, self.vy = 0, 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Check if reached target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        if math.hypot(dx, dy) < self.speed:
            self.active = False
            
    def draw(self, surface):
        if not self.active: return
        pos = (int(self.x), int(self.y))
        pygame.draw.circle(surface, self.color, pos, self.size)
        if self.tail:
            tail_pos = (int(self.x - self.vx * 3), int(self.y - self.vy * 3))
            pygame.draw.line(surface, self.color, tail_pos, pos, self.size)

class BrainVisualizer:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.center_x = self.rect.x + self.rect.width // 2
        self.center_y = self.rect.y + self.rect.height // 2 - 50
        
        # Define functional regions mapping
        self.nodes = {
            "lpi_l": (self.center_x - 70, self.center_y - 60),
            "lpi_r": (self.center_x + 70, self.center_y - 60),
            "lplc2_l": (self.center_x - 130, self.center_y),
            "lplc2_r": (self.center_x + 130, self.center_y),
            "central_complex": (self.center_x, self.center_y),
            "mn": (self.center_x, self.center_y + 190)
        }
        
        self.gf_l_start = (self.center_x - 15, self.center_y + 30)
        self.gf_r_start = (self.center_x + 15, self.center_y + 30)
        self.gf_l_end = (self.center_x - 10, self.center_y + 190)
        self.gf_r_end = (self.center_x + 10, self.center_y + 190)
        
        self.particles = []
        self.gf_flash_alpha = 0
        self.mn_flash_alpha = 0
        self.font = pygame.font.SysFont("Consolas", 12)
        
        # 1. Precomputed Anatomical Filament Mesh
        self.fibers = []
        self._generate_fibers()

    def _generate_fibers(self):
        def add_fibers(f_type, cx, cy, count, base_angle, spread_angle, min_r, max_r, segments, seg_len, curve):
            for _ in range(count):
                pts = []
                angle = base_angle + random.uniform(-spread_angle, spread_angle)
                r = random.uniform(min_r, max_r)
                px, py = cx + math.cos(angle)*r, cy + math.sin(angle)*r
                pts.append((px, py))
                
                cur_angle = angle
                for _ in range(segments):
                    cur_angle += random.uniform(-curve, curve)
                    sl = random.uniform(seg_len * 0.5, seg_len * 1.5)
                    px += math.cos(cur_angle)*sl
                    py += math.sin(cur_angle)*sl
                    pts.append((px, py))
                self.fibers.append({"type": f_type, "points": pts})

        # Optic Lobes Left (Kidney-shaped outward fanning)
        add_fibers("optic_l", self.center_x - 120, self.center_y, 70, math.pi, math.pi/1.5, 10, 40, 8, 12, 0.4)
        
        # Optic Lobes Right
        add_fibers("optic_r", self.center_x + 120, self.center_y, 70, 0, math.pi/1.5, 10, 40, 8, 12, 0.4)
        
        # Central Complex & Fan-Shaped Body (Center Core)
        add_fibers("central", self.center_x, self.center_y, 70, 0, math.pi, 5, 25, 8, 8, 1.2)
        
        # Antennal Lobes & Subesophageal Zone (Lower Center)
        add_fibers("antennal", self.center_x, self.center_y + 50, 60, 0, math.pi, 5, 20, 6, 8, 1.5)
        
        # Dorsal Calyces / Mushroom Bodies (Upper Arch)
        add_fibers("mushroom", self.center_x, self.center_y - 60, 60, -math.pi/2, math.pi/2.5, 5, 25, 7, 15, 0.5)
        
        # Descending Giant Fiber Tracts
        for _ in range(15):
            pts_l = []
            pts_r = []
            px_l = self.center_x - 12 + random.uniform(-6, 6)
            px_r = self.center_x + 12 + random.uniform(-6, 6)
            py = self.center_y + 20
            
            pts_l.append((px_l, py))
            pts_r.append((px_r, py))
            
            for _ in range(10):
                py += random.uniform(12, 20)
                px_l += random.uniform(-4, 4)
                px_r += random.uniform(-4, 4)
                pts_l.append((px_l, py))
                pts_r.append((px_r, py))
                
            self.fibers.append({"type": "giant", "points": pts_l})
            self.fibers.append({"type": "giant", "points": pts_r})

    def spawn_particles(self, src_key, dst_key, color, count=1, speed=3.0, size=2, tail=False):
        src = self.nodes[src_key]
        dst = self.nodes[dst_key]
        for _ in range(count):
            offset_x = random.uniform(-30, 30)
            offset_y = random.uniform(-30, 30)
            start_pos = (src[0] + offset_x, src[1] + offset_y)
            self.particles.append(Particle(start_pos, dst, color, speed, size, tail))
            
    def spawn_descending_wave(self):
        # High-velocity shockwave cascade
        for _ in range(8):
            offset_x = random.uniform(-5, 5)
            start_l = (self.gf_l_start[0] + offset_x, self.gf_l_start[1])
            start_r = (self.gf_r_start[0] + offset_x, self.gf_r_start[1])
            end_l = (self.gf_l_end[0] + offset_x, self.gf_l_end[1])
            end_r = (self.gf_r_end[0] + offset_x, self.gf_r_end[1])
            self.particles.append(Particle(start_l, end_l, (200, 255, 255), speed=20.0, size=4, tail=True))
            self.particles.append(Particle(start_r, end_r, (200, 255, 255), speed=20.0, size=4, tail=True))

    def update_and_draw(self, surface, inputs, gf_v, spiked):
        looming = inputs[0]
        gap_offset = inputs[1]
        vel = inputs[2]
        ground = inputs[3]
        
        lpi_act = max(0.0, -gap_offset)
        lplc2_act = max(0.0, gap_offset) + looming + ground
        
        # Dynamic glow calculations
        v_rest = -70.0
        v_thresh = -50.0 
        gf_fill = np.clip((gf_v - v_rest) / (v_thresh - v_rest), 0.0, 1.0)
        
        cyan_base = (0, 68, 102) # #004466
        cyan_glow = (0, 255, 255) # #00FFFF
        amber_base = (153, 92, 0) # Dimmed #FF9900
        amber_glow = (255, 204, 0) # #FFCC00
        crimson_base = (100, 20, 30)
        crimson_glow = (255, 51, 85) # #FF3355
        magenta_color = (204, 0, 255) # #CC00FF
        
        # Additive blend surfaces
        brain_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        glow_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        
        # Silhouette
        capsule_color = hex_to_rgb("#0D1B2A")
        
        center_x_local = self.rect.width // 2
        center_y_local = self.rect.height // 2 - 50
        
        # Central brain mass
        pygame.draw.ellipse(brain_surf, (*capsule_color, 89), (center_x_local - 80, center_y_local - 100, 160, 200))
        # Left optic lobe
        pygame.draw.ellipse(brain_surf, (*capsule_color, 89), (center_x_local - 180, center_y_local - 70, 140, 140))
        # Right optic lobe
        pygame.draw.ellipse(brain_surf, (*capsule_color, 89), (center_x_local + 40, center_y_local - 70, 140, 140))
        
        # Background structural fibers
        for f in self.fibers:
            pts = [(int(p[0] - self.rect.x), int(p[1] - self.rect.y)) for p in f["points"]]
            if len(pts) > 1:
                pygame.draw.lines(brain_surf, (13, 27, 42, 100), False, pts, 1)

        # Dynamic active foreground fibers
        for f in self.fibers:
            pts = [(int(p[0] - self.rect.x), int(p[1] - self.rect.y)) for p in f["points"]]
            if len(pts) < 2: continue
            
            if f["type"] in ("optic_l", "optic_r"):
                # Interpolate cyan
                glow_int = min(1.0, lplc2_act)
                r = int(cyan_base[0] + (cyan_glow[0] - cyan_base[0]) * glow_int)
                g = int(cyan_base[1] + (cyan_glow[1] - cyan_base[1]) * glow_int)
                b = int(cyan_base[2] + (cyan_glow[2] - cyan_base[2]) * glow_int)
                pygame.draw.lines(glow_surf, (r, g, b, 150), False, pts, 1)
            elif f["type"] == "central":
                glow_int = gf_fill
                r = int(amber_base[0] + (amber_glow[0] - amber_base[0]) * glow_int)
                g = int(amber_base[1] + (amber_glow[1] - amber_base[1]) * glow_int)
                b = int(amber_base[2] + (amber_glow[2] - amber_base[2]) * glow_int)
                pygame.draw.lines(glow_surf, (r, g, b, 180), False, pts, 1)
            elif f["type"] == "antennal":
                pygame.draw.lines(glow_surf, (80, 100, 120, 100), False, pts, 1)
            elif f["type"] == "mushroom":
                glow_int = min(1.0, lpi_act)
                r = int(crimson_base[0] + (crimson_glow[0] - crimson_base[0]) * glow_int)
                g = int(crimson_base[1] + (crimson_glow[1] - crimson_base[1]) * glow_int)
                b = int(crimson_base[2] + (crimson_glow[2] - crimson_base[2]) * glow_int)
                pygame.draw.lines(glow_surf, (r, g, b, 150), False, pts, 1)
            elif f["type"] == "giant":
                pygame.draw.lines(glow_surf, (*magenta_color, 120), False, pts, 1)

        # Spawns sparks
        if random.random() < lpi_act * 0.5:
            self.spawn_particles("lpi_l", "central_complex", crimson_glow, speed=6.0)
            self.spawn_particles("lpi_r", "central_complex", crimson_glow, speed=6.0)
            
        if random.random() < lplc2_act * 0.6:
            self.spawn_particles("lplc2_l", "central_complex", cyan_glow, speed=6.0)
            self.spawn_particles("lplc2_r", "central_complex", cyan_glow, speed=6.0)
            
        if spiked:
            self.gf_flash_alpha = 255
            self.mn_flash_alpha = 255
            self.spawn_descending_wave()
            
        self.gf_flash_alpha = max(0, self.gf_flash_alpha - 12)
        self.mn_flash_alpha = max(0, self.mn_flash_alpha - 12)
        
        # Central Complex Flash
        if self.gf_flash_alpha > 0:
            pygame.draw.circle(glow_surf, (150, 255, 255, int(self.gf_flash_alpha * 0.7)), (center_x_local, center_y_local), 60)
            pygame.draw.circle(glow_surf, (255, 255, 255, self.gf_flash_alpha), (center_x_local, center_y_local), 30)

        # Thoracic Motor Ganglion Flash
        mn_pos_local = (int(self.nodes["mn"][0] - self.rect.x), int(self.nodes["mn"][1] - self.rect.y))
        if self.mn_flash_alpha > 0:
            pygame.draw.circle(glow_surf, (0, 255, 0, int(self.mn_flash_alpha * 0.7)), mn_pos_local, 40)
            pygame.draw.circle(glow_surf, (100, 255, 100, self.mn_flash_alpha), mn_pos_local, 15)

        # Draw particles
        for p in self.particles[:]:
            p.update()
            if p.active:
                p_local = (int(p.x - self.rect.x), int(p.y - self.rect.y))
                pygame.draw.circle(glow_surf, p.color, p_local, p.size)
                if p.tail:
                    tail_pos = (int(p_local[0] - p.vx * 3), int(p_local[1] - p.vy * 3))
                    pygame.draw.line(glow_surf, p.color, tail_pos, p_local, p.size)
            else:
                self.particles.remove(p)

        # Blit layers
        surface.blit(brain_surf, (self.rect.x, self.rect.y))
        surface.blit(glow_surf, (self.rect.x, self.rect.y), special_flags=pygame.BLEND_ADD)
        
        # Labels
        cc_label = self.font.render("Central Complex", True, COLOR_TEXT)
        surface.blit(cc_label, (self.nodes["central_complex"][0] - cc_label.get_width()//2, self.nodes["central_complex"][1] - 80))
        
        mn_label = self.font.render("Thoracic Motor Ganglion", True, COLOR_TEXT)
        surface.blit(mn_label, (self.nodes["mn"][0] - mn_label.get_width()//2, self.nodes["mn"][1] + 20))
