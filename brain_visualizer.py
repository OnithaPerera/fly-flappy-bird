import pygame
import math
import random
import numpy as np
from config import COLOR_PANEL, COLOR_GRID, COLOR_TEXT, COLOR_ACCENT

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
        
        # Define node positions
        self.nodes = {
            "lpi_l": (self.center_x - 120, self.center_y - 160),
            "lpi_r": (self.center_x + 120, self.center_y - 160),
            "lplc2_l": (self.center_x - 140, self.center_y + 80),
            "lplc2_r": (self.center_x + 140, self.center_y + 80),
            "haltere": (self.center_x - 180, self.center_y - 30),
            "gf": (self.center_x, self.center_y),
            "mn": (self.center_x, self.center_y + 200)
        }
        
        self.particles = []
        self.gf_flash_alpha = 0
        self.mn_flash_alpha = 0
        
        self.font = pygame.font.SysFont("Consolas", 12)
        
    def spawn_particles(self, src_key, dst_key, color, count=1, speed=3.0, size=2, tail=False):
        src = self.nodes[src_key]
        dst = self.nodes[dst_key]
        for _ in range(count):
            offset_x = random.uniform(-10, 10)
            offset_y = random.uniform(-10, 10)
            start_pos = (src[0] + offset_x, src[1] + offset_y)
            self.particles.append(Particle(start_pos, dst, color, speed, size, tail))

    def update_and_draw(self, surface, inputs, gf_v, spiked):
        """
        inputs: [looming, gap_offset, vel, ground]
        """
        # Parse inputs
        looming = inputs[0]
        gap_offset = inputs[1]
        vel = inputs[2]
        ground = inputs[3]
        
        lpi_act = max(0.0, -gap_offset) # Dorsal Inhibitory
        lplc2_act = max(0.0, gap_offset) + looming + ground # Ventral Excitatory
        haltere_act = min(1.0, abs(vel))
        
        # Spawn sensory particles
        if random.random() < lpi_act * 0.5:
            self.spawn_particles("lpi_l", "gf", (255, 50, 50), speed=4.0)
            self.spawn_particles("lpi_r", "gf", (255, 50, 50), speed=4.0)
            
        if random.random() < lplc2_act * 0.5:
            self.spawn_particles("lplc2_l", "gf", (0, 255, 200), speed=4.0)
            self.spawn_particles("lplc2_r", "gf", (0, 255, 200), speed=4.0)
            
        if random.random() < haltere_act * 0.5:
            self.spawn_particles("haltere", "gf", (255, 180, 0), speed=3.0)
            
        if spiked:
            self.gf_flash_alpha = 255
            self.mn_flash_alpha = 255
            self.spawn_particles("gf", "mn", (200, 255, 255), count=10, speed=12.0, size=3, tail=True)
            
        # Decay flashes
        self.gf_flash_alpha = max(0, self.gf_flash_alpha - 15)
        self.mn_flash_alpha = max(0, self.mn_flash_alpha - 15)
        
        # Draw connections
        pygame.draw.line(surface, COLOR_GRID, self.nodes["lpi_l"], self.nodes["gf"], 2)
        pygame.draw.line(surface, COLOR_GRID, self.nodes["lpi_r"], self.nodes["gf"], 2)
        pygame.draw.line(surface, COLOR_GRID, self.nodes["lplc2_l"], self.nodes["gf"], 3)
        pygame.draw.line(surface, COLOR_GRID, self.nodes["lplc2_r"], self.nodes["gf"], 3)
        pygame.draw.line(surface, COLOR_GRID, self.nodes["haltere"], self.nodes["gf"], 2)
        
        # Trunk
        pygame.draw.line(surface, (50, 60, 80), self.nodes["gf"], self.nodes["mn"], 6)
        
        # Update and draw particles
        for p in self.particles[:]:
            p.update()
            p.draw(surface)
            if not p.active:
                self.particles.remove(p)
                
        # Draw Nodes
        def draw_node(key, color, radius, label, glow_alpha=0):
            pos = self.nodes[key]
            # Outer glow
            if glow_alpha > 0:
                glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color, int(glow_alpha)), (radius*2, radius*2), radius*2)
                surface.blit(glow_surf, (pos[0]-radius*2, pos[1]-radius*2))
            
            # Core
            pygame.draw.circle(surface, (30, 40, 50), pos, radius)
            pygame.draw.circle(surface, color, pos, radius, 2)
            
            label_surf = self.font.render(label, True, COLOR_TEXT)
            surface.blit(label_surf, (pos[0] - label_surf.get_width()//2, pos[1] + radius + 5))
            
        # Draw LPi
        lpi_glow = min(255, int(lpi_act * 255))
        draw_node("lpi_l", (255, 50, 50), 16, "LPi (L)", lpi_glow)
        draw_node("lpi_r", (255, 50, 50), 16, "LPi (R)", lpi_glow)
        
        # Draw LPLC2
        lplc2_glow = min(255, int(lplc2_act * 150))
        draw_node("lplc2_l", (0, 255, 200), 18, "LPLC2 (L)", lplc2_glow)
        draw_node("lplc2_r", (0, 255, 200), 18, "LPLC2 (R)", lplc2_glow)
        
        # Draw Haltere
        haltere_glow = min(255, int(haltere_act * 200))
        draw_node("haltere", (255, 180, 0), 14, "Haltere", haltere_glow)
        
        # Draw GF
        # Vm ranges from ~ -70 to -50
        v_rest = -70.0
        v_thresh = -50.0 # Approx
        gf_fill = np.clip((gf_v - v_rest) / (v_thresh - v_rest), 0.0, 1.0)
        gf_core_color = (int(gf_fill * 200), int(gf_fill * 255), int(gf_fill * 255))
        
        pos = self.nodes["gf"]
        pygame.draw.circle(surface, gf_core_color, pos, 24)
        pygame.draw.circle(surface, (0, 255, 255), pos, 24, 3)
        if self.gf_flash_alpha > 0:
            glow = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.circle(glow, (200, 255, 255, self.gf_flash_alpha), (50, 50), 40)
            surface.blit(glow, (pos[0]-50, pos[1]-50))
            
        label_surf = self.font.render("Giant Fiber (GF)", True, COLOR_TEXT)
        surface.blit(label_surf, (pos[0] - label_surf.get_width()//2, pos[1] + 28))
        
        # Draw MN
        pos = self.nodes["mn"]
        pygame.draw.circle(surface, (20, 30, 40), pos, 20)
        pygame.draw.circle(surface, (0, 255, 100) if self.mn_flash_alpha > 0 else (50, 100, 50), pos, 20, 3)
        if self.mn_flash_alpha > 0:
            glow = pygame.Surface((80, 80), pygame.SRCALPHA)
            pygame.draw.circle(glow, (0, 255, 100, self.mn_flash_alpha), (40, 40), 30)
            surface.blit(glow, (pos[0]-40, pos[1]-40))
        
        label_surf = self.font.render("Thoracic MN", True, COLOR_TEXT)
        surface.blit(label_surf, (pos[0] - label_surf.get_width()//2, pos[1] + 24))
