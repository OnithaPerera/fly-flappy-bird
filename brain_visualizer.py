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
        
        # Define node positions based on FlyWire anatomical connectome layout
        self.nodes = {
            "lpi_l": (self.center_x - 120, self.center_y - 120),  # Dorsal tracts
            "lpi_r": (self.center_x + 120, self.center_y - 120),
            "lplc2_l": (self.center_x - 150, self.center_y + 20), # Lateral optic lobes
            "lplc2_r": (self.center_x + 150, self.center_y + 20),
            "central_complex": (self.center_x, self.center_y),    # Central core
            "mn": (self.center_x, self.center_y + 200)            # Thoracic Motor Ganglion
        }
        
        # Helper points for curved descending twin tracts
        self.gf_l_start = (self.center_x - 15, self.center_y + 30)
        self.gf_r_start = (self.center_x + 15, self.center_y + 30)
        self.gf_l_end = (self.center_x - 10, self.center_y + 190)
        self.gf_r_end = (self.center_x + 10, self.center_y + 190)
        
        self.particles = []
        self.gf_flash_alpha = 0
        self.mn_flash_alpha = 0
        
        self.font = pygame.font.SysFont("Consolas", 12)
        
    def spawn_particles(self, src_key, dst_key, color, count=1, speed=3.0, size=2, tail=False):
        src = self.nodes[src_key]
        dst = self.nodes[dst_key]
        for _ in range(count):
            offset_x = random.uniform(-20, 20)
            offset_y = random.uniform(-20, 20)
            start_pos = (src[0] + offset_x, src[1] + offset_y)
            self.particles.append(Particle(start_pos, dst, color, speed, size, tail))
            
    def spawn_descending_wave(self):
        # Spawn particles specifically for the descending twin tracts
        self.particles.append(Particle(self.gf_l_start, self.gf_l_end, (255, 255, 255), speed=15.0, size=4, tail=True))
        self.particles.append(Particle(self.gf_r_start, self.gf_r_end, (255, 255, 255), speed=15.0, size=4, tail=True))

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
            self.spawn_particles("lpi_l", "central_complex", (255, 30, 50), speed=5.0)
            self.spawn_particles("lpi_r", "central_complex", (255, 30, 50), speed=5.0)
            
        if random.random() < lplc2_act * 0.6:
            self.spawn_particles("lplc2_l", "central_complex", (0, 255, 255), speed=5.0)
            self.spawn_particles("lplc2_r", "central_complex", (0, 255, 255), speed=5.0)
            
        if spiked:
            self.gf_flash_alpha = 255
            self.mn_flash_alpha = 255
            self.spawn_descending_wave()
            
        # Decay flashes
        self.gf_flash_alpha = max(0, self.gf_flash_alpha - 15)
        self.mn_flash_alpha = max(0, self.mn_flash_alpha - 15)
        
        # --- DRAW ANATOMICAL SILHOUETTE ---
        # Draw translucent outer neuropil shell (Navy Blue Capsule)
        capsule_rect = pygame.Rect(0, 0, 360, 260)
        capsule_rect.center = (self.center_x, self.center_y)
        pygame.draw.ellipse(surface, (10, 15, 30), capsule_rect)
        pygame.draw.ellipse(surface, (20, 30, 60), capsule_rect, 2)
        
        # --- DRAW FIBER TRACTS ---
        # Connections from LPi to Center
        pygame.draw.line(surface, (60, 20, 30), self.nodes["lpi_l"], self.nodes["central_complex"], 4)
        pygame.draw.line(surface, (60, 20, 30), self.nodes["lpi_r"], self.nodes["central_complex"], 4)
        
        # Connections from LPLC2 to Center
        pygame.draw.line(surface, (10, 60, 60), self.nodes["lplc2_l"], self.nodes["central_complex"], 6)
        pygame.draw.line(surface, (10, 60, 60), self.nodes["lplc2_r"], self.nodes["central_complex"], 6)
        
        # Twin Giant Fiber Descending Nerve Cords
        # Vm ranges from ~ -70 to -50
        v_rest = -70.0
        v_thresh = -50.0 
        gf_fill = np.clip((gf_v - v_rest) / (v_thresh - v_rest), 0.0, 1.0)
        cord_color = (int(60 + gf_fill * 140), 20, int(100 + gf_fill * 155)) # Magenta/Purple shift
        
        pygame.draw.line(surface, cord_color, self.gf_l_start, self.gf_l_end, 8)
        pygame.draw.line(surface, cord_color, self.gf_r_start, self.gf_r_end, 8)
        
        # Update and draw particles
        for p in self.particles[:]:
            p.update()
            p.draw(surface)
            if not p.active:
                self.particles.remove(p)
                
        # --- DRAW DENDRITIC CLUSTERS / LOBES ---
        def draw_lobe(pos, color, intensity, radius, label, num_branches=5):
            glow_alpha = min(255, int(intensity * 255))
            if glow_alpha > 0:
                glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color, int(glow_alpha*0.4)), (radius*2, radius*2), radius*2)
                surface.blit(glow_surf, (pos[0]-radius*2, pos[1]-radius*2))
                
            # Draw core cluster
            pygame.draw.circle(surface, (color[0]//2, color[1]//2, color[2]//2), pos, radius)
            
            # Draw branches
            for i in range(num_branches):
                angle = (i / num_branches) * math.pi * 2 + (intensity * 2) # Slowly rotate/animate slightly with intensity
                branch_len = radius * (1.0 + intensity * 0.5)
                end_x = pos[0] + math.cos(angle) * branch_len
                end_y = pos[1] + math.sin(angle) * branch_len
                pygame.draw.line(surface, color, pos, (int(end_x), int(end_y)), 2)
                
            label_surf = self.font.render(label, True, COLOR_TEXT)
            surface.blit(label_surf, (pos[0] - label_surf.get_width()//2, pos[1] + radius + 15))
            
        # Draw LPi (Dorsal Inhibitory)
        draw_lobe(self.nodes["lpi_l"], (255, 30, 50), lpi_act, 15, "LPi (L)")
        draw_lobe(self.nodes["lpi_r"], (255, 30, 50), lpi_act, 15, "LPi (R)")
        
        # Draw LPLC2 (Lateral Optic Lobes)
        draw_lobe(self.nodes["lplc2_l"], (0, 255, 255), lplc2_act, 25, "LPLC2 / Lobula (L)", num_branches=8)
        draw_lobe(self.nodes["lplc2_r"], (0, 255, 255), lplc2_act, 25, "LPLC2 / Lobula (R)", num_branches=8)
        
        # --- DRAW CENTRAL COMPLEX (Ellipsoid Body) ---
        cc_pos = self.nodes["central_complex"]
        cc_glow = min(255, int(haltere_act * 255))
        cc_color = (255, 180, 0) # Amber / Gold
        
        # Outer ring
        pygame.draw.circle(surface, (cc_color[0]//3, cc_color[1]//3, 0), cc_pos, 35, 4)
        pygame.draw.circle(surface, cc_color, cc_pos, 35, max(1, int(haltere_act * 5)))
        
        # Inner core
        pygame.draw.circle(surface, (20, 20, 20), cc_pos, 25)
        
        # Haltere Label
        hal_surf = self.font.render("Central Complex", True, COLOR_TEXT)
        surface.blit(hal_surf, (cc_pos[0] - hal_surf.get_width()//2, cc_pos[1] - 50))
        
        # Spike Flash (White/Cyan Blast on Central Complex)
        if self.gf_flash_alpha > 0:
            flash_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.circle(flash_surf, (200, 255, 255, self.gf_flash_alpha), (50, 50), 40)
            pygame.draw.circle(flash_surf, (255, 255, 255, self.gf_flash_alpha), (50, 50), 20)
            surface.blit(flash_surf, (cc_pos[0]-50, cc_pos[1]-50))
            
        # --- DRAW THORACIC MOTOR GANGLION ---
        mn_pos = self.nodes["mn"]
        pygame.draw.circle(surface, (20, 30, 40), mn_pos, 25)
        pygame.draw.circle(surface, (0, 150, 50), mn_pos, 25, 3)
        
        if self.mn_flash_alpha > 0:
            mn_glow = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.circle(mn_glow, (0, 255, 100, self.mn_flash_alpha), (50, 50), 35)
            surface.blit(mn_glow, (mn_pos[0]-50, mn_pos[1]-50))
            
        mn_label = self.font.render("Thoracic Motor Ganglion", True, COLOR_TEXT)
        surface.blit(mn_label, (mn_pos[0] - mn_label.get_width()//2, mn_pos[1] + 30))
