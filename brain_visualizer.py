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
        self.particles.append(Particle(self.gf_l_start, self.gf_l_end, (200, 255, 255), speed=18.0, size=4, tail=True))
        self.particles.append(Particle(self.gf_r_start, self.gf_r_end, (200, 255, 255), speed=18.0, size=4, tail=True))

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
        
        # Colors based on requested spec
        cyan_base = hex_to_rgb("#00AACC")
        cyan_glow = hex_to_rgb("#00FFFF")
        red_color = hex_to_rgb("#FF3366")
        amber_color = hex_to_rgb("#FF9900")
        magenta_color = hex_to_rgb("#CC00FF")
        
        # Spawn sensory particles
        if random.random() < lpi_act * 0.5:
            self.spawn_particles("lpi_l", "central_complex", red_color, speed=5.0)
            self.spawn_particles("lpi_r", "central_complex", red_color, speed=5.0)
            
        if random.random() < lplc2_act * 0.6:
            self.spawn_particles("lplc2_l", "central_complex", cyan_glow, speed=5.0)
            self.spawn_particles("lplc2_r", "central_complex", cyan_glow, speed=5.0)
            
        if spiked:
            self.gf_flash_alpha = 255
            self.mn_flash_alpha = 255
            self.spawn_descending_wave()
            
        # Decay flashes
        self.gf_flash_alpha = max(0, self.gf_flash_alpha - 15)
        self.mn_flash_alpha = max(0, self.mn_flash_alpha - 15)
        
        # --- DRAW ANATOMICAL SILHOUETTE ---
        # Draw translucent outer neuropil shell (Navy Blue: #081224)
        capsule_color = hex_to_rgb("#081224")
        cap_surf = pygame.Surface((400, 300), pygame.SRCALPHA)
        
        # Draw actual contour of insect brain (broad lateral lobes tapering inward)
        # We can approximate this using a polygon or multiple overlapping ellipses
        center_pt = (200, 150)
        
        # Central brain mass
        pygame.draw.ellipse(cap_surf, (*capsule_color, 89), (120, 50, 160, 200)) # 35% alpha approx 89
        # Left optic lobe
        pygame.draw.ellipse(cap_surf, (*capsule_color, 89), (20, 80, 140, 140))
        # Right optic lobe
        pygame.draw.ellipse(cap_surf, (*capsule_color, 89), (240, 80, 140, 140))
        
        # Outline (Central)
        pygame.draw.ellipse(cap_surf, (*capsule_color, 255), (120, 50, 160, 200), 2)
        # Outline (Lobes)
        pygame.draw.ellipse(cap_surf, (*capsule_color, 255), (20, 80, 140, 140), 2)
        pygame.draw.ellipse(cap_surf, (*capsule_color, 255), (240, 80, 140, 140), 2)
        
        surface.blit(cap_surf, (self.center_x - 200, self.center_y - 150))
        
        # --- DRAW FIBER TRACTS ---
        # Connections from LPi to Center (Dorsal Inhibitory Tracts)
        pygame.draw.line(surface, red_color, self.nodes["lpi_l"], self.nodes["central_complex"], 4)
        pygame.draw.line(surface, red_color, self.nodes["lpi_r"], self.nodes["central_complex"], 4)
        
        # Connections from LPLC2 to Center
        pygame.draw.line(surface, cyan_base, self.nodes["lplc2_l"], self.nodes["central_complex"], 6)
        pygame.draw.line(surface, cyan_base, self.nodes["lplc2_r"], self.nodes["central_complex"], 6)
        
        # Twin Giant Fiber Descending Nerve Cords
        pygame.draw.line(surface, magenta_color, self.gf_l_start, self.gf_l_end, 8)
        pygame.draw.line(surface, magenta_color, self.gf_r_start, self.gf_r_end, 8)
        
        # Update and draw particles
        for p in self.particles[:]:
            p.update()
            p.draw(surface)
            if not p.active:
                self.particles.remove(p)
                
        # --- DRAW DENDRITIC CLUSTERS / LOBES ---
        def draw_lobe(pos, base_color, glow_color, intensity, radius, label, num_branches=40):
            glow_alpha = min(255, int(intensity * 255))
            if glow_alpha > 0:
                glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*glow_color, int(glow_alpha*0.15)), (radius*2, radius*2), radius*2)
                pygame.draw.circle(glow_surf, (*glow_color, int(glow_alpha*0.3)), (radius*2, radius*2), radius)
                surface.blit(glow_surf, (pos[0]-radius*2, pos[1]-radius*2), special_flags=pygame.BLEND_ADD)
                
            # Draw dense network of branching filaments
            filament_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
            for i in range(num_branches):
                angle = (i / num_branches) * math.pi * 2 + (intensity * 1.5) + random.uniform(-0.1, 0.1)
                branch_len = radius * random.uniform(0.5, 1.5) * (1.0 + intensity * 0.3)
                start_x, start_y = radius*2, radius*2
                end_x = start_x + math.cos(angle) * branch_len
                end_y = start_y + math.sin(angle) * branch_len
                
                # Draw main filament
                pygame.draw.line(filament_surf, (*base_color, 180), (int(start_x), int(start_y)), (int(end_x), int(end_y)), 2)
                # Draw sub-branch
                if i % 3 == 0:
                    sub_angle = angle + random.uniform(-0.5, 0.5)
                    sub_len = branch_len * 0.5
                    sub_end_x = end_x + math.cos(sub_angle) * sub_len
                    sub_end_y = end_y + math.sin(sub_angle) * sub_len
                    pygame.draw.line(filament_surf, (*glow_color, int(glow_alpha*0.6)), (int(end_x), int(end_y)), (int(sub_end_x), int(sub_end_y)), 1)
            
            surface.blit(filament_surf, (pos[0]-radius*2, pos[1]-radius*2), special_flags=pygame.BLEND_ADD)
            
            label_surf = self.font.render(label, True, COLOR_TEXT)
            surface.blit(label_surf, (pos[0] - label_surf.get_width()//2, pos[1] + radius + 15))
            
        # Draw LPi (Dorsal Inhibitory)
        draw_lobe(self.nodes["lpi_l"], (100, 0, 0), red_color, lpi_act, 15, "LPi (L)")
        draw_lobe(self.nodes["lpi_r"], (100, 0, 0), red_color, lpi_act, 15, "LPi (R)")
        
        # Draw LPLC2 (Lateral Optic Lobes)
        draw_lobe(self.nodes["lplc2_l"], cyan_base, cyan_glow, lplc2_act, 25, "LPLC2 / Lobula (L)", num_branches=8)
        draw_lobe(self.nodes["lplc2_r"], cyan_base, cyan_glow, lplc2_act, 25, "LPLC2 / Lobula (R)", num_branches=8)
        
        # --- DRAW CENTRAL COMPLEX (Ellipsoid Body) ---
        cc_pos = self.nodes["central_complex"]
        
        # Vm ranges from ~ -70 to -50 for visual scaling
        v_rest = -70.0
        v_thresh = -50.0 
        gf_fill = np.clip((gf_v - v_rest) / (v_thresh - v_rest), 0.0, 1.0)
        
        # Core brightness dynamically scales with Giant Fiber membrane potential (dim amber to blinding gold)
        core_color = (
            min(255, int(amber_color[0] + gf_fill * (255 - amber_color[0]))),
            min(255, int(amber_color[1] + gf_fill * (255 - amber_color[1]))),
            min(255, int(amber_color[2] + gf_fill * (255 - amber_color[2])))
        )
        
        # Outer soft glow
        cc_glow = pygame.Surface((140, 140), pygame.SRCALPHA)
        pygame.draw.circle(cc_glow, (*amber_color, int(gf_fill * 80)), (70, 70), 50)
        surface.blit(cc_glow, (cc_pos[0]-70, cc_pos[1]-70), special_flags=pygame.BLEND_ADD)
        
        # Draw toroidal fiber loops (Mushroom Body / Central Complex)
        loop_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
        num_loops = 36
        for i in range(num_loops):
            angle = (i / num_loops) * math.pi * 2
            r1 = 15
            r2 = 35 + gf_fill * 10
            x1 = 50 + math.cos(angle) * r1
            y1 = 50 + math.sin(angle) * r1
            x2 = 50 + math.cos(angle + 0.3) * r2
            y2 = 50 + math.sin(angle + 0.3) * r2
            pygame.draw.line(loop_surf, (*amber_color, 120), (int(x1), int(y1)), (int(x2), int(y2)), 2)
            pygame.draw.line(loop_surf, (*core_color, 200), (int(x2), int(y2)), (int(50 + math.cos(angle+0.6)*r1), int(50 + math.sin(angle+0.6)*r1)), 1)
        
        surface.blit(loop_surf, (cc_pos[0]-50, cc_pos[1]-50), special_flags=pygame.BLEND_ADD)
        
        # Central Complex Label
        hal_surf = self.font.render("Central Complex", True, COLOR_TEXT)
        surface.blit(hal_surf, (cc_pos[0] - hal_surf.get_width()//2, cc_pos[1] - 50))
        
        # Spike Flash (White/Cyan Blast on Central Complex)
        if self.gf_flash_alpha > 0:
            flash_surf = pygame.Surface((160, 160), pygame.SRCALPHA)
            pygame.draw.circle(flash_surf, (150, 255, 255, self.gf_flash_alpha), (80, 80), 60)
            pygame.draw.circle(flash_surf, (255, 255, 255, self.gf_flash_alpha), (80, 80), 30)
            surface.blit(flash_surf, (cc_pos[0]-80, cc_pos[1]-80), special_flags=pygame.BLEND_ADD)
            
        # --- DRAW THORACIC MOTOR GANGLION ---
        mn_pos = self.nodes["mn"]
        pygame.draw.circle(surface, (20, 30, 40), mn_pos, 25)
        pygame.draw.circle(surface, hex_to_rgb("#00FF00") if self.mn_flash_alpha > 0 else (0, 100, 0), mn_pos, 25, 3)
        
        if self.mn_flash_alpha > 0:
            mn_glow = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.circle(mn_glow, (*hex_to_rgb("#00FF00"), self.mn_flash_alpha), (50, 50), 35)
            surface.blit(mn_glow, (mn_pos[0]-50, mn_pos[1]-50), special_flags=pygame.BLEND_ADD)
            
        mn_label = self.font.render("Thoracic Motor Ganglion", True, COLOR_TEXT)
        surface.blit(mn_label, (mn_pos[0] - mn_label.get_width()//2, mn_pos[1] + 30))
