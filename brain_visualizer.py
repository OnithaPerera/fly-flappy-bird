import pygame
import math
import random
import numpy as np
from config import COLOR_PANEL, COLOR_GRID, COLOR_TEXT, COLOR_ACCENT

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

class BrainVisualizer:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.cx = self.rect.width / 2
        self.cy = self.rect.height / 2 - 50
        
        self.time = 0.0
        
        # Fiber generation
        self.fibers_3d = []
        self._generate_3d_fibers()
        
        # Neural state
        self.gf_flash_alpha = 0.0
        self.mn_flash_alpha = 0.0
        self.sparks = []
        
        # Colors
        self.cyan_base = (0, 68, 102)
        self.cyan_glow = (0, 255, 255)
        self.amber_base = (153, 92, 0)
        self.amber_glow = (255, 204, 0)
        self.crimson_glow = (255, 51, 85)
        self.magenta_glow = (204, 0, 255)
        
        self.font = pygame.font.SysFont("Consolas", 12)

    def _generate_3d_fibers(self):
        def add_fibers(f_type, cx, cy, cz, count, radius_x, radius_y, radius_z, segments, curve):
            for _ in range(count):
                # Start point on surface of ellipsoid
                u = random.uniform(0, 2 * math.pi)
                v = random.uniform(0, math.pi)
                
                x = cx + radius_x * math.sin(v) * math.cos(u)
                y = cy + radius_y * math.sin(v) * math.sin(u)
                z = cz + radius_z * math.cos(v)
                
                pts = [[x, y, z]]
                
                # Direction vector
                dx = (x - cx) * 0.2
                dy = (y - cy) * 0.2
                dz = (z - cz) * 0.2
                
                for _ in range(segments):
                    dx += random.uniform(-curve, curve)
                    dy += random.uniform(-curve, curve)
                    dz += random.uniform(-curve, curve)
                    
                    x += dx
                    y += dy
                    z += dz
                    pts.append([x, y, z])
                    
                self.fibers_3d.append({"type": f_type, "points": np.array(pts, dtype=np.float32)})
                
        # Optic Lobes
        add_fibers("optic_l", -120, 0, 0, 70, 20, 40, 20, 8, 3.0)
        add_fibers("optic_r", 120, 0, 0, 70, 20, 40, 20, 8, 3.0)
        
        # Central Complex (Torus)
        for _ in range(80):
            angle = random.uniform(0, 2*math.pi)
            r = random.uniform(15, 25)
            x = math.cos(angle) * r
            z = math.sin(angle) * r
            y = random.uniform(-5, 5)
            
            pts = [[x, y, z]]
            for _ in range(6):
                x += random.uniform(-3, 3)
                y += random.uniform(-3, 3)
                z += random.uniform(-3, 3)
                pts.append([x, y, z])
            self.fibers_3d.append({"type": "central", "points": np.array(pts, dtype=np.float32)})
            
        # Mushroom Bodies (Dorsal vertical lobes)
        add_fibers("mushroom", 0, -60, -20, 60, 20, 10, 20, 6, 4.0)
        
        # Descending Giant Fibers
        for _ in range(15):
            x_l = -12 + random.uniform(-4, 4)
            x_r = 12 + random.uniform(-4, 4)
            y = 30
            z = random.uniform(-5, 5)
            
            pts_l = [[x_l, y, z]]
            pts_r = [[x_r, y, z]]
            
            for _ in range(10):
                y += random.uniform(12, 18)
                x_l += random.uniform(-1.5, 1.5)
                x_r += random.uniform(-1.5, 1.5)
                z += random.uniform(-1.5, 1.5)
                pts_l.append([x_l, y, z])
                pts_r.append([x_r, y, z])
                
            self.fibers_3d.append({"type": "giant", "points": np.array(pts_l, dtype=np.float32)})
            self.fibers_3d.append({"type": "giant", "points": np.array(pts_r, dtype=np.float32)})

    def update_and_draw(self, surface, inputs, gf_v, spiked):
        self.time += 0.015
        
        looming = inputs[0]
        gap_offset = inputs[1]
        vel = inputs[2]
        ground = inputs[3]
        
        lpi_act = max(0.0, -gap_offset)
        lplc2_act = max(0.0, gap_offset) + looming + ground
        
        v_rest = -70.0
        v_thresh = -50.0 
        gf_fill = np.clip((gf_v - v_rest) / (v_thresh - v_rest), 0.0, 1.0)
        
        if spiked:
            self.gf_flash_alpha = 255.0
            self.mn_flash_alpha = 255.0
            
            # Spawn descending wave sparks down the Giant Fiber
            for _ in range(15):
                self.sparks.append({
                    "x": random.uniform(-15, 15),
                    "y": 30.0,
                    "z": random.uniform(-5, 5),
                    "vy": random.uniform(8.0, 18.0),
                    "life": 1.0
                })
            
        self.gf_flash_alpha = max(0, self.gf_flash_alpha - 15)
        self.mn_flash_alpha = max(0, self.mn_flash_alpha - 10)
        
        # 3D Rotation Matrix (Idle drift)
        angle_y = math.sin(self.time * 0.5) * 0.3
        angle_x = math.cos(self.time * 0.3) * 0.15
        
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        
        def project(pts_3d):
            # Rotate Y
            x = pts_3d[:, 0] * cos_y - pts_3d[:, 2] * sin_y
            z1 = pts_3d[:, 0] * sin_y + pts_3d[:, 2] * cos_y
            # Rotate X
            y = pts_3d[:, 1] * cos_x - z1 * sin_x
            z = pts_3d[:, 1] * sin_x + z1 * cos_x
            
            # Perspective
            f = 400.0
            d = 350.0
            scale = f / (z + d)
            
            px = x * scale + self.cx
            py = y * scale + self.cy
            
            return np.column_stack((px, py))
            
        brain_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        glow_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        
        # Outer Neuropil Boundary Shell (Translucent Silhouette)
        capsule_color = (0, 40, 60, 40)
        pygame.draw.ellipse(brain_surf, capsule_color, (self.cx - 160, self.cy - 90, 320, 180))
        pygame.draw.ellipse(brain_surf, (0, 80, 120, 60), (self.cx - 160, self.cy - 90, 320, 180), 2) # Cyan rim

        # Render Fibers
        for f in self.fibers_3d:
            pts_2d = project(f["points"])
            
            if f["type"] in ("optic_l", "optic_r"):
                glow_int = min(1.0, lplc2_act)
                r = int(self.cyan_base[0] + (self.cyan_glow[0] - self.cyan_base[0]) * glow_int)
                g = int(self.cyan_base[1] + (self.cyan_glow[1] - self.cyan_base[1]) * glow_int)
                b = int(self.cyan_base[2] + (self.cyan_glow[2] - self.cyan_base[2]) * glow_int)
                color = (r, g, b, 140)
            elif f["type"] == "central":
                glow_int = gf_fill
                r = int(self.amber_base[0] + (self.amber_glow[0] - self.amber_base[0]) * glow_int)
                g = int(self.amber_base[1] + (self.amber_glow[1] - self.amber_base[1]) * glow_int)
                b = int(self.amber_base[2] + (self.amber_glow[2] - self.amber_base[2]) * glow_int)
                
                if self.gf_flash_alpha > 0:
                    color = (255, 255, 255, 200) # White-cyan flash
                else:
                    color = (r, g, b, 180)
            elif f["type"] == "mushroom":
                glow_int = min(1.0, lpi_act)
                r = int(self.amber_base[0] + (self.crimson_glow[0] - self.amber_base[0]) * glow_int)
                g = int(self.amber_base[1] + (self.crimson_glow[1] - self.amber_base[1]) * glow_int)
                b = int(self.amber_base[2] + (self.crimson_glow[2] - self.amber_base[2]) * glow_int)
                color = (r, g, b, 150)
            elif f["type"] == "giant":
                color = (*self.magenta_glow, 120)

            pts_list = pts_2d.tolist()
            if len(pts_list) > 1:
                pygame.draw.lines(glow_surf, color, False, pts_list, 2)
                
        # Draw sparks
        for spark in self.sparks[:]:
            spark["y"] += spark["vy"]
            spark["life"] -= 0.05
            if spark["life"] <= 0 or spark["y"] > 200:
                self.sparks.remove(spark)
            else:
                pts_3d = np.array([[spark["x"], spark["y"], spark["z"]]])
                pt_2d = project(pts_3d)[0]
                pygame.draw.circle(glow_surf, (200, 255, 255, int(spark["life"] * 255)), (int(pt_2d[0]), int(pt_2d[1])), 3)

        # Central Complex Flash Bloom
        if self.gf_flash_alpha > 0:
            pygame.draw.circle(glow_surf, (150, 255, 255, int(self.gf_flash_alpha * 0.4)), (self.cx, self.cy), 70)
            pygame.draw.circle(glow_surf, (255, 255, 255, int(self.gf_flash_alpha * 0.8)), (self.cx, self.cy), 30)

        # Thoracic Motor Ganglion Flash
        mn_y = self.cy + 180
        if self.mn_flash_alpha > 0:
            pygame.draw.circle(glow_surf, (0, 255, 0, int(self.mn_flash_alpha * 0.5)), (self.cx, mn_y), 50)
            pygame.draw.circle(glow_surf, (100, 255, 100, int(self.mn_flash_alpha * 0.9)), (self.cx, mn_y), 20)

        surface.blit(brain_surf, (self.rect.x, self.rect.y))
        surface.blit(glow_surf, (self.rect.x, self.rect.y), special_flags=pygame.BLEND_ADD)
        
        cc_label = self.font.render("Central Complex", True, COLOR_TEXT)
        surface.blit(cc_label, (self.rect.x + self.cx - cc_label.get_width()//2, self.rect.y + self.cy - 90))
        
        mn_label = self.font.render("Thoracic Motor Ganglion", True, COLOR_TEXT)
        surface.blit(mn_label, (self.rect.x + self.cx - mn_label.get_width()//2, self.rect.y + mn_y + 30))
