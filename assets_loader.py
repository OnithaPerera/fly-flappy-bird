import os
import urllib.request
import pygame
from config import ARENA_WIDTH, CANVAS_HEIGHT

ASSETS_DIR = "assets"
BASE_URL = "https://raw.githubusercontent.com/samuelcust/flappy-bird-assets/master/sprites/"
ASSET_FILES = {
    "background": "background-day.png",
    "pipe": "pipe-green.png",
    "ground": "base.png"
}

def generate_procedural_fly(flap_state):
    """
    Generates a retro procedural fruit fly sprite.
    flap_state: 'up', 'mid', 'down'
    """
    width, height = 34, 24
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    
    # Colors
    c_outline = (0, 0, 0)
    c_thorax = (40, 40, 40)
    c_abd_dark = (160, 90, 20)
    c_abd_light = (220, 150, 40)
    c_eye = (220, 20, 40)
    c_eye_hl = (255, 100, 100)
    c_wing = (200, 230, 255, 150)
    c_wing_vein = (150, 180, 220, 180)
    
    # Abdomen (striped oval)
    pygame.draw.ellipse(surf, c_outline, (2, 8, 16, 12))
    pygame.draw.ellipse(surf, c_abd_dark, (3, 9, 14, 10))
    for i in range(5, 14, 3):
        pygame.draw.line(surf, c_abd_light, (i, 10), (i, 17), 2)
        
    # Thorax
    pygame.draw.ellipse(surf, c_outline, (14, 7, 12, 10))
    pygame.draw.ellipse(surf, c_thorax, (15, 8, 10, 8))
    
    # Head and Eye
    pygame.draw.ellipse(surf, c_outline, (22, 6, 10, 10))
    pygame.draw.ellipse(surf, c_thorax, (23, 7, 8, 8))
    
    # Compound Eye (ruby red)
    pygame.draw.ellipse(surf, c_outline, (24, 5, 8, 8))
    pygame.draw.ellipse(surf, c_eye, (25, 6, 6, 6))
    pygame.draw.rect(surf, c_eye_hl, (28, 7, 2, 2)) # highlight
    
    # Legs (simple lines)
    pygame.draw.line(surf, c_outline, (16, 16), (14, 20), 1)
    pygame.draw.line(surf, c_outline, (20, 16), (20, 21), 1)
    pygame.draw.line(surf, c_outline, (24, 14), (26, 19), 1)
    
    # Wings based on flap state
    if flap_state == "up":
        wing_rect = (8, 0, 16, 10)
    elif flap_state == "down":
        wing_rect = (8, 14, 16, 10)
    else: # mid
        wing_rect = (6, 5, 18, 6)
        
    pygame.draw.ellipse(surf, c_outline, wing_rect)
    pygame.draw.ellipse(surf, c_wing, (wing_rect[0]+1, wing_rect[1]+1, wing_rect[2]-2, wing_rect[3]-2))
    
    # Vein
    pygame.draw.line(surf, c_wing_vein, 
                     (wing_rect[0] + 4, wing_rect[1] + wing_rect[3]//2), 
                     (wing_rect[0] + wing_rect[2] - 4, wing_rect[1] + wing_rect[3]//2), 1)
                     
    # Scale up (similar to the 1.5x in original code)
    surf = pygame.transform.scale(surf, (int(width * 1.5), int(height * 1.5)))
    return surf

def load_or_fetch_assets():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        
    assets = {}
    fallback_mode = False
    
    for key, filename in ASSET_FILES.items():
        filepath = os.path.join(ASSETS_DIR, filename)
        if not os.path.exists(filepath):
            url = BASE_URL + filename
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to download {filename}: {e}. Enabling fallback textures.")
                fallback_mode = True
                break
                
    if not fallback_mode:
        try:
            assets["background"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["background"])).convert()
            assets["background"] = pygame.transform.scale(assets["background"], (ARENA_WIDTH, CANVAS_HEIGHT))
            
            assets["pipe"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["pipe"])).convert_alpha()
            pipe_rect = assets["pipe"].get_rect()
            pipe_width = int(ARENA_WIDTH * 0.15)
            pipe_height = int(pipe_rect.height * (pipe_width / pipe_rect.width))
            assets["pipe"] = pygame.transform.scale(assets["pipe"], (pipe_width, pipe_height))
            
            cap_h = int(26 * (pipe_height / pipe_rect.height))
            assets["pipe_cap"] = assets["pipe"].subsurface((0, 0, pipe_width, cap_h)).copy()
            assets["pipe_body"] = assets["pipe"].subsurface((0, cap_h, pipe_width, pipe_height - cap_h)).copy()
            
            assets["ground"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["ground"])).convert()
            assets["ground"] = pygame.transform.scale(assets["ground"], (ARENA_WIDTH * 2, int(CANVAS_HEIGHT * 0.2)))
                
        except Exception as e:
            print(f"Error loading images: {e}. Enabling fallback textures.")
            fallback_mode = True
            
    if fallback_mode:
        print("[Assets Loader] Using procedural fallback assets.")
        assets["background"] = pygame.Surface((ARENA_WIDTH, CANVAS_HEIGHT))
        assets["background"].fill((112, 197, 206))
        
        pipe_w = int(ARENA_WIDTH * 0.15)
        pipe_h = CANVAS_HEIGHT
        pipe_surf = pygame.Surface((pipe_w, pipe_h), pygame.SRCALPHA)
        pygame.draw.rect(pipe_surf, (116, 191, 46), (0, 0, pipe_w, pipe_h))
        pygame.draw.rect(pipe_surf, (84, 155, 33), (0, 0, pipe_w, pipe_h), 2)
        pygame.draw.rect(pipe_surf, (116, 191, 46), (-2, 0, pipe_w+4, 30))
        pygame.draw.rect(pipe_surf, (84, 155, 33), (-2, 0, pipe_w+4, 30), 2)
        assets["pipe"] = pipe_surf
        assets["pipe_cap"] = pipe_surf.subsurface((0, 0, pipe_w, 30)).copy()
        assets["pipe_body"] = pipe_surf.subsurface((0, 30, pipe_w, pipe_h - 30)).copy()
        
        ground_h = int(CANVAS_HEIGHT * 0.2)
        ground_surf = pygame.Surface((ARENA_WIDTH * 2, ground_h))
        ground_surf.fill((221, 216, 148))
        pygame.draw.rect(ground_surf, (115, 190, 46), (0, 0, ARENA_WIDTH * 2, 10))
        assets["ground"] = ground_surf

    # Always procedurally generate the fruit fly
    fly_up = generate_procedural_fly("up")
    fly_mid = generate_procedural_fly("mid")
    fly_down = generate_procedural_fly("down")
    
    assets["fly_0"] = fly_up
    assets["fly_1"] = fly_mid
    assets["fly_2"] = fly_down
    
    assets["bird_0"] = fly_up
    assets["bird_1"] = fly_mid
    assets["bird_2"] = fly_down
    
    # Legacy keys
    assets["bird_up"] = fly_up
    assets["bird_mid"] = fly_mid
    assets["bird_down"] = fly_down
            
    return assets
