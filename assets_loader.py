import os
import urllib.request
import pygame
from config import ARENA_WIDTH, WINDOW_HEIGHT

ASSETS_DIR = "assets"
BASE_URL = "https://raw.githubusercontent.com/samuelcust/flappy-bird-assets/master/sprites/"
ASSET_FILES = {
    "background": "background-day.png",
    "pipe": "pipe-green.png",
    "ground": "base.png",
    "bird_up": "yellowbird-upflap.png",
    "bird_mid": "yellowbird-midflap.png",
    "bird_down": "yellowbird-downflap.png"
}

def load_or_fetch_assets():
    """
    Downloads authentic Flappy Bird assets if missing.
    Returns a dictionary of loaded Pygame surfaces, or procedural fallbacks on failure.
    """
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
            # Scale background to fill arena
            assets["background"] = pygame.transform.scale(assets["background"], (ARENA_WIDTH, WINDOW_HEIGHT))
            
            assets["pipe"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["pipe"])).convert_alpha()
            # Scale pipe appropriately
            pipe_rect = assets["pipe"].get_rect()
            pipe_width = int(ARENA_WIDTH * 0.15) # 15% of screen width
            pipe_height = int(pipe_rect.height * (pipe_width / pipe_rect.width))
            assets["pipe"] = pygame.transform.scale(assets["pipe"], (pipe_width, pipe_height))
            
            assets["ground"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["ground"])).convert()
            # Scale ground
            assets["ground"] = pygame.transform.scale(assets["ground"], (ARENA_WIDTH * 2, int(WINDOW_HEIGHT * 0.2)))
            
            assets["bird_up"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["bird_up"])).convert_alpha()
            assets["bird_mid"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["bird_mid"])).convert_alpha()
            assets["bird_down"] = pygame.image.load(os.path.join(ASSETS_DIR, ASSET_FILES["bird_down"])).convert_alpha()
            
            # Scale birds up a bit (e.g., 1.5x)
            for b_key in ["bird_up", "bird_mid", "bird_down"]:
                w, h = assets[b_key].get_width(), assets[b_key].get_height()
                assets[b_key] = pygame.transform.scale(assets[b_key], (int(w * 1.5), int(h * 1.5)))
                
        except Exception as e:
            print(f"Error loading images: {e}. Enabling fallback textures.")
            fallback_mode = True
            
    if fallback_mode:
        print("[Assets Loader] Using procedural fallback assets.")
        # Background
        assets["background"] = pygame.Surface((ARENA_WIDTH, WINDOW_HEIGHT))
        assets["background"].fill((112, 197, 206)) # Flappy blue sky
        
        # Pipe
        pipe_w = int(ARENA_WIDTH * 0.15)
        pipe_h = WINDOW_HEIGHT
        pipe_surf = pygame.Surface((pipe_w, pipe_h), pygame.SRCALPHA)
        pygame.draw.rect(pipe_surf, (116, 191, 46), (0, 0, pipe_w, pipe_h)) # Body
        pygame.draw.rect(pipe_surf, (84, 155, 33), (0, 0, pipe_w, pipe_h), 2) # Outline
        # Bevel cap
        pygame.draw.rect(pipe_surf, (116, 191, 46), (-2, 0, pipe_w+4, 30))
        pygame.draw.rect(pipe_surf, (84, 155, 33), (-2, 0, pipe_w+4, 30), 2)
        assets["pipe"] = pipe_surf
        
        # Ground
        ground_h = int(WINDOW_HEIGHT * 0.2)
        ground_surf = pygame.Surface((ARENA_WIDTH * 2, ground_h))
        ground_surf.fill((221, 216, 148))
        pygame.draw.rect(ground_surf, (115, 190, 46), (0, 0, ARENA_WIDTH * 2, 10))
        assets["ground"] = ground_surf
        
        # Birds
        for b_key in ["bird_up", "bird_mid", "bird_down"]:
            bird_surf = pygame.Surface((40, 30), pygame.SRCALPHA)
            pygame.draw.ellipse(bird_surf, (244, 215, 60), (0, 0, 40, 30))
            pygame.draw.ellipse(bird_surf, (0, 0, 0), (0, 0, 40, 30), 2)
            pygame.draw.circle(bird_surf, (255, 255, 255), (30, 10), 6)
            pygame.draw.circle(bird_surf, (0, 0, 0), (32, 10), 2)
            # Orange beak
            pygame.draw.rect(bird_surf, (241, 104, 35), (30, 15, 15, 8))
            assets[b_key] = bird_surf
            
    return assets
