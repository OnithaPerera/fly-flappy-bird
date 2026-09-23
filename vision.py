import cv2
import numpy as np
import pygame
from config import EYE_RES, ARENA_WIDTH, WINDOW_HEIGHT, GROUND_Y

def extract_forward_binary_grid(surface, bird_x):
    """
    Crops a full-height region forward of the bird, converts to grayscale,
    applies binary thresholding, downsamples to 8x8, and returns a 64-element
    float32 array normalized to [0.0, 1.0] and the 8x8 uint8 visualization.
    """
    start_x = int(bird_x)
    
    # Crop horizontally from x = bird_x to x = min(ARENA_WIDTH, bird_x + 320)
    # Crop vertically from y = 0 to y = GROUND_Y
    crop_w = min(320, ARENA_WIDTH - start_x)
    crop_h = GROUND_Y
    
    # Ensure we don't go out of bounds of the surface
    surf_w, surf_h = surface.get_size()
    
    # If the bird is too close to the right edge of the *entire* surface, clamp it (though ARENA_WIDTH should prevent this if surface is large enough)
    if start_x + crop_w > surf_w:
        crop_w = surf_w - start_x
        
    if crop_w <= 0:
        return np.zeros(64, dtype=np.float32), np.zeros((8, 8), dtype=np.uint8)
        
    crop_rect = pygame.Rect(start_x, 0, crop_w, crop_h)
    
    # Extract subsurface and convert to numpy array
    sub_surf = surface.subsurface(crop_rect)
    frame_rgb = pygame.surfarray.array3d(sub_surf)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply binary thresholding so pipes/ground are 1.0 (hazard) and sky is 0.0 (clear).
    # Sky is bright, pipes are dark. We use THRESH_BINARY_INV so dark pixels become 255 (hazard).
    _, binary_frame = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Downsample to 8x8 grid using block averaging
    pooled_curr = cv2.resize(binary_frame, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Flatten into 64-element vector normalized to [0.0, 1.0]
    flat_vector = (pooled_curr.astype(np.float32) / 255.0).flatten()
    
    return flat_vector, pooled_curr

def get_colored_heatmap(matrix_8x8):
    """
    Applies a color map for the HUD thermal display (8x8).
    """
    heatmap = cv2.applyColorMap(matrix_8x8, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.transpose(heatmap, (1, 0, 2))
