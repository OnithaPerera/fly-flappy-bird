import cv2
import numpy as np
import pygame
from config import EYE_RES, ARENA_WIDTH, WINDOW_HEIGHT

def extract_forward_binary_grid(surface, bird_x):
    """
    Crops a full-height region forward of the bird, converts to grayscale,
    applies binary thresholding, downsamples to 8x8, and returns a 64-element
    float32 array normalized to [0.0, 1.0] and the 8x8 uint8 visualization.
    """
    crop_w = min(ARENA_WIDTH - bird_x, 300)
    if crop_w <= 0:
        return np.zeros(64, dtype=np.float32), np.zeros((8, 8), dtype=np.uint8)
        
    crop_rect = pygame.Rect(bird_x, 0, crop_w, 500)
    
    # Extract subsurface and convert to numpy array
    sub_surf = surface.subsurface(crop_rect)
    frame_rgb = pygame.surfarray.array3d(sub_surf)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # We will use normal binary thresholding so obstacles (darker) become 255 (1.0) and sky (brighter) becomes 0.
    # Wait, sky is bright cyan (~180-220 grayscale) and pipes are green (~120-150 grayscale).
    # THRESH_BINARY_INV means > thresh -> 0, < thresh -> 255.
    # If sky was yellow (255), it means sky was < thresh. So the sky value was < 180.
    # To fix this, we should lower the threshold to 100, or just use cv2.THRESH_OTSU.
    # Let's use THRESH_OTSU with THRESH_BINARY_INV.
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
