import cv2
import numpy as np
import pygame
from config import EYE_RES, ARENA_WIDTH, WINDOW_HEIGHT

def get_sensory_vector(surface, bird_rect, prev_frame, bird_vel):
    """
    Crops a full-height region forward of the bird, downsamples to 32x32 grayscale,
    compensates for vertical egomotion, spatially pools to 4x4, and computes 
    the 16-element fused expansion tensor.
    Returns: (flattened_tensor, current_downsampled_frame, visual_display_matrix_4x4)
    """
    # Define crop region: bird.right to +280, y=0 to y=500
    crop_x = bird_rect.right
    crop_w = min(ARENA_WIDTH - crop_x, 280)
    if crop_w <= 0:
        return np.zeros(16, dtype=np.float32), None, np.zeros((4,4), dtype=np.uint8)
    
    crop_rect = pygame.Rect(crop_x, 0, crop_w, 500)
    
    # Extract subsurface and convert to numpy array
    sub_surf = surface.subsurface(crop_rect)
    frame_rgb = pygame.surfarray.array3d(sub_surf)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    curr_frame = cv2.resize(frame_gray, (EYE_RES, EYE_RES), interpolation=cv2.INTER_AREA)
    
    if prev_frame is None:
        return np.zeros(16, dtype=np.float32), curr_frame, np.zeros((4,4), dtype=np.uint8)
        
    # Vertical egomotion shift
    shift_y = int(round(-bird_vel * 0.7))
    shifted_prev = np.roll(prev_frame, shift_y, axis=0)
    
    if shift_y > 0:
        shifted_prev[:shift_y, :] = 0
    elif shift_y < 0:
        shifted_prev[shift_y:, :] = 0
        
    # Spatial pooling to 4x4
    pooled_curr = cv2.resize(curr_frame, (4, 4), interpolation=cv2.INTER_AREA)
    pooled_prev = cv2.resize(shifted_prev, (4, 4), interpolation=cv2.INTER_AREA)
        
    # Compute static contrast against sky (clip 200 - frame)
    C = np.clip((200.0 - pooled_curr.astype(np.float32)) / 150.0, 0.0, 1.0)
    
    # Compute temporal difference
    D = np.abs(pooled_curr.astype(np.float32) - pooled_prev.astype(np.float32)) / 255.0
    
    Visual_Signal = (0.7 * C) + (0.3 * D)
    
    # Normalize to [0, 1] range
    Visual_Signal = np.clip(Visual_Signal, 0.0, 1.0)
    
    display_matrix = (Visual_Signal * 255).astype(np.uint8)
    
    return Visual_Signal.flatten(), curr_frame, display_matrix

def get_colored_heatmap(matrix_4x4):
    """
    Applies a color map for the HUD thermal display (4x4).
    """
    heatmap = cv2.applyColorMap(matrix_4x4, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.transpose(heatmap, (1, 0, 2))
