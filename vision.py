import cv2
import numpy as np
import pygame
from config import EYE_RES, ARENA_WIDTH, WINDOW_HEIGHT

def get_sensory_vector(surface, bird_rect, prev_frame, bird_vel):
    """
    Crops a 160x160 region forward of the bird, downsamples to 32x32 grayscale,
    compensates for vertical egomotion, spatially pools to 4x4, and computes 
    the 16-element expansion tensor.
    Returns: (flattened_tensor, current_downsampled_frame)
    """
    # Define crop region: 160x160 ahead of bird
    crop_x = bird_rect.right
    crop_y = bird_rect.centery - 80
    
    # Clamp to screen boundaries
    if crop_x < 0: crop_x = 0
    if crop_y < 0: crop_y = 0
    if crop_x + 160 > ARENA_WIDTH: crop_x = ARENA_WIDTH - 160
    if crop_y + 160 > WINDOW_HEIGHT: crop_y = WINDOW_HEIGHT - 160
    
    crop_rect = pygame.Rect(crop_x, crop_y, 160, 160)
    
    # Extract subsurface and convert to numpy array
    sub_surf = surface.subsurface(crop_rect)
    frame_rgb = pygame.surfarray.array3d(sub_surf)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    curr_frame = cv2.resize(frame_gray, (EYE_RES, EYE_RES), interpolation=cv2.INTER_AREA)
    
    if prev_frame is None:
        return np.zeros(16, dtype=np.float32), curr_frame
        
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
        
    diff = cv2.absdiff(pooled_curr, pooled_prev).astype(np.float32)
    
    # Cancel horizontal parallax scrolling
    diff = np.maximum(0, diff - np.median(diff, axis=1, keepdims=True))
    
    # Normalize to [0, 1] range
    diff = diff / 255.0
    
    return diff.flatten(), curr_frame

def get_colored_heatmap(matrix_4x4):
    """
    Applies a color map for the HUD thermal display (4x4).
    """
    disp_eye = matrix_4x4.copy()
    max_val = np.max(disp_eye)
    if max_val > 0:
        disp_eye = (disp_eye / max_val * 255).astype(np.uint8)
    else:
        disp_eye = disp_eye.astype(np.uint8)
        
    heatmap = cv2.applyColorMap(disp_eye, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.transpose(heatmap, (1, 0, 2))
