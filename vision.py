import cv2
import numpy as np
import pygame
from config import EYE_RES
from diagnostics import safe_execute, validate_frame_tensor

@safe_execute(fallback_value=np.zeros((EYE_RES, EYE_RES), dtype=np.uint8), error_context="vision.preprocess_frame")
def preprocess_frame(surface):
    """
    Captures the offscreen buffer using pygame.surfarray, converts to grayscale,
    and downsamples to the compound eye resolution.
    """
    frame_rgb = pygame.surfarray.array3d(surface)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    small_frame = cv2.resize(frame_gray, (EYE_RES, EYE_RES), interpolation=cv2.INTER_AREA)
    
    return small_frame

def compute_looming_stimulus(curr_frame, prev_frame, spatial_weights=None):
    """
    Calculates absolute luminance difference to detect expanding edges (looming).
    Accepts an optional spatial_weights matrix (genome) or defaults to the baseline mask.
    """
    if prev_frame is None:
        return 0.0, np.zeros((EYE_RES, EYE_RES), dtype=np.float32)
        
    diff = cv2.absdiff(curr_frame, prev_frame).astype(np.float32)
    
    if spatial_weights is None:
        # Default baseline mask to prioritize lower-forward field
        spatial_weights = np.ones((EYE_RES, EYE_RES), dtype=np.float32)
        spatial_weights[0:EYE_RES//2, :] = 0.2
        spatial_weights[EYE_RES//2:, :] = 2.0
    
    # Ensure spatial_weights is a numpy array
    if isinstance(spatial_weights, list):
        spatial_weights = np.array(spatial_weights, dtype=np.float32)
        
    masked_diff = diff * spatial_weights
    total_drive = np.sum(masked_diff)
    
    return total_drive, masked_diff

def get_colored_heatmap(matrix_32x32):
    """
    Applies a color map for the HUD thermal display.
    """
    disp_eye = matrix_32x32.copy()
    max_val = np.max(disp_eye)
    if max_val > 0:
        disp_eye = (disp_eye / max_val * 255).astype(np.uint8)
    else:
        disp_eye = disp_eye.astype(np.uint8)
        
    # Apply colormap (e.g., INFERNO or VIRIDIS)
    heatmap = cv2.applyColorMap(disp_eye, cv2.COLORMAP_INFERNO)
    
    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Transpose for Pygame
    return np.transpose(heatmap, (1, 0, 2))
