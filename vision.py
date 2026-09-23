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

from config import EYE_RES, LPI_INHIBITION_WEIGHT

def compute_temporal_looming(curr_frame, prev_frame_1, prev_frame_2, spatial_weights=None):
    """
    Calculates 3-frame temporal expansion tensor.
    Splits field into Ventral (excitatory) and Dorsal (inhibitory) zones.
    """
    if prev_frame_1 is None or prev_frame_2 is None:
        return 0.0, np.zeros((EYE_RES, EYE_RES), dtype=np.float32)
        
    D1 = cv2.absdiff(curr_frame, prev_frame_1).astype(np.float32)
    D2 = cv2.absdiff(prev_frame_1, prev_frame_2).astype(np.float32)
    
    # Cancel horizontal parallax scrolling by subtracting median row displacement
    D1 = np.maximum(0, D1 - np.median(D1, axis=1, keepdims=True))
    D2 = np.maximum(0, D2 - np.median(D2, axis=1, keepdims=True))
    
    # Optical Looming Acceleration
    looming = D1 + np.maximum(0.0, D1 - D2)
    
    if spatial_weights is None:
        spatial_weights = np.ones((EYE_RES, EYE_RES), dtype=np.float32)
        spatial_weights[0:13, :] = 0.2
        spatial_weights[13:, :] = 2.0
    
    if isinstance(spatial_weights, list):
        spatial_weights = np.array(spatial_weights, dtype=np.float32)
        
    masked_looming = looming * spatial_weights
    
    # Ventral (rows 13 to 31, excitatory)
    # Dorsal (rows 0 to 12, inhibitory)
    ventral = masked_looming[13:, :]
    dorsal = masked_looming[:13, :]
    
    i_excitatory = np.sum(ventral)
    i_inhibitory = np.sum(dorsal)
    
    total_drive = i_excitatory - (i_inhibitory * LPI_INHIBITION_WEIGHT)
    
    return total_drive, masked_looming

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
