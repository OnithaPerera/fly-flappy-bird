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

def compute_stabilized_looming(curr_frame, prev_frame, bird_vel):
    """
    Calculates stabilized expansion tensor.
    Compensates for vertical egomotion (self-bobbing).
    Returns a normalized 1024-element 1D vector.
    """
    if prev_frame is None:
        return np.zeros(EYE_RES * EYE_RES, dtype=np.float32)
        
    # Vertical egomotion shift
    shift_y = int(round(-bird_vel * 0.8))
    
    # We can shift prev_frame using np.roll
    shifted_prev = np.roll(prev_frame, shift_y, axis=0)
    
    # Handle wrap-around from np.roll by zeroing out the rolled-in edges
    if shift_y > 0:
        shifted_prev[:shift_y, :] = 0
    elif shift_y < 0:
        shifted_prev[shift_y:, :] = 0
        
    diff = cv2.absdiff(curr_frame, shifted_prev).astype(np.float32)
    
    # Cancel horizontal parallax scrolling by subtracting median row displacement
    diff = np.maximum(0, diff - np.median(diff, axis=1, keepdims=True))
    
    # Normalize to [0, 1] range approximately (max diff is 255)
    diff = diff / 255.0
    
    return diff.flatten()


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
