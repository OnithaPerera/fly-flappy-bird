import cv2
import numpy as np
import pygame
from config import EYE_RES

def preprocess_frame(surface):
    """
    Captures the offscreen buffer using pygame.surfarray, converts to grayscale,
    and downsamples to the compound eye resolution.
    """
    # array3d gives (width, height, 3)
    frame_rgb = pygame.surfarray.array3d(surface)
    
    # Transpose to standard (height, width, 3) for cv2 processing
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    
    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Downsample directly to EYE_RES x EYE_RES
    # OpenCV resize is very fast, usually < 1ms for this size
    small_frame = cv2.resize(frame_gray, (EYE_RES, EYE_RES), interpolation=cv2.INTER_AREA)
    
    return small_frame

def compute_looming_stimulus(curr_frame, prev_frame):
    """
    Calculates absolute luminance difference to detect expanding edges (looming).
    Applies an asymmetrical mask to counteract the top-pipe paradox.
    """
    if prev_frame is None:
        return 0.0, np.zeros((EYE_RES, EYE_RES), dtype=np.float32)
        
    diff = cv2.absdiff(curr_frame, prev_frame).astype(np.float32)
    
    # Mask to prioritize lower-forward field and avoid top-pipe paradox
    mask = np.ones((EYE_RES, EYE_RES), dtype=np.float32)
    
    # Attenuate top half (sky and upper pipes)
    mask[0:EYE_RES//2, :] = 0.2
    
    # Amplify lower half (ground and lower pipes)
    mask[EYE_RES//2:, :] = 2.0
    
    masked_diff = diff * mask
    
    # Sum it up to represent total LPLC2 drive to the Giant Fiber
    total_drive = np.sum(masked_diff)
    
    return total_drive, masked_diff
