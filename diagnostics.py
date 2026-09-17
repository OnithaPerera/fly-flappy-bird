import os
import sys
import numpy as np

def run_preflight_checks():
    """
    Validates dependencies and configures headless dummy display if needed.
    """
    # Check if headless CI/cloud environment
    # Just check if display exists on unix, but on windows it's different.
    # For now, we rely on the --train headless flags in main, but we can set 
    # dummy driver if needed.
    import pygame
    try:
        pygame.display.init()
    except pygame.error:
        print("[Diagnostics] Display init failed. Falling back to SDL_VIDEODRIVER='dummy'")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
    
    print("[Diagnostics] Preflight checks passed.")

def validate_frame_tensor(frame, expected_shape=(32, 32)):
    """
    Asserts and validates array shapes, types, and NaN/Inf anomalies.
    """
    if frame.shape != expected_shape:
        raise ValueError(f"[Diagnostics] Expected frame shape {expected_shape}, got {frame.shape}")
    
    if np.isnan(frame).any() or np.isinf(frame).any():
        raise ValueError("[Diagnostics] Frame contains NaN or Inf values!")
        
    return True

class safe_execute:
    """
    Decorator and context manager to catch rendering/math exceptions gracefully.
    """
    def __init__(self, fallback_value=None, error_context=""):
        self.fallback_value = fallback_value
        self.error_context = error_context
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"[Diagnostics] SafeExecute caught error in {self.error_context}: {exc_val}")
            # Returning True suppresses the exception
            return True
        return False
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[Diagnostics] SafeExecute caught error in {func.__name__} ({self.error_context}): {e}")
                return self.fallback_value
        return wrapper
