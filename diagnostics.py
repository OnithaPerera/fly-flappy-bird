import os


def run_preflight_checks():
    """Fall back to SDL's dummy video driver when there is no display (CI, SSH)."""
    import pygame
    try:
        pygame.display.init()
    except pygame.error:
        print("[Diagnostics] Display init failed. Falling back to SDL_VIDEODRIVER='dummy'")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
