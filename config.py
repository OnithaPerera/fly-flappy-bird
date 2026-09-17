# config.py

# Window dimensions
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
FPS = 60

# Vision parameters
EYE_RES = 32

# Biological LIF parameters
V_REST = -70.0
V_RESET = -75.0
V_THRESH = -50.0
BETA = 0.85
REFRACTORY_PERIOD = 6
SYNAPTIC_GAIN = 0.003 # Tuned for Pygame absolute luminance difference magnitude

# Physics config
GRAVITY = 0.8
FLAP_STRENGTH = -10.0
PIPE_SPEED = 4
PIPE_SPAWN_FRAMES = 80
PIPE_GAP = 160
