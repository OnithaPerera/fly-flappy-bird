# config.py
#
# All lengths are in arena pixels, all times in simulation frames unless a name
# says otherwise. One frame is 1/60 s of simulated time.

# ---------------------------------------------------------------- display ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
FPS = 60

# ------------------------------------------------------------------ arena ---
ARENA_WIDTH = 720
ARENA_HEIGHT = 900
GROUND_H = 180
GROUND_Y = ARENA_HEIGHT - GROUND_H   # top surface of the ground (lethal)
CEILING_Y = 0                        # top of the arena (lethal, and visible to the eye)

FLY_X = 120
FLY_RADIUS = 12                      # collision half-size of the fly

GRAVITY = 0.5
FLAP_STRENGTH = -7.0               # one stroke lifts the fly ~49 px (v^2 / 2g)
FLAP_COOLDOWN = 4                    # frames: a new wing stroke can't start mid-stroke

PIPE_SPEED = 2.8
PIPE_SPACING = 320
PIPE_GAP = 190
PIPE_WIDTH = 108
FIRST_PIPE_X = 420
GAP_MARGIN = 60                      # min distance between a gap edge and ceiling/ground
MAX_GAP_SHIFT = 180                  # max change in gap height from one pipe to the next

MAX_FRAMES = 6000                    # episode cap (100 s of flight, ~52 pipes)

# ----------------------------------------------------------- compound eye ---
# A 1-D vertical panorama: the arena is 2-D, so the eye samples elevation only.
# Like a real fly eye it is nearly panoramic, reaching slightly behind straight up and down.
N_OMMATIDIA = 33
EYE_ELEVATION_DEG = 100.0            # ommatidia span +100 deg (dorsal, past vertical) .. -100 deg (ventral)
ACUTE_ZONE = 0.6                     # 0 = even spacing; higher packs ommatidia towards the horizon
EYE_RANGE = 420.0                    # px; surfaces further than this read as empty sky

# ---------------------------------------------------------------- circuit ---
SUBSTEPS = 4                         # neuron integration steps per frame (dt ~ 4.2 ms)
DT_MS = 1000.0 / FPS / SUBSTEPS

V_REST = -70.0
V_RESET = -75.0
V_THRESH = -50.0                     # threshold of the visual and proprioceptive units
TAU_UNIT_MS = 20.0
REFRACTORY_MS = 2.0
TAU_SYN_MS = 5.0

N_LPLC2 = 8                          # retinotopic looming units, dorsal -> ventral
N_LPI = 8
RF_SIGMA = 0.8                       # receptive-field width, in units of unit spacing
LOOM_MAX = 4.0                       # clip on nearness change, in nearness units per second
PROP_GAIN = 30.0                     # mV drive of a proprioceptor at |vy| = 10 px/frame

# --------------------------------------------------------------- evolution ---
GA_POPULATION_SIZE = 96
GA_ELITE_COUNT = 4
GA_TOURNAMENT = 3
SEEDS_PER_GENERATION = 4
SHAPING_BONUS = 160.0                # px of credit for dying close to the gap (selection only)
MUT_RATE = 0.15
MUT_SCALE = 0.10                     # Gaussian sigma as a fraction of each gene's range
STAGNATION_LIMIT = 6                 # generations without a record before a diversity pulse
VALIDATION_SEED_BASE = 2_000_000     # courses used only to pick the final champion
TEST_SEED_BASE = 1_000_000           # held-out courses used only for reporting

# --------------------------------------------------------------------- UI ---
COLOR_BG = (11, 15, 25)
COLOR_PANEL = (18, 24, 38)
COLOR_GRID = (30, 41, 59)
COLOR_TEXT = (226, 232, 240)
COLOR_ACCENT = (0, 229, 255)
COLOR_PHOSPHOR = (0, 255, 102)
COLOR_LEADER = (255, 179, 0)

AUDIO_SAMPLE_RATE = 44100
AUDIO_CLICK_MS = 12
