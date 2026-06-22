# –– environment –––––––––––––
NUM_ADVERSARIES    = 3
NUM_GOOD_AGENTS    = 1
NUM_OBSTACLES      = 2
MAX_CYCLES         = 25   # steps per episode (MPE default)
CONTINUOUS_ACTIONS = True 

# –– run/output identity –––––––––––––
RUN_MARKER     =  "src_v2"
RESULTS_DIR    = f"results/{RUN_MARKER}"
CHECKPOINT_DIR = f"checkpoints/{RUN_MARKER}"
GIF_DIR        = f"gifs/{RUN_MARKER}"

# –– environment safety / rendering –––––––
STABLE_ENVIRONMENT          = True
ENFORCE_OBSTACLE_COLLISIONS = True
RESET_MIN_GAP               = 0.02
RESET_MAX_ATTEMPTS          = 1_000
FIXED_RENDER_CAMERA         = True
RENDER_CAMERA_RANGE         = 1.25

# –– training ––––––––––––––––
NUM_EPS     = 35_000       # total training episodes
BATCH_SIZE  = 1024
BUFFER_SIZE = 1_000_000  
GAMMA       = 0.95         # discount factor
TAU         = 0.01         # soft update rate
LR_ACTOR    = 1e-4
LR_CRITIC   = 1e-3

# –– exploration ––––––––––––––
NOISE_STD_START = 0.3      # initial exploration noise
NOISE_STD_END   = 0.05     # final exploration noise
NOISE_DECAY_EPS = 25_000

# –– logging ––––––––––––––––––
LOG_FREQ  = 500            # print/save every N episodes
EVAL_FREQ = 5_000          # record a GIF every N episodes
CKPT_FREQ = 1_000
