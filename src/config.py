# –– environment –––––––––––––
NUM_ADVERSARIES    =  3
NUM_GOOD_AGENTS    =  1
NUM_OBSTACLES      =  2
MAX_CYCLES         = 25    # steps per episode (MPE default)
CONTINUOUS_ACTIONS = True 

# –– training ––––––––––––––––
NUM_EPS     = 60_000       # total training episodes
BATCH_SIZE  = 1024
BUFFER_SIZE = 1_000_000  
GAMMA       = 0.95         # discount factor
TAU         = 0.01         # soft update rate
LR_ACTOR    = 1e-4
LR_CRITIC   = 1e-3

# –– exploration ––––––––––––––
NOISE_STD_START = 0.3      # initial exploration noise
NOISE_STD_END   = 0.05     # final exploration noise
NOISE_DECAY_EPS = 40_000

# –– logging ––––––––––––––––––
LOG_FREQ  = 500            # print/save every N episodes
EVAL_FREQ = 5_000          # record a GIF every N episodes
CKPT_FREQ = 10_000
