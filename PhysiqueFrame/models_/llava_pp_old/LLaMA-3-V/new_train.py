import warnings
import os
from transformers import set_seed

SEED = 123
set_seed(SEED)

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_DIR = '/kaggle/input/nlp-getting-started/'

DIR = '/kaggle/working/'

NUM_WORKERS = os.cpu_count()
NUM_CLASSES = 2

# To speed up the calculation,we set epochs to 1
EPOCHS,R,LORA_ALPHA,LORA_DROPOUT = 1,32,32,0.1
BATCH_SIZE = 64

MODEL_ID = "LLaVA-Meta-Llama-3-8B-Instruct-FT"

