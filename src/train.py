import src.config as model_config
from src.utils import get_project_root_path
import src.data
import src.metrics

from src.model import (
    T5EncoderModelForTokenClassification,
)

import gc
import copy
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    T5Tokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

import peft
from peft import (
    LoraConfig,
)

USE_CRF = model_config.use_crf
ROOT = get_project_root_path()
# EXPERT = model_config.selected_expert
MODEL_VERSION = model_config.model_version

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))




    SEED = model_config.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("Base Model:\t", model_config.base_model_name)
    print("MPS Availible:\t", torch.backends.mps.is_available())
    print("Path:\t\t", ROOT)
    print(f"Using device:\t {device}")

    for expert, encoding in model_config.select_encoding_type.items():
        adapter_location = f'/models/moe_v{MODEL_VERSION}_linear_expert_{EXPERT}'
        print(expert, encoding)
