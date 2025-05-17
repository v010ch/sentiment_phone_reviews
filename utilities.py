# basic random seed
import os
import random
from typing import Optional

import numpy as np

DEFAULT_RANDOM_SEED = 1984


def seedBasic(seed: Optional[int] = DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# tensorflow random seed 
try:
    import tensorflow as tf 
    def seedTF(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        tf.random.set_seed(seed)
except:
    def seedTF(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        pass

# torch random seed
try:
    import torch
    def seedTorch(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except:
    def seedTorch(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        pass

# basic + tensorflow + torch 
# original from https://www.kaggle.com/code/rhythmcam/random-seed-everything
def seedEverything(seed: Optional[int] = DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTF(seed)
    seedTorch(seed)
