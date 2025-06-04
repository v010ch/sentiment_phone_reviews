'''Утилиты, регулярно применяемые в проектах'''

import os
import random
from typing import Optional

import numpy as np

DEFAULT_RANDOM_SEED = 1984


def seedBasic(seed: Optional[int] = DEFAULT_RANDOM_SEED):
    '''
    Функция установки random_seed основных библиотек для
    воспроизводимости решений
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# tensorflow random seed
try:
    import tensorflow as tf 
    def seedTF(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        '''
        Функция усnановки random_seed для TensorFlow
        при наличии усnановленного TensorFlow
        '''
        tf.random.set_seed(seed)
except:
    def seedTF(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        '''Функция-заглушка при остутствии установленного TensorFlow'''
        pass

# torch random seed
try:
    import torch

    def seedTorch(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        '''
        Функция усnановки random_seed для PyTorch
        при наличии усnановленного Pytorch
        '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except:
    def seedTorch(seed: Optional[int] = DEFAULT_RANDOM_SEED):
        '''Функция-заглушка при остутствии установленного PyTorch'''
        pass


# basic + tensorflow + torch
# original from https://www.kaggle.com/code/rhythmcam/random-seed-everything
def seedEverything(seed: Optional[int] = DEFAULT_RANDOM_SEED):
    '''
    Функция установки глобального random_seed для воспроизводимости решений
    '''
    seedBasic(seed)
    seedTF(seed)
    seedTorch(seed)
