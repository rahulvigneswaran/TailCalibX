# Imports
import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import numpy as np
import warnings
import pdb

# Custom imports
from libs.core.core_base import model as base_model
from libs.utils.utils import *
from libs.utils.logger import Logger
import libs.utils.globals as g
if g.wandb_log:
    import wandb

class model(base_model):
    """Basic CrossEntropy los training

    Args:
        base_model (): [description]
    """    
    def batch_forward(self, inputs):
        """Batch-wise forward prop

        Args:
            inputs (float Tensor): batch_size x image_size
            labels ([type], optional): [description]. Defaults to None.
        """        
        # Calculate Features and outputs
        self.features = self.networks["feat_model"](inputs)                    
        self.logits = self.networks["classifier"](self.features)
        
# This is there so that we can use source_import from the utils to import model
def get_core(*args):
    return model(*args)