# Softmax Loss

# Imports
import torch.nn as nn


def create_loss():
    """Generic Cross entropy loss
    """    
    print("Loading Softmax Loss.")
    return nn.CrossEntropyLoss()
