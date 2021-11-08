# Imports
import torch.nn as nn
from os import path
import torch
import torch.nn.functional as F

# Custom imports
from libs.utils.utils import *

class ecbd_converter(nn.Module):
    """A linear layer that converts the student's feature to the same dimension as the concatenated teachers' dimensions.

    Args:
        nn ([type]): [description]
    """    
    def __init__(self, feat_in, feat_out, *args):
        super(ecbd_converter, self).__init__()
        self.fc = nn.Linear(feat_in, feat_out)

    def forward(self, x, *args):
        return self.fc(x)


def create_model(feat_in, feat_out, pretrain=False, pretrain_dir=None, *args):
    """Initialize the model

    Args:
        feat_dim (int): output dimension of the previous feature extractor
        num_classes (int, optional): Number of classes. Defaults to 1000.

    Returns:
        Class: Model
    """
    print("ECBD Converter.")
    clf = ecbd_converter(feat_in, feat_out)

    if pretrain:
        if path.exists(pretrain_dir):
            print("===> Load Pretrain Initialization for CosineDotProductClassfier")
            weights = torch.load(pretrain_dir)["state_dict_best"]["classifier"]

            weights = {
                k: weights["module." + k]
                if "module." + k in weights
                else clf.state_dict()[k]
                for k in clf.state_dict()
            }
            clf.load_state_dict(weights)
        else:        
            raise Exception("Pretrain path doesn't exist!!")
    else:
        print("===> Train classifier from the scratch")

    return clf
