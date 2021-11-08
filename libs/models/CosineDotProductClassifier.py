# Imports
import torch.nn as nn
from os import path
import torch
import torch.nn.functional as F

# Custom imports
from libs.utils.utils import *

class CosineDotProduct_Classifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048, scale=10.0, *args):
        super(CosineDotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(scale), requires_grad=True)

    def forward(self, x, *args):
        x = F.softplus(self.scale)*(torch.mm(x, F.normalize(self.fc.weight.T, dim=1))) + self.fc.bias     # (Batch, feature) x (out, feature).T -> (Batch, out)
        wandb_log({"Classifier Scale": self.scale.item()})
        return x


def create_model(feat_dim, num_classes=1000, scale=10.0, pretrain=False, pretrain_dir=None, *args):
    """Initialize the model

    Args:
        feat_dim (int): output dimension of the previous feature extractor
        num_classes (int, optional): Number of classes. Defaults to 1000.

    Returns:
        Class: Model
    """
    print("Loading Cosine Dot Product Classifier.")
    clf = CosineDotProduct_Classifier(num_classes, feat_dim, scale)

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
            raise Exception(f"Pretrain path doesn't exist!!-{pretrain_dir}")
    else:
        print("===> Train classifier from the scratch")

    return clf
