# Imports
import torch.nn as nn
from os import path
import torch
import torch.nn.functional as F

class DotProduct_Classifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, *args):
        x = self.fc(x)
        return x


def create_model(feat_dim, num_classes=1000, pretrain=False, pretrain_dir=None, *args):
    """Initialize the model

    Args:
        feat_dim (int): output dimension of the previous feature extractor
        num_classes (int, optional): Number of classes. Defaults to 1000.

    Returns:
        Class: Model
    """
    print("Loading Dot Product Classifier.")
    clf = DotProduct_Classifier(num_classes, feat_dim)

    if pretrain:
        if path.exists(pretrain_dir):
            print("===> Load Pretrain Initialization for DotProductClassfier")
            weights = torch.load(pretrain_dir)["state_dict_best"]["classifier"]

            weights = {
                k: weights["module." + k]
                if "module." + k in weights
                else clf.state_dict()[k]
                for k in clf.state_dict()
            }
            clf.load_state_dict(weights)
        else:        
            raise Exception(f"Pretrain path doesn't exist!!--{pretrain_dir}")
    else:
        print("===> Train classifier from the scratch")

    return clf
