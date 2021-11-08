"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# Imports
from os import path
from collections import OrderedDict
import torch

# Custom imports
from libs.models.ResNextFeature import *
from libs.utils.utils import *

def create_model(pretrain=False, pretrain_dir=None, *args):
    """Initialize/load the model

    Args:
        pretrain (bool, optional): Use pre-trained model?. Defaults to False.
        pretrain_dir (str, optional): Directory of the pre-trained model. Defaults to None.

    Returns:
        class: Model
    """

    print("Loading ResNext 50 Feature Model.")
    resnext50 = ResNext(Bottleneck, [3, 4, 6, 3], use_fc=False, dropout=None,
                       groups=32, width_per_group=4, last_relu=True)

    if pretrain:
        if path.exists(pretrain_dir):
            print("===> Load Pretrain Initialization for ResNext50")
            model_dict = resnext50.state_dict()
            new_dict = load_model(pretrain_dir=pretrain_dir)
            model_dict.update(new_dict)
            resnext50.load_state_dict(model_dict)
            print("Backbone model has been loaded......")
            
        else: 
            raise Exception(f"Pretrain path doesn't exist!!-{pretrain_dir}")
    else:
        print("===> Train backbone from the scratch")

    return resnext50

def load_model(pretrain_dir):
    """Load a pre-trained model

    Args:
        pretrain_dir (str): path of pretrained model
    """
    print(f"Loading Backbone pretrain model from {pretrain_dir}......")
    pretrain_dict = torch.load(pretrain_dir)["state_dict_best"]["feat_model"]

    new_dict = OrderedDict()

    # Removing FC and Classifier layers
    for k, v in pretrain_dict.items():
        if k.startswith("module"):
            k = k[7:]
        if "fc" not in k and "classifier" not in k:
            new_dict[k] = v
    
    return new_dict
    