# Imports
from torch.optim import optimizer
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Custom imports
from libs.core.core_base import model as base_model
from libs.utils.utils import *
from libs.utils.logger import Logger
import libs.utils.globals as g
if g.wandb_log:
    import wandb


# For repo specific to CBD paper with much more detailed instructions, check https://github.com/rahulvigneswaran/Class-Balanced-Distillation-for-Long-Tailed-Visual-Recognition.pytorch
class model(base_model):
    def batch_forward(self, inputs):
        """Batch Forward

        Args:
            inputs (float Tensor): batch_size x image_size
        """
        
        self.features_temp = self.networks["feat_model"](inputs)     
        self.features_temp = F.normalize(self.features_temp, p=2, dim=1)
        
        if len(self.networks.keys()) > 3 :
            # Convert student feature to match with concatenated teachers' features
            self.features = self.networks["ecbd_converter"](self.features_temp)
            self.features = F.normalize(self.features, p=2, dim=1)
        else:
            self.features = self.features_temp

        # Calculate Features and outputs
        self.features_teacher = []
        for i in self.networks.keys():
            if not(("feat_model" in i) or ("classifier" in i) or ("ecbd_converter" in i)):
                self.temp = self.networks[i](inputs) 
                self.features_teacher.append(F.normalize(self.temp, p=2, dim=1))
        self.features_teacher = torch.hstack(self.features_teacher)
        self.features_teacher = F.normalize(self.features_teacher, p=2, dim=1)
                       
        self.logits = self.networks["classifier"](self.features_temp)

    def batch_loss(self, labels):
        """Calculate training loss
        """
        self.loss = 0

        # Calculating loss
        if "DistillLoss" in self.criterions.keys():
            self.loss_distill = self.criterions["DistillLoss"](self.features, self.features_teacher)
            self.loss_distill *= self.criterion_weights["DistillLoss"]
            self.loss += self.loss_distill
        
        # Calculating loss
        if "ClassifierLoss" in self.criterions.keys():
            self.loss_classifier = self.criterions["ClassifierLoss"](self.logits, labels)
            self.loss_classifier *= self.criterion_weights["ClassifierLoss"]
            self.loss += self.loss_classifier

# This is there so that we can use source_import from the utils to import model
def get_core(*args):
    return model(*args)