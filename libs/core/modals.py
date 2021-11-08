# Imports
from torch.optim import optimizer
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

# Custom imports
from libs.core.core_base import model as base_model
from libs.utils.utils import *
from libs.utils.logger import Logger
import libs.utils.globals as g
if g.wandb_log:
    import wandb

class model(base_model):
    def batch_forward(self, inputs, labels=None, phase="train", retrain= False):
        """Batch Forward

        Args:
            inputs (float Tensor): batch_size x image_size
            labels (int, optional): Labels. Defaults to None.
            phase (str, optional): Train or Test?. Defaults to "train".
            retrain (bool, optional): When retraining, only the classifier is forward propagated. Defaults to False.
        """

        # Calculate Features and outputs
        if not(retrain):
            self.features = self.networks["feat_model"](inputs)
        else:
            self.features = inputs

        self.logits = self.networks["classifier"](self.features, labels)

    def accumulate(self, phase):
        """Accumulates features of all the datapoints in a particular split

        Args:
            phase ([type]): Which split of dataset should be accumulated?
        """        
        print_str = ['Accumulating features: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        # Iterate over dataset
        self.feat = {}
        self.labs = {}

        accum_features = []
        accum_labels = []

        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()
            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, phase=phase)

                accum_features.append(self.features)
                accum_labels.append(labels)
                torch.cuda.empty_cache()

        # Appending and stacking is same as just concatenating. In this you just dont have to declare a torch.empty of particular size.
        accum_features = torch.vstack(accum_features)
        accum_labels = torch.hstack(accum_labels)

        for i in accum_labels.unique().cpu().numpy():
            self.feat[i] = accum_features[accum_labels == i]
            self.labs[i] = torch.full((self.feat[i].size()[0],), i).cuda()

    def generate_points(self):
        """Generate new datapoints
        """        

        if not os.path.isdir(self.config["training_opt"]["log_generate"]):
            os.makedirs(self.config["training_opt"]["log_generate"])
        else:
            raise Exception("Generation Directory already exists!!")

        # Class statistics
        self.base_std = []
        for i in self.feat.keys():  
            self.base_std.append(torch.std(self.feat[i], dim=0))

        self.base_std = torch.vstack(self.base_std)    

        
        if self.config["pg"]["generate"]:
            sample_from_each = self.get_sample_count(self.config["training_opt"]["data_count"],self.feat.keys())

            self.generated_points = {}

            for i in tqdm(self.feat.keys()):
                if np.sum(sample_from_each[i]) == 0 and self.config["pg"]["extra_points"] == 0 :
                    continue

                self.generated_points[i] = []
                for k, x_ij in zip(sample_from_each[i], self.feat[i]):
                    if k == 0:
                        continue
                    # Adding a small gaussian noise to the existing train datapoints
                    gen = x_ij + self.config["pg"]["lambda"]*torch.randn((int(k), 128), device='cuda')*self.base_std[i] #FIXME should be standard deviation across samples or dims or dim_samples?
                    self.generated_points[i].append(gen)

                self.generated_points[i] = torch.vstack(self.generated_points[i])

            torch.save(self.generated_points, self.config["training_opt"]["log_generate"] + "/generated_points.pt")
            print("\nPoint Generation Completed!\n")
            
        else:
            self.generated_points ={}
            print("\nPoint Generation is False!\n")
    
    def retrain(self,):
        """Creates a new dataloader, reinits everything and trains just the classifier part.
        """        
        # Prepare a dataloader for all splits which includes the generated points and is also tukey transformed.
        self.prepare_updated_dataset(include_generated_points = self.config["pg"]["generate"])

        # Change log_dir so it wont change the weights of the parent model
        self.config["training_opt"]["log_dir"] = self.config["training_opt"]["log_retrain"]

        # Create retrain directory
        if not os.path.isdir(self.config["training_opt"]["log_dir"]):
            os.makedirs(self.config["training_opt"]["log_dir"])
        else:
            raise Exception("Retrained Directory already exists!!")

        g.log_dir  = self.config["training_opt"]["log_dir"]
        if g.log_offline:
            if not os.path.isdir(f"{g.log_dir}/metrics"):
                os.makedirs(f"{g.log_dir}/metrics")

        # Reinitialize everything
        print("Using steps for training.")
        self.training_data_num = len(self.my_dataloader["train"].dataset)
        self.epoch_steps = int(
            self.training_data_num / self.training_opt["batch_size"]
        )
        
        # Init logger
        self.logger = Logger(self.training_opt["log_dir"])
        self.log_file = os.path.join(self.training_opt["log_dir"], "log.txt")
        self.logger.log_cfg(self.config)

        # Initialize loss
        self.init_criterions()

        # Initialize model
        self.init_models()

        # Initialize model optimizer and scheduler
        print("Initializing model optimizer.")
        self.init_optimizers(self.model_optim_params_dict)

        self.train(retrain=True)

    def prepare_updated_dataset(self, include_generated_points = True):
        """Prepares a dataloader for all splits which includes the generated points and is also tukey transformed.


        Args:
            include_generated_points (bool, optional): Do you wanna include the newly generated points. Defaults to True.
        """        
        self.my_dataloader = {}
        self.for_distance_analysis = {}
        for phase in ["train", "val", "test"]:

            self.accumulate(phase=phase)

            feat_all = []
            labs_all = []

            if self.config["pg"]["tukey"]: 
                for i in self.labs.keys():
                    self.feat[i] = self.tukey_transform(self.feat[i], lam=self.config["pg"]["tukey_value"])

            for i in self.labs.keys():
                feat_all.append(self.feat[i])
                labs_all.append(self.labs[i])

            feat_all = torch.vstack(feat_all)
            labs_all = torch.hstack(labs_all).cuda()

            if include_generated_points and phase == "train":
                generated_points = torch.load(self.config["training_opt"]["log_generate"] + "/generated_points.pt")
                for i in generated_points.keys():
                    feat_all = torch.cat((feat_all, generated_points[i].cuda()))
                    labs_all = torch.cat((labs_all, torch.full((generated_points[i].size()[0],), int(i)).cuda()))

            # Create dataloader
            my_dataset = TensorDataset(feat_all, labs_all, labs_all)
            self.my_dataloader[phase] = DataLoader(my_dataset, batch_size=self.config["training_opt"]["batch_size"], shuffle=True)

# This is there so that we can use source_import from the utils to import model
def get_core(*args):
    return model(*args)
