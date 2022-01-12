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
    def batch_forward(self, inputs, retrain=False):
        """Batch Forward

        Args:
            inputs (float Tensor): batch_size x image_size
            retrain (bool, optional): Incase of retraining, different dataloaders are used. Defaults to False.
        """

        # Calculate Features and outputs
        if not(retrain):
            self.features = self.networks["feat_model"](inputs)
            self.features = F.normalize(self.features, dim=1)
        else:
            self.features = inputs

        self.logits = self.networks["classifier"](self.features)

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

        # Appending and stacking is same as just concatenating. In this you just dont have to declare a torch.empty of particular size.
        accum_features = torch.vstack(accum_features)
        accum_labels = torch.hstack(accum_labels)

        for i in accum_labels.unique().cpu().numpy():
            self.feat[i] = accum_features[accum_labels == i]
            self.labs[i] = torch.full((self.feat[i].size()[0],), i).cuda()

    def generate_points(self, tailcalibX=False):
        """Generate new datapoints
        """        
        
        if not os.path.isdir(self.config["training_opt"]["log_generate"]):
            os.makedirs(self.config["training_opt"]["log_generate"])
        else:
            if not(tailcalibX):
                raise Exception("Generation Directory already exists!!")

        # Class statistics
        self.base_means = []
        self.base_covs = []
        for i in self.feat.keys():  
            self.base_means.append(self.feat[i].mean(dim=0))
            self.base_covs.append(self.get_cov(self.feat[i]).unsqueeze(dim=0))

        self.base_means = torch.vstack(self.base_means)
        self.base_covs = torch.vstack(self.base_covs)       

        # Tukey's transform
        if self.config["pg"]["tukey"]:
            for i in self.feat.keys():
                self.feat[i] = self.tukey_transform(self.feat[i], self.config["pg"]["tukey_value"])

        # Distribution calibration and feature sampling
        if self.config["pg"]["generate"]:
            sample_from_each = self.get_sample_count(self.config["training_opt"]["data_count"],self.feat.keys())

            K = self.config["pg"]["topk"]
            self.generated_points = {}

            for i in tqdm(self.feat.keys()):
                if np.sum(sample_from_each[i]) == 0 and self.config["pg"]["extra_points"] == 0 :
                    continue

                self.generated_points[i] = []
                for k, x_ij in zip(sample_from_each[i], self.feat[i]):
                    if k == 0:
                        continue
                    # Getting the top k nearest classes based on l2 distance
                    distances = torch.cdist(self.base_means, x_ij.unsqueeze(0)).squeeze()
                    topk_idx_nn_analysis = torch.topk(-distances, k=int(self.config["pg"]["nn_analysis_k"]))[1]
                    topk_idx = topk_idx_nn_analysis[:K]
                    
                    # Calibrating mean and covariance
                    calibrated_mean, calibrated_cov = self.calibrate_distribution(self.base_means[topk_idx], self.base_covs[topk_idx], K, x_ij, self.config["pg"]["alpha"])
                    
                    # Trick to avoid cholesky decomposition from failing. Look at https://juanitorduz.github.io/multivariate_normal/
                    EPS = 1e-4
                    calibrated_cov += (torch.eye(calibrated_cov.shape[0])*EPS).cuda() 

                    # Note that providng the scal_tril is faster than providing the covariance matrix directly.
                    new_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=calibrated_mean, scale_tril=torch.linalg.cholesky(calibrated_cov)) 

                    gen = new_dist.sample((int(k),))

                    self.generated_points[i].append(gen)    

                self.generated_points[i] = torch.vstack(self.generated_points[i])
                torch.cuda.empty_cache()

            torch.save(self.generated_points, self.config["training_opt"]["log_generate"] + "/generated_points.pt")
            print("\nPoint Generation Completed!\n")
            
        else:
            self.generated_points ={}
            print("\nPoint Generation is False!\n")

    def get_cov(self, X):
        """Calculate the covariance matrix for X

        Args:
            X (torch.tensor): Features

        Returns:
            [torch.tensor]: Covariance matrix of X
        """        
        n = X.shape[0]
        mu = X.mean(dim=0) 
        X = (X - mu)  
        return 1/(n-1) * (X.transpose(0, 1) @ X)  # X^TX -> feat_size x num_of_samples @ num_of_samples x feat_size -> feat_size x feat_size

    def get_sample_count(self, count, keys):
        """Decides how many samples must be generated based on each existing train datapoints.

        Args:
            count (list): Number of samples in each class
            keys (dict.keys): Class keys

        Returns:
            dict: dict consists that has the info as to how many samples must be generated based on each existing train datapoints.
        """        
        sample_count_dict = {}
        for i in keys:
            current = count[i]
            head = max(count)
            # head is the sample count that we must match after the generation. This can be offset by "self.config["pg"]["extra_points"]". In our experiments this is set to 0 as it worked better.
            num_sample = head - current + self.config["pg"]["extra_points"]
            ratio = num_sample / current
            #  Makes sure each datapoint is being used atleast once
            new_sample_from_each = [np.floor(ratio)] * current

            # Rest of the datapoints used for generation are decided randomly
            while True:
                if sum(new_sample_from_each) == num_sample:
                    break
                idx = np.random.randint(0, current)
                new_sample_from_each[idx] += 1
            
            # Sanity checks
            assert sum(new_sample_from_each) == num_sample
            assert len(new_sample_from_each) == current

            sample_count_dict[i] = new_sample_from_each

        return sample_count_dict


    def calibrate_distribution(self, base_means, base_cov, k, x_ij, alpha=0.0):
        """Calibration of the distribution for generation. Check equation 7 and 8 from our paper - Feature Generation for Long-tail Classification. 

        Args:
            base_means (torch.tensor): List of all the means that are used for calibration.
            base_cov (torch.tensor): List of all the covariance matrices used for calibraton.
            k (int): Number of classes used for calibration.
            x_ij (torch.tensor): Datapoint chosen to be used for generation.
            alpha (float, optional): Decides the spread of the generated samples. Defaults to 0.0.

        Returns:
            torch.tensor : Calibrated mean and covariance matrix
        """        

        calibrated_mean = (base_means.sum(dim=0) + x_ij)/(k+1)
        calibrated_cov = base_cov.sum(dim=0)/k + alpha

        return calibrated_mean, calibrated_cov

    def tukey_transform(self, x, lam=0.2):
        """Transforms any distribution into normal-distribution like.

        Args:
            x (torch.tensor): Features
            lam (float, optional): Adjusts how close the transformed features will be to the origin. Defaults to 0.2.

        Returns:
            torch.tensor: Normal distribution like features.
        """        
        if lam == 0:
            EPS = 1e-6
            x = x + EPS
            return x.log()
        else :
            return x**lam

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
            # Note that here we l2 normalize the features before creating the dataloader
            my_dataset = TensorDataset(F.normalize(feat_all, dim=1), labs_all, labs_all)
            self.my_dataloader[phase] = DataLoader(my_dataset, batch_size=self.config["training_opt"]["batch_size"], shuffle=True)

# This is there so that we can use source_import from the utils to import model
def get_core(*args):
    return model(*args)
            
