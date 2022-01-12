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
from libs.core.TailCalib import model as base_model # Note that we are trying to inherit the TailCalib (which in turn inherits core_base) instead of just core_base
from libs.utils.utils import *
from libs.utils.logger import Logger
import libs.utils.globals as g
if g.wandb_log:
    import wandb

class model(base_model):
    def batch_forward(self, inputs):
        """Batch Forward

        Args:
            inputs (float Tensor): batch_size x image_size
        """
        # Calculate Features and outputs
        if self.accumulation:
            self.features = self.networks["feat_model"](inputs)
            self.features = F.normalize(self.features, dim=1)
        else:
            self.features = inputs

        self.logits = self.networks["classifier"](self.features)

    def train(self, retrain=False):
        """Main training 

        Args:
            retrain (bool, optional): Incase of retraining different dataloaders are used. Defaults to False.
        """ 
        phase = "train"
        print_str = ["Phase: train"]
        print_write(print_str, self.log_file)

        # Inits
        best_acc = 0.0
        best_epoch = 0
        self.retrain = retrain
        self.end_epoch = self.training_opt["num_epochs"]
        self.accumulation = False
        self.accumulation_step = 1

        # Initialize best model and other variables
        self.best_model_weights = {}
        for key, _ in self.config["networks"].items():
            if self.config["networks"][key]["trainable"]:
                self.best_model_weights[key] = copy.deepcopy(self.networks[key].state_dict())

        # Loop over epochs
        for epoch in range(self.start_epoch, self.end_epoch + 1):
           # global config
            g.epoch_global = epoch 

            # "Accumulate features -> Generate points -> Prepare a new dataloader" cycle.
            self.accumulate(phase="train")
            self.generate_points(tailcalibX=True)
            self.prepare_updated_dataset(include_generated_points = self.config["pg"]["generate"])
            data_load = self.my_dataloader["train"]

            # Switch to train mode
            for key, model in self.networks.items():
                if self.config["networks"][key]["trainable"]:
                    # only train the module with lr > 0
                    if self.config["networks"][key]["optim_params"]["lr"] == 0.0:
                        model.eval()
                    else:
                        model.train()

            # Empty cuda cache
            torch.cuda.empty_cache()

            # Step the schedulers
            if self.model_scheduler_dict:
                for key, scheduler in self.model_scheduler_dict.items():
                    scheduler.step()
            if self.criterion_optimizer_scheduler:
                self.criterion_optimizer_scheduler.step()

            print_write([self.training_opt["log_dir"]], self.log_file)

            # print learning rate
            current_lr = self.show_current_lr()
            current_lr = min(current_lr * 50, 1.0)

            self.step = 0
            total_preds = []
            total_labels = []
            for inputs, labels, _ in data_load:
                # Break when step equal to epoch step
                if self.step == self.epoch_steps:
                    break

                # Force shuffle option
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)

                # Pushing to GPU
                inputs, labels = inputs, labels.cuda()

                with torch.set_grad_enabled(True):
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, phase="train")
                    self.batch_loss(labels)
                    self.batch_backward()

                    # Tracking and printing predictions
                    _, preds = torch.max(self.logits, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))

                    # Output minibatch training results
                    if self.step % self.training_opt['display_step'] == 0:

                        minibatch_loss_classifier = self.loss_classifier.item() if 'ClassifierLoss' in self.criterions else None
                        minibatch_loss_embed = self.loss_embed.item() if 'EmbeddingLoss' in self.criterions else None
                        minibatch_loss_embed_proto = self.loss_embed_proto.item() if 'EmbeddingLoss' in self.criterions else None
                        minibatch_loss_embed_biasreduc = self.loss_embed_biasreduc.item() if 'EmbeddingLoss' in self.criterions else None
                        minibatch_loss_total = self.loss.item()
                        minibatch_acc = mic_acc_cal(preds, labels)


                        print_str = ['Epoch: [%d/%d]'
                                    % (epoch, self.training_opt['num_epochs']),
                                    'Step: [%d/%d]' 
                                    % (self.step, self.epoch_steps),
                                    'Minibatch_loss_embedding: %.3f'
                                    % (minibatch_loss_embed) if minibatch_loss_embed else '',
                                    'Minibatch_loss_classifier: %.3f'
                                    % (minibatch_loss_classifier) if minibatch_loss_classifier else '',
                                    'Minibatch_accuracy_micro: %.3f'
                                    % (minibatch_acc)]
                        print_write(print_str, self.log_file)

                        loss_info = {
                            'epoch': epoch,
                            'Step': self.step,
                            'Total': minibatch_loss_total,
                            'Embedding (Total)': minibatch_loss_embed,
                            'Proto': minibatch_loss_embed_proto,
                            'BiasReduc': minibatch_loss_embed_biasreduc,
                            'Classifier': minibatch_loss_classifier,
                        }

                        self.logger.log_loss(loss_info)

                # wandb logging
                wandb_log({"Training Loss": minibatch_loss_total})

                # batch-level: sampler update
                if hasattr(self.data["train"].sampler, "update_weights"):
                    if hasattr(self.data["train"].sampler, "ptype"):
                        ptype = self.data["train"].sampler.ptype
                    else:
                        ptype = "score"
                    ws = get_priority(ptype, self.logits.detach(), labels)

                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt["sampler"]["type"] == "ClassPrioritySampler":
                        inlist.append(labels.cpu().numpy())
                    self.data["train"].sampler.update_weights(*inlist)

                # Clear things out (optional)
                del inputs, labels, self.logits, self.features, preds 

                # Update steps
                self.step+=1
                g.step_global += 1

            # epoch-level: reset sampler weight
            if hasattr(self.data["train"].sampler, "get_weights"):
                self.logger.log_ws(epoch, self.data["train"].sampler.get_weights())
            if hasattr(self.data["train"].sampler, "reset_weights"):
                self.data["train"].sampler.reset_weights(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval, _ , _ , _ = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            # Reset class weights for sampling if pri_mode is valid
            if hasattr(self.data["train"].sampler, "reset_priority"):
                ws = get_priority(
                    self.data["train"].sampler.ptype,
                    self.total_logits.detach(),
                    self.total_labels,
                )
                self.data["train"].sampler.reset_priority(
                    ws, self.total_labels.cpu().numpy()
                )

            self.logger.log_acc(rsls)

            # # Under validation, the best model need to be updated
            if rsls_eval["val_all"] > best_acc:
                best_epoch = epoch
                best_acc = rsls_eval["val_all"]
                for key, _ in self.config["networks"].items():
                    if self.config["networks"][key]["trainable"]: 
                        self.best_model_weights[key] = copy.deepcopy(self.networks[key].state_dict())

                # wandb log best epoch, train accuracy, based on best validation accuracy
                wandb_log({"Best Val": 100*best_acc, "Best Epoch": best_epoch})    
                wandb_log({"Best train": 100*rsls_train["train_all"], "Best Epoch": best_epoch})    

                wandb_log({'B_val_all': self.eval_acc_mic_top1,
                    'B_val_many': self.many_acc_top1,
                    'B_val_median': self.median_acc_top1,
                    'B_val_low': self.low_acc_top1})
                
                wandb_log({'B_train_all': rsls_train["train_all"],
                    'B_train_many': rsls_train["train_many"],
                    'B_train_median': rsls_train["train_median"],
                    'B_train_low': rsls_train["train_low"]})

            print("===> Saving checkpoint")
            self.save_latest(epoch)

            # Clear things out (optional)
            del rsls_eval
            del rsls_train
            del rsls

        # Resetting the model with the best weights
        self.reset_model(self.best_model_weights)

        # Save the best model
        self.save_model(epoch, best_epoch, self.best_model_weights, best_acc)

        # After training is complete, gets the classwise accuracies of all the splits and saves it based on the based model
        for i in list(self.data.keys()):
            # wandb is switched off temprorily so that the this validation loop is not logged
            g.wandb_log = False
            accs_dict , _ , _ , cls_acc = self.eval(phase=i)
            if g.log_offline:
                torch.save((accs_dict,cls_acc),g.log_dir+f"/metrics/{i}_cls_acc.pt")
            print(accs_dict)
            g.wandb_log = True

        print("Training Complete.")
        print_str = [f"Best validation accuracy is {best_acc} at epoch {best_epoch}"]
        print_write(print_str, self.log_file)

        # Empty cuda cache
        torch.cuda.empty_cache()

    def accumulate(self, phase):
        """Accumulates features of all the datapoints in a particular split

        Args:
            phase ([type]): Which split of dataset should be accumulated?
        """     
        print_str = ['Accumulating features: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)
        self.accumulation = True

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        # Iterate over dataset
        self.feat = {}
        self.labs = {}

        accum_features = []
        accum_labels = []

        for inputs, labels, _ in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()
            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, phase=phase)
                accum_features.append(self.features)
                accum_labels.append(labels)

        accum_features = torch.vstack(accum_features)
        accum_labels = torch.hstack(accum_labels)

        for i in accum_labels.unique().cpu().numpy():
            self.feat[i] = accum_features[accum_labels == i]
            self.labs[i] = torch.full((self.feat[i].size()[0],), i).cuda()

        self.accumulation = False

# This is there so that we can use source_import from the utils to import model
def get_core(*args):
    return model(*args)
