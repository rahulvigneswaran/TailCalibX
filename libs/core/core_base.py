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
from libs.utils.utils import *
from libs.utils.logger import Logger
import libs.utils.globals as g
if g.wandb_log:
    import wandb

class model:
    def __init__(self, config, data):
        """Initialize

        Args:
            config (Dict): Dictionary of all the configurations
            data (list): Train, val, test splits
        """

        self.config = config
        self.training_opt = self.config["training_opt"]
        self.data = data
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config["shuffle"] if "shuffle" in config else False
        self.start_epoch = 1

        # For gradient accumulation
        self.accumulation_step = self.training_opt["accumulation_step"]
        
        #-----Offline Logger
        self.logger = Logger(self.training_opt["log_dir"])
        self.log_file = os.path.join(self.training_opt["log_dir"], "log.txt")
        self.logger.log_cfg(self.config)
        
        # If using steps for training, we need to calculate training steps
        # for each epoch based on actual number of training data instead of
        # oversampled data number
        print("Using steps for training.")
        self.training_data_num = len(self.data["train"].dataset)
        self.epoch_steps = int(self.training_data_num / self.training_opt["batch_size"])

        # Initialize loss
        self.init_criterions()
        
        # Initialize model
        self.init_models()

        # Initialize model optimizer and scheduler
        print("Initializing model optimizer.")
        self.init_optimizers(self.model_optim_params_dict)

        
    def init_models(self):
        """Initialize models
        """
        networks_defs = self.config["networks"]
        self.networks = {}
        self.model_optim_params_dict = {}

        print("Using", torch.cuda.device_count(), "GPUs.")

        # Create the models in loop
        for key, val in networks_defs.items():
            def_file = val["def_file"]
            model_args = val["params"]
            
            # Create/load model
            self.networks[key] = source_import(def_file).create_model(**model_args)
            if networks_defs[key]["trainable"]: 
                self.networks[key] = nn.DataParallel(self.networks[key]).cuda()
            
                # Freezing part or entire model
                if "fix" in val and val["fix"]:
                    print(f"Freezing weights of module {key}")
                    for param_name, param in self.networks[key].named_parameters():
                        param.requires_grad = False
                if "fix_set" in val:
                    for fix_layer in val["fix_set"]:
                        for param_name, param in self.networks[key].named_parameters():
                            if fix_layer == param_name:
                                param.requires_grad = False
                                print(f"=====> Freezing: {param_name} | {param.requires_grad}")
    
                # wandb logging
                if g.wandb_log:
                    wandb.watch(self.networks[key], log="all")

                # Optimizer list to add to the optimizer in the "init_optimizer" step
                optim_params = val["optim_params"]
                self.model_optim_params_dict[key] = {
                    "params": self.networks[key].parameters(),
                    "lr": optim_params["lr"],
                    "momentum": optim_params["momentum"],
                    "weight_decay": optim_params["weight_decay"],
                }

    def init_optimizers(self, optim_params_dict):
        """Init optimizer with/without scheduler for it

        Args:
            optim_params_dict (Dict): A dictonary with all the params for the optimizer
        """
        networks_defs = self.config["networks"]
        self.model_optimizer_dict = {}
        self.model_scheduler_dict = {}

        for key, val in networks_defs.items():
            if networks_defs[key]["trainable"]: 
                # optimizer
                if ("optimizer" in self.training_opt and self.training_opt["optimizer"] == "adam"):
                    print("=====> Using Adam optimizer")
                    optimizer = optim.Adam([optim_params_dict[key],])
                else:
                    print("=====> Using SGD optimizer")
                    optimizer = optim.SGD([optim_params_dict[key],])
                self.model_optimizer_dict[key] = optimizer
                
                # scheduler
                if val["scheduler_params"]:
                    scheduler_params = val["scheduler_params"]
                    
                    if scheduler_params["coslr"]:
                        self.model_scheduler_dict[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            self.training_opt["num_epochs"],
                        )
                    elif scheduler_params['warmup']:
                        print("===> Module {} : Using warmup".format(key))
                        self.model_scheduler_dict[key] = WarmupMultiStepLR(optimizer, scheduler_params['lr_step'], 
                                                            gamma=scheduler_params['lr_factor'], warmup_epochs=scheduler_params['warm_epoch'])
                    else:
                        self.model_scheduler_dict[key] = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=scheduler_params["step_size"],
                            gamma=scheduler_params["gamma"],
                        )

    def init_criterions(self):
        """Initialize criterion (loss) and if required optimizer, scheduler for trainable params in it.
        """
        criterion_defs = self.config["criterions"]
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val["def_file"]
            loss_args = list(val["loss_params"].values())

            self.criterions[key] = (source_import(def_file).create_loss(*loss_args).cuda())
            self.criterion_weights[key] = val["weight"]

            if val["optim_params"]:
                print("Initializing criterion optimizer.")
                optim_params = val["optim_params"]
                optim_params = [
                    {
                        "params": self.criterions[key].parameters(),
                        "lr": optim_params["lr"],
                        "momentum": optim_params["momentum"],
                        "weight_decay": optim_params["weight_decay"],
                    }
                ]
                
                # Initialize criterion optimizer
                if ("optimizer" in self.training_opt and self.training_opt["optimizer"] == "adam"):
                    print("=====> Using Adam optimizer")
                    optimizer = optim.Adam(optim_params)
                else:
                    print("=====> Using SGD optimizer")
                    optimizer = optim.SGD(optim_params)
                self.criterion_optimizer = optimizer
                
                # Initialize criterion scheduler
                if "scheduler_params" in val and val["scheduler_params"]:
                    scheduler_params = val["scheduler_params"]
                    if scheduler_params["coslr"]:                        
                        self.criterion_optimizer_scheduler = (
                            torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer,
                                self.training_opt["num_epochs"],
                            )
                        )
                    elif scheduler_params['warmup']:
                        print("===> Module {} : Using warmup".format(key))
                        self.criterion_optimizer_scheduler = WarmupMultiStepLR(optimizer, scheduler_params['lr_step'], 
                                                            gamma=scheduler_params['lr_factor'], warmup_epochs=scheduler_params['warm_epoch'])
                    else:
                        self.criterion_optimizer_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=scheduler_params["step_size"],
                            gamma=scheduler_params["gamma"],
                        )

                else:
                    self.criterion_optimizer_scheduler = None
            else:
                self.criterion_optimizer = None
                self.criterion_optimizer_scheduler = None


    def show_current_lr(self):
        """Shows current learning rate

        Returns:
            float: Current learning rate
        """
        max_lr = 0.0
        for key, val in self.model_optimizer_dict.items():
            lr_set = list(set([para["lr"] for para in val.param_groups]))
            if max(lr_set) > max_lr:
                max_lr = max(lr_set)
            lr_set = ",".join([str(i) for i in lr_set])
            print_str = [f"=====> Current Learning Rate of model {key} : {str(lr_set)}"]
            
            print_write(print_str, self.log_file)
            wandb_log({f"LR - {key}": float(lr_set)})

        if self.criterion_optimizer:
            lr_set_rad = list(set([para["lr"] for para in self.criterion_optimizer.param_groups]))
            lr_set_rad = ",".join([str(i) for i in lr_set_rad])
            
            wandb_log({f"LR - Radius": float(lr_set_rad)})

        return max_lr

    def batch_forward(self, inputs):
        """Batch Forward

        Args:
            inputs (float Tensor): batch_size x image_size
        """

        # Calculate Features and outputs
        self.features = self.networks["feat_model"](inputs)     
        self.features = F.normalize(self.features, p=2, dim=1)
                       
        self.logits = self.networks["classifier"](self.features)

    def batch_backward(self):
        """Backprop
        """
        if self.accumulation_step == 1:
            # Zero out optimizer gradients
            for key, optimizer in self.model_optimizer_dict.items():
                optimizer.zero_grad()
            if self.criterion_optimizer:
                self.criterion_optimizer.zero_grad()
            
        # Back-propagation from loss outputs
        self.loss.backward()
        
        # Gradient accumulation incase the batch doesnt fit inside a single GPU
        if (self.step+1) % self.accumulation_step == 0: 
            # Step optimizers
            for key, optimizer in self.model_optimizer_dict.items():
                optimizer.step()
            if self.criterion_optimizer:
                self.criterion_optimizer.step()
            
            if self.accumulation_step != 1:
                # Zero out optimizer gradients
                for key, optimizer in self.model_optimizer_dict.items():
                    optimizer.zero_grad()
                if self.criterion_optimizer:
                    self.criterion_optimizer.zero_grad()

    def batch_loss(self, labels):
        """Calculate training loss

        Args:
            labels (int): Dim = Batch_size
        """
        self.loss = 0

        # Calculating loss
        if "ClassifierLoss" in self.criterions.keys():
            self.loss_classifier = self.criterions["ClassifierLoss"](self.logits, labels)
            self.loss_classifier *= self.criterion_weights["ClassifierLoss"]
            self.loss += self.loss_classifier       

        self.loss = self.loss / self.accumulation_step
        
    def shuffle_batch(self, x, y):
        """Force shuffle data

        Args:
            x (float Tensor): Datapoints
            y (int): Labels

        Returns:
            floatTensor, int: Return shuffled datapoints and corresponding labels
        """
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def print_test(self, path=None):
        """Loads best model and prints accuracies of all the splits

        Args:
            path ([type], optional): [description]. Defaults to None.
        """        
        if path != None:
            self.reset_model(torch.load(path)["state_dict_best"])
        else:
            self.reset_model(torch.load(f"{self.training_opt['log_dir']}/final_model_checkpoint.pth"))

        for i in list(self.data.keys()):
            accs_dict , _ , _ , cls_acc = self.eval(phase=i)
            print(accs_dict)

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

        # Initialize best model and other variables
        self.best_model_weights = {}
        for key, _ in self.config["networks"].items():
            if self.config["networks"][key]["trainable"]: 
                self.best_model_weights[key] = copy.deepcopy(self.networks[key].state_dict())

        # Loop over epochs
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            # global config
            g.epoch_global = epoch 
            
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

            # Choose a different dataloader based on whether this is training or retraining
            if self.retrain:
                data_enum = self.my_dataloader[phase]
            else:
                data_enum = self.data[phase]

            self.step = 0
            total_preds = []
            total_labels = []
            for inputs, labels, indexes in data_enum:             
                # Break when step equal to epoch step
                if self.step == self.epoch_steps:
                    break

                # Force shuffle option
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                    
                # Pushing to GPU    
                inputs, labels = inputs.cuda(), labels.cuda()

                with torch.set_grad_enabled(True):
                    # If training, forward with loss, and no top 5 accuracy calculation 
                    self.batch_forward(inputs, labels, phase="train", retrain=retrain)
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
                del inputs, labels, self.logits, self.features, preds, indexes
                
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
            
            # Under validation, the best model need to be updated
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

    def eval_with_preds(self, preds, labels):
        """Train accuracy 

        Args:
            preds (int): Predictions
            labels (int): Ground Truth

        Returns:
            dict: dictionary of all training accuracies
        """
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)

        # Calculate normal prediction accuracy
        rsl = {
            "train_all": 0.0,
            "train_many": 0.0,
            "train_median": 0.0,
            "train_low": 0.0,
        }
        
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(
                map(np.concatenate, [normal_preds, normal_labels])
            )
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            (
                n_top1_many,
                n_top1_median,
                n_top1_low,
            ) = shot_acc(normal_preds, normal_labels, self.data["train"])
            rsl["train_all"] += len(normal_preds) / n_total * n_top1
            rsl["train_many"] += len(normal_preds) / n_total * n_top1_many
            rsl["train_median"] += len(normal_preds) / n_total * n_top1_median
            rsl["train_low"] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = list(
                map(
                    np.concatenate,
                    [mixup_preds * 2, mixup_labels1 + mixup_labels2, mixup_ws],
                )
            )
            mixup_ws = np.concatenate([mixup_ws, 1 - mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, n_top1_median, n_top1_low, = weighted_shot_acc(
                mixup_preds, mixup_labels, mixup_ws, self.data["train"]
            )
            rsl["train_all"] += len(mixup_preds) / 2 / n_total * n_top1
            rsl["train_many"] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl["train_median"] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl["train_low"] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = [
            "\n Training acc Top1: %.3f \n" % (rsl["train_all"]),
            "Many_top1: %.3f" % (rsl["train_many"]),
            "Median_top1: %.3f" % (rsl["train_median"]),
            "Low_top1: %.3f" % (rsl["train_low"]),
            "\n",
        ]

        print_write(print_str, self.log_file)
        phase = "train"
        wandb_log({phase + '_all': rsl["train_all"]*100,
               phase + '_many': rsl["train_many"]*100,
               phase + '_median': rsl["train_median"]*100,
               phase + '_low': rsl["train_low"]*100,
               phase + ' Accuracy': rsl["train_all"]*100,})

        return rsl

    def eval(self, phase='val'):
        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_labels = torch.empty(0, dtype=torch.long) 
        self.total_preds = []
        minibatch_loss_total = [] 

        # Choose a different dataloader based on whether this is training or retraining
        if self.retrain:
            data_enum = self.my_dataloader[phase]
        else:
            data_enum = self.data[phase]

        # Iterate over dataset
        stepval = 0
        for inputs, labels, paths in tqdm(data_enum):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, phase=phase)
                self.batch_loss(labels)
                minibatch_loss_total.append(self.loss.item())

                _, preds = F.softmax(self.logits, dim=1).max(dim=1)
                self.total_preds.append(preds.cpu())
                self.total_labels = torch.cat((self.total_labels, labels.cpu()))

                #----Clear things out
                del preds, inputs, labels
                torch.cuda.empty_cache()

            stepval+=1
            
        wandb_log({"Validation Loss": np.mean(minibatch_loss_total)})

        preds = torch.hstack(self.total_preds)
        
        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds,
                                            self.total_labels)
        self.eval_f_measure = F_measure(preds, self.total_labels, theta=self.training_opt['open_threshold'])

        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds,
                                 self.total_labels, 
                                 self.data['train'],
                                 acc_per_cls=True)
                                 
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f' 
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f' 
                     % (self.low_acc_top1),
                     '\n']
        
        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure,
               phase + '_loss': self.loss.item()}

        wandb_log({phase + '_all': self.eval_acc_mic_top1*100,
               phase + '_many': self.many_acc_top1*100,
               phase + '_median': self.median_acc_top1*100,
               phase + '_low': self.low_acc_top1*100,
               phase + ' Accuracy': self.eval_acc_mic_top1*100,
               phase + ' Loss': self.loss.item(),})

                
        print_write(print_str, self.log_file)
        print(f"------------->{self.eval_acc_mic_top1 * 100}")
        
        return rsl, preds, self.total_labels, self.cls_accs
          
    def save_latest(self, epoch):
        """Saves necessary model states for resuming.

        Args:
            epoch (int): Epoch number
        """
        # Model's state_dict
        model_weights = {}
        for key, _ in self.config["networks"].items():
            if self.config["networks"][key]["trainable"]: 
                model_weights[key] = copy.deepcopy(
                    self.networks[key].state_dict()
                )
        
        # Optimizer's state_dict
        optimizer_state_dict = {}
        for key, _ in self.model_optimizer_dict.items():
                optimizer_state_dict[key] = copy.deepcopy(
                    self.model_optimizer_dict[key].state_dict()
                )
        
        # Criterion's Optimizer's state_dict
        criterion_optimizer_state_dict = self.criterion_optimizer.state_dict() if self.criterion_optimizer else None
            
         Scheduler's state dict
        scheduler_state_dict = {}
        if self.model_scheduler_dict:
            for key, _ in self.model_scheduler_dict.items():
                    scheduler_state_dict[key] = copy.deepcopy(
                        self.model_scheduler_dict[key].state_dict()
                    )
        else:
           scheduler_state_dict = None 
        
        # Criterion's Scheduler's state_dict
        criterion_scheduler_state_dict = self.criterion_optimizer_scheduler.state_dict() if self.criterion_optimizer_scheduler else None
         

        model_states = {
            "epoch": epoch,
            "state_dict": model_weights,
            "opt_state_dict": optimizer_state_dict,
            "opt_crit_state_dict": criterion_optimizer_state_dict,
            "sch_state_dict": scheduler_state_dict,
            "sch_crit_state_dict": criterion_scheduler_state_dict,   
            "wandb_id": self.config["wandb_id"],        
        }

        model_dir = os.path.join(
            self.training_opt["log_dir"], "latest_model_checkpoint.pth"
        )
        torch.save(model_states, model_dir)

    def save_model(self, epoch, best_epoch, best_model_weights, best_acc):
        """Saves the best model's weights

        Args:
            epoch (int): Epoch number
            best_epoch (int): Epoch with the best accuracy or val loss
            best_model_weights (float Tensor): Best model's weights
            best_acc (float): Best accuracy
        """

        model_states = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "state_dict_best": best_model_weights,
            "best_acc": best_acc,
        }

        model_dir = os.path.join(
            self.training_opt["log_dir"], "final_model_checkpoint.pth"
        )

        torch.save(model_states, model_dir)
        
    def reset_model(self, model_state):
        """Resets the model with the best weight

        Args:
            model_state (dict): dict with best weight
        """
        for key, model in self.networks.items():
            if self.config["networks"][key]["trainable"]: 
                weights = model_state[key]
                weights = {k: weights[k] for k in weights if k in model.state_dict()}
                model.load_state_dict(weights)
    
    def resume_run(self, saved_dict):
        """Resumes the run based on the states saved by "self.save_latest()"

        Args:
            model_state (dict): dict with best weight
        """
        loaded_dict = torch.load(saved_dict)
        model_state = loaded_dict["state_dict"]
        optimizer_state_dict = loaded_dict["opt_state_dict"]
        criterion_optimizer_state_dict = loaded_dict["opt_crit_state_dict"]
        scheduler_state_dict = loaded_dict["sch_state_dict"]
        criterion_scheduler_state_dict = loaded_dict["sch_crit_state_dict"]

        for key, model in self.networks.items():
            if self.config["networks"][key]["trainable"]: 
                weights = model_state[key]
                weights = {k: weights[k] for k in weights if k in model.state_dict()}
                model.load_state_dict(weights)
        
        # Optimizer's state_dict
        for key, _ in self.model_optimizer_dict.items():
                self.model_optimizer_dict[key].load_state_dict(optimizer_state_dict[key])
        
        # Criterion's Optimizer's state_dict
        if self.criterion_optimizer :
            self.criterion_optimizer.load_state_dict(criterion_optimizer_state_dict)  
            
        # Scheduler's state dict
        if self.model_scheduler_dict:
            for key, _ in self.model_scheduler_dict.items():
                    self.model_scheduler_dict[key].load_state_dict(scheduler_state_dict[key])
        
        # Criterion's Scheduler's state_dict
        if self.criterion_optimizer_scheduler :
            self.criterion_optimizer_scheduler.load_state_dict(criterion_scheduler_state_dict)

        self.start_epoch = loaded_dict["epoch"] + 1

        print(f"\nResuming from Epoch: {self.start_epoch}!\n")

#-----------------------------------------------------

# This is there so that we can use source_import from the utils to import model
def get_core(*args):
    return model(*args)