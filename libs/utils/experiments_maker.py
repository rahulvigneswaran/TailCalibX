# imports
import yaml
import pprint
import libs.utils.globals as g
import torch

# Look here for experiments list
experiments = {
#---Table 4
#----------------------------------------------------------
#---Baseline
    0.1 : "CE",                                 # Pre-requisite: None 
    0.2 : "CosineCE",                           # Pre-requisite: None 
#----------------------------------------------------------
#---Decouple
    0.3 : "cRT",                                # Pre-requisite: Experiment "0.1" 
#----------------------------------------------------------
#---Distillation  
    0.4: "CBD",                                 # Pre-requisite: Experiment "0.2" but with seeds 10, 20, 30 
# For repo specific to CBD paper with much more detailed instructions, check https://github.com/rahulvigneswaran/Class-Balanced-Distillation-for-Long-Tailed-Visual-Recognition.pytorch
#----------------------------------------------------------
#---Generation
    0.5: "MODALS",  # learning rate bug         # Pre-requisite: Experiment "0.1" 
#----------------------------------------------------------
#---Ours
    1.2: "CosineCE+TailCalib",                  # Pre-requisite: Experiment "0.2" 
    2.2 : "CosineCE+TailCalibX",                # Pre-requisite: Experiment "0.2" 
    2.4 : "CBD+TailCalibX",                     # Pre-requisite: Experiment "0.4" 
#----------------------------------------------------------
}

# Dataset list
datasets = {
    1: "CIFAR100",
    2: "mini-imagenet",
}

# Others
imbalance_ratios = {
    0: 1,     # No Imbalance
    1: 0.01,  # Imbalance ratio = 100
    2: 0.02,  # Imbalance ratio = 50
    3: 0.1,   # Imbalance ratio = 10
}

imbalance_ratio_names = {
    0: 1,     # No Imbalance
    1: 100,   # Imbalance ratio = 100
    2: 50,    # Imbalance ratio = 50
    3: 10,    # Imbalance ratio = 10
    5: 1000,
}


# Detailed experiments
def experiment_maker(experiment, dataset, imb_ratio, data_root, seed=1, valtype="vft", custom_var1="0", custom_var2="0", custom_var3="0", custom_var4="0", custom_var5="0", custom_var6="0", custom_var7="0", custom_var8="0.9", custom_var9="0"):
    """Creates an experiment and outputs an appropriate yaml file

    Args:
        experiment (float): Experiment number (Check 'libs/utils/experiment_maker.py'.
        dataset (float): Dataset number. Choice: 0 - CIFAR100, 1 - mini-imagenet.
        imb_ratio (float): Select Imbalance factor. Choice: 0: 1, 1: 100, 2: 50, 3: 10.
        data_root (dict): Dict of the root directories of all the datasets
        seed (int, optional): Which seed is being used ? Defaults to 1.
        valtype (str, optional): Choose which dataset split to use. Choice: vt: val_from_test, vtr: val_from_train, vit: val_is_test. Defaults to "vft".
        custom_var1 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.
        custom_var2 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.
        custom_var3 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.
        custom_var4 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.
        custom_var5 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.
        custom_var6 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.
        custom_var7 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.
        custom_var8 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0.9". Always make sure to convert to the desired type in experiment_maker.
        custom_var9 (str, optional): Custom variable to use in experiments - purpose changes according to the experiment. Defaults to "0". Always make sure to convert to the desired type in experiment_maker.

    Returns:
        [dictionary]: Dict of config.
    """
    assert experiment in experiments.keys(), "Wrong Experiment!"
    assert dataset in datasets.keys(), "Wrong Dataset!"
    assert imb_ratio in imbalance_ratios.keys(), "Wrong Imbalance Ratio!"

    # Load Default configuration
    with open("default_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Fixing dataset details and imbalance factor
    if dataset == 1:
        num_of_classes = 100
        config["training_opt"]["cifar_imb_ratio"] = imbalance_ratios[imb_ratio]
    elif dataset == 2:
        num_of_classes = 1000
        imb_ratio = 0
    else:
        print("Wrong dataset!")

    if imb_ratio == 0:
        dataset_name = datasets[dataset]
        exp_name_template = f'{dataset_name}'
    else:
        dataset_name = f"{datasets[dataset]}_LT"
        exp_name_template = f'{dataset_name}_imb_{imbalance_ratio_names[imb_ratio]}'

    # Have a separate root folders and experiments names for all seeds except for seed 1
    if seed == 1:
        init_dir = f"logs/{valtype}"
    else:
        init_dir = f"logs/other_seeds/{valtype}/seed_{seed}"
        exp_name_template = f"seed_{seed}_{exp_name_template}"

    # Common configs
    config["training_opt"]["num_classes"] = num_of_classes
    config["training_opt"]["dataset"] = dataset_name

    # Experiments start here
    if experiment in [0.1]: #CE
        config["core"] = "./libs/core/ce.py"

        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        if dataset in [2]:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = False
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = None

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/DotProductClassifier.py"
        if dataset in [2]:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.2
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = False
        config["networks"]["classifier"]["params"]["pretrain_dir"] = None

        if dataset in [2]:
            #delete
            del(config["networks"]["classifier"]["scheduler_params"])
            config["networks"]["classifier"]["scheduler_params"] = False
            del(config["networks"]["feat_model"]["scheduler_params"])
            config["networks"]["feat_model"]["scheduler_params"] = False

        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150
        
        config["training_opt"]["sampler"] = False

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'

        # name of experiment in wandb
        config["training_opt"]["stage"] = exp_name

    elif experiment in [0.2]: #CosineCE
        config["core"] = "./libs/core/core_base.py"

        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        if dataset in [2]:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = False
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = None

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        if dataset in [2]:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.2        
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["scale"] = 10.0
        config["networks"]["classifier"]["params"]["pretrain"] = False
        config["networks"]["classifier"]["params"]["pretrain_dir"] = None

        if dataset in [2]:
            #delete
            del(config["networks"]["classifier"]["scheduler_params"])
            config["networks"]["classifier"]["scheduler_params"] = False
            del(config["networks"]["feat_model"]["scheduler_params"])
            config["networks"]["feat_model"]["scheduler_params"] = False

        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150
        
        config["training_opt"]["sampler"] = False

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'
        
        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'

        # name of experiment in wandb
        config["training_opt"]["stage"] = exp_name

    elif experiment in [0.3]: #cRT
        config["core"] = "./libs/core/ce.py"

        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        if dataset in [2]:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = True
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.1]}_{exp_name_template}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
        config["networks"]["feat_model"]["fix"] = True

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/DotProductClassifier.py"
        if dataset in [2]:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.2        
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = False
        config["networks"]["classifier"]["params"]["pretrain_dir"] = None

        if dataset in [2]:
            #delete
            del(config["networks"]["classifier"]["scheduler_params"])
            config["networks"]["classifier"]["scheduler_params"] = False
            del(config["networks"]["feat_model"]["scheduler_params"])
            config["networks"]["feat_model"]["scheduler_params"] = False

        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150

        config["training_opt"]["sampler"] = {"def_file": "./libs/samplers/ClassAwareSampler.py", "num_samples_cls": 4, "type": "ClassAwareSampler"}

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'

        # name of experiment in wandb
        config["training_opt"]["stage"] = exp_name

    elif experiment in [0.4]: #CBD
        # For repo specific to CBD paper with much more detailed instructions, check https://github.com/rahulvigneswaran/Class-Balanced-Distillation-for-Long-Tailed-Visual-Recognition.pytorch
        config["core"] = "./libs/core/ecbd.py"

        # Seed selection for teachers
        if seed == 1:
            normal_teacher = [10]
            aug_teacher = []
        elif seed == 2:
            normal_teacher = [20]
            aug_teacher = []
        elif seed == 3:
            normal_teacher = [30]
            aug_teacher = []

        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0 - float(custom_var1)

        # Distill loss (Just doing cosine distance between teacher and student features)
        config["criterions"]["DistillLoss"] = {}
        config["criterions"]["DistillLoss"]["def_file"] = "./libs/loss/CosineDistill.py"
        config["criterions"]["DistillLoss"]["loss_params"] = {}
        config["criterions"]["DistillLoss"]["loss_params"]["beta"] = float(custom_var2)
        config["criterions"]["DistillLoss"]["optim_params"] = False
        config["criterions"]["DistillLoss"]["weight"] = float(custom_var1)

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        if dataset in [2]:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = False
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = None

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        if dataset in [2]:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.2 
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = False
        config["networks"]["classifier"]["params"]["pretrain_dir"] = None

        # other training configs
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
        else:
            config["training_opt"]["backbone"] = "resnet32"

        # standard model teacher in CBD paper
        for i,j in zip(range(len(normal_teacher)), normal_teacher):
            dataset_name_t = f"{datasets[dataset]}_LT"
            exp_name_template_t = f'{dataset_name}_imb_{imbalance_ratio_names[imb_ratio]}'
            seed_t = j
            # Have a separate root folders and experiments names for all seeds except for seed 1
            if seed_t == 1:
                init_dir_t = f"logs/{valtype}"
            else:
                init_dir_t = f"logs/other_seeds/{valtype}/seed_{seed_t}"
                exp_name_template_t = f"seed_{seed_t}_{exp_name_template_t}"

            config["networks"][f"normal_t{i}_model"] = {}
            config["networks"][f"normal_t{i}_model"]["trainable"] = True
            if dataset in [2]:
                config["networks"][f"normal_t{i}_model"]["def_file"] = "./libs/models/ResNext50Feature.py"       
                config["networks"][f"normal_t{i}_model"]["optim_params"] = {}
                config["networks"][f"normal_t{i}_model"]["optim_params"]["lr"] = 0.01
            else:
                config["networks"][f"normal_t{i}_model"]["def_file"] = "./libs/models/ResNet32Feature.py"        
                config["networks"][f"normal_t{i}_model"]["optim_params"] = {}
                config["networks"][f"normal_t{i}_model"]["optim_params"]["lr"] = 0.2
            config["networks"][f"normal_t{i}_model"]["optim_params"]["momentum"] = 0.9
            config["networks"][f"normal_t{i}_model"]["optim_params"]["weight_decay"] = 0.0005
            if dataset in [2]:
                config["networks"][f"normal_t{i}_model"]["scheduler_params"] = False
            else:
                config["networks"][f"normal_t{i}_model"]["scheduler_params"] = {}
                config["networks"][f"normal_t{i}_model"]["scheduler_params"]["coslr"] = True
                config["networks"][f"normal_t{i}_model"]["scheduler_params"]["endlr"] = 0.0
                config["networks"][f"normal_t{i}_model"]["scheduler_params"]["step_size"] = 30
            config["networks"][f"normal_t{i}_model"]["params"] = {}
            config["networks"][f"normal_t{i}_model"]["params"]["pretrain"] = True
            config["networks"][f"normal_t{i}_model"]["params"]["pretrain_dir"] = f'./{init_dir_t}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.2]}_{exp_name_template_t}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
            config["networks"][f"normal_t{i}_model"]["fix"] = True

        # augmentation model teacher in CBD paper
        for i,j in zip(range(len(aug_teacher)), aug_teacher):
            dataset_name_t = f"{datasets[dataset]}_LT"
            exp_name_template_t = f'{dataset_name}_imb_{imbalance_ratio_names[imb_ratio]}'
            seed_t = j
            # Have a separate root folders and experiments names for all seeds except for seed 1
            if seed_t == 1:
                init_dir_t = f"logs/{valtype}"
            else:
                init_dir_t = f"logs/other_seeds/{valtype}/seed_{seed_t}"
                exp_name_template_t = f"seed_{seed_t}_{exp_name_template_t}"

            config["networks"][f"aug_t{i}_model"] = {}
            config["networks"][f"aug_t{i}_model"]["trainable"] = True
            if dataset == 2:
                config["networks"][f"aug_t{i}_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
                config["networks"][f"aug_t{i}_model"]["optim_params"] = {}
                config["networks"][f"aug_t{i}_model"]["optim_params"]["lr"] = 0.01
            else:
                config["networks"][f"aug_t{i}_model"]["def_file"] = "./libs/models/ResNet32Feature.py"    
                config["networks"][f"aug_t{i}_model"]["optim_params"] = {}
                config["networks"][f"aug_t{i}_model"]["optim_params"]["lr"] = 0.2
            config["networks"][f"aug_t{i}_model"]["optim_params"]["momentum"] = 0.9
            config["networks"][f"aug_t{i}_model"]["optim_params"]["weight_decay"] = 0.0005
            if dataset == 2:
                config["networks"][f"aug_t{i}_model"]["scheduler_params"] = False
            else:
                config["networks"][f"aug_t{i}_model"]["scheduler_params"] = {}
                config["networks"][f"aug_t{i}_model"]["scheduler_params"]["coslr"] = True
                config["networks"][f"aug_t{i}_model"]["scheduler_params"]["endlr"] = 0.0
                config["networks"][f"aug_t{i}_model"]["scheduler_params"]["step_size"] = 30
                config["networks"][f"aug_t{i}_model"]["params"] = {}
            config["networks"][f"aug_t{i}_model"]["params"]["pretrain"] = True
            config["networks"][f"aug_t{i}_model"]["params"]["pretrain_dir"] = f'./{init_dir_t}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.3]}_{exp_name_template_t}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
            config["networks"][f"aug_t{i}_model"]["fix"] = True

        # model which converts the student features dim to concatenated feature dims of teachers
        if (len(normal_teacher) + len(aug_teacher)) > 1 :
            config["networks"]["ecbd_converter"] = {}
            config["networks"]["ecbd_converter"]["trainable"] = True
            config["networks"]["ecbd_converter"]["def_file"] = "./libs/models/ecbd_converter.py"
            config["networks"]["ecbd_converter"]["optim_params"] = {}
            if dataset == 2:
                config["networks"]["ecbd_converter"]["optim_params"]["lr"] = 0.01
            else:
                config["networks"]["ecbd_converter"]["optim_params"]["lr"] = 0.2
            config["networks"]["ecbd_converter"]["optim_params"]["momentum"] = 0.9
            config["networks"]["ecbd_converter"]["optim_params"]["weight_decay"] = 0.0005
            if dataset == 2:
                config["networks"]["ecbd_converter"]["scheduler_params"] = False
            else:
                config["networks"]["ecbd_converter"]["scheduler_params"] = {}
                config["networks"]["ecbd_converter"]["scheduler_params"]["coslr"] = True
                config["networks"]["ecbd_converter"]["scheduler_params"]["endlr"] = 0.0
                config["networks"]["ecbd_converter"]["scheduler_params"]["step_size"] = 30
            config["networks"]["ecbd_converter"]["params"] = {}
            config["networks"]["ecbd_converter"]["params"]["feat_in"] = config["networks"]["classifier"]["params"]["feat_dim"]
            config["networks"]["ecbd_converter"]["params"]["feat_out"] = config["networks"]["classifier"]["params"]["feat_dim"]*(len(normal_teacher) + len(aug_teacher))


        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150

        config["training_opt"]["sampler"] = {"def_file": "./libs/samplers/ClassAwareSampler.py", "num_samples_cls": 4, "type": "ClassAwareSampler"}
        
        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'

        # name of experiment in wandb
        config["training_opt"]["stage"] = exp_name

    elif experiment in [0.5]: #MODALS
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
        else:
            config["training_opt"]["backbone"] = "resnet32"
        config["core"] = "./libs/core/modals.py"

        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        if dataset in [2]:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
            config["networks"]["feat_model"]["optim_params"]["lr"] = 0.2
        config["networks"]["feat_model"]["optim_params"]["momentum"] = 0.9
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = True
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.1]}_{exp_name_template}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
        config["networks"]["feat_model"]["fix"] = True

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/DotProductClassifier.py"
        if dataset in [2]:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.01
        else:
            config["networks"]["classifier"]["optim_params"]["lr"] = 0.2     
        config["networks"]["classifier"]["optim_params"]["momentum"] = 0.9
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = True
        config["networks"]["classifier"]["params"]["pretrain_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.1]}_{exp_name_template}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
 
        # No LR scheduler used
        del(config["networks"]["classifier"]["scheduler_params"])
        config["networks"]["classifier"]["scheduler_params"] = False
        del(config["networks"]["feat_model"]["scheduler_params"])
        config["networks"]["feat_model"]["scheduler_params"] = False

        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        # backbone has already been init at the start
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150

        config["training_opt"]["sampler"] = False

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'

        # name of experiment in wandb
        config["training_opt"]["stage"] = exp_name

        # point generation sequence related config
        config["pg"]["generate"] = True 
        config["pg"]["lambda"] = float(custom_var1)
        config["pg"]["extra_points"] = 0

        common = f"MODALS_GaussNoise_lambda_{float(custom_var1)}_LR_{float(custom_var2)}_momen_{float(custom_var3)}_NoScheduler_Extra_{config['pg']['extra_points']}"
        # generation log directory
        config["training_opt"]["log_generate"] = config["training_opt"]["log_dir"] + f"/generate/" + common

        # retraining log directory
        config["training_opt"]["log_retrain"] = config["training_opt"]["log_dir"] + f"/retrain/" + common + f"_retrain_{float(custom_var6)},{float(custom_var7)}_LR_{float(custom_var8)}_BS_{float(custom_var9)}"
    
    elif experiment in [1.2]: #CosineCE+TailCalib
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
        else:
            config["training_opt"]["backbone"] = "resnet32"
        config["core"] = "./libs/core/TailCalib.py"
            
        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        if dataset in [2]:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
        else:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
        config["networks"]["feat_model"]["optim_params"]["lr"] = float(custom_var2)
        config["networks"]["feat_model"]["optim_params"]["momentum"] = float(custom_var3)
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = True
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.2]}_{exp_name_template}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
        config["networks"]["feat_model"]["fix"] = True

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        config["networks"]["classifier"]["optim_params"]["lr"] = float(custom_var2) 
        config["networks"]["classifier"]["optim_params"]["momentum"] = float(custom_var3)
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["scale"] = 10.0
        config["networks"]["classifier"]["params"]["pretrain"] = True
        config["networks"]["classifier"]["params"]["pretrain_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.2]}_{exp_name_template}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
        config["networks"]["classifier"]["fix"] = False

        # No LR scheduler used as it works better without it
        del(config["networks"]["classifier"]["scheduler_params"])
        config["networks"]["classifier"]["scheduler_params"] = False
        del(config["networks"]["feat_model"]["scheduler_params"])
        config["networks"]["feat_model"]["scheduler_params"] = False

        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        # backbone has already been init at the start
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150

        config["training_opt"]["sampler"] = False

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'

        # name of experiment in wandb
        config["training_opt"]["stage"] = exp_name

        # point generation sequence related config
        config["pg"]["generate"] = True #Make this True to generate points
        config["pg"]["tukey"] = True
        config["pg"]["tukey_value"] = float(custom_var1)
        config["pg"]["alpha"] = float(custom_var4)
        config["pg"]["extra_points"] = 0
        config["pg"]["topk"] = int(custom_var5)
        config["pg"]["distance_analysis"] = False
        config["pg"]["nn_analysis"] = False
        config["pg"]["nn_analysis_k"] = 10
        config["pg"]["tsne"] = False

        common = f"FreeLunch_Tukey_{float(custom_var1)}_LR_{float(custom_var2)}_momen_{float(custom_var3)}_alpha_{float(custom_var4)}_topK_{float(custom_var5)}_NoScheduler_Extra_{config['pg']['extra_points']}"
        # generation log directory
        config["training_opt"]["log_generate"] = config["training_opt"]["log_dir"] + f"/generate/" + common

        # retraining sequence related config
        config["training_opt"]["log_retrain"] = config["training_opt"]["log_dir"] + f"/retrain/" + common + f"_retrain_{float(custom_var6)},{float(custom_var7)}_LR_{float(custom_var8)}_BS_{float(custom_var9)}"

    elif experiment in [2.2]: #CosineCE+TailCalibX
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
        else:
            config["training_opt"]["backbone"] = "resnet32"
        config["core"] = "./libs/core/TailCalibX.py"

        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        if dataset in [2]:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNext50Feature.py"
        else:
            config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
        config["networks"]["feat_model"]["optim_params"]["lr"] = float(custom_var2)
        config["networks"]["feat_model"]["optim_params"]["momentum"] = float(custom_var3)
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = True
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.2]}_{exp_name_template}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
        config["networks"]["feat_model"]["fix"] = True

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        config["networks"]["classifier"]["optim_params"]["lr"] = float(custom_var2)
        config["networks"]["classifier"]["optim_params"]["momentum"] = float(custom_var3)
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["pretrain"] = True
        config["networks"]["classifier"]["params"]["pretrain_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{experiments[0.2]}_{exp_name_template}_{config["training_opt"]["backbone"]}/final_model_checkpoint.pth'
        config["networks"]["classifier"]["fix"] = False

        # No LR scheduler used as it works better without it
        del(config["networks"]["classifier"]["scheduler_params"])
        config["networks"]["classifier"]["scheduler_params"] = False
        del(config["networks"]["feat_model"]["scheduler_params"])
        config["networks"]["feat_model"]["scheduler_params"] = False

        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        # backbone has already been init at the start
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150

        config["training_opt"]["sampler"] = False

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'
        
        # run name for wandb
        config["training_opt"]["stage"] = exp_name

        # point generation sequence related config
        config["pg"]["generate"] = True #Make this True to generate points
        config["pg"]["tukey"] = True
        config["pg"]["tukey_value"] = float(custom_var1)
        config["pg"]["alpha"] = float(custom_var4)
        config["pg"]["extra_points"] = 0
        config["pg"]["topk"] = int(custom_var5)
        config["pg"]["distance_analysis"] = False
        config["pg"]["nn_analysis"] = False
        config["pg"]["nn_analysis_k"] = 10
        config["pg"]["tsne"] = False
        config["pg"]["start_after"] = -1

        common = f"FreeLunch_Tukey_{float(custom_var1)}_LR_{float(custom_var2)}_momen_{float(custom_var3)}_alpha_{float(custom_var4)}_topK_{float(custom_var5)}_NoScheduler_Extra_{config['pg']['extra_points']}"
        # generation log directory
        config["training_opt"]["log_generate"] = config["training_opt"]["log_dir"] + f"/generate/" + common

        # retraining sequence related config
        config["training_opt"]["log_retrain"] = config["training_opt"]["log_dir"] + f"/retrain/" + common + f"_retrain_{float(custom_var6)},{float(custom_var7)}_LR_{float(custom_var8)}_BS_{float(custom_var9)}"

    elif experiment == 2.4: #CBD+TailCalibX
        config["training_opt"]["backbone"] = "resnet32"
        config["core"] = "./libs/core/TailCalibX.py"

        # loss
        config["criterions"]["ClassifierLoss"]["def_file"] = "./libs/loss/SoftmaxLoss.py"
        config["criterions"]["ClassifierLoss"]["loss_params"] = {}
        config["criterions"]["ClassifierLoss"]["optim_params"] = False
        config["criterions"]["ClassifierLoss"]["weight"] = 1.0

        # network
        # part 1
        config["networks"]["feat_model"]["trainable"] = True
        config["networks"]["feat_model"]["def_file"] = "./libs/models/ResNet32Feature.py"
        config["networks"]["feat_model"]["optim_params"]["lr"] = float(custom_var2)
        config["networks"]["feat_model"]["optim_params"]["momentum"] = float(custom_var3)
        config["networks"]["feat_model"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        config["networks"]["feat_model"]["params"]["pretrain"] = True

        # CBD pretrain directory
        if dataset in [2]:
            exp_n = f'{experiments[0.4]}_{exp_name_template}_resnext50'
            a = 0.6
            b = 200
        else:
            exp_n = f'{experiments[0.4]}_{exp_name_template}_resnet32'
            if imb_ratio == 1:
                a = 0.8 
                b = 100 
            elif imb_ratio == 2:
                a = 0.8 
                b = 200 
            elif imb_ratio == 3:
                a = 0.8 
                b = 100
            else:
                raise Exception("Wrong Imb ratio!!")
        pre = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_n}/alpha_{float(a)},beta_{float(b)}_normal_k_1_aug_k_0'
        config["networks"]["feat_model"]["params"]["pretrain_dir"] = f'{pre}/final_model_checkpoint.pth'
        config["networks"]["feat_model"]["fix"] = True

        # part 2
        config["networks"]["classifier"]["trainable"] = True
        config["networks"]["classifier"]["def_file"] = "./libs/models/CosineDotProductClassifier.py"
        config["networks"]["classifier"]["optim_params"]["lr"] = float(custom_var2)
        config["networks"]["classifier"]["optim_params"]["momentum"] = float(custom_var3)
        config["networks"]["classifier"]["optim_params"]["weight_decay"] = 0.0005
        config["networks"]["feat_model"]["scheduler_params"]["coslr"] = True
        if dataset in [2]:
            config["networks"]["classifier"]["params"]["feat_dim"] = 2048
        else:
            config["networks"]["classifier"]["params"]["feat_dim"] = 128     
        config["networks"]["classifier"]["params"]["num_classes"] = num_of_classes
        config["networks"]["classifier"]["params"]["scale"] = 10.0
        config["networks"]["classifier"]["params"]["pretrain"] = True
        config["networks"]["classifier"]["params"]["pretrain_dir"] = f'{pre}/final_model_checkpoint.pth'

        # No LR scheduler used as it works better without it
        del(config["networks"]["classifier"]["scheduler_params"])
        config["networks"]["classifier"]["scheduler_params"] = False
        del(config["networks"]["feat_model"]["scheduler_params"])
        config["networks"]["feat_model"]["scheduler_params"] = False

        # force shuffle dataset
        config["shuffle"] = False

        # tags for wandb
        config["wandb_tags"] = [experiments[experiment]]

        # other training configs
        # backbone has already been init at the start
        if dataset in [2]:
            config["training_opt"]["backbone"] = "resnext50"
            config["training_opt"]["batch_size"] = 128 
            config["training_opt"]["accumulation_step"] = int(int(512/int(torch.cuda.device_count()))/config["training_opt"]["batch_size"])
            config["training_opt"]["feature_dim"] = 2048
            config["training_opt"]["num_epochs"] = 111
            config["training_opt"]["num_workers"] = 12
            config["training_opt"]["jitter"] = True
        else:
            config["training_opt"]["backbone"] = "resnet32"
            config["training_opt"]["batch_size"] = 128
            config["training_opt"]["accumulation_step"] = 1 
            config["training_opt"]["feature_dim"] = 128
            config["training_opt"]["num_epochs"] = 150

        config["training_opt"]["sampler"] = False

        # final name of the experiment
        exp_name = f'{experiments[experiment]}_{exp_name_template}_{config["training_opt"]["backbone"]}'

        # directory for logging
        config["training_opt"]["log_dir"] = f'./{init_dir}/{dataset_name}/{imbalance_ratio_names[imb_ratio]}/{exp_name}'

        # run name for wandb
        config["training_opt"]["stage"] = exp_name

        # point generation sequence related config
        config["pg"]["generate"] = True #Make this True to generate points
        config["pg"]["tukey"] = True
        config["pg"]["tukey_value"] = float(custom_var1)
        config["pg"]["alpha"] = float(custom_var4)
        config["pg"]["extra_points"] = 0
        config["pg"]["topk"] = int(custom_var5)
        config["pg"]["distance_analysis"] = False
        config["pg"]["nn_analysis"] = False
        config["pg"]["nn_analysis_k"] = 10
        config["pg"]["tsne"] = False

        common = f"FreeLunch_Tukey_{float(custom_var1)}_LR_{float(custom_var2)}_momen_{float(custom_var3)}_alpha_{float(custom_var4)}_topK_{float(custom_var5)}_NoScheduler_Extra_{config['pg']['extra_points']}"
        # generation log directory
        config["training_opt"]["log_generate"] = config["training_opt"]["log_dir"] + f"/generate/" + common

        # retraining sequence related config
        config["training_opt"]["log_retrain"] = config["training_opt"]["log_dir"] + f"/retrain/" + common + f"_retrain_{float(custom_var6)},{float(custom_var7)}_LR_{float(custom_var8)}_BS_{float(custom_var9)}"

    else:
        print(f"Wrong experiments setup!-{experiment}")

    if g.log_offline:
        g.log_dir = config["training_opt"]["log_dir"]
    return config

# Use for any debugging/checking
# for exp in [experiments.keys()][0]:
#     for data in datasets.keys():
#         for imb_r in imbalance_ratios.keys():
#             config = experiment_maker(exp, data, imb_r, data_root="./datasets/CIFAR100")
#             print(config)
