# Imports
import os
import argparse
import yaml
import resource
import torch
import random
import numpy as np
import yaml

# Increase resource limit
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4048, rlimit[1]))

# argparsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int, help="Select seed for fixing it.")
parser.add_argument("--gpu", default="0,1,2,3", type=str, help="Select the GPUs to be used.")

parser.add_argument("--experiment", default=0.1, type=float, help="Experiment number (Check 'libs/utils/experiment_maker.py').")
parser.add_argument("--dataset", default=0, type=int, help="Dataset number. Choice: 0 - CIFAR100, 1 - mini-imagenet.")
parser.add_argument("--imbalance", default=1, type=int, help="Select Imbalance factor. Choice: 0: 1, 1: 100, 2: 50, 3: 10.")
parser.add_argument("--type_of_val", type=str, default="vit", help="Choose which dataset split to use. Choice: vt: val_from_test, vtr: val_from_train, vit: val_is_test")   

parser.add_argument("--cv1", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    
parser.add_argument("--cv2", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    
parser.add_argument("--cv3", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    
parser.add_argument("--cv4", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    
parser.add_argument("--cv5", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    
parser.add_argument("--cv6", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    
parser.add_argument("--cv7", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    
parser.add_argument("--cv8", type=str, default="0.9", help="Custom variable to use in experiments - purpose changes according to the experiment.")  
parser.add_argument("--cv9", type=str, default="1", help="Custom variable to use in experiments - purpose changes according to the experiment.")    

parser.add_argument("--train", default=False, action="store_true", help="Run training sequence?")
parser.add_argument("--generate", default=False, action="store_true", help="Run generation sequence?")
parser.add_argument("--retraining", default=False, action="store_true", help="Run retraining sequence?")
parser.add_argument("--resume", default=False, action="store_true", help="Will resume from the 'latest_model_checkpoint.pth' and wandb if applicable.")

parser.add_argument("--save_features", default=False, action="store_true", help="Collect feature representations.")
parser.add_argument("--save_features_phase", type=str, default="train", help="Dataset split of representations to collect.")

parser.add_argument("--config", type=str, default=None, help="If you have a yaml file with appropriate config, provide the path here. Will override the 'experiment_maker'.")

args = parser.parse_args()

# CUDA devices used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# custom imports
from libs.utils.experiments_maker import experiment_maker
from libs.data import dataloader
from libs.utils.utils import *
import libs.utils.globals as g

# global configs
g.wandb_log = True
g.epoch_global = 0
g.log_offline = True

# Fixing random seed
print(f"=======> Using seed: {args.seed} <========")
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g.seed = args.seed
g.resume = args.resume

# Roots of datasets (Change this according to your current directory)
data_root = {
    "CIFAR100": "./datasets/CIFAR100",
    "mini-imagenet": "/home/rahul_intern/Imagenet/mini_imagenet",
}

# Either use an existing yaml file or use "experiment_maker"
if args.config == None :
    config = experiment_maker(args.experiment, args.dataset, args.imbalance, data_root, args.seed, args.type_of_val, args.cv1, args.cv2, args.cv3, args.cv4, args.cv5, args.cv6, args.cv7, args.cv8, args.cv9)
else:
    config = yaml.load(args.config)

# wandb inits/resume
if g.wandb_log:
    import wandb
    config_dictionary = config
    if args.resume:
        id = torch.load(config["training_opt"]["log_dir"]+"/latest_model_checkpoint.pth")['wandb_id']
        print(f"\nResuming wandb id: {id}!\n")
    else:
        id = wandb.util.generate_id()
        print(f"\nStarting wandb id: {id}!\n")
    wandb.init(
        project="long-tail",
        entity="long-tail",
        reinit=True,
        name=f"{config['training_opt']['stage']}",
        allow_val_change=True,
        save_code=True,
        config=config_dictionary,
        tags=config["wandb_tags"],
        id=id,
        resume="allow",
    )
    wandb.config.update(args, allow_val_change=True)
    config["wandb_id"] = id
else:
    config["wandb_id"] = None

# Create necessary directories for logging
if (args.train or args.generate) and not(args.resume):
    if not os.path.isdir(config["training_opt"]["log_dir"]):
        os.makedirs(config["training_opt"]["log_dir"])
    else:
        raise Exception("Directory already exists!!")
g.log_dir = config["training_opt"]["log_dir"]
if g.log_offline:
    if not os.path.isdir(f"{g.log_dir}/metrics"):
        os.makedirs(f"{g.log_dir}/metrics")

# Save the config as yaml file
ff = open(f'{config["training_opt"]["log_dir"]}/config.yaml', 'w+')
yaml.dump(config, ff, default_flow_style=False, allow_unicode=True)

# Splits
splits = ["train", "val", "test"]

# Generate dataloader for all the splits
data = {
    x: dataloader.load_data(
        data_root=data_root[config["training_opt"]["dataset"].rstrip("_LT")],
        dataset=config["training_opt"]["dataset"],
        phase=x,
        batch_size=config["training_opt"]["batch_size"], #Use 512 above to shave 1 min off the FreeLunch accumulate
        sampler_dic=get_sampler_dict(config["training_opt"]["sampler"]),
        num_workers=config["training_opt"]["num_workers"],
        top_k_class=config["training_opt"]["top_k"] if "top_k" in config["training_opt"] else None,
        cifar_imb_ratio=config["training_opt"]["cifar_imb_ratio"] if "cifar_imb_ratio" in config["training_opt"] else None,
        imb_type=config["training_opt"]["imb_type"] if "imb_type" in config["training_opt"] else None,
        class_order=config["training_opt"]["class_order"] if "class_order" in config["training_opt"] else None,
        balanced=config["training_opt"]["balanced"] if "balanced" in config["training_opt"] else None,
        special_aug=config["training_opt"]["special_aug"] if "special_aug" in config["training_opt"] else False,
        seed=args.seed,
        jitter=config["training_opt"]["jitter"] if "jitter" in config["training_opt"] else True,
        type_of_val=args.type_of_val
    )
    for x in splits
}

# Number of samples in each class
config["training_opt"]["data_count"] = data["train"].dataset.img_num_list
print(config["training_opt"]["data_count"])

# import appropriate core
training_model = source_import(config["core"]).get_core(config, data)

if args.train:
    # training sequence
    print("\nInitiating training sequence!")
    if args.resume:
        training_model.resume_run(config["training_opt"]["log_dir"]+"/latest_model_checkpoint.pth")
    training_model.train()

if not(args.generate) and args.save_features:
    # Accumulate and save features alone
    training_model.reset_model(torch.load(config["training_opt"]["log_dir"]+"/final_model_checkpoint.pth")['state_dict_best'])
    print("Model reset to best model!")
    training_model.accumulate(phase=args.save_features_phase, save=args.save_features)

if args.generate:
    # Point generation sequence
    print("\nInitiating point generation sequence!")
    print("Model reset to best model!")
    training_model.accumulate(phase="train", save=args.save_features)
    training_model.generate_points()

if args.retraining:
    # Retraining sequence
    if not(args.generate) and not(args.save_features):
        training_model.reset_model(torch.load(config["training_opt"]["log_dir"]+"/final_model_checkpoint.pth")['state_dict_best'])
        print("Model reset to best model!")
        training_model.accumulate(phase="train", save=args.save_features)
    print("\nInitiating retraining sequence!")
    training_model.retrain()

print("=" * 25, " ALL COMPLETED ", "=" * 25)
