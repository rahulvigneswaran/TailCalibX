import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image

# Image statistics
RGB_statistics = {
    "iNaturalist18": {"mean": [0.466, 0.471, 0.380], "std": [0.195, 0.194, 0.192]},
    "default": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key="default", jitter=True, special_aug=False):
    if split == "train" and key=="default" :
        if jitter and not(special_aug): 
            return transforms.Compose(
                [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
                ]
                )
        elif special_aug:
            return  transforms.Compose(
                [   transforms.RandomApply([transforms.ColorJitter(brightness=(0.1, 0.3), contrast=(0.1, 0.3), saturation=(0.1, 0.3), hue=(0.1, 0.3))], p=0.8),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rbg_std),
                    AddGaussianNoise(0., 0.01),
                ]
                )
        else :
            return transforms.Compose(
                [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
                ]
                )
    elif split == "train" and key=="iNaturalist18" :
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
            )   
    else :
        data_transforms = {
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rbg_std),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rbg_std),
                ]
            ),
        }
        return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, template=None, top_k=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                rootalt = root
                self.img_path.append(os.path.join(rootalt, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        # select top k class
        if top_k:
            # only select top k in training, in case train/val/test not matching.
            if "train" in txt:
                max_len = max(self.labels) + 1
                dist = [[i, 0] for i in range(max_len)]
                for i in self.labels:
                    dist[i][-1] += 1
                dist.sort(key=lambda x: x[1], reverse=True)
                # saving
                torch.save(dist, template + "_top_{}_mapping".format(top_k))
            else:
                # loading
                dist = torch.load(template + "_top_{}_mapping".format(top_k))
            selected_labels = {item[0]: i for i, item in enumerate(dist[:top_k])}
            # replace original path and labels
            self.new_img_path = []
            self.new_labels = []
            for path, label in zip(self.img_path, self.labels):
                if label in selected_labels:
                    self.new_img_path.append(path)
                    self.new_labels.append(selected_labels[label])
            self.img_path = self.new_img_path
            self.labels = self.new_labels
        self.img_num_list = list(np.unique(self.labels, return_counts=True)[1])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index


# Load datasets
def load_data(
    data_root,
    dataset,
    phase,
    batch_size,
    top_k_class=None,
    sampler_dic=None,
    num_workers=4,
    shuffle=True,
    cifar_imb_ratio=None,
    imb_type="exp",
    class_order=None,
    balanced=None,
    type_of_val="vft",    #val_from_test or val_from_train or val_is_test
    special_aug=False,
    seed = 1,
    jitter = True,
):
    if imb_type == None:
        imb_type = "exp"
    txt_split = phase
    txt = "./libs/data/%s/%s_%s.txt" % (dataset, dataset, txt_split)
    template = "./libs/data/%s/%s" % (dataset, dataset)

    print("Loading data from %s" % (txt))

    if "CIFAR" in dataset:
        from libs.data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
            
    if dataset in ["iNaturalist18","iNaturalist18_insecta"]:
        print("===> Loading iNaturalist18 statistics")
        key = "iNaturalist18"
    else:
        key = "default"


    if dataset == "CIFAR100_LT":
        if cifar_imb_ratio == 1:
            print("====> CIFAR100 No Imbalance")
        else:
            print("====> CIFAR100 Imbalance Ratio: ", cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(
            phase, imbalance_ratio=cifar_imb_ratio, root=data_root, imb_type=imb_type, class_order=class_order, balanced=balanced, special_aug=special_aug, seed=seed
        )
    else:
        rgb_mean, rgb_std = RGB_statistics[key]["mean"], RGB_statistics[key]["std"]
        if phase not in ["train", "val"]:
            transform = get_data_transform("test", rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key, jitter, special_aug)
        print("Use data transformation:", transform)

        set_ = LT_Dataset(
            data_root, txt, transform, template=template, top_k=top_k_class
        )

    print(len(set_))

    if sampler_dic and phase == "train":
        print("=====> Using sampler: ", sampler_dic["sampler"])
        print("=====> Sampler parameters: ", sampler_dic["params"])
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_dic["sampler"](set_, **sampler_dic["params"]),
            num_workers=num_workers,
        )
    elif phase == "train":
        print("=====> No sampler.")
        print("=====> Shuffle is %s." % (shuffle))
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        print("=====> No sampler.")
        print("=====> Shuffle is %s." % (shuffle))
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

 # ECBD - # For repo specific to CBD paper with much more detailed instructions, check https://github.com/rahulvigneswaran/Class-Balanced-Distillation-for-Long-Tailed-Visual-Recognition.pytorch
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)