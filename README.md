# TailCalibX : Feature Generation for Long-tail Classification
by [Rahul Vigneswaran](https://rahulvigneswaran.github.io/), [Marc T. Law](http://www.cs.toronto.edu/~law/), [Vineeth N. Balasubramanian](https://lab1055.github.io/), [Makarand Tapaswi](https://makarandtapaswi.github.io/)

[[arXiv](https://arxiv.org/abs/2111.05956#:~:text=The%20visual%20world%20naturally%20exhibits,models%20based%20on%20deep%20learning.)] [[Code](https://github.com/rahulvigneswaran/TailCalibX)] [[pip Package](https://pypi.org/project/tailcalib/0.0.1/)] [[Video](https://www.youtube.com/watch?v=BhOtYW2a_pU)] 
![TailCalibX methodology](readme_assets/method.svg "TailCalibX methodology")


# Table of contents
  - [๐ฃ Easy Usage (Recommended way to use our method)](#-easy-usage-recommended-way-to-use-our-method)
    - [๐ป Installation](#-installation)
    - [๐จโ๐ป Example Code](#-example-code)
  - [๐งช Advanced Usage](#-advanced-usage)
    - [โ Things to do before you run the code from this repo](#-things-to-do-before-you-run-the-code-from-this-repo)
    - [๐ How to use?](#-how-to-use)
    - [๐ How to create the mini-ImageNet-LT dataset?](#-how-to-create-the-mini-imagenet-lt-dataset)
    - [โ Arguments](#-arguments)
  - [๐๏ธโโ๏ธ Trained weights](#%EF%B8%8F%EF%B8%8F-trained-weights)
  - [๐ช Results on a Toy Dataset](#-results-on-a-toy-dataset)
  - [๐ด Directory Tree](#-directory-tree)
  - [๐ Citation](#-citation)
  - [๐ Contributing](#-contributing)
  - [โค About me](#-about-me)
  - [โจ Extras](#-extras)
  - [๐ License](#-license)
  
## ๐ฃ Easy Usage (Recommended way to use our method)
โ  **Caution:**  TailCalibX is just TailCalib employed multiple times. Specifically, we generate a set of features once every epoch and use them to train the classifier. In order to mimic that, three things must be done at __every epoch__ in the following order:
1. Collect all the features from your dataloader.
2. Use the `tailcalib` package to make the features balanced by generating samples.
3. Train the classifier.
4. Repeat.

### ๐ป Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install __tailcalib__.

```bash
pip install tailcalib
```

### ๐จโ๐ป Example Code
 Check the instruction [here](tailcalib_pip/README.md) for a much more detailed python package information.

```python
# Import
from tailcalib import tailcalib

# Initialize
a = tailcalib(base_engine="numpy")   # Options: "numpy", "pytorch"

# Imbalanced random fake data
import numpy as np
X = np.random.rand(200,100)
y = np.random.randint(0,10, (200,))

# Balancing the data using "tailcalib"
feat, lab, gen = a.generate(X=X, y=y)

# Output comparison
print(f"Before: {np.unique(y, return_counts=True)}")
print(f"After: {np.unique(lab, return_counts=True)}")
```

## ๐งช Advanced Usage

### โ Things to do before you run the code from this repo
- Change the `data_root` for your dataset in `main.py`.
- If you are using wandb logging ([Weights & Biases](https://docs.wandb.ai/quickstart)), make sure to change the `wandb.init` in `main.py` accordingly.

### ๐ How to use?
- For just the methods proposed in this paper :
    - For CIFAR100-LT: `run_TailCalibX_CIFAR100-LT.sh`
    - For mini-ImageNet-LT : `run_TailCalibX_mini-ImageNet-LT.sh`
- For all the results show in the paper :
    - For CIFAR100-LT: `run_all_CIFAR100-LT.sh`
    - For mini-ImageNet-LT : `run_all_mini-ImageNet-LT.sh`

### ๐ How to create the mini-ImageNet-LT dataset?
Check `Notebooks/Create_mini-ImageNet-LT.ipynb` for the script that generates the mini-ImageNet-LT dataset with varying imbalance ratios and train-test-val splits.
### โ Arguments
- `--seed` : Select seed for fixing it. 
    - Default : `1`
- `--gpu` : Select the GPUs to be used. 
    - Default : `"0,1,2,3"`

- `--experiment`: Experiment number (Check 'libs/utils/experiment_maker.py'). 
    - Default : `0.1`
- `--dataset` : Dataset number. 
    - Choices : `0 - CIFAR100, 1 - mini-imagenet` 
    - Default : `0`
- `--imbalance` : Select Imbalance factor. 
    - Choices : `0: 1, 1: 100, 2: 50, 3: 10` 
    - Default : `1`
- `--type_of_val` : Choose which dataset split to use. 
    - Choices: `"vt": val_from_test, "vtr": val_from_train, "vit": val_is_test`
    - Default : `"vit"`

- `--cv1` to `--cv9` : Custom variable to use in experiments - purpose changes according to the experiment.
    - Default : `"1"`    

- `--train` : Run training sequence
    - Default : `False`
- `--generate` : Run generation sequence
    - Default : `False`
- `--retraining` : Run retraining sequence
    - Default : `False`
- `--resume` : Will resume from the 'latest_model_checkpoint.pth' and wandb if applicable.
    - Default : `False`

- `--save_features` : Collect feature representations.
    - Default : `False`
- `--save_features_phase` : Dataset split of representations to collect.
    - Choices : `"train", "val", "test"`
    - Default : `"train"`

- `--config` : If you have a yaml file with appropriate config, provide the path here. Will override the 'experiment_maker'.
    - Default : `None`

## ๐๏ธโโ๏ธ Trained weights
| Experiment       | CIFAR100-LT (ResNet32, seed 1, Imb 100) | mini-ImageNet-LT (ResNeXt50)|
| -----------      | -----------        | -----------        |
| TailCalib        | [Git-LFS](https://downgit.github.io/#/home?url=https://github.com/rahulvigneswaran/trained_models/tree/main/TailCalibX/CIFAR100-LT/TailCalib)       | [Git-LFS](https://downgit.github.io/#/home?url=https://github.com/rahulvigneswaran/trained_models/tree/main/TailCalibX/mini-ImageNet-LT/TailCalib)       |
| TailCalibX       | [Git-LFS](https://downgit.github.io/#/home?url=https://github.com/rahulvigneswaran/trained_models/tree/main/TailCalibX/CIFAR100-LT/TailCalibX)        |[Git-LFS](https://downgit.github.io/#/home?url=https://github.com/rahulvigneswaran/trained_models/tree/main/TailCalibX/mini-ImageNet-LT/TailCalibX)        |
| CBD + TailCalibX | [Git-LFS](https://downgit.github.io/#/home?url=https://github.com/rahulvigneswaran/trained_models/tree/main/TailCalibX/CIFAR100-LT/CBD%2BTailCalibX)        |[Git-LFS](https://downgit.github.io/#/home?url=https://github.com/rahulvigneswaran/trained_models/tree/main/TailCalibX/mini-ImageNet-LT/CBD%2BTailCalibX)        |

## ๐ช Results on a Toy Dataset 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Yj2qymSm3NgCBqvKn5r_cOiEFl9wGp3J?usp=sharing)

The higher the `Imb ratio`, the more imbalanced the dataset is.
`Imb ratio = maximum_sample_count / minimum_sample_count`.

Check [this notebook](https://colab.research.google.com/drive/1Yj2qymSm3NgCBqvKn5r_cOiEFl9wGp3J?usp=sharing) to play with the toy example from which the plot below was generated.
![](readme_assets/toy_example_output.svg)

## ๐ด Directory Tree
```bash
TailCalibX
โโโ libs
โ   โโโ core
โ   โ   โโโ ce.py
โ   โ   โโโ core_base.py
โ   โ   โโโ ecbd.py
โ   โ   โโโ modals.py
โ   โ   โโโ TailCalib.py
โ   โ   โโโ TailCalibX.py
โ   โโโ data
โ   โ   โโโ dataloader.py
โ   โ   โโโ ImbalanceCIFAR.py
โ   โ   โโโ mini-imagenet
โ   โ       โโโ 0.01_test.txt
โ   โ       โโโ 0.01_train.txt
โ   โ       โโโ 0.01_val.txt
โ   โโโ loss
โ   โ   โโโ CosineDistill.py
โ   โ   โโโ SoftmaxLoss.py
โ   โโโ models
โ   โ   โโโ CosineDotProductClassifier.py
โ   โ   โโโ DotProductClassifier.py
โ   โ   โโโ ecbd_converter.py
โ   โ   โโโ ResNet32Feature.py
โ   โ   โโโ ResNext50Feature.py
โ   โ   โโโ ResNextFeature.py
โ   โโโ samplers
โ   โ   โโโ ClassAwareSampler.py
โ   โโโ utils
โ       โโโ Default_config.yaml
โ       โโโ experiments_maker.py
โ       โโโ globals.py
โ       โโโ logger.py
โ       โโโ utils.py
โโโ LICENSE
โโโ main.py
โโโ Notebooks
โ   โโโ Create_mini-ImageNet-LT.ipynb
โ   โโโ toy_example.ipynb
โโโ readme_assets
โ   โโโ method.svg
โ   โโโ toy_example_output.svg
โโโ README.md
โโโ run_all_CIFAR100-LT.sh
โโโ run_all_mini-ImageNet-LT.sh
โโโ run_TailCalibX_CIFAR100-LT.sh
โโโ run_TailCalibX_mini-imagenet-LT.sh
```
Ignored `tailcalib_pip` as it is for the `tailcalib` pip package.


## ๐ Citation
```
@inproceedings{rahul2021tailcalibX,
    title   = {{Feature Generation for Long-tail Classification}},
    author  = {Rahul Vigneswaran and Marc T. Law and Vineeth N. Balasubramanian and Makarand Tapaswi},
    booktitle = {ICVGIP},
    year = {2021}
}
```

## ๐ Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## โค About me
[Rahul Vigneswaran](https://rahulvigneswaran.github.io/)

## โจ Extras
[๐  Long-tail buzz](https://rahulvigneswaran.github.io/longtail-buzz/) : If you are interested in deep learning research which involves __long-tailed / imbalanced__ dataset, take a look at [Long-tail buzz](https://rahulvigneswaran.github.io/longtail-buzz/) to learn about the recent trending papers in this field.

![](/readme_assets/long_tail-buzz_ss.png)

## ๐ License
[MIT](LICENSE)
