import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import importlib
import pdb
import wandb
import pandas as pd
from pandas.plotting import table

import libs.utils.globals as g


def update(config, args):
    # Change parameters
    config["model_dir"] = get_value(config["model_dir"], args.model_dir)
    config["training_opt"]["batch_size"] = get_value(
        config["training_opt"]["batch_size"], args.batch_size
    )
    return config


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location("", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def print_write(print_str, log_file):
    print(*print_str)
    if g.log_offline:
        if log_file is None:
            return
        with open(log_file, "a") as f:
            print(*print_str, file=f)


def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print(
        "Pretrained %s weights path: %s"
        % ("classifier" if classifier else "feature model", weights_path)
    )
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {
                k: weights[k] if k in weights else model.state_dict()[k]
                for k in model.state_dict()
            }
        else:
            weights = weights["state_dict_best"]["feat_model"]
            weights = {
                k: weights["module." + k]
                if "module." + k in weights
                else model.state_dict()[k]
                for k in model.state_dict()
            }
    else:
        weights = weights["state_dict_best"]["classifier"]
        weights = {
            k: weights["module.fc." + k]
            if "module.fc." + k in weights
            else model.state_dict()[k]
            for k in model.state_dict()
        }
    model.load_state_dict(weights)
    return model


def init_weights_rahul(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print(
        "Pretrained %s weights path: %s"
        % ("classifier" if classifier else "feature model", weights_path)
    )
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {
                k: weights[k] if k in weights else model.state_dict()[k]
                for k in model.state_dict()
            }
        else:
            weights = weights["state_dict_best"]["feat_model"]
            weights = {
                k: weights["module." + k]
                if "module." + k in weights
                else model.state_dict()[k]
                for k in model.state_dict()
            }
    else:
        weights = weights["state_dict_best"]["classifier"]
        weights = {
            k: weights["module." + k]
            if "module." + k in weights
            else model.state_dict()[k]
            for k in model.state_dict()
        }
    model.load_state_dict(weights)
    return model


def shot_acc(
    preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False
):

    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError("Type ({}) of preds not supported".format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def weighted_shot_acc(
    preds, labels, ws, train_data, many_shot_thr=100, low_shot_thr=20
):

    training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError("Type ({}) of preds not supported".format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(ws[labels == l].sum())
        class_correct.append(
            ((preds[labels == l] == labels[labels == l]) * ws[labels == l]).sum()
        )

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def F_measure(preds, labels, theta=None):
    # Regular f1 score
    return f1_score(
        labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro"
    )


def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (
            lam * preds.eq(targets_a.data).cpu().sum().float()
            + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()
        ) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def weighted_mic_acc_cal(preds, labels, ws):
    acc_mic_top1 = ws[preds == labels].sum() / ws.sum()
    return acc_mic_top1


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


# New Added
def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x


def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score


def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce


def get_priority(ptype, logits, labels):
    if ptype == "score":
        ws = 1 - logits2score(logits, labels)
    elif ptype == "entropy":
        ws = logits2entropy(logits)
    elif ptype == "CE":
        ws = logits2CE(logits, labels)

    return ws


def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv


# Tang Kaihua New Add
def print_grad_norm(named_parameters, logger_func, log_file, verbose=False):
    if not verbose:
        return None

    total_norm = 0.0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters.items():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1.0 / 2)

    logger_func(
        ["----------Total norm {:.5f}-----------------".format(total_norm)], log_file
    )
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
        logger_func(
            ["{:<50s}: {:.5f}, ({})".format(name, norm, param_to_shape[name])], log_file
        )
    logger_func(["-------------------------------"], log_file)

    return total_norm


def smooth_l1_loss(input, target, beta=1.0 / 9, reduction="mean"):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        print("XXXXXX Error Reduction Type for smooth_l1_loss, use default mean")
        return loss.mean()


def l2_loss(input, target, reduction="mean"):
    return F.mse_loss(input, target, reduction=reduction)


def regression_loss(
    input,
    target,
    l2=False,
    pre_mean=True,
    l1=False,
    moving_average=False,
    moving_ratio=0.01,
):
    assert (l2 + l1 + moving_average) == 1
    if l2:
        if input.shape[0] == target.shape[0]:
            assert not pre_mean
            loss = l2_loss(input, target.clone().detach())
        else:
            assert pre_mean
            loss = l2_loss(input, target.clone().detach().mean(0, keepdim=True))
    elif l1:
        loss = smooth_l1_loss(input, target.clone().detach())
    elif moving_average:
        # input should be register_buffer rather than nn.Parameter
        with torch.no_grad():
            input = (
                1 - moving_ratio
            ) * input + moving_ratio * target.clone().detach().mean(0, keepdim=True)
        loss = None
    return loss


def gumbel_softmax(logits, tau=1, hard=False, gumbel=True, dim=-1):
    if gumbel:
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
    else:
        y_soft = logits.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_sigmoid(logits, tau=1, hard=False, gumbel=True):
    if gumbel:
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = torch.sigmoid(gumbels)
    else:
        y_soft = torch.sigmoid(logits)

    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

# Warmup from BBN
from bisect import bisect_right

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

# Rahul Vigneswaran
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import confusion_matrix
# from cuml.manifold import TSNE
from sklearn.manifold import TSNE
# from tsnecuda import TSNE
from more_itertools import sort_together

import libs.utils.globals as g



def plot_tsne(embedding, labels, phase="train"):
    """Function to plot tsne

    Args:
        embedding (float Tensor): Embedding of data. Batch Size x Embedding Size
        labels (int): Ground truth.
        phase (str, optional): Is the plot for train data or validation data or test data? Defaults to "train".
    """
    X_tsne = TSNE(n_components=2).fit_transform(embedding)
    tsne_x = X_tsne[:, 0]
    tsne_y = X_tsne[:, 1]

    tsne_x = sort_together([labels,tsne_x])[1]
    tsne_y = sort_together([labels,tsne_y])[1]
    labels = sort_together([labels,labels])[1]
    
    sym = [0, 1, 4, 24, 5, 3, 17, 13, 26, 20]
    classes = {
        0: "plane",
        1: "car",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    class_label = [classes[i] for i in labels]

    df = pd.DataFrame(
        list(zip(tsne_x, tsne_y, class_label)), columns=["x", "y", "Class"]
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Class",
        symbol="Class",
        symbol_sequence=sym,
        hover_name=class_label,
        labels={"color": "Class"},
    )

    if g.wandb_log:
        if phase == "train":
            wandb.log({"t-SNE": fig, "epoch": g.epoch_global})
        elif phase == "val":
            wandb.log({"t-SNE Eval": fig, "epoch": g.epoch_global})
        elif phase == "test":
            wandb.log({"t-SNE Test": fig, "epoch": g.epoch_global})
        else:
            raise Exception("Invalid data split!!")
    
    if g.log_offline:
        if phase == "train":
            fig.write_image(f"{g.log_dir}/metrics/tsne.png")
        elif phase == "val":
            fig.write_image(f"{g.log_dir}/metrics/tsneEval.png")
        elif phase == "test":    
            fig.write_image(f"{g.log_dir}/metrics/tsneTest.png")
        else:
            raise Exception("Invalid data split!!")

def plot_tsne_with_name(embedding, labels, name="train"):
    """Function to plot tsne

    Args:
        embedding (float Tensor): Embedding of data. Batch Size x Embedding Size
        labels (int): Ground truth.
        phase (str, optional): Is the plot for train data or validation data or test data? Defaults to "train".
    """
    X_tsne = TSNE(n_components=2).fit_transform(embedding)
    tsne_x = X_tsne[:, 0]
    tsne_y = X_tsne[:, 1]

    tsne_x = sort_together([labels,tsne_x])[1]
    tsne_y = sort_together([labels,tsne_y])[1]
    labels = sort_together([labels,labels])[1]
    
    sym = [0, 1, 4, 24, 5, 3, 17, 13, 26, 20]
    classes = {
        0: "plane",
        1: "car",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    class_label = [classes[i] for i in labels]

    df = pd.DataFrame(
        list(zip(tsne_x, tsne_y, class_label)), columns=["x", "y", "Class"]
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Class",
        symbol="Class",
        symbol_sequence=sym,
        hover_name=class_label,
        labels={"color": "Class"},
    )

    if g.wandb_log:
        actual_name = f"{name} t-SNE"
        wandb.log({actual_name: fig})

    if g.log_offline:
        actual_name = f"{name} t-SNE"
        fig.write_image(f"{g.log_dir}/metrics/{actual_name}.png")

def plot_tsne_with_name_with_mark_key(embedding, labels, mark_key, name="all", dir=None):
    """Function to plot tsne

    Args:
        embedding (float Tensor): Embedding of data. Batch Size x Embedding Size
        labels (int): Ground truth.
        phase (str, optional): Is the plot for train data or validation data or test data? Defaults to "train".
    """
    X_tsne = TSNE(n_components=2).fit_transform(embedding)
    tsne_x = X_tsne[:, 0]
    tsne_y = X_tsne[:, 1]

    tsne_x = sort_together([labels,tsne_x])[1]
    tsne_y = sort_together([labels,tsne_y])[1]
    labs = sort_together([labels,labels])[1]
    marker_keys = sort_together([labels, mark_key])[1]

    sym = [0, 26, 29, 41] #[0, 1, 4, 24, 5, 3, 17, 13, 26, 20]
    classes = {
        0: "plane",
        1: "car",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    class_label = [classes[int(i.cpu().numpy())] for i in labs]
    marker_keys = [int(i.cpu().numpy()) for i in marker_keys]

    df = pd.DataFrame(
        list(zip(tsne_x, tsne_y, class_label, marker_keys)), columns=["x", "y", "Class", "Keys"]
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Class",
        symbol="Keys",
        symbol_sequence=sym,
        hover_name=class_label,
        labels={"color": "Keys"},
    )

    # fig.show()
    if g.wandb_log:
        actual_name = f"{name} t-SNE"
        wandb_log({actual_name: fig})

    if g.log_offline:
        actual_name = f"{name} t-SNE"
        fig.write_image(f"{dir}/{actual_name}.png")

def plot_confusion(preds, labels, phase = "train"):
    """Function to plot confusion matrix (both usual and normalized plots)

    Args:
        labels (int): N
        preds (int): N
        phase (str, optional): Is the plot for train data or validation data or test data? Defaults to "train".
    """
    cfm = confusion_matrix(labels, preds)
    
    for normalize in [False, True]:
        z = cfm
        
        if normalize:
            n_digits = 4
            
            z -= z.min()
            z = z/z.max()
            
            z = np.around(z, n_digits)
            
            # z = torch.from_numpy(np.around(z.numpy(), n_digits))
            

        x = y = [
            "0 : plane",
            "1: car",
            "2: bird",
            "3: cat",
            "4: deer",
            "5: dog",
            "6: frog",
            "7: horse",
            "8: ship",
            "9: truck",
        ]

        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]

        # set up figure
        fig = ff.create_annotated_heatmap(
            z, x=x, y=y, annotation_text=z_text, colorscale="Viridis"
        )

        # add custom xaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=0.5,
                y=-0.15,
                showarrow=False,
                text="Predicted Label",
                xref="paper",
                yref="paper",
            )
        )

        # add custom yaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=-0.35,
                y=0.5,
                showarrow=False,
                text="True Label",
                textangle=-90,
                xref="paper",
                yref="paper",
            )
        )

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig["data"][0]["showscale"] = True

        if g.wandb_log:
            if phase == "train":
                if normalize:
                    wandb.log({"Train Confusion Martix (Normalized)": fig})
                else:
                    wandb.log({"Train Confusion Martix": fig})
            elif phase == "val":
                if normalize:
                    wandb.log({"Eval Confusion Martix (Normalized)": fig})
                else:
                    wandb.log({"Eval Confusion Martix": fig})
            elif phase == "test":
                if normalize:
                    wandb.log({"Test Confusion Martix (Normalized)": fig})
                else:
                    wandb.log({"Test Confusion Martix": fig})
            else:
                raise Exception("Invalid data split!!")
                    
        if g.log_offline:
            if phase == "train":
                if normalize:
                    fig.write_image(f"{g.log_dir}/metrics/Train_Confusion_Martix_(Normalized).png")
                else:
                    fig.write_image(f"{g.log_dir}/metrics/Train_Confusion_Martix.png")
            elif phase == "val":
                if normalize:
                    fig.write_image(f"{g.log_dir}/metrics/Eval_Confusion_Martix_(Normalized).png")
                else:
                    fig.write_image(f"{g.log_dir}/metrics/Eval_Confusion_Martix.png")
            elif phase == "test":
                if normalize:
                    fig.write_image(f"{g.log_dir}/metrics/Test_Confusion_Martix_(Normalized).png")
                else:
                    fig.write_image(f"{g.log_dir}/metrics/Test_Confusion_Martix.png")
            else:
                raise Exception("Invalid data split!!")


def plot_confusion_with_name(preds, labels, phase="train", name = "none"):
    """Same confusion matrix function as above but with an addition name for logging purpose (both usual and normalized plots)

    Args:
        labels (int): N
        preds (int): N
        phase (str, optional): Is the plot for train data or validation data or test data? Defaults to "train".
        name (str, optional): Addition name str for certain logging scenarios. Defaults to "none".
    """
    cfm = confusion_matrix(labels, preds)
    
    for normalize in [False, True]:
        z = cfm
    
        if normalize:
            n_digits = 4
            
            z -= z.min()
            z = z/z.max()
            
            z = np.around(z, n_digits)

        x = y = [
            "0 : plane",
            "1: car",
            "2: bird",
            "3: cat",
            "4: deer",
            "5: dog",
            "6: frog",
            "7: horse",
            "8: ship",
            "9: truck",
        ]

        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]

        # set up figure
        fig = ff.create_annotated_heatmap(
            z, x=x, y=y, annotation_text=z_text, colorscale="Viridis"
        )

        # add custom xaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=0.5,
                y=-0.15,
                showarrow=False,
                text="Predicted Label",
                xref="paper",
                yref="paper",
            )
        )

        # add custom yaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=-0.35,
                y=0.5,
                showarrow=False,
                text="True Label",
                textangle=-90,
                xref="paper",
                yref="paper",
            )
        )

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig["data"][0]["showscale"] = True
        
        if g.wandb_log:
            if phase == "train":
                if normalize:
                    wandb.log({f"Train Confusion Martix (Normalized) - {name}": fig})
                else:
                    wandb.log({f"Train Confusion Martix - {name}": fig})
            elif phase == "val":
                if normalize:
                    wandb.log({f"Eval Confusion Martix (Normalized) - {name}": fig})
                else:
                    wandb.log({f"Eval Confusion Martix - {name}": fig})
            elif phase == "test":
                if normalize:
                    wandb.log({f"Test Confusion Martix (Normalized) - {name}": fig})
                else:
                    wandb.log({f"Test Confusion Martix - {name}": fig})
            else:
                raise Exception("Invalid data split!!")
                    
        if g.log_offline:
            if phase == "train":
                if normalize:
                    fig.write_image(f"{g.log_dir}/metrics/Train_Confusion_Martix_(Normalized)_{name}.png")
                else:
                    fig.write_image(f"{g.log_dir}/metrics/Train_Confusion_Martix_{name}.png")
            elif phase == "val":
                if normalize:
                    fig.write_image(f"{g.log_dir}/metrics/Eval_Confusion_Martix_(Normalized)_{name}.png")
                else:
                    fig.write_image(f"{g.log_dir}/metrics/Eval_Confusion_Martix_{name}.png")
            elif phase == "test":
                if normalize:
                    fig.write_image(f"{g.log_dir}/metrics/Test_Confusion_Martix_(Normalized)_{name}.png")
                else:
                    fig.write_image(f"{g.log_dir}/metrics/Test_Confusion_Martix_{name}.png")
            else:
                raise Exception("Invalid data split!!")


def prediction_change_finder(old, new, labels, class_names, phase):
    """[summary]

    Args:
        old (Python list): List of predictions from the Nearest Neighbour on the features directly from the feature extractor.
        new (Python list): List of predictions on the embedding after being trained on any desired loss (eg, protoloss)
        labels (Python list): List of ground truth labels
        class_names (Python list): List of class names
        phase (str): Is the data from train or eval set or test set?

    Returns:
        corrected (Python list): List of counts of samples that were misclassified in old and corrected in new
        wronged (Python list): List of counts of samples that were correct in old and misclassified in new
        stayed_correct (Python list): List of counts of samples that were correct in both old and new
        stayed_wrong (Python list): List of counts of samples that were wrong in both old and new
        class_total (Python list): List of total number of samples in each class
    """

    nums = len(class_names)
    
    corrected = list(0 for i in range(nums))
    wronged = list(0 for i in range(nums))
    stayed_correct = list(0 for i in range(nums))
    stayed_wrong = list(0 for i in range(nums))
    
    classwise_correct_after = list(0 for i in range(nums))
    classwise_correct_before = list(0 for i in range(nums))
    
    class_correct = list(0 for i in range(nums))
    class_total = list(0 for i in range(nums))
    
    for i in range(len(labels)):
        correct_label = labels[i].cpu()
        if old[i] == correct_label :
            classwise_correct_before[correct_label] +=1
            if new[i] == correct_label :
                classwise_correct_after[correct_label] +=1
                stayed_correct[correct_label] += 1
            else:
                wronged[correct_label] += 1
        else:
            if new[i] == correct_label :
                classwise_correct_after[correct_label] +=1
                corrected[correct_label] += 1
            else:
                stayed_wrong[correct_label] += 1      
        class_total[correct_label] += 1
    
    # corrected = torch.FloatTensor(corrected).unsqueeze(1)
    # wronged = torch.FloatTensor(wronged).unsqueeze(1)
    # stayed_correct = torch.FloatTensor(stayed_correct).unsqueeze(1)
    # stayed_wrong = torch.FloatTensor(stayed_wrong).unsqueeze(1) 
    # class_total = torch.FloatTensor(class_total).unsqueeze(1)
    
    metrics = {}
    metrics["Wrong -> Correct"] = corrected
    metrics["(W->C)%"] = [(corrected[i]/class_total[i])*100 for i in range(len(class_total))]
    metrics["Correct -> Wrong"] = wronged
    metrics["(C->W)%"] = [(wronged[i]/class_total[i])*100 for i in range(len(class_total))]
    metrics["Stayed Correct"] = stayed_correct
    metrics["(Stayed Correct)%"] = [(stayed_correct[i]/class_total[i])*100 for i in range(len(class_total))]
    metrics["Stayed Wrong"] = stayed_wrong
    metrics["(Stayed Wrong)%"] = [(stayed_wrong[i]/class_total[i])*100 for i in range(len(class_total))]
    metrics["Class Total"] = class_total
    metrics["Accuracy_FeatureSpace"] = [(classwise_correct_before[i]/class_total[i])*100 for i in range(len(class_total))]
    metrics["Accuracy_EmbeddingSpace"] = [(classwise_correct_after[i]/class_total[i])*100 for i in range(len(class_total))]
    
    if g.wandb_log:
        if phase == "train":
            df = pd.DataFrame(metrics).astype("float").round(5)
            df.insert(0, column="Class", value=class_names)
            wandb.log({f"Train: Raw to Trained metrics": wandb.Table(dataframe=df)})
        elif phase == "val":
            df = pd.DataFrame(metrics).astype("float").round(5)
            df.insert(0, column="Class", value=class_names)
            wandb.log({f"Eval: Raw to Trained metrics": wandb.Table(dataframe=df)})
        elif phase == "test":
            df = pd.DataFrame(metrics).astype("float").round(5)
            df.insert(0, column="Class", value=class_names)
            wandb.log({f"Test: Raw to Trained metrics": wandb.Table(dataframe=df)})
        else:
            raise Exception("Invalid data split!!")
    
    if g.log_offline:
        if phase == "train":
            df = pd.DataFrame(metrics).astype("float").round(5)
            df.insert(0, column="Class", value=class_names)
            fig, ax = plt.subplots(figsize=(25, 5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            tabla = table(ax, df, loc='upper right', colWidths=[0.05]*len(df.columns))  # where df is your data frame
            tabla.auto_set_font_size(True) # Activate set fontsize manually
            tabla.set_fontsize(15) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.7, 2) # change size table
            plt.savefig(f"{g.log_dir}/metrics/TrainRawtoTrainedmetrics.png", transparent=False)
        elif phase == "val":
            df = pd.DataFrame(metrics).astype("float").round(5)
            df.insert(0, column="Class", value=class_names)
            fig, ax = plt.subplots(figsize=(25, 5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            tabla = table(ax, df, loc='upper right', colWidths=[0.05]*len(df.columns))  # where df is your data frame
            tabla.auto_set_font_size(True) # Activate set fontsize manually
            tabla.set_fontsize(15) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.7, 2) # change size table
            plt.savefig(f"{g.log_dir}/metrics/EvalRawtoTrainedmetrics.png", transparent=False)
        elif phase == "test":
            df = pd.DataFrame(metrics).astype("float").round(5)
            df.insert(0, column="Class", value=class_names)
            fig, ax = plt.subplots(figsize=(25, 5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            tabla = table(ax, df, loc='upper right', colWidths=[0.05]*len(df.columns))  # where df is your data frame
            tabla.auto_set_font_size(True) # Activate set fontsize manually
            tabla.set_fontsize(15) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.7, 2) # change size table
            plt.savefig(f"{g.log_dir}/metrics/TestRawtoTrainedmetrics.png", transparent=False)
        else:
            raise Exception("Invalid data split!!")
                        
    # return corrected, wronged, stayed_correct, stayed_wrong, class_total
    
    
def wandb_log(dict1):
    if g.wandb_log:
        dict1["epoch"] = g.epoch_global
        dict1["iter"] = g.step_global
        wandb.log(dict1)

def get_sampler_dict(sampler_defs):
    if sampler_defs:
        if sampler_defs["type"] == "ClassAwareSampler": # Inverse Sampler
            sampler_dic = {
                "sampler": source_import(sampler_defs["def_file"]).get_sampler(),
                "params": {"num_samples_cls": sampler_defs["num_samples_cls"]},
            }
        elif sampler_defs["type"] in ["MixedPrioritizedSampler","ClassPrioritySampler",]:
            sampler_dic = {
                "sampler": source_import(sampler_defs["def_file"]).get_sampler(),
                "params": {k: v for k, v in sampler_defs.items() if k not in ["type", "def_file"]},
            }
    else:
        sampler_dic = None

    return sampler_dic

            
