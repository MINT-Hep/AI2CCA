import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_TITAN(batch):
    feature = torch.cat([item[0] for item in batch], dim=0)
    coords = torch.from_numpy(np.vstack([item[1] for item in batch])).long()
    patch_size_lv0 = torch.LongTensor([item[2] for item in batch])
    label = torch.LongTensor([item[3] for item in batch])
    return [feature, coords, patch_size_lv0, label]


def get_split_loader(split_dataset, training=False, testing=False, weighted=False, batch_size=1, num_workers=0):
    """
        return either the validation loader or training loader
    """

    kwargs = {'num_workers': num_workers} if device.type == "cuda" else {}

    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size,
                                    sampler=WeightedRandomSampler(weights, len(weights)),
                                    collate_fn=collate_TITAN, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate_TITAN, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler=SequentialSampler(split_dataset),
                                collate_fn=collate_TITAN, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler=SubsetSequentialSampler(ids),
                            collate_fn=collate_TITAN, **kwargs)

    return loader


def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 0, 'pin_memory': False} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset),
                        collate_fn=collate_TITAN, **kwargs)

    return loader


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    unique_classes: Optional[List[int]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return them in a dictionary.

    Args:
        targets_all: True labels.
        preds_all: Predicted labels.
        probs_all: Predicted probabilities for each class. Optional.
        unique_classes: Explicit list of class labels. If None, inferred from targets_all.
        get_report: Whether to compute classification_report (used for weighted F1).
        prefix: Unused in this simplified version, kept for compatibility.
        roc_kwargs: Additional kwargs for roc_auc_score.

    Returns:
        A dict with keys:
            acc, bacc, kappa, nw_kappa, weighted_f1
        And if probs_all is provided and targets are not all the same:
            loss, auroc
    """
    unique_classes = unique_classes if unique_classes is not None else np.unique(targets_all)
    bacc = balanced_accuracy_score(targets_all, preds_all) if len(targets_all) > 1 else 0
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    nw_kappa = cohen_kappa_score(targets_all, preds_all, weights="linear")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(
        targets_all,
        preds_all,
        output_dict=True,
        zero_division=0,
        labels=unique_classes,
    )

    eval_metrics: Dict[str, Any] = {
        "acc": acc,
        "bacc": bacc,
        "kappa": kappa,
        "nw_kappa": nw_kappa,
        "weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if probs_all is not None:
        if len(np.unique(targets_all)) > 1:
            try:
                loss = log_loss(targets_all, probs_all, labels=unique_classes)
                roc_auc = roc_auc_score(targets_all, probs_all, labels=unique_classes, **roc_kwargs)
            except ValueError:
                roc_auc = -1
                loss = -1
            eval_metrics["loss"] = loss
            eval_metrics["auroc"] = roc_auc

    return eval_metrics


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def seed_torch(device, seed: int = 0):
    """
    Set random seeds for reproducibility across Python, NumPy and PyTorch.
    """
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True