# engine.py

import os
import math
from typing import Any, Dict, List, Optional, Union
import json
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from utils import get_eval_metrics


def _as_logits(out):
    """Return logits when model output is either logits or a tuple (logits, h, g)."""
    return out[0] if isinstance(out, tuple) else out


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = False):
        """
        Args:
            patience: How many epochs to wait after the last improvement.
            min_delta: Minimum loss decrease to qualify as improvement.
            verbose: If True, prints each time the metric improves or the counter increases.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.best_model_weights = None

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_weights = model.state_dict()
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_weights = model.state_dict()


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """
    Cosine learning rate scheduler (OpenCLIP style).
    Returns a function lr_adjuster(step) that updates optimizer.param_groups in-place.
    """

    def warmup(step):
        return base_lr * (step + 1) / warmup_length

    def assign_lr(new_lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = new_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = new_lr

    def lr_adjuster(step):
        if step < warmup_length:
            lr = warmup(step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_lr(lr)
        return lr

    return lr_adjuster


def train(
    train_loader,
    val_loader,
    model: nn.Module,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    accum_steps: int = 1,
    **kwargs,
):
    """
    Train the model with gradient accumulation, cosine LR schedule, and early stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        model: Model to train.
        num_epochs: Number of epochs.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW optimizer.
        device: torch.device("cuda") or torch.device("cpu").
        accum_steps: Gradient accumulation steps.
        **kwargs: Extra arguments forwarded to model(...).

    Returns:
        The model with the best validation loss weights loaded.
    """
    named_params = list(model.named_parameters())
    exclude = lambda n, p: (
        p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)

    gain_or_bias = [p for n, p in named_params if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_params if include(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=lr,
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(1, accum_steps))
    total_updates = max(1, updates_per_epoch * max(1, num_epochs))

    lr_scheduler = cosine_lr(
        optimizer=optimizer,
        base_lr=lr,
        warmup_length=int(total_updates * 0.1),
        steps=total_updates,
    )

    loss_fn = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    opt_step = 0  # optimizer step counter

    for epoch in tqdm(range(num_epochs)):
        model.train()
        preds_all = []
        targets_all = []
        total_train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for bidx, (features, coords, patch_size_lv0, labels) in enumerate(tqdm(train_loader)):
            features = features.to(device)
            coords = coords.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(features, coords, patch_size_lv0.to(device), **kwargs)
                logits = _as_logits(out)
                loss = loss_fn(logits, labels)
                loss_scaled = loss / max(1, accum_steps)

            scaler.scale(loss_scaled).backward()
            total_train_loss += float(loss.detach().cpu())

            if ((bidx + 1) % max(1, accum_steps) == 0) or (bidx + 1 == len(train_loader)):
                lr_scheduler(opt_step)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            preds_all.append(logits.argmax(1).detach().cpu().numpy())
            targets_all.append(labels.detach().cpu().numpy())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)
        bacc = balanced_accuracy_score(targets_all, preds_all)

        # ----- Validation phase -----
        if epoch > 1:
            model.eval()
            preds_val = []
            targets_val = []
            total_val_loss = 0.0

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                for features, coords, patch_size_lv0, labels in tqdm(val_loader):
                    try:
                        out = model(
                            features.to(device),
                            coords.to(device),
                            patch_size_lv0.to(device),
                            **kwargs,
                        )
                    except Exception:
                        # Fallback to CPU if needed
                        model.cpu()
                        out = model(
                            features.cpu(),
                            coords.cpu(),
                            patch_size_lv0.cpu(),
                            **kwargs,
                        )
                        model.to(device)

                    logits = _as_logits(out).float()
                    val_loss = loss_fn(logits, labels.to(logits.device))
                    preds_val.append(logits.argmax(1).cpu().numpy())
                    targets_val.append(labels.detach().cpu().numpy())
                    total_val_loss += float(val_loss.detach().cpu())

            avg_val_loss = total_val_loss / max(1, len(val_loader))
            preds_val = np.concatenate(preds_val)
            targets_val = np.concatenate(targets_val)
            bacc_val = balanced_accuracy_score(targets_val, preds_val)

            tqdm.write(
                f"Epoch {epoch}, "
                f"Train BACC: {bacc:.4f}, Val BACC: {bacc_val:.4f}, "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        else:
            tqdm.write(
                f"Epoch {epoch}, "
                f"Train BACC: {bacc:.4f}, Train Loss: {avg_train_loss:.4f}"
            )

    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
    model.eval()
    return model


def eval(test_loader, model: nn.Module, num_classes: int, device: torch.device, prefix: str, **kwargs):
    """
    Evaluate the model on a test set.
    Supports outputs of form logits or (logits, h, g) for OOD scores.

    Args:
        test_loader: DataLoader for test data.
        model: Trained model.
        num_classes: Number of classes.
        device: torch.device.
        prefix: String prefix for metric keys (kept for compatibility).
        **kwargs: Extra arguments forwarded to model(...).

    Returns:
        eval_metrics (dict), outputs (dict)
    """
    slide_ids = test_loader.dataset.slide_data["slide_id"].tolist()
    preds_all, probs_all, targets_all, ids_all = [], [], [], []
    s_h_all, s_g_all = [], []

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for idx, (features, coords, patch_size_lv0, labels) in enumerate(tqdm(test_loader)):
            out = model(
                features.to(device),
                coords.to(device),
                patch_size_lv0.to(device),
                **kwargs,
            )

            if isinstance(out, tuple):
                logits, h, g = out
                s_h_all.append(h.max(dim=1).values.cpu().numpy())
                s_g_all.append(g.squeeze(1).cpu().numpy())
            else:
                logits = out

            logits = logits.float()
            preds = logits.argmax(1)

            if num_classes == 2:
                probs = nn.functional.softmax(logits, dim=1)[:, 1]
                roc_kwargs: Dict[str, Any] = {}
            else:
                probs = nn.functional.softmax(logits, dim=1)
                roc_kwargs = {"multi_class": "ovo", "average": "macro"}

            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(labels.detach().cpu().numpy())
            ids_all.append(slide_ids[idx])

    preds_all = np.concatenate(preds_all)
    probs_all = np.concatenate(probs_all)
    targets_all = np.concatenate(targets_all)

    eval_metrics = get_eval_metrics(
        targets_all=targets_all,
        preds_all=preds_all,
        probs_all=probs_all,
        roc_kwargs=roc_kwargs,
        prefix=prefix,
    )

    outputs: Dict[str, Any] = {
        "slide_id": ids_all,
        "Y": targets_all,
        "Y_hat": preds_all,
        "probs": probs_all,
    }

    if len(s_h_all) > 0:
        outputs["s_h"] = np.concatenate(s_h_all)
        outputs["s_g"] = np.concatenate(s_g_all)

    return eval_metrics, outputs


@torch.no_grad()
def _softmax_T(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Softmax with temperature T applied to logits (h/g)/T."""
    return F.softmax(logits / T, dim=1)

def odin_perturb_features(
    model: nn.Module,
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size_lv0: torch.Tensor,
    T: float,
    eps: float,
    **kwargs,
) -> torch.Tensor:
    """
    Single-step FGSM on input features to maximize -log S_{ŷ}(x; T).
    Uses AMP to reduce memory. Returns a detached perturbed tensor x_tilde.
    """
    if eps <= 0:
        return features.detach()

    x = features.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    # Use AMP for forward to save memory
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits_first = _as_logits(model(x, coords, patch_size_lv0, **kwargs))
        scaled = logits_first / T
        y_hat = scaled.argmax(dim=1)
        loss = F.cross_entropy(scaled, y_hat, reduction="sum")  # == -log S_{ŷ}

    loss.backward()
    # Gradient wrt x may be float16; that's fine for sign()
    x_tilde = (x - eps * x.grad.detach().sign()).detach()
    return x_tilde


def eval_odin(
    loader,
    model: nn.Module,
    num_classes: int,
    device: torch.device,
    T: float,
    eps: float,
    **kwargs,
):
    """
    Evaluate once with given (T, eps). Temperature T is applied to logits.
    If eps>0, do a feature-space FGSM step before the forward.

    Returns:
        metrics (dict), outputs (dict), mean_pmax (float)
    """
    slide_ids = loader.dataset.slide_data["slide_id"].tolist()
    preds_all, probs_all, targets_all, ids_all = [], [], [], []
    s_h_all, s_g_all = [], []

    model.eval()

    for bidx, (features, coords, patch_size_lv0, labels) in enumerate(loader):
        features = features.to(device)
        coords = coords.to(device)
        patch_size_lv0 = patch_size_lv0.to(device)

        if eps > 0:
            with torch.enable_grad():
                x_tilde = odin_perturb_features(
                    model, features, coords, patch_size_lv0, T=T, eps=eps, **kwargs
                )
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(x_tilde, coords, patch_size_lv0, **kwargs)
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(features, coords, patch_size_lv0, **kwargs)

        if isinstance(out, tuple):
            logits, h, g = out
            s_h_all.append(h.max(dim=1).values.cpu().numpy())
            s_g_all.append(g.squeeze(1).cpu().numpy())
        else:
            logits = out

        logits = logits.float()
        probs_T = _softmax_T(logits, T)

        preds = probs_T.argmax(1)
        if num_classes == 2:
            probs = probs_T[:, 1]  # probability of class-1
            roc_kwargs = {}
        else:
            probs = probs_T
            roc_kwargs = {"multi_class": "ovo", "average": "macro"}

        preds_all.append(preds.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        targets_all.append(labels.cpu().numpy())
        ids_all.append(slide_ids[bidx])

    preds_all = np.concatenate(preds_all)
    probs_all = np.concatenate(probs_all)
    targets_all = np.concatenate(targets_all)

    metrics = get_eval_metrics(
        targets_all=targets_all,
        preds_all=preds_all,
        probs_all=probs_all,
        roc_kwargs=roc_kwargs,
        prefix="",
    )

    outputs = {
        "slide_id": ids_all,
        "Y": targets_all,
        "Y_hat": preds_all,
        "probs": probs_all,
    }
    if len(s_h_all) > 0:
        outputs["s_h"] = np.concatenate(s_h_all)
        outputs["s_g"] = np.concatenate(s_g_all)

    # Mean p_max as selection score (keep same behavior as your snippet)
    if num_classes == 2:
        pmax = probs_all  # class-1 prob (note: not max(p,1-p); kept as requested)
    else:
        pmax = probs_all.max(axis=1)
    mean_pmax = float(np.mean(pmax))

    return metrics, outputs, mean_pmax


def search_epsilon(
    val_loader,
    model: nn.Module,
    num_classes: int,
    device: torch.device,
    T: float,
    eps_list: list,
    allow_acc_drop: float,
    json_path: str,
    **kwargs,
):
    """
    Grid-search epsilon on the validation set.

    Strategy:
      1) Evaluate baseline eps=0.0 → base_acc, base_mean_pmax
      2) For eps in eps_list (excluding 0), evaluate (acc, mean_pmax)
      3) Keep the eps with max mean_pmax subject to (base_acc - acc) <= allow_acc_drop
      4) Save a JSON summary

    Returns:
      best_eps (float), base_acc (float), base_mean_pmax (float)
    """
    # Sanitize grid
    grid = sorted(set(abs(float(x)) for x in eps_list))
    if 0.0 not in grid:
        grid = [0.0] + grid

    # Baseline
    base_metrics, _, base_mean_pmax = eval_odin(
        val_loader, model, num_classes, device, T=T, eps=0.0, **kwargs
    )
    base_acc = float(base_metrics.get("acc", 0.0))

    # Search
    best_eps, best_score = 0.0, -1.0
    for eps in grid:
        if eps == 0.0:
            continue
        metrics_g, _, mean_pmax = eval_odin(
            val_loader, model, num_classes, device, T=T, eps=eps, **kwargs
        )
        acc_g = float(metrics_g.get("acc", 0.0))
        acc_drop = max(0.0, base_acc - acc_g)
        score = float(mean_pmax)

        # select by constraint then by score
        if acc_drop <= float(allow_acc_drop) and score > best_score:
            best_score, best_eps = score, eps

    # Persist JSON
    payload = {
        "best_eps": float(best_eps),
        "T": float(T),
        "grid": [float(x) for x in grid],
        "base_val_acc": float(base_acc),
        "base_mean_pmax": float(base_mean_pmax),
        "allow_acc_drop": float(allow_acc_drop),
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return best_eps, base_acc, base_mean_pmax

