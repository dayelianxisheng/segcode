"""Common utilities for all models"""
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)


def create_scheduler(optimizer, patience=5, factor=0.5):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )


def save_checkpoint(model, optimizer, epoch, loss, is_best, filepath, **kwargs):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, filepath)
    if is_best:
        # Save best model separately
        best_path = filepath.replace(".pth", "_best.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(filepath, model=None, optimizer=None, device=None):
    if not os.path.exists(filepath):
        return 1, float("inf")
    try:
        checkpoint = torch.load(filepath, map_location=device)
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_val_loss", float("inf"))
        return start_epoch, best_loss
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 1, float("inf")
