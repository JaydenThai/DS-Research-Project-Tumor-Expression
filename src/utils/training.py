from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_best_device(prefer_cuda: bool = True) -> Tuple[torch.device, Dict[str, Any], str]:
    """Select the best available device and return DataLoader kwargs.

    Returns (device, loader_kwargs, device_name)
    - On CUDA: enables cudnn.benchmark and returns pin_memory=True
    - On MPS/CPU: returns empty loader kwargs
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            torch.backends.cudnn.benchmark = True  # speed up for fixed-size batches
        except Exception:
            pass
        return device, {"pin_memory": True}, "cuda"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps"), {}, "mps"

    return torch.device("cpu"), {}, "cpu"


def train_epoch(model, train_loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        non_blocking = getattr(device, "type", None) == "cuda"
        sequences = batch["sequence"].to(device, non_blocking=non_blocking)
        targets = batch["target"].to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        logits = model(sequences)
        log_probs = F.log_softmax(logits, dim=1)
        loss = criterion(log_probs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))


def validate_epoch(model, val_loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            non_blocking = getattr(device, "type", None) == "cuda"
            sequences = batch["sequence"].to(device, non_blocking=non_blocking)
            targets = batch["target"].to(device, non_blocking=non_blocking)
            logits = model(sequences)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)
            total_loss += loss.item()
    return total_loss / max(1, len(val_loader))


def evaluate_model(model, test_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            non_blocking = getattr(device, "type", None) == "cuda"
            sequences = batch["sequence"].to(device, non_blocking=non_blocking)
            targets = batch["target"].to(device, non_blocking=non_blocking)
            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return np.vstack(all_predictions), np.vstack(all_targets)



def train_epoch_ce(model, train_loader, criterion: nn.Module, optimizer, device) -> float:
    """Train for one epoch using cross-entropy with integer labels.

    Expects batches with keys "sequence" and "target" where target is LongTensor.
    """
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        non_blocking = getattr(device, "type", None) == "cuda"
        sequences = batch["sequence"].to(device, non_blocking=non_blocking)
        labels = batch["target"].to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))


def validate_epoch_ce(model, val_loader, criterion: nn.Module, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            non_blocking = getattr(device, "type", None) == "cuda"
            sequences = batch["sequence"].to(device, non_blocking=non_blocking)
            labels = batch["target"].to(device, non_blocking=non_blocking)
            logits = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / max(1, len(val_loader))


def evaluate_model_ce(model, test_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            non_blocking = getattr(device, "type", None) == "cuda"
            sequences = batch["sequence"].to(device, non_blocking=non_blocking)
            labels = batch["target"].to(device, non_blocking=non_blocking)
            logits = model(sequences)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_logits), np.concatenate(all_labels)

