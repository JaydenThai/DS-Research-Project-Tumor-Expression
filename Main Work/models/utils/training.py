from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def train_epoch(model, train_loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        sequences = batch["sequence"].to(device)
        targets = batch["target"].to(device)

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
            sequences = batch["sequence"].to(device)
            targets = batch["target"].to(device)
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
            sequences = batch["sequence"].to(device)
            targets = batch["target"].to(device)
            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return np.vstack(all_predictions), np.vstack(all_targets)


