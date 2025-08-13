import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BaselineLinear(nn.Module):
    """A minimal baseline that predicts 4-component probabilities.

    The model averages inputs across sequence length to get 5-base frequencies
    (A, T, G, C, N), then applies a linear (or small MLP) classifier.
    Accepts inputs shaped (batch, 5, L) or (batch, 5).
    """

    def __init__(self, input_channels: int = 5, num_classes: int = 4, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden and hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_channels, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.classifier = nn.Linear(input_channels, num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (B, 5, L) or (B, 5)
        if sequence.dim() == 3:
            x = sequence.mean(dim=2)
        else:
            x = sequence
        logits = self.classifier(x)
        return logits


class BaselineFrequencyDataset(Dataset):
    """Dataset that converts DNA sequences to 5-base frequency vectors.

    Frequencies are computed for A, T, G, C, and N. Targets are normalized
    to a valid probability distribution across 4 classes.
    """

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        self.dna_dict = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}

    def __len__(self):
        return len(self.sequences)

    def _sequence_to_frequencies(self, sequence: str) -> np.ndarray:
        counts = np.zeros(5, dtype=np.float32)
        total = 0
        for base in sequence:
            idx = self.dna_dict.get(base.upper(), 4)
            counts[idx] += 1.0
            total += 1
        if total > 0:
            counts /= float(total)
        return counts

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        target = self.targets[idx].astype(np.float32)
        total = float(np.sum(target))
        if total <= 0:
            target = np.ones_like(target, dtype=np.float32) / target.shape[0]
        else:
            target = target / total

        features = self._sequence_to_frequencies(seq)
        return {
            "sequence": torch.FloatTensor(features),
            "target": torch.FloatTensor(target),
        }


