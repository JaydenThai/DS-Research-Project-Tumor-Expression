from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PromoterDataset(Dataset):
    """Dataset for promoter sequences only.

    Produces one-hot encoded sequences shaped (5, L) suitable for Conv1d.
    Targets are normalized to a valid probability distribution across 4 classes.
    """

    def __init__(self, sequences: List[str], targets: np.ndarray, max_length: int = 600):
        self.sequences = sequences
        self.targets = targets
        self.max_length = max_length
        # A, T, G, C, N
        self.dna_dict = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}

    def __len__(self) -> int:
        return len(self.sequences)

    def encode_sequence(self, sequence: str) -> np.ndarray:
        seq = sequence
        if len(seq) > self.max_length:
            seq = seq[: self.max_length]
        else:
            seq = seq + "N" * (self.max_length - len(seq))

        encoded = np.array([self.dna_dict.get(base.upper(), 4) for base in seq])
        one_hot = np.zeros((self.max_length, 5), dtype=np.float32)
        one_hot[np.arange(self.max_length), encoded] = 1.0
        return one_hot.T

    def __getitem__(self, idx: int):
        sequence = self.encode_sequence(self.sequences[idx])
        target = self.targets[idx].astype(np.float32)
        total = float(np.sum(target))
        if total <= 0:
            target = np.ones_like(target, dtype=np.float32) / target.shape[0]
        else:
            target = target / total
        return {
            "sequence": torch.FloatTensor(sequence),
            "target": torch.FloatTensor(target),
        }


def load_and_prepare_data(file_path: str) -> Tuple[List[str], np.ndarray]:
    """Load and prepare data for training from a CSV file.

    Expects columns:
    - ProSeq: DNA sequence string
    - Component_1_Probability ... Component_4_Probability
    """

    df = pd.read_csv(file_path)
    prob_cols = [
        "Component_1_Probability",
        "Component_2_Probability",
        "Component_3_Probability",
        "Component_4_Probability",
    ]

    df = df.dropna(subset=["ProSeq"]).dropna(subset=prob_cols)

    sequences = df["ProSeq"].tolist()
    targets = df[prob_cols].values

    valid_sequences: List[str] = []
    valid_targets: list = []
    for i, seq in enumerate(sequences):
        if isinstance(seq, str) and len(seq) > 0:
            valid_sequences.append(seq)
            valid_targets.append(targets[i])

    return valid_sequences, np.array(valid_targets)


