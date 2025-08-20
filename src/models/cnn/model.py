# models_promoter_cnn.py
# -*- coding: utf-8 -*-
"""
Lightweight CNN models for promoter sequence classification.

This module provides two architectures with identical public APIs to the user's originals,
but refactored internally for better training stability and to reduce mode collapse
(always predicting one class).

Classes
-------
- PromoterCNN: single-stream CNN with BN, gentler downsampling, and a slightly richer head.
- MultiKernelCNN: three-branch CNN (k=7/9/11) with BN and consistent padding.

Both expect inputs shaped (batch_size, 5, sequence_length) where channels correspond to
[A, T, G, C, N] one-hot. Outputs are logits of shape (batch_size, 5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# PromoterCNN (refactored: BN, stride-2 early downsample, milder dropout)
# ---------------------------------------------------------------------
class PromoterCNN(nn.Module):
    """Improved CNN for promoter sequence classification.

    Input:  (B, 5, L) one-hot (A,T,G,C,N), default L=600
    Output: (B, 5) logits

    Key changes vs original:
    - BatchNorm after convs to stabilise optimisation.
    - Replace aggressive MaxPool1d(4) with stride-2 conv to keep more signal.
    - Keep dropout but tone it down (default 0.2) and distribute it more sensibly.
    - Slightly richer classifier head for class separation.
    """

    def __init__(
        self,
        sequence_length: int = 600,
        num_blocks: int = 4,
        base_channels: int = 64,
        dropout: float = 0.2,
        num_classes: int = 5,
    ):
        super().__init__()
        assert num_blocks >= 1, "num_blocks must be >= 1"

        conv_layers = []
        in_ch = 5
        out_ch = base_channels

        # First block: stride-2 to gently downsample (instead of MaxPool(4))
        conv_layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=11, padding=5, stride=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        in_ch = out_ch

        # Additional blocks (no further downsampling; keep receptive field growth)
        for _ in range(num_blocks - 1):
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.sequence_conv = nn.Sequential(*conv_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # (B, C, 1) -> (B, C)
            nn.LayerNorm(in_ch),
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, num_classes),
        )

        # Kaiming init for convs
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.sequence_conv(sequence)  # (B, C, 1)
        logits = self.classifier(x)       # (B, num_classes)
        return logits

    @classmethod
    def from_best_config(cls, sequence_length: int = 600):
        """
        Create model instance using the best hyperparameters from tuning.

        Returns:
            PromoterCNN instance with optimised hyperparameters if available,
            otherwise falls back to defaults.
        """
        try:
            import sys
            from pathlib import Path

            hyperparameter_path = Path(__file__).parent.parent.parent / "hyperparameter_tuning"
            sys.path.append(str(hyperparameter_path))
            from config import load_best_config

            config = load_best_config("promoter_cnn")
            return cls(
                sequence_length=sequence_length,
                num_blocks=config.depth,
                base_channels=config.base_channels,
                dropout=config.dropout,
                num_classes=config.num_classes,
            )
        except ImportError:
            print("⚠️  Could not load hyperparameter config, using defaults")
            return cls(sequence_length=sequence_length)
        except Exception as e:
            print(f"⚠️  Error loading hyperparameter config: {e}")
            return cls(sequence_length=sequence_length)


# ---------------------------------------------------------------------
# MultiKernelCNN (refactored: BN + consistent padding and naming)
# ---------------------------------------------------------------------
class MultiKernelCNN(nn.Module):
    """CNN with multiple fixed kernel sizes (7, 9, 11) for promoter sequence classification.

    Architecture:
    - Three parallel conv branches with kernel sizes 7, 9, 11
      Each branch: Conv1D -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout
    - Global average pooling
    - Feature concatenation and final classification

    Input:  (B, 5, L) one-hot (A,T,G,C,N), default L=600
    Output: (B, 5) logits
    """

    def __init__(
        self,
        sequence_length: int = 600,
        base_channels: int = 32,
        dropout: float = 0.2,
        num_classes: int = 5,
    ):
        super().__init__()

        in_channels = 5

        # Branch 1: k=7
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        # Branch 2: k=9
        self.branch9 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        # Branch 3: k=11
        self.branch11 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        total_features = base_channels * 3

        self.classifier = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, num_classes),
        )

        # Kaiming init for convs
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x7 = self.branch7(sequence)
        x9 = self.branch9(sequence)
        x11 = self.branch11(sequence)

        # Global average pool each branch and flatten
        x7 = self.global_pool(x7).squeeze(-1)    # (B, base_channels)
        x9 = self.global_pool(x9).squeeze(-1)    # (B, base_channels)
        x11 = self.global_pool(x11).squeeze(-1)  # (B, base_channels)

        # Concatenate features across branches
        x = torch.cat([x7, x9, x11], dim=1)      # (B, base_channels * 3)
        logits = self.classifier(x)              # (B, num_classes)
        return logits


__all__ = ["PromoterCNN", "MultiKernelCNN"]
