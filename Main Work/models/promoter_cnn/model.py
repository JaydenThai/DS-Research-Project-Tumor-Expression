import torch
import torch.nn as nn


class PromoterCNN(nn.Module):
    """CNN for predicting component probabilities with configurable depth."""

    def __init__(self, sequence_length: int = 600, num_blocks: int = 2, base_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        assert num_blocks >= 1

        conv_layers = []
        in_ch = 5
        out_ch = base_channels
        for i in range(num_blocks):
            k = 11 if i == 0 else 7
            p = 5 if i == 0 else 3
            conv_layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, padding=p))
            conv_layers.append(nn.ReLU())
            if i < min(2, num_blocks):
                conv_layers.append(nn.MaxPool1d(kernel_size=4))
            conv_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
            out_ch = min(out_ch * 2, base_channels * (2 ** (num_blocks - 1)))
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.sequence_conv = nn.Sequential(*conv_layers)

        final_ch = in_ch
        hidden = max(32, final_ch // 2)
        self.classifier = nn.Sequential(
            nn.Linear(final_ch, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.sequence_conv(sequence)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits


