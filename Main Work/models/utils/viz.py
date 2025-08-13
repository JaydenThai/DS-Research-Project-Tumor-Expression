from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_results(train_losses, val_losses, metrics: Dict[str, Dict[str, float]], predictions: np.ndarray, targets: np.ndarray, out_path: str = "cnn_results.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(train_losses, label="Train Loss", color="blue", alpha=0.7)
    axes[0, 0].plot(val_losses, label="Validation Loss", color="red", alpha=0.7)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (MSE)")
    axes[0, 0].set_title("Training Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    component_names_short = ["Comp_1", "Comp_2", "Comp_3", "Comp_4"]
    r2_scores = [metrics[f"Component_{i+1}"]["R2"] for i in range(4)]
    colors = ["blue", "orange", "green", "red"]

    axes[0, 1].bar(component_names_short, r2_scores, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel("R² Score")
    axes[0, 1].set_title("R² Scores by Component")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    mse_scores = [metrics[f"Component_{i+1}"]["MSE"] for i in range(4)]
    axes[0, 2].bar(component_names_short, mse_scores, color=colors, alpha=0.7)
    axes[0, 2].set_ylabel("MSE")
    axes[0, 2].set_title("MSE by Component")
    axes[0, 2].grid(True, alpha=0.3)

    for i in range(3):
        row = 1
        col = i
        axes[row, col].scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10, color=colors[i])
        axes[row, col].plot([0, 1], [0, 1], "r--", alpha=0.8)
        axes[row, col].set_xlabel("Actual Probability")
        axes[row, col].set_ylabel("Predicted Probability")
        axes[row, col].set_title(f"Component {i+1} (R² = {metrics[f'Component_{i+1}']['R2']:.3f})")
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


