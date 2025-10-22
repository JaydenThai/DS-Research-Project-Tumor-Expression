"""
Embedding-based CNN models for binary classification of promoter sequences.

This module implements:
1. EmbeddingCNN: One-hot (5x600) → Linear transforms (128→64→32) → CNN
2. PatchingCNN: Non-overlapping patches for motif detection
3. Hyperparameter optimization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import pandas as pd


class EmbeddingCNN(nn.Module):
    """
    CNN with learnable embedding transformations.
    
    Architecture:
    - Input: One-hot encoded DNA (batch, 600, 5) 
    - Linear transforms: 5 → 128 → 64 → 32
    - CNN: 1D/2D convolutions on embedded sequences
    - Output: Binary classification
    """
    
    def __init__(
        self,
        input_channels: int = 5,  # A, T, G, C, N
        sequence_length: int = 600,
        embedding_dims: List[int] = [128, 64, 32],
        conv_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 3],
        use_2d_conv: bool = False,
        dropout: float = 0.3
    ):
        super(EmbeddingCNN, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.embedding_dims = embedding_dims
        self.use_2d_conv = use_2d_conv
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        
        # Learnable linear transformations for embedding
        self.embeddings = nn.ModuleList()
        prev_dim = input_channels
        
        for embed_dim in embedding_dims:
            self.embeddings.append(
                nn.Sequential(
                    nn.Linear(prev_dim, embed_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5)  # Lighter dropout in embeddings
                )
            )
            prev_dim = embed_dim
        
        # Final embedding dimension
        final_embed_dim = embedding_dims[-1]
        
        if use_2d_conv:
            # 2D Convolution approach
            # Treat embedded sequence as 2D: (batch, 1, seq_len, embed_dim)
            self.conv_layers = nn.ModuleList()
            in_channels = 1
            
            for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, filters, 
                                 kernel_size=(kernel_size, min(kernel_size, final_embed_dim)), 
                                 padding=(kernel_size//2, 0)),
                        nn.BatchNorm2d(filters),
                        nn.ReLU(),
                        nn.MaxPool2d((2, 1)),  # Pool only along sequence dimension
                        nn.Dropout2d(dropout)
                    )
                )
                in_channels = filters
            
            # Calculate final dimensions after pooling
            final_seq_len = sequence_length // (2 ** len(conv_filters))
            final_embed_after_conv = final_embed_dim - sum(k-1 for k in kernel_sizes if k <= final_embed_dim)
            final_embed_after_conv = max(1, final_embed_after_conv)
            
            self.fc_input_size = conv_filters[-1] * final_seq_len * final_embed_after_conv
            
        else:
            # 1D Convolution approach
            # Transpose to (batch, embed_dim, seq_len) for Conv1d
            self.conv_layers = nn.ModuleList()
            in_channels = final_embed_dim
            
            for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(filters),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Dropout(dropout)
                    )
                )
                in_channels = filters
            
            # Calculate final sequence length after pooling
            final_seq_len = sequence_length // (2 ** len(conv_filters))
            self.fc_input_size = conv_filters[-1] * final_seq_len
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_channels)
        batch_size = x.size(0)
        
        # Apply embedding transformations
        # Process each position independently
        for embedding_layer in self.embeddings:
            x = embedding_layer(x)  # (batch, seq_len, embed_dim)
        
        if self.use_2d_conv:
            # Add channel dimension for 2D conv: (batch, 1, seq_len, embed_dim)
            x = x.unsqueeze(1)
            
            # Apply 2D convolutions
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            
            # Flatten for FC layers
            x = x.view(batch_size, -1)
            
        else:
            # Transpose for 1D conv: (batch, embed_dim, seq_len)
            x = x.transpose(1, 2)
            
            # Apply 1D convolutions
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            
            # Flatten for FC layers
            x = x.view(batch_size, -1)
        
        # Classification
        output = self.classifier(x)
        return output.squeeze(-1)  # Remove last dimension for binary classification
    
    def log_architecture(self):
        """Log the EmbeddingCNN model architecture details."""
        print(f"Model Architecture:")
        print(f"  Input: One-hot encoded DNA (batch, {self.sequence_length}, {self.input_channels})")
        
        # Embedding layers
        for i, dim in enumerate(self.embedding_dims):
            if i == 0:
                prev_dim = self.input_channels
            else:
                prev_dim = self.embedding_dims[i-1]
            print(f"  Embedding {i+1}: Linear({prev_dim} -> {dim}) + ReLU + Dropout")
        
        # Convolution type
        conv_type = "Conv2D" if self.use_2d_conv else "Conv1D"
        
        # Conv layers
        in_channels = self.embedding_dims[-1]
        for i, (filters, kernel_size) in enumerate(zip(self.conv_filters, self.kernel_sizes)):
            print(f"  {conv_type} {i+1}: ({in_channels} -> {filters}, kernel={kernel_size}) + BatchNorm + ReLU + MaxPool")
            in_channels = filters
        
        print(f"  Classifier: Linear layers -> Sigmoid")
        print(f"  Output: Binary probability")
        print(f"  Total Parameters: {count_parameters(self):,}")


class PatchingCNN(nn.Module):
    """
    CNN with non-overlapping patching for motif detection.
    
    Architecture:
    - Input: One-hot encoded DNA (batch, 600, 5)
    - Patching: Group consecutive nucleotides into patches
    - Embedding: Transform patches to higher dimensions
    - CNN: Detect motifs in patch space
    - Output: Binary classification
    """
    
    def __init__(
        self,
        input_channels: int = 5,  # A, T, G, C, N
        sequence_length: int = 600,
        patch_size: int = 3,  # Group 3 consecutive nucleotides
        patch_embed_dim: int = 20,  # Transform each patch to 20 dimensions
        conv_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3],  # For motif detection (6-11 range)
        motif_sizes: List[int] = [6, 7, 8, 9, 10, 11],  # Explicit motif detection
        dropout: float = 0.3
    ):
        super(PatchingCNN, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.patch_embed_dim = patch_embed_dim
        self.motif_sizes = motif_sizes
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        
        # Calculate number of patches
        self.num_patches = sequence_length // patch_size
        self.patch_input_dim = input_channels * patch_size  # Flattened patch
        
        # Patch embedding: Transform flattened patches to higher dimensions
        self.patch_embedding = nn.Sequential(
            nn.Linear(self.patch_input_dim, patch_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(patch_embed_dim * 2, patch_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Multi-scale motif detection
        self.motif_convs = nn.ModuleList()
        for motif_size in motif_sizes:
            self.motif_convs.append(
                nn.Sequential(
                    nn.Conv1d(patch_embed_dim, 32, kernel_size=motif_size, padding=motif_size//2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)  # Global max pooling
                )
            )
        
        # Standard CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = patch_embed_dim
        
        for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = filters
        
        # Calculate dimensions
        final_seq_len = self.num_patches // (2 ** len(conv_filters))
        motif_features = len(motif_sizes) * 32  # From motif detection
        conv_features = conv_filters[-1] * final_seq_len
        
        total_features = motif_features + conv_features
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_channels)
        batch_size = x.size(0)
        
        # Create non-overlapping patches
        # Truncate sequence to be divisible by patch_size
        truncated_len = (self.sequence_length // self.patch_size) * self.patch_size
        x = x[:, :truncated_len, :]  # (batch, truncated_len, input_channels)
        
        # Reshape to patches: (batch, num_patches, patch_size * input_channels)
        x = x.view(batch_size, self.num_patches, self.patch_input_dim)
        
        # Apply patch embedding
        x = self.patch_embedding(x)  # (batch, num_patches, patch_embed_dim)
        
        # Transpose for Conv1d: (batch, patch_embed_dim, num_patches)
        x = x.transpose(1, 2)
        
        # Multi-scale motif detection
        motif_features = []
        for motif_conv in self.motif_convs:
            motif_out = motif_conv(x)  # (batch, 32, 1)
            motif_features.append(motif_out.squeeze(-1))  # (batch, 32)
        
        motif_features = torch.cat(motif_features, dim=1)  # (batch, len(motif_sizes) * 32)
        
        # Standard CNN processing
        conv_out = x
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(conv_out)
        
        conv_features = conv_out.view(batch_size, -1)  # Flatten
        
        # Combine motif and conv features
        combined_features = torch.cat([motif_features, conv_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        return output.squeeze(-1)
    
    def log_architecture(self):
        """Log the PatchingCNN model architecture details."""
        print(f"Model Architecture:")
        print(f"  Input: One-hot encoded DNA (batch, {self.sequence_length}, {self.input_channels})")
        print(f"  Patching: Group {self.patch_size} nucleotides -> patches (batch, {self.num_patches}, {self.patch_input_dim})")
        print(f"  Patch Embedding: Linear({self.patch_input_dim} -> {self.patch_embed_dim}) + ReLU + Dropout")
        print(f"  Motif Detection: Multi-scale convolutions {self.motif_sizes}")
        
        # Conv layers
        in_channels = self.patch_embed_dim
        for i, (filters, kernel_size) in enumerate(zip(self.conv_filters, self.kernel_sizes)):
            print(f"  Conv1D {i+1}: ({in_channels} -> {filters}, kernel={kernel_size}) + BatchNorm + ReLU + MaxPool")
            in_channels = filters
        
        print(f"  Classifier: Combine features -> Linear layers -> Sigmoid")
        print(f"  Output: Binary probability")
        print(f"  Total Parameters: {count_parameters(self):,}")


class BinaryClassificationDataset(Dataset):
    """Dataset for binary classification with enhanced one-hot encoding."""
    
    def __init__(self, data: pd.DataFrame, target_length: int = 600):
        self.data = data
        self.dna_dict = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
        self.target_length = target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['ProSeq']
        target = self.data.iloc[idx]['binary_classification']
        one_hot_sequence = self.one_hot_encode(sequence)
        return one_hot_sequence, target
    
    def one_hot_encode(self, sequence: str) -> torch.Tensor:
        """One-hot encode DNA sequence with N handling."""
        # Truncate or pad sequence
        if len(sequence) > self.target_length:
            sequence = sequence[:self.target_length]
        else:
            sequence = sequence + "N" * (self.target_length - len(sequence))
        
        # Create one-hot encoding
        one_hot = np.zeros((self.target_length, 5), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide.upper() in self.dna_dict:
                one_hot[i, self.dna_dict[nucleotide.upper()]] = 1.0
        
        return torch.FloatTensor(one_hot)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_embedding_cnn(
    embedding_dims: List[int] = [128, 64, 32],
    conv_filters: List[int] = [64, 128, 256],
    use_2d_conv: bool = False,
    dropout: float = 0.3
) -> EmbeddingCNN:
    """Factory function to create EmbeddingCNN with specified parameters."""
    return EmbeddingCNN(
        embedding_dims=embedding_dims,
        conv_filters=conv_filters,
        use_2d_conv=use_2d_conv,
        dropout=dropout
    )


def create_patching_cnn(
    patch_size: int = 3,
    patch_embed_dim: int = 20,
    conv_filters: List[int] = [32, 64, 128],
    motif_sizes: List[int] = [6, 7, 8, 9, 10, 11],
    dropout: float = 0.3
) -> PatchingCNN:
    """Factory function to create PatchingCNN with specified parameters."""
    return PatchingCNN(
        patch_size=patch_size,
        patch_embed_dim=patch_embed_dim,
        conv_filters=conv_filters,
        motif_sizes=motif_sizes,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test model creation and forward pass
    batch_size, seq_len, input_channels = 4, 600, 5
    
    # Test data
    x = torch.randn(batch_size, seq_len, input_channels)
    
    # Test EmbeddingCNN
    print("Testing EmbeddingCNN...")
    embedding_model_1d = create_embedding_cnn(use_2d_conv=False)
    embedding_model_2d = create_embedding_cnn(use_2d_conv=True)
    
    print(f"EmbeddingCNN (1D) parameters: {count_parameters(embedding_model_1d):,}")
    print(f"EmbeddingCNN (2D) parameters: {count_parameters(embedding_model_2d):,}")
    
    out_1d = embedding_model_1d(x)
    out_2d = embedding_model_2d(x)
    print(f"1D output shape: {out_1d.shape}")
    print(f"2D output shape: {out_2d.shape}")
    
    # Test PatchingCNN
    print("\nTesting PatchingCNN...")
    patching_model = create_patching_cnn()
    print(f"PatchingCNN parameters: {count_parameters(patching_model):,}")
    
    out_patch = patching_model(x)
    print(f"Patching output shape: {out_patch.shape}")
    
    print("\nAll models created successfully!")
