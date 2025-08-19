import torch
import torch.nn as nn


class PromoterCNN(nn.Module):
    """Lightweight CNN for promoter sequence classification.

    This model is designed to work with the hyperparameter tuning system
    and automatically loads the best hyperparameters when available.
    
    Architecture features:
    - Configurable depth (number of conv blocks)
    - Configurable base channels
    - Dropout for regularization
    - Adaptive pooling for variable sequence lengths
    """

    def __init__(self, sequence_length: int = 600, num_blocks: int = 3, base_channels: int = 24, dropout: float = 0.2, num_classes: int = 5):
        super().__init__()
        assert num_blocks >= 1

        conv_layers = []
        in_ch = 5
        out_ch = base_channels
        # First (and main) conv block
        conv_layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=11, padding=5))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(kernel_size=4))
        conv_layers.append(nn.Dropout(dropout))
        in_ch = out_ch

        # Optional second conv block if requested
        if num_blocks >= 2:
            conv_layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=7, padding=3))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.sequence_conv = nn.Sequential(*conv_layers)

        final_ch = in_ch
        # Minimal classifier for simplicity
        self.classifier = nn.Linear(final_ch, num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.sequence_conv(sequence)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits
    
    @classmethod
    def from_best_config(cls, sequence_length: int = 600):
        """
        Create model instance using the best hyperparameters from tuning.
        
        Args:
            sequence_length: Length of input sequences
            
        Returns:
            PromoterCNN instance with optimized hyperparameters
        """
        try:
            # Try to import hyperparameter config
            import sys
            from pathlib import Path
            
            # Add hyperparameter_tuning to path
            hyperparameter_path = Path(__file__).parent.parent.parent / 'hyperparameter_tuning'
            sys.path.append(str(hyperparameter_path))
            
            from config import load_best_config
            
            config = load_best_config('promoter_cnn')
            
            return cls(
                sequence_length=sequence_length,
                num_blocks=config.depth,
                base_channels=config.base_channels,
                dropout=config.dropout,
                num_classes=config.num_classes
            )
            
        except ImportError:
            print("⚠️  Could not load hyperparameter config, using defaults")
            return cls(sequence_length=sequence_length)
        except Exception as e:
            print(f"⚠️  Error loading hyperparameter config: {e}")
            return cls(sequence_length=sequence_length)


class MultiKernelCNN(nn.Module):
    """CNN with multiple fixed kernel sizes (3, 5, 7) for promoter sequence classification.
    
    This model uses three parallel convolutional branches with different kernel sizes
    to capture features at different scales, then combines them for classification.
    No extendable blocks - fixed architecture for simplicity and interpretability.
    
    Architecture:
    - Three parallel conv branches with kernel sizes 3, 5, 7
    - Each branch: Conv1D -> ReLU -> MaxPool -> Dropout
    - Feature concatenation and final classification
    """
    
    def __init__(self, sequence_length: int = 600, base_channels: int = 32, dropout: float = 0.2, num_classes: int = 5):
        super().__init__()
        
        # Input channels (assuming one-hot encoded DNA: A, T, G, C, N)
        in_channels = 5
        
        # Three parallel convolutional branches with different kernel sizes
        # Branch 1: Kernel size 3 (local features)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        
        # Branch 2: Kernel size 5 (medium-range features)  
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=base_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        
        # Branch 3: Kernel size 7 (long-range features)
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=base_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        
        # Global average pooling to handle variable sequence lengths
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate total features after concatenation (3 branches * base_channels)
        total_features = base_channels * 3
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, num_classes)
        )
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-kernel CNN.
        
        Args:
            sequence: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Apply each convolutional branch in parallel
        x3 = self.conv3(sequence)  # Features from kernel size 3
        x5 = self.conv5(sequence)  # Features from kernel size 5  
        x7 = self.conv7(sequence)  # Features from kernel size 7
        
        # Apply global average pooling to each branch
        x3 = self.global_pool(x3).squeeze(-1)  # (batch_size, base_channels)
        x5 = self.global_pool(x5).squeeze(-1)  # (batch_size, base_channels)
        x7 = self.global_pool(x7).squeeze(-1)  # (batch_size, base_channels)
        
        # Concatenate features from all branches
        x = torch.cat([x3, x5, x7], dim=1)  # (batch_size, base_channels * 3)
        
        # Classification
        logits = self.classifier(x)
        return logits


