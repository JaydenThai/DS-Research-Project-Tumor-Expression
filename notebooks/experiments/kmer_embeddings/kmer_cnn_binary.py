#!/usr/bin/env python3
"""
K-mer Embedding CNN for Binary Classification

This implementation uses k-mer embeddings instead of one-hot encoding for DNA sequences.
K-mer encoding has been shown to achieve 93%+ accuracy in DNA sequence classification tasks.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from itertools import product
from typing import List, Optional, Tuple
from pathlib import Path

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# --- K-mer Utility Functions ---
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

def build_kmer_vocab(k: int):
    """Build vocabulary for k-mers of length k."""
    bases = ["A", "C", "G", "T"]
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    vocab = SPECIALS + kmers
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos

def seq_to_kmers(seq: str, k: int) -> List[str]:
    """Convert DNA sequence to k-mer tokens."""
    seq = seq.upper()
    toks: List[str] = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(c not in "ACGT" for c in kmer):
            toks.append("[UNK]")
        else:
            toks.append(kmer)
    return ["[CLS]"] + toks + ["[SEP]"]

class KmerBinaryClassificationDataset(Dataset):
    """Dataset for binary classification using k-mer embeddings."""
    def __init__(self, data, k=6, max_length=None):
        self.data = data.reset_index(drop=True)
        self.k = k
        self.vocab, self.stoi, self.itos = build_kmer_vocab(k)
        self.vocab_size = len(self.vocab)
        
        # Calculate max sequence length in k-mers if not provided
        if max_length is None:
            max_kmers = max(len(seq_to_kmers(seq, k)) for seq in self.data['ProSeq'])
            self.max_length = min(max_kmers, 512)  # Cap at reasonable length
        else:
            self.max_length = max_length
            
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['ProSeq']
        target = self.data.iloc[idx]['binary_classification']
        
        # Convert sequence to k-mer tokens
        kmers = seq_to_kmers(sequence, self.k)
        
        # Convert to token IDs with padding/truncation
        pad_id = self.stoi["[PAD]"]
        unk_id = self.stoi["[UNK]"]
        
        ids = [self.stoi.get(kmer, unk_id) for kmer in kmers[:self.max_length]]
        attention_mask = [1] * len(ids)
        
        # Pad if necessary
        if len(ids) < self.max_length:
            pad_n = self.max_length - len(ids)
            ids += [pad_id] * pad_n
            attention_mask += [0] * pad_n
            
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.bool), target

class KmerEmbeddingCNN(nn.Module):
    """CNN architecture using k-mer embeddings for binary classification."""
    def __init__(self, vocab_size, embedding_dim=128, num_conv_layers=3, 
                 conv_channels=[32, 64, 128], kernel_sizes=[3, 5, 7], pool_sizes=[2, 2, 2],
                 num_fc_layers=1, fc_sizes=[64], dropout_rate=0.3, 
                 use_batch_norm=True, activation='relu', pooling_type='max'):
        super(KmerEmbeddingCNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.dropout_rate = dropout_rate
        
        # Activation function
        activation_map = {
            'relu': nn.ReLU, 
            'leaky_relu': nn.LeakyReLU, 
            'gelu': nn.GELU, 
            'swish': nn.SiLU, 
            'elu': nn.ELU
        }
        self.activation = activation_map[activation]()
        
        # K-mer embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # 0 is [PAD] token
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = embedding_dim
        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_sizes[i], padding=kernel_sizes[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity())
            self.pool_layers.append(nn.MaxPool1d(pool_sizes[i]))
            in_channels = out_channels
            
        # Global pooling
        self.pooling_type = pooling_type
        if pooling_type == 'avg': 
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max': 
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:  # 'both'
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)
            in_channels *= 2
            
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()
        
        if num_fc_layers > 0:
            fc_input_size = in_channels
            for i in range(num_fc_layers):
                fc_output_size = fc_sizes[i]
                self.fc_layers.append(nn.Linear(fc_input_size, fc_output_size))
                self.fc_bn_layers.append(nn.BatchNorm1d(fc_output_size) if use_batch_norm else nn.Identity())
                fc_input_size = fc_output_size
            self.output_layer = nn.Linear(fc_input_size, 1)
        else: 
            self.output_layer = nn.Linear(in_channels, 1)
            
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                
    def forward(self, input_ids, attention_mask=None):
        # Embed k-mer tokens: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x = self.embedding(input_ids)
        x = self.embedding_dropout(x)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        
        # Transpose for Conv1d: [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for i in range(self.num_conv_layers):
            x = self.activation(self.bn_layers[i](self.conv_layers[i](x)))
            x = self.pool_layers[i](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        # Global pooling
        if self.pooling_type == 'both': 
            x = torch.cat([self.global_avg_pool(x), self.global_max_pool(x)], dim=1).squeeze(-1)
        else: 
            x = self.global_pool(x).squeeze(-1)
            
        # Fully connected layers
        for i in range(self.num_fc_layers):
            x = self.activation(self.fc_bn_layers[i](self.fc_layers[i](x)))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        return self.output_layer(x)  # Remove sigmoid for BCEWithLogitsLoss

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=10):
    """Train the model with early stopping."""
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_aucs = []
    
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (input_ids, attention_mask, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_probs = []
        all_val_targets = []
        
        with torch.no_grad():
            for input_ids, attention_mask, targets in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.float().to(device)
                
                outputs = model(input_ids, attention_mask)
                outputs = outputs.squeeze()
                
                val_loss = criterion(outputs, targets)
                epoch_val_loss += val_loss.item()
                
                # Collect predictions for metrics
                probs = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid for probabilities
                preds = (probs > 0.5).astype(int)
                targets_np = targets.cpu().numpy()
                
                all_val_probs.extend(probs)
                all_val_preds.extend(preds)
                all_val_targets.extend(targets_np)
        
        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_targets, all_val_preds)
        val_auc = roc_auc_score(all_val_targets, all_val_probs)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_aucs.append(val_auc)
        
        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc
    }

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test set."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for input_ids, attention_mask, targets in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids, attention_mask)
            outputs = outputs.squeeze()
            
            probs = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid for probabilities
            preds = (probs > 0.5).astype(int)
            targets_np = targets.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets_np)
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets,
        'confusion_matrix': confusion_matrix(all_targets, all_preds)
    }

def plot_training_curves(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Training and validation loss
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation accuracy
    axes[0, 1].plot(epochs, history['val_accuracies'], 'g-', label='Validation Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Validation AUC
    axes[1, 0].plot(epochs, history['val_aucs'], 'm-', label='Validation AUC')
    axes[1, 0].set_title('Validation AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Overfitting indicator
    loss_diff = [val - train for val, train in zip(history['val_losses'], history['train_losses'])]
    axes[1, 1].plot(epochs, loss_diff, 'orange', label='Val Loss - Train Loss')
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, accuracy, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def main():
    print("K-mer Embedding CNN for Binary Classification")
    print("=" * 60)
    
    # Load data
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / 'data' / 'processed' / 'ProSeq_binary_classification.csv'
    print(f"Loading data from: {data_path}")
    
    data_binary = pd.read_csv(data_path)
    data_filtered = data_binary[data_binary['ProSeq'].str.len() >= 600].copy()
    print(f"Dataset size: {len(data_filtered)}")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', 
                                       classes=np.unique(data_filtered['binary_classification']),
                                       y=data_filtered['binary_classification'])
    print(f"Class weights: {class_weights}")
    
    # Split data
    train_val_data, test_data = train_test_split(
        data_filtered, test_size=0.2, random_state=SEED, 
        stratify=data_filtered['binary_classification']
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=0.2, random_state=SEED, 
        stratify=train_val_data['binary_classification']
    )
    
    # Create k-mer datasets
    K_MER_SIZE = 6
    print(f"\nUsing k-mer size: {K_MER_SIZE}")
    
    train_dataset = KmerBinaryClassificationDataset(train_data, k=K_MER_SIZE)
    val_dataset = KmerBinaryClassificationDataset(val_data, k=K_MER_SIZE)
    test_dataset = KmerBinaryClassificationDataset(test_data, k=K_MER_SIZE)
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Max sequence length (k-mers): {train_dataset.max_length}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model - simpler architecture for better training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KmerEmbeddingCNN(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=64,  # Smaller embedding
        num_conv_layers=2,  # Fewer layers
        conv_channels=[32, 64],
        kernel_sizes=[3, 5],
        pool_sizes=[2, 2],
        num_fc_layers=1,
        fc_sizes=[32],  # Smaller FC layer
        dropout_rate=0.2,  # Less dropout
        use_batch_norm=True,
        activation='relu',
        pooling_type='max'
    ).to(device)
    
    # Training setup with class weights
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Lower LR
    
    # Train model
    print(f"\nStarting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                         num_epochs=100, patience=15)
    
    # Plot training curves
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    plot_training_curves(history, save_path=results_dir / 'training_curves.png')
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")
    print(f"AUC: {test_results['auc']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(test_results['targets'], test_results['predictions']))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_results['confusion_matrix'], test_results['accuracy'],
                         save_path=results_dir / 'confusion_matrix.png')
    
    # Save model
    model_path = results_dir / 'kmer_cnn_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    print(f"\nBest validation AUC: {history['best_val_auc']:.4f}")
    print(f"Final test accuracy: {test_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
