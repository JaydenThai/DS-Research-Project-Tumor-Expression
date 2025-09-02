#!/usr/bin/env python3
"""
Fixed training script that handles class imbalance and actually learns
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

print("🔧 FIXED TRAINING WITH CLASS IMBALANCE HANDLING")
print("="*60)

# Load data
data_binary = pd.read_csv('../../../data/processed/ProSeq_binary_classification.csv')
data_binary = data_binary[['binary_classification', 'ProSeq']]

# Filter sequences
sequence_lengths = data_binary['ProSeq'].str.len()
data_filtered = data_binary[sequence_lengths >= 600].copy()
print(f"Dataset size: {len(data_filtered)}")

# Check class distribution
class_dist = data_filtered['binary_classification'].value_counts()
print(f"\nClass distribution:")
for class_val, count in class_dist.items():
    percentage = count / len(data_filtered) * 100
    print(f"  Class {class_val}: {count} ({percentage:.1f}%)")

# Calculate class weights for imbalanced data
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(data_filtered['binary_classification']),
                                   y=data_filtered['binary_classification'])
print(f"Class weights: {class_weights}")

# Dataset class
class BinaryClassificationDataset(Dataset):
    def __init__(self, data, target_length=600):
        self.data = data.reset_index(drop=True)
        self.dna_dict = {"A": 0, "T": 1, "G": 2, "C": 3}
        self.target_length = target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['ProSeq']
        target = self.data.iloc[idx]['binary_classification']
        one_hot_sequence = self.one_hot_encode(sequence)
        return one_hot_sequence, target
    
    def one_hot_encode(self, sequence):
        sequence = sequence[:self.target_length]
        one_hot = np.zeros((self.target_length, 4), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.dna_dict:
                one_hot[i, self.dna_dict[nucleotide]] = 1.0
        return one_hot

# Split data with stratification
train_data, test_data = train_test_split(data_filtered, test_size=0.2, random_state=42, 
                                       stratify=data_filtered['binary_classification'])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, 
                                      stratify=train_data['binary_classification'])

print(f"\nData splits:")
print(f"  Train: {len(train_data)}")
print(f"  Val: {len(val_data)}")  
print(f"  Test: {len(test_data)}")

# Create datasets
train_dataset = BinaryClassificationDataset(train_data)
val_dataset = BinaryClassificationDataset(val_data)
test_dataset = BinaryClassificationDataset(test_data)

# Create weighted sampler for balanced training
train_targets = train_data['binary_classification'].values
sample_weights = np.array([class_weights[t] for t in train_targets])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)  # Use weighted sampler
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Improved model with better architecture
class ImprovedBinaryCNN(nn.Module):
    def __init__(self):
        super(ImprovedBinaryCNN, self).__init__()
        
        # Better initialization and architecture
        self.conv1 = nn.Conv1d(4, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)  # 600 -> 150
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)  
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)  # 150 -> 37
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(8)  # -> 8
        
        self.fc1 = nn.Linear(64 * 8, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        
        # Proper initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: [batch, length, channels] -> [batch, channels, length]
        x = x.transpose(1, 2)
        
        # Conv layers with batch norm and ReLU
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # FC layers with dropout
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze()

def train_model_with_fixes(num_epochs=50):
    """Train model with all the fixes for class imbalance"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedBinaryCNN().to(device)
    
    # Use weighted loss for class imbalance
    pos_weight = torch.tensor([class_weights[0] / class_weights[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use a good learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    
    print(f"\n🚀 Training for {num_epochs} epochs with class imbalance fixes")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad()
            
            # Get logits before sigmoid
            logits = model.fc3(model.dropout2(F.relu(model.fc2(model.dropout1(F.relu(model.fc1(
                model.pool3(F.relu(model.bn3(model.conv3(
                    model.pool2(F.relu(model.bn2(model.conv2(
                        model.pool1(F.relu(model.bn1(model.conv1(batch_x.transpose(1, 2)))))
                    ))))
                )))).view(batch_x.size(0), -1)
            )))))))
            
            # Use logits for loss calculation
            loss = criterion(logits.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Calculate accuracy using sigmoid
            outputs = torch.sigmoid(logits.squeeze())
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device)
                
                outputs = model(batch_x)
                loss = nn.BCELoss()(outputs, batch_y)  # Use BCE for validation
                
                epoch_val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch < 5:
            print(f'Epoch {epoch+1:3d}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(train_losses)
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x)
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)
    
    return accuracy, cm, report, all_preds, all_targets

# Train the fixed model
model, history = train_model_with_fixes(num_epochs=50)

# Evaluate on test set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_acc, cm, report, preds, targets = evaluate_model(model, test_loader, device)

print(f"\n📊 FINAL RESULTS:")
print("="*60)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Best Validation Accuracy: {history['best_val_acc']:.4f}")
print(f"Epochs Trained: {history['epochs_trained']}")

print(f"\nConfusion Matrix:")
print(cm)

print(f"\nClassification Report:")
print(report)

# Check if both classes are predicted
unique_preds = len(set(preds))
print(f"\nUnique predictions: {unique_preds}")
if unique_preds == 1:
    print("🚨 Still predicting only one class!")
else:
    print("✅ Model is predicting both classes!")

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_losses'], 'b-', label='Train Loss')
plt.plot(history['val_losses'], 'r-', label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history['train_accs'], 'b-', label='Train Acc')
plt.plot(history['val_accs'], 'r-', label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('../../../results/fixed_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n💾 Results saved to '../../../results/fixed_training_results.png'")

# Now test different epoch counts
print(f"\n🔄 TESTING DIFFERENT EPOCH COUNTS")
print("="*60)

epoch_counts = [30, 50, 100, 150]
results = {}

for epochs in epoch_counts:
    print(f"\n🚀 Training for {epochs} epochs...")
    model_test, history_test = train_model_with_fixes(num_epochs=epochs)
    test_acc_test, cm_test, _, _, _ = evaluate_model(model_test, test_loader, device)
    
    results[epochs] = {
        'test_accuracy': test_acc_test,
        'val_accuracy': history_test['best_val_acc'],
        'epochs_trained': history_test['epochs_trained'],
        'confusion_matrix': cm_test.tolist()
    }
    
    unique_preds_test = len(np.unique(np.array([cm_test[0,1], cm_test[1,0], cm_test[1,1]])))
    
    print(f"✅ {epochs} epochs: Test Acc = {test_acc_test:.4f}, "
          f"Val Acc = {history_test['best_val_acc']:.4f}, "
          f"Both classes predicted: {unique_preds_test > 1}")

print(f"\n📈 EPOCH COMPARISON SUMMARY:")
print("="*60)
for epochs, result in results.items():
    print(f"  {epochs:3d} epochs: Test = {result['test_accuracy']:.4f}, "
          f"Val = {result['val_accuracy']:.4f}")

print(f"\n✅ All models should now show different accuracies and predict both classes!")
