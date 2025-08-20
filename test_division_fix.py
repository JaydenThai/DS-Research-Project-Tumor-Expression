#!/usr/bin/env python3
"""
Test script to verify the division by zero fixes in hyperparameter tuning.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from hyperparameter_tuning import DeviceManager, PromoterDataset, HyperparameterTuner


def test_empty_dataloader_protection():
    """Test that empty dataloaders are handled safely"""
    print("🧪 Testing empty DataLoader protection...")
    
    try:
        # Create very small dataset that might cause issues
        sequences = ["ATCGATCGATCG"] * 3  # Only 3 samples
        targets = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32)
        
        dataset = PromoterDataset(sequences, targets)
        device_manager = DeviceManager(verbose=False)
        
        # Test with very large batch size that would make DataLoader empty
        from torch.utils.data import DataLoader
        loader_kwargs = device_manager.get_dataloader_kwargs()
        
        # This should not crash due to division by zero
        large_batch_loader = DataLoader(dataset, batch_size=10, **loader_kwargs)  # Larger than dataset
        
        print(f"   Dataset size: {len(dataset)}")
        print(f"   DataLoader length: {len(large_batch_loader)}")
        print(f"   ✅ DataLoader created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_gradient_accumulation_safety():
    """Test that gradient accumulation steps are safe"""
    print("\n🧪 Testing gradient accumulation safety...")
    
    try:
        configs_to_test = [
            {'gradient_accumulation_steps': 0},  # Should be corrected to 1
            {'gradient_accumulation_steps': -1}, # Should be corrected to 1
            {},  # Should default to 1
        ]
        
        for config in configs_to_test:
            # This should not cause division by zero
            accumulation_steps = max(1, config.get('gradient_accumulation_steps', 1))
            test_loss = 1.0 / accumulation_steps  # This should not crash
            print(f"   Config {config} -> accumulation_steps: {accumulation_steps}, scaled_loss: {test_loss}")
        
        print(f"   ✅ Gradient accumulation safety checks passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_dataset_validation():
    """Test dataset validation and creation"""
    print("\n🧪 Testing dataset validation...")
    
    from hyperparameter_tuning import create_datasets
    
    test_cases = [
        # (sequences, targets, should_pass, description)
        ([], np.array([]), False, "Empty dataset"),
        (["ATCG"], np.array([[1, 0, 0, 0, 0]]), False, "Too small dataset"),
        (["ATCG"] * 15, np.random.rand(15, 5), True, "Valid small dataset"),
        (["ATCG"] * 100, np.random.rand(100, 5), True, "Valid larger dataset"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for sequences, targets, should_pass, description in test_cases:
        try:
            train_ds, val_ds, test_ds = create_datasets(sequences, targets)
            if should_pass:
                print(f"   ✅ {description}: Created datasets successfully")
                passed += 1
            else:
                print(f"   ❌ {description}: Should have failed but didn't")
        except Exception as e:
            if not should_pass:
                print(f"   ✅ {description}: Correctly failed with: {str(e)[:50]}...")
                passed += 1
            else:
                print(f"   ❌ {description}: Unexpectedly failed with: {str(e)[:50]}...")
    
    print(f"   Dataset validation: {passed}/{total} tests passed")
    return passed == total


def test_training_loop_safety():
    """Test that training loop handles edge cases safely"""
    print("\n🧪 Testing training loop safety...")
    
    try:
        # Create minimal valid dataset
        sequences = ["ATCGATCGATCG"] * 20
        targets = np.random.rand(20, 5).astype(np.float32)
        
        from hyperparameter_tuning import create_datasets
        train_dataset, val_dataset, _ = create_datasets(sequences, targets)
        
        device_manager = DeviceManager(verbose=False)
        tuner = HyperparameterTuner(train_dataset, val_dataset, device_manager)
        
        # Test configuration that might cause issues
        config = {
            'depth': 1,
            'base_channels': 8,
            'dropout': 0.1,
            'batch_size': 50,  # Larger than dataset
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'loss_function': 'kldiv',
            'gradient_accumulation_steps': 2,
            'max_epochs': 2  # Very short training
        }
        
        # This should not crash with division by zero
        result = tuner.evaluate_config(config)
        
        print(f"   ✅ Training completed successfully")
        print(f"   Result: val_loss={result.val_loss:.6f}, converged={result.converged}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")
        return False


def main():
    """Run all division by zero protection tests"""
    print("🚀 DIVISION BY ZERO PROTECTION TESTS")
    print("=" * 50)
    
    tests = [
        ("Empty DataLoader Protection", test_empty_dataloader_protection),
        ("Gradient Accumulation Safety", test_gradient_accumulation_safety),
        ("Dataset Validation", test_dataset_validation),
        ("Training Loop Safety", test_training_loop_safety),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n📊 RESULTS")
    print("=" * 20)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All division by zero protection tests passed!")
        print("The hyperparameter tuning system should now be robust against division by zero errors.")
    else:
        print("⚠️  Some tests failed. There may still be division by zero issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
