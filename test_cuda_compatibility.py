#!/usr/bin/env python3
"""
Test CUDA compatibility for the hyperparameter tuning system.
"""

import sys
from pathlib import Path
import torch

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from hyperparameter_tuning import DeviceManager, PromoterCNN


def test_cuda_detection():
    """Test CUDA detection and functionality"""
    print("🧪 CUDA COMPATIBILITY TEST")
    print("=" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("❌ CUDA not available - will fall back to CPU/MPS")
    
    return torch.cuda.is_available()


def test_device_manager():
    """Test DeviceManager with CUDA optimizations"""
    print(f"\n🧪 Testing DeviceManager...")
    
    try:
        device_manager = DeviceManager(verbose=True)
        print(f"✅ DeviceManager initialized successfully")
        
        # Test memory info
        if device_manager.device_name == "cuda":
            memory_info = device_manager.get_memory_info()
            print(f"   Memory info: {memory_info}")
            
            # Test cache clearing
            device_manager.clear_cache()
            
        # Test DataLoader kwargs
        kwargs = device_manager.get_dataloader_kwargs()
        print(f"   DataLoader kwargs: {kwargs}")
        
        return device_manager
        
    except Exception as e:
        print(f"❌ DeviceManager failed: {e}")
        return None


def test_cuda_tensor_operations(device_manager):
    """Test CUDA tensor operations and transfers"""
    if device_manager.device_name != "cuda":
        print(f"\n⏭️  Skipping CUDA tensor tests (using {device_manager.device_name})")
        return True
    
    print(f"\n🧪 Testing CUDA tensor operations...")
    
    try:
        # Test basic tensor creation and operations
        test_tensor = torch.randn(1000, 1000, dtype=torch.float32)
        print(f"   Created tensor: {test_tensor.shape} on {test_tensor.device}")
        
        # Move to CUDA
        cuda_tensor = device_manager.to_device(test_tensor)
        print(f"   Moved to CUDA: {cuda_tensor.device}")
        
        # Test operations
        result = torch.mm(cuda_tensor, cuda_tensor.t())
        print(f"   Matrix multiplication successful: {result.shape}")
        
        # Test mixed precision
        with torch.cuda.amp.autocast():
            mixed_result = torch.mm(cuda_tensor, cuda_tensor.t()) * 0.5
        print(f"   Mixed precision operations successful: {mixed_result.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA tensor operations failed: {e}")
        return False


def test_model_cuda_compatibility(device_manager):
    """Test model creation and training on CUDA"""
    if device_manager.device_name != "cuda":
        print(f"\n⏭️  Skipping CUDA model tests (using {device_manager.device_name})")
        return True
    
    print(f"\n🧪 Testing model CUDA compatibility...")
    
    try:
        # Create model
        model = PromoterCNN(num_blocks=2, base_channels=32, dropout=0.3)
        model = model.to(device_manager.device)
        print(f"   Model created and moved to CUDA")
        
        # Test forward pass
        batch_size = 4
        sequence_length = 100
        test_input = torch.randn(batch_size, 5, sequence_length, dtype=torch.float32)
        test_input = device_manager.to_device(test_input)
        
        # Regular forward pass
        with torch.no_grad():
            output = model(test_input)
        print(f"   Forward pass successful: {output.shape}")
        
        # Test with autocast (mixed precision)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                mixed_output = model(test_input)
        print(f"   Mixed precision forward pass successful: {mixed_output.shape}")
        
        # Test gradient computation
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        
        # Dummy targets
        targets = torch.randn(batch_size, 5).softmax(dim=1)
        targets = device_manager.to_device(targets)
        
        optimizer.zero_grad()
        output = model(test_input)
        output = torch.log_softmax(output, dim=1)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        print(f"   Training step successful: loss={loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model CUDA compatibility failed: {e}")
        return False


def main():
    """Run all CUDA compatibility tests"""
    print("🚀 CUDA COMPATIBILITY TEST SUITE")
    print("=" * 50)
    
    # Test CUDA detection
    cuda_available = test_cuda_detection()
    
    # Test device manager
    device_manager = test_device_manager()
    if not device_manager:
        print("❌ Cannot continue without DeviceManager")
        return
    
    # Run CUDA-specific tests
    tests = [
        ("CUDA Tensor Operations", lambda: test_cuda_tensor_operations(device_manager)),
        ("Model CUDA Compatibility", lambda: test_model_cuda_compatibility(device_manager)),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n📊 TEST RESULTS")
    print("=" * 30)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if device_manager.device_name == "cuda":
        if passed == total:
            print("🎉 All CUDA tests passed!")
            print("Your system is fully optimized for CUDA hyperparameter tuning.")
            
            # Show optimization features
            print(f"\n🚀 CUDA Optimizations Enabled:")
            print("- Non-blocking tensor transfers")
            print("- cuDNN benchmark mode")
            print("- Mixed precision training (autocast)")
            print("- Optimized DataLoader (pin_memory, multiple workers)")
            print("- Memory management and cache clearing")
            print("- Real-time memory monitoring")
            
        else:
            print("⚠️  Some CUDA tests failed.")
            print("The system will work but may not be fully optimized.")
    else:
        print(f"ℹ️  Using {device_manager.device_name} - CUDA tests skipped")
        print("The system will work but without CUDA optimizations.")


if __name__ == "__main__":
    main()
