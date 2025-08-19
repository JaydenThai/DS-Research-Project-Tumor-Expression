#!/usr/bin/env python3
"""
GPU Utilization Optimizer for Hyperparameter Tuning

This script helps you find the optimal settings for maximum GPU utilization
during hyperparameter tuning.
"""

import sys
from pathlib import Path
import torch

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from hyperparameter_tuning import DeviceManager, PromoterCNN, get_optimized_search_space


def analyze_gpu_capabilities():
    """Analyze GPU capabilities and provide optimization recommendations"""
    print("🔍 GPU UTILIZATION ANALYSIS")
    print("=" * 50)
    
    device_manager = DeviceManager(verbose=True)
    
    if device_manager.device_name != "cuda":
        print(f"❌ Not using CUDA - GPU utilization optimization not applicable")
        print(f"   Current device: {device_manager.device_name}")
        return None
    
    # Get memory info
    memory_info = device_manager.get_memory_info()
    total_memory = memory_info['total_gb']
    
    print(f"\n📊 GPU Analysis:")
    print(f"   Total Memory: {total_memory:.1f}GB")
    print(f"   Current Usage: {memory_info['utilization']:.1f}%")
    print(f"   Available: {memory_info['free_gb']:.1f}GB")
    
    # Get compute capability
    capability = torch.cuda.get_device_capability()
    print(f"   Compute Capability: {capability[0]}.{capability[1]}")
    
    # Check for Tensor Cores (compute capability >= 7.0)
    has_tensor_cores = capability[0] >= 7
    print(f"   Tensor Cores: {'✅ Available' if has_tensor_cores else '❌ Not Available'}")
    
    return device_manager, memory_info, has_tensor_cores


def test_optimal_batch_sizes(device_manager):
    """Test different batch sizes to find optimal GPU utilization"""
    print(f"\n🧪 BATCH SIZE OPTIMIZATION")
    print("=" * 40)
    
    # Create test model
    test_model = PromoterCNN(num_blocks=3, base_channels=32, dropout=0.3)
    
    # Find optimal batch size
    sample_shape = (1, 5, 600)  # Typical input shape
    optimal_batch_size = device_manager.find_optimal_batch_size(test_model, sample_shape)
    
    return optimal_batch_size


def recommend_hyperparameter_settings(device_manager, optimal_batch_size, has_tensor_cores):
    """Provide recommendations for hyperparameter tuning settings"""
    print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 45)
    
    memory_info = device_manager.get_memory_info()
    total_memory = memory_info['total_gb']
    
    # Get optimized search space
    search_space = get_optimized_search_space(device_manager)
    
    print(f"📋 Recommended Settings:")
    print(f"   Optimal Batch Size: {optimal_batch_size}")
    print(f"   Batch Size Range: {search_space['batch_size']}")
    print(f"   Gradient Accumulation: {search_space['gradient_accumulation_steps']}")
    
    # Mixed precision recommendations
    if has_tensor_cores:
        print(f"   Mixed Precision: ✅ Enabled (Tensor Cores available)")
        print(f"   Expected Speedup: 1.5-2x faster training")
    else:
        print(f"   Mixed Precision: ⚠️  Enabled but limited benefit (no Tensor Cores)")
    
    # DataLoader settings
    dataloader_kwargs = device_manager.get_dataloader_kwargs()
    print(f"   Data Workers: {dataloader_kwargs['num_workers']}")
    print(f"   Prefetch Factor: {dataloader_kwargs['prefetch_factor']}")
    print(f"   Pin Memory: {dataloader_kwargs['pin_memory']}")
    
    # Memory-based recommendations
    if total_memory >= 16.0:
        print(f"\n🎯 High-End GPU Optimizations:")
        print(f"   - Use largest batch sizes (512-1024)")
        print(f"   - Enable gradient accumulation for effective batch sizes up to 2048")
        print(f"   - Consider running multiple trials in parallel")
    elif total_memory >= 8.0:
        print(f"\n🎯 Mid-Range GPU Optimizations:")
        print(f"   - Use medium-large batch sizes (128-512)")
        print(f"   - Enable gradient accumulation for effective batch sizes up to 1024")
        print(f"   - Monitor memory usage during training")
    else:
        print(f"\n🎯 Limited Memory GPU Optimizations:")
        print(f"   - Use smaller batch sizes (64-256)")
        print(f"   - Enable gradient accumulation to maintain training stability")
        print(f"   - Consider gradient checkpointing if needed")
    
    return {
        'optimal_batch_size': optimal_batch_size,
        'search_space': search_space,
        'has_tensor_cores': has_tensor_cores,
        'dataloader_kwargs': dataloader_kwargs
    }


def generate_optimized_command(recommendations):
    """Generate optimized command for hyperparameter tuning"""
    print(f"\n💻 OPTIMIZED COMMAND")
    print("=" * 30)
    
    # Basic command
    cmd = "python hyperparameter_tuning.py"
    
    # Add trials based on GPU capability
    memory_gb = recommendations['search_space']['batch_size'][-1] / 64 * 4  # Rough estimate
    if memory_gb >= 16:
        trials = 100
    elif memory_gb >= 8:
        trials = 75
    else:
        trials = 50
    
    cmd += f" --trials {trials}"
    cmd += " --plot"
    
    print(f"Recommended command:")
    print(f"   {cmd}")
    
    print(f"\nThis command will:")
    print(f"   - Run {trials} trials optimized for your GPU")
    print(f"   - Use batch sizes: {recommendations['search_space']['batch_size']}")
    print(f"   - Enable mixed precision training")
    print(f"   - Use {recommendations['dataloader_kwargs']['num_workers']} data workers")
    print(f"   - Generate analysis plots")


def monitor_gpu_during_training():
    """Provide tips for monitoring GPU utilization during training"""
    print(f"\n📊 MONITORING GPU UTILIZATION")
    print("=" * 40)
    
    print(f"To monitor GPU utilization during training:")
    print(f"   1. Open a new terminal")
    print(f"   2. Run: nvidia-smi -l 1")
    print(f"   3. Look for GPU utilization percentage")
    print(f"   4. Aim for 80-95% GPU utilization")
    
    print(f"\nSigns of good GPU utilization:")
    print(f"   ✅ GPU utilization: 80-95%")
    print(f"   ✅ Memory usage: 70-90%")
    print(f"   ✅ Consistent utilization (not spiky)")
    
    print(f"\nSigns of poor GPU utilization:")
    print(f"   ❌ GPU utilization: <50%")
    print(f"   ❌ Memory usage: <50%")
    print(f"   ❌ Highly variable utilization")
    
    print(f"\nIf utilization is low, try:")
    print(f"   - Increasing batch size")
    print(f"   - Increasing gradient accumulation steps")
    print(f"   - Increasing number of data workers")
    print(f"   - Using larger models (more base_channels)")


def main():
    """Main function to analyze and optimize GPU utilization"""
    print("🚀 GPU UTILIZATION OPTIMIZER")
    print("=" * 60)
    
    # Analyze GPU capabilities
    result = analyze_gpu_capabilities()
    if result is None:
        return
    
    device_manager, memory_info, has_tensor_cores = result
    
    # Test optimal batch sizes
    optimal_batch_size = test_optimal_batch_sizes(device_manager)
    
    # Get recommendations
    recommendations = recommend_hyperparameter_settings(device_manager, optimal_batch_size, has_tensor_cores)
    
    # Generate optimized command
    generate_optimized_command(recommendations)
    
    # Monitoring tips
    monitor_gpu_during_training()
    
    print(f"\n🎉 GPU optimization analysis complete!")
    print(f"Your hyperparameter tuning should now achieve much higher GPU utilization.")


if __name__ == "__main__":
    main()
