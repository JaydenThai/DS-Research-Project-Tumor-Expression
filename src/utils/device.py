import torch
import gc
from typing import Tuple, Dict, Any, Union, Optional
import logging
import warnings

# Set up logging
logger = logging.getLogger(__name__)


class DeviceManager:
    """
    A comprehensive device manager for CUDA and MPS compatibility.
    
    Provides utilities for:
    - Device detection and selection
    - Memory management
    - Tensor movement with compatibility checks
    - Device-specific optimizations
    """
    
    def __init__(self, prefer_cuda: bool = True, verbose: bool = False):
        self.prefer_cuda = prefer_cuda
        self.verbose = verbose
        self.device, self.loader_kwargs, self.device_name = self._select_device()
        self._setup_device_optimizations()
        
        if self.verbose:
            logger.info(f"Selected device: {self.device_name} ({self.device})")
    
    def _select_device(self) -> Tuple[torch.device, Dict[str, Any], str]:
        """Select the best available device and return DataLoader kwargs."""
        
        # CUDA preference
        if self.prefer_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            if self.verbose:
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return device, {"pin_memory": True, "num_workers": 4}, "cuda"
        
        # MPS (Apple Silicon) preference
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            if self.verbose:
                logger.info("MPS (Apple Silicon) acceleration available")
            return device, {"num_workers": 0}, "mps"  # MPS doesn't support pin_memory
        
        # CPU fallback
        if self.verbose:
            logger.info("Using CPU (no GPU acceleration available)")
        return torch.device("cpu"), {"num_workers": 2}, "cpu"
    
    def _setup_device_optimizations(self):
        """Apply device-specific optimizations."""
        if self.device_name == "cuda":
            try:
                # Enable cuDNN benchmark for consistent input sizes
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                if self.verbose:
                    logger.info("Enabled cuDNN optimizations")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not enable cuDNN optimizations: {e}")
        
        elif self.device_name == "mps":
            # MPS-specific settings if needed
            if self.verbose:
                logger.info("MPS device configured")
    
    def to_device(self, tensor_or_batch: Union[torch.Tensor, Dict, Any]) -> Union[torch.Tensor, Dict, Any]:
        """
        Move tensor(s) to the selected device with automatic compatibility handling.
        
        This method provides a unified interface regardless of device type (CUDA/MPS/CPU).
        All device-specific optimizations are handled internally.
        
        Args:
            tensor_or_batch: Tensor, dict of tensors, or other data structure
        
        Returns:
            Data moved to the appropriate device
        """
        # Handle dictionaries (common for batches)
        if isinstance(tensor_or_batch, dict):
            return {key: self.to_device(value) for key, value in tensor_or_batch.items()}
        
        # Handle lists/tuples
        elif isinstance(tensor_or_batch, (list, tuple)):
            container_type = type(tensor_or_batch)
            return container_type(self.to_device(item) for item in tensor_or_batch)
        
        # Handle tensors
        elif isinstance(tensor_or_batch, torch.Tensor):
            try:
                # Use device-specific optimized transfer
                return self._transfer_tensor_optimized(tensor_or_batch)
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to move tensor to {self.device_name}, falling back to CPU: {e}")
                return tensor_or_batch.to("cpu")
        
        # Return unchanged for non-tensor types
        else:
            return tensor_or_batch
    
    def _transfer_tensor_optimized(self, tensor: torch.Tensor) -> torch.Tensor:
        """Internal method for device-optimized tensor transfer."""
        if self.device_name == "cuda":
            # CUDA supports non-blocking transfers
            return tensor.to(self.device, non_blocking=True)
        elif self.device_name == "mps":
            # MPS doesn't support non_blocking, use standard transfer
            return tensor.to(self.device)
        else:
            # CPU transfer
            return tensor.to(self.device)
    
    def empty_cache(self):
        """Clear device cache to free up memory."""
        if self.device_name == "cuda":
            torch.cuda.empty_cache()
            if self.verbose:
                logger.info("Cleared CUDA cache")
        elif self.device_name == "mps":
            # MPS doesn't have explicit cache clearing, but we can trigger garbage collection
            gc.collect()
            if self.verbose:
                logger.info("Triggered garbage collection for MPS")
        else:
            gc.collect()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get device memory information in GB."""
        if self.device_name == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated": allocated,
                "cached": cached,
                "total": total,
                "free": total - cached
            }
        else:
            # For MPS and CPU, we can't easily get memory info
            return {"allocated": 0.0, "cached": 0.0, "total": 0.0, "free": 0.0}
    
    def check_tensor_device(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is on the correct device."""
        return tensor.device == self.device
    
    def get_dataloader_kwargs(self, **additional_kwargs) -> Dict[str, Any]:
        """Get DataLoader kwargs optimized for the current device."""
        kwargs = self.loader_kwargs.copy()
        kwargs.update(additional_kwargs)
        return kwargs
    
    def create_model_wrapper(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to device and apply device-specific optimizations."""
        model = model.to(self.device)
        
        if self.device_name == "cuda":
            # Enable automatic mixed precision if supported
            try:
                model = torch.jit.optimize_for_inference(model)
                if self.verbose:
                    logger.info("Applied CUDA optimizations to model")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not apply CUDA optimizations: {e}")
        
        return model
    
    def __str__(self) -> str:
        return f"DeviceManager(device={self.device_name}, device_obj={self.device})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Backward compatibility functions
def get_best_device(prefer_cuda: bool = True) -> Tuple[torch.device, Dict[str, Any], str]:
    """
    Legacy function for backward compatibility.
    
    Returns: (device, loader_kwargs, device_name)
    """
    manager = DeviceManager(prefer_cuda=prefer_cuda, verbose=False)
    return manager.device, manager.loader_kwargs, manager.device_name


def create_device_manager(prefer_cuda: bool = True, verbose: bool = False) -> DeviceManager:
    """
    Create a new DeviceManager instance.
    
    Args:
        prefer_cuda: Whether to prefer CUDA over MPS
        verbose: Whether to print device information
    
    Returns:
        DeviceManager instance
    """
    return DeviceManager(prefer_cuda=prefer_cuda, verbose=verbose)


