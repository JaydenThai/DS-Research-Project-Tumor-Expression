"""
Shared utilities for model training and data handling.

Modules:
- data: dataset and sequence encoding utilities for promoter sequences
- training: epoch loops and evaluation helpers
- viz: plotting helpers for training curves and metrics
"""

__all__ = [
    "data",
    "training",
    "viz",
]

# Convenience import for device helpers
try:
    from .device import get_best_device, DeviceManager, create_device_manager  # noqa: F401
except Exception:
    # Optional
    pass


