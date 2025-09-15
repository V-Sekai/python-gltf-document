"""
Godot GLTF Module - Python Implementation

A Python module that replicates the functionality of Godot's GLTF module,
providing GLTF 2.0 file parsing and scene generation capabilities.

Supports both CPU and GPU-accelerated operations via PyTorch integration.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Core functionality
from .gltf_document import GLTFDocument
from .gltf_state import GLTFState
from .structures import *
from .accessor_decoder import GLTFAccessorDecoder
from .scene_generator import GLTFSceneGenerator
from .skin_tool import GLTFSkinTool
from .gltf_exporter import GLTFExporter
from .logger import logger, get_logger, set_log_level

# PyTorch-compatible versions
try:
    import torch
    _torch_available = True
    from .torch_structures import *
    from .torch_decoder import TorchGLTFAccessorDecoder
except ImportError:
    _torch_available = False
    # Define dummy classes for when PyTorch is not available
    class TorchGLTFAccessorDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GPU-accelerated decoding")

__all__ = [
    'GLTFDocument',
    'GLTFState',
    'GLTFSkinTool',
    # Include PyTorch classes if available
    'TorchGLTFAccessorDecoder' if _torch_available else None,
]

# Filter out None values
__all__ = [item for item in __all__ if item is not None]

def is_torch_available() -> bool:
    """Check if PyTorch is available for GPU acceleration"""
    return _torch_available

def get_torch_device(device: str = 'auto') -> 'torch.device':
    """Get PyTorch device for computations"""
    if not _torch_available:
        raise ImportError("PyTorch is not available")

    import torch
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device('cuda')
    elif device == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device(device)
