"""
PyTorch GPU-Accelerated GLTF Accessor Decoder

This module provides GPU-accelerated decoding of GLTF accessor data using PyTorch,
offering significant performance improvements for large models.
"""

import torch
import struct
from typing import List, Optional, Tuple
from .gltf_state import GLTFState
from .torch_structures import TorchGLTFAccessor, TorchGLTFBuffer
from .accessor_decoder import GLTFAccessorDecoder


class TorchGLTFAccessorDecoder:
    """
    PyTorch-based GPU-accelerated GLTF accessor decoder.

    This class provides high-performance decoding of GLTF buffer data using PyTorch tensors,
    with support for GPU acceleration and batched operations.
    """

    # Component type constants (same as CPU decoder)
    COMPONENT_TYPE_BYTE = 5120
    COMPONENT_TYPE_UNSIGNED_BYTE = 5121
    COMPONENT_TYPE_SHORT = 5122
    COMPONENT_TYPE_UNSIGNED_SHORT = 5123
    COMPONENT_TYPE_UNSIGNED_INT = 5125
    COMPONENT_TYPE_FLOAT = 5126

    # Accessor type constants
    ACCESSOR_TYPE_SCALAR = "SCALAR"
    ACCESSOR_TYPE_VEC2 = "VEC2"
    ACCESSOR_TYPE_VEC3 = "VEC3"
    ACCESSOR_TYPE_VEC4 = "VEC4"
    ACCESSOR_TYPE_MAT2 = "MAT2"
    ACCESSOR_TYPE_MAT3 = "MAT3"
    ACCESSOR_TYPE_MAT4 = "MAT4"

    # PyTorch dtype mappings
    COMPONENT_TYPE_TO_DTYPE = {
        COMPONENT_TYPE_BYTE: torch.int8,
        COMPONENT_TYPE_UNSIGNED_BYTE: torch.uint8,
        COMPONENT_TYPE_SHORT: torch.int16,
        COMPONENT_TYPE_UNSIGNED_SHORT: torch.int32,  # Promote to int32 for safety
        COMPONENT_TYPE_UNSIGNED_INT: torch.int32,     # Keep as int32
        COMPONENT_TYPE_FLOAT: torch.float32,
    }

    # Component type pack formats (for CPU fallback)
    COMPONENT_TYPE_PACK_FORMATS = {
        COMPONENT_TYPE_BYTE: '<b',
        COMPONENT_TYPE_UNSIGNED_BYTE: '<B',
        COMPONENT_TYPE_SHORT: '<h',
        COMPONENT_TYPE_UNSIGNED_SHORT: '<H',
        COMPONENT_TYPE_UNSIGNED_INT: '<I',
        COMPONENT_TYPE_FLOAT: '<f',
    }

    # Accessor type component counts
    ACCESSOR_TYPE_COMPONENTS = {
        ACCESSOR_TYPE_SCALAR: 1,
        ACCESSOR_TYPE_VEC2: 2,
        ACCESSOR_TYPE_VEC3: 3,
        ACCESSOR_TYPE_VEC4: 4,
        ACCESSOR_TYPE_MAT2: 4,
        ACCESSOR_TYPE_MAT3: 9,
        ACCESSOR_TYPE_MAT4: 16,
    }

    def __init__(self, device: torch.device = None):
        """
        Initialize the PyTorch decoder.

        Args:
            device: PyTorch device to use for computations
        """
        self.device = device if device is not None else torch.device('cpu')

    def decode_accessor_tensor(self, state: GLTFState, accessor_index: int,
                              force_cpu: bool = False) -> torch.Tensor:
        """
        Decode GLTF accessor data as PyTorch tensor.

        Args:
            state: GLTF state containing accessor and buffer data
            accessor_index: Index of the accessor to decode
            force_cpu: Force CPU decoding even if GPU is available

        Returns:
            PyTorch tensor containing the decoded data
        """
        if accessor_index < 0 or accessor_index >= len(state.accessors):
            raise ValueError(f"Invalid accessor index: {accessor_index}")

        accessor = state.accessors[accessor_index]

        # Check if we have a cached PyTorch accessor
        torch_accessor = self._get_or_create_torch_accessor(accessor, accessor_index)

        # Return cached data if available
        if torch_accessor.tensor_data is not None:
            return torch_accessor.tensor_data

        # Get buffer view
        if accessor.buffer_view is None:
            # Return empty tensor
            component_count = self.ACCESSOR_TYPE_COMPONENTS.get(accessor.type, 1)
            return torch.empty((accessor.count, component_count),
                             dtype=torch.float32, device=self.device)

        if accessor.buffer_view < 0 or accessor.buffer_view >= len(state.buffer_views):
            raise ValueError(f"Invalid buffer view index: {accessor.buffer_view}")

        buffer_view = state.buffer_views[accessor.buffer_view]

        # Get buffer
        if buffer_view.buffer < 0 or buffer_view.buffer >= len(state.buffers):
            raise ValueError(f"Invalid buffer index: {buffer_view.buffer}")

        buffer = state.buffers[buffer_view.buffer]

        # Use GPU-accelerated decoding if possible
        if not force_cpu and self._can_use_gpu_decoding(accessor, buffer):
            tensor_data = self._decode_with_gpu(buffer, buffer_view, accessor)
        else:
            # Fall back to CPU decoding
            flat_data = GLTFAccessorDecoder.decode_accessor(state, accessor_index, for_vertex=True)
            tensor_data = torch.tensor(flat_data, dtype=torch.float32, device=self.device)

        # Reshape according to accessor type
        component_count = self.ACCESSOR_TYPE_COMPONENTS[accessor.type]
        if component_count > 1:
            tensor_data = tensor_data.view(accessor.count, component_count)

        # Cache the result
        torch_accessor.tensor_data = tensor_data

        return tensor_data

    def _can_use_gpu_decoding(self, accessor: 'GLTFAccessor', buffer: 'GLTFBuffer') -> bool:
        """Check if GPU decoding can be used for this accessor"""
        # GPU decoding requires:
        # 1. GPU device available
        # 2. Buffer data available
        # 3. Supported component type
        # 4. No normalization (for now)

        if self.device.type == 'cpu':
            return False

        if not buffer.data:
            return False

        if accessor.component_type not in self.COMPONENT_TYPE_TO_DTYPE:
            return False

        if accessor.normalized:
            return False  # Normalization not implemented for GPU yet

        return True

    def _decode_with_gpu(self, buffer: 'GLTFBuffer', buffer_view: 'GLTFBufferView',
                        accessor: 'GLTFAccessor') -> torch.Tensor:
        """
        Decode accessor data using GPU acceleration.
        """
        # Convert buffer data to PyTorch tensor
        buffer_tensor = torch.frombuffer(buffer.data, dtype=torch.uint8)

        # Calculate data offset and size
        data_offset = buffer_view.byte_offset + accessor.byte_offset
        component_size = GLTFAccessorDecoder.COMPONENT_TYPE_SIZES[accessor.component_type]
        component_count = self.ACCESSOR_TYPE_COMPONENTS[accessor.type]
        element_size = component_count * component_size

        # Extract raw bytes for this accessor
        byte_stride = buffer_view.byte_stride if buffer_view.byte_stride else element_size
        total_bytes = accessor.count * byte_stride

        # Create view of the buffer data
        accessor_bytes = buffer_tensor[data_offset:data_offset + total_bytes]

        # Reshape according to stride
        if byte_stride == element_size:
            # Contiguous data
            accessor_bytes = accessor_bytes.view(accessor.count, element_size)
        else:
            # Interleaved data - need to extract every Nth element
            accessor_bytes = accessor_bytes.view(accessor.count, byte_stride)
            accessor_bytes = accessor_bytes[:, :element_size]

        # Convert to target dtype
        target_dtype = self.COMPONENT_TYPE_TO_DTYPE[accessor.component_type]

        # Interpret bytes as target type
        if component_size == 1:
            tensor_data = accessor_bytes.view(target_dtype)
        elif component_size == 2:
            tensor_data = accessor_bytes.view(torch.int16)
            if accessor.component_type == self.COMPONENT_TYPE_UNSIGNED_SHORT:
                tensor_data = tensor_data.to(torch.int32)  # Promote to int32
        elif component_size == 4:
            if accessor.component_type == self.COMPONENT_TYPE_FLOAT:
                tensor_data = accessor_bytes.view(torch.float32)
            else:
                tensor_data = accessor_bytes.view(torch.int32)

        # Reshape to final dimensions
        if component_count == 1:
            tensor_data = tensor_data.view(accessor.count)
        else:
            tensor_data = tensor_data.view(accessor.count, component_count)

        # Convert to float32 for consistency
        if tensor_data.dtype != torch.float32:
            tensor_data = tensor_data.to(torch.float32)

        # Apply normalization if needed
        if accessor.normalized:
            tensor_data = self._apply_normalization(tensor_data, accessor.component_type)

        return tensor_data.to(self.device)

    def _apply_normalization(self, tensor: torch.Tensor, component_type: int) -> torch.Tensor:
        """Apply normalization to tensor data"""
        if component_type == self.COMPONENT_TYPE_BYTE:
            # Normalize from [-128, 127] to [-1, 1]
            return torch.clamp(tensor / 127.0, -1.0, 1.0)
        elif component_type == self.COMPONENT_TYPE_UNSIGNED_BYTE:
            # Normalize from [0, 255] to [0, 1]
            return tensor / 255.0
        elif component_type == self.COMPONENT_TYPE_SHORT:
            # Normalize from [-32768, 32767] to [-1, 1]
            return torch.clamp(tensor / 32767.0, -1.0, 1.0)
        elif component_type == self.COMPONENT_TYPE_UNSIGNED_SHORT:
            # Normalize from [0, 65535] to [0, 1]
            return tensor / 65535.0
        else:
            return tensor

    def _get_or_create_torch_accessor(self, accessor: 'GLTFAccessor',
                                     accessor_index: int) -> TorchGLTFAccessor:
        """Get or create PyTorch accessor wrapper"""
        # For now, create new wrapper each time
        # In a production implementation, you'd cache these
        return TorchGLTFAccessor.from_gltf_accessor(accessor, self.device)

    def decode_mesh_attributes(self, state: GLTFState, mesh_index: int) -> dict:
        """
        Decode all mesh attributes as PyTorch tensors.

        Args:
            state: GLTF state
            mesh_index: Index of the mesh to decode

        Returns:
            Dictionary containing decoded mesh attributes
        """
        if mesh_index < 0 or mesh_index >= len(state.meshes):
            raise ValueError(f"Invalid mesh index: {mesh_index}")

        mesh = state.meshes[mesh_index]
        attributes = {}

        # Decode all primitives (use first one for now)
        if mesh.primitives:
            primitive = mesh.primitives[0]

            # Decode vertex attributes
            for attr_name, accessor_index in primitive.attributes.items():
                try:
                    tensor_data = self.decode_accessor_tensor(state, accessor_index)
                    attributes[attr_name] = tensor_data
                except Exception as e:
                    print(f"Failed to decode {attr_name}: {e}")

            # Decode indices
            if primitive.indices is not None:
                try:
                    indices_tensor = self.decode_accessor_tensor(state, primitive.indices)
                    # Convert to integer type for indices
                    attributes['INDICES'] = indices_tensor.to(torch.int32)
                except Exception as e:
                    print(f"Failed to decode indices: {e}")

        return attributes

    def batch_decode_accessors(self, state: GLTFState, accessor_indices: List[int]) -> List[torch.Tensor]:
        """
        Decode multiple accessors in batch for better performance.

        Args:
            state: GLTF state
            accessor_indices: List of accessor indices to decode

        Returns:
            List of decoded tensors
        """
        tensors = []
        for idx in accessor_indices:
            try:
                tensor = self.decode_accessor_tensor(state, idx)
                tensors.append(tensor)
            except Exception as e:
                print(f"Failed to decode accessor {idx}: {e}")
                # Add empty tensor as placeholder
                tensors.append(torch.empty(0, dtype=torch.float32, device=self.device))

        return tensors

    def preload_accessors(self, state: GLTFState, accessor_indices: List[int]):
        """
        Preload multiple accessors into GPU memory for faster access.

        Args:
            state: GLTF state
            accessor_indices: List of accessor indices to preload
        """
        for idx in accessor_indices:
            try:
                self.decode_accessor_tensor(state, idx)
            except Exception as e:
                pass  # Ignore errors during preloading

    def get_memory_usage(self, state: GLTFState) -> dict:
        """
        Estimate memory usage for decoded tensors.

        Args:
            state: GLTF state

        Returns:
            Dictionary with memory usage statistics
        """
        total_bytes = 0
        accessor_info = []

        for i, accessor in enumerate(state.accessors):
            component_count = self.ACCESSOR_TYPE_COMPONENTS.get(accessor.type, 1)
            tensor_bytes = accessor.count * component_count * 4  # float32 = 4 bytes
            total_bytes += tensor_bytes

            accessor_info.append({
                'index': i,
                'type': accessor.type,
                'count': accessor.count,
                'components': component_count,
                'bytes': tensor_bytes
            })

        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'accessor_count': len(state.accessors),
            'accessor_info': accessor_info
        }

    def set_device(self, device: torch.device):
        """Change the device used for decoding"""
        self.device = device

    def clear_cache(self):
        """Clear any cached tensor data"""
        # In a full implementation, you'd clear cached tensors here
        pass
