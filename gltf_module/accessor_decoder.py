"""
GLTF Accessor Decoder

This module provides functionality for decoding GLTF accessor data from buffers,
equivalent to Godot's accessor decoding functionality.
"""

import struct
from typing import List, Optional, Any, Tuple
from .gltf_state import GLTFState
from .structures import *


class GLTFAccessorDecoder:
    """
    Decoder for GLTF accessor data.

    This class provides methods to decode various types of data from GLTF buffers
    based on accessor specifications.
    """

    # Component type constants
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

    # Component type sizes
    COMPONENT_TYPE_SIZES = {
        COMPONENT_TYPE_BYTE: 1,
        COMPONENT_TYPE_UNSIGNED_BYTE: 1,
        COMPONENT_TYPE_SHORT: 2,
        COMPONENT_TYPE_UNSIGNED_SHORT: 2,
        COMPONENT_TYPE_UNSIGNED_INT: 4,
        COMPONENT_TYPE_FLOAT: 4,
    }

    # Component type pack formats
    COMPONENT_TYPE_PACK_FORMATS = {
        COMPONENT_TYPE_BYTE: '<b',           # Signed byte
        COMPONENT_TYPE_UNSIGNED_BYTE: '<B',  # Unsigned byte
        COMPONENT_TYPE_SHORT: '<h',          # Signed short
        COMPONENT_TYPE_UNSIGNED_SHORT: '<H', # Unsigned short
        COMPONENT_TYPE_UNSIGNED_INT: '<I',   # Unsigned int
        COMPONENT_TYPE_FLOAT: '<f',          # Float
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

    @staticmethod
    def decode_accessor(state: GLTFState, accessor_index: int, for_vertex: bool = True) -> List[float]:
        """
        Decode accessor data as a flat list of floats.

        Args:
            state: GLTF state containing the accessor and buffer data
            accessor_index: Index of the accessor to decode
            for_vertex: Whether this is for vertex data (affects normalization)

        Returns:
            List of decoded float values
        """
        if accessor_index < 0 or accessor_index >= len(state.accessors):
            raise ValueError(f"Invalid accessor index: {accessor_index}")

        accessor = state.accessors[accessor_index]

        # Get buffer view
        if accessor.buffer_view is None:
            return []

        if accessor.buffer_view < 0 or accessor.buffer_view >= len(state.buffer_views):
            raise ValueError(f"Invalid buffer view index: {accessor.buffer_view}")

        buffer_view = state.buffer_views[accessor.buffer_view]

        # Get buffer
        if buffer_view.buffer < 0 or buffer_view.buffer >= len(state.buffers):
            raise ValueError(f"Invalid buffer index: {buffer_view.buffer}")

        buffer = state.buffers[buffer_view.buffer]

        # Check if buffer has data
        if buffer.data is None:
            raise ValueError(f"Buffer {buffer_view.buffer} has no data loaded")

        # Calculate data parameters
        component_size = GLTFAccessorDecoder.COMPONENT_TYPE_SIZES.get(accessor.component_type, 0)
        if component_size == 0:
            raise ValueError(f"Unsupported component type: {accessor.component_type}")

        component_count = GLTFAccessorDecoder.ACCESSOR_TYPE_COMPONENTS.get(accessor.type, 0)
        if component_count == 0:
            raise ValueError(f"Unsupported accessor type: {accessor.type}")

        element_size = component_count * component_size
        byte_stride = buffer_view.byte_stride if buffer_view.byte_stride else element_size

        # Calculate data offset
        data_offset = buffer_view.byte_offset + accessor.byte_offset

        # Check bounds for the entire accessor
        total_data_needed = data_offset + (accessor.count - 1) * byte_stride + element_size
        if total_data_needed > len(buffer.data):
            raise ValueError(f"Accessor data out of bounds: needs {total_data_needed} bytes, buffer has {len(buffer.data)}")

        # Decode the data
        result = []
        for i in range(accessor.count):
            element_offset = data_offset + i * byte_stride

            # Read component values
            component_values = []
            for j in range(component_count):
                comp_offset = element_offset + j * component_size

                # Unpack the component value
                pack_format = GLTFAccessorDecoder.COMPONENT_TYPE_PACK_FORMATS[accessor.component_type]
                value = struct.unpack(pack_format, buffer.data[comp_offset:comp_offset + component_size])[0]

                # Convert to float and apply normalization if needed
                if accessor.normalized:
                    if accessor.component_type == GLTFAccessorDecoder.COMPONENT_TYPE_BYTE:
                        value = max(-1.0, min(1.0, value / 127.0))
                    elif accessor.component_type == GLTFAccessorDecoder.COMPONENT_TYPE_UNSIGNED_BYTE:
                        value = value / 255.0
                    elif accessor.component_type == GLTFAccessorDecoder.COMPONENT_TYPE_SHORT:
                        value = max(-1.0, min(1.0, value / 32767.0))
                    elif accessor.component_type == GLTFAccessorDecoder.COMPONENT_TYPE_UNSIGNED_SHORT:
                        value = value / 65535.0

                component_values.append(float(value))

            result.extend(component_values)

        return result

    @staticmethod
    def decode_accessor_as_vec3(state: GLTFState, accessor_index: int, for_vertex: bool = True) -> List[List[float]]:
        """
        Decode accessor data as a list of 3D vectors.

        Args:
            state: GLTF state containing the accessor and buffer data
            accessor_index: Index of the accessor to decode
            for_vertex: Whether this is for vertex data

        Returns:
            List of 3D vectors [x, y, z]
        """
        flat_data = GLTFAccessorDecoder.decode_accessor(state, accessor_index, for_vertex)

        if not flat_data:
            return []

        # Group into vectors of 3
        vectors = []
        for i in range(0, len(flat_data), 3):
            if i + 2 < len(flat_data):
                vectors.append([flat_data[i], flat_data[i + 1], flat_data[i + 2]])

        return vectors

    @staticmethod
    def decode_accessor_as_vec2(state: GLTFState, accessor_index: int, for_vertex: bool = True) -> List[List[float]]:
        """
        Decode accessor data as a list of 2D vectors.

        Args:
            state: GLTF state containing the accessor and buffer data
            accessor_index: Index of the accessor to decode
            for_vertex: Whether this is for vertex data

        Returns:
            List of 2D vectors [x, y]
        """
        flat_data = GLTFAccessorDecoder.decode_accessor(state, accessor_index, for_vertex)

        if not flat_data:
            return []

        # Group into vectors of 2
        vectors = []
        for i in range(0, len(flat_data), 2):
            if i + 1 < len(flat_data):
                vectors.append([flat_data[i], flat_data[i + 1]])

        return vectors

    @staticmethod
    def decode_accessor_as_indices(state: GLTFState, accessor_index: int) -> List[int]:
        """
        Decode accessor data as a list of indices.

        Args:
            state: GLTF state containing the accessor and buffer data
            accessor_index: Index of the accessor to decode

        Returns:
            List of integer indices
        """
        if accessor_index < 0 or accessor_index >= len(state.accessors):
            raise ValueError(f"Invalid accessor index: {accessor_index}")

        accessor = state.accessors[accessor_index]

        # Get buffer view
        if accessor.buffer_view is None:
            return []

        if accessor.buffer_view < 0 or accessor.buffer_view >= len(state.buffer_views):
            raise ValueError(f"Invalid buffer view index: {accessor.buffer_view}")

        buffer_view = state.buffer_views[accessor.buffer_view]

        # Get buffer
        if buffer_view.buffer < 0 or buffer_view.buffer >= len(state.buffers):
            raise ValueError(f"Invalid buffer index: {buffer_view.buffer}")

        buffer = state.buffers[buffer_view.buffer]

        # Check if buffer has data
        if buffer.data is None:
            raise ValueError(f"Buffer {buffer_view.buffer} has no data loaded")

        # Calculate data parameters
        component_size = GLTFAccessorDecoder.COMPONENT_TYPE_SIZES.get(accessor.component_type, 0)
        if component_size == 0:
            raise ValueError(f"Unsupported component type: {accessor.component_type}")

        byte_stride = buffer_view.byte_stride if buffer_view.byte_stride else component_size

        # Calculate data offset
        data_offset = buffer_view.byte_offset + accessor.byte_offset

        # Check bounds for the entire accessor
        total_data_needed = data_offset + (accessor.count - 1) * byte_stride + component_size
        if total_data_needed > len(buffer.data):
            raise ValueError(f"Accessor data out of bounds: needs {total_data_needed} bytes, buffer has {len(buffer.data)}")

        # Decode indices
        indices = []
        for i in range(accessor.count):
            element_offset = data_offset + i * byte_stride

            # Unpack the index value
            pack_format = GLTFAccessorDecoder.COMPONENT_TYPE_PACK_FORMATS[accessor.component_type]
            index = struct.unpack(pack_format, buffer.data[element_offset:element_offset + component_size])[0]

            indices.append(int(index))

        return indices

    @staticmethod
    def decode_accessor_as_colors(state: GLTFState, accessor_index: int, for_vertex: bool = True) -> List[List[float]]:
        """
        Decode accessor data as a list of colors.

        Args:
            state: GLTF state containing the accessor and buffer data
            accessor_index: Index of the accessor to decode
            for_vertex: Whether this is for vertex data

        Returns:
            List of RGBA colors [r, g, b, a]
        """
        flat_data = GLTFAccessorDecoder.decode_accessor(state, accessor_index, for_vertex)

        if not flat_data:
            return []

        accessor = state.accessors[accessor_index]
        component_count = GLTFAccessorDecoder.ACCESSOR_TYPE_COMPONENTS[accessor.type]

        # Handle different color formats
        colors = []
        for i in range(0, len(flat_data), component_count):
            if component_count == 3:
                # RGB
                r, g, b = flat_data[i:i+3]
                colors.append([r, g, b, 1.0])
            elif component_count == 4:
                # RGBA
                colors.append(flat_data[i:i+4])
            else:
                # Invalid color format
                break

        return colors

    @staticmethod
    def get_accessor_bounds(state: GLTFState, accessor_index: int) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Get the min/max bounds of an accessor.

        Args:
            state: GLTF state containing the accessor
            accessor_index: Index of the accessor

        Returns:
            Tuple of (min_values, max_values) or (None, None) if not available
        """
        if accessor_index < 0 or accessor_index >= len(state.accessors):
            return None, None

        accessor = state.accessors[accessor_index]
        return accessor.min, accessor.max

    @staticmethod
    def validate_accessor_data(state: GLTFState, accessor_index: int) -> List[str]:
        """
        Validate accessor data for consistency and correctness.

        Args:
            state: GLTF state containing the accessor
            accessor_index: Index of the accessor to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if accessor_index < 0 or accessor_index >= len(state.accessors):
            errors.append(f"Invalid accessor index: {accessor_index}")
            return errors

        accessor = state.accessors[accessor_index]

        # Validate component type
        if accessor.component_type not in GLTFAccessorDecoder.COMPONENT_TYPE_SIZES:
            errors.append(f"Unsupported component type: {accessor.component_type}")

        # Validate accessor type
        if accessor.type not in GLTFAccessorDecoder.ACCESSOR_TYPE_COMPONENTS:
            errors.append(f"Unsupported accessor type: {accessor.type}")

        # Validate buffer view
        if accessor.buffer_view is not None:
            if accessor.buffer_view < 0 or accessor.buffer_view >= len(state.buffer_views):
                errors.append(f"Invalid buffer view index: {accessor.buffer_view}")
            else:
                buffer_view = state.buffer_views[accessor.buffer_view]

                # Validate buffer
                if buffer_view.buffer < 0 or buffer_view.buffer >= len(state.buffers):
                    errors.append(f"Invalid buffer index: {buffer_view.buffer}")
                else:
                    buffer = state.buffers[buffer_view.buffer]

                    # Check if buffer has data
                    if not buffer.data and not buffer.uri:
                        errors.append("Buffer has no data or URI")

        # Validate count
        if accessor.count <= 0:
            errors.append("Accessor count must be positive")

        return errors
