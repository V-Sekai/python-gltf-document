#!/usr/bin/env python3
"""
GLB Parsing Tests using Slash Testing Framework

Tests for GLB (binary) GLTF file parsing functionality.
"""

import os
import struct
import tempfile
import slash
from pathlib import Path

# Import GLTF module
from gltf_module import GLTFDocument, GLTFState
from gltf_module.logger import get_logger, set_log_level

# Configure logging for tests
set_log_level(logging.WARNING)


@slash.fixture
def gltf_document():
    """Fixture providing a fresh GLTFDocument instance"""
    return GLTFDocument()


@slash.fixture
def gltf_state():
    """Fixture providing a fresh GLTFState instance"""
    return GLTFState()


@slash.fixture
def temp_dir():
    """Fixture providing a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def create_minimal_glb_json():
    """Create minimal valid GLTF JSON for testing"""
    return {
        "asset": {
            "version": "2.0",
            "generator": "GLTF Test Suite"
        },
        "scene": 0,
        "scenes": [
            {
                "name": "Test Scene",
                "nodes": [0]
            }
        ],
        "nodes": [
            {
                "name": "Test Node",
                "mesh": 0
            }
        ],
        "meshes": [
            {
                "name": "Test Mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0
                        },
                        "indices": 1,
                        "material": 0
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "Test Material",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0
                }
            }
        ],
        "buffers": [
            {
                "byteLength": 48,
                "uri": None  # Will be embedded in GLB
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": 24,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": 24,
                "byteLength": 6,
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": 2,
                "type": "VEC3",
                "max": [1.0, 1.0, 1.0],
                "min": [-1.0, -1.0, -1.0]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": 3,
                "type": "SCALAR",
                "max": [1],
                "min": [0]
            }
        ]
    }


def create_minimal_glb_buffer():
    """Create minimal binary buffer data for GLTF"""
    # Two vertices: (-1,-1,-1) and (1,1,1)
    vertices = struct.pack('<6f', -1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    # Three indices: 0, 1, 0 (triangle)
    indices = struct.pack('<3H', 0, 1, 0)
    return vertices + indices


def create_glb_file(json_data, bin_data=None, output_path=None):
    """Create a GLB file from JSON and binary data"""
    import json as json_module

    json_bytes = json_module.dumps(json_data, separators=(',', ':')).encode('utf-8')

    # Pad JSON to 4-byte alignment
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b'\x20' * json_padding

    # Create chunks
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes  # "JSON"

    chunks = json_chunk

    if bin_data:
        # Pad binary data to 4-byte alignment
        bin_padding = (4 - (len(bin_data) % 4)) % 4
        bin_data += b'\x00' * bin_padding

        bin_chunk = struct.pack('<II', len(bin_data), 0x004E4942) + bin_data  # "BIN\x00"
        chunks += bin_chunk

    # Create GLB header
    total_length = 12 + len(chunks)  # 12 bytes header + chunks
    header = struct.pack('<III', 0x46546C67, 2, total_length)  # magic, version, length

    glb_data = header + chunks

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(glb_data)

    return glb_data


def test_glb_magic_detection(gltf_document):
    """Test GLB magic number detection"""
    # Valid GLB header
    valid_glb = struct.pack('<III', 0x46546C67, 2, 100) + b'dummy data'
    slash.assert_true(gltf_document._is_glb_file(valid_glb))

    # Invalid magic
    invalid_magic = struct.pack('<III', 0x12345678, 2, 100) + b'dummy data'
    slash.assert_false(gltf_document._is_glb_file(invalid_magic))

    # Too short
    short_data = b'short'
    slash.assert_false(gltf_document._is_glb_file(short_data))


def test_glb_version_validation(gltf_document, temp_dir):
    """Test GLB version validation"""
    # Create GLB with wrong version
    json_data = create_minimal_glb_json()
    bin_data = create_minimal_glb_buffer()

    # Wrong version (3 instead of 2)
    header = struct.pack('<III', 0x46546C67, 3, 0)  # Wrong version
    json_bytes = json.dumps(json_data).encode('utf-8')
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    glb_data = header + json_chunk

    glb_path = temp_dir / "wrong_version.glb"
    with open(glb_path, 'wb') as f:
        f.write(glb_data)

    result = gltf_document.append_from_file(str(glb_path), GLTFState())
    slash.assert_equal(result, 1)  # Should fail due to wrong version


def test_minimal_glb_parsing(gltf_document, gltf_state, temp_dir):
    """Test parsing a minimal valid GLB file"""
    # Create minimal GLB
    json_data = create_minimal_glb_json()
    bin_data = create_minimal_glb_buffer()

    glb_path = temp_dir / "minimal.glb"
    create_glb_file(json_data, bin_data, glb_path)

    # Parse GLB
    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed

    # Verify parsed data
    slash.assert_is_not_none(gltf_state.json)
    slash.assert_equal(gltf_state.json['asset']['version'], '2.0')
    slash.assert_equal(len(gltf_state.nodes), 1)
    slash.assert_equal(len(gltf_state.meshes), 1)
    slash.assert_equal(len(gltf_state.materials), 1)
    slash.assert_equal(len(gltf_state.buffers), 1)
    slash.assert_equal(len(gltf_state.buffer_views), 2)
    slash.assert_equal(len(gltf_state.accessors), 2)


def test_glb_without_binary_chunk(gltf_document, gltf_state, temp_dir):
    """Test GLB file without binary chunk"""
    json_data = create_minimal_glb_json()
    # Remove buffer reference since no binary data
    json_data['buffers'] = []
    json_data['bufferViews'] = []
    json_data['accessors'] = []

    glb_path = temp_dir / "no_bin.glb"
    create_glb_file(json_data, None, glb_path)

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed


def test_glb_corrupted_header(gltf_document, gltf_state, temp_dir):
    """Test GLB with corrupted header"""
    glb_path = temp_dir / "corrupted.glb"

    # Create file with incomplete header
    with open(glb_path, 'wb') as f:
        f.write(b'incomplete header')

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 1)  # Should fail


def test_glb_invalid_json_chunk(gltf_document, gltf_state, temp_dir):
    """Test GLB with invalid JSON in chunk"""
    glb_path = temp_dir / "invalid_json.glb"

    # Create GLB with invalid JSON
    header = struct.pack('<III', 0x46546C67, 2, 0)
    invalid_json = b'invalid json content'
    json_chunk = struct.pack('<II', len(invalid_json), 0x4E4F534A) + invalid_json

    with open(glb_path, 'wb') as f:
        f.write(header + json_chunk)

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 1)  # Should fail due to invalid JSON


def test_glb_chunk_alignment(gltf_document, gltf_state, temp_dir):
    """Test GLB chunk alignment handling"""
    json_data = create_minimal_glb_json()
    bin_data = create_minimal_glb_buffer()

    # Manually create GLB with unaligned chunks to test padding
    json_bytes = json.dumps(json_data).encode('utf-8')
    # Add padding to test alignment
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b'\x20' * json_padding

    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes

    # Add padding to binary data
    bin_padding = (4 - (len(bin_data) % 4)) % 4
    bin_data += b'\x00' * bin_padding

    bin_chunk = struct.pack('<II', len(bin_data), 0x004E4942) + bin_data

    header = struct.pack('<III', 0x46546C67, 2, 12 + len(json_chunk) + len(bin_chunk))

    glb_path = temp_dir / "aligned.glb"
    with open(glb_path, 'wb') as f:
        f.write(header + json_chunk + bin_chunk)

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed with proper alignment


def test_glb_buffer_embedded_data(gltf_document, gltf_state, temp_dir):
    """Test that GLB binary data is properly embedded in buffers"""
    json_data = create_minimal_glb_json()
    bin_data = create_minimal_glb_buffer()

    glb_path = temp_dir / "embedded.glb"
    create_glb_file(json_data, bin_data, glb_path)

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 0)

    # Check that binary data was embedded
    slash.assert_equal(len(gltf_state.buffers), 1)
    slash.assert_is_not_none(gltf_state.buffers[0].data)
    slash.assert_equal(len(gltf_state.buffers[0].data), len(bin_data))


def test_glb_multiple_chunks(gltf_document, gltf_state, temp_dir):
    """Test GLB with multiple chunks (should ignore extra chunks)"""
    json_data = create_minimal_glb_json()
    bin_data = create_minimal_glb_buffer()

    # Create GLB with extra dummy chunk
    json_bytes = json.dumps(json_data).encode('utf-8')
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<II', len(bin_data), 0x004E4942) + bin_data
    dummy_chunk = struct.pack('<II', 4, 0x12345678) + b'dummy'

    total_length = 12 + len(json_chunk) + len(bin_chunk) + len(dummy_chunk)
    header = struct.pack('<III', 0x46546C67, 2, total_length)

    glb_path = temp_dir / "multi_chunk.glb"
    with open(glb_path, 'wb') as f:
        f.write(header + json_chunk + bin_chunk + dummy_chunk)

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed, ignoring unknown chunks


def test_glb_chunk_bounds_checking(gltf_document, gltf_state, temp_dir):
    """Test GLB chunk bounds checking"""
    glb_path = temp_dir / "bounds.glb"

    # Create GLB where chunk length exceeds file bounds
    header = struct.pack('<III', 0x46546C67, 2, 1000)  # Large claimed length
    json_chunk = struct.pack('<II', 500, 0x4E4F534A) + b'x' * 10  # But only 10 bytes

    with open(glb_path, 'wb') as f:
        f.write(header + json_chunk)

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 1)  # Should fail due to bounds error


def test_glb_empty_chunks(gltf_document, gltf_state, temp_dir):
    """Test GLB with empty chunks"""
    json_data = create_minimal_glb_json()

    # Create GLB with empty binary chunk
    json_bytes = json.dumps(json_data).encode('utf-8')
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    empty_bin_chunk = struct.pack('<II', 0, 0x004E4942)  # Empty binary chunk

    total_length = 12 + len(json_chunk) + len(empty_bin_chunk)
    header = struct.pack('<III', 0x46546C67, 2, total_length)

    glb_path = temp_dir / "empty_chunks.glb"
    with open(glb_path, 'wb') as f:
        f.write(header + json_chunk + empty_bin_chunk)

    result = gltf_document.append_from_file(str(glb_path), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed with empty binary chunk
