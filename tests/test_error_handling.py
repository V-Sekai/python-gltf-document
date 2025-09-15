#!/usr/bin/env python3
"""
Error Handling Tests using Slash Testing Framework

Tests for error handling and edge cases in GLTF processing.
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


def test_nonexistent_file(gltf_document, gltf_state):
    """Test loading a file that doesn't exist"""
    result = gltf_document.append_from_file("definitely_does_not_exist.gltf", gltf_state)
    slash.assert_equal(result, 1)  # Should return error code


def test_empty_file(gltf_document, gltf_state, temp_dir):
    """Test loading an empty file"""
    empty_file = temp_dir / "empty.gltf"
    empty_file.write_text("")

    result = gltf_document.append_from_file(str(empty_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on empty file


def test_invalid_json(gltf_document, gltf_state, temp_dir):
    """Test loading a file with invalid JSON"""
    invalid_file = temp_dir / "invalid.json"
    invalid_file.write_text("{invalid json content")

    result = gltf_document.append_from_file(str(invalid_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on invalid JSON


def test_missing_asset_field(gltf_document, gltf_state, temp_dir):
    """Test GLTF file missing required asset field"""
    gltf_data = {
        "scenes": [{"nodes": []}],
        "nodes": []
    }

    gltf_file = temp_dir / "no_asset.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail without asset field


def test_missing_version_field(gltf_document, gltf_state, temp_dir):
    """Test GLTF file with asset but no version"""
    gltf_data = {
        "asset": {
            "generator": "Test"
        },
        "scenes": [{"nodes": []}],
        "nodes": []
    }

    gltf_file = temp_dir / "no_version.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail without version


def test_unsupported_version(gltf_document, gltf_state, temp_dir):
    """Test GLTF file with unsupported version"""
    gltf_data = {
        "asset": {
            "version": "3.0"  # Unsupported version
        },
        "scenes": [{"nodes": []}],
        "nodes": []
    }

    gltf_file = temp_dir / "unsupported_version.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on unsupported version


def test_corrupted_glb_header(gltf_document, gltf_state, temp_dir):
    """Test GLB file with corrupted header"""
    glb_file = temp_dir / "corrupted_header.glb"

    # Write incomplete header
    with open(glb_file, 'wb') as f:
        f.write(b'glTF')  # Incomplete header

    result = gltf_document.append_from_file(str(glb_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on corrupted header


def test_invalid_glb_magic(gltf_document, gltf_state, temp_dir):
    """Test GLB file with invalid magic number"""
    glb_file = temp_dir / "invalid_magic.glb"

    # Write header with wrong magic
    with open(glb_file, 'wb') as f:
        f.write(b'XXXX')  # Wrong magic
        f.write(struct.pack('<III', 2, 100, 0))  # Version, length, chunk info

    result = gltf_document.append_from_file(str(glb_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on invalid magic


def test_glb_wrong_version(gltf_document, gltf_state, temp_dir):
    """Test GLB file with wrong version"""
    glb_file = temp_dir / "wrong_version.glb"

    with open(glb_file, 'wb') as f:
        f.write(b'glTF')  # Magic
        f.write(struct.pack('<III', 99, 100, 0))  # Wrong version (99)

    result = gltf_document.append_from_file(str(glb_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on wrong version


def test_glb_no_json_chunk(gltf_document, gltf_state, temp_dir):
    """Test GLB file without JSON chunk"""
    glb_file = temp_dir / "no_json.glb"

    with open(glb_file, 'wb') as f:
        f.write(b'glTF')  # Magic
        f.write(struct.pack('<III', 2, 12, 0))  # Version, length, no chunks

    result = gltf_document.append_from_file(str(glb_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail without JSON chunk


def test_glb_invalid_json_chunk(gltf_document, gltf_state, temp_dir):
    """Test GLB file with invalid JSON in chunk"""
    glb_file = temp_dir / "invalid_json_chunk.glb"

    json_data = b'{invalid json'
    chunk_length = len(json_data)

    with open(glb_file, 'wb') as f:
        f.write(b'glTF')  # Magic
        f.write(struct.pack('<II', 2, 12 + 8 + chunk_length))  # Version, total length
        f.write(struct.pack('<II', chunk_length, 0x4E4F534A))  # Chunk length, JSON type
        f.write(json_data)  # Invalid JSON

    result = gltf_document.append_from_file(str(glb_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on invalid JSON


def test_buffer_uri_handling(gltf_document, gltf_state, temp_dir):
    """Test various buffer URI handling scenarios"""
    # Test with data URI
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "buffers": [{
            "byteLength": 4,
            "uri": "data:application/octet-stream;base64,AAAA"  # base64 for 4 zero bytes
        }]
    }

    gltf_file = temp_dir / "data_uri.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed with data URI


def test_missing_external_buffer(gltf_document, gltf_state, temp_dir):
    """Test handling of missing external buffer files"""
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "buffers": [{
            "byteLength": 100,
            "uri": "missing_file.bin"
        }]
    }

    gltf_file = temp_dir / "missing_buffer.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed (buffer loading is optional)


def test_malformed_data_uri(gltf_document, gltf_state, temp_dir):
    """Test handling of malformed data URIs"""
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "buffers": [{
            "byteLength": 4,
            "uri": "data:invalid"
        }]
    }

    gltf_file = temp_dir / "malformed_data_uri.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed (graceful handling of malformed URIs)


def test_unicode_filename_handling(gltf_document, gltf_state, temp_dir):
    """Test handling of Unicode characters in filenames"""
    unicode_name = "测试文件_ñáéíóú.gltf"
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": []
    }

    gltf_file = temp_dir / unicode_name
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should handle Unicode filenames


def test_extremely_large_file_simulation(gltf_document, gltf_state, temp_dir):
    """Test handling of files that claim to be extremely large"""
    # This tests bounds checking without actually creating large files
    glb_file = temp_dir / "large_claim.glb"

    with open(glb_file, 'wb') as f:
        f.write(b'glTF')  # Magic
        f.write(struct.pack('<II', 2, 0xFFFFFFFF))  # Version, huge claimed length
        # No actual chunks

    result = gltf_document.append_from_file(str(glb_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on bounds checking


def test_empty_scenes_and_nodes(gltf_document, gltf_state, temp_dir):
    """Test handling of GLTF files with empty scenes/nodes"""
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],  # Empty scene
        "nodes": []  # No nodes
    }

    gltf_file = temp_dir / "empty.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed with empty content


def test_invalid_node_references(gltf_document, gltf_state, temp_dir):
    """Test handling of invalid node references"""
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [999]}],  # Reference to non-existent node
        "nodes": []
    }

    gltf_file = temp_dir / "invalid_refs.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed (validation is minimal)


def test_malformed_extensions(gltf_document, gltf_state, temp_dir):
    """Test handling of malformed extensions"""
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "extensions": "not an object",  # Invalid extensions
        "extensionsRequired": "also not an object"  # Invalid extensionsRequired
    }

    gltf_file = temp_dir / "malformed_extensions.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should succeed (extensions are optional)


def test_nested_empty_objects(gltf_document, gltf_state, temp_dir):
    """Test handling of deeply nested empty objects"""
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [{
            "mesh": 0,
            "extras": {"nested": {"empty": {}}}
        }],
        "meshes": [{
            "primitives": [{
                "attributes": {},
                "extras": {"deeply": {"nested": {"structure": {}}}}
            }]
        }]
    }

    gltf_file = temp_dir / "nested_empty.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)  # Should handle nested empty objects


def test_buffer_overflow_simulation(gltf_document, gltf_state, temp_dir):
    """Test handling of potential buffer overflow scenarios"""
    # Create a GLB where chunk extends beyond file
    glb_file = temp_dir / "overflow.glb"

    with open(glb_file, 'wb') as f:
        f.write(b'glTF')  # Magic
        f.write(struct.pack('<III', 2, 1000, 500))  # Version, length, chunk length (too big)
        f.write(b'JSON')  # Chunk type
        f.write(b'x' * 10)  # Only 10 bytes of data, but claimed 500

    result = gltf_document.append_from_file(str(glb_file), gltf_state)
    slash.assert_equal(result, 1)  # Should fail on overflow detection
