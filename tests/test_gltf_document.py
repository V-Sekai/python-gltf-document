#!/usr/bin/env python3
"""
GLTF Document Tests using Slash Testing Framework

Tests for GLTFDocument class functionality including loading, parsing,
and basic operations.
"""

import os
import tempfile
import json
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


def test_gltf_document_initialization(gltf_document):
    """Test GLTFDocument initialization"""
    slash.assert_is_not_none(gltf_document)
    slash.assert_is_instance(gltf_document, GLTFDocument)

    # Check default properties
    slash.assert_equal(gltf_document.get_naming_version(), 2)
    slash.assert_equal(gltf_document.get_image_format(), "PNG")
    slash.assert_equal(gltf_document.get_lossy_quality(), 0.75)


def test_gltf_state_initialization(gltf_state):
    """Test GLTFState initialization"""
    slash.assert_is_not_none(gltf_state)
    slash.assert_is_instance(gltf_state, GLTFState)

    # Check initial state
    slash.assert_is_none(gltf_state.filename)
    slash.assert_is_none(gltf_state.base_path)
    slash.assert_is_none(gltf_state.json)
    slash.assert_equal(len(gltf_state.nodes), 0)
    slash.assert_equal(len(gltf_state.meshes), 0)
    slash.assert_equal(len(gltf_state.materials), 0)


def test_gltf_document_configuration(gltf_document):
    """Test GLTFDocument configuration methods"""
    # Test naming version
    gltf_document.set_naming_version(3)
    slash.assert_equal(gltf_document.get_naming_version(), 3)

    # Test image format
    gltf_document.set_image_format("WEBP")
    slash.assert_equal(gltf_document.get_image_format(), "WEBP")

    # Test lossy quality
    gltf_document.set_lossy_quality(0.9)
    slash.assert_equal(gltf_document.get_lossy_quality(), 0.9)

    # Test fallback settings
    gltf_document.set_fallback_image_format("JPEG")
    slash.assert_equal(gltf_document.get_fallback_image_format(), "JPEG")

    gltf_document.set_fallback_image_quality(0.8)
    slash.assert_equal(gltf_document.get_fallback_image_quality(), 0.8)


def test_gltf_document_root_node_modes(gltf_document):
    """Test GLTFDocument root node mode settings"""
    modes = [
        GLTFDocument.RootNodeMode.ROOT_NODE_MODE_SINGLE_ROOT,
        GLTFDocument.RootNodeMode.ROOT_NODE_MODE_KEEP_ROOT,
        GLTFDocument.RootNodeMode.ROOT_NODE_MODE_MULTI_ROOT
    ]

    for mode in modes:
        gltf_document.set_root_node_mode(mode)
        slash.assert_equal(gltf_document.get_root_node_mode(), mode)


def test_gltf_document_visibility_modes(gltf_document):
    """Test GLTFDocument visibility mode settings"""
    modes = [
        GLTFDocument.VisibilityMode.VISIBILITY_MODE_INCLUDE_REQUIRED,
        GLTFDocument.VisibilityMode.VISIBILITY_MODE_INCLUDE_OPTIONAL,
        GLTFDocument.VisibilityMode.VISIBILITY_MODE_EXCLUDE
    ]

    for mode in modes:
        gltf_document.set_visibility_mode(mode)
        slash.assert_equal(gltf_document.get_visibility_mode(), mode)


def test_gltf_document_string_representation(gltf_document):
    """Test GLTFDocument string representation"""
    str_repr = str(gltf_document)
    slash.assert_in("GLTFDocument", str_repr)


def test_gltf_state_string_representation(gltf_state):
    """Test GLTFState string representation"""
    str_repr = str(gltf_state)
    slash.assert_in("GLTFState", str_repr)


def test_gltf_document_clear(gltf_document):
    """Test GLTFDocument clear method"""
    # Modify document state
    gltf_document.set_naming_version(5)
    gltf_document.set_image_format("TIFF")

    # Clear should reset to defaults
    gltf_document.clear()
    slash.assert_equal(gltf_document.get_naming_version(), 2)
    slash.assert_equal(gltf_document.get_image_format(), "PNG")


def test_gltf_state_clear(gltf_state):
    """Test GLTFState clear method"""
    # Populate state with some data
    gltf_state.filename = "test.gltf"
    gltf_state.base_path = "/tmp"
    gltf_state.json = {"test": "data"}

    # Clear should reset everything
    gltf_state.clear()
    slash.assert_is_none(gltf_state.filename)
    slash.assert_is_none(gltf_state.base_path)
    slash.assert_is_none(gltf_state.json)


def test_gltf_document_get_state(gltf_document):
    """Test GLTFDocument get_state method"""
    state = gltf_document.get_state()
    slash.assert_is_not_none(state)
    slash.assert_is_instance(state, GLTFState)


def test_invalid_file_loading(gltf_document, gltf_state):
    """Test loading invalid/nonexistent files"""
    # Test with nonexistent file
    result = gltf_document.append_from_file("nonexistent_file.gltf", gltf_state)
    slash.assert_equal(result, 1)  # Should return error code

    # Test with invalid path
    result = gltf_document.append_from_file("", gltf_state)
    slash.assert_equal(result, 1)


def test_gltf_document_append_from_buffer(gltf_document, gltf_state):
    """Test append_from_buffer method"""
    # Test with None buffer
    result = gltf_document.append_from_buffer(None, "", gltf_state)
    slash.assert_equal(result, 1)

    # Test with empty buffer
    result = gltf_document.append_from_buffer(b"", "", gltf_state)
    slash.assert_equal(result, 1)

    # Test with invalid data
    result = gltf_document.append_from_buffer(b"invalid data", "", gltf_state)
    slash.assert_equal(result, 1)


def test_gltf_document_scene_generation_empty(gltf_document, gltf_state):
    """Test scene generation with empty state"""
    scene = gltf_document.generate_scene(gltf_state)
    # Should not crash, may return None for empty scenes
    slash.assert_true(scene is None or hasattr(scene, '__dict__'))


def test_gltf_document_generate_buffer_empty(gltf_document, gltf_state):
    """Test buffer generation with empty state"""
    buffer_data = gltf_document.generate_buffer(gltf_state)
    slash.assert_is_instance(buffer_data, bytes)


def test_gltf_document_write_to_filesystem_empty(gltf_document, gltf_state, temp_dir):
    """Test export with empty state"""
    output_path = temp_dir / "empty_test.gltf"
    result = gltf_document.write_to_filesystem(gltf_state, str(output_path))
    # May succeed or fail depending on implementation, but shouldn't crash
    slash.assert_in(result, [0, 1])


def test_gltf_document_append_from_scene(gltf_document, gltf_state):
    """Test append_from_scene method (placeholder)"""
    # This is currently a placeholder implementation
    result = gltf_document.append_from_scene(None, gltf_state)
    slash.assert_equal(result, 1)  # Should return error for now


@slash.parametrize('config_name,config_value', [
    ('naming_version', 1),
    ('naming_version', 3),
    ('image_format', 'JPEG'),
    ('image_format', 'WEBP'),
    ('lossy_quality', 0.5),
    ('lossy_quality', 1.0),
])
def test_gltf_document_configuration_range(gltf_document, config_name, config_value):
    """Test various configuration values"""
    if config_name == 'naming_version':
        gltf_document.set_naming_version(config_value)
        slash.assert_equal(gltf_document.get_naming_version(), config_value)
    elif config_name == 'image_format':
        gltf_document.set_image_format(config_value)
        slash.assert_equal(gltf_document.get_image_format(), config_value)
    elif config_name == 'lossy_quality':
        gltf_document.set_lossy_quality(config_value)
        slash.assert_equal(gltf_document.get_lossy_quality(), config_value)


def test_gltf_constants(gltf_document):
    """Test GLTF constant values"""
    slash.assert_equal(gltf_document.GLTF_VERSION, "2.0")
    slash.assert_equal(gltf_document.GLTF_MAGIC, 0x46546C67)  # "glTF"
    slash.assert_equal(gltf_document.GLTF_JSON_CHUNK_TYPE, 0x4E4F534A)  # "JSON"
    slash.assert_equal(gltf_document.GLTF_BIN_CHUNK_TYPE, 0x004E4942)  # "BIN\x00"
