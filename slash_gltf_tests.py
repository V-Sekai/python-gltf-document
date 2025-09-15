#!/usr/bin/env python3
"""
GLTF Compatibility Tests using Slash Testing Framework

This module provides comprehensive GLTF validation tests using the Slash testing framework.
Tests cover loading, processing, and export validation across the entire glTF-Sample-Assets collection.

See: https://slash.readthedocs.io/en/master/
"""

import os
import tempfile
import slash
from pathlib import Path
from typing import Dict, List, Any

# Import GLTF module
from gltf_module import GLTFDocument, GLTFState, GLTFExporter
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
def gltf_exporter():
    """Fixture providing a fresh GLTFExporter instance"""
    return GLTFExporter()


@slash.fixture
def temp_output_dir():
    """Fixture providing a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def discover_gltf_files(assets_dir: str = "thirdparty/glTF-Sample-Assets/Models") -> List[Dict[str, Any]]:
    """Discover all GLTF files and return test parameters, focusing on core features"""
    assets_path = Path(assets_dir)
    test_cases = []

    if not assets_path.exists():
        slash.logger.warning(f"Assets directory not found: {assets_path}")
        return test_cases

    # Skip advanced compression and extension directories (focus on core GLTF)
    skip_patterns = [
        'node_modules', '.git', '__pycache__',
        # Skip advanced compression formats
        'KTX', 'Basis', 'Draco', 'ETC1S', 'UASTC',
        # Skip advanced material extensions
        'ClearCoat', 'Sheen', 'Transmission', 'Volume',
        'Iridescence', 'Anisotropy', 'Dispersion',
        # Skip complex lighting
        'Punctual', 'Environment',
        # Skip specialized formats
        'Sparse', 'Instancing', 'Variants'
    ]

    # Walk through all subdirectories
    for root, dirs, files in os.walk(assets_path):
        root_path = Path(root)

        # Skip directories with advanced features
        should_skip = any(skip in str(root_path) for skip in skip_patterns)
        if should_skip:
            continue

        for file in files:
            if file.endswith(('.gltf', '.glb')):
                file_path = root_path / file
                variant = root_path.name

                # Additional filtering for file-level exclusions
                if any(skip in str(file_path) for skip in ['Draco', 'KTX', 'Basis']):
                    continue

                test_cases.append({
                    'file_path': str(file_path),
                    'file_name': file,
                    'variant': variant,
                    'is_glb': file.endswith('.glb'),
                    'relative_path': str(file_path.relative_to(assets_path))
                })

    return sorted(test_cases, key=lambda x: x['file_path'])


@slash.parametrize('gltf_file', discover_gltf_files())
def test_gltf_loading(gltf_document, gltf_state, gltf_file):
    """Test GLTF file loading capability"""
    slash.logger.info(f"Testing GLTF loading: {gltf_file['file_name']} ({gltf_file['variant']})")

    # Load GLTF file
    result = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)

    # Assert successful loading
    slash.assert_equal(result, 0, f"Failed to load GLTF file: {gltf_file['file_name']}")

    # Verify basic structure
    slash.assert_greater(len(gltf_state.nodes), 0, "GLTF file should contain at least one node")

    # Store metadata for subsequent tests
    slash.store.add_global('gltf_metadata', {
        'file': gltf_file,
        'node_count': len(gltf_state.nodes),
        'mesh_count': len(gltf_state.meshes),
        'material_count': len(gltf_state.materials),
        'texture_count': len(gltf_state.textures),
        'animation_count': len(gltf_state.animations),
        'skin_count': len(gltf_state.skins),
        'light_count': len(gltf_state.lights),
        'camera_count': len(gltf_state.cameras)
    })


@slash.parametrize('gltf_file', discover_gltf_files())
def test_gltf_scene_generation(gltf_document, gltf_state, gltf_file):
    """Test GLTF scene generation"""
    slash.logger.info(f"Testing scene generation: {gltf_file['file_name']} ({gltf_file['variant']})")

    # First load the file
    result = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)
    slash.assert_equal(result, 0, f"Failed to load GLTF file for scene generation: {gltf_file['file_name']}")

    # Generate scene
    scene = gltf_document.generate_scene(gltf_state)

    # Scene generation should succeed (may return None for empty scenes)
    # We don't assert it's not None since some GLTF files might not have scenes
    slash.logger.info(f"Scene generation completed for {gltf_file['file_name']}")


@slash.parametrize('gltf_file', discover_gltf_files())
def test_gltf_export_gltf(gltf_document, gltf_state, gltf_exporter, temp_output_dir, gltf_file):
    """Test GLTF export to .gltf format"""
    slash.logger.info(f"Testing GLTF export: {gltf_file['file_name']} ({gltf_file['variant']})")

    # Load and prepare GLTF
    result = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)
    slash.assert_equal(result, 0, f"Failed to load GLTF file for export: {gltf_file['file_name']}")

    # Export to GLTF
    output_path = temp_output_dir / f"export_{gltf_file['file_name']}"
    export_result = gltf_exporter.write_to_filesystem(gltf_state, str(output_path))

    # Assert successful export
    slash.assert_equal(export_result, 0, f"Failed to export GLTF file: {gltf_file['file_name']}")

    # Verify output file exists
    slash.assert_true(output_path.exists(), f"Exported GLTF file not found: {output_path}")

    # Verify file has content
    slash.assert_greater(output_path.stat().st_size, 0, f"Exported GLTF file is empty: {output_path}")


@slash.parametrize('gltf_file', discover_gltf_files())
def test_gltf_export_glb(gltf_document, gltf_state, gltf_exporter, temp_output_dir, gltf_file):
    """Test GLTF export to .glb format"""
    slash.logger.info(f"Testing GLB export: {gltf_file['file_name']} ({gltf_file['variant']})")

    # Load and prepare GLTF
    result = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)
    slash.assert_equal(result, 0, f"Failed to load GLTF file for GLB export: {gltf_file['file_name']}")

    # Export to GLB
    output_path = temp_output_dir / f"export_{Path(gltf_file['file_name']).stem}.glb"
    export_result = gltf_exporter.write_to_filesystem(gltf_state, str(output_path))

    # Assert successful export
    slash.assert_equal(export_result, 0, f"Failed to export GLB file: {gltf_file['file_name']}")

    # Verify output file exists
    slash.assert_true(output_path.exists(), f"Exported GLB file not found: {output_path}")

    # Verify file has content
    slash.assert_greater(output_path.stat().st_size, 0, f"Exported GLB file is empty: {output_path}")


@slash.parametrize('gltf_file', discover_gltf_files())
def test_gltf_round_trip(gltf_document, gltf_state, gltf_exporter, temp_output_dir, gltf_file):
    """Test complete round-trip: Load -> Export -> Reload"""
    slash.logger.info(f"Testing round-trip: {gltf_file['file_name']} ({gltf_file['variant']})")

    # Phase 1: Load original
    result1 = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)
    slash.assert_equal(result1, 0, f"Failed to load original GLTF: {gltf_file['file_name']}")

    original_node_count = len(gltf_state.nodes)

    # Phase 2: Export
    export_path = temp_output_dir / f"roundtrip_{gltf_file['file_name']}"
    export_result = gltf_exporter.write_to_filesystem(gltf_state, str(export_path))
    slash.assert_equal(export_result, 0, f"Failed to export GLTF: {gltf_file['file_name']}")

    # Phase 3: Reload exported file
    new_document = GLTFDocument()
    new_state = GLTFState()
    result2 = new_document.append_from_file(str(export_path), new_state)
    slash.assert_equal(result2, 0, f"Failed to reload exported GLTF: {gltf_file['file_name']}")

    # Verify structure is preserved (basic check)
    slash.assert_equal(len(new_state.nodes), original_node_count,
                      f"Node count mismatch after round-trip: {gltf_file['file_name']}")


@slash.parametrize('gltf_file', discover_gltf_files())
def test_gltf_validation_basic(gltf_document, gltf_state, gltf_file):
    """Test basic GLTF validation (structure and required fields)"""
    slash.logger.info(f"Testing validation: {gltf_file['file_name']} ({gltf_file['variant']})")

    # Load GLTF
    result = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)
    slash.assert_equal(result, 0, f"Failed to load GLTF for validation: {gltf_file['file_name']}")

    # Basic validation checks
    slash.assert_true(hasattr(gltf_state, 'json'), "GLTF state should have json attribute")
    slash.assert_is_not_none(gltf_state.json, "GLTF json should not be None")

    # Check required GLTF fields
    json_data = gltf_state.json
    slash.assert_in('asset', json_data, "GLTF should have asset field")
    slash.assert_in('version', json_data['asset'], "GLTF asset should have version")

    # Version should be 2.0
    slash.assert_equal(json_data['asset']['version'], '2.0', "GLTF version should be 2.0")


@slash.parametrize('gltf_file', discover_gltf_files())
def test_gltf_feature_detection(gltf_document, gltf_state, gltf_file):
    """Test GLTF feature detection and reporting"""
    slash.logger.info(f"Testing feature detection: {gltf_file['file_name']} ({gltf_file['variant']})")

    # Load GLTF
    result = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)
    slash.assert_equal(result, 0, f"Failed to load GLTF for feature detection: {gltf_file['file_name']}")

    # Collect feature information
    features = {
        'has_meshes': len(gltf_state.meshes) > 0,
        'has_materials': len(gltf_state.materials) > 0,
        'has_textures': len(gltf_state.textures) > 0,
        'has_animations': len(gltf_state.animations) > 0,
        'has_skins': len(gltf_state.skins) > 0,
        'has_lights': len(gltf_state.lights) > 0,
        'has_cameras': len(gltf_state.cameras) > 0,
        'has_extensions': bool(gltf_state.extensions.used),
    }

    # Store feature data for reporting
    slash.store.add_global(f"features_{gltf_file['file_name']}", features)

    # Basic assertions
    slash.assert_greater(len(gltf_state.nodes), 0, "Should have at least one node")


@slash.fixture
def gltf_test_session():
    """Session-level fixture for GLTF testing"""
    slash.logger.info("Starting GLTF compatibility test session")

    # Setup code here
    yield

    # Cleanup code here
    slash.logger.info("Completed GLTF compatibility test session")


@slash.sessionstart
def session_start():
    """Session start hook"""
    slash.logger.info("GLTF Compatibility Test Suite Starting")
    slash.logger.info("=" * 60)


@slash.sessionend
def session_end():
    """Session end hook - generate summary report"""
    slash.logger.info("=" * 60)
    slash.logger.info("GLTF Compatibility Test Suite Completed")

    # Generate summary from stored data
    summary = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'features_detected': {},
        'variant_coverage': {}
    }

    # Collect data from global store
    for key, value in slash.store._global_store.items():
        if key.startswith('features_'):
            for feature, present in value.items():
                if feature not in summary['features_detected']:
                    summary['features_detected'][feature] = 0
                if present:
                    summary['features_detected'][feature] += 1
        elif key == 'gltf_metadata':
            # Count variants
            variant = value['file']['variant']
            if variant not in summary['variant_coverage']:
                summary['variant_coverage'][variant] = 0
            summary['variant_coverage'][variant] += 1

    # Log summary
    slash.logger.info(f"Feature Detection Summary:")
    for feature, count in summary['features_detected'].items():
        slash.logger.info(f"  {feature}: {count} files")

    slash.logger.info(f"Variant Coverage:")
    for variant, count in summary['variant_coverage'].items():
        slash.logger.info(f"  {variant}: {count} files")


if __name__ == "__main__":
    # Allow running with python directly for debugging
    print("GLTF Compatibility Tests using Slash Framework")
    print("Run with: slash run slash_gltf_tests.py")
    print("Or: python -m slash run slash_gltf_tests.py")
