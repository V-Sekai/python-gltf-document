#!/usr/bin/env python3
"""
Performance Tests using Slash Testing Framework

Tests for performance characteristics of GLTF processing.
"""

import os
import time
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


def create_large_gltf(num_nodes=1000, num_meshes=100):
    """Create a large GLTF file for performance testing"""
    nodes = []
    meshes = []
    materials = []

    # Create materials
    for i in range(10):
        materials.append({
            "name": f"Material_{i}",
            "pbrMetallicRoughness": {
                "baseColorFactor": [i/10, (i+1)/10, (i+2)/10, 1.0],
                "metallicFactor": i/10,
                "roughnessFactor": (i+5)/10
            }
        })

    # Create meshes
    for i in range(num_meshes):
        meshes.append({
            "name": f"Mesh_{i}",
            "primitives": [{
                "attributes": {
                    "POSITION": i * 4,  # Reference accessor
                    "NORMAL": i * 4 + 1,
                    "TEXCOORD_0": i * 4 + 2
                },
                "indices": i * 4 + 3,
                "material": i % len(materials)
            }]
        })

    # Create nodes
    for i in range(num_nodes):
        node = {
            "name": f"Node_{i}",
            "translation": [i * 0.1, i * 0.05, i * 0.02],
            "scale": [1.0, 1.0, 1.0]
        }

        if i < num_meshes:
            node["mesh"] = i

        if i > 0:
            node["children"] = []  # Will be populated below

        nodes.append(node)

    # Set up parent-child relationships
    for i in range(1, min(100, num_nodes)):  # Create hierarchy for first 100 nodes
        parent_idx = (i - 1) // 10
        if parent_idx < len(nodes):
            nodes[parent_idx]["children"].append(i)

    # Create accessors (simplified)
    accessors = []
    for i in range(num_meshes * 4):  # 4 accessors per mesh
        accessors.append({
            "bufferView": i,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": 24,  # 8 vertices * 3 components
            "type": "VEC3",
            "max": [1.0, 1.0, 1.0],
            "min": [-1.0, -1.0, -1.0]
        })

    # Create buffer views
    buffer_views = []
    for i in range(num_meshes * 4):
        buffer_views.append({
            "buffer": 0,
            "byteOffset": i * 72,  # 24 floats * 4 bytes each
            "byteLength": 72,
            "target": 34962  # ARRAY_BUFFER
        })

    # Create buffers
    total_buffer_size = num_meshes * 4 * 72
    buffers = [{
        "byteLength": total_buffer_size,
        "uri": f"data:application/octet-stream;base64,{total_buffer_size * 'A'}"
    }]

    return {
        "asset": {
            "version": "2.0",
            "generator": f"Performance Test - {num_nodes} nodes, {num_meshes} meshes"
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": buffers,
        "bufferViews": buffer_views,
        "accessors": accessors
    }


def test_small_file_performance(gltf_document, gltf_state, temp_dir):
    """Test performance with a small GLTF file"""
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Test", "mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1
            }]
        }],
        "buffers": [{"byteLength": 24, "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAAAA"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 24}],
        "accessors": [{
            "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3",
            "max": [1,1,1], "min": [-1,-1,-1]
        }, {
            "bufferView": 0, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR",
            "max": [1], "min": [0]
        }]
    }

    gltf_file = temp_dir / "small.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    start_time = time.time()
    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    load_time = time.time() - start_time

    slash.assert_equal(result, 0)
    slash.assert_less(load_time, 0.1)  # Should load in under 100ms

    # Store performance data
    slash.store.add_global('perf_small_load', load_time)


def test_medium_file_performance(gltf_document, gltf_state, temp_dir):
    """Test performance with a medium-sized GLTF file"""
    gltf_data = create_large_gltf(num_nodes=100, num_meshes=20)

    gltf_file = temp_dir / "medium.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    start_time = time.time()
    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    load_time = time.time() - start_time

    slash.assert_equal(result, 0)
    slash.assert_less(load_time, 0.5)  # Should load in under 500ms

    # Store performance data
    slash.store.add_global('perf_medium_load', load_time)


def test_large_file_performance(gltf_document, gltf_state, temp_dir):
    """Test performance with a large GLTF file"""
    gltf_data = create_large_gltf(num_nodes=500, num_meshes=50)

    gltf_file = temp_dir / "large.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    start_time = time.time()
    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    load_time = time.time() - start_time

    slash.assert_equal(result, 0)
    slash.assert_less(load_time, 2.0)  # Should load in under 2 seconds

    # Store performance data
    slash.store.add_global('perf_large_load', load_time)


def test_scene_generation_performance(gltf_document, gltf_state, temp_dir):
    """Test scene generation performance"""
    gltf_data = create_large_gltf(num_nodes=200, num_meshes=30)

    gltf_file = temp_dir / "scene_perf.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    # Load file
    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)

    # Time scene generation
    start_time = time.time()
    scene = gltf_document.generate_scene(gltf_state)
    scene_time = time.time() - start_time

    slash.assert_less(scene_time, 1.0)  # Should generate scene in under 1 second

    # Store performance data
    slash.store.add_global('perf_scene_gen', scene_time)


def test_export_performance(gltf_document, gltf_state, temp_dir):
    """Test export performance"""
    gltf_data = create_large_gltf(num_nodes=100, num_meshes=20)

    gltf_file = temp_dir / "export_perf.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    # Load file
    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)

    # Time GLTF export
    output_file = temp_dir / "exported.gltf"
    start_time = time.time()
    export_result = gltf_document.write_to_filesystem(gltf_state, str(output_file))
    export_time = time.time() - start_time

    slash.assert_equal(export_result, 0)
    slash.assert_less(export_time, 1.0)  # Should export in under 1 second

    # Verify output file exists and has content
    slash.assert_true(output_file.exists())
    slash.assert_greater(output_file.stat().st_size, 1000)  # Should be substantial

    # Store performance data
    slash.store.add_global('perf_export_gltf', export_time)


def test_glb_export_performance(gltf_document, gltf_state, temp_dir):
    """Test GLB export performance"""
    gltf_data = create_large_gltf(num_nodes=100, num_meshes=20)

    gltf_file = temp_dir / "glb_export_perf.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    # Load file
    result = gltf_document.append_from_file(str(gltf_file), gltf_state)
    slash.assert_equal(result, 0)

    # Time GLB export
    output_file = temp_dir / "exported.glb"
    start_time = time.time()
    export_result = gltf_document.write_to_filesystem(gltf_state, str(output_file))
    export_time = time.time() - start_time

    slash.assert_equal(export_result, 0)
    slash.assert_less(export_time, 1.0)  # Should export in under 1 second

    # Verify output file exists and has content
    slash.assert_true(output_file.exists())
    slash.assert_greater(output_file.stat().st_size, 1000)  # Should be substantial

    # Store performance data
    slash.store.add_global('perf_export_glb', export_time)


def test_memory_efficiency(gltf_document, gltf_state, temp_dir):
    """Test memory efficiency with repeated operations"""
    gltf_data = create_large_gltf(num_nodes=50, num_meshes=10)

    gltf_file = temp_dir / "memory_test.gltf"
    gltf_file.write_text(json.dumps(gltf_data))

    # Perform multiple load operations
    for i in range(5):
        fresh_doc = GLTFDocument()
        fresh_state = GLTFState()

        start_time = time.time()
        result = fresh_doc.append_from_file(str(gltf_file), fresh_state)
        load_time = time.time() - start_time

        slash.assert_equal(result, 0)
        slash.assert_less(load_time, 0.2)  # Each should be fast

        # Clear state between runs
        fresh_state.clear()

    # Store performance data
    slash.store.add_global('perf_memory_test', 'completed')


def test_round_trip_performance(gltf_document, gltf_state, temp_dir):
    """Test complete round-trip performance"""
    gltf_data = create_large_gltf(num_nodes=50, num_meshes=10)

    original_file = temp_dir / "round_trip_original.gltf"
    original_file.write_text(json.dumps(gltf_data))

    # Phase 1: Load
    load_start = time.time()
    result1 = gltf_document.append_from_file(str(original_file), gltf_state)
    load_time = time.time() - load_start

    slash.assert_equal(result1, 0)
    slash.assert_less(load_time, 0.2)

    # Phase 2: Export
    exported_file = temp_dir / "round_trip_exported.gltf"
    export_start = time.time()
    export_result = gltf_document.write_to_filesystem(gltf_state, str(exported_file))
    export_time = time.time() - export_start

    slash.assert_equal(export_result, 0)
    slash.assert_less(export_time, 0.5)

    # Phase 3: Reload
    reload_doc = GLTFDocument()
    reload_state = GLTFState()
    reload_start = time.time()
    result2 = reload_doc.append_from_file(str(exported_file), reload_state)
    reload_time = time.time() - reload_start

    slash.assert_equal(result2, 0)
    slash.assert_less(reload_time, 0.2)

    total_time = load_time + export_time + reload_time
    slash.assert_less(total_time, 1.0)  # Complete round-trip under 1 second

    # Store performance data
    slash.store.add_global('perf_round_trip', {
        'load_time': load_time,
        'export_time': export_time,
        'reload_time': reload_time,
        'total_time': total_time
    })


@slash.sessionend
def performance_summary():
    """Generate performance summary at end of test session"""
    slash.logger.info("=" * 60)
    slash.logger.info("PERFORMANCE TEST SUMMARY")
    slash.logger.info("=" * 60)

    # Collect performance data
    perf_data = {}
    for key, value in slash.store._global_store.items():
        if key.startswith('perf_'):
            perf_data[key] = value

    # Report individual metrics
    slash.logger.info("Individual Performance Metrics:")
    for metric, value in perf_data.items():
        if isinstance(value, dict):
            slash.logger.info(f"  {metric}:")
            for sub_metric, sub_value in value.items():
                slash.logger.info(f"    {sub_metric}: {sub_value:.4f}s")
        else:
            slash.logger.info(f"  {metric}: {value:.4f}s")

    # Calculate averages and report
    load_times = []
    for key, value in perf_data.items():
        if 'load' in key and isinstance(value, (int, float)):
            load_times.append(value)

    if load_times:
        avg_load = sum(load_times) / len(load_times)
        slash.logger.info(f"Average load time: {avg_load:.4f}s")

    slash.logger.info("Performance tests completed!")
