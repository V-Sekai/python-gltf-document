#!/usr/bin/env python3
"""
Example usage of the Godot GLTF Module

This script demonstrates how to use the GLTF module to load and process GLTF files,
both with and without PyTorch acceleration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gltf_module import GLTFDocument, is_torch_available, get_torch_device
import json


def create_sample_gltf():
    """Create a simple sample GLTF file for demonstration"""
    # Simple triangle mesh
    vertices = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    indices = [0, 1, 2]

    # Convert to base64
    import base64
    import struct

    # Pack vertex data (9 floats = 36 bytes)
    vertex_data = b''.join(struct.pack('<f', v) for v in vertices)
    vertex_b64 = base64.b64encode(vertex_data).decode('ascii')

    # Pack index data (3 shorts = 6 bytes)
    index_data = b''.join(struct.pack('<H', i) for i in indices)
    index_b64 = base64.b64encode(index_data).decode('ascii')

    gltf_data = {
        "asset": {
            "version": "2.0",
            "generator": "Godot GLTF Python Module Example"
        },
        "scene": 0,
        "scenes": [{
            "name": "SampleScene",
            "nodes": [0]
        }],
        "nodes": [{
            "name": "TriangleNode",
            "mesh": 0
        }],
        "meshes": [{
            "name": "TriangleMesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 0
                },
                "indices": 1,
                "mode": 4  # TRIANGLES
            }]
        }],
        "buffers": [
            {
                "byteLength": 36,
                "uri": f"data:application/octet-stream;base64,{vertex_b64}"
            },
            {
                "byteLength": 6,
                "uri": f"data:application/octet-stream;base64,{index_b64}"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": 36,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 1,
                "byteOffset": 0,
                "byteLength": 6,
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": 3,
                "type": "VEC3",
                "max": [1.0, 1.0, 0.0],
                "min": [0.0, 0.0, 0.0]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": 3,
                "type": "SCALAR",
                "max": [2],
                "min": [0]
            }
        ]
    }

    return gltf_data


def demonstrate_basic_usage():
    """Demonstrate basic GLTF loading and processing"""
    print("=== Basic GLTF Loading Demo ===")

    # Create sample GLTF data
    gltf_data = create_sample_gltf()

    # Load GLTF from string
    doc = GLTFDocument()
    success = doc.load_from_string(json.dumps(gltf_data))

    if success:
        print("[OK] GLTF loaded successfully!")
        print(f"  - Nodes: {len(doc.state.nodes)}")
        print(f"  - Meshes: {len(doc.state.meshes)}")
        print(f"  - Accessors: {len(doc.state.accessors)}")
        print(f"  - Buffers: {len(doc.state.buffers)}")

        # Decode vertex positions
        from gltf_module import GLTFAccessorDecoder
        decoder = GLTFAccessorDecoder()
        vertices = decoder.decode_accessor_as_vec3(doc.state, 0)
        indices = decoder.decode_accessor_as_indices(doc.state, 1)

        print(f"  - Vertices: {vertices}")
        print(f"  - Indices: {indices}")

    else:
        print("[FAIL] Failed to load GLTF")

    print()


def demonstrate_pytorch_usage():
    """Demonstrate PyTorch-accelerated GLTF processing"""
    print("=== PyTorch GLTF Processing Demo ===")

    if not is_torch_available():
        print("[FAIL] PyTorch not available - skipping PyTorch demo")
        return

    # Create sample GLTF data
    gltf_data = create_sample_gltf()

    # Load GLTF
    doc = GLTFDocument()
    success = doc.load_from_string(json.dumps(gltf_data))

    if not success:
        print("[FAIL] Failed to load GLTF")
        return

    # Use PyTorch decoder
    device = get_torch_device('cpu')
    from gltf_module import TorchGLTFAccessorDecoder
    torch_decoder = TorchGLTFAccessorDecoder(device=device)

    print(f"[OK] Using device: {device}")

    # Decode with PyTorch
    vertex_tensor = torch_decoder.decode_accessor_tensor(doc.state, 0)
    index_tensor = torch_decoder.decode_accessor_tensor(doc.state, 1)

    print(f"  - Vertex tensor shape: {vertex_tensor.shape}")
    print(f"  - Vertex tensor dtype: {vertex_tensor.dtype}")
    print(f"  - Index tensor shape: {index_tensor.shape}")
    print(f"  - Index tensor dtype: {index_tensor.dtype}")

    # Demonstrate tensor operations
    print(f"  - Vertex positions:\n{vertex_tensor}")
    print(f"  - Triangle indices: {index_tensor}")

    # Calculate bounding box
    if vertex_tensor.numel() > 0:
        min_coords = torch.min(vertex_tensor, dim=0)[0]
        max_coords = torch.max(vertex_tensor, dim=0)[0]
        print(f"  - Bounding box: min={min_coords}, max={max_coords}")

    print()


def demonstrate_scene_generation():
    """Demonstrate scene generation from GLTF"""
    print("=== Scene Generation Demo ===")

    # Create sample GLTF data
    gltf_data = create_sample_gltf()

    # Load GLTF
    doc = GLTFDocument()
    success = doc.load_from_string(json.dumps(gltf_data))

    if not success:
        print("[FAIL] Failed to load GLTF")
        return

    # Generate scene
    from gltf_module import GLTFSceneGenerator
    generator = GLTFSceneGenerator(doc.state)
    scene_nodes = generator.generate_scene()

    print(f"[OK] Generated {len(scene_nodes)} root scene nodes")

    for i, node in enumerate(scene_nodes):
        print(f"  - Node {i}: {node.name}")
        if node.mesh_data:
            print(f"    - Mesh: {node.mesh_data.num_vertices} vertices, {node.mesh_data.num_faces} faces")
        print(f"    - Transform: {node.transform}")

    print()


def main():
    """Main demonstration function"""
    print("Godot GLTF Module - Usage Examples")
    print("=" * 40)
    print()

    # Check PyTorch availability
    print(f"PyTorch available: {is_torch_available()}")
    print()

    # Run demonstrations
    demonstrate_basic_usage()
    demonstrate_pytorch_usage()
    demonstrate_scene_generation()

    print("Demo completed!")


if __name__ == "__main__":
    main()
