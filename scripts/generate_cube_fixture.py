#!/usr/bin/env python3
"""
Generate Cube Test Fixture

This script generates test fixtures for the GPU GLTF pipeline including:
- test_cube.bin: Binary buffer data
- test_cube.gltf: GLTF JSON with external buffer reference
- test_cube.base64: Base64 encoded version for CLI conversion
- test_cube_embedded.gltf: GLTF with embedded base64 data
"""

import struct
import json
import os
from pathlib import Path


def generate_cube_geometry():
    """Generate cube geometry data"""
    # Cube vertex positions (8 vertices)
    positions = [
        -0.5, -0.5,  0.5,   # 0: left-bottom-front
         0.5, -0.5,  0.5,   # 1: right-bottom-front
         0.5,  0.5,  0.5,   # 2: right-top-front
        -0.5,  0.5,  0.5,   # 3: left-top-front
        -0.5, -0.5, -0.5,   # 4: left-bottom-back
         0.5, -0.5, -0.5,   # 5: right-bottom-back
         0.5,  0.5, -0.5,   # 6: right-top-back
        -0.5,  0.5, -0.5    # 7: left-top-back
    ]

    # Cube normals (6 face normals)
    normals = [
        # Front face
         0.0,  0.0,  1.0,   # +Z
         0.0,  0.0,  1.0,
         0.0,  0.0,  1.0,
         0.0,  0.0,  1.0,
        # Back face
         0.0,  0.0, -1.0,   # -Z
         0.0,  0.0, -1.0,
         0.0,  0.0, -1.0,
         0.0,  0.0, -1.0
    ]

    # UV coordinates (simple planar mapping)
    uvs = [
        0.0, 0.0,  # 0
        1.0, 0.0,  # 1
        1.0, 1.0,  # 2
        0.0, 1.0,  # 3
        0.0, 0.0,  # 4
        1.0, 0.0,  # 5
        1.0, 1.0,  # 6
        0.0, 1.0   # 7
    ]

    # Cube indices (36 indices for 12 triangles)
    indices = [
        # Front face
        0, 1, 2,  0, 2, 3,
        # Back face
        5, 4, 7,  5, 7, 6,
        # Left face
        4, 0, 3,  4, 3, 7,
        # Right face
        1, 5, 6,  1, 6, 2,
        # Top face
        3, 2, 6,  3, 6, 7,
        # Bottom face
        4, 5, 1,  4, 1, 0
    ]

    return positions, normals, uvs, indices


def create_binary_buffer(positions, normals, uvs, indices):
    """Create binary buffer data"""
    # Pack binary data
    pos_data = b''.join(struct.pack('<3f', *positions[i:i+3]) for i in range(0, len(positions), 3))
    nor_data = b''.join(struct.pack('<3f', *normals[i:i+3]) for i in range(0, len(normals), 3))
    uv_data = b''.join(struct.pack('<2f', *uvs[i:i+2]) for i in range(0, len(uvs), 2))
    idx_data = b''.join(struct.pack('<H', idx) for idx in indices)

    # Combine all data
    buffer_data = pos_data + nor_data + uv_data + idx_data

    return buffer_data, len(pos_data), len(nor_data), len(uv_data), len(idx_data)


def create_gltf_json(buffer_size):
    """Create GLTF JSON structure"""
    gltf_data = {
        "asset": {
            "version": "2.0",
            "generator": "GPU Pipeline Test Fixture"
        },
        "scene": 0,
        "scenes": [{
            "name": "TestScene",
            "nodes": [0]
        }],
        "nodes": [{
            "name": "TestCube",
            "mesh": 0,
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0]
        }],
        "meshes": [{
            "name": "CubeMesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                },
                "indices": 3,
                "material": 0,
                "mode": 4
            }]
        }],
        "materials": [{
            "name": "CubeMaterial",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.5, 0.2, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.7
            }
        }],
        "buffers": [
            {
                "byteLength": buffer_size,
                "uri": "test_cube.bin"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": 96,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": 96,
                "byteLength": 96,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": 192,
                "byteLength": 64,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": 256,
                "byteLength": 72,
                "target": 34963
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3",
                "max": [0.5, 0.5, 0.5],
                "min": [-0.5, -0.5, -0.5]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC2"
            },
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5123,
                "count": 36,
                "type": "SCALAR",
                "max": [7],
                "min": [0]
            }
        ]
    }

    return gltf_data


def main():
    """Main function to generate all test fixtures"""
    import sys

    # Check for command line arguments
    embedded_only = "--embedded-only" in sys.argv

    print("🎨 Generating GPU GLTF Pipeline Test Fixtures")
    print("=" * 50)

    # Create test_fixtures directory
    fixtures_dir = Path("test_fixtures")
    fixtures_dir.mkdir(exist_ok=True)

    if not embedded_only:
        # Generate geometry data
        print("📐 Generating cube geometry...")
        positions, normals, uvs, indices = generate_cube_geometry()

        # Create binary buffer
        print("💾 Creating binary buffer...")
        buffer_data, pos_size, nor_size, uv_size, idx_size = create_binary_buffer(
            positions, normals, uvs, indices
        )

        # Write binary file
        bin_path = fixtures_dir / "test_cube.bin"
        with open(bin_path, 'wb') as f:
            f.write(buffer_data)

        print(f"✅ Created {bin_path} ({len(buffer_data)} bytes)")
        print(f"   Positions: {pos_size} bytes")
        print(f"   Normals: {nor_size} bytes")
        print(f"   UVs: {uv_size} bytes")
        print(f"   Indices: {idx_size} bytes")

        # Create GLTF JSON
        print("📄 Creating GLTF JSON...")
        gltf_data = create_gltf_json(len(buffer_data))

        # Write GLTF file
        gltf_path = fixtures_dir / "test_cube.gltf"
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)

        print(f"✅ Created {gltf_path}")

        # Create base64 version
        print("🔄 Creating base64 version...")
        import base64
        base64_data = base64.b64encode(buffer_data).decode('ascii')

        base64_path = fixtures_dir / "test_cube.base64"
        with open(base64_path, 'w') as f:
            f.write(base64_data)

        print(f"✅ Created {base64_path}")

    # Create embedded GLTF (always generated)
    print("📦 Creating embedded GLTF...")
    if embedded_only:
        # Read existing base64 data
        base64_path = fixtures_dir / "test_cube.base64"
        if not base64_path.exists():
            print("❌ test_cube.base64 not found. Run without --embedded-only first.")
            return

        with open(base64_path, 'r') as f:
            base64_data = f.read().strip()

        # Read existing GLTF data
        gltf_path = fixtures_dir / "test_cube.gltf"
        if not gltf_path.exists():
            print("❌ test_cube.gltf not found. Run without --embedded-only first.")
            return

        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
    else:
        # Use the data we just generated
        pass

    embedded_gltf_data = gltf_data.copy()
    embedded_gltf_data["asset"]["generator"] = "GPU Pipeline Test Fixture (Embedded)"
    embedded_gltf_data["buffers"][0]["uri"] = f"data:application/octet-stream;base64,{base64_data}"

    embedded_path = fixtures_dir / "test_cube_embedded.gltf"
    with open(embedded_path, 'w') as f:
        json.dump(embedded_gltf_data, f, indent=2)

    print(f"✅ Created {embedded_path}")

    if not embedded_only:
        print("\n" + "=" * 50)
        print("🎉 Test fixture generation complete!")
        print("=" * 50)
        print("Generated files:")
        print(f"  📁 {bin_path}")
        print(f"  📄 {gltf_path}")
        print(f"  🔄 {base64_path}")
        print(f"  📦 {embedded_path}")
        print()
        print("You can now use these fixtures for testing:")
        print("  just verify-fixtures  # Verify all fixtures")
        print("  just test-gpu         # Run GPU pipeline tests")
    else:
        print("✅ Embedded GLTF fixture updated!")


if __name__ == "__main__":
    main()
