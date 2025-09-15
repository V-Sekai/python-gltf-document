#!/usr/bin/env python3
"""
GPU Process Script

Simple script to process GLTF files with the GPU pipeline.
Used by the justfile to avoid syntax issues with inline Python code.
"""

import sys
from gltf_module.gpu_pipeline import process_gltf_on_gpu

def main():
    if len(sys.argv) < 4:
        print("Usage: python gpu_process.py <input_file> <output_file> <operations>")
        print("Example: python gpu_process.py test_cube.gltf output.gltf transform,normals")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    operations_str = sys.argv[3]

    # Parse operations
    operations = [op.strip() for op in operations_str.split(',') if op.strip()]

    print(f"Processing {input_file} -> {output_file}")
    print(f"Operations: {operations}")

    result = process_gltf_on_gpu(input_file, output_file, operations)
    print(f"GPU processing result: {result}")

    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
