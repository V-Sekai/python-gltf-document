#!/usr/bin/env python3
"""
Test GLTF loading functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gltf_module import GLTFDocument, GLTFState

def test_gltf_loading():
    """Test loading a GLTF file"""
    print("Testing GLTF Loading Functionality")
    print("=" * 40)

    doc = GLTFDocument()
    state = GLTFState()

    # Try to load a basic GLTF file from the sample assets
    sample_dir = 'thirdparty/glTF-Sample-Assets/Models'
    if os.path.exists(sample_dir):
        # Find a simple GLTF file
        for root, dirs, files in os.walk(sample_dir):
            for file in files:
                if file.endswith('.gltf'):
                    gltf_path = os.path.join(root, file)
                    print(f'Testing GLTF loading with: {file}')
                    try:
                        result = doc.append_from_file(gltf_path, state)
                        if result == 0:
                            print('✓ GLTF loading successful!')
                            print(f'  Nodes: {len(state.nodes)}')
                            print(f'  Meshes: {len(state.meshes)}')
                            print(f'  Materials: {len(state.materials)}')
                            print(f'  Accessors: {len(state.accessors)}')
                            return True
                        else:
                            print(f'✗ GLTF loading failed with code: {result}')
                    except Exception as e:
                        print(f'✗ Exception during loading: {e}')
                        import traceback
                        traceback.print_exc()
                    break
            else:
                continue
            break
        else:
            print('No GLTF files found in sample assets')
            return False
    else:
        print('Sample assets directory not found')
        return False

if __name__ == "__main__":
    success = test_gltf_loading()
    print(f"\nGLTF loading test: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
