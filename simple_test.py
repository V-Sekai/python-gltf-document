#!/usr/bin/env python3
"""
Simple test to verify GLTF module functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_functionality():
    """Test basic GLTF module functionality"""
    print("Testing GLTF Module Basic Functionality")
    print("=" * 40)

    try:
        from gltf_module import GLTFDocument, GLTFSkinTool, is_torch_available
        print("[OK] Imports successful")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

    try:
        # Test GLTFDocument creation
        doc = GLTFDocument()
        print("[OK] GLTFDocument created")

        # Test API compatibility
        print(f"  JOINT_GROUP_SIZE: {doc.JOINT_GROUP_SIZE}")
        print(f"  ARRAY_BUFFER: {doc.ARRAY_BUFFER}")
        print(f"  RootNodeMode enum: {doc.RootNodeMode.ROOT_NODE_MODE_SINGLE_ROOT}")
        print(f"  VisibilityMode enum: {doc.VisibilityMode.VISIBILITY_MODE_INCLUDE_REQUIRED}")

        # Test configuration methods
        doc.set_naming_version(2)
        doc.set_image_format("PNG")
        doc.set_root_node_mode(doc.RootNodeMode.ROOT_NODE_MODE_MULTI_ROOT)
        doc.set_visibility_mode(doc.VisibilityMode.VISIBILITY_MODE_INCLUDE_OPTIONAL)
        print("[OK] Configuration methods work")

    except Exception as e:
        print(f"[FAIL] GLTFDocument test failed: {e}")
        return False

    try:
        # Test skin tool
        result = GLTFSkinTool.sanitize_bone_name("Bone:Left/Upper")
        expected = "Bone_Left_Upper"
        if result == expected:
            print("[OK] GLTFSkinTool bone naming works")
        else:
            print(f"[FAIL] GLTFSkinTool bone naming failed: got '{result}', expected '{expected}'")
            return False

    except Exception as e:
        print(f"[FAIL] GLTFSkinTool test failed: {e}")
        return False

    try:
        # Test PyTorch availability
        torch_available = is_torch_available()
        print(f"[OK] PyTorch {'available' if torch_available else 'not available'}")

        if torch_available:
            from gltf_module import get_torch_device
            device = get_torch_device('cpu')
            print(f"[OK] PyTorch device: {device}")

    except Exception as e:
        print(f"[FAIL] PyTorch test failed: {e}")
        return False

    print("\n[OK] All basic functionality tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
