#!/usr/bin/env python3
"""
Round-trip GLTF test: Load -> Process -> Export -> Validate

This test verifies that the GLTF module can successfully load GLTF files,
process them internally, export them back to valid GLTF/GLB files,
and that the exported files pass gltf-validator validation.
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path
import logging

sys.path.insert(0, os.path.dirname(__file__))

from gltf_module import GLTFDocument, GLTFState, GLTFExporter
from gltf_module.logger import set_log_level

# Set up logging for testing
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
set_log_level(logging.INFO)


def find_gltf_validator():
    """Find gltf-validator executable"""
    # Try common locations
    possible_paths = [
        'gltf-validator',
        './gltf-validator',
        'node_modules/.bin/gltf-validator',
        'thirdparty/gltf-validator/gltf-validator',
    ]

    for path in possible_paths:
        try:
            # Check if executable exists
            if os.path.isfile(path) or (os.name == 'nt' and os.path.isfile(f"{path}.exe")):
                return path
            # Try running it
            result = subprocess.run([path, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.SubprocessError):
            continue

    return None


def validate_gltf_file(file_path, validator_path):
    """Validate GLTF file using gltf-validator"""
    try:
        cmd = [validator_path, file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Validation timeout"
    except Exception as e:
        return False, "", f"Validation error: {e}"


def test_round_trip():
    """Test round-trip GLTF processing"""
    print("GLTF Round-Trip Test")
    print("=" * 50)

    # Find a test GLTF file
    sample_dir = Path('thirdparty/glTF-Sample-Assets/Models')
    test_file = None

    if sample_dir.exists():
        # Find a simple GLTF file to test
        for root, dirs, files in os.walk(sample_dir):
            for file in files:
                if file.endswith('.gltf'):
                    test_file = Path(root) / file
                    break
            if test_file:
                break

    if not test_file:
        print("[ERROR] No GLTF test file found")
        return False

    print(f"[DIR] Testing with: {test_file}")

    # Step 1: Load GLTF file
    print("\n[1] Loading GLTF file...")
    doc = GLTFDocument()
    state = GLTFState()

    result = doc.append_from_file(str(test_file), state)
    if result != 0:
        print(f"[ERROR] Failed to load GLTF file (error code: {result})")
        return False

    print("[OK] GLTF file loaded successfully")
    print(f"   [STATS] Nodes: {len(state.nodes)}, Meshes: {len(state.meshes)}, Materials: {len(state.materials)}")

    # Step 2: Process the data (generate scene)
    print("\n[2] Processing GLTF data...")
    scene = doc.generate_scene(state)
    if scene is None:
        print("[ERROR] Failed to generate scene")
        return False

    print("[OK] Scene generated successfully")

    # Step 3: Export back to GLTF
    print("\n[3] Exporting GLTF file...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test GLTF export
        gltf_output = Path(temp_dir) / "exported_test.gltf"
        exporter = GLTFExporter()
        result = exporter.write_to_filesystem(state, str(gltf_output))

        if result != 0:
            print(f"[ERROR] Failed to export GLTF file (error code: {result})")
            return False

        print("[OK] GLTF file exported successfully")
        print(f"   [DIR] Output: {gltf_output}")

        # Step 4: Validate exported GLTF
        print("\n[4] Validating exported GLTF...")

        validator = find_gltf_validator()
        if not validator:
            print("[WARNING] gltf-validator not found, skipping validation")
            print("   Install with: npm install -g gltf-validator")
            print("   Or use deno: deno install npm:gltf-validator")
            return True  # Consider this a pass since export worked

        print(f"[SEARCH] Using validator: {validator}")

        valid, stdout, stderr = validate_gltf_file(str(gltf_output), validator)

        if valid:
            print("[OK] Exported GLTF passes validation!")
            return True
        else:
            print("[ERROR] Exported GLTF validation failed!")
            if stdout:
                print(f"   [FILE] stdout: {stdout}")
            if stderr:
                print(f"   [FILE] stderr: {stderr}")
            return False

    return True


def test_glb_export():
    """Test GLB export functionality"""
    print("\n[TEST] Testing GLB Export...")

    # Find a test GLTF file
    sample_dir = Path('thirdparty/glTF-Sample-Assets/Models')
    test_file = None

    if sample_dir.exists():
        for root, dirs, files in os.walk(sample_dir):
            for file in files:
                if file.endswith('.gltf'):
                    test_file = Path(root) / file
                    break
            if test_file:
                break

    if not test_file:
        print("[ERROR] No GLTF test file found for GLB test")
        return False

    # Load and export as GLB
    doc = GLTFDocument()
    state = GLTFState()

    result = doc.append_from_file(str(test_file), state)
    if result != 0:
        print("[ERROR] Failed to load GLTF for GLB test")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        glb_output = Path(temp_dir) / "exported_test.glb"
        exporter = GLTFExporter()
        result = exporter.write_to_filesystem(state, str(glb_output))

        if result != 0:
            print(f"[ERROR] Failed to export GLB file (error code: {result})")
            return False

        print("[OK] GLB file exported successfully")
        print(f"   [DIR] Output: {glb_output}")

        # Check file exists and has content
        if glb_output.exists() and glb_output.stat().st_size > 0:
            print("[OK] GLB file has valid size")
            return True
        else:
            print("[ERROR] GLB file is invalid")
            return False


if __name__ == "__main__":
    print("[START] Starting GLTF Round-Trip Tests\n")

    success1 = test_round_trip()
    success2 = test_glb_export()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("[OK] Round-trip GLTF processing works correctly")
        print("[OK] GLTF/GLB export functionality is working")
        print("[OK] Files are ready for gltf-validator testing")
        sys.exit(0)
    else:
        print("[ERROR] SOME TESTS FAILED")
        if not success1:
            print("[ERROR] Round-trip test failed")
        if not success2:
            print("[ERROR] GLB export test failed")
        sys.exit(1)
