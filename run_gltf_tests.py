#!/usr/bin/env python3
"""
GLTF Test Runner

Convenience script to run GLTF compatibility tests using different frameworks.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_slash_tests(max_files=None, verbose=False):
    """Run tests using Slash framework"""
    print("🚀 Running GLTF Compatibility Tests with Slash Framework")
    print("=" * 60)

    cmd = ["slash", "run", "slash_gltf_tests.py"]

    if max_files:
        cmd.extend(["--max-files", str(max_files)])

    if verbose:
        cmd.append("-v")

    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()

    try:
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ Slash not found. Install with: pip install slash")
        return False
    except Exception as e:
        print(f"❌ Error running Slash tests: {e}")
        return False

def run_comprehensive_tests(max_files=None):
    """Run comprehensive compatibility tests"""
    print("🧪 Running Comprehensive GLTF Compatibility Tests")
    print("=" * 60)

    cmd = ["python", "comprehensive_compatibility_test.py"]

    if max_files:
        cmd.extend(["--max-files", str(max_files)])

    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running comprehensive tests: {e}")
        return False

def run_round_trip_test():
    """Run basic round-trip test"""
    print("🔄 Running GLTF Round-Trip Test")
    print("=" * 60)

    try:
        result = subprocess.run([sys.executable, "round_trip_test.py"], cwd=os.getcwd())
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running round-trip test: {e}")
        return False

def install_dependencies():
    """Install test dependencies"""
    print("📦 Installing test dependencies...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
        ], cwd=os.getcwd())
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="GLTF Test Runner")
    parser.add_argument('--framework', choices=['slash', 'comprehensive', 'roundtrip'],
                       default='comprehensive', help='Test framework to use')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies first')

    args = parser.parse_args()

    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)

    success = False

    if args.framework == 'slash':
        success = run_slash_tests(args.max_files, args.verbose)
    elif args.framework == 'comprehensive':
        success = run_comprehensive_tests(args.max_files)
    elif args.framework == 'roundtrip':
        success = run_round_trip_test()

    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    print("GLTF Module Test Runner")
    print("Usage examples:")
    print("  python run_gltf_tests.py --framework slash --max-files 10")
    print("  python run_gltf_tests.py --framework comprehensive --max-files 5")
    print("  python run_gltf_tests.py --framework roundtrip")
    print("  python run_gltf_tests.py --install-deps --framework slash")
    print()

    main()
