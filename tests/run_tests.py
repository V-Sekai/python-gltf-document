#!/usr/bin/env python3
"""
Test Runner for GLTF Module Test Suite

Runs all Slash-based tests for the GLTF module.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_slash_tests(test_files=None, verbose=False):
    """Run Slash tests for specified test files"""

    # Add current directory to Python path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))

    if test_files is None:
        # Run all test files in tests directory
        test_files = [
            'tests/test_gltf_document.py',
            'tests/test_glb_parsing.py',
            'tests/test_scene_generation.py',
            'tests/test_error_handling.py',
            'tests/test_performance.py'
        ]

    print("[TEST] Running GLTF Module Test Suite")
    print("=" * 50)

    success_count = 0
    total_count = len(test_files)

    for test_file in test_files:
        test_path = current_dir / test_file
        if not test_path.exists():
            print(f"[ERROR] Test file not found: {test_file}")
            continue

        print(f"\n[RUN] Running: {test_file}")
        print("-" * 30)

        # Build slash command
        cmd = ['python', '-m', 'slash', 'run', str(test_path)]

        if verbose:
            cmd.append('-v')

        try:
            result = subprocess.run(cmd, cwd=current_dir, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("[PASS] PASSED")
                success_count += 1
            else:
                print("[FAIL] FAILED")
                if verbose:
                    print("STDOUT:")
                    print(result.stdout)
                    print("STDERR:")
                    print(result.stderr)

        except subprocess.TimeoutExpired:
            print("[TIMEOUT] TIMEOUT (5 minutes)")
        except Exception as e:
            print(f"[CRASH] ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"[STATS] Test Results: {success_count}/{total_count} test files passed")

    if success_count == total_count:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[WARNING] Some tests failed. Check output above for details.")
        return 1


def run_single_test(test_file, verbose=False):
    """Run a single test file"""
    return run_slash_tests([test_file], verbose)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="GLTF Module Test Runner")
    parser.add_argument('--test', help='Run specific test file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list', action='store_true', help='List available tests')

    args = parser.parse_args()

    if args.list:
        print("Available test files:")
        test_dir = Path(__file__).parent
        for test_file in test_dir.glob('test_*.py'):
            print(f"  {test_file.name}")
        return 0

    if args.test:
        return run_single_test(args.test, args.verbose)
    else:
        return run_slash_tests(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
