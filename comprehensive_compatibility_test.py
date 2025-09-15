#!/usr/bin/env python3
"""
Comprehensive GLTF Compatibility Test

Tests the GLTF module against the entire glTF-Sample-Assets collection
to determine compatibility across different GLTF features and variants.

This test uses a Slash-like testing framework approach to validate
GLTF loading, processing, and export capabilities.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))

from gltf_module import GLTFDocument, GLTFState, GLTFExporter
from gltf_module.logger import get_logger, set_log_level

# Set up logging
logger = get_logger('compatibility_test')
set_log_level(logging.WARNING)  # Reduce noise during testing


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class CompatibilityResult:
    """Result of testing a single GLTF file"""
    file_path: str
    file_name: str
    variant: str
    load_success: bool
    scene_generation_success: bool
    export_gltf_success: bool
    export_glb_success: bool
    load_error: str = ""
    scene_generation_error: str = ""
    export_gltf_error: str = ""
    export_glb_error: str = ""
    load_time: float = 0.0
    scene_time: float = 0.0
    export_time: float = 0.0
    node_count: int = 0
    mesh_count: int = 0
    material_count: int = 0
    texture_count: int = 0
    animation_count: int = 0
    skin_count: int = 0
    light_count: int = 0
    camera_count: int = 0

    @property
    def overall_result(self) -> TestResult:
        """Determine overall test result"""
        if not self.load_success:
            return TestResult.FAIL
        if not self.scene_generation_success:
            return TestResult.FAIL
        if not (self.export_gltf_success or self.export_glb_success):
            return TestResult.FAIL
        return TestResult.PASS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'variant': self.variant,
            'result': self.overall_result.value,
            'load_success': self.load_success,
            'load_error': self.load_error,
            'scene_generation_success': self.scene_generation_success,
            'scene_generation_error': self.scene_generation_error,
            'export_gltf_success': self.export_gltf_success,
            'export_gltf_error': self.export_gltf_error,
            'export_glb_success': self.export_glb_success,
            'export_glb_error': self.export_glb_error,
            'load_time': self.load_time,
            'scene_time': self.scene_time,
            'export_time': self.export_time,
            'node_count': self.node_count,
            'mesh_count': self.mesh_count,
            'material_count': self.material_count,
            'texture_count': self.texture_count,
            'animation_count': self.animation_count,
            'skin_count': self.skin_count,
            'light_count': self.light_count,
            'camera_count': self.camera_count,
        }


class GLTFCompatibilityTest:
    """Comprehensive GLTF compatibility testing framework"""

    def __init__(self, assets_dir: str = "thirdparty/glTF-Sample-Assets/Models"):
        self.assets_dir = Path(assets_dir)
        self.results: List[CompatibilityResult] = []
        self.test_start_time = 0.0
        self.logger = get_logger('compatibility')

    def discover_gltf_files(self) -> List[Tuple[Path, str]]:
        """Discover all GLTF files in the assets directory, focusing on core features"""
        gltf_files = []

        if not self.assets_dir.exists():
            self.logger.error(f"Assets directory not found: {self.assets_dir}")
            return gltf_files

        # Skip advanced compression and extension directories
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
        for root, dirs, files in os.walk(self.assets_dir):
            root_path = Path(root)

            # Skip directories with advanced features
            should_skip = any(skip in str(root_path) for skip in skip_patterns)
            if should_skip:
                continue

            for file in files:
                if file.endswith(('.gltf', '.glb')):
                    file_path = root_path / file
                    # Determine variant (directory name)
                    variant = root_path.name

                    # Additional filtering for file-level exclusions
                    if any(skip in str(file_path) for skip in ['Draco', 'KTX', 'Basis']):
                        continue

                    gltf_files.append((file_path, variant))

        return sorted(gltf_files)

    def test_single_file(self, file_path: Path, variant: str) -> CompatibilityResult:
        """Test a single GLTF file for compatibility"""
        result = CompatibilityResult(
            file_path=str(file_path),
            file_name=file_path.name,
            variant=variant,
            load_success=False,
            scene_generation_success=False,
            export_gltf_success=False,
            export_glb_success=False
        )

        try:
            # Test 1: Load GLTF file
            load_start = time.time()
            doc = GLTFDocument()
            state = GLTFState()

            load_result = doc.append_from_file(str(file_path), state)
            result.load_time = time.time() - load_start

            if load_result != 0:
                result.load_success = False
                result.load_error = f"Load failed with code: {load_result}"
                return result

            result.load_success = True
            result.node_count = len(state.nodes)
            result.mesh_count = len(state.meshes)
            result.material_count = len(state.materials)
            result.texture_count = len(state.textures)
            result.animation_count = len(state.animations)
            result.skin_count = len(state.skins)
            result.light_count = len(state.lights)
            result.camera_count = len(state.cameras)

            # Test 2: Generate scene
            scene_start = time.time()
            try:
                scene = doc.generate_scene(state)
                result.scene_generation_success = scene is not None
                result.scene_time = time.time() - scene_start
            except Exception as e:
                result.scene_generation_success = False
                result.scene_generation_error = str(e)
                result.scene_time = time.time() - scene_start

            # Test 3: Export to GLTF
            export_start = time.time()
            try:
                exporter = GLTFExporter()
                export_result = exporter.write_to_filesystem(state, f"/tmp/test_{file_path.stem}.gltf")
                result.export_gltf_success = export_result == 0
                if not result.export_gltf_success:
                    result.export_gltf_error = f"Export failed with code: {export_result}"
            except Exception as e:
                result.export_gltf_success = False
                result.export_gltf_error = str(e)

            # Test 4: Export to GLB
            try:
                export_result = exporter.write_to_filesystem(state, f"/tmp/test_{file_path.stem}.glb")
                result.export_glb_success = export_result == 0
                if not result.export_glb_success:
                    result.export_glb_error = f"Export failed with code: {export_result}"
            except Exception as e:
                result.export_glb_success = False
                result.export_glb_error = str(e)

            result.export_time = time.time() - export_start

        except Exception as e:
            result.load_success = False
            result.load_error = f"Unexpected error: {str(e)}"

        return result

    def run_compatibility_tests(self, max_files: int = None) -> Dict[str, Any]:
        """Run compatibility tests on all discovered GLTF files"""
        self.test_start_time = time.time()
        self.results = []

        gltf_files = self.discover_gltf_files()

        if not gltf_files:
            return self._generate_report()

        print(f"🔍 Discovered {len(gltf_files)} GLTF files to test")
        print("=" * 80)

        tested_count = 0
        for file_path, variant in gltf_files:
            if max_files and tested_count >= max_files:
                break

            print(f"🧪 Testing: {file_path.name} ({variant})")
            result = self.test_single_file(file_path, variant)
            self.results.append(result)

            # Show result
            status_emoji = {
                TestResult.PASS: "✅",
                TestResult.FAIL: "❌",
                TestResult.SKIP: "⏭️",
                TestResult.ERROR: "💥"
            }[result.overall_result]

            print(f"   {status_emoji} {result.overall_result.value}")
            if result.overall_result != TestResult.PASS:
                if not result.load_success:
                    print(f"      Load failed: {result.load_error}")
                elif not result.scene_generation_success:
                    print(f"      Scene generation failed: {result.scene_generation_error}")
                elif not (result.export_gltf_success or result.export_glb_success):
                    print(f"      Export failed: GLTF={result.export_gltf_error}, GLB={result.export_glb_error}")

            tested_count += 1

        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive compatibility report"""
        total_time = time.time() - self.test_start_time
        total_files = len(self.results)

        if total_files == 0:
            return {
                'summary': {
                    'total_files': 0,
                    'pass_count': 0,
                    'fail_count': 0,
                    'skip_count': 0,
                    'error_count': 0,
                    'pass_rate': 0.0,
                    'total_time': total_time
                },
                'results': [],
                'variant_breakdown': {},
                'feature_support': {}
            }

        # Calculate statistics
        pass_count = sum(1 for r in self.results if r.overall_result == TestResult.PASS)
        fail_count = sum(1 for r in self.results if r.overall_result == TestResult.FAIL)
        skip_count = sum(1 for r in self.results if r.overall_result == TestResult.SKIP)
        error_count = sum(1 for r in self.results if r.overall_result == TestResult.ERROR)

        pass_rate = (pass_count / total_files) * 100.0

        # Variant breakdown
        variant_stats = {}
        for result in self.results:
            variant = result.variant
            if variant not in variant_stats:
                variant_stats[variant] = {'total': 0, 'pass': 0, 'fail': 0}
            variant_stats[variant]['total'] += 1
            if result.overall_result == TestResult.PASS:
                variant_stats[variant]['pass'] += 1
            else:
                variant_stats[variant]['fail'] += 1

        # Feature support analysis
        feature_support = {
            'meshes': sum(1 for r in self.results if r.mesh_count > 0),
            'materials': sum(1 for r in self.results if r.material_count > 0),
            'textures': sum(1 for r in self.results if r.texture_count > 0),
            'animations': sum(1 for r in self.results if r.animation_count > 0),
            'skins': sum(1 for r in self.results if r.skin_count > 0),
            'lights': sum(1 for r in self.results if r.light_count > 0),
            'cameras': sum(1 for r in self.results if r.camera_count > 0),
        }

        # Performance statistics
        load_times = [r.load_time for r in self.results if r.load_success]
        scene_times = [r.scene_time for r in self.results if r.scene_generation_success]
        export_times = [r.export_time for r in self.results if r.export_gltf_success or r.export_glb_success]

        report = {
            'summary': {
                'total_files': total_files,
                'pass_count': pass_count,
                'fail_count': fail_count,
                'skip_count': skip_count,
                'error_count': error_count,
                'pass_rate': round(pass_rate, 2),
                'total_time': round(total_time, 2),
                'avg_load_time': round(sum(load_times) / len(load_times), 3) if load_times else 0,
                'avg_scene_time': round(sum(scene_times) / len(scene_times), 3) if scene_times else 0,
                'avg_export_time': round(sum(export_times) / len(export_times), 3) if export_times else 0,
            },
            'results': [r.to_dict() for r in self.results],
            'variant_breakdown': variant_stats,
            'feature_support': feature_support,
            'failed_tests': [r.to_dict() for r in self.results if r.overall_result != TestResult.PASS]
        }

        return report

    def save_report(self, report: Dict[str, Any], output_file: str = "gltf_compatibility_report.json"):
        """Save compatibility report to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to: {output_file}")

    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable test summary"""
        summary = report['summary']

        print("\n" + "=" * 80)
        print("🎯 GLTF COMPATIBILITY TEST SUMMARY")
        print("=" * 80)

        print(f"📊 Total Files Tested: {summary['total_files']}")
        print(f"✅ Passed: {summary['pass_count']} ({summary['pass_rate']}%)")
        print(f"❌ Failed: {summary['fail_count']}")
        print(f"⏭️ Skipped: {summary['skip_count']}")
        print(f"💥 Errors: {summary['error_count']}")

        print(f"\n⏱️ Performance:")
        print(f"   Total Time: {summary['total_time']}s")
        print(f"   Avg Load Time: {summary['avg_load_time']}s")
        print(f"   Avg Scene Gen: {summary['avg_scene_time']}s")
        print(f"   Avg Export: {summary['avg_export_time']}s")

        print(f"\n🔧 Feature Support:")
        features = report['feature_support']
        for feature, count in features.items():
            percentage = (count / summary['total_files']) * 100 if summary['total_files'] > 0 else 0
            print(f"   {feature.capitalize()}: {count} files ({percentage:.1f}%)")

        if report['failed_tests']:
            print(f"\n❌ Failed Tests ({len(report['failed_tests'])}):")
            for failed in report['failed_tests'][:10]:  # Show first 10 failures
                print(f"   • {failed['file_name']} ({failed['variant']})")
            if len(report['failed_tests']) > 10:
                print(f"   ... and {len(report['failed_tests']) - 10} more")


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="GLTF Compatibility Test Suite")
    parser.add_argument('--assets-dir', default='thirdparty/glTF-Sample-Assets/Models',
                       help='Directory containing GLTF sample assets')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to test')
    parser.add_argument('--output', default='gltf_compatibility_report.json',
                       help='Output report file')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')

    args = parser.parse_args()

    # Run compatibility tests
    tester = GLTFCompatibilityTest(args.assets_dir)
    report = tester.run_compatibility_tests(args.max_files)

    # Save and display results
    tester.save_report(report, args.output)
    tester.print_summary(report)

    # Exit with appropriate code
    pass_rate = report['summary']['pass_rate']
    if pass_rate >= 95.0:
        print("🎉 Excellent compatibility! Almost all tests passed.")
        sys.exit(0)
    elif pass_rate >= 80.0:
        print("👍 Good compatibility! Most tests passed.")
        sys.exit(0)
    else:
        print("⚠️ Compatibility issues detected. Review failed tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()
