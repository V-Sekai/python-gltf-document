#!/usr/bin/env python3
"""
Comprehensive test script for GLTF module using sample assets

This script tests the GLTF module against the extensive collection of
GLTF-Sample-Assets to verify functionality, performance, and compatibility.
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import psutil
import gc

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from gltf_module import (
    GLTFDocument, GLTFState, is_torch_available, get_torch_device,
    GLTFAccessorDecoder, TorchGLTFAccessorDecoder
)
from gltf_module.scene_generator import GLTFSceneGenerator


@dataclass
class TestResult:
    """Result of a single test case"""
    model_name: str
    variant: str
    success: bool
    load_time: float
    decode_time: float
    scene_gen_time: float
    memory_usage: int
    error_message: Optional[str] = None
    warnings: List[str] = None
    stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.stats is None:
            self.stats = {}


@dataclass
class TestReport:
    """Comprehensive test report"""
    total_models: int
    successful_loads: int
    successful_decodes: int
    successful_scene_gen: int
    total_time: float
    average_load_time: float
    average_decode_time: float
    average_scene_gen_time: float
    peak_memory_usage: int
    results: List[TestResult]
    errors: List[str]
    torch_available: bool
    torch_tests: int
    torch_successes: int


class GLTFAssetValidator:
    """Comprehensive validator for GLTF assets"""

    def __init__(self, assets_dir: str = "thirdparty/glTF-Sample-Assets/Models"):
        self.assets_dir = Path(assets_dir)
        self.model_index_path = self.assets_dir / "model-index.json"
        self.results: List[TestResult] = []
        self.errors: List[str] = []
        self.torch_available = is_torch_available()

        if self.torch_available:
            self.device = get_torch_device('auto')
        else:
            self.device = None

        # Load model index
        with open(self.model_index_path, 'r', encoding='utf-8') as f:
            self.model_index = json.load(f)

    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def load_model_index(self) -> List[Dict]:
        """Load and return the model index"""
        return self.model_index

    def find_model_files(self, model_name: str, variant: str) -> List[Path]:
        """Find all files for a specific model variant"""
        model_dir = self.assets_dir / model_name
        if not model_dir.exists():
            return []

        files = []
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                files.append(file_path)
        return files

    def test_basic_loading(self, model_data: Dict, variant: str) -> TestResult:
        """Test basic GLTF loading functionality"""
        model_name = model_data['name']
        model_dir = self.assets_dir / model_name

        start_time = time.time()
        initial_memory = self.get_memory_usage()

        result = TestResult(
            model_name=model_name,
            variant=variant,
            success=False,
            load_time=0.0,
            decode_time=0.0,
            scene_gen_time=0.0,
            memory_usage=0
        )

        try:
            # Find the GLTF file for this variant
            gltf_file = None
            if variant in model_data.get('variants', {}):
                variant_file = model_data['variants'][variant]
                gltf_path = model_dir / "glTF" / variant_file
                if gltf_path.exists():
                    gltf_file = gltf_path
                else:
                    # Try glTF-Binary format
                    glb_path = model_dir / "glTF-Binary" / variant_file.replace('.gltf', '.glb')
                    if glb_path.exists():
                        gltf_file = glb_path

            if not gltf_file:
                result.error_message = f"No GLTF file found for variant {variant}"
                return result

            # Load GLTF document
            doc = GLTFDocument()
            success = doc.load_from_file(str(gltf_file))

            if not success:
                result.error_message = "Failed to load GLTF document"
                return result

            load_time = time.time() - start_time
            result.load_time = load_time
            result.success = True

            # Collect basic statistics
            result.stats.update({
                'nodes': len(doc.state.nodes),
                'meshes': len(doc.state.meshes),
                'materials': len(doc.state.materials),
                'textures': len(doc.state.textures),
                'animations': len(doc.state.animations),
                'accessors': len(doc.state.accessors),
                'buffers': len(doc.state.buffers),
                'file_size': gltf_file.stat().st_size
            })

            # Check for extensions
            extensions = set()
            if hasattr(doc.state, 'extensions_used'):
                extensions.update(doc.state.extensions_used)
            if hasattr(doc.state, 'extensions_required'):
                extensions.update(doc.state.extensions_required)

            result.stats['extensions'] = list(extensions)

            # Memory usage
            result.memory_usage = self.get_memory_usage() - initial_memory

        except Exception as e:
            result.error_message = f"Exception during loading: {str(e)}"
            result.load_time = time.time() - start_time

        return result

    def test_accessor_decoding(self, result: TestResult) -> None:
        """Test accessor decoding for a loaded model"""
        if not result.success:
            return

        try:
            start_time = time.time()

            # Create CPU decoder
            cpu_decoder = GLTFAccessorDecoder()

            # Test decoding key accessors
            doc = GLTFDocument()
            gltf_file = self._find_gltf_file(result.model_name, result.variant)
            if not gltf_file:
                return

            doc.load_from_file(str(gltf_file))

            # Find POSITION accessor
            position_accessor = None
            for mesh in doc.state.meshes:
                for primitive in mesh.primitives:
                    if 'POSITION' in primitive.attributes:
                        position_accessor = primitive.attributes['POSITION']
                        break
                if position_accessor is not None:
                    break

            if position_accessor is not None:
                # Test CPU decoding
                positions = cpu_decoder.decode_accessor_as_vec3(doc.state, position_accessor)
                result.stats['cpu_decoded_vertices'] = len(positions) if positions else 0

                # Test PyTorch decoding if available
                if self.torch_available:
                    torch_decoder = TorchGLTFAccessorDecoder(device=self.device)
                    torch_positions = torch_decoder.decode_accessor_tensor(doc.state, position_accessor)
                    result.stats['torch_decoded_vertices'] = torch_positions.shape[0] if torch_positions.numel() > 0 else 0

            result.decode_time = time.time() - start_time

        except Exception as e:
            result.warnings.append(f"Accessor decoding failed: {str(e)}")
            result.decode_time = time.time() - start_time

    def test_scene_generation(self, result: TestResult) -> None:
        """Test scene generation for a loaded model"""
        if not result.success:
            return

        try:
            start_time = time.time()

            # Load document
            doc = GLTFDocument()
            gltf_file = self._find_gltf_file(result.model_name, result.variant)
            if not gltf_file:
                return

            doc.load_from_file(str(gltf_file))

            # Generate scene
            generator = GLTFSceneGenerator(doc.state)
            scene_nodes = generator.generate_scene()

            result.scene_gen_time = time.time() - start_time
            result.stats['scene_nodes'] = len(scene_nodes)

            # Count total meshes in scene
            total_meshes = 0
            total_vertices = 0
            total_faces = 0

            for node in scene_nodes:
                if node.mesh_data:
                    total_meshes += 1
                    total_vertices += node.mesh_data.num_vertices
                    total_faces += node.mesh_data.num_faces

            result.stats.update({
                'total_scene_meshes': total_meshes,
                'total_scene_vertices': total_vertices,
                'total_scene_faces': total_faces
            })

        except Exception as e:
            result.warnings.append(f"Scene generation failed: {str(e)}")
            result.scene_gen_time = time.time() - start_time

    def _find_gltf_file(self, model_name: str, variant: str) -> Optional[Path]:
        """Find GLTF file for a model variant"""
        model_dir = self.assets_dir / model_name

        for model_data in self.model_index:
            if model_data['name'] == model_name:
                if variant in model_data.get('variants', {}):
                    variant_file = model_data['variants'][variant]
                    gltf_path = model_dir / "glTF" / variant_file
                    if gltf_path.exists():
                        return gltf_path
                    # Try glTF-Binary
                    glb_path = model_dir / "glTF-Binary" / variant_file.replace('.gltf', '.glb')
                    if glb_path.exists():
                        return glb_path
        return None

    def run_comprehensive_test(self, max_models: int = None, variants_to_test: List[str] = None) -> TestReport:
        """Run comprehensive tests on all models"""
        if variants_to_test is None:
            variants_to_test = ['glTF', 'glTF-Binary']

        start_time = time.time()
        peak_memory = 0

        models_tested = 0

        for model_data in self.model_index:
            if max_models and models_tested >= max_models:
                break

            model_name = model_data['name']
            print(f"\nTesting model: {model_name}")

            for variant in variants_to_test:
                if variant not in model_data.get('variants', {}):
                    continue

                print(f"  Testing variant: {variant}")

                # Test basic loading
                result = self.test_basic_loading(model_data, variant)

                # Test accessor decoding
                self.test_accessor_decoding(result)

                # Test scene generation
                self.test_scene_generation(result)

                self.results.append(result)

                # Track peak memory
                current_memory = self.get_memory_usage()
                peak_memory = max(peak_memory, current_memory)

                # Force garbage collection between tests
                gc.collect()

                models_tested += 1

                # Progress indicator
                successful = sum(1 for r in self.results if r.success)
                print(f"    Result: {'✓' if result.success else '✗'} "
                      f"(Total: {len(self.results)}, Success: {successful})")

        total_time = time.time() - start_time

        # Calculate statistics
        successful_loads = sum(1 for r in self.results if r.success)
        successful_decodes = sum(1 for r in self.results if r.decode_time > 0)
        successful_scene_gen = sum(1 for r in self.results if r.scene_gen_time > 0)

        torch_tests = sum(1 for r in self.results if 'torch_decoded_vertices' in r.stats)
        torch_successes = sum(1 for r in self.results
                             if 'torch_decoded_vertices' in r.stats and r.stats['torch_decoded_vertices'] > 0)

        avg_load_time = sum(r.load_time for r in self.results) / len(self.results) if self.results else 0
        avg_decode_time = sum(r.decode_time for r in self.results if r.decode_time > 0) / successful_decodes if successful_decodes > 0 else 0
        avg_scene_gen_time = sum(r.scene_gen_time for r in self.results if r.scene_gen_time > 0) / successful_scene_gen if successful_scene_gen > 0 else 0

        return TestReport(
            total_models=len(self.results),
            successful_loads=successful_loads,
            successful_decodes=successful_decodes,
            successful_scene_gen=successful_scene_gen,
            total_time=total_time,
            average_load_time=avg_load_time,
            average_decode_time=avg_decode_time,
            average_scene_gen_time=avg_scene_gen_time,
            peak_memory_usage=peak_memory,
            results=self.results,
            errors=self.errors,
            torch_available=self.torch_available,
            torch_tests=torch_tests,
            torch_successes=torch_successes
        )

    def generate_report(self, report: TestReport) -> str:
        """Generate a detailed test report"""
        lines = []
        lines.append("=" * 80)
        lines.append("GLTF MODULE COMPREHENSIVE TEST REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append("SUMMARY:")
        lines.append(f"  Total Models Tested: {report.total_models}")
        lines.append(f"  Successful Loads: {report.successful_loads} ({report.successful_loads/report.total_models*100:.1f}%)")
        lines.append(f"  Successful Decodes: {report.successful_decodes} ({report.successful_decodes/report.total_models*100:.1f}%)")
        lines.append(f"  Successful Scene Gen: {report.successful_scene_gen} ({report.successful_scene_gen/report.total_models*100:.1f}%)")
        lines.append(f"  Total Test Time: {report.total_time:.2f}s")
        lines.append(f"  Average Load Time: {report.average_load_time:.3f}s")
        lines.append(f"  Average Decode Time: {report.average_decode_time:.3f}s")
        lines.append(f"  Average Scene Gen Time: {report.average_scene_gen_time:.3f}s")
        lines.append(f"  Peak Memory Usage: {report.peak_memory_usage / 1024 / 1024:.1f} MB")
        lines.append("")

        lines.append("PYTORCH SUPPORT:")
        lines.append(f"  PyTorch Available: {report.torch_available}")
        if report.torch_available:
            lines.append(f"  PyTorch Tests: {report.torch_tests}")
            lines.append(f"  PyTorch Successes: {report.torch_successes} ({report.torch_successes/report.torch_tests*100:.1f}%)")
        lines.append("")

        # Extension usage statistics
        all_extensions = set()
        for result in report.results:
            if result.success and 'extensions' in result.stats:
                all_extensions.update(result.stats['extensions'])

        if all_extensions:
            lines.append("EXTENSIONS FOUND:")
            for ext in sorted(all_extensions):
                count = sum(1 for r in report.results
                          if r.success and ext in r.stats.get('extensions', []))
                lines.append(f"  {ext}: {count} models")
            lines.append("")

        # Model size statistics
        if report.results:
            file_sizes = [r.stats.get('file_size', 0) for r in report.results if r.success]
            if file_sizes:
                lines.append("MODEL SIZE STATISTICS:")
                lines.append(f"  Average File Size: {sum(file_sizes)/len(file_sizes)/1024:.1f} KB")
                lines.append(f"  Min File Size: {min(file_sizes)/1024:.1f} KB")
                lines.append(f"  Max File Size: {max(file_sizes)/1024:.1f} KB")
                lines.append("")

        # Failed models
        failed_results = [r for r in report.results if not r.success]
        if failed_results:
            lines.append("FAILED MODELS:")
            for result in failed_results[:20]:  # Show first 20 failures
                lines.append(f"  {result.model_name} ({result.variant}): {result.error_message}")
            if len(failed_results) > 20:
                lines.append(f"  ... and {len(failed_results) - 20} more")
            lines.append("")

        # Performance outliers
        if report.results:
            slow_loads = sorted([r for r in report.results if r.load_time > 1.0],
                              key=lambda r: r.load_time, reverse=True)[:5]
            if slow_loads:
                lines.append("SLOWEST LOADING MODELS:")
                for result in slow_loads:
                    lines.append(f"  {result.model_name}: {result.load_time:.2f}s")
                lines.append("")

        return "\n".join(lines)


def main():
    """Main test function"""
    print("GLTF Module Comprehensive Testing")
    print("=" * 40)

    # Check if assets directory exists
    assets_dir = "thirdparty/glTF-Sample-Assets/Models"
    if not os.path.exists(assets_dir):
        print(f"Error: Assets directory {assets_dir} not found!")
        print("Please ensure GLTF-Sample-Assets is properly set up.")
        return 1

    # Create validator
    validator = GLTFAssetValidator(assets_dir)

    print(f"PyTorch available: {validator.torch_available}")
    if validator.torch_available:
        print(f"Using device: {validator.device}")
    print()

    # Run tests (limit to first 20 models for quick testing, remove limit for full test)
    max_models = 20  # Set to None for full test suite
    print(f"Running comprehensive tests on {'all' if max_models is None else max_models} models...")

    try:
        report = validator.run_comprehensive_test(max_models=max_models)

        # Generate and print report
        report_text = validator.generate_report(report)
        print("\n" + report_text)

        # Save detailed results to file
        with open("test_results.json", "w") as f:
            json.dump({
                'summary': {
                    'total_models': report.total_models,
                    'successful_loads': report.successful_loads,
                    'successful_decodes': report.successful_decodes,
                    'successful_scene_gen': report.successful_scene_gen,
                    'total_time': report.total_time,
                    'torch_available': report.torch_available,
                    'torch_tests': report.torch_tests,
                    'torch_successes': report.torch_successes
                },
                'results': [
                    {
                        'model_name': r.model_name,
                        'variant': r.variant,
                        'success': r.success,
                        'load_time': r.load_time,
                        'decode_time': r.decode_time,
                        'scene_gen_time': r.scene_gen_time,
                        'memory_usage': r.memory_usage,
                        'error_message': r.error_message,
                        'warnings': r.warnings,
                        'stats': r.stats
                    } for r in report.results
                ]
            }, f, indent=2)

        print("Detailed results saved to test_results.json")

        # Return success/failure based on test results
        success_rate = report.successful_loads / report.total_models if report.total_models > 0 else 0
        return 0 if success_rate >= 0.8 else 1  # 80% success threshold

    except Exception as e:
        print(f"Test suite failed with exception: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
