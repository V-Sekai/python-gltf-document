#!/usr/bin/env python3
"""
GPU Benchmark Script

Benchmark the GPU GLTF pipeline performance.
Used by the justfile to avoid syntax issues with inline Python code.
"""

import time
from gltf_module.gpu_pipeline import GPUGLTFPipeline

def main():
    print("📊 Benchmarking GPU GLTF Pipeline")
    print("=" * 40)

    try:
        # Create GPU pipeline
        pipeline = GPUGLTFPipeline()

        # Benchmark processing
        start_time = time.time()
        success = pipeline.process_gltf_file_gpu(
            'test_fixtures/test_cube.gltf',
            'benchmark_output.gltf'
        )
        end_time = time.time()

        if success:
            processing_time = end_time - start_time
            print(".3f"
            # Get performance stats
            stats = pipeline.get_performance_stats()
            print("\n📈 Performance Statistics:")
            print(f"  Load Times: {stats.get('avg_load_times', 'N/A')}")
            print(f"  Process Times: {stats.get('avg_process_times', 'N/A')}")
            print(f"  Save Times: {stats.get('avg_save_times', 'N/A')}")
            print(f"  Total Operations: {stats.get('total_operations', 0)}")

            gpu_stats = stats.get('gpu_memory', {})
            if gpu_stats:
                print("
🎮 GPU Memory:"                print(f"  Allocated: {gpu_stats.get('allocated_mb', 0):.1f} MB")
                print(f"  Reserved: {gpu_stats.get('reserved_mb', 0):.1f} MB")
                print(f"  Utilization: {gpu_stats.get('utilization_percent', 0):.1f}%")

            print("\n✅ Benchmark completed successfully!")
        else:
            print("❌ GPU processing failed during benchmark")
            return 1

    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
