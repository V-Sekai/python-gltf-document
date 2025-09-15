#!/usr/bin/env just --justfile
# GPU GLTF Pipeline - Development Commands
# Run `just --list` to see all available commands

# Default recipe (run when you just type `just`)
default: test

# =============================================================================
# SETUP & ENVIRONMENT
# =============================================================================

# Set up development environment
setup:
    #!/usr/bin/env bash
    echo "🚀 Setting up GPU GLTF Pipeline development environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate and install dependencies
    echo "📦 Installing dependencies..."
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -r requirements-test.txt

    # Install development dependencies
    pip install black isort flake8 mypy pytest pytest-cov

    echo "✅ Setup complete! Run 'just test' to verify everything works."

# Update dependencies
update-deps:
    #!/usr/bin/env bash
    echo "📦 Updating dependencies..."
    source .venv/bin/activate
    pip install --upgrade -r requirements.txt
    pip install --upgrade -r requirements-test.txt

# =============================================================================
# TEST FIXTURES
# =============================================================================

# Generate all test fixtures
fixtures: fixture-cube fixture-convert

# Generate cube test fixture
fixture-cube:
    #!/usr/bin/env bash
    echo "📦 Generating cube test fixtures..."
    mkdir -p test_fixtures
    python3 scripts/generate_cube_fixture.py

# Convert binary to base64
convert-bin-to-base64:
    #!/usr/bin/env bash
    if [ -f "test_fixtures/test_cube.bin" ]; then
        base64 test_fixtures/test_cube.bin > test_fixtures/test_cube.base64
        echo "✅ Converted test_cube.bin to test_cube.base64"
    else
        echo "❌ test_cube.bin not found"
        exit 1
    fi

# Convert base64 to binary
convert-base64-to-bin:
    #!/usr/bin/env bash
    if [ -f "test_fixtures/test_cube.base64" ]; then
        base64 -d test_fixtures/test_cube.base64 > test_fixtures/test_cube_converted.bin
        echo "✅ Converted test_cube.base64 to test_cube_converted.bin"
    else
        echo "❌ test_cube.base64 not found"
        exit 1
    fi

# Generate embedded GLTF with base64
fixture-convert:
    #!/usr/bin/env bash
    echo "📦 Generating embedded GLTF fixture..."
    python3 scripts/generate_cube_fixture.py --embedded-only

# Verify fixture conversion works
verify-fixtures:
    #!/usr/bin/env bash
    echo "🔍 Verifying test fixtures..."

    # Check files exist
    for file in test_fixtures/test_cube.gltf test_fixtures/test_cube.bin test_fixtures/test_cube.base64 test_fixtures/test_cube_embedded.gltf; do
        if [ -f "$file" ]; then
            echo "✅ $file exists"
        else
            echo "❌ $file missing"
        fi
    done

    # Test base64 conversion
    echo "Testing base64 conversion..."
    base64 -d test_fixtures/test_cube.base64 > /tmp/test_cube_verify.bin
    if cmp test_fixtures/test_cube.bin /tmp/test_cube_verify.bin; then
        echo "✅ Base64 conversion works correctly"
    else
        echo "❌ Base64 conversion failed"
    fi
    rm -f /tmp/test_cube_verify.bin

# =============================================================================
# TESTING
# =============================================================================

# Run all tests
test: test-unit test-integration test-gpu

# Run unit tests
test-unit:
    #!/usr/bin/env bash
    echo "🧪 Running unit tests..."
    source .venv/bin/activate
    python -m pytest tests/ -v --tb=short

# Run integration tests
test-integration:
    #!/usr/bin/env bash
    echo "🔗 Running integration tests..."
    source .venv/bin/activate
    python test_comprehensive.py

# Run GPU pipeline tests
test-gpu:
    #!/usr/bin/env bash
    echo "🚀 Running GPU pipeline tests..."
    source .venv/bin/activate
    python test_gpu_pipeline.py

# Run PTE model tests
test-pte:
    #!/usr/bin/env bash
    echo "⚡ Running ExecuTorch PTE tests..."
    source .venv/bin/activate
    python test_pte_models.py

# Run all tests with coverage
test-cov:
    #!/usr/bin/env bash
    echo "📊 Running tests with coverage..."
    source .venv/bin/activate
    python -m pytest --cov=gltf_module --cov-report=html --cov-report=term-missing

# =============================================================================
# GPU PIPELINE
# =============================================================================

# Run GPU pipeline on test file
gpu-process FILE="test_fixtures/test_cube.gltf":
    #!/usr/bin/env bash
    echo "🚀 Processing {{FILE}} with GPU pipeline..."
    source .venv/bin/activate
    python scripts/gpu_process.py "{{FILE}}" "output_processed.gltf" "transform,normals"

# Benchmark GPU pipeline
gpu-benchmark:
    #!/usr/bin/env bash
    echo "📊 Benchmarking GPU pipeline..."
    source .venv/bin/activate
    python scripts/gpu_benchmark.py

# =============================================================================
# CODE QUALITY
# =============================================================================

# Format code
format:
    #!/usr/bin/env bash
    echo "🎨 Formatting code..."
    source .venv/bin/activate
    black gltf_module/ tests/ *.py
    isort gltf_module/ tests/ *.py

# Lint code
lint:
    #!/usr/bin/env bash
    echo "🔍 Linting code..."
    source .venv/bin/activate
    flake8 gltf_module/ tests/ *.py --max-line-length=100

# Type check
type-check:
    #!/usr/bin/env bash
    echo "📝 Type checking..."
    source .venv/bin/activate
    mypy gltf_module/ --ignore-missing-imports

# Run all quality checks
quality: format lint type-check

# =============================================================================
# UTILITIES
# =============================================================================

# Clean build artifacts
clean:
    #!/usr/bin/env bash
    echo "🧹 Cleaning build artifacts..."
    rm -rf __pycache__/ */__pycache__/ *.pyc */*.pyc
    rm -rf .pytest_cache/ .coverage htmlcov/
    rm -rf test_models/ output_*.gltf benchmark_output.gltf
    rm -f test_fixtures/test_cube_converted.bin temp_buffer.json

# Clean everything including fixtures
clean-all: clean
    #!/usr/bin/env bash
    echo "🧹 Cleaning all artifacts including fixtures..."
    rm -rf test_fixtures/ .venv/

# Show project info
info:
    #!/usr/bin/env bash
    echo "📋 GPU GLTF Pipeline Project Info"
    echo "=================================="
    echo "Python version: $(python3 --version)"
    echo "Project root: $(pwd)"
    echo ""
    echo "Key directories:"
    echo "  gltf_module/     - Main GLTF processing modules"
    echo "  tests/          - Unit tests"
    echo "  test_fixtures/  - Test data files"
    echo "  test_models/    - Generated PTE models"
    echo ""
    echo "Available commands:"
    just --list

# Help
help:
    #!/usr/bin/env bash
    echo "🤖 GPU GLTF Pipeline - Available Commands"
    echo "=========================================="
    echo ""
    echo "Setup & Environment:"
    echo "  just setup          - Set up development environment"
    echo "  just update-deps    - Update Python dependencies"
    echo ""
    echo "Test Fixtures:"
    echo "  just fixtures       - Generate all test fixtures"
    echo "  just fixture-cube   - Generate cube test fixture"
    echo "  just verify-fixtures - Verify fixture integrity"
    echo ""
    echo "Testing:"
    echo "  just test           - Run all tests"
    echo "  just test-unit      - Run unit tests"
    echo "  just test-integration - Run integration tests"
    echo "  just test-gpu       - Run GPU pipeline tests"
    echo "  just test-pte       - Run ExecuTorch PTE tests"
    echo "  just test-cov       - Run tests with coverage"
    echo ""
    echo "GPU Pipeline:"
    echo "  just gpu-process FILE - Process GLTF file with GPU pipeline"
    echo "  just gpu-benchmark    - Benchmark GPU pipeline performance"
    echo ""
    echo "Code Quality:"
    echo "  just format          - Format code with black"
    echo "  just lint            - Lint code with flake8"
    echo "  just type-check      - Type check with mypy"
    echo "  just quality         - Run all quality checks"
    echo ""
    echo "Utilities:"
    echo "  just clean           - Clean build artifacts"
    echo "  just clean-all       - Clean everything"
    echo "  just info            - Show project information"
    echo "  just help            - Show this help message"
