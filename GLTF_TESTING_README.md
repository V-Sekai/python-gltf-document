# GLTF Compatibility Testing with Slash Framework

This document describes how to use the Slash testing framework to validate GLTF compatibility across the entire glTF-Sample-Assets collection.

## Overview

The GLTF module now includes comprehensive testing using the [Slash testing framework](https://slash.readthedocs.io/en/master/), which provides:

- **Parametrized testing** across all GLTF sample files
- **Detailed reporting** and test session management
- **Fixture-based testing** with proper resource management
- **Session-level hooks** for setup/teardown and reporting

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements-test.txt
```

Or install Slash directly:

```bash
pip install slash
```

### 2. Verify Installation

```bash
slash --version
```

## Test Structure

### Test Files

- **`slash_gltf_tests.py`** - Main Slash test suite
- **`comprehensive_compatibility_test.py`** - Alternative comprehensive test runner
- **`run_gltf_tests.py`** - Convenience test runner script

### Test Categories

1. **`test_gltf_loading`** - Tests GLTF/GLB file loading
2. **`test_gltf_scene_generation`** - Tests scene hierarchy generation
3. **`test_gltf_export_gltf`** - Tests export to .gltf format
4. **`test_gltf_export_glb`** - Tests export to .glb format
5. **`test_gltf_round_trip`** - Tests complete load → export → reload cycle
6. **`test_gltf_validation_basic`** - Tests basic GLTF structure validation
7. **`test_gltf_feature_detection`** - Tests feature detection and reporting

## Running Tests

### Basic Usage

```bash
# Run all tests
slash run slash_gltf_tests.py

# Run with verbose output
slash run slash_gltf_tests.py -v

# Run specific test
slash run slash_gltf_tests.py::test_gltf_loading

# Run tests for specific file patterns
slash run slash_gltf_tests.py -k "loading"
```

### Using the Convenience Script

```bash
# Install dependencies and run Slash tests
python run_gltf_tests.py --install-deps --framework slash

# Run comprehensive tests (limited files)
python run_gltf_tests.py --framework comprehensive --max-files 10

# Run basic round-trip test
python run_gltf_tests.py --framework roundtrip
```

### Limiting Test Scope

```bash
# Test only first 5 files
slash run slash_gltf_tests.py --max-files 5

# Test specific variants
slash run slash_gltf_tests.py -k "Box"  # Test Box-related files
```

## Test Fixtures

### Available Fixtures

- **`gltf_document`** - Fresh GLTFDocument instance
- **`gltf_state`** - Fresh GLTFState instance
- **`gltf_exporter`** - Fresh GLTFExporter instance
- **`temp_output_dir`** - Temporary directory for test outputs

### Session Management

- **`session_start()`** - Executed at test session start
- **`session_end()`** - Executed at test session end with summary reporting

## Test Discovery

The test suite automatically discovers GLTF files from:

```
thirdparty/glTF-Sample-Assets/Models/
```

It scans all subdirectories and identifies:
- `.gltf` files (JSON format)
- `.glb` files (binary format)

Each discovered file becomes a test parameter with metadata:
- File path and name
- Variant (parent directory name)
- Format type (GLTF/GLB)
- Relative path

## Test Results and Reporting

### Console Output

Tests provide real-time feedback:
```
🧪 Testing: ABeautifulGame.gltf (ABeautifulGame)
   ✅ PASS
🧪 Testing: Box.gltf (Box)
   ✅ PASS
```

### Session Summary

At session end, Slash provides:
- Test counts (passed/failed/total)
- Execution time
- Feature detection summary
- Variant coverage report

### Detailed Reporting

```bash
# Generate HTML report
slash run slash_gltf_tests.py --html-report report.html

# Generate JUnit XML
slash run slash_gltf_tests.py --junit-xml results.xml
```

## Test Configuration

### Environment Variables

```bash
# Set log level
export SLASH_LOG_LEVEL=INFO

# Custom assets directory
export GLTF_ASSETS_DIR=/path/to/custom/assets
```

### Command Line Options

```bash
slash run slash_gltf_tests.py \
  --verbose \
  --parallel 4 \
  --max-files 50 \
  --html-report gltf_test_report.html
```

## Advanced Usage

### Custom Test Selection

```python
# In your test file
@slash.parametrize('gltf_file', discover_gltf_files())
def test_custom_validation(gltf_document, gltf_state, gltf_file):
    # Your custom validation logic
    pass
```

### Adding New Test Categories

```python
@slash.parametrize('gltf_file', discover_gltf_files())
def test_performance(gltf_document, gltf_state, gltf_file):
    """Test GLTF loading performance"""
    import time

    start_time = time.time()
    result = gltf_document.append_from_file(gltf_file['file_path'], gltf_state)
    load_time = time.time() - start_time

    slash.assert_less(load_time, 5.0, "Loading should complete within 5 seconds")
```

### Integration with CI/CD

```yaml
# .github/workflows/test.yml
- name: Run GLTF Compatibility Tests
  run: |
    pip install -r requirements-test.txt
    slash run slash_gltf_tests.py --junit-xml test-results.xml
```

## Troubleshooting

### Common Issues

1. **Assets directory not found**
   ```
   Solution: Ensure thirdparty/glTF-Sample-Assets/Models/ exists
   ```

2. **Buffer loading errors**
   ```
   Solution: Check that external buffer files are present
   ```

3. **Memory issues with large files**
   ```
   Solution: Use --max-files to limit test scope
   ```

### Debug Mode

```bash
# Run with debug logging
SLASH_LOG_LEVEL=DEBUG slash run slash_gltf_tests.py -v

# Run single failing test
slash run slash_gltf_tests.py::test_gltf_loading -k "Box"
```

## Performance Considerations

- **Large test suites**: Use `--max-files` to limit scope
- **Parallel execution**: Use `--parallel` for faster runs
- **Resource cleanup**: Temporary files are automatically cleaned up

## Integration with Your Testing Framework

The Slash tests can be integrated with your existing test infrastructure:

```python
# Import and extend
from slash_gltf_tests import discover_gltf_files, test_gltf_loading

# Create your own test suite
@slash.parametrize('gltf_file', discover_gltf_files())
def test_your_custom_validation(gltf_document, gltf_state, gltf_file):
    # Your validation logic here
    pass
```

## Results Interpretation

### Success Criteria

- ✅ **PASS**: File loads, processes, and exports successfully
- ❌ **FAIL**: Any step in the pipeline fails
- ⏭️ **SKIP**: Test intentionally skipped

### Feature Coverage

The test suite reports feature usage across the asset collection:
- Meshes, materials, textures
- Animations, skins, lights, cameras
- Extensions and advanced features

This helps identify which GLTF features are well-supported vs. those needing improvement.

## Contributing

When adding new tests:

1. Follow Slash conventions
2. Use appropriate fixtures
3. Include proper assertions
4. Add documentation strings
5. Test with multiple GLTF variants

## See Also

- [Slash Documentation](https://slash.readthedocs.io/en/master/)
- [GLTF 2.0 Specification](https://www.khronos.org/gltf/)
- [glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
