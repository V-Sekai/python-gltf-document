# Godot GLTF Module - Python Implementation

A comprehensive Python implementation of Godot's GLTF module, providing GLTF 2.0 file parsing and scene generation capabilities with optional PyTorch acceleration.

## Features

- **Complete GLTF 2.0 Support**: Parse both JSON (.gltf) and binary (.glb) GLTF files
- **PyTorch Integration**: Optional GPU-accelerated processing with seamless PyTorch tensor support
- **Scene Generation**: Convert GLTF data to hierarchical scene structures
- **Accessor Decoding**: Efficient decoding of vertex data, indices, and other attributes
- **Extension Support**: Framework for handling GLTF extensions
- **Comprehensive Testing**: Full test suite using Slash testing framework
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Installation

### Requirements

- Python 3.7+
- PyTorch (optional, for GPU acceleration)
- Slash (for testing)

### Basic Installation

```bash
# Install with basic functionality
pip install -r requirements.txt

# Install with PyTorch support
pip install torch
pip install -r requirements.txt
```

### From Source

```bash
git clone <repository-url>
cd godot-gltf-python
pip install -e .
```

## Quick Start

### Basic Usage

```python
from gltf_module import GLTFDocument

# Load a GLTF file
doc = GLTFDocument()
success = doc.load_from_file('path/to/model.gltf')

if success:
    print(f"Loaded {len(doc.state.nodes)} nodes")
    print(f"Loaded {len(doc.state.meshes)} meshes")

    # Access parsed data
    for node in doc.state.nodes:
        print(f"Node: {node.name}")
```

### PyTorch-Accelerated Usage

```python
import torch
from gltf_module import GLTFDocument, TorchGLTFAccessorDecoder, get_torch_device

# Load GLTF
doc = GLTFDocument()
doc.load_from_file('model.gltf')

# Use GPU acceleration
device = get_torch_device('auto')  # Auto-detect GPU
decoder = TorchGLTFAccessorDecoder(device=device)

# Decode vertex data as PyTorch tensors
vertices = decoder.decode_accessor_tensor(doc.state, position_accessor_index)
normals = decoder.decode_accessor_tensor(doc.state, normal_accessor_index)

print(f"Vertices shape: {vertices.shape}")  # torch.Size([N, 3])
print(f"Device: {vertices.device}")        # cuda:0 or cpu
```

### Scene Generation

```python
from gltf_module import GLTFDocument, GLTFSceneGenerator

# Load and generate scene
doc = GLTFDocument()
doc.load_from_file('model.gltf')

generator = GLTFSceneGenerator(doc.state)
scene_nodes = generator.generate_scene()

# Traverse scene hierarchy
for node in scene_nodes:
    print(f"Node: {node.name}")
    if node.mesh_data:
        print(f"  Mesh: {node.mesh_data.num_vertices} vertices")
```

## Architecture

### Core Components

- **`GLTFDocument`**: Main interface for loading GLTF files
- **`GLTFState`**: Container for parsed GLTF data structures
- **`GLTFAccessorDecoder`**: CPU-based accessor data decoding
- **`GLTFSceneGenerator`**: Scene hierarchy generation
- **`TorchGLTFAccessorDecoder`**: GPU-accelerated decoding (PyTorch)
- **`TorchMeshData`**: PyTorch tensor-based mesh data

### Data Structures

The module provides both standard Python and PyTorch-compatible data structures:

```python
# Standard structures
from gltf_module.structures import GLTFNode, GLTFMesh, GLTFMaterial

# PyTorch-compatible structures
from gltf_module.torch_structures import TorchMeshData, TorchSceneNode
```

## API Reference

### GLTFDocument

```python
class GLTFDocument:
    def __init__(self)
    def load_from_file(self, file_path: str) -> bool
    def load_from_string(self, json_string: str) -> bool
    def get_state(self) -> GLTFState
    def clear(self)
```

### GLTFAccessorDecoder

```python
class GLTFAccessorDecoder:
    def decode_accessor_as_vec3(self, state, accessor_index) -> List[List[float]]
    def decode_accessor_as_vec2(self, state, accessor_index) -> List[List[float]]
    def decode_accessor_as_indices(self, state, accessor_index) -> List[int]
    def decode_accessor_as_colors(self, state, accessor_index) -> List[List[float]]
```

### TorchGLTFAccessorDecoder

```python
class TorchGLTFAccessorDecoder:
    def __init__(self, device: torch.device = None)
    def decode_accessor_tensor(self, state, accessor_index) -> torch.Tensor
    def batch_decode_accessors(self, state, accessor_indices) -> List[torch.Tensor]
    def get_memory_usage(self, state) -> dict
```

## PyTorch Integration

### Device Management

```python
from gltf_module import get_torch_device, is_torch_available

if is_torch_available():
    device = get_torch_device('auto')  # Auto-detect best device
    # or
    device = get_torch_device('cuda')  # Force CUDA
    # or
    device = get_torch_device('cpu')   # Force CPU
```

### GPU Acceleration Benefits

- **Faster Decoding**: Direct tensor operations on GPU
- **Memory Efficiency**: Zero-copy operations where possible
- **Batch Processing**: Process multiple accessors simultaneously
- **Integration**: Seamless use with PyTorch ML pipelines

### Example: GPU-Accelerated Mesh Processing

```python
import torch
from gltf_module import TorchMeshData

# Create mesh data on GPU
mesh = TorchMeshData(device=torch.device('cuda'))
mesh.vertices = torch.randn(1000, 3, device='cuda')
mesh.normals = torch.randn(1000, 3, device='cuda')

# Compute face normals on GPU
face_normals = mesh.compute_face_normals()
print(face_normals.device)  # cuda:0

# Move to CPU for further processing
mesh.to(torch.device('cpu'))
```

## Testing

The module includes comprehensive tests using the Slash testing framework:

```bash
# Run all tests
slash run

# Run specific test file
slash run tests/test_gltf_document.py

# Run with PyTorch tests
slash run tests/test_torch_integration.py

# Run tests with coverage
slash run --coverage
```

### Test Structure

- **`test_gltf_document.py`**: GLTFDocument functionality
- **`test_accessor_decoder.py`**: Accessor decoding
- **`test_torch_integration.py`**: PyTorch integration
- **`test_scene_generator.py`**: Scene generation

## Examples

See `example_usage.py` for comprehensive usage examples including:

- Basic GLTF loading
- PyTorch-accelerated processing
- Scene generation
- Memory usage optimization

## Performance Considerations

### CPU vs GPU

- **Small Models**: CPU decoding is often faster due to overhead
- **Large Models**: GPU decoding provides significant speedup
- **Memory**: GPU tensors use GPU memory, consider VRAM limits

### Memory Usage

```python
from gltf_module import TorchGLTFAccessorDecoder

decoder = TorchGLTFAccessorDecoder()
memory_info = decoder.get_memory_usage(gltf_state)
print(f"Estimated usage: {memory_info['total_mb']} MB")
```

### Optimization Tips

1. **Batch Processing**: Use `batch_decode_accessors()` for multiple accessors
2. **Device Selection**: Choose appropriate device based on model size
3. **Caching**: Reuse decoded tensors when possible
4. **Memory Management**: Move unused tensors to CPU or delete them

## Extensions

The module provides a framework for handling GLTF extensions:

```python
# Check supported extensions
from gltf_module.gltf_state import GLTFState
state = GLTFState()
print("Supported extensions:", state.extensions.used)

# Extension-specific handling can be added here
```

## Compatibility

### GLTF 2.0 Features Supported

- ✅ JSON and binary (.glb) files
- ✅ All accessor types (SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4)
- ✅ All component types (BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT)
- ✅ Buffer and buffer view handling
- ✅ Data URI support
- ✅ Node hierarchy and transforms
- ✅ Mesh primitives with attributes
- ✅ Material definitions
- ✅ Texture and image references
- ✅ Animation data structures
- ✅ Skinning data structures

### PyTorch Compatibility

- ✅ PyTorch 1.7+ (optional)
- ✅ CUDA support (if available)
- ✅ CPU fallback
- ✅ Tensor operations
- ✅ Memory management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd godot-gltf-python

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
slash run

# Run linting
flake8 gltf_module tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Godot Engine's GLTF module implementation
- PyTorch integration for GPU acceleration
- Slash testing framework for comprehensive testing
- GLTF 2.0 specification compliance

## Changelog

### Version 1.0.0
- Initial release
- Complete GLTF 2.0 parsing support
- PyTorch integration
- Comprehensive test suite
- Scene generation
- GPU-accelerated decoding
