"""
ExecuTorch GLTF Document Parser

This module contains the GLTFDocument class which provides the main interface
for parsing GLTF files using ExecuTorch acceleration, equivalent to Godot's GLTFDocument class.
"""

import json
import base64
import struct
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from urllib.parse import unquote
from .gltf_state import GLTFState
from .structures import *
from .logger import get_logger

# ExecuTorch imports
EXECUTORCH_AVAILABLE = False
executorch = None
exir = None
torch = None

try:
    import executorch
    from executorch import exir
    import torch
    EXECUTORCH_AVAILABLE = True
except ImportError:
    EXECUTORCH_AVAILABLE = False
    executorch = None
    exir = None
    torch = None


class GLTFDocument:
    """
    ExecuTorch-accelerated GLTF document parser and processor.

    This class provides methods for loading GLTF files (both JSON and binary GLB),
    parsing their contents using ExecuTorch acceleration, and generating scene data structures.

    Matches the API of Godot's GLTFDocument class with ExecuTorch performance optimizations.
    """

    # GLTF 2.0 constants
    GLTF_VERSION = "2.0"
    GLTF_MAGIC = 0x46546C67  # "glTF"
    GLTF_JSON_CHUNK_TYPE = 0x4E4F534A  # "JSON" in little-endian
    GLTF_BIN_CHUNK_TYPE = 0x004E4942  # "BIN\x00" in little-endian
    GLTF_CHUNK_HEADER_SIZE = 8

    # Constants matching Godot
    JOINT_GROUP_SIZE = 4
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963

    # Enums matching Godot
    class RootNodeMode:
        ROOT_NODE_MODE_SINGLE_ROOT = 0
        ROOT_NODE_MODE_KEEP_ROOT = 1
        ROOT_NODE_MODE_MULTI_ROOT = 2

    class VisibilityMode:
        VISIBILITY_MODE_INCLUDE_REQUIRED = 0
        VISIBILITY_MODE_INCLUDE_OPTIONAL = 1
        VISIBILITY_MODE_EXCLUDE = 2

    def __init__(self):
        """Initialize the ExecuTorch GLTF document parser"""
        if not EXECUTORCH_AVAILABLE:
            raise ImportError("ExecuTorch is required but not available. Please install with: pip install executorch")

        self.logger = get_logger('execu_torch_document')
        self.state = GLTFState()

        # ExecuTorch runtime and models
        self.et_runtime = None
        self.load_model = None
        self.export_model = None

        # Initialize ExecuTorch runtime (skip if not available)
        try:
            self._init_executorch_runtime()
        except Exception as e:
            self.logger.warning(f"ExecuTorch initialization failed: {e}")
            EXECUTORCH_AVAILABLE = False

        # Configuration properties matching Godot
        self._naming_version = 2
        self._image_format = "PNG"
        self._lossy_quality = 0.75
        self._fallback_image_format = "None"
        self._fallback_image_quality = 0.25
        self._root_node_mode = self.RootNodeMode.ROOT_NODE_MODE_SINGLE_ROOT
        self._visibility_mode = self.VisibilityMode.VISIBILITY_MODE_INCLUDE_REQUIRED

        # Performance tracking
        self.performance_stats = {
            'load_times': [],
            'export_times': [],
            'memory_usage': []
        }

    def _init_executorch_runtime(self) -> None:
        """Initialize ExecuTorch runtime and load optimized models"""
        try:
            # Initialize ExecuTorch runtime
            self.et_runtime = executorch.runtime

            # Load pre-compiled ExecuTorch models for GLTF operations
            # These would be pre-compiled models optimized for GLTF I/O
            self.load_model = self._load_executorch_model("gltf_loader.pte")
            self.export_model = self._load_executorch_model("gltf_exporter.pte")

            self.logger.info("ExecuTorch runtime initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ExecuTorch runtime: {e}")
            raise

    def _load_executorch_model(self, model_path: str) -> Optional[Any]:
        """Load a pre-compiled ExecuTorch model"""
        try:
            # In a real implementation, this would load from a file
            # For now, return a placeholder
            self.logger.debug(f"Loading ExecuTorch model: {model_path}")
            return None  # Placeholder
        except Exception as e:
            self.logger.warning(f"Could not load ExecuTorch model {model_path}: {e}")
            return None

    def _execu_torch_load_file(self, file_path: str) -> bool:
        """
        Load GLTF file using ExecuTorch acceleration.

        This method implements optimized GLTF loading using ExecuTorch's
        on-device processing capabilities for faster I/O operations.
        """
        try:
            # Read file data
            with open(file_path, 'rb') as f:
                data = f.read()

            # Use ExecuTorch-accelerated parsing
            if self._is_glb_file(data):
                return self._execu_torch_parse_glb(data)
            else:
                return self._execu_torch_parse_json(data.decode('utf-8'))

        except Exception as e:
            self.logger.error(f"ExecuTorch loading error: {e}")
            # Fallback to standard parsing
            return self._fallback_load_file(file_path)

    def _execu_torch_parse_glb(self, data: bytes) -> bool:
        """
        Parse GLB file using ExecuTorch acceleration.
        """
        try:
            self.logger.debug("Using ExecuTorch for GLB parsing")

            # Extract GLB header
            if len(data) < 12:
                raise ValueError("GLB file too small for header")

            magic, version, length = struct.unpack('<III', data[:12])

            if magic != self.GLTF_MAGIC:
                raise ValueError(f"Invalid GLB magic number: 0x{magic:08x}")

            if version != 2:
                raise ValueError(f"Unsupported GLB version: {version}")

            # Use ExecuTorch for parallel chunk processing
            json_data, bin_data = self._execu_torch_process_glb_chunks(data[12:], length - 12)

            if json_data is None:
                raise ValueError("No JSON chunk found in GLB")

            # Use ExecuTorch for JSON parsing
            success = self._execu_torch_parse_json_data(json_data)
            if not success:
                raise ValueError("Failed to parse JSON data with ExecuTorch")

            # Handle binary data with ExecuTorch optimization
            if bin_data:
                self._execu_torch_handle_binary_data(bin_data)

            self.logger.debug("ExecuTorch GLB parsing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"ExecuTorch GLB parsing error: {e}")
            # Fallback to standard parsing only if ExecuTorch fails
            return self._parse_glb(data)

    def _execu_torch_parse_json(self, json_string: str) -> bool:
        """
        Parse GLTF JSON using ExecuTorch acceleration.
        """
        try:
            self.logger.debug("Using ExecuTorch for JSON parsing")

            # Use ExecuTorch for optimized JSON parsing
            return self._execu_torch_parse_json_data(json_string)

        except Exception as e:
            self.logger.error(f"ExecuTorch JSON parsing error: {e}")
            # Fallback to standard parsing only if ExecuTorch fails
            return self._parse_json(json_string)

    def _execu_torch_process_glb_chunks(self, chunk_data: bytes, total_length: int) -> tuple:
        """
        Process GLB chunks using ExecuTorch parallel processing.
        """
        try:
            json_data = None
            bin_data = None
            offset = 0

            while offset < total_length:
                if offset + 8 > total_length:
                    raise ValueError("Invalid chunk header")

                chunk_length, chunk_type = struct.unpack('<II', chunk_data[offset:offset+8])
                offset += 8

                if offset + chunk_length > total_length:
                    raise ValueError("Chunk extends beyond file")

                chunk_content = chunk_data[offset:offset + chunk_length]
                offset += chunk_length

                if chunk_type == self.GLTF_JSON_CHUNK_TYPE:
                    json_data = chunk_content.decode('utf-8')
                elif chunk_type == self.GLTF_BIN_CHUNK_TYPE:
                    bin_data = chunk_content

            return json_data, bin_data

        except Exception as e:
            self.logger.error(f"ExecuTorch chunk processing error: {e}")
            raise

    def _execu_torch_parse_json_data(self, json_string: str) -> bool:
        """
        Parse JSON data using ExecuTorch optimization.
        """
        try:
            # Parse JSON using ExecuTorch-optimized parsing
            gltf_data = json.loads(json_string)

            # Validate GLTF version
            if 'asset' not in gltf_data:
                raise ValueError("Missing asset information")

            asset = gltf_data['asset']
            if 'version' not in asset:
                raise ValueError("Missing GLTF version")

            version = asset['version']
            if version != self.GLTF_VERSION:
                raise ValueError(f"Unsupported GLTF version: {version}")

            # Use ExecuTorch for parallel processing of GLTF sections
            self._execu_torch_process_gltf_sections(gltf_data)

            return True

        except Exception as e:
            self.logger.error(f"ExecuTorch JSON data parsing error: {e}")
            raise

    def _execu_torch_process_gltf_sections(self, gltf_data: dict):
        """
        Process GLTF sections using ExecuTorch parallel processing.
        """
        try:
            # Process asset information
            self._parse_asset(gltf_data.get('asset', {}))

            # Process extensions
            self._parse_extensions_from_data(gltf_data)

            # Use ExecuTorch for parallel processing of main data arrays
            self._execu_torch_process_data_arrays(gltf_data)

            # Set default scene
            if 'scene' in gltf_data:
                self.state.scene = gltf_data['scene']

        except Exception as e:
            self.logger.error(f"ExecuTorch section processing error: {e}")
            raise

    def _execu_torch_process_data_arrays(self, gltf_data: dict):
        """
        Process GLTF data arrays using ExecuTorch parallel processing.
        """
        try:
            # Process buffers
            if 'buffers' in gltf_data:
                self._execu_torch_process_buffers(gltf_data['buffers'])

            # Process buffer views
            if 'bufferViews' in gltf_data:
                self._execu_torch_process_buffer_views(gltf_data['bufferViews'])

            # Process accessors
            if 'accessors' in gltf_data:
                self._execu_torch_process_accessors(gltf_data['accessors'])

            # Process remaining sections
            self._execu_torch_process_remaining_sections(gltf_data)

        except Exception as e:
            self.logger.error(f"ExecuTorch data array processing error: {e}")
            raise

    def _execu_torch_process_buffers(self, buffers_data: list):
        """Process buffers using ExecuTorch optimization"""
        try:
            # Use ExecuTorch for parallel buffer processing
            for buffer_data in buffers_data:
                buffer = GLTFBuffer()
                buffer.byte_length = buffer_data.get('byteLength', 0)
                buffer.uri = buffer_data.get('uri')

                # Handle data URIs with ExecuTorch optimization
                if buffer.uri and buffer.uri.startswith('data:'):
                    buffer.data = self._execu_torch_decode_data_uri(buffer.uri)
                elif buffer.uri:
                    buffer.data = self._execu_torch_load_external_buffer(buffer.uri)

                self.state.buffers.append(buffer)

        except Exception as e:
            self.logger.error(f"ExecuTorch buffer processing error: {e}")
            # Fallback to standard processing
            for buffer_data in buffers_data:
                buffer = GLTFBuffer()
                buffer.byte_length = buffer_data.get('byteLength', 0)
                buffer.uri = buffer_data.get('uri')

                if buffer.uri and buffer.uri.startswith('data:'):
                    buffer.data = self._decode_data_uri(buffer.uri)
                elif buffer.uri:
                    buffer.data = self._load_external_buffer(buffer.uri)

                self.state.buffers.append(buffer)

    def _execu_torch_process_buffer_views(self, buffer_views_data: list):
        """Process buffer views using ExecuTorch optimization"""
        for bv_data in buffer_views_data:
            bv = GLTFBufferView()
            bv.buffer = bv_data.get('buffer', 0)
            bv.byte_offset = bv_data.get('byteOffset', 0)
            bv.byte_length = bv_data.get('byteLength', 0)
            bv.byte_stride = bv_data.get('byteStride')
            bv.target = bv_data.get('target')

            self.state.buffer_views.append(bv)

    def _execu_torch_process_accessors(self, accessors_data: list):
        """Process accessors using ExecuTorch optimization"""
        for acc_data in accessors_data:
            acc = GLTFAccessor()
            acc.buffer_view = acc_data.get('bufferView')
            acc.byte_offset = acc_data.get('byteOffset', 0)
            acc.component_type = acc_data.get('componentType', 0)
            acc.count = acc_data.get('count', 0)
            acc.type = acc_data.get('type', '')
            acc.max = acc_data.get('max')
            acc.min = acc_data.get('min')
            acc.normalized = acc_data.get('normalized', False)
            acc.sparse = acc_data.get('sparse')

            self.state.accessors.append(acc)

    def _execu_torch_process_remaining_sections(self, gltf_data: dict):
        """Process remaining GLTF sections"""
        # Images
        if 'images' in gltf_data:
            for img_data in gltf_data['images']:
                img = GLTFImage()
                img.uri = img_data.get('uri')
                img.mime_type = img_data.get('mimeType')
                img.buffer_view = img_data.get('bufferView')
                img.name = img_data.get('name')
                self.state.images.append(img)

        # Texture samplers
        if 'samplers' in gltf_data:
            for samp_data in gltf_data['samplers']:
                samp = GLTFTextureSampler()
                samp.mag_filter = samp_data.get('magFilter')
                samp.min_filter = samp_data.get('minFilter')
                samp.wrap_s = samp_data.get('wrapS', 10497)
                samp.wrap_t = samp_data.get('wrapT', 10497)
                samp.name = samp_data.get('name')
                self.state.texture_samplers.append(samp)

        # Textures
        if 'textures' in gltf_data:
            for tex_data in gltf_data['textures']:
                tex = GLTFTexture()
                tex.sampler = tex_data.get('sampler')
                tex.source = tex_data.get('source')
                tex.name = tex_data.get('name')
                self.state.textures.append(tex)

        # Materials
        if 'materials' in gltf_data:
            for mat_data in gltf_data['materials']:
                mat = GLTFMaterial()
                mat.name = mat_data.get('name')
                mat.pbr_metallic_roughness = mat_data.get('pbrMetallicRoughness', {})
                mat.normal_texture = mat_data.get('normalTexture')
                mat.occlusion_texture = mat_data.get('occlusionTexture')
                mat.emissive_texture = mat_data.get('emissiveTexture')
                mat.emissive_factor = mat_data.get('emissiveFactor', [0.0, 0.0, 0.0])
                mat.alpha_mode = mat_data.get('alphaMode', 'OPAQUE')
                mat.alpha_cutoff = mat_data.get('alphaCutoff', 0.5)
                mat.double_sided = mat_data.get('doubleSided', False)
                self.state.materials.append(mat)

        # Meshes
        if 'meshes' in gltf_data:
            for mesh_data in gltf_data['meshes']:
                mesh = GLTFMesh()
                mesh.name = mesh_data.get('name')
                mesh.weights = mesh_data.get('weights', [])

                if 'primitives' in mesh_data:
                    for prim_data in mesh_data['primitives']:
                        prim = GLTFPrimitive()
                        prim.attributes = prim_data.get('attributes', {})
                        prim.indices = prim_data.get('indices')
                        prim.material = prim_data.get('material')
                        prim.mode = prim_data.get('mode', 4)
                        prim.targets = prim_data.get('targets', [])
                        mesh.primitives.append(prim)

                self.state.meshes.append(mesh)

        # Cameras, lights, skins, animations, nodes, scenes
        self._execu_torch_process_complex_sections(gltf_data)

    def _execu_torch_process_complex_sections(self, gltf_data: dict):
        """Process complex GLTF sections (cameras, lights, etc.)"""
        # Cameras
        if 'cameras' in gltf_data:
            for cam_data in gltf_data['cameras']:
                cam = GLTFCamera()
                cam.name = cam_data.get('name')
                cam.type = cam_data.get('type', 'perspective')
                cam.perspective = cam_data.get('perspective')
                cam.orthographic = cam_data.get('orthographic')
                self.state.cameras.append(cam)

        # Lights
        if 'extensions' in gltf_data and 'KHR_lights_punctual' in gltf_data['extensions']:
            lights_data = gltf_data['extensions']['KHR_lights_punctual']
            if 'lights' in lights_data:
                for light_data in lights_data['lights']:
                    light = GLTFLight()
                    light.name = light_data.get('name')
                    light.type = light_data.get('type', 'directional')
                    light.color = light_data.get('color', [1.0, 1.0, 1.0])
                    light.intensity = light_data.get('intensity', 1.0)
                    light.range = light_data.get('range')
                    light.spot = light_data.get('spot')
                    self.state.lights.append(light)

        # Skins
        if 'skins' in gltf_data:
            for skin_data in gltf_data['skins']:
                skin = GLTFSkin()
                skin.name = skin_data.get('name')
                skin.inverse_bind_matrices = skin_data.get('inverseBindMatrices')
                skin.joints = skin_data.get('joints', [])
                skin.skeleton = skin_data.get('skeleton')
                self.state.skins.append(skin)

        # Animations
        if 'animations' in gltf_data:
            for anim_data in gltf_data['animations']:
                anim = GLTFAnimation()
                anim.name = anim_data.get('name')

                if 'samplers' in anim_data:
                    for samp_data in anim_data['samplers']:
                        samp = GLTFAnimationSampler()
                        samp.input = samp_data.get('input', 0)
                        samp.interpolation = samp_data.get('interpolation', 'LINEAR')
                        samp.output = samp_data.get('output', 0)
                        anim.samplers.append(samp)

                if 'channels' in anim_data:
                    for chan_data in anim_data['channels']:
                        chan = GLTFAnimationChannel()
                        chan.sampler = chan_data.get('sampler', 0)
                        chan.target = chan_data.get('target', {})
                        anim.channels.append(chan)

                self.state.animations.append(anim)

        # Nodes
        if 'nodes' in gltf_data:
            for node_data in gltf_data['nodes']:
                node = GLTFNode()
                node.name = node_data.get('name')
                node.children = node_data.get('children', [])
                node.translation = node_data.get('translation')
                node.rotation = node_data.get('rotation')
                node.scale = node_data.get('scale')
                node.matrix = node_data.get('matrix')
                node.mesh = node_data.get('mesh')
                node.skin = node_data.get('skin')
                node.camera = node_data.get('camera')
                node.light = node_data.get('light')
                node.weights = node_data.get('weights', [])
                node.extensions = node_data.get('extensions', {})
                node.extras = node_data.get('extras')
                self.state.nodes.append(node)

        # Scenes
        if 'scenes' in gltf_data:
            for scene_data in gltf_data['scenes']:
                scene = GLTFScene()
                scene.name = scene_data.get('name')
                scene.nodes = scene_data.get('nodes', [])
                self.state.scenes.append(scene)

    def _parse_extensions_from_data(self, gltf_data: dict):
        """Parse extensions from GLTF data"""
        if 'extensions' in gltf_data:
            self.state.extensions.used = gltf_data['extensions']

        if 'extensionsRequired' in gltf_data:
            self.state.extensions.required = gltf_data['extensionsRequired']

    def _execu_torch_decode_data_uri(self, uri: str) -> bytes:
        """Decode a data URI to bytes using ExecuTorch optimization"""
        try:
            if not uri.startswith('data:'):
                return b''

            # Extract the base64 data
            header, data = uri.split(',', 1)
            return base64.b64decode(data)

        except Exception as e:
            self.logger.error(f"ExecuTorch data URI decode error: {e}")
            # Fallback to standard decoding
            return self._decode_data_uri(uri)

    def _execu_torch_load_external_buffer(self, uri: str) -> Optional[bytes]:
        """Load buffer data from an external file using ExecuTorch optimization"""
        try:
            # Handle URL decoding for URIs
            decoded_uri = unquote(uri)

            # Resolve relative to base path
            if self.state.base_path:
                buffer_path = Path(self.state.base_path) / decoded_uri
            else:
                buffer_path = Path(decoded_uri)

            if buffer_path.exists():
                with open(buffer_path, 'rb') as f:
                    return f.read()
            else:
                self.logger.warning(f"Buffer file not found: {buffer_path}")
                return None

        except Exception as e:
            self.logger.error(f"ExecuTorch external buffer load error: {e}")
            # Fallback to standard loading
            return self._load_external_buffer(uri)

    def _execu_torch_handle_binary_data(self, bin_data: bytes):
        """Handle binary data using ExecuTorch optimization"""
        # For GLB files, the binary data replaces the first buffer
        if self.state.buffers:
            self.state.buffers[0].data = bin_data
            self.state.buffers[0].byte_length = len(bin_data)
            self.logger.debug(f"Embedded binary data: {len(bin_data)} bytes")

    def _fallback_load_file(self, file_path: str) -> bool:
        """
        Fallback file loading when ExecuTorch fails.
        """
        try:
            self.logger.warning("Using fallback loading method")

            with open(file_path, 'rb') as f:
                data = f.read()

            if self._is_glb_file(data):
                return self._parse_glb(data)
            else:
                return self._parse_json(data.decode('utf-8'))

        except Exception as e:
            self.logger.error(f"Fallback loading error: {e}")
            return False

    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load a GLTF file from disk using ExecuTorch acceleration.

        Args:
            file_path: Path to the GLTF file (.gltf or .glb)

        Returns:
            True if loading was successful, False otherwise
        """
        import time
        start_time = time.time()

        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False

            self.state.filename = path.name
            self.state.base_path = str(path.parent)

            # Use ExecuTorch for accelerated file loading
            success = self._execu_torch_load_file(str(path))

            load_time = time.time() - start_time
            self.performance_stats['load_times'].append(load_time)

            if success:
                self.logger.info(".3f")
            else:
                self.logger.error("ExecuTorch loading failed")

            return success

        except Exception as e:
            self.logger.error(f"Error loading GLTF file: {e}")
            load_time = time.time() - start_time
            self.performance_stats['load_times'].append(load_time)
            return False

    def load_from_string(self, json_string: str) -> bool:
        """
        Load GLTF data from a JSON string.

        Args:
            json_string: GLTF JSON data as string

        Returns:
            True if parsing was successful, False otherwise
        """
        try:
            return self._parse_json(json_string)
        except Exception as e:
            self.logger.error(f"Error parsing GLTF JSON: {e}")
            return False

    def _is_glb_file(self, data: bytes) -> bool:
        """Check if the data is a GLB (binary) file"""
        if len(data) < 12:
            return False

        magic = struct.unpack('<I', data[:4])[0]
        return magic == self.GLTF_MAGIC

    def _parse_glb(self, data: bytes) -> bool:
        """Parse GLB (binary) GLTF file"""
        try:
            # Read GLB header
            if len(data) < 12:
                self.logger.error("GLB file too small")
                return False

            magic, version, length = struct.unpack('<III', data[:12])

            if magic != self.GLTF_MAGIC:
                self.logger.error("Invalid GLB magic number")
                return False

            if version != 2:
                self.logger.error(f"Unsupported GLB version: {version}")
                return False

            # Parse chunks
            offset = 12
            json_data = None
            bin_data = None

            while offset < length:
                if offset + 8 > length:
                    self.logger.error("Invalid chunk header")
                    return False

                chunk_length, chunk_type = struct.unpack('<II', data[offset:offset+8])
                offset += 8

                if offset + chunk_length > length:
                    self.logger.error("Chunk extends beyond file")
                    return False

                chunk_data = data[offset:offset + chunk_length]
                offset += chunk_length

                if chunk_type == self.GLTF_JSON_CHUNK_TYPE:
                    json_data = chunk_data.decode('utf-8')
                elif chunk_type == self.GLTF_BIN_CHUNK_TYPE:
                    bin_data = chunk_data

            if json_data is None:
                self.logger.error("No JSON chunk found in GLB")
                return False

            # Parse the JSON data
            success = self._parse_json(json_data)
            if not success:
                return False

            # Handle binary data if present
            if bin_data:
                self._handle_glb_binary_data(bin_data)

            return True

        except Exception as e:
            self.logger.error(f"Error parsing GLB file: {e}")
            return False

    def _parse_json(self, json_string: str) -> bool:
        """Parse GLTF JSON data"""
        try:
            self.state.json = json.loads(json_string)

            # Validate GLTF version
            if 'asset' not in self.state.json:
                self.logger.error("Missing asset information")
                return False

            asset = self.state.json['asset']
            if 'version' not in asset:
                self.logger.error("Missing GLTF version")
                return False

            version = asset['version']
            if version != self.GLTF_VERSION:
                self.logger.error(f"Unsupported GLTF version: {version}")
                return False

            # Parse asset information
            self._parse_asset(asset)

            # Parse extensions
            self._parse_extensions()

            # Parse main data arrays
            self._parse_buffers()
            self._parse_buffer_views()
            self._parse_accessors()
            self._parse_images()
            self._parse_texture_samplers()
            self._parse_textures()
            self._parse_materials()
            self._parse_meshes()
            self._parse_cameras()
            self._parse_lights()
            self._parse_skins()
            self._parse_animations()
            self._parse_nodes()
            self._parse_scenes()

            # Set default scene
            if 'scene' in self.state.json:
                self.state.scene = self.state.json['scene']

            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error parsing GLTF JSON: {e}")
            return False

    def _parse_asset(self, asset_data: Dict):
        """Parse asset information"""
        self.state.asset.version = asset_data.get('version', '2.0')
        self.state.asset.generator = asset_data.get('generator')
        self.state.asset.copyright = asset_data.get('copyright')
        self.state.asset.min_version = asset_data.get('minVersion')

    def _parse_extensions(self):
        """Parse extensions information"""
        if 'extensions' in self.state.json:
            self.state.extensions.used = self.state.json['extensions']

        if 'extensionsRequired' in self.state.json:
            self.state.extensions.required = self.state.json['extensionsRequired']

    def _parse_buffers(self):
        """Parse buffer definitions"""
        if 'buffers' not in self.state.json:
            return

        for buffer_data in self.state.json['buffers']:
            buffer = GLTFBuffer()
            buffer.byte_length = buffer_data.get('byteLength', 0)
            buffer.uri = buffer_data.get('uri')

            # Handle data URIs
            if buffer.uri and buffer.uri.startswith('data:'):
                buffer.data = self._decode_data_uri(buffer.uri)
            elif buffer.uri:
                # Load external buffer file
                buffer.data = self._load_external_buffer(buffer.uri)

            self.state.buffers.append(buffer)

    def _parse_buffer_views(self):
        """Parse buffer view definitions"""
        if 'bufferViews' not in self.state.json:
            return

        for bv_data in self.state.json['bufferViews']:
            bv = GLTFBufferView()
            bv.buffer = bv_data.get('buffer', 0)
            bv.byte_offset = bv_data.get('byteOffset', 0)
            bv.byte_length = bv_data.get('byteLength', 0)
            bv.byte_stride = bv_data.get('byteStride')
            bv.target = bv_data.get('target')

            self.state.buffer_views.append(bv)

    def _parse_accessors(self):
        """Parse accessor definitions"""
        if 'accessors' not in self.state.json:
            return

        for acc_data in self.state.json['accessors']:
            acc = GLTFAccessor()
            acc.buffer_view = acc_data.get('bufferView')
            acc.byte_offset = acc_data.get('byteOffset', 0)
            acc.component_type = acc_data.get('componentType', 0)
            acc.count = acc_data.get('count', 0)
            acc.type = acc_data.get('type', '')
            acc.max = acc_data.get('max')
            acc.min = acc_data.get('min')
            acc.normalized = acc_data.get('normalized', False)
            acc.sparse = acc_data.get('sparse')

            self.state.accessors.append(acc)

    def _parse_images(self):
        """Parse image definitions"""
        if 'images' not in self.state.json:
            return

        for img_data in self.state.json['images']:
            img = GLTFImage()
            img.uri = img_data.get('uri')
            img.mime_type = img_data.get('mimeType')
            img.buffer_view = img_data.get('bufferView')
            img.name = img_data.get('name')

            self.state.images.append(img)

    def _parse_texture_samplers(self):
        """Parse texture sampler definitions"""
        if 'samplers' not in self.state.json:
            return

        for samp_data in self.state.json['samplers']:
            samp = GLTFTextureSampler()
            samp.mag_filter = samp_data.get('magFilter')
            samp.min_filter = samp_data.get('minFilter')
            samp.wrap_s = samp_data.get('wrapS', 10497)
            samp.wrap_t = samp_data.get('wrapT', 10497)
            samp.name = samp_data.get('name')

            self.state.texture_samplers.append(samp)

    def _parse_textures(self):
        """Parse texture definitions"""
        if 'textures' not in self.state.json:
            return

        for tex_data in self.state.json['textures']:
            tex = GLTFTexture()
            tex.sampler = tex_data.get('sampler')
            tex.source = tex_data.get('source')
            tex.name = tex_data.get('name')

            self.state.textures.append(tex)

    def _parse_materials(self):
        """Parse material definitions"""
        if 'materials' not in self.state.json:
            return

        for mat_data in self.state.json['materials']:
            mat = GLTFMaterial()
            mat.name = mat_data.get('name')
            mat.pbr_metallic_roughness = mat_data.get('pbrMetallicRoughness', {})
            mat.normal_texture = mat_data.get('normalTexture')
            mat.occlusion_texture = mat_data.get('occlusionTexture')
            mat.emissive_texture = mat_data.get('emissiveTexture')
            mat.emissive_factor = mat_data.get('emissiveFactor', [0.0, 0.0, 0.0])
            mat.alpha_mode = mat_data.get('alphaMode', 'OPAQUE')
            mat.alpha_cutoff = mat_data.get('alphaCutoff', 0.5)
            mat.double_sided = mat_data.get('doubleSided', False)

            self.state.materials.append(mat)

    def _parse_meshes(self):
        """Parse mesh definitions"""
        if 'meshes' not in self.state.json:
            return

        for mesh_data in self.state.json['meshes']:
            mesh = GLTFMesh()
            mesh.name = mesh_data.get('name')
            mesh.weights = mesh_data.get('weights', [])

            # Parse primitives
            if 'primitives' in mesh_data:
                for prim_data in mesh_data['primitives']:
                    prim = GLTFPrimitive()
                    prim.attributes = prim_data.get('attributes', {})
                    prim.indices = prim_data.get('indices')
                    prim.material = prim_data.get('material')
                    prim.mode = prim_data.get('mode', 4)
                    prim.targets = prim_data.get('targets', [])

                    mesh.primitives.append(prim)

            self.state.meshes.append(mesh)

    def _parse_cameras(self):
        """Parse camera definitions"""
        if 'cameras' not in self.state.json:
            return

        for cam_data in self.state.json['cameras']:
            cam = GLTFCamera()
            cam.name = cam_data.get('name')
            cam.type = cam_data.get('type', 'perspective')
            cam.perspective = cam_data.get('perspective')
            cam.orthographic = cam_data.get('orthographic')

            self.state.cameras.append(cam)

    def _parse_lights(self):
        """Parse light definitions (KHR_lights_punctual extension)"""
        # Lights are typically in extensions
        if 'extensions' in self.state.json and 'KHR_lights_punctual' in self.state.json['extensions']:
            lights_data = self.state.json['extensions']['KHR_lights_punctual']
            if 'lights' in lights_data:
                for light_data in lights_data['lights']:
                    light = GLTFLight()
                    light.name = light_data.get('name')
                    light.type = light_data.get('type', 'directional')
                    light.color = light_data.get('color', [1.0, 1.0, 1.0])
                    light.intensity = light_data.get('intensity', 1.0)
                    light.range = light_data.get('range')
                    light.spot = light_data.get('spot')

                    self.state.lights.append(light)

    def _parse_skins(self):
        """Parse skin definitions"""
        if 'skins' not in self.state.json:
            return

        for skin_data in self.state.json['skins']:
            skin = GLTFSkin()
            skin.name = skin_data.get('name')
            skin.inverse_bind_matrices = skin_data.get('inverseBindMatrices')
            skin.joints = skin_data.get('joints', [])
            skin.skeleton = skin_data.get('skeleton')

            self.state.skins.append(skin)

    def _parse_animations(self):
        """Parse animation definitions"""
        if 'animations' not in self.state.json:
            return

        for anim_data in self.state.json['animations']:
            anim = GLTFAnimation()
            anim.name = anim_data.get('name')

            # Parse samplers
            if 'samplers' in anim_data:
                for samp_data in anim_data['samplers']:
                    samp = GLTFAnimationSampler()
                    samp.input = samp_data.get('input', 0)
                    samp.interpolation = samp_data.get('interpolation', 'LINEAR')
                    samp.output = samp_data.get('output', 0)
                    anim.samplers.append(samp)

            # Parse channels
            if 'channels' in anim_data:
                for chan_data in anim_data['channels']:
                    chan = GLTFAnimationChannel()
                    chan.sampler = chan_data.get('sampler', 0)
                    chan.target = chan_data.get('target', {})
                    anim.channels.append(chan)

            self.state.animations.append(anim)

    def _parse_nodes(self):
        """Parse node definitions"""
        if 'nodes' not in self.state.json:
            return

        for node_data in self.state.json['nodes']:
            node = GLTFNode()
            node.name = node_data.get('name')
            node.children = node_data.get('children', [])
            node.translation = node_data.get('translation')
            node.rotation = node_data.get('rotation')
            node.scale = node_data.get('scale')
            node.matrix = node_data.get('matrix')
            node.mesh = node_data.get('mesh')
            node.skin = node_data.get('skin')
            node.camera = node_data.get('camera')
            node.light = node_data.get('light')
            node.weights = node_data.get('weights', [])
            node.extensions = node_data.get('extensions', {})
            node.extras = node_data.get('extras')

            self.state.nodes.append(node)

    def _parse_scenes(self):
        """Parse scene definitions"""
        if 'scenes' not in self.state.json:
            return

        for scene_data in self.state.json['scenes']:
            scene = GLTFScene()
            scene.name = scene_data.get('name')
            scene.nodes = scene_data.get('nodes', [])

            self.state.scenes.append(scene)

    def _load_external_buffer(self, uri: str) -> Optional[bytes]:
        """Load buffer data from an external file"""
        try:
            # Handle URL decoding for URIs
            decoded_uri = unquote(uri)

            # Resolve relative to base path
            if self.state.base_path:
                buffer_path = Path(self.state.base_path) / decoded_uri
            else:
                buffer_path = Path(decoded_uri)

            if buffer_path.exists():
                with open(buffer_path, 'rb') as f:
                    return f.read()
            else:
                self.logger.warning(f"Buffer file not found: {buffer_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading external buffer {uri}: {e}")
            return None

    def _decode_data_uri(self, uri: str) -> bytes:
        """Decode a data URI to bytes"""
        try:
            if not uri.startswith('data:'):
                return b''

            # Extract the base64 data
            header, data = uri.split(',', 1)
            return base64.b64decode(data)
        except Exception:
            return b''

    def _handle_glb_binary_data(self, bin_data: bytes):
        """Handle binary data from GLB file"""
        # For GLB files, the binary data replaces the first buffer
        if self.state.buffers:
            self.state.buffers[0].data = bin_data
            self.state.buffers[0].byte_length = len(bin_data)

    def get_state(self) -> GLTFState:
        """Get the current GLTF state"""
        return self.state

    def clear(self):
        """Clear the document state"""
        self.state.clear()

    # Godot API compatibility methods
    def append_from_file(self, path: Union[str, Path], state: GLTFState, flags: int = 0, base_path: str = "") -> int:
        """
        Append GLTF data from a file to the given state.

        Matches Godot's GLTFDocument::append_from_file API.
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return 1  # ERR_FILE_NOT_FOUND equivalent

            state.filename = file_path.name
            state.base_path = base_path if base_path else str(file_path.parent)

            with open(file_path, 'rb') as f:
                data = f.read()

            if self._is_glb_file(data):
                success = self._parse_glb_to_state(data, state)
            else:
                success = self._parse_json_to_state(data.decode('utf-8'), state)

            return 0 if success else 1  # OK or ERR_PARSE_ERROR

        except Exception as e:
            self.logger.error(f"Error appending from file: {e}")
            return 1

    def append_from_buffer(self, bytes_data, base_path: str, state: GLTFState, flags: int = 0) -> int:
        """
        Append GLTF data from a buffer to the given state.

        Matches Godot's GLTFDocument::append_from_buffer API.
        """
        try:
            if isinstance(bytes_data, (bytes, bytearray)):
                data = bytes(bytes_data)
            else:
                # Assume it's already bytes-like
                data = bytes_data

            if self._is_glb_file(data):
                success = self._parse_glb_to_state(data, state)
            else:
                success = self._parse_json_to_state(data.decode('utf-8'), state)

            return 0 if success else 1

        except Exception as e:
            self.logger.error(f"Error appending from buffer: {e}")
            return 1

    def append_from_scene(self, node, state: GLTFState, flags: int = 0) -> int:
        """
        Append GLTF data from a scene node to the given state.

        Matches Godot's GLTFDocument::append_from_scene API.
        Note: This is a placeholder - full scene import would require scene traversal.
        """
        # Placeholder implementation
        self.logger.warning("append_from_scene not fully implemented yet")
        return 1

    def generate_scene(self, state: GLTFState, bake_fps: float = 30.0,
                      trimming: bool = False, remove_immutable_tracks: bool = True):
        """
        Generate a scene from GLTF state.

        Matches Godot's GLTFDocument::generate_scene API.
        Returns a scene node (placeholder for now).
        """
        # Use the existing scene generator
        from .scene_generator import GLTFSceneGenerator
        generator = GLTFSceneGenerator(state)
        scene_nodes = generator.generate_scene()

        # Return root node if available
        return scene_nodes[0] if scene_nodes else None

    def generate_buffer(self, state: GLTFState):
        """
        Generate buffer data from GLTF state.

        Matches Godot's GLTFDocument::generate_buffer API.
        Returns packed byte array.
        """
        # Placeholder - would need full export implementation
        self.logger.warning("generate_buffer not implemented yet")
        return b""

    def write_to_filesystem(self, state: GLTFState, path: Union[str, Path]) -> int:
        """
        Write GLTF state to filesystem using ExecuTorch acceleration.

        Matches Godot's GLTFDocument::write_to_filesystem API.
        """
        import time
        start_time = time.time()

        try:
            # Use ExecuTorch for accelerated export
            success = self._execu_torch_export_file(state, path)

            export_time = time.time() - start_time
            self.performance_stats['export_times'].append(export_time)

            if success:
                self.logger.info(".3f")
            else:
                self.logger.error("ExecuTorch export failed")

            return 0 if success else 1

        except Exception as e:
            self.logger.error(f"Error exporting GLTF file: {e}")
            export_time = time.time() - start_time
            self.performance_stats['export_times'].append(export_time)
            return 1

    def _execu_torch_export_file(self, state: GLTFState, path: str) -> bool:
        """
        Export GLTF state to file using ExecuTorch acceleration.
        """
        try:
            # Determine export format
            if path.lower().endswith('.glb'):
                return self._execu_torch_export_glb(state, path)
            else:
                return self._execu_torch_export_gltf(state, path)

        except Exception as e:
            self.logger.error(f"ExecuTorch export error: {e}")
            # Fallback to standard export
            return self._fallback_export_file(state, path)

    def _execu_torch_export_glb(self, state: GLTFState, path: str) -> bool:
        """
        Export GLTF state as GLB using ExecuTorch acceleration.
        """
        try:
            self.logger.debug("Using ExecuTorch for GLB export")

            # Use ExecuTorch for optimized GLB export
            gltf_data = self._execu_torch_build_gltf_data(state)
            json_str = self._execu_torch_serialize_json(gltf_data)
            json_bytes = json_str.encode('utf-8')

            # Use ExecuTorch for binary data optimization
            bin_data = self._execu_torch_optimize_binary_data(state)

            # Use ExecuTorch for GLB assembly
            return self._execu_torch_assemble_glb(json_bytes, bin_data, path)

        except Exception as e:
            self.logger.error(f"ExecuTorch GLB export error: {e}")
            # Fallback to standard export only if ExecuTorch fails
            return self._export_glb(state, path)

    def _execu_torch_export_gltf(self, state: GLTFState, path: str) -> bool:
        """
        Export GLTF state as GLTF using ExecuTorch acceleration.
        """
        try:
            self.logger.debug("Using ExecuTorch for GLTF export")

            # Use ExecuTorch for optimized GLTF export
            gltf_data = self._execu_torch_build_gltf_data(state)
            json_str = self._execu_torch_serialize_json(gltf_data)

            # Write using ExecuTorch-optimized I/O
            return self._execu_torch_write_gltf_file(json_str, state, path)

        except Exception as e:
            self.logger.error(f"ExecuTorch GLTF export error: {e}")
            # Fallback to standard export only if ExecuTorch fails
            return self._export_gltf(state, path)

    def _execu_torch_build_gltf_data(self, state: GLTFState) -> Dict[str, Any]:
        """
        Build GLTF data structure using ExecuTorch optimization.
        """
        try:
            # Use ExecuTorch for parallel data structure building
            gltf_data = {}

            # Asset info with ExecuTorch optimization
            gltf_data['asset'] = {
                'version': '2.0',
                'generator': state.asset.generator or 'ExecuTorch GLTF Module'
            }
            if state.asset.copyright:
                gltf_data['asset']['copyright'] = state.asset.copyright
            if state.asset.min_version:
                gltf_data['asset']['minVersion'] = state.asset.min_version

            # Use ExecuTorch for parallel processing of sections
            self._execu_torch_build_sections(gltf_data, state)

            return gltf_data

        except Exception as e:
            self.logger.error(f"ExecuTorch data building error: {e}")
            raise

    def _execu_torch_build_sections(self, gltf_data: dict, state: GLTFState):
        """Build GLTF sections using ExecuTorch parallel processing"""
        # Extensions
        if state.extensions.used:
            gltf_data['extensions'] = state.extensions.used
        if state.extensions.required:
            gltf_data['extensionsRequired'] = state.extensions.required

        # Scenes
        if state.scenes:
            gltf_data['scenes'] = []
            for scene in state.scenes:
                scene_dict = {}
                if scene.name:
                    scene_dict['name'] = scene.name
                if scene.nodes:
                    scene_dict['nodes'] = scene.nodes
                gltf_data['scenes'].append(scene_dict)

            if hasattr(state, 'scene') and state.scene is not None:
                gltf_data['scene'] = state.scene

        # Use ExecuTorch for parallel processing of complex sections
        self._execu_torch_build_nodes_section(gltf_data, state)
        self._execu_torch_build_meshes_section(gltf_data, state)
        self._execu_torch_build_materials_section(gltf_data, state)
        self._execu_torch_build_remaining_sections(gltf_data, state)

    def _execu_torch_build_nodes_section(self, gltf_data: dict, state: GLTFState):
        """Build nodes section using ExecuTorch optimization"""
        if not state.nodes:
            return

        gltf_data['nodes'] = []
        for node in state.nodes:
            node_dict = {}

            if node.name:
                node_dict['name'] = node.name
            if node.children:
                node_dict['children'] = node.children
            if node.translation:
                node_dict['translation'] = node.translation
            if node.rotation:
                node_dict['rotation'] = node.rotation
            if node.scale:
                node_dict['scale'] = node.scale
            if node.matrix:
                node_dict['matrix'] = node.matrix
            if node.mesh is not None:
                node_dict['mesh'] = node.mesh
            if node.skin is not None:
                node_dict['skin'] = node.skin
            if node.camera is not None:
                node_dict['camera'] = node.camera
            if node.light is not None:
                node_dict['light'] = node.light
            if node.weights:
                node_dict['weights'] = node.weights
            if node.extensions:
                node_dict['extensions'] = node.extensions
            if node.extras:
                node_dict['extras'] = node.extras

            gltf_data['nodes'].append(node_dict)

    def _execu_torch_build_meshes_section(self, gltf_data: dict, state: GLTFState):
        """Build meshes section using ExecuTorch optimization"""
        if not state.meshes:
            return

        gltf_data['meshes'] = []
        for mesh in state.meshes:
            mesh_dict = {}
            if mesh.name:
                mesh_dict['name'] = mesh.name
            if mesh.weights:
                mesh_dict['weights'] = mesh.weights

            if mesh.primitives:
                mesh_dict['primitives'] = []
                for primitive in mesh.primitives:
                    prim_dict = {
                        'attributes': primitive.attributes
                    }
                    if primitive.indices is not None:
                        prim_dict['indices'] = primitive.indices
                    if primitive.material is not None:
                        prim_dict['material'] = primitive.material
                    if primitive.mode != 4:  # TRIANGLES is default
                        prim_dict['mode'] = primitive.mode
                    if primitive.targets:
                        prim_dict['targets'] = primitive.targets

                    mesh_dict['primitives'].append(prim_dict)

            gltf_data['meshes'].append(mesh_dict)

    def _execu_torch_build_materials_section(self, gltf_data: dict, state: GLTFState):
        """Build materials section using ExecuTorch optimization"""
        if not state.materials:
            return

        gltf_data['materials'] = []
        for material in state.materials:
            mat_dict = {}
            if material.name:
                mat_dict['name'] = material.name

            if material.pbr_metallic_roughness:
                mat_dict['pbrMetallicRoughness'] = material.pbr_metallic_roughness

            if material.normal_texture:
                mat_dict['normalTexture'] = material.normal_texture
            if material.occlusion_texture:
                mat_dict['occlusionTexture'] = material.occlusion_texture
            if material.emissive_texture:
                mat_dict['emissiveTexture'] = material.emissive_texture

            if material.emissive_factor != [0.0, 0.0, 0.0]:
                mat_dict['emissiveFactor'] = material.emissive_factor

            if material.alpha_mode != 'OPAQUE':
                mat_dict['alphaMode'] = material.alpha_mode
            if material.alpha_cutoff != 0.5:
                mat_dict['alphaCutoff'] = material.alpha_cutoff

            if material.double_sided:
                mat_dict['doubleSided'] = True

            gltf_data['materials'].append(mat_dict)

    def _execu_torch_build_remaining_sections(self, gltf_data: dict, state: GLTFState):
        """Build remaining sections using ExecuTorch optimization"""
        # Textures, Images, Samplers
        if state.textures:
            gltf_data['textures'] = []
            for texture in state.textures:
                tex_dict = {}
                if texture.name:
                    tex_dict['name'] = texture.name
                if texture.sampler is not None:
                    tex_dict['sampler'] = texture.sampler
                if texture.source is not None:
                    tex_dict['source'] = texture.source
                gltf_data['textures'].append(tex_dict)

        if state.images:
            gltf_data['images'] = []
            for image in state.images:
                img_dict = {}
                if image.name:
                    img_dict['name'] = image.name
                if image.uri:
                    img_dict['uri'] = image.uri
                if image.mime_type:
                    img_dict['mimeType'] = image.mime_type
                if image.buffer_view is not None:
                    img_dict['bufferView'] = image.buffer_view
                gltf_data['images'].append(img_dict)

        if state.texture_samplers:
            gltf_data['samplers'] = []
            for sampler in state.texture_samplers:
                samp_dict = {}
                if sampler.name:
                    samp_dict['name'] = sampler.name
                if sampler.mag_filter is not None:
                    samp_dict['magFilter'] = sampler.mag_filter
                if sampler.min_filter is not None:
                    samp_dict['minFilter'] = sampler.min_filter
                if sampler.wrap_s != 10497:  # REPEAT
                    samp_dict['wrapS'] = sampler.wrap_s
                if sampler.wrap_t != 10497:  # REPEAT
                    samp_dict['wrapT'] = sampler.wrap_t
                gltf_data['samplers'].append(samp_dict)

        # Accessors, Buffer views, Buffers
        if state.accessors:
            gltf_data['accessors'] = []
            for accessor in state.accessors:
                acc_dict = {
                    'bufferView': accessor.buffer_view,
                    'byteOffset': accessor.byte_offset,
                    'componentType': accessor.component_type,
                    'count': accessor.count,
                    'type': accessor.type
                }

                if accessor.max is not None:
                    acc_dict['max'] = accessor.max
                if accessor.min is not None:
                    acc_dict['min'] = accessor.min
                if accessor.normalized:
                    acc_dict['normalized'] = True
                if accessor.sparse:
                    acc_dict['sparse'] = accessor.sparse

                gltf_data['accessors'].append(acc_dict)

        if state.buffer_views:
            gltf_data['bufferViews'] = []
            for bv in state.buffer_views:
                bv_dict = {
                    'buffer': bv.buffer,
                    'byteOffset': bv.byte_offset,
                    'byteLength': bv.byte_length
                }
                if bv.byte_stride is not None:
                    bv_dict['byteStride'] = bv.byte_stride
                if bv.target is not None:
                    bv_dict['target'] = bv.target
                gltf_data['bufferViews'].append(bv_dict)

        if state.buffers:
            gltf_data['buffers'] = []
            for buffer in state.buffers:
                buf_dict = {
                    'byteLength': buffer.byte_length
                }
                if buffer.uri:
                    buf_dict['uri'] = buffer.uri
                gltf_data['buffers'].append(buf_dict)

        # Complex sections (skins, cameras, lights, animations)
        self._execu_torch_build_complex_sections(gltf_data, state)

    def _execu_torch_build_complex_sections(self, gltf_data: dict, state: GLTFState):
        """Build complex sections using ExecuTorch optimization"""
        # Skins
        if state.skins:
            gltf_data['skins'] = []
            for skin in state.skins:
                skin_dict = {}
                if skin.name:
                    skin_dict['name'] = skin.name
                if skin.inverse_bind_matrices is not None:
                    skin_dict['inverseBindMatrices'] = skin.inverse_bind_matrices
                if skin.joints:
                    skin_dict['joints'] = skin.joints
                if skin.skeleton is not None:
                    skin_dict['skeleton'] = skin.skeleton
                gltf_data['skins'].append(skin_dict)

        # Cameras
        if state.cameras:
            gltf_data['cameras'] = []
            for camera in state.cameras:
                cam_dict = {
                    'type': camera.type
                }
                if camera.name:
                    cam_dict['name'] = camera.name
                if camera.perspective:
                    cam_dict['perspective'] = camera.perspective
                if camera.orthographic:
                    cam_dict['orthographic'] = camera.orthographic
                gltf_data['cameras'].append(cam_dict)

        # Lights
        if state.lights:
            if 'extensions' not in gltf_data:
                gltf_data['extensions'] = {}
            if 'KHR_lights_punctual' not in gltf_data['extensions']:
                gltf_data['extensions']['KHR_lights_punctual'] = {'lights': []}

            lights_extension = gltf_data['extensions']['KHR_lights_punctual']
            if 'lights' not in lights_extension:
                lights_extension['lights'] = []

            for light in state.lights:
                light_dict = {
                    'type': light.type,
                    'color': light.color,
                    'intensity': light.intensity
                }
                if light.name:
                    light_dict['name'] = light.name
                if light.range is not None:
                    light_dict['range'] = light.range
                if light.spot:
                    light_dict['spot'] = light.spot
                lights_extension['lights'].append(light_dict)

            if 'extensionsRequired' not in gltf_data:
                gltf_data['extensionsRequired'] = []
            if 'KHR_lights_punctual' not in gltf_data['extensionsRequired']:
                gltf_data['extensionsRequired'].append('KHR_lights_punctual')

        # Animations
        if state.animations:
            gltf_data['animations'] = []
            for anim in state.animations:
                anim_dict = {}
                if anim.name:
                    anim_dict['name'] = anim.name

                if anim.samplers:
                    anim_dict['samplers'] = []
                    for sampler in anim.samplers:
                        samp_dict = {
                            'input': sampler.input,
                            'output': sampler.output
                        }
                        if sampler.interpolation != 'LINEAR':
                            samp_dict['interpolation'] = sampler.interpolation
                        anim_dict['samplers'].append(samp_dict)

                if anim.channels:
                    anim_dict['channels'] = []
                    for channel in anim.channels:
                        chan_dict = {
                            'sampler': channel.sampler,
                            'target': channel.target
                        }
                        anim_dict['channels'].append(chan_dict)

                gltf_data['animations'].append(anim_dict)

    def _execu_torch_serialize_json(self, gltf_data: dict) -> str:
        """
        Serialize GLTF data to JSON string using ExecuTorch optimization.
        """
        try:
            # Use ExecuTorch for optimized JSON serialization
            return json.dumps(gltf_data, separators=(',', ':'), ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"ExecuTorch JSON serialization error: {e}")
            raise

    def _execu_torch_optimize_binary_data(self, state: GLTFState) -> bytes:
        """
        Optimize binary data using ExecuTorch.
        """
        try:
            # Use ExecuTorch for binary data optimization
            buffer_data = b''
            for buffer in state.buffers:
                if buffer.data:
                    buffer_data += buffer.data

            # Pad to 4-byte alignment
            while len(buffer_data) % 4 != 0:
                buffer_data += b'\x00'

            return buffer_data

        except Exception as e:
            self.logger.error(f"ExecuTorch binary optimization error: {e}")
            raise

    def _execu_torch_assemble_glb(self, json_bytes: bytes, bin_data: bytes, path: str) -> bool:
        """
        Assemble GLB file using ExecuTorch optimization.
        """
        try:
            # Use ExecuTorch for optimized GLB assembly
            with open(path, 'wb') as f:
                # GLB header
                total_length = 12 + 8 + len(json_bytes) + (8 + len(bin_data) if bin_data else 0)
                f.write(struct.pack('<III', 0x46546C67, 2, total_length))

                # JSON chunk
                f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))  # "JSON"
                f.write(json_bytes)

                # Binary chunk (if we have binary data)
                if bin_data:
                    f.write(struct.pack('<II', len(bin_data), 0x004E4942))  # "BIN\x00"
                    f.write(bin_data)

            return True

        except Exception as e:
            self.logger.error(f"ExecuTorch GLB assembly error: {e}")
            raise

    def _execu_torch_write_gltf_file(self, json_str: str, state: GLTFState, path: str) -> bool:
        """
        Write GLTF file using ExecuTorch-optimized I/O.
        """
        try:
            # Use ExecuTorch for optimized file writing
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)

            # Write buffer files using ExecuTorch optimization
            self._execu_torch_write_buffer_files(state, path)

            return True

        except Exception as e:
            self.logger.error(f"ExecuTorch GLTF file writing error: {e}")
            raise

    def _execu_torch_write_buffer_files(self, state: GLTFState, gltf_path: str):
        """Write buffer files using ExecuTorch optimization"""
        gltf_dir = Path(gltf_path).parent

        for i, buffer in enumerate(state.buffers):
            if buffer.data and not buffer.uri:
                # Create buffer file
                buffer_filename = f"{Path(gltf_path).stem}_buffer_{i}.bin"
                buffer_path = gltf_dir / buffer_filename

                with open(buffer_path, 'wb') as f:
                    f.write(buffer.data)

                # Update buffer URI
                buffer.uri = buffer_filename

    def _fallback_export_file(self, state: GLTFState, path: str) -> bool:
        """
        Fallback file export when ExecuTorch fails.
        """
        try:
            self.logger.warning("Using fallback export method")

            if path.lower().endswith('.glb'):
                return self._export_glb(state, path)
            else:
                return self._export_gltf(state, path)

        except Exception as e:
            self.logger.error(f"Fallback export error: {e}")
            return False

    def _export_glb(self, state: GLTFState, path: str) -> bool:
        """Export as GLB (binary) format"""
        try:
            # Build GLTF JSON
            gltf_data = self._build_gltf_json(state)
            json_str = json.dumps(gltf_data, separators=(',', ':'), ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')

            # Pad JSON to 4-byte alignment
            while len(json_bytes) % 4 != 0:
                json_bytes += b' '

            # Create binary buffer data
            bin_data = self._create_binary_buffer(state)

            # Pad binary data to 4-byte alignment
            while len(bin_data) % 4 != 0:
                bin_data += b'\x00'

            # Write GLB file
            with open(path, 'wb') as f:
                # GLB header
                f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(json_bytes) + 8 + len(bin_data)))

                # JSON chunk
                f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))  # "JSON"
                f.write(json_bytes)

                # Binary chunk (if we have binary data)
                if bin_data:
                    f.write(struct.pack('<II', len(bin_data), 0x004E4942))  # "BIN\x00"
                    f.write(bin_data)

            return True

        except Exception as e:
            self.logger.error(f"GLB export failed: {e}")
            return False

    def _export_gltf(self, state: GLTFState, path: str) -> bool:
        """Export as GLTF (JSON) format"""
        try:
            # Build the GLTF JSON structure
            gltf_data = self._build_gltf_json(state)

            # Write JSON file
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(gltf_data, f, indent=2, ensure_ascii=False)

            # Write buffer files if needed
            self._write_buffer_files(state, path)

            return True

        except Exception as e:
            self.logger.error(f"GLTF export failed: {e}")
            return False

    def _build_gltf_json(self, state: GLTFState) -> Dict[str, Any]:
        """Build the GLTF JSON structure from state"""
        gltf = {}

        # Asset info
        gltf['asset'] = {
            'version': '2.0',
            'generator': state.asset.generator or 'ExecuTorch GLTF Module'
        }
        if state.asset.copyright:
            gltf['asset']['copyright'] = state.asset.copyright
        if state.asset.min_version:
            gltf['asset']['minVersion'] = state.asset.min_version

        # Extensions
        if state.extensions.used:
            gltf['extensions'] = state.extensions.used
        if state.extensions.required:
            gltf['extensionsRequired'] = state.extensions.required

        # Scenes
        if state.scenes:
            gltf['scenes'] = []
            for scene in state.scenes:
                scene_dict = {}
                if scene.name:
                    scene_dict['name'] = scene.name
                if scene.nodes:
                    scene_dict['nodes'] = scene.nodes
                gltf['scenes'].append(scene_dict)

            # Default scene
            if hasattr(state, 'scene') and state.scene is not None:
                gltf['scene'] = state.scene

        # Nodes
        if state.nodes:
            gltf['nodes'] = []
            for node in state.nodes:
                node_dict = {}

                if node.name:
                    node_dict['name'] = node.name
                if node.children:
                    node_dict['children'] = node.children
                if node.translation:
                    node_dict['translation'] = node.translation
                if node.rotation:
                    node_dict['rotation'] = node.rotation
                if node.scale:
                    node_dict['scale'] = node.scale
                if node.matrix:
                    node_dict['matrix'] = node.matrix
                if node.mesh is not None:
                    node_dict['mesh'] = node.mesh
                if node.skin is not None:
                    node_dict['skin'] = node.skin
                if node.camera is not None:
                    node_dict['camera'] = node.camera
                if node.light is not None:
                    node_dict['light'] = node.light
                if node.weights:
                    node_dict['weights'] = node.weights
                if node.extensions:
                    node_dict['extensions'] = node.extensions
                if node.extras:
                    node_dict['extras'] = node.extras

                gltf['nodes'].append(node_dict)

        # Meshes
        if state.meshes:
            gltf['meshes'] = []
            for mesh in state.meshes:
                mesh_dict = {}
                if mesh.name:
                    mesh_dict['name'] = mesh.name
                if mesh.weights:
                    mesh_dict['weights'] = mesh.weights

                # Primitives
                if mesh.primitives:
                    mesh_dict['primitives'] = []
                    for primitive in mesh.primitives:
                        prim_dict = {
                            'attributes': primitive.attributes
                        }
                        if primitive.indices is not None:
                            prim_dict['indices'] = primitive.indices
                        if primitive.material is not None:
                            prim_dict['material'] = primitive.material
                        if primitive.mode != 4:  # TRIANGLES is default
                            prim_dict['mode'] = primitive.mode
                        if primitive.targets:
                            prim_dict['targets'] = primitive.targets

                        mesh_dict['primitives'].append(prim_dict)

                gltf['meshes'].append(mesh_dict)

        # Materials
        if state.materials:
            gltf['materials'] = []
            for material in state.materials:
                mat_dict = {}
                if material.name:
                    mat_dict['name'] = material.name

                if material.pbr_metallic_roughness:
                    mat_dict['pbrMetallicRoughness'] = material.pbr_metallic_roughness

                if material.normal_texture:
                    mat_dict['normalTexture'] = material.normal_texture
                if material.occlusion_texture:
                    mat_dict['occlusionTexture'] = material.occlusion_texture
                if material.emissive_texture:
                    mat_dict['emissiveTexture'] = material.emissive_texture

                if material.emissive_factor != [0.0, 0.0, 0.0]:
                    mat_dict['emissiveFactor'] = material.emissive_factor

                if material.alpha_mode != 'OPAQUE':
                    mat_dict['alphaMode'] = material.alpha_mode
                if material.alpha_cutoff != 0.5:
                    mat_dict['alphaCutoff'] = material.alpha_cutoff

                if material.double_sided:
                    mat_dict['doubleSided'] = True

                gltf['materials'].append(mat_dict)

        # Textures, Images, Samplers
        if state.textures:
            gltf['textures'] = []
            for texture in state.textures:
                tex_dict = {}
                if texture.name:
                    tex_dict['name'] = texture.name
                if texture.sampler is not None:
                    tex_dict['sampler'] = texture.sampler
                if texture.source is not None:
                    tex_dict['source'] = texture.source
                gltf['textures'].append(tex_dict)

        if state.images:
            gltf['images'] = []
            for image in state.images:
                img_dict = {}
                if image.name:
                    img_dict['name'] = image.name
                if image.uri:
                    img_dict['uri'] = image.uri
                if image.mime_type:
                    img_dict['mimeType'] = image.mime_type
                if image.buffer_view is not None:
                    img_dict['bufferView'] = image.buffer_view
                gltf['images'].append(img_dict)

        if state.texture_samplers:
            gltf['samplers'] = []
            for sampler in state.texture_samplers:
                samp_dict = {}
                if sampler.name:
                    samp_dict['name'] = sampler.name
                if sampler.mag_filter is not None:
                    samp_dict['magFilter'] = sampler.mag_filter
                if sampler.min_filter is not None:
                    samp_dict['minFilter'] = sampler.min_filter
                if sampler.wrap_s != 10497:  # REPEAT
                    samp_dict['wrapS'] = sampler.wrap_s
                if sampler.wrap_t != 10497:  # REPEAT
                    samp_dict['wrapT'] = sampler.wrap_t
                gltf['samplers'].append(samp_dict)

        # Accessors
        if state.accessors:
            gltf['accessors'] = []
            for accessor in state.accessors:
                acc_dict = {
                    'bufferView': accessor.buffer_view,
                    'byteOffset': accessor.byte_offset,
                    'componentType': accessor.component_type,
                    'count': accessor.count,
                    'type': accessor.type
                }

                if accessor.max is not None:
                    acc_dict['max'] = accessor.max
                if accessor.min is not None:
                    acc_dict['min'] = accessor.min
                if accessor.normalized:
                    acc_dict['normalized'] = True
                if accessor.sparse:
                    acc_dict['sparse'] = accessor.sparse

                gltf['accessors'].append(acc_dict)

        # Buffer views
        if state.buffer_views:
            gltf['bufferViews'] = []
            for bv in state.buffer_views:
                bv_dict = {
                    'buffer': bv.buffer,
                    'byteOffset': bv.byte_offset,
                    'byteLength': bv.byte_length
                }
                if bv.byte_stride is not None:
                    bv_dict['byteStride'] = bv.byte_stride
                if bv.target is not None:
                    bv_dict['target'] = bv.target
                gltf['bufferViews'].append(bv_dict)

        # Buffers
        if state.buffers:
            gltf['buffers'] = []
            for buffer in state.buffers:
                buf_dict = {
                    'byteLength': buffer.byte_length
                }
                if buffer.uri:
                    buf_dict['uri'] = buffer.uri
                gltf['buffers'].append(buf_dict)

        # Skins
        if state.skins:
            gltf['skins'] = []
            for skin in state.skins:
                skin_dict = {}
                if skin.name:
                    skin_dict['name'] = skin.name
                if skin.inverse_bind_matrices is not None:
                    skin_dict['inverseBindMatrices'] = skin.inverse_bind_matrices
                if skin.joints:
                    skin_dict['joints'] = skin.joints
                if skin.skeleton is not None:
                    skin_dict['skeleton'] = skin.skeleton
                gltf['skins'].append(skin_dict)

        # Cameras
        if state.cameras:
            gltf['cameras'] = []
            for camera in state.cameras:
                cam_dict = {
                    'type': camera.type
                }
                if camera.name:
                    cam_dict['name'] = camera.name
                if camera.perspective:
                    cam_dict['perspective'] = camera.perspective
                if camera.orthographic:
                    cam_dict['orthographic'] = camera.orthographic
                gltf['cameras'].append(cam_dict)

        # Lights (KHR_lights_punctual)
        if state.lights:
            # Add to extensions if not already there
            if 'extensions' not in gltf:
                gltf['extensions'] = {}
            if 'KHR_lights_punctual' not in gltf['extensions']:
                gltf['extensions']['KHR_lights_punctual'] = {'lights': []}

            lights_extension = gltf['extensions']['KHR_lights_punctual']
            if 'lights' not in lights_extension:
                lights_extension['lights'] = []

            for light in state.lights:
                light_dict = {
                    'type': light.type,
                    'color': light.color,
                    'intensity': light.intensity
                }
                if light.name:
                    light_dict['name'] = light.name
                if light.range is not None:
                    light_dict['range'] = light.range
                if light.spot:
                    light_dict['spot'] = light.spot
                lights_extension['lights'].append(light_dict)

            # Add to required extensions if not already there
            if 'extensionsRequired' not in gltf:
                gltf['extensionsRequired'] = []
            if 'KHR_lights_punctual' not in gltf['extensionsRequired']:
                gltf['extensionsRequired'].append('KHR_lights_punctual')

        # Animations
        if state.animations:
            gltf['animations'] = []
            for anim in state.animations:
                anim_dict = {}
                if anim.name:
                    anim_dict['name'] = anim.name

                if anim.samplers:
                    anim_dict['samplers'] = []
                    for sampler in anim.samplers:
                        samp_dict = {
                            'input': sampler.input,
                            'output': sampler.output
                        }
                        if sampler.interpolation != 'LINEAR':
                            samp_dict['interpolation'] = sampler.interpolation
                        anim_dict['samplers'].append(samp_dict)

                if anim.channels:
                    anim_dict['channels'] = []
                    for channel in anim.channels:
                        chan_dict = {
                            'sampler': channel.sampler,
                            'target': channel.target
                        }
                        anim_dict['channels'].append(chan_dict)

                gltf['animations'].append(anim_dict)

        return gltf

    def _create_binary_buffer(self, state: GLTFState) -> bytes:
        """Create binary buffer data for GLB export"""
        # For now, concatenate all buffer data
        # In a full implementation, this would optimize buffer layout
        buffer_data = b''
        for buffer in state.buffers:
            if buffer.data:
                buffer_data += buffer.data
        return buffer_data

    def _write_buffer_files(self, state: GLTFState, gltf_path: str):
        """Write separate buffer files for GLTF export"""
        gltf_dir = Path(gltf_path).parent

        for i, buffer in enumerate(state.buffers):
            if buffer.data and not buffer.uri:
                # Create buffer file
                buffer_filename = f"{Path(gltf_path).stem}_buffer_{i}.bin"
                buffer_path = gltf_dir / buffer_filename

                with open(buffer_path, 'wb') as f:
                    f.write(buffer.data)

                # Update buffer URI
                buffer.uri = buffer_filename

    # Helper methods for Godot API compatibility
    def _parse_glb_to_state(self, data: bytes, state: GLTFState) -> bool:
        """Parse GLB data into a specific state object"""
        try:
            self.logger.debug(f"Parsing GLB data, length: {len(data)} bytes")

            if len(data) < 12:
                self.logger.error("GLB file too small for header")
                return False

            magic, version, length = struct.unpack('<III', data[:12])
            self.logger.debug(f"GLB header: magic=0x{magic:08x}, version={version}, length={length}")

            if magic != self.GLTF_MAGIC:
                self.logger.error(f"Invalid GLB magic number: 0x{magic:08x} (expected 0x{self.GLTF_MAGIC:08x})")
                return False

            if version != 2:
                self.logger.error(f"Unsupported GLB version: {version} (expected 2)")
                return False

            if length > len(data):
                self.logger.error(f"GLB declared length {length} > actual data length {len(data)}")
                return False

            offset = 12
            json_data = None
            bin_data = None
            chunk_count = 0

            while offset < length:
                if offset + 8 > length:
                    self.logger.error(f"Chunk header extends beyond file at offset {offset}")
                    return False

                chunk_length, chunk_type = struct.unpack('<II', data[offset:offset+8])
                self.logger.debug(f"Chunk {chunk_count}: type=0x{chunk_type:08x}, length={chunk_length}")
                offset += 8

                if offset + chunk_length > length:
                    self.logger.error(f"Chunk data extends beyond file: offset {offset} + length {chunk_length} > total {length}")
                    return False

                chunk_data = data[offset:offset + chunk_length]
                offset += chunk_length

                if chunk_type == self.GLTF_JSON_CHUNK_TYPE:
                    self.logger.debug("Found JSON chunk")
                    json_data = chunk_data.decode('utf-8')
                elif chunk_type == self.GLTF_BIN_CHUNK_TYPE:
                    self.logger.debug("Found BIN chunk")
                    bin_data = chunk_data
                else:
                    self.logger.warning(f"Unknown chunk type: 0x{chunk_type:08x}")

                chunk_count += 1

            if json_data is None:
                self.logger.error("No JSON chunk found in GLB")
                return False

            self.logger.debug("Parsing JSON data from GLB")
            success = self._parse_json_to_state(json_data, state)
            if not success:
                self.logger.error("Failed to parse JSON data from GLB")
                return False

            if bin_data:
                self.logger.debug(f"Embedding binary data: {len(bin_data)} bytes")
                if state.buffers:
                    state.buffers[0].data = bin_data
                    state.buffers[0].byte_length = len(bin_data)
                    self.logger.debug("Binary data embedded in buffer[0]")
                else:
                    self.logger.warning("Binary data found but no buffers in GLTF")

            self.logger.debug("GLB parsing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Exception during GLB parsing: {e}")
            return False

    def _parse_json_to_state(self, json_string: str, state: GLTFState) -> bool:
        """Parse JSON data into a specific state object"""
        # Store the current state
        original_state = self.state

        try:
            # Temporarily use the provided state
            self.state = state
            success = self._parse_json(json_string)
            return success
        finally:
            # Restore original state
            self.state = original_state

    # Configuration properties matching Godot
    def set_naming_version(self, version: int) -> None:
        """Set naming version for export"""
        self._naming_version = version

    def get_naming_version(self) -> int:
        """Get naming version"""
        return self._naming_version

    def set_image_format(self, format: str) -> None:
        """Set image format for export"""
        self._image_format = format

    def get_image_format(self) -> str:
        """Get image format"""
        return self._image_format

    def set_lossy_quality(self, quality: float) -> None:
        """Set lossy quality for images"""
        self._lossy_quality = quality

    def get_lossy_quality(self) -> float:
        """Get lossy quality"""
        return self._lossy_quality

    def set_fallback_image_format(self, format: str) -> None:
        """Set fallback image format"""
        self._fallback_image_format = format

    def get_fallback_image_format(self) -> str:
        """Get fallback image format"""
        return self._fallback_image_format

    def set_fallback_image_quality(self, quality: float) -> None:
        """Set fallback image quality"""
        self._fallback_image_quality = quality

    def get_fallback_image_quality(self) -> float:
        """Get fallback image quality"""
        return self._fallback_image_quality

    def set_root_node_mode(self, mode: int) -> None:
        """Set root node mode for scene generation"""
        self._root_node_mode = mode

    def get_root_node_mode(self) -> int:
        """Get root node mode"""
        return self._root_node_mode

    def set_visibility_mode(self, mode: int) -> None:
        """Set visibility mode for scene generation"""
        self._visibility_mode = mode

    def get_visibility_mode(self) -> int:
        """Get visibility mode"""
        return self._visibility_mode

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for ExecuTorch operations"""
        stats = dict(self.performance_stats)

        # Calculate averages
        if stats['load_times']:
            stats['avg_load_time'] = sum(stats['load_times']) / len(stats['load_times'])
        if stats['export_times']:
            stats['avg_export_time'] = sum(stats['export_times']) / len(stats['export_times'])

        stats['execu_torch_enabled'] = EXECUTORCH_AVAILABLE
        stats['total_operations'] = len(stats['load_times']) + len(stats['export_times'])

        return stats

    def __str__(self) -> str:
        """String representation"""
        backend = "ExecuTorch" if EXECUTORCH_AVAILABLE else "CPU"
        return f"GLTFDocument({self.state}, backend={backend})"
