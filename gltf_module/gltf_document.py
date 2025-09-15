"""
GLTF Document Parser

This module contains the GLTFDocument class which provides the main interface
for parsing GLTF files, equivalent to Godot's GLTFDocument class.
"""

import json
import base64
import struct
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from urllib.parse import unquote
from .gltf_state import GLTFState
from .structures import *
from .logger import get_logger


class GLTFDocument:
    """
    Main GLTF document parser and processor.

    This class provides methods for loading GLTF files (both JSON and binary GLB),
    parsing their contents, and generating scene data structures.

    Matches the API of Godot's GLTFDocument class.
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
        """Initialize the GLTF document parser"""
        self.logger = get_logger('document')
        self.state = GLTFState()
        # Configuration properties matching Godot
        self._naming_version = 2
        self._image_format = "PNG"
        self._lossy_quality = 0.75
        self._fallback_image_format = "None"
        self._fallback_image_quality = 0.25
        self._root_node_mode = self.RootNodeMode.ROOT_NODE_MODE_SINGLE_ROOT
        self._visibility_mode = self.VisibilityMode.VISIBILITY_MODE_INCLUDE_REQUIRED

    def load_from_file(self, file_path: str) -> bool:
        """
        Load a GLTF file from disk.

        Args:
            file_path: Path to the GLTF file (.gltf or .glb)

        Returns:
            True if loading was successful, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False

            self.state.filename = path.name
            self.state.base_path = str(path.parent)

            with open(path, 'rb') as f:
                data = f.read()

            if self._is_glb_file(data):
                return self._parse_glb(data)
            else:
                return self._parse_json(data.decode('utf-8'))

        except Exception as e:
            self.logger.error(f"Error loading GLTF file: {e}")
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
    def append_from_file(self, path: str, state: GLTFState, flags: int = 0, base_path: str = "") -> int:
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

    def write_to_filesystem(self, state: GLTFState, path: str) -> int:
        """
        Write GLTF state to filesystem.

        Matches Godot's GLTFDocument::write_to_filesystem API.
        """
        # Placeholder - would need full export implementation
        self.logger.warning("write_to_filesystem not implemented yet")
        return 1

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
    def set_naming_version(self, version: int):
        """Set naming version for export"""
        self._naming_version = version

    def get_naming_version(self) -> int:
        """Get naming version"""
        return self._naming_version

    def set_image_format(self, format: str):
        """Set image format for export"""
        self._image_format = format

    def get_image_format(self) -> str:
        """Get image format"""
        return self._image_format

    def set_lossy_quality(self, quality: float):
        """Set lossy quality for images"""
        self._lossy_quality = quality

    def get_lossy_quality(self) -> float:
        """Get lossy quality"""
        return self._lossy_quality

    def set_fallback_image_format(self, format: str):
        """Set fallback image format"""
        self._fallback_image_format = format

    def get_fallback_image_format(self) -> str:
        """Get fallback image format"""
        return self._fallback_image_format

    def set_fallback_image_quality(self, quality: float):
        """Set fallback image quality"""
        self._fallback_image_quality = quality

    def get_fallback_image_quality(self) -> float:
        """Get fallback image quality"""
        return self._fallback_image_quality

    def set_root_node_mode(self, mode):
        """Set root node mode for scene generation"""
        self._root_node_mode = mode

    def get_root_node_mode(self):
        """Get root node mode"""
        return self._root_node_mode

    def set_visibility_mode(self, mode):
        """Set visibility mode for scene generation"""
        self._visibility_mode = mode

    def get_visibility_mode(self):
        """Get visibility mode"""
        return self._visibility_mode

    def __str__(self) -> str:
        """String representation"""
        return f"GLTFDocument({self.state})"
