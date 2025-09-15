"""
GLTF State Management

This module contains the GLTFState class which manages the parsed GLTF data,
equivalent to Godot's GLTFState class.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from .structures import *


@dataclass
class GLTFState:
    """
    Main container for parsed GLTF data.

    This class holds all the parsed GLTF data structures and provides
    methods for managing and accessing them.
    """

    # Core GLTF data
    json: Dict = field(default_factory=dict)
    gltf_version: str = "2.0"

    # Asset information
    asset: GLTFAsset = field(default_factory=GLTFAsset)

    # Main data arrays
    nodes: List[GLTFNode] = field(default_factory=list)
    scenes: List[GLTFScene] = field(default_factory=list)
    meshes: List[GLTFMesh] = field(default_factory=list)
    materials: List[GLTFMaterial] = field(default_factory=list)
    textures: List[GLTFTexture] = field(default_factory=list)
    images: List[GLTFImage] = field(default_factory=list)
    texture_samplers: List[GLTFTextureSampler] = field(default_factory=list)
    buffers: List[GLTFBuffer] = field(default_factory=list)
    buffer_views: List[GLTFBufferView] = field(default_factory=list)
    accessors: List[GLTFAccessor] = field(default_factory=list)
    skins: List[GLTFSkin] = field(default_factory=list)
    cameras: List[GLTFCamera] = field(default_factory=list)
    lights: List[GLTFLight] = field(default_factory=list)
    animations: List[GLTFAnimation] = field(default_factory=list)

    # Extensions
    extensions: GLTFExtensions = field(default_factory=GLTFExtensions)

    # Scene management
    scene: Optional[int] = None
    root_nodes: List[int] = field(default_factory=list)

    # Additional metadata
    filename: str = ""
    base_path: str = ""
    major_version: int = 2
    minor_version: int = 0

    # Unique name tracking (for generating unique names)
    unique_names: Dict[str, int] = field(default_factory=dict)
    unique_animation_names: Dict[str, int] = field(default_factory=dict)
    unique_bone_names: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values"""
        if not self.asset.generator:
            self.asset.generator = "Godot GLTF Python Module"

    def get_scene_count(self) -> int:
        """Get the number of scenes"""
        return len(self.scenes)

    def get_node_count(self) -> int:
        """Get the number of nodes"""
        return len(self.nodes)

    def get_mesh_count(self) -> int:
        """Get the number of meshes"""
        return len(self.meshes)

    def get_material_count(self) -> int:
        """Get the number of materials"""
        return len(self.materials)

    def get_texture_count(self) -> int:
        """Get the number of textures"""
        return len(self.textures)

    def get_buffer_count(self) -> int:
        """Get the number of buffers"""
        return len(self.buffers)

    def get_accessor_count(self) -> int:
        """Get the number of accessors"""
        return len(self.accessors)

    def get_animation_count(self) -> int:
        """Get the number of animations"""
        return len(self.animations)

    def get_skin_count(self) -> int:
        """Get the number of skins"""
        return len(self.skins)

    def get_camera_count(self) -> int:
        """Get the number of cameras"""
        return len(self.cameras)

    def get_light_count(self) -> int:
        """Get the number of lights"""
        return len(self.lights)

    def add_node(self, node: GLTFNode) -> GLTFNodeIndex:
        """Add a node and return its index"""
        index = len(self.nodes)
        self.nodes.append(node)
        return index

    def add_mesh(self, mesh: GLTFMesh) -> GLTFMeshIndex:
        """Add a mesh and return its index"""
        index = len(self.meshes)
        self.meshes.append(mesh)
        return index

    def add_material(self, material: GLTFMaterial) -> GLTFMaterialIndex:
        """Add a material and return its index"""
        index = len(self.materials)
        self.materials.append(material)
        return index

    def add_texture(self, texture: GLTFTexture) -> GLTFTextureIndex:
        """Add a texture and return its index"""
        index = len(self.textures)
        self.textures.append(texture)
        return index

    def add_buffer(self, buffer: GLTFBuffer) -> GLTFBufferIndex:
        """Add a buffer and return its index"""
        index = len(self.buffers)
        self.buffers.append(buffer)
        return index

    def add_accessor(self, accessor: GLTFAccessor) -> GLTFAccessorIndex:
        """Add an accessor and return its index"""
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index

    def add_animation(self, animation: GLTFAnimation) -> GLTFAnimationIndex:
        """Add an animation and return its index"""
        index = len(self.animations)
        self.animations.append(animation)
        return index

    def get_node(self, index: GLTFNodeIndex) -> Optional[GLTFNode]:
        """Get a node by index"""
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None

    def get_mesh(self, index: GLTFMeshIndex) -> Optional[GLTFMesh]:
        """Get a mesh by index"""
        if 0 <= index < len(self.meshes):
            return self.meshes[index]
        return None

    def get_material(self, index: GLTFMaterialIndex) -> Optional[GLTFMaterial]:
        """Get a material by index"""
        if 0 <= index < len(self.materials):
            return self.materials[index]
        return None

    def get_texture(self, index: GLTFTextureIndex) -> Optional[GLTFTexture]:
        """Get a texture by index"""
        if 0 <= index < len(self.textures):
            return self.textures[index]
        return None

    def get_buffer(self, index: GLTFBufferIndex) -> Optional[GLTFBuffer]:
        """Get a buffer by index"""
        if 0 <= index < len(self.buffers):
            return self.buffers[index]
        return None

    def get_accessor(self, index: GLTFAccessorIndex) -> Optional[GLTFAccessor]:
        """Get an accessor by index"""
        if 0 <= index < len(self.accessors):
            return self.accessors[index]
        return None

    def get_animation(self, index: GLTFAnimationIndex) -> Optional[GLTFAnimation]:
        """Get an animation by index"""
        if 0 <= index < len(self.animations):
            return self.animations[index]
        return None

    def generate_unique_name(self, base_name: str) -> str:
        """Generate a unique name"""
        if base_name not in self.unique_names:
            self.unique_names[base_name] = 0
            return base_name

        self.unique_names[base_name] += 1
        return f"{base_name}_{self.unique_names[base_name]}"

    def generate_unique_animation_name(self, base_name: str) -> str:
        """Generate a unique animation name"""
        if base_name not in self.unique_animation_names:
            self.unique_animation_names[base_name] = 0
            return base_name

        self.unique_animation_names[base_name] += 1
        return f"{base_name}_{self.unique_animation_names[base_name]}"

    def generate_unique_bone_name(self, base_name: str) -> str:
        """Generate a unique bone name"""
        if base_name not in self.unique_bone_names:
            self.unique_bone_names[base_name] = 0
            return base_name

        self.unique_bone_names[base_name] += 1
        return f"{base_name}_{self.unique_bone_names[base_name]}"

    def clear(self):
        """Clear all data"""
        self.nodes.clear()
        self.scenes.clear()
        self.meshes.clear()
        self.materials.clear()
        self.textures.clear()
        self.images.clear()
        self.texture_samplers.clear()
        self.buffers.clear()
        self.buffer_views.clear()
        self.accessors.clear()
        self.skins.clear()
        self.cameras.clear()
        self.lights.clear()
        self.animations.clear()
        self.unique_names.clear()
        self.unique_animation_names.clear()
        self.unique_bone_names.clear()
        self.root_nodes.clear()
        self.scene = None

    def is_empty(self) -> bool:
        """Check if the state is empty"""
        return (len(self.nodes) == 0 and
                len(self.meshes) == 0 and
                len(self.materials) == 0 and
                len(self.textures) == 0 and
                len(self.buffers) == 0 and
                len(self.accessors) == 0)

    def __str__(self) -> str:
        """String representation of the GLTF state"""
        return (f"GLTFState(nodes={len(self.nodes)}, meshes={len(self.meshes)}, "
                f"materials={len(self.materials)}, textures={len(self.textures)}, "
                f"buffers={len(self.buffers)}, accessors={len(self.accessors)}, "
                f"animations={len(self.animations)})")
