"""
GLTF Data Structures

This module contains the core data structures used to represent GLTF data,
equivalent to the structures in Godot's GLTF module.
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import json


# Type aliases for better code readability
Vector3 = List[float]
Vector4 = List[float]
Quaternion = List[float]
Matrix4x4 = List[List[float]]
Color = List[float]


@dataclass
class Transform:
    """Represents a 3D transformation"""
    origin: Vector3 = field(default_factory=lambda: [0.0, 0.0, 0.0])
    basis: Any = None  # Could be a matrix or quaternion representation

    def __post_init__(self):
        if self.basis is None:
            # Default to identity matrix
            self.basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@dataclass
class GLTFBuffer:
    """Represents a GLTF buffer"""
    uri: Optional[str] = None
    byte_length: int = 0
    data: bytes = field(default_factory=bytes)


@dataclass
class GLTFBufferView:
    """Represents a GLTF buffer view"""
    buffer: int = 0
    byte_offset: int = 0
    byte_length: int = 0
    byte_stride: Optional[int] = None
    target: Optional[int] = None


@dataclass
class GLTFAccessor:
    """Represents a GLTF accessor"""
    buffer_view: Optional[int] = None
    byte_offset: int = 0
    component_type: int = 0
    count: int = 0
    type: str = ""
    max: Optional[List[float]] = None
    min: Optional[List[float]] = None
    normalized: bool = False
    sparse: Optional[Dict] = None


@dataclass
class GLTFTexture:
    """Represents a GLTF texture"""
    sampler: Optional[int] = None
    source: Optional[int] = None
    name: Optional[str] = None


@dataclass
class GLTFImage:
    """Represents a GLTF image"""
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    buffer_view: Optional[int] = None
    name: Optional[str] = None


@dataclass
class GLTFTextureSampler:
    """Represents a GLTF texture sampler"""
    mag_filter: Optional[int] = None
    min_filter: Optional[int] = None
    wrap_s: int = 10497  # REPEAT
    wrap_t: int = 10497  # REPEAT
    name: Optional[str] = None


@dataclass
class GLTFMaterial:
    """Represents a GLTF material"""
    name: Optional[str] = None
    pbr_metallic_roughness: Dict = field(default_factory=dict)
    normal_texture: Optional[Dict] = None
    occlusion_texture: Optional[Dict] = None
    emissive_texture: Optional[Dict] = None
    emissive_factor: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False


@dataclass
class GLTFPrimitive:
    """Represents a GLTF mesh primitive"""
    attributes: Dict[str, int] = field(default_factory=dict)
    indices: Optional[int] = None
    material: Optional[int] = None
    mode: int = 4  # TRIANGLES
    targets: List[Dict[str, int]] = field(default_factory=list)


@dataclass
class GLTFMesh:
    """Represents a GLTF mesh"""
    name: Optional[str] = None
    primitives: List[GLTFPrimitive] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)


@dataclass
class GLTFSkin:
    """Represents a GLTF skin"""
    name: Optional[str] = None
    inverse_bind_matrices: Optional[int] = None
    joints: List[int] = field(default_factory=list)
    joints_original: List[int] = field(default_factory=list)
    non_joints: List[int] = field(default_factory=list)
    roots: List[int] = field(default_factory=list)
    skeleton: Optional[int] = None
    godot_skin: Optional[Any] = None


@dataclass
class GLTFSkeleton:
    """Represents a GLTF skeleton"""
    name: Optional[str] = None
    joints: List[int] = field(default_factory=list)
    roots: List[int] = field(default_factory=list)
    godot_skeleton: Optional[Any] = None


@dataclass
class GLTFCamera:
    """Represents a GLTF camera"""
    name: Optional[str] = None
    type: str = "perspective"
    perspective: Optional[Dict] = None
    orthographic: Optional[Dict] = None


@dataclass
class GLTFLight:
    """Represents a GLTF light (KHR_lights_punctual extension)"""
    name: Optional[str] = None
    type: str = "directional"
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity: float = 1.0
    range: Optional[float] = None
    spot: Optional[Dict] = None


@dataclass
class GLTFAnimationSampler:
    """Represents a GLTF animation sampler"""
    input: int = 0
    interpolation: str = "LINEAR"
    output: int = 0


@dataclass
class GLTFAnimationChannel:
    """Represents a GLTF animation channel"""
    sampler: int = 0
    target: Dict = field(default_factory=dict)


@dataclass
class GLTFAnimation:
    """Represents a GLTF animation"""
    name: Optional[str] = None
    samplers: List[GLTFAnimationSampler] = field(default_factory=list)
    channels: List[GLTFAnimationChannel] = field(default_factory=list)


@dataclass
class GLTFNode:
    """Represents a GLTF node"""
    name: Optional[str] = None
    children: List[int] = field(default_factory=list)
    translation: Optional[Vector3] = None
    rotation: Optional[Quaternion] = None
    scale: Optional[Vector3] = None
    matrix: Optional[Matrix4x4] = None
    mesh: Optional[int] = None
    skin: Optional[int] = None
    camera: Optional[int] = None
    light: Optional[int] = None
    weights: List[float] = field(default_factory=list)
    extensions: Dict = field(default_factory=dict)
    extras: Optional[Dict] = None
    # Additional fields for skinning support
    parent: int = -1
    joint: bool = False
    skeleton: int = -1
    height: int = 0
    transform: Optional[Any] = None  # Transform object


@dataclass
class GLTFScene:
    """Represents a GLTF scene"""
    name: Optional[str] = None
    nodes: List[int] = field(default_factory=list)


@dataclass
class GLTFAsset:
    """Represents GLTF asset information"""
    version: str = "2.0"
    generator: Optional[str] = None
    copyright: Optional[str] = None
    min_version: Optional[str] = None


@dataclass
class GLTFExtensions:
    """Represents GLTF extensions"""
    used: Dict = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


# Type indices (equivalent to Godot's typedefs)
GLTFNodeIndex = int
GLTFMeshIndex = int
GLTFSkinIndex = int
GLTFTextureIndex = int
GLTFMaterialIndex = int
GLTFBufferIndex = int
GLTFBufferViewIndex = int
GLTFAccessorIndex = int
GLTFLightIndex = int
GLTFCameraIndex = int
GLTFImageIndex = int
GLTFSkeletonIndex = int
GLTFAnimationIndex = int
GLTFSceneIndex = int


def create_default_gltf_node(name: Optional[str] = None) -> GLTFNode:
    """Create a default GLTF node"""
    return GLTFNode(
        name=name,
        translation=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0, 1.0],
        scale=[1.0, 1.0, 1.0]
    )


def create_default_gltf_material(name: Optional[str] = None) -> GLTFMaterial:
    """Create a default GLTF material"""
    return GLTFMaterial(
        name=name,
        pbr_metallic_roughness={
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0
        }
    )
