"""
PyTorch-compatible GLTF Data Structures

This module contains PyTorch tensor-based data structures for GLTF data,
providing GPU acceleration and seamless PyTorch integration.
"""

import torch
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from .structures import *


@dataclass
class TorchGLTFBuffer:
    """PyTorch-compatible GLTF buffer with tensor data"""
    uri: Optional[str] = None
    byte_length: int = 0
    data: Optional[torch.Tensor] = None  # Byte tensor for buffer data
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    def to(self, device: torch.device) -> 'TorchGLTFBuffer':
        """Move buffer to specified device"""
        if self.data is not None:
            self.data = self.data.to(device)
        self.device = device
        return self

    @classmethod
    def from_gltf_buffer(cls, buffer: GLTFBuffer, device: torch.device = None) -> 'TorchGLTFBuffer':
        """Convert from standard GLTF buffer to PyTorch buffer"""
        if device is None:
            device = torch.device('cpu')

        torch_buffer = cls(
            uri=buffer.uri,
            byte_length=buffer.byte_length,
            device=device
        )

        if buffer.data:
            torch_buffer.data = torch.frombuffer(buffer.data, dtype=torch.uint8).to(device)

        return torch_buffer


@dataclass
class TorchGLTFAccessor:
    """PyTorch-compatible GLTF accessor with tensor data"""
    buffer_view: Optional[int] = None
    byte_offset: int = 0
    component_type: int = 0
    count: int = 0
    type: str = ""
    max: Optional[torch.Tensor] = None
    min: Optional[torch.Tensor] = None
    normalized: bool = False
    sparse: Optional[Dict] = None

    # Cached decoded tensor data
    tensor_data: Optional[torch.Tensor] = None
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    def to(self, device: torch.device) -> 'TorchGLTFAccessor':
        """Move accessor data to specified device"""
        if self.tensor_data is not None:
            self.tensor_data = self.tensor_data.to(device)
        if self.max is not None:
            self.max = self.max.to(device)
        if self.min is not None:
            self.min = self.min.to(device)
        self.device = device
        return self

    @classmethod
    def from_gltf_accessor(cls, accessor: GLTFAccessor, device: torch.device = None) -> 'TorchGLTFAccessor':
        """Convert from standard GLTF accessor to PyTorch accessor"""
        if device is None:
            device = torch.device('cpu')

        torch_accessor = cls(
            buffer_view=accessor.buffer_view,
            byte_offset=accessor.byte_offset,
            component_type=accessor.component_type,
            count=accessor.count,
            type=accessor.type,
            normalized=accessor.normalized,
            sparse=accessor.sparse,
            device=device
        )

        # Convert min/max to tensors if present
        if accessor.max is not None:
            torch_accessor.max = torch.tensor(accessor.max, dtype=torch.float32, device=device)
        if accessor.min is not None:
            torch_accessor.min = torch.tensor(accessor.min, dtype=torch.float32, device=device)

        return torch_accessor


@dataclass
class TorchMeshData:
    """PyTorch-compatible mesh data with tensor attributes"""
    vertices: Optional[torch.Tensor] = None  # Shape: (N, 3)
    normals: Optional[torch.Tensor] = None   # Shape: (N, 3)
    texcoords: Optional[torch.Tensor] = None # Shape: (N, 2)
    colors: Optional[torch.Tensor] = None    # Shape: (N, 4) - RGBA
    indices: Optional[torch.Tensor] = None   # Shape: (M,) or (M, 3) for triangle indices
    material_name: Optional[str] = None
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    def to(self, device: torch.device) -> 'TorchMeshData':
        """Move mesh data to specified device"""
        if self.vertices is not None:
            self.vertices = self.vertices.to(device)
        if self.normals is not None:
            self.normals = self.normals.to(device)
        if self.texcoords is not None:
            self.texcoords = self.texcoords.to(device)
        if self.colors is not None:
            self.colors = self.colors.to(device)
        if self.indices is not None:
            self.indices = self.indices.to(device)
        self.device = device
        return self

    @property
    def num_vertices(self) -> int:
        """Get number of vertices"""
        return self.vertices.shape[0] if self.vertices is not None else 0

    @property
    def num_faces(self) -> int:
        """Get number of faces (triangles)"""
        if self.indices is None:
            return 0
        if self.indices.dim() == 1:
            return self.indices.shape[0] // 3
        return self.indices.shape[0]

    def compute_face_normals(self) -> torch.Tensor:
        """Compute face normals from vertices and indices"""
        if self.vertices is None or self.indices is None:
            return torch.empty(0, 3, dtype=torch.float32, device=self.device)

        if self.indices.dim() == 1:
            # Triangle indices as flat array
            faces = self.indices.view(-1, 3)
        else:
            faces = self.indices

        # Get triangle vertices
        v0 = self.vertices[faces[:, 0]]
        v1 = self.vertices[faces[:, 1]]
        v2 = self.vertices[faces[:, 2]]

        # Compute face normals
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = torch.cross(edge1, edge2, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)

        return face_normals

    def compute_vertex_normals(self) -> torch.Tensor:
        """Compute vertex normals by averaging face normals"""
        if self.vertices is None or self.indices is None:
            return torch.empty(0, 3, dtype=torch.float32, device=self.device)

        face_normals = self.compute_face_normals()

        if self.indices.dim() == 1:
            faces = self.indices.view(-1, 3)
        else:
            faces = self.indices

        # Initialize vertex normals
        vertex_normals = torch.zeros_like(self.vertices)

        # Accumulate face normals to vertices
        for i, face in enumerate(faces):
            for vertex_idx in face:
                vertex_normals[vertex_idx] += face_normals[i]

        # Normalize
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
        return vertex_normals

    @classmethod
    def from_mesh_data(cls, mesh_data: 'MeshData', device: torch.device = None) -> 'TorchMeshData':
        """Convert from standard MeshData to PyTorch MeshData"""
        if device is None:
            device = torch.device('cpu')

        torch_mesh = cls(device=device, material_name=mesh_data.material_name)

        # Convert lists to tensors
        if mesh_data.vertices:
            torch_mesh.vertices = torch.tensor(mesh_data.vertices, dtype=torch.float32, device=device)
        if mesh_data.normals:
            torch_mesh.normals = torch.tensor(mesh_data.normals, dtype=torch.float32, device=device)
        if mesh_data.texcoords:
            torch_mesh.texcoords = torch.tensor(mesh_data.texcoords, dtype=torch.float32, device=device)
        if mesh_data.colors:
            torch_mesh.colors = torch.tensor(mesh_data.colors, dtype=torch.float32, device=device)
        if mesh_data.indices:
            torch_mesh.indices = torch.tensor(mesh_data.indices, dtype=torch.int32, device=device)

        return torch_mesh


@dataclass
class TorchSceneNode:
    """PyTorch-compatible scene node"""
    name: str = "Node"
    transform: 'TorchTransform' = field(default_factory=lambda: TorchTransform())
    children: List['TorchSceneNode'] = field(default_factory=list)
    parent: Optional['TorchSceneNode'] = None

    # Node data
    mesh_data: Optional[TorchMeshData] = None
    light_data: Optional[Dict[str, Any]] = None  # Keep as dict for now
    camera_data: Optional[Dict[str, Any]] = None  # Keep as dict for now

    # GLTF-specific data
    gltf_node_index: Optional[int] = None
    gltf_mesh_index: Optional[int] = None
    gltf_material_index: Optional[int] = None

    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    def to(self, device: torch.device) -> 'TorchSceneNode':
        """Move node and all children to specified device"""
        self.transform.to(device)
        if self.mesh_data is not None:
            self.mesh_data.to(device)
        self.device = device

        for child in self.children:
            child.to(device)

        return self

    def add_child(self, child: 'TorchSceneNode'):
        """Add a child node"""
        child.parent = self
        child.to(self.device)  # Ensure child is on same device
        self.children.append(child)

    def get_world_transform(self) -> 'TorchTransform':
        """Get the world transform of this node"""
        if self.parent is None:
            return self.transform
        else:
            return self.parent.get_world_transform() * self.transform

    def traverse(self) -> List['TorchSceneNode']:
        """Get all nodes in subtree including this one"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.traverse())
        return nodes

    def get_all_meshes(self) -> List[TorchMeshData]:
        """Get all mesh data in subtree"""
        meshes = []
        if self.mesh_data is not None:
            meshes.append(self.mesh_data)

        for child in self.children:
            meshes.extend(child.get_all_meshes())

        return meshes

    @classmethod
    def from_scene_node(cls, scene_node: 'SceneNode', device: torch.device = None) -> 'TorchSceneNode':
        """Convert from standard SceneNode to PyTorch SceneNode"""
        if device is None:
            device = torch.device('cpu')

        torch_node = cls(
            name=scene_node.name,
            device=device,
            gltf_node_index=scene_node.gltf_node_index,
            gltf_mesh_index=scene_node.gltf_mesh_index,
            gltf_material_index=scene_node.gltf_material_index
        )

        # Convert transform
        torch_node.transform = TorchTransform.from_transform(scene_node.transform, device)

        # Convert mesh data
        if scene_node.mesh_data is not None:
            torch_node.mesh_data = TorchMeshData.from_mesh_data(scene_node.mesh_data, device)

        # Copy other data as-is for now
        torch_node.light_data = scene_node.light_data
        torch_node.camera_data = scene_node.camera_data

        return torch_node


@dataclass
class TorchTransform:
    """PyTorch-compatible 3D transformation"""
    translation: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32))
    rotation: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32))
    scale: torch.Tensor = field(default_factory=lambda: torch.ones(3, dtype=torch.float32))
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    def to(self, device: torch.device) -> 'TorchTransform':
        """Move transform to specified device"""
        self.translation = self.translation.to(device)
        self.rotation = self.rotation.to(device)
        self.scale = self.scale.to(device)
        self.device = device
        return self

    def __mul__(self, other: 'TorchTransform') -> 'TorchTransform':
        """Combine two transforms"""
        result = TorchTransform(device=self.device)

        # Combine translations
        result.translation = self.translation + other.translation

        # Combine rotations (simplified - would need proper quaternion multiplication)
        result.rotation = self.rotation  # Placeholder

        # Combine scales
        result.scale = self.scale * other.scale

        return result

    def get_matrix(self) -> torch.Tensor:
        """Get 4x4 transformation matrix"""
        # This is a simplified implementation
        matrix = torch.eye(4, dtype=torch.float32, device=self.device)

        # Apply scale
        matrix[0, 0] = self.scale[0]
        matrix[1, 1] = self.scale[1]
        matrix[2, 2] = self.scale[2]

        # Apply translation
        matrix[0, 3] = self.translation[0]
        matrix[1, 3] = self.translation[1]
        matrix[2, 3] = self.translation[2]

        # TODO: Apply rotation properly
        # For now, return simplified matrix

        return matrix

    @classmethod
    def from_transform(cls, transform: 'Transform', device: torch.device = None) -> 'TorchTransform':
        """Convert from standard Transform to PyTorch Transform"""
        if device is None:
            device = torch.device('cpu')

        torch_transform = cls(device=device)
        torch_transform.translation = torch.tensor(transform.translation, dtype=torch.float32, device=device)
        torch_transform.rotation = torch.tensor(transform.rotation, dtype=torch.float32, device=device)
        torch_transform.scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)

        return torch_transform


# Utility functions for PyTorch operations
def batch_mesh_data(meshes: List[TorchMeshData]) -> TorchMeshData:
    """Batch multiple meshes into a single mesh (for batched processing)"""
    if not meshes:
        return TorchMeshData()

    device = meshes[0].device

    # Concatenate all vertex data
    all_vertices = []
    all_normals = []
    all_texcoords = []
    all_colors = []
    all_indices = []

    vertex_offset = 0
    for mesh in meshes:
        if mesh.vertices is not None:
            all_vertices.append(mesh.vertices)
        if mesh.normals is not None:
            all_normals.append(mesh.normals)
        if mesh.texcoords is not None:
            all_texcoords.append(mesh.texcoords)
        if mesh.colors is not None:
            all_colors.append(mesh.colors)
        if mesh.indices is not None:
            # Offset indices by current vertex count
            offset_indices = mesh.indices + vertex_offset
            all_indices.append(offset_indices)
            vertex_offset += mesh.num_vertices

    batched_mesh = TorchMeshData(device=device)

    if all_vertices:
        batched_mesh.vertices = torch.cat(all_vertices, dim=0)
    if all_normals:
        batched_mesh.normals = torch.cat(all_normals, dim=0)
    if all_texcoords:
        batched_mesh.texcoords = torch.cat(all_texcoords, dim=0)
    if all_colors:
        batched_mesh.colors = torch.cat(all_colors, dim=0)
    if all_indices:
        batched_mesh.indices = torch.cat(all_indices, dim=0)

    return batched_mesh


def move_scene_to_device(scene_nodes: List[TorchSceneNode], device: torch.device) -> List[TorchSceneNode]:
    """Move entire scene to specified device"""
    for node in scene_nodes:
        node.to(device)
    return scene_nodes


def get_scene_device(scene_nodes: List[TorchSceneNode]) -> torch.device:
    """Get the device of the scene (assumes all nodes are on same device)"""
    if scene_nodes:
        return scene_nodes[0].device
    return torch.device('cpu')
