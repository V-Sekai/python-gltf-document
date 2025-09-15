"""
GLTF Scene Generator

This module provides functionality for generating scene structures from parsed GLTF data,
equivalent to Godot's scene generation from GLTF.
"""

from typing import List, Dict, Optional, Any, Tuple
from .gltf_state import GLTFState
from .structures import *
from .accessor_decoder import GLTFAccessorDecoder


class SceneNode:
    """
    Represents a node in the generated scene.

    This is a simplified scene node structure that can be used by applications
    to represent the GLTF scene hierarchy.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "Node"
        self.transform = Transform()
        self.children: List['SceneNode'] = []
        self.parent: Optional['SceneNode'] = None

        # Node data
        self.mesh_data: Optional[MeshData] = None
        self.light_data: Optional[LightData] = None
        self.camera_data: Optional[CameraData] = None

        # GLTF-specific data
        self.gltf_node_index: Optional[int] = None
        self.gltf_mesh_index: Optional[int] = None
        self.gltf_material_index: Optional[int] = None

    def add_child(self, child: 'SceneNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)

    def get_world_transform(self) -> 'Transform':
        """Get the world transform of this node"""
        if self.parent is None:
            return self.transform
        else:
            return self.parent.get_world_transform() * self.transform

    def __str__(self) -> str:
        return f"SceneNode(name='{self.name}', children={len(self.children)})"


class Transform:
    """Represents a 3D transformation"""

    def __init__(self):
        self.translation = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0, 1.0]  # Quaternion
        self.scale = [1.0, 1.0, 1.0]

    def __mul__(self, other: 'Transform') -> 'Transform':
        """Combine two transforms"""
        result = Transform()

        # Combine translations
        result.translation = [
            self.translation[0] + other.translation[0],
            self.translation[1] + other.translation[1],
            self.translation[2] + other.translation[2]
        ]

        # Combine rotations (simplified - would need proper quaternion multiplication)
        result.rotation = self.rotation  # Placeholder

        # Combine scales
        result.scale = [
            self.scale[0] * other.scale[0],
            self.scale[1] * other.scale[1],
            self.scale[2] * other.scale[2]
        ]

        return result

    def __str__(self) -> str:
        return f"Transform(t={self.translation}, r={self.rotation}, s={self.scale})"


class MeshData:
    """Represents mesh data for a scene node"""

    def __init__(self):
        self.vertices: List[List[float]] = []
        self.normals: List[List[float]] = []
        self.texcoords: List[List[float]] = []
        self.colors: List[List[float]] = []
        self.indices: List[int] = []
        self.material_name: Optional[str] = None

    def __str__(self) -> str:
        return f"MeshData(vertices={len(self.vertices)}, indices={len(self.indices)})"


class LightData:
    """Represents light data for a scene node"""

    def __init__(self):
        self.type = "directional"
        self.color = [1.0, 1.0, 1.0]
        self.intensity = 1.0
        self.range: Optional[float] = None

    def __str__(self) -> str:
        return f"LightData(type='{self.type}', color={self.color})"


class CameraData:
    """Represents camera data for a scene node"""

    def __init__(self):
        self.type = "perspective"
        self.fov: Optional[float] = None
        self.aspect_ratio: Optional[float] = None
        self.near: Optional[float] = None
        self.far: Optional[float] = None

    def __str__(self) -> str:
        return f"CameraData(type='{self.type}')"


class GLTFSceneGenerator:
    """
    Generator for scene structures from GLTF data.

    This class takes parsed GLTF data and generates a scene hierarchy
    that can be used by applications.
    """

    def __init__(self, state: GLTFState):
        self.state = state
        self.scene_nodes: List[SceneNode] = []
        self.node_map: Dict[int, SceneNode] = {}  # GLTF node index -> SceneNode

    def generate_scene(self, scene_index: Optional[int] = None) -> List[SceneNode]:
        """
        Generate scene nodes from GLTF data.

        Args:
            scene_index: Index of the scene to generate (uses default if None)

        Returns:
            List of root scene nodes
        """
        # Clear previous generation
        self.scene_nodes.clear()
        self.node_map.clear()

        # Determine which scene to use
        if scene_index is None:
            scene_index = self.state.scene

        if scene_index is None or scene_index >= len(self.state.scenes):
            # Generate all nodes if no scene specified
            for i in range(len(self.state.nodes)):
                if not self._is_node_in_hierarchy(i):
                    self._generate_node(i)
        else:
            scene = self.state.scenes[scene_index]
            for node_index in scene.nodes:
                self._generate_node(node_index)

        return self.scene_nodes

    def _is_node_in_hierarchy(self, node_index: int) -> bool:
        """Check if a node is already part of the hierarchy"""
        for scene in self.state.scenes:
            if node_index in scene.nodes:
                return True
        return False

    def _generate_node(self, node_index: int) -> SceneNode:
        """
        Generate a scene node from a GLTF node.

        Args:
            node_index: Index of the GLTF node

        Returns:
            Generated SceneNode
        """
        if node_index in self.node_map:
            return self.node_map[node_index]

        if node_index < 0 or node_index >= len(self.state.nodes):
            raise ValueError(f"Invalid node index: {node_index}")

        gltf_node = self.state.nodes[node_index]

        # Create scene node
        scene_node = SceneNode(gltf_node.name)
        scene_node.gltf_node_index = node_index

        # Set transform
        self._set_node_transform(scene_node, gltf_node)

        # Add mesh data if present
        if gltf_node.mesh is not None:
            scene_node.mesh_data = self._generate_mesh_data(gltf_node.mesh)
            scene_node.gltf_mesh_index = gltf_node.mesh

        # Add light data if present
        if gltf_node.light is not None:
            scene_node.light_data = self._generate_light_data(gltf_node.light)

        # Add camera data if present
        if gltf_node.camera is not None:
            scene_node.camera_data = self._generate_camera_data(gltf_node.camera)

        # Store in map
        self.node_map[node_index] = scene_node

        # Generate children
        for child_index in gltf_node.children:
            child_node = self._generate_node(child_index)
            scene_node.add_child(child_node)

        # Add to root nodes if it has no parent
        if not gltf_node.children or node_index not in [c for n in self.state.nodes for c in n.children]:
            self.scene_nodes.append(scene_node)

        return scene_node

    def _set_node_transform(self, scene_node: SceneNode, gltf_node: GLTFNode):
        """Set the transform of a scene node from GLTF node data"""
        if gltf_node.translation:
            scene_node.transform.translation = gltf_node.translation

        if gltf_node.rotation:
            scene_node.transform.rotation = gltf_node.rotation

        if gltf_node.scale:
            scene_node.transform.scale = gltf_node.scale

        # Note: Matrix transforms would need more complex handling

    def _generate_mesh_data(self, mesh_index: int) -> Optional[MeshData]:
        """Generate mesh data from GLTF mesh"""
        if mesh_index < 0 or mesh_index >= len(self.state.meshes):
            return None

        gltf_mesh = self.state.meshes[mesh_index]
        mesh_data = MeshData()

        # Process all primitives (we'll use the first one for simplicity)
        if gltf_mesh.primitives:
            primitive = gltf_mesh.primitives[0]

            # Get vertex positions
            if 'POSITION' in primitive.attributes:
                pos_accessor = primitive.attributes['POSITION']
                try:
                    mesh_data.vertices = GLTFAccessorDecoder.decode_accessor_as_vec3(
                        self.state, pos_accessor, for_vertex=True)
                except Exception as e:
                    print(f"Error decoding positions: {e}")

            # Get normals
            if 'NORMAL' in primitive.attributes:
                normal_accessor = primitive.attributes['NORMAL']
                try:
                    mesh_data.normals = GLTFAccessorDecoder.decode_accessor_as_vec3(
                        self.state, normal_accessor, for_vertex=True)
                except Exception as e:
                    print(f"Error decoding normals: {e}")

            # Get texture coordinates
            if 'TEXCOORD_0' in primitive.attributes:
                texcoord_accessor = primitive.attributes['TEXCOORD_0']
                try:
                    mesh_data.texcoords = GLTFAccessorDecoder.decode_accessor_as_vec2(
                        self.state, texcoord_accessor, for_vertex=True)
                except Exception as e:
                    print(f"Error decoding texcoords: {e}")

            # Get colors
            if 'COLOR_0' in primitive.attributes:
                color_accessor = primitive.attributes['COLOR_0']
                try:
                    mesh_data.colors = GLTFAccessorDecoder.decode_accessor_as_colors(
                        self.state, color_accessor, for_vertex=True)
                except Exception as e:
                    print(f"Error decoding colors: {e}")

            # Get indices
            if primitive.indices is not None:
                try:
                    mesh_data.indices = GLTFAccessorDecoder.decode_accessor_as_indices(
                        self.state, primitive.indices)
                except Exception as e:
                    print(f"Error decoding indices: {e}")

            # Set material
            if primitive.material is not None and primitive.material < len(self.state.materials):
                material = self.state.materials[primitive.material]
                mesh_data.material_name = material.name

        return mesh_data

    def _generate_light_data(self, light_index: int) -> Optional[LightData]:
        """Generate light data from GLTF light"""
        if light_index < 0 or light_index >= len(self.state.lights):
            return None

        gltf_light = self.state.lights[light_index]
        light_data = LightData()

        light_data.type = gltf_light.type
        light_data.color = gltf_light.color
        light_data.intensity = gltf_light.intensity
        light_data.range = gltf_light.range

        return light_data

    def _generate_camera_data(self, camera_index: int) -> Optional[CameraData]:
        """Generate camera data from GLTF camera"""
        if camera_index < 0 or camera_index >= len(self.state.cameras):
            return None

        gltf_camera = self.state.cameras[camera_index]
        camera_data = CameraData()

        camera_data.type = gltf_camera.type

        if gltf_camera.perspective:
            camera_data.fov = gltf_camera.perspective.get('yfov')
            camera_data.aspect_ratio = gltf_camera.perspective.get('aspectRatio')
            camera_data.near = gltf_camera.perspective.get('znear')
            camera_data.far = gltf_camera.perspective.get('zfar')

        return camera_data

    def get_scene_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated scene"""
        total_nodes = len(self.node_map)
        total_meshes = sum(1 for node in self.node_map.values() if node.mesh_data)
        total_lights = sum(1 for node in self.node_map.values() if node.light_data)
        total_cameras = sum(1 for node in self.node_map.values() if node.camera_data)

        total_vertices = sum(len(node.mesh_data.vertices) for node in self.node_map.values()
                           if node.mesh_data)
        total_indices = sum(len(node.mesh_data.indices) for node in self.node_map.values()
                          if node.mesh_data)

        return {
            'total_nodes': total_nodes,
            'total_meshes': total_meshes,
            'total_lights': total_lights,
            'total_cameras': total_cameras,
            'total_vertices': total_vertices,
            'total_indices': total_indices,
            'root_nodes': len(self.scene_nodes)
        }

    def print_scene_hierarchy(self, node: Optional[SceneNode] = None, indent: int = 0):
        """Print the scene hierarchy"""
        if node is None:
            if not self.scene_nodes:
                print("No scene generated")
                return
            for root_node in self.scene_nodes:
                self.print_scene_hierarchy(root_node, 0)
            return

        prefix = "  " * indent
        print(f"{prefix}{node}")

        for child in node.children:
            self.print_scene_hierarchy(child, indent + 1)
