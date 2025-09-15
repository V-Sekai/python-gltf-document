#!/usr/bin/env python3
"""
GPU Operators for GLTF Processing

This module contains ExecuTorch operators for GPU-accelerated GLTF processing
including geometry transformations, skinning calculations, and mesh operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

# ExecuTorch imports
try:
    from executorch import exir
    EXECUTORCH_AVAILABLE = True
except ImportError:
    EXECUTORCH_AVAILABLE = False


class GPUTransformOperator(nn.Module):
    """GPU operator for applying transformations to vertex data"""

    def __init__(self):
        super(GPUTransformOperator, self).__init__()

    def forward(self, vertices: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply 4x4 transformation matrix to vertices

        Args:
            vertices: Vertex positions (N, 3)
            transform_matrix: 4x4 transformation matrix

        Returns:
            Transformed vertices (N, 3)
        """
        # Convert vertices to homogeneous coordinates
        ones = torch.ones(vertices.shape[0], 1, dtype=vertices.dtype, device=vertices.device)
        homogeneous_vertices = torch.cat([vertices, ones], dim=1)  # (N, 4)

        # Apply transformation
        transformed = torch.matmul(homogeneous_vertices, transform_matrix.t())  # (N, 4)

        # Convert back to 3D coordinates
        return transformed[:, :3] / transformed[:, 3:4]  # Perspective divide


class GPUSkinningOperator(nn.Module):
    """GPU operator for skeletal animation skinning"""

    def __init__(self):
        super(GPUSkinningOperator, self).__init__()

    def forward(self, vertices: torch.Tensor, normals: torch.Tensor,
                bone_weights: torch.Tensor, bone_indices: torch.Tensor,
                bone_matrices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply skeletal animation skinning

        Args:
            vertices: Vertex positions (N, 3)
            normals: Vertex normals (N, 3)
            bone_weights: Bone weights (N, 4)
            bone_indices: Bone indices (N, 4)
            bone_matrices: Bone transformation matrices (M, 4, 4)

        Returns:
            Tuple of (transformed_vertices, transformed_normals)
        """
        batch_size, num_bones = bone_weights.shape

        # Initialize transformed vertices and normals
        transformed_vertices = torch.zeros_like(vertices)
        transformed_normals = torch.zeros_like(normals)

        # Apply skinning for each bone
        for i in range(num_bones):
            # Get bone weight and index for this bone
            weight = bone_weights[:, i:i+1]  # (N, 1)
            bone_idx = bone_indices[:, i].long()  # (N,)

            # Get bone transformation matrix
            bone_matrix = bone_matrices[bone_idx]  # (N, 4, 4)

            # Apply transformation to vertices
            vertex_transform = self._transform_points(vertices, bone_matrix)
            transformed_vertices += weight * vertex_transform

            # Apply transformation to normals (without translation)
            normal_transform = self._transform_vectors(normals, bone_matrix)
            transformed_normals += weight * normal_transform

        return transformed_vertices, transformed_normals

    def _transform_points(self, points: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
        """Transform points using 4x4 matrices"""
        # Convert to homogeneous coordinates
        ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
        homogeneous = torch.cat([points, ones], dim=1)  # (N, 4)

        # Apply transformation
        transformed = torch.matmul(homogeneous.unsqueeze(1), matrices.transpose(-2, -1))
        transformed = transformed.squeeze(1)

        # Convert back to 3D
        return transformed[:, :3] / transformed[:, 3:4]

    def _transform_vectors(self, vectors: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
        """Transform vectors using 4x4 matrices (no translation)"""
        # Apply rotation/scale part only (top-left 3x3)
        rotation_matrices = matrices[:, :3, :3]  # (N, 3, 3)
        transformed = torch.matmul(vectors.unsqueeze(1), rotation_matrices.transpose(-2, -1))
        return transformed.squeeze(1)


class GPUNormalCalculationOperator(nn.Module):
    """GPU operator for calculating vertex normals from triangle data"""

    def __init__(self):
        super(GPUNormalCalculationOperator, self).__init__()

    def forward(self, vertices: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate vertex normals from triangle indices

        Args:
            vertices: Vertex positions (N, 3)
            indices: Triangle indices (M, 3)

        Returns:
            Vertex normals (N, 3)
        """
        num_vertices = vertices.shape[0]
        normals = torch.zeros(num_vertices, 3, dtype=vertices.dtype, device=vertices.device)

        # Calculate face normals and accumulate
        face_normals = self._calculate_face_normals(vertices, indices)

        # Accumulate face normals to vertices
        for i in range(indices.shape[0]):
            for j in range(3):
                vertex_idx = indices[i, j]
                normals[vertex_idx] += face_normals[i]

        # Normalize
        normal_lengths = torch.norm(normals, dim=1, keepdim=True)
        normal_lengths = torch.clamp(normal_lengths, min=1e-8)  # Avoid division by zero
        normals = normals / normal_lengths

        return normals

    def _calculate_face_normals(self, vertices: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Calculate normals for each triangle face"""
        # Get triangle vertices
        v0 = vertices[indices[:, 0]]  # (M, 3)
        v1 = vertices[indices[:, 1]]  # (M, 3)
        v2 = vertices[indices[:, 2]]  # (M, 3)

        # Calculate edges
        edge1 = v1 - v0  # (M, 3)
        edge2 = v2 - v0  # (M, 3)

        # Calculate face normals using cross product
        face_normals = torch.cross(edge1, edge2, dim=1)  # (M, 3)

        # Normalize face normals
        lengths = torch.norm(face_normals, dim=1, keepdim=True)
        lengths = torch.clamp(lengths, min=1e-8)
        face_normals = face_normals / lengths

        return face_normals


class GPUUVAtlasOperator(nn.Module):
    """GPU operator for UV atlas generation and optimization"""

    def __init__(self):
        super(GPUUVAtlasOperator, self).__init__()

    def forward(self, vertices: torch.Tensor, triangles: torch.Tensor,
                target_resolution: int = 1024) -> torch.Tensor:
        """
        Generate UV coordinates for mesh atlas

        Args:
            vertices: Vertex positions (N, 3)
            triangles: Triangle indices (M, 3)
            target_resolution: Target atlas resolution

        Returns:
            UV coordinates (N, 2)
        """
        # Simple box projection for now
        # In a full implementation, this would use advanced UV unwrapping algorithms

        # Normalize vertices to [0, 1] range
        min_coords = torch.min(vertices, dim=0)[0]
        max_coords = torch.max(vertices, dim=0)[0]
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords + 1e-8)

        # Use different projections for different faces
        uvs = torch.zeros(vertices.shape[0], 2, dtype=vertices.dtype, device=vertices.device)

        # Simple planar projection on YZ plane
        uvs[:, 0] = normalized_vertices[:, 2]  # Z -> U
        uvs[:, 1] = normalized_vertices[:, 1]  # Y -> V

        return uvs


class GPUMeshOptimizationOperator(nn.Module):
    """GPU operator for mesh optimization operations"""

    def __init__(self):
        super(GPUMeshOptimizationOperator, self).__init__()

    def forward(self, vertices: torch.Tensor, indices: torch.Tensor,
                optimization_type: str = "simplify") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mesh optimization

        Args:
            vertices: Vertex positions (N, 3)
            indices: Triangle indices (M, 3)
            optimization_type: Type of optimization ("simplify", "smooth", etc.)

        Returns:
            Tuple of (optimized_vertices, optimized_indices)
        """
        if optimization_type == "simplify":
            return self._simplify_mesh(vertices, indices)
        elif optimization_type == "smooth":
            return self._smooth_mesh(vertices, indices)
        else:
            return vertices, indices

    def _simplify_mesh(self, vertices: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplify mesh by removing vertices (placeholder implementation)"""
        # For now, return unchanged
        # In a full implementation, this would use quadric error metrics or similar
        return vertices, indices

    def _smooth_mesh(self, vertices: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Laplacian smoothing to mesh"""
        # Calculate vertex neighbors
        num_vertices = vertices.shape[0]
        neighbor_sum = torch.zeros_like(vertices)
        neighbor_count = torch.zeros(num_vertices, dtype=torch.int32, device=vertices.device)

        # Accumulate neighbor positions
        for i in range(indices.shape[0]):
            for j in range(3):
                v1 = indices[i, j]
                v2 = indices[i, (j + 1) % 3]
                neighbor_sum[v1] += vertices[v2]
                neighbor_count[v1] += 1

        # Calculate smoothed positions
        smoothed_vertices = torch.where(
            neighbor_count.unsqueeze(1) > 0,
            neighbor_sum / neighbor_count.unsqueeze(1).float(),
            vertices
        )

        # Blend with original (conservative smoothing)
        smoothed_vertices = 0.5 * vertices + 0.5 * smoothed_vertices

        return smoothed_vertices, indices


class GPUBatchProcessor(nn.Module):
    """GPU operator for batch processing multiple meshes"""

    def __init__(self):
        super(GPUBatchProcessor, self).__init__()
        self.transform_op = GPUTransformOperator()
        self.skinning_op = GPUSkinningOperator()
        self.normal_op = GPUNormalCalculationOperator()

    def forward(self, batch_vertices: List[torch.Tensor], batch_indices: List[torch.Tensor],
                batch_transforms: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process batch of meshes on GPU

        Args:
            batch_vertices: List of vertex tensors
            batch_indices: List of index tensors
            batch_transforms: List of transformation matrices

        Returns:
            List of processed vertex tensors
        """
        processed_vertices = []

        for i, (vertices, indices, transform) in enumerate(zip(batch_vertices, batch_indices, batch_transforms)):
            # Apply transformation
            transformed_vertices = self.transform_op(vertices, transform)

            # Recalculate normals if needed
            if vertices.shape[1] >= 6:  # Has normals
                normals = vertices[:, 3:6]
                # Transform normals
                transformed_normals = self._transform_normals(normals, transform)
                transformed_vertices = torch.cat([transformed_vertices, transformed_normals], dim=1)

            processed_vertices.append(transformed_vertices)

        return processed_vertices

    def _transform_normals(self, normals: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
        """Transform normals using rotation part of matrix"""
        rotation_matrix = transform_matrix[:3, :3]
        transformed_normals = torch.matmul(normals, rotation_matrix.t())

        # Renormalize
        lengths = torch.norm(transformed_normals, dim=1, keepdim=True)
        lengths = torch.clamp(lengths, min=1e-8)
        return transformed_normals / lengths


def create_executorch_program(model: nn.Module, sample_input: Tuple) -> Optional[object]:
    """Create ExecuTorch program from PyTorch model"""
    if not EXECUTORCH_AVAILABLE:
        print("ExecuTorch not available, skipping program creation")
        return None

    try:
        # Try new API first
        try:
            from executorch.exir import EdgeProgramManager, to_edge

            with torch.no_grad():
                edge_program = to_edge(model, sample_input)
                program_manager = EdgeProgramManager(
                    edge_program,
                    [],
                    compile_config=None
                )
                return program_manager.to_executorch()

        except Exception as api_error:
            print(f"New API failed: {api_error}, trying legacy API...")

            # Fallback to legacy API
            try:
                program = exir.capture_program(model, sample_input)
                return program.to_executorch()
            except Exception as legacy_error:
                print(f"Legacy API also failed: {legacy_error}")
                return None

    except Exception as e:
        print(f"Error creating ExecuTorch program: {e}")
        return None


# Pre-compiled GPU operators (would be created at build time)
def get_gpu_transform_program():
    """Get pre-compiled GPU transform operator"""
    model = GPUTransformOperator()
    sample_vertices = torch.randn(100, 3)
    sample_matrix = torch.eye(4)
    return create_executorch_program(model, (sample_vertices, sample_matrix))


def get_gpu_skinning_program():
    """Get pre-compiled GPU skinning operator"""
    model = GPUSkinningOperator()
    sample_vertices = torch.randn(100, 3)
    sample_normals = torch.randn(100, 3)
    sample_weights = torch.randn(100, 4)
    sample_indices = torch.randint(0, 10, (100, 4))
    sample_matrices = torch.eye(4).unsqueeze(0).repeat(10, 1, 1)
    return create_executorch_program(model, (sample_vertices, sample_normals,
                                           sample_weights, sample_indices, sample_matrices))


def get_gpu_normal_program():
    """Get pre-compiled GPU normal calculation operator"""
    model = GPUNormalCalculationOperator()
    sample_vertices = torch.randn(100, 3)
    sample_indices = torch.randint(0, 100, (50, 3))
    return create_executorch_program(model, (sample_vertices, sample_indices))


def get_gpu_batch_program():
    """Get pre-compiled GPU batch processing operator"""
    model = GPUBatchProcessor()
    sample_vertices = [torch.randn(50, 3), torch.randn(30, 3)]
    sample_indices = [torch.randint(0, 50, (20, 3)), torch.randint(0, 30, (10, 3))]
    sample_transforms = [torch.eye(4), torch.eye(4)]
    return create_executorch_program(model, (sample_vertices, sample_indices, sample_transforms))
