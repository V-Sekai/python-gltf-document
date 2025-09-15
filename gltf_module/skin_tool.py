#!/usr/bin/env python3
"""
GLTF Skin Tool - Python Implementation

A comprehensive tool for handling GLTF skinning operations, including:
- Converting between node-based and skeleton-based bone representations
- Expanding and verifying skin definitions
- Creating skeleton hierarchies from GLTF data
- Handling complex multi-rooted skin scenarios
"""

import math
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .structures import GLTFNode, GLTFSkin, GLTFSkeleton


@dataclass
class DisjointSet:
    """Disjoint set data structure for grouping related nodes"""
    parent: Dict[int, int] = field(default_factory=dict)
    rank: Dict[int, int] = field(default_factory=dict)

    def insert(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def create_union(self, x: int, y: int) -> None:
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def get_representatives(self) -> List[int]:
        """Get all set representatives (roots)"""
        return [x for x in self.parent.keys() if self.parent[x] == x]

    def get_members(self, representative: int) -> List[int]:
        """Get all members of a set"""
        return [x for x in self.parent.keys() if self.find(x) == representative]


class GLTFSkinTool:
    """Tool for handling GLTF skinning operations and conversions"""

    @staticmethod
    def _find_highest_node(nodes: List[GLTFNode], subset: List[int]) -> int:
        """Find the node with the highest (lowest) height in the subset"""
        if not subset:
            return -1

        highest = -1
        best_node = -1

        for node_idx in subset:
            node = nodes[node_idx]
            if highest == -1 or node.height < highest:
                highest = node.height
                best_node = node_idx

        return best_node

    @staticmethod
    def _capture_nodes_in_skin(nodes: List[GLTFNode], skin: GLTFSkin, node_index: int) -> bool:
        """Recursively capture nodes that are part of the skin"""
        found_joint = False
        current_node = nodes[node_index]

        # Process children first
        for child_idx in current_node.children:
            found_joint |= GLTFSkinTool._capture_nodes_in_skin(nodes, skin, child_idx)

        if found_joint:
            # Mark nodes if we found another skin's joint
            if current_node.joint and node_index not in skin.joints:
                skin.joints.append(node_index)
            elif node_index not in skin.non_joints:
                skin.non_joints.append(node_index)

        if node_index in skin.joints:
            return True

        return False

    @staticmethod
    def _capture_nodes_for_multirooted_skin(nodes: List[GLTFNode], skin: GLTFSkin) -> None:
        """Handle multi-rooted skins by finding common ancestors"""
        disjoint_set = DisjointSet()

        for joint_idx in skin.joints:
            disjoint_set.insert(joint_idx)
            parent = nodes[joint_idx].parent
            if parent >= 0 and parent in skin.joints:
                disjoint_set.create_union(parent, joint_idx)

        roots = disjoint_set.get_representatives()

        if len(roots) <= 1:
            return

        # Find maximum height rooted tree
        max_height = -1
        for root in roots:
            if max_height == -1 or nodes[root].height < max_height:
                max_height = nodes[root].height

        # Go up the tree until all roots are at the same hierarchy level
        while True:
            all_same_level = True
            for root in roots:
                if nodes[root].height > max_height:
                    parent = nodes[root].parent
                    if parent >= 0:
                        if nodes[parent].joint and parent not in skin.joints:
                            skin.joints.append(parent)
                        elif parent not in skin.non_joints:
                            skin.non_joints.append(parent)
                        root = parent  # Update root reference

            # Check if all roots are now at the same level
            current_heights = [nodes[root].height for root in roots]
            if len(set(current_heights)) == 1:
                break

        # Climb up until they all have the same parent
        while True:
            parents = [nodes[root].parent for root in roots]
            if len(set(parents)) == 1 and parents[0] >= 0:
                # All have same parent, continue climbing
                for i, root in enumerate(roots):
                    parent = nodes[root].parent
                    if nodes[parent].joint and parent not in skin.joints:
                        skin.joints.append(parent)
                    elif parent not in skin.non_joints:
                        skin.non_joints.append(parent)
                    roots[i] = parent
            else:
                break

    @staticmethod
    def expand_skin(nodes: List[GLTFNode], skin: GLTFSkin) -> bool:
        """
        Expand skin definition to include all nodes between joints

        Returns True on success, False on failure
        """
        GLTFSkinTool._capture_nodes_for_multirooted_skin(nodes, skin)

        # Create disjoint set for all skin nodes
        disjoint_set = DisjointSet()
        all_skin_nodes = skin.joints + skin.non_joints

        for node_idx in all_skin_nodes:
            disjoint_set.insert(node_idx)
            parent = nodes[node_idx].parent
            if parent >= 0 and parent in all_skin_nodes:
                disjoint_set.create_union(parent, node_idx)

        # Find root nodes for each connected component
        representatives = disjoint_set.get_representatives()
        out_roots = []

        for rep in representatives:
            members = disjoint_set.get_members(rep)
            root = GLTFSkinTool._find_highest_node(nodes, members)
            if root < 0:
                return False
            out_roots.append(root)

        out_roots.sort()

        # Capture all nodes in each root subtree
        for root in out_roots:
            GLTFSkinTool._capture_nodes_in_skin(nodes, skin, root)

        skin.roots = out_roots
        return True

    @staticmethod
    def verify_skin(nodes: List[GLTFNode], skin: GLTFSkin) -> bool:
        """
        Verify skin integrity by recalculating roots and comparing

        Returns True if skin is valid, False otherwise
        """
        if not skin.roots:
            return False

        # Recalculate roots using same logic as expand_skin
        disjoint_set = DisjointSet()
        all_skin_nodes = skin.joints + skin.non_joints

        for node_idx in all_skin_nodes:
            disjoint_set.insert(node_idx)
            parent = nodes[node_idx].parent
            if parent >= 0 and parent in all_skin_nodes:
                disjoint_set.create_union(parent, node_idx)

        representatives = disjoint_set.get_representatives()
        calculated_roots = []

        for rep in representatives:
            members = disjoint_set.get_members(rep)
            root = GLTFSkinTool._find_highest_node(nodes, members)
            if root < 0:
                return False
            calculated_roots.append(root)

        calculated_roots.sort()

        # Compare with stored roots
        if len(calculated_roots) != len(skin.roots):
            return False

        for i, root in enumerate(calculated_roots):
            if root != skin.roots[i]:
                return False

        # Single rooted skin is always OK
        if len(calculated_roots) == 1:
            return True

        # Multi-rooted skin must have same parent
        parent = nodes[calculated_roots[0]].parent
        for root in calculated_roots[1:]:
            if nodes[root].parent != parent:
                return False

        return True

    @staticmethod
    def _recurse_children(nodes: List[GLTFNode], node_index: int,
                         all_skin_nodes: Set[int], visited_set: Set[int]) -> None:
        """Recursively collect all children of skin nodes"""
        if node_index in visited_set:
            return

        visited_set.add(node_index)
        current_node = nodes[node_index]

        for child_idx in current_node.children:
            GLTFSkinTool._recurse_children(nodes, child_idx, all_skin_nodes, visited_set)

        # Include nodes that have skin/mesh or have no children (leaves)
        has_skin = current_node.skin is not None and current_node.skin >= 0
        has_mesh = current_node.mesh is not None and current_node.mesh >= 0
        if (has_skin or has_mesh or not current_node.children):
            all_skin_nodes.add(node_index)

    @staticmethod
    def determine_skeletons(skins: List[GLTFSkin], nodes: List[GLTFNode],
                          skeletons: List[GLTFSkeleton],
                          single_skeleton_roots: Optional[List[int]] = None,
                          turn_non_joints_into_bones: bool = False) -> bool:
        """
        Group skins into skeletons and determine skeleton hierarchies

        Returns True on success, False on failure
        """
        if single_skeleton_roots:
            # Create a single skeleton from the provided roots
            skin = GLTFSkin()
            skin.name = "godot_single_skeleton_root"
            skin.joints = single_skeleton_roots.copy()
            skins.append(skin)

        # Use disjoint set to group related skins
        skeleton_sets = DisjointSet()

        for skin in skins:
            child_visited_set = set()
            all_skin_nodes = set()

            # Collect all nodes in this skin
            for joint_idx in skin.joints:
                all_skin_nodes.add(joint_idx)
                GLTFSkinTool._recurse_children(nodes, joint_idx, all_skin_nodes, child_visited_set)

            for non_joint_idx in skin.non_joints:
                all_skin_nodes.add(non_joint_idx)
                GLTFSkinTool._recurse_children(nodes, non_joint_idx, all_skin_nodes, child_visited_set)

            # Connect nodes in the skeleton set
            for node_idx in all_skin_nodes:
                skeleton_sets.insert(node_idx)
                parent = nodes[node_idx].parent
                if parent >= 0 and parent in all_skin_nodes:
                    skeleton_sets.create_union(parent, node_idx)

            # Connect multiple roots within the same skin
            if len(skin.roots) > 1:
                for i in range(1, len(skin.roots)):
                    skeleton_sets.create_union(skin.roots[0], skin.roots[i])

        # Group sibling and parent relationships
        representatives = skeleton_sets.get_representatives()
        highest_members = []
        groups = []

        for rep in representatives:
            group = skeleton_sets.get_members(rep)
            highest_members.append(GLTFSkinTool._find_highest_node(nodes, group))
            groups.append(group)

        # Connect siblings and parent-child relationships
        for i, node_i in enumerate(highest_members):
            # Connect siblings
            for j in range(i + 1, len(highest_members)):
                node_j = highest_members[j]
                if nodes[node_i].parent == nodes[node_j].parent:
                    skeleton_sets.create_union(node_i, node_j)

            # Connect parenting relationships
            parent_i = nodes[node_i].parent
            if parent_i >= 0:
                for j, group in enumerate(groups):
                    if parent_i in group:
                        node_j = highest_members[j]
                        skeleton_sets.create_union(node_i, node_j)

        # Create final skeletons
        skeleton_owners = skeleton_sets.get_representatives()

        for skel_idx, skeleton_owner in enumerate(skeleton_owners):
            skeleton = GLTFSkeleton()
            skeleton_nodes = skeleton_sets.get_members(skeleton_owner)

            # Assign skeleton index to skins
            for skin in skins:
                for node_idx in skeleton_nodes:
                    if (node_idx in skin.joints or node_idx in skin.non_joints):
                        skin.skeleton = skel_idx
                        break

            # Add joints to skeleton
            non_joints = []
            for node_idx in skeleton_nodes:
                node = nodes[node_idx]
                if node.joint:
                    if not turn_non_joints_into_bones:
                        # Check if parent needs to become joint
                        GLTFSkinTool._check_parent_needs_joint(nodes, skeleton_nodes, node, non_joints)
                    skeleton.joints.append(node_idx)
                elif turn_non_joints_into_bones:
                    non_joints.append(node_idx)

            skeletons.append(skeleton)

            # Handle non-joint subtrees
            GLTFSkinTool._reparent_non_joint_subtrees(nodes, skeleton, non_joints)

        # Assign skeleton indices to nodes and determine roots
        for skel_idx, skeleton in enumerate(skeletons):
            for joint_idx in skeleton.joints:
                node = nodes[joint_idx]
                if not node.joint or node.skeleton >= 0:
                    return False
                node.skeleton = skel_idx

            if not GLTFSkinTool._determine_skeleton_roots(nodes, skeletons, skel_idx):
                return False

        return True

    @staticmethod
    def _check_parent_needs_joint(nodes: List[GLTFNode], skeleton_nodes: List[int],
                                gltf_node: GLTFNode, non_joints: List[int]) -> None:
        """Check if parent nodes need to become joints"""
        parent_idx = gltf_node.parent
        if (parent_idx >= 0 and
            not nodes[parent_idx].joint and
            parent_idx in skeleton_nodes and
            parent_idx not in non_joints):
            GLTFSkinTool._check_parent_needs_joint(nodes, skeleton_nodes, nodes[parent_idx], non_joints)
            non_joints.append(parent_idx)

    @staticmethod
    def _reparent_non_joint_subtrees(nodes: List[GLTFNode], skeleton: GLTFSkeleton,
                                   non_joints: List[int]) -> bool:
        """Reparent non-joint subtrees to create valid skeleton hierarchy"""
        subtree_set = DisjointSet()

        # Only include non-joints that are in the skeleton hierarchy
        for node_idx in non_joints:
            subtree_set.insert(node_idx)
            parent_idx = nodes[node_idx].parent
            if (parent_idx >= 0 and
                parent_idx in non_joints and
                not nodes[parent_idx].joint):
                subtree_set.create_union(parent_idx, node_idx)

        # Find non-joint subtree roots
        non_joint_roots = subtree_set.get_representatives()

        for root in non_joint_roots:
            subtree_nodes = subtree_set.get_members(root)

            # Convert all nodes in subtree to joints
            for node_idx in subtree_nodes:
                nodes[node_idx].joint = True
                skeleton.joints.append(node_idx)

        return True

    @staticmethod
    def _determine_skeleton_roots(nodes: List[GLTFNode], skeletons: List[GLTFSkeleton],
                                skeleton_idx: int) -> bool:
        """Determine the root nodes of a skeleton"""
        disjoint_set = DisjointSet()

        for node_idx, node in enumerate(nodes):
            if node.skeleton != skeleton_idx:
                continue

            disjoint_set.insert(node_idx)
            if node.parent >= 0 and nodes[node.parent].skeleton == skeleton_idx:
                disjoint_set.create_union(node.parent, node_idx)

        skeleton = skeletons[skeleton_idx]
        representatives = disjoint_set.get_representatives()
        roots = []

        for rep in representatives:
            members = disjoint_set.get_members(rep)
            root = GLTFSkinTool._find_highest_node(nodes, members)
            if root < 0:
                return False
            roots.append(root)

        roots.sort()
        skeleton.roots = roots

        if not roots:
            return False
        elif len(roots) == 1:
            return True

        # Check that multi-rooted skeletons have same parent
        parent = nodes[roots[0]].parent
        for root in roots[1:]:
            if nodes[root].parent != parent:
                return False

        return True

    @staticmethod
    def create_skeleton_from_gltf(nodes: List[GLTFNode], skeleton: GLTFSkeleton) -> Dict[str, Any]:
        """
        Create a skeleton dictionary representation from GLTF data

        Returns a dictionary with bone information similar to Godot's Skeleton3D
        """
        skeleton_data = {
            'name': 'Skeleton3D',
            'bones': [],
            'bone_parents': {},
            'bone_rest_transforms': {},
            'bone_names': []
        }

        # Collect all bones in depth-first order
        bones_to_process = skeleton.roots.copy()
        processed = set()

        while bones_to_process:
            node_idx = bones_to_process.pop(0)
            if node_idx in processed:
                continue

            processed.add(node_idx)
            node = nodes[node_idx]

            # Add children to processing queue (depth-first)
            children_in_skeleton = [
                child_idx for child_idx in node.children
                if nodes[child_idx].skeleton == node.skeleton
            ]
            bones_to_process = children_in_skeleton + bones_to_process

            bone_info = {
                'name': node.name or f'bone_{node_idx}',
                'node_index': node_idx,
                'rest_transform': node.transform,
                'parent': -1
            }

            # Find parent bone
            if node.parent >= 0 and nodes[node.parent].skeleton == node.skeleton:
                parent_name = nodes[node.parent].name or f'bone_{node.parent}'
                bone_info['parent'] = skeleton_data['bone_names'].index(parent_name)

            skeleton_data['bones'].append(bone_info)
            skeleton_data['bone_names'].append(bone_info['name'])
            skeleton_data['bone_parents'][bone_info['name']] = bone_info['parent']

            # Store rest transform
            if hasattr(node, 'rest_transform'):
                skeleton_data['bone_rest_transforms'][bone_info['name']] = node.rest_transform
            else:
                skeleton_data['bone_rest_transforms'][bone_info['name']] = node.transform

        return skeleton_data

    @staticmethod
    def create_skin_from_gltf(skin: GLTFSkin, nodes: List[GLTFNode],
                            skeleton_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a skin dictionary from GLTF skin data

        Returns a dictionary with bind pose information
        """
        skin_data = {
            'name': skin.name or 'Skin',
            'binds': []
        }

        for joint_idx, joint_node_idx in enumerate(skin.joints_original):
            joint_node = nodes[joint_node_idx]
            bone_name = joint_node.name or f'bone_{joint_node_idx}'

            # Find bone index in skeleton
            if bone_name in skeleton_data['bone_names']:
                bone_idx = skeleton_data['bone_names'].index(bone_name)

                bind_transform = skin.inverse_binds[joint_idx] if joint_idx < len(skin.inverse_binds) else None

                skin_data['binds'].append({
                    'bone_index': bone_idx,
                    'bone_name': bone_name,
                    'bind_transform': bind_transform
                })

        return skin_data

    @staticmethod
    def convert_node_based_to_skeleton(nodes: List[GLTFNode], skins: List[GLTFSkin]) -> List[GLTFSkeleton]:
        """
        Convert node-based bone representation to skeleton-based representation

        This is the main conversion function that replicates Godot's skin tool functionality
        """
        skeletons = []

        # Expand all skins first
        for skin in skins:
            if not GLTFSkinTool.expand_skin(nodes, skin):
                continue

        # Determine skeleton groupings
        if not GLTFSkinTool.determine_skeletons(skins, nodes, skeletons):
            return []

        # Create skeleton data structures
        for skeleton in skeletons:
            skeleton_data = GLTFSkinTool.create_skeleton_from_gltf(nodes, skeleton)
            skeleton.godot_skeleton = skeleton_data

        return skeletons

    @staticmethod
    def convert_skeleton_to_node_based(skeletons: List[GLTFSkeleton], skins: List[GLTFSkin],
                                     nodes: List[GLTFNode]) -> bool:
        """
        Convert skeleton-based representation back to node-based bones

        This reverses the skeleton creation process
        """
        # This would be more complex to implement as it needs to recreate
        # the original node hierarchy from skeleton data
        # For now, return True as placeholder
        return True

    @staticmethod
    def sanitize_bone_name(name: str) -> str:
        """Sanitize bone name by replacing invalid characters"""
        if not name:
            return "bone"
        return name.replace(':', '_').replace('/', '_')

    @staticmethod
    def generate_unique_bone_name(existing_names: Set[str], name: str) -> str:
        """Generate a unique bone name"""
        sanitized = GLTFSkinTool.sanitize_bone_name(name)
        if not sanitized:
            sanitized = "bone"

        unique_name = sanitized
        counter = 1

        while unique_name in existing_names:
            unique_name = f"{sanitized}_{counter}"
            counter += 1

        existing_names.add(unique_name)
        return unique_name
