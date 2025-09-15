#!/usr/bin/env python3
"""
Example usage of the GLTF Skin Tool

This script demonstrates how to use the GLTFSkinTool to convert between
Blender-style node-based bones and GLTF-style skeleton systems.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gltf_module import GLTFSkinTool, GLTFDocument, GLTFNode, GLTFSkin, GLTFSkeleton
from gltf_module.structures import Transform


def create_sample_skinned_gltf():
    """Create a sample GLTF with skinned mesh for demonstration"""
    # Create nodes for a simple character rig
    nodes = []

    # Root node
    root = GLTFNode()
    root.name = "CharacterRoot"
    root.transform = Transform()
    nodes.append(root)

    # Hip bone (root of skeleton)
    hip = GLTFNode()
    hip.name = "Hip"
    hip.parent = 0  # Child of root
    hip.joint = True
    hip.transform = Transform()
    nodes.append(hip)

    # Spine bone
    spine = GLTFNode()
    spine.name = "Spine"
    spine.parent = 1  # Child of hip
    spine.joint = True
    spine.transform = Transform()
    spine.transform.origin[1] = 1.0  # Offset up (Y axis)
    nodes.append(spine)

    # Left leg bones
    left_upper_leg = GLTFNode()
    left_upper_leg.name = "LeftUpperLeg"
    left_upper_leg.parent = 1  # Child of hip
    left_upper_leg.joint = True
    left_upper_leg.transform = Transform()
    left_upper_leg.transform.origin[0] = -0.5  # Offset left (X axis)
    nodes.append(left_upper_leg)

    left_lower_leg = GLTFNode()
    left_lower_leg.name = "LeftLowerLeg"
    left_lower_leg.parent = 3  # Child of left upper leg
    left_lower_leg.joint = True
    left_lower_leg.transform = Transform()
    left_lower_leg.transform.origin[1] = -1.0  # Offset down (Y axis)
    nodes.append(left_lower_leg)

    # Right leg bones
    right_upper_leg = GLTFNode()
    right_upper_leg.name = "RightUpperLeg"
    right_upper_leg.parent = 1  # Child of hip
    right_upper_leg.joint = True
    right_upper_leg.transform = Transform()
    right_upper_leg.transform.origin[0] = 0.5  # Offset right (X axis)
    nodes.append(right_upper_leg)

    right_lower_leg = GLTFNode()
    right_lower_leg.name = "RightLowerLeg"
    right_lower_leg.parent = 5  # Child of right upper leg
    right_lower_leg.joint = True
    right_lower_leg.transform = Transform()
    right_lower_leg.transform.origin[1] = -1.0  # Offset down (Y axis)
    nodes.append(right_lower_leg)

    # Mesh node (not a joint)
    mesh_node = GLTFNode()
    mesh_node.name = "CharacterMesh"
    mesh_node.parent = 0  # Child of root
    mesh_node.mesh = 0  # References a mesh
    mesh_node.skin = 0  # References a skin
    nodes.append(mesh_node)

    # Set up parent-child relationships
    nodes[0].children = [1, 6]  # Root has hip and mesh
    nodes[1].children = [2, 3, 5]  # Hip has spine, left leg, right leg
    nodes[2].children = []  # Spine has no children
    nodes[3].children = [4]  # Left upper leg has left lower leg
    nodes[4].children = []  # Left lower leg has no children
    nodes[5].children = [6]  # Right upper leg has right lower leg (wait, this should be 7)
    nodes[5].children = [6]  # Right upper leg has right lower leg
    nodes[6].children = []  # Right lower leg has no children
    nodes[7].children = []  # Mesh has no children

    # Fix the right leg parent reference
    nodes[5].children = [6]  # Right upper leg has right lower leg (index 6)
    nodes[6].children = []  # Right lower leg has no children

    # Create skin
    skin = GLTFSkin()
    skin.name = "CharacterSkin"
    skin.joints = [1, 2, 3, 4, 5, 6]  # All bone nodes
    skin.joints_original = [1, 2, 3, 4, 5, 6]  # Same as joints for this example
    skin.non_joints = []  # No non-joint nodes in skin

    return nodes, [skin]


def demonstrate_skin_expansion():
    """Demonstrate skin expansion functionality"""
    print("=== Skin Expansion Demo ===")

    nodes, skins = create_sample_skinned_gltf()
    skin = skins[0]

    print(f"Original skin joints: {skin.joints}")
    print(f"Original skin non_joints: {skin.non_joints}")

    # Expand the skin
    success = GLTFSkinTool.expand_skin(nodes, skin)

    if success:
        print("[OK] Skin expansion successful!")
        print(f"Expanded skin joints: {skin.joints}")
        print(f"Expanded skin non_joints: {skin.non_joints}")
        print(f"Skin roots: {skin.roots}")
    else:
        print("[FAIL] Skin expansion failed!")

    print()


def demonstrate_skin_verification():
    """Demonstrate skin verification functionality"""
    print("=== Skin Verification Demo ===")

    nodes, skins = create_sample_skinned_gltf()
    skin = skins[0]

    # First expand the skin
    GLTFSkinTool.expand_skin(nodes, skin)

    # Verify the skin
    is_valid = GLTFSkinTool.verify_skin(nodes, skin)

    if is_valid:
        print("[OK] Skin verification passed!")
    else:
        print("[FAIL] Skin verification failed!")

    print()


def demonstrate_skeleton_determination():
    """Demonstrate skeleton determination functionality"""
    print("=== Skeleton Determination Demo ===")

    nodes, skins = create_sample_skinned_gltf()
    skeletons = []

    success = GLTFSkinTool.determine_skeletons(skins, nodes, skeletons)

    if success:
        print(f"[OK] Skeleton determination successful! Created {len(skeletons)} skeletons")

        for i, skeleton in enumerate(skeletons):
            print(f"Skeleton {i}:")
            print(f"  Joints: {skeleton.joints}")
            print(f"  Roots: {skeleton.roots}")

            # Create skeleton data
            skeleton_data = GLTFSkinTool.create_skeleton_from_gltf(nodes, skeleton)
            print(f"  Bones: {[bone['name'] for bone in skeleton_data['bones']]}")
    else:
        print("[FAIL] Skeleton determination failed!")

    print()


def demonstrate_full_conversion():
    """Demonstrate full node-to-skeleton conversion"""
    print("=== Full Node-to-Skeleton Conversion Demo ===")

    nodes, skins = create_sample_skinned_gltf()

    print("Original node structure:")
    for i, node in enumerate(nodes):
        parent_info = f" (parent: {node.parent})" if node.parent >= 0 else ""
        joint_info = " [JOINT]" if node.joint else ""
        print(f"  {i}: {node.name}{joint_info}{parent_info}")

    print(f"\nOriginal skins: {len(skins)}")
    for i, skin in enumerate(skins):
        print(f"  Skin {i}: {skin.name} - joints: {skin.joints}")

    # Perform full conversion
    skeletons = GLTFSkinTool.convert_node_based_to_skeleton(nodes, skins)

    if skeletons:
        print(f"\n[OK] Conversion successful! Created {len(skeletons)} skeletons")

        for i, skeleton in enumerate(skeletons):
            print(f"\nSkeleton {i}:")
            skeleton_data = skeleton.godot_skeleton
            print(f"  Name: {skeleton_data['name']}")
            print(f"  Bones: {len(skeleton_data['bones'])}")
            for bone in skeleton_data['bones']:
                parent_info = f" (parent: {skeleton_data['bone_names'][bone['parent']]})" if bone['parent'] >= 0 else ""
                print(f"    - {bone['name']}{parent_info}")
    else:
        print("\n[FAIL] Conversion failed!")

    print()


def demonstrate_bone_naming():
    """Demonstrate bone name sanitization and uniqueness"""
    print("=== Bone Naming Demo ===")

    existing_names = {"Bone", "Armature"}

    test_names = [
        "Bone",
        "Bone:Left",
        "Bone/Right",
        "Bone:Left/Upper",
        "",
        "Bone:Left/Upper/Finger1"
    ]

    print("Original names -> Sanitized unique names:")
    for name in test_names:
        sanitized = GLTFSkinTool.sanitize_bone_name(name)
        unique = GLTFSkinTool.generate_unique_bone_name(existing_names.copy(), name)
        print(f"  '{name}' -> '{sanitized}' -> '{unique}'")

    print()


def main():
    """Main demonstration function"""
    print("GLTF Skin Tool - Usage Examples")
    print("=" * 40)
    print()

    demonstrate_skin_expansion()
    demonstrate_skin_verification()
    demonstrate_skeleton_determination()
    demonstrate_full_conversion()
    demonstrate_bone_naming()

    print("Demo completed!")


if __name__ == "__main__":
    main()
