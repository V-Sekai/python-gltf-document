#!/usr/bin/env python3
"""
GLTF Exporter

This module provides GLTF export functionality, converting internal GLTF state
back to GLTF/GLB files that pass validation.
"""

import json
import base64
import struct
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from .gltf_state import GLTFState
from .structures import *
from .logger import get_logger


class GLTFExporter:
    """
    GLTF exporter for writing GLTF state back to files.

    Matches Godot's GLTF export functionality.
    """

    def __init__(self):
        """Initialize the GLTF exporter"""
        self.logger = get_logger('exporter')

    def write_to_filesystem(self, state: GLTFState, path: str) -> int:
        """
        Write GLTF state to filesystem.

        Matches Godot's GLTFDocument::write_to_filesystem API.
        Returns 0 on success, non-zero on failure.
        """
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Determine if we should export as GLB or GLTF
            if path.lower().endswith('.glb'):
                return self._export_glb(state, path)
            else:
                return self._export_gltf(state, path)

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return 1

    def _export_gltf(self, state: GLTFState, path: str) -> int:
        """Export as GLTF (JSON) format"""
        try:
            # Build the GLTF JSON structure
            gltf_data = self._build_gltf_json(state)

            # Write JSON file
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(gltf_data, f, indent=2, ensure_ascii=False)

            # Write buffer files if needed
            self._write_buffer_files(state, path)

            return 0

        except Exception as e:
            self.logger.error(f"GLTF export failed: {e}")
            return 1

    def _export_glb(self, state: GLTFState, path: str) -> int:
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

            return 0

        except Exception as e:
            self.logger.error(f"GLB export failed: {e}")
            return 1

    def _build_gltf_json(self, state: GLTFState) -> Dict[str, Any]:
        """Build the GLTF JSON structure from state"""
        gltf = {}

        # Asset info
        gltf['asset'] = {
            'version': '2.0',
            'generator': state.asset.generator or 'GLTF Python Module'
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
