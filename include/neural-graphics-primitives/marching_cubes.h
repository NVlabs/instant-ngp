/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   marching_cubes.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_host.h>

#include <tiny-cuda-nn/common.h>

namespace ngp {

ivec3 get_marching_cubes_res(uint32_t res_1d, const BoundingBox& render_aabb);

void marching_cubes_gpu(cudaStream_t stream, BoundingBox render_aabb, mat3 render_aabb_to_local, ivec3 res_3d, float thresh, const GPUMemory<float>& density, GPUMemory<vec3>& vert_out, GPUMemory<uint32_t>& indices_out);

// computes the average of the 1ring of all verts, as homogenous coordinates
void compute_mesh_1ring(const GPUMemory<vec3>& verts, const GPUMemory<uint32_t>& indices, GPUMemory<vec4>& output_pos, GPUMemory<vec3>& output_normals);

void compute_mesh_opt_gradients(
	float thresh,
	const GPUMemory<vec3>& verts,
	const GPUMemory<vec3>& vert_normals,
	const GPUMemory<vec4>& verts_smoothed,
	const network_precision_t* densities,
	uint32_t input_gradient_width,
	const float* input_gradients,
	GPUMemory<vec3>& verts_gradient_out,
	float k_smooth_amount,
	float k_density_amount,
	float k_inflate_amount
);

void save_mesh(
	GPUMemory<vec3>& verts,
	GPUMemory<vec3>& normals,
	GPUMemory<vec3>& colors,
	GPUMemory<uint32_t>& indices,
	const fs::path& path,
	bool unwrap_it,
	float nerf_scale,
	vec3 nerf_offset
);

#ifdef NGP_GUI
void draw_mesh_gl(
	const GPUMemory<vec3>& verts,
	const GPUMemory<vec3>& normals,
	const GPUMemory<vec3>& cols,
	const GPUMemory<uint32_t>& indices,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	int mesh_render_mode
);

void glCheckError(const char* file, unsigned int line);
uint32_t compile_shader(bool pixel, const char* code);
bool check_shader(uint32_t handle, const char* desc, bool program);
#endif

void save_density_grid_to_png(const GPUMemory<float>& density, const fs::path& path, ivec3 res3d, float thresh, bool swap_y_z = true, float density_range = 4.f);
void save_rgba_grid_to_png_sequence(const GPUMemory<vec4>& rgba, const fs::path& path, ivec3 res3d, bool swap_y_z = true);
void save_rgba_grid_to_raw_file(const GPUMemory<vec4>& rgba, const fs::path& path, ivec3 res3d, bool swap_y_z, int cascade);

}
