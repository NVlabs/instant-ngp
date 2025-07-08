/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   triangle_bvh.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>

#include <tiny-cuda-nn/gpu_memory.h>

#include <memory>

namespace ngp {

struct TriangleBvhNode {
	BoundingBox bb;
	int left_idx; // negative values indicate leaves
	int right_idx;
};

__host__ __device__ std::pair<int, float> trianglebvh_ray_intersect(
	const vec3& ro, const vec3& rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles
);

class TriangleBvh {
public:
	virtual void signed_distance_gpu(
		uint32_t n_elements,
		EMeshSdfMode mode,
		const vec3* gpu_positions,
		float* gpu_distances,
		const Triangle* gpu_triangles,
		bool use_existing_distances_as_upper_bounds,
		cudaStream_t stream
	) = 0;
	virtual void
		ray_trace_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
	virtual bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const = 0;
	virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;
	virtual void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) = 0;

	static std::unique_ptr<TriangleBvh> make();

	TriangleBvhNode* nodes_gpu() const { return m_nodes_gpu.data(); }

protected:
	std::vector<TriangleBvhNode> m_nodes;
	GPUMemory<TriangleBvhNode> m_nodes_gpu;
	TriangleBvh() {};
};

} // namespace ngp
