/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_sdf.cuh
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/sdf_device.cuh>
#include <neural-graphics-primitives/triangle_octree_device.cuh>

using namespace ngp;

__global__ void trace_sdf(
	const ivec2 resolution,
	const float zero_offset,
	vec3* __restrict__ positions,
	float* __restrict__ distances,
	SdfPayload* __restrict__ payloads,
	BoundingBox aabb,
	float floor_y,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	int max_octree_depth,
	float distance_scale,
	float maximum_distance,
	float k,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility,
	const network_precision_t* __restrict__ params
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	bool valid = x < resolution.x && y < resolution.y;
	uint32_t idx = x + resolution.x * y;

	SdfPayload& payload = payloads[idx];
	Ray ray = {positions[idx], payload.dir};
	ray.d = normalize(ray.d);

	bool alive = valid && ray.is_valid() && payload.alive && aabb.contains(ray.o);

	float total_dist = 0.0f;
	float dist = 0.0f;
	float prev_dist = 1e20f;
	float min_vis = 1.0f;

	uint32_t n_steps = 0;

	while (true) {
		dist = ((float)eval_sdf(ray.o, params)[0] - zero_offset) * distance_scale;

		// Implicit sync is fine here, because the following nerf model wants synchronized warps anyway
		if (__all_sync(0xFFFFFFFF, !alive)) {
			break;
		}

		// All threads in the warp must execute the above MLPs for coherence reasons.
		// Starting from here, it's fine to skip computation.
		if (!alive) {
			continue;
		}

		ray.o = ray(dist);
		++n_steps;

		// Skip over regions not covered by the octree
		if (octree_nodes && !contains(octree_nodes, max_octree_depth, ray.o)) {
			float octree_dist = ray_intersect(octree_nodes, max_octree_depth, ray.o, payload.dir) + 1e-6f;
			dist += octree_dist;
			ray.o = ray(octree_dist);
		}

		// Collide with floor
		if (ray.o.y < floor_y && ray.d.y < 0.f) {
			float floor_dist = -(ray.o.y - floor_y) / ray.d.y;
			dist += floor_dist;
			ray.o = ray(floor_dist);

			alive = false;
		}

		// Shadow ray per https://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
		if (total_distances && dist > 0.0f) {
			float y = dist * dist / (2.0f * prev_dist);
			float d = sqrtf(dist * dist - y * y);

			min_vis = fminf(min_vis, k * d / fmaxf(0.0f, total_dist - y));
			prev_dist = dist;
			total_dist += dist;
		}

		alive &= dist > maximum_distance && fabsf(dist / 2) > 3 * maximum_distance && aabb.contains(ray.o);
	}

	if (!valid) {
		return;
	}

	positions[idx] = ray.o;
	distances[idx] = dist;
	payload.n_steps += n_steps;
	payload.alive = false; // All payloads should have become inactive at this point!

	// Shadow ray
	if (total_distances) {
		prev_distances[idx] = prev_dist;
		total_distances[idx] = total_dist;
		min_visibility[idx] = min_vis;
	}
}
