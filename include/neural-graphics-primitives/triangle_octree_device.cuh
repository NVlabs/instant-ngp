/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   triangle_octree_device.cuh
 *  @author Thomas Müller, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_device.cuh>

namespace ngp {

struct TriangleOctreeNode {
	int children[8];
	u16vec3 pos;
	uint8_t depth;
};

struct TriangleOctreeDualNode {
	uint32_t vertices[8];
};

template <typename F>
__device__ uint8_t traverse(const TriangleOctreeNode* nodes, const TriangleOctreeDualNode* dual_nodes, int max_depth, vec3 pos, F fun) {
	int node_idx = 0;

	for (uint8_t depth = 0; true; ++depth) {
		fun(dual_nodes[node_idx], depth, pos);

		// Dual nodes are one layer deeper than regular nodes
		if (depth >= max_depth - 1) {
			return depth + 1;
		}

		// Traverse

		uint8_t child_in_node = 0;

		NGP_PRAGMA_UNROLL
		for (uint8_t i = 0; i < 3; ++i) {
			if (pos[i] >= 0.5f) {
				child_in_node |= (1 << i);
				pos[i] = (pos[i] - 0.5f) * 2;
			} else {
				pos[i] *= 2;
			}
		}

		node_idx = nodes[node_idx].children[child_in_node];

		if (node_idx < 0) {
			return depth + 1;
		}
	}
	return max_depth;
}

inline __device__ bool contains(const TriangleOctreeNode* nodes, int max_depth, vec3 pos) {
	const TriangleOctreeNode* node = &nodes[0];

	for (uint8_t depth = 0; depth < max_depth - 1; ++depth) {
		uint8_t child_in_node = 0;

		NGP_PRAGMA_UNROLL
		for (uint8_t i = 0; i < 3; ++i) {
			if (pos[i] >= 0.5f) {
				child_in_node |= (1 << i);
				pos[i] = (pos[i] - 0.5f) * 2;
			} else {
				pos[i] *= 2;
			}
		}

		int child_idx = node->children[child_in_node];

		if (child_idx < 0) {
			return false;
		}

		node = &nodes[child_idx];
	}

	return true;
}

inline __device__ float ray_intersect(const TriangleOctreeNode* nodes, int max_depth, const vec3& ro, const vec3& rd) {
	FixedStack<int, 64> query_stack;
	query_stack.push(0);

	static constexpr float MAX_DIST = 1000.f;

	float mint = MAX_DIST;

	// Ensure that closer children are checked last such that they rise to the top of the stack
	uint8_t reorder_mask = 0;
	if (rd.x > 0) {
		reorder_mask |= 1;
	}
	if (rd.y > 0) {
		reorder_mask |= 2;
	}
	if (rd.z > 0) {
		reorder_mask |= 4;
	}

	while (!query_stack.empty()) {
		int idx = query_stack.pop();

		const TriangleOctreeNode& node = nodes[idx];

		for (uint8_t aux = 0; aux < 8; ++aux) {
			uint8_t i = aux ^ reorder_mask;
			int child_idx = node.children[i];

			uint8_t depth = node.depth + 1;
			// Intermediate leaf
			if (child_idx == -1) {
				continue;
			}

			u16vec3 pos = node.pos * (uint16_t)2;
			if (i & 1) {
				++pos.x;
			}
			if (i & 2) {
				++pos.y;
			}
			if (i & 4) {
				++pos.z;
			}

			float size = scalbnf(1.0f, -depth);

			BoundingBox bb = {size * vec3(pos), size * vec3(pos + u16vec3(1))};
			vec2 t = bb.ray_intersect(ro, rd);
			if (t.y >= 0 && t.y < MAX_DIST && t.x < mint) {
				// All children's children are gonna be leaves,
				// so we can cut to the chase and terminate here already.
				if (depth == max_depth - 1) {
					if (t.x >= 0) {
						mint = t.x;
					}
				} else {
					query_stack.push(child_idx);
				}
			}
		}
	}

	return mint;
}

} // namespace ngp
