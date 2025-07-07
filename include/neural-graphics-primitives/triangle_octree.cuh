/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   triangle_octree.cuh
 *  @author Thomas Müller, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/thread_pool.h>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree_device.cuh>

#include <tiny-cuda-nn/gpu_memory.h>

namespace std {
template <> struct less<tcnn::u16vec4> {
	bool operator()(const tcnn::u16vec4& a, const tcnn::u16vec4& b) const {
		for (size_t i = 0; i < 4; ++i) {
			if (a[i] < b[i]) {
				return true;
			}
			if (a[i] > b[i]) {
				return false;
			}
		}
		return false;
	}
};

template <> struct hash<tcnn::u16vec4> {
	size_t operator()(const tcnn::u16vec4& x) const {
		return (size_t)x.x * 73856093 + (size_t)x.y * 19349663 + (size_t)x.z * 83492791 + (size_t)x.w * 25165843;
	}
};
} // namespace std

namespace ngp {

class TriangleOctree {
public:
	void build(const TriangleBvh& bvh, const std::vector<Triangle>& triangles, uint32_t max_depth) {
		m_nodes.clear();
		m_dual_nodes.clear();

		// Root
		m_nodes.emplace_back();
		m_dual_nodes.emplace_back();

		// Start counting at the 1 root node that we already have
		std::atomic<int> node_counter{1};
		int n_nodes = 0;

		ThreadPool pool;

		// Only generate nodes up to max_depth-1! The dual nodes will truly reach to the max depth
		for (uint8_t depth = 0; depth < max_depth - 1; ++depth) {
			int n_pending_nodes_to_build = node_counter - n_nodes;

			// Reserve enough memory as if the next layer was dense. This allows parallelizing the following code
			if (depth < max_depth - 2) {
				m_nodes.resize(node_counter + n_pending_nodes_to_build * 8);
			} else {
				// Tighten the memory back down when we're done.
				m_nodes.resize(node_counter);
			}

			int last_n_nodes = n_nodes;
			n_nodes = node_counter;
			pool.parallel_for<int>(last_n_nodes, node_counter, [&](size_t parent_idx) {
				u16vec3 child_pos_base = m_nodes[parent_idx].pos * (uint16_t)2;
				float size = std::scalbnf(1.0f, -depth - 1);

				for (uint32_t i = 0; i < 8; ++i) {
					u16vec3 child_pos = child_pos_base;
					if (i & 1) {
						++child_pos.x;
					}
					if (i & 2) {
						++child_pos.y;
					}
					if (i & 4) {
						++child_pos.z;
					}

					BoundingBox bb = {size * vec3(child_pos), size * vec3(child_pos + u16vec3(1))};

					if (!bvh.touches_triangle(bb, triangles.data())) {
						m_nodes[parent_idx].children[i] = -1;
						continue;
					}

					int node_idx = node_counter++;
					m_nodes[parent_idx].children[i] = node_idx;

					// Create regular nodes one layer less deep as the dual nodes
					if (depth < max_depth - 2) {
						m_nodes[node_idx].pos = {(uint16_t)child_pos.x, (uint16_t)child_pos.y, (uint16_t)child_pos.z};
						m_nodes[node_idx].depth = (uint8_t)(depth + 1);
					}
				}
			});
		}

		m_dual_nodes.resize(node_counter);

		tlog::success() << "Built TriangleOctree: depth=" << max_depth << " nodes=" << m_nodes.size()
						<< " dual_nodes=" << m_dual_nodes.size() << ". Populating dual nodes...";

		// TODO: find a fast lockfree hashmap implementation and parallelize the bottom for loop
		std::unordered_map<u16vec4, uint32_t> coords;
		coords.reserve(m_dual_nodes.size() * 8);
		m_n_vertices = 0;
		auto generate_dual_coords = [&](TriangleOctreeDualNode& dual_node, int depth, const u16vec3 pos) {
			for (uint32_t i = 0; i < 8; ++i) {
				u16vec4 coord = {(uint16_t)pos.x, (uint16_t)pos.y, (uint16_t)pos.z, (uint16_t)depth};
				if (i & 1) {
					++coord.x;
				}
				if (i & 2) {
					++coord.y;
				}
				if (i & 4) {
					++coord.z;
				}

				auto p = coords.insert({coord, m_n_vertices});
				if (p.second) {
					++m_n_vertices;
				}

				dual_node.vertices[i] = p.first->second;
			}
		};

		generate_dual_coords(m_dual_nodes[0], 0, {(uint16_t)0, (uint16_t)0, (uint16_t)0});
		for (auto& node : m_nodes) {
			for (uint32_t i = 0; i < 8; ++i) {
				auto child_idx = node.children[i];
				if (child_idx == -1) {
					continue;
				}

				u16vec3 child_pos = node.pos * (uint16_t)2;
				if (i & 1) {
					++child_pos.x;
				}
				if (i & 2) {
					++child_pos.y;
				}
				if (i & 4) {
					++child_pos.z;
				}

				generate_dual_coords(m_dual_nodes[child_idx], node.depth + 1, child_pos);
			}
		}

		m_depth = max_depth;

		m_nodes_gpu.resize_and_copy_from_host(m_nodes);
		m_dual_nodes_gpu.resize_and_copy_from_host(m_dual_nodes);
	}

	uint32_t depth() const { return m_depth; }

	uint32_t n_vertices() const { return m_n_vertices; }

	uint32_t n_nodes() const { return (uint32_t)m_nodes.size(); }

	uint32_t n_dual_nodes() const { return (uint32_t)m_dual_nodes.size(); }

	const TriangleOctreeNode* nodes_gpu() const { return m_nodes_gpu.data(); }

	const TriangleOctreeDualNode* dual_nodes_gpu() const { return m_dual_nodes_gpu.data(); }

private:
	std::vector<TriangleOctreeNode> m_nodes;
	std::vector<TriangleOctreeDualNode> m_dual_nodes;

	GPUMemory<TriangleOctreeNode> m_nodes_gpu;
	GPUMemory<TriangleOctreeDualNode> m_dual_nodes_gpu;

	uint32_t m_n_vertices = 0;
	uint32_t m_depth = 0;
};

} // namespace ngp
