/*
* Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/thread_pool.h>

#include <Eigen/Dense>

#include <tiny-cuda-nn/gpu_memory.h>

#include <stack>

namespace std {
	template<>
	struct less<ngp::Vector4i16> {
		bool operator()(const ngp::Vector4i16& a, const ngp::Vector4i16& b) const {
			for(size_t i = 0; i < 4; ++i) {
				if (a[i] < b[i]) return true;
				if (a[i] > b[i]) return false;
			}
			return false;
		}
	};

	template <>
	struct hash<ngp::Vector4i16> {
		size_t operator()(const ngp::Vector4i16& x) const {
			return (size_t)x.x() * 73856093 + (size_t)x.y() * 19349663 + (size_t)x.z() * 83492791 + (size_t)x.w() * 25165843;
		}
	};
}

NGP_NAMESPACE_BEGIN

struct TriangleOctreeNode {
	int children[8];
	Vector3i16 pos;
	uint8_t depth;
};

struct TriangleOctreeDualNode {
	uint32_t vertices[8];
};

inline void write_brick_voxel_positions(Eigen::Vector3f* dst, uint32_t B, float size, const Eigen::Vector3f& base_pos) {
	float rstep = size / (B - 1);
	for (uint32_t z = 0; z < B; ++z) {
		for (uint32_t y = 0; y < B; ++y) {
			for (uint32_t x = 0; x < B; ++x) {
				*dst++ = base_pos + Eigen::Vector3f{x * rstep, y * rstep, z * rstep};
			}
		}
	}
}

class TriangleOctree {
public:
	std::vector<Eigen::Vector3f> build_brick_voxel_position_list(uint32_t B) const { // brick size B^3
		std::vector<Eigen::Vector3f> brick_pos;
		brick_pos.resize(m_dual_nodes.size() * B * B * B);
		float rstep = 1.f / (B - 1);
		write_brick_voxel_positions(brick_pos.data(), B, 1.f, Eigen::Vector3f::Zero());
		uint32_t last_level = 0;
		uint32_t prev_ni = 0, ni = 0;
		for (ni = 0; ni < (uint32_t)m_nodes.size(); ++ni) {
			auto const &n = m_nodes[ni];
			if (n.depth > last_level) {
				tlog::info() << (ni-prev_ni) << " bricks at level " << last_level << " with " << (ni-prev_ni)*B*B*B << " values.";
				last_level = n.depth;
				prev_ni = ni;
			}
			float child_size = std::scalbnf(1.f, -(int)n.depth - 1);
			Vector3i16 child_pos_base = n.pos * (uint16_t)2;
			for (int i = 0; i < 8; ++i) {
				int child_idx = n.children[i];
				if (child_idx < 0)
					continue;
				Vector3i16 child_pos = child_pos_base;
				if (i&1) ++child_pos.x();
				if (i&2) ++child_pos.y();
				if (i&4) ++child_pos.z();
				Eigen::Vector3f base_pos = child_pos.cast<float>() * child_size;
				write_brick_voxel_positions(brick_pos.data() + (child_idx * B * B * B), B, child_size, base_pos);
			}
		}
		tlog::info() << (ni-prev_ni) << " bricks at level " << last_level << " with " << (ni-prev_ni)*B*B*B << " values.";
		tlog::info() << (m_dual_nodes.size()-ni) << " bricks at level " << last_level+1 << " with " << (m_dual_nodes.size()-ni)*B*B*B << " values.";
		return brick_pos;
	}

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
		for (uint8_t depth = 0; depth < max_depth-1; ++depth) {
			int n_pending_nodes_to_build = node_counter - n_nodes;

			// Reserve enough memory as if the next layer was dense. This allows parallelizing the following code
			if (depth < max_depth-2) {
				m_nodes.resize(node_counter + n_pending_nodes_to_build*8);
			} else {
				// Tighten the memory back down when we're done.
				m_nodes.resize(node_counter);
			}

			int last_n_nodes = n_nodes;
			n_nodes = node_counter;
			pool.parallel_for<int>(last_n_nodes, node_counter, [&](size_t parent_idx) {
				Vector3i16 child_pos_base = m_nodes[parent_idx].pos * (uint16_t)2;
				float size = std::scalbnf(1.0f, -depth-1);

				for (uint32_t i = 0; i < 8; ++i) {
					Vector3i16 child_pos = child_pos_base;
					if (i&1) ++child_pos.x();
					if (i&2) ++child_pos.y();
					if (i&4) ++child_pos.z();

					BoundingBox bb = {size * child_pos.cast<float>(), size * (child_pos + Vector3i16::Constant(1)).cast<float>()};

					if (!bvh.touches_triangle(bb, triangles.data())) {
						m_nodes[parent_idx].children[i] = -1;
						continue;
					}

					int node_idx = node_counter++;
					m_nodes[parent_idx].children[i] = node_idx;

					// Create regular nodes one layer less deep as the dual nodes
					if (depth < max_depth-2) {
						m_nodes[node_idx].pos = {(uint16_t)child_pos.x(), (uint16_t)child_pos.y(), (uint16_t)child_pos.z()};
						m_nodes[node_idx].depth = (uint8_t)(depth+1);
					}
				}
			});
		}

		m_dual_nodes.resize(node_counter);

		tlog::success() << "Built TriangleOctree: depth=" << max_depth << " nodes=" << m_nodes.size() << " dual_nodes=" << m_dual_nodes.size() << ". Populating dual nodes...";

		// TODO: find a fast lockfree hashmap implementation and parallelize the bottom for loop
		std::unordered_map<Vector4i16, uint32_t> coords;
		coords.reserve(m_dual_nodes.size()*8);
		m_n_vertices = 0;
		auto generate_dual_coords = [&](TriangleOctreeDualNode& dual_node, int depth, const Vector3i16 pos) {
			for (uint32_t i = 0; i < 8; ++i) {
				Vector4i16 coord = {(uint16_t)pos.x(), (uint16_t)pos.y(), (uint16_t)pos.z(), (uint16_t)depth};
				if (i&1) ++coord.x();
				if (i&2) ++coord.y();
				if (i&4) ++coord.z();

				auto p = coords.insert({coord, m_n_vertices});
				if (p.second) {
					++m_n_vertices;
				}

				dual_node.vertices[i] = p.first->second;
			}
		};

		generate_dual_coords(m_dual_nodes[0], 0, {0, 0, 0});
		for (auto& node : m_nodes) {
			for (uint32_t i = 0; i < 8; ++i) {
				auto child_idx = node.children[i];
				if (child_idx == -1) {
					continue;
				}

				Vector3i16 child_pos = node.pos * (uint16_t)2;
				if (i&1) ++child_pos.x();
				if (i&2) ++child_pos.y();
				if (i&4) ++child_pos.z();

				generate_dual_coords(m_dual_nodes[child_idx], node.depth+1, child_pos);
			}
		}

		m_depth = max_depth;

		m_nodes_gpu.resize_and_copy_from_host(m_nodes);
		m_dual_nodes_gpu.resize_and_copy_from_host(m_dual_nodes);
	}

	uint32_t depth() const {
		return m_depth;
	}

	uint32_t n_vertices() const {
		return m_n_vertices;
	}

	uint32_t n_nodes() const {
		return (uint32_t)m_nodes.size();
	}

	uint32_t n_dual_nodes() const {
		return (uint32_t)m_dual_nodes.size();
	}

	template <typename F>
	__device__ static uint8_t traverse(const TriangleOctreeNode* nodes, const TriangleOctreeDualNode* dual_nodes, int max_depth, Eigen::Vector3f pos, F fun) {
		int node_idx = 0;

		for (uint8_t depth = 0; true; ++depth) {
			fun(dual_nodes[node_idx], depth, pos);

			// Dual nodes are one layer deeper than regular nodes
			if (depth >= max_depth-1) {
				return depth+1;
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
				return depth+1;
			}
		}
		return max_depth;
	}

	__device__ static bool contains(const TriangleOctreeNode* nodes, int max_depth, Eigen::Vector3f pos) {
		const TriangleOctreeNode* node = &nodes[0];

		for (uint8_t depth = 0; depth < max_depth-1; ++depth) {
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

	__device__ static float ray_intersect(const TriangleOctreeNode* nodes, int max_depth, const Eigen::Vector3f& ro, const Eigen::Vector3f& rd) {
		FixedStack<int, 64> query_stack;
		query_stack.push(0);

		static constexpr float MAX_DIST = 1000.f;

		float mint = MAX_DIST;

		// Ensure that closer children are checked last such that they rise to the top of the stack
		uint8_t reorder_mask = 0;
		if (rd.x() > 0) reorder_mask |= 1;
		if (rd.y() > 0) reorder_mask |= 2;
		if (rd.z() > 0) reorder_mask |= 4;

		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const TriangleOctreeNode& node = nodes[idx];

			for (uint8_t aux = 0; aux < 8; ++aux) {
				uint8_t i = aux ^ reorder_mask;
				int child_idx = node.children[i];

				uint8_t depth = node.depth+1;
				// Intermediate leaf
				if (child_idx == -1) {
					continue;
				}

				Vector3i16 pos = node.pos * (uint16_t)2;
				if (i&1) ++pos.x();
				if (i&2) ++pos.y();
				if (i&4) ++pos.z();

				float size = scalbnf(1.0f, -depth);

				BoundingBox bb = {size * pos.cast<float>(), size * (pos + Vector3i16::Constant(1)).cast<float>()};
				Eigen::Vector2f t = bb.ray_intersect(ro, rd);
				if (t.y() >= 0 && t.y() < MAX_DIST && t.x() < mint) {
					// All children's children are gonna be leaves,
					// so we can cut to the chase and terminate here already.
					if (depth == max_depth-1) {
						if (t.x() >= 0) {
							mint = t.x();
						}
					} else {
						query_stack.push(child_idx);
					}
				}
			}
		}

		return mint;
	}

	const TriangleOctreeNode* nodes_gpu() const {
		return m_nodes_gpu.data();
	}

	const TriangleOctreeDualNode* dual_nodes_gpu() const {
		return m_dual_nodes_gpu.data();
	}

private:
	std::vector<TriangleOctreeNode> m_nodes;
	std::vector<TriangleOctreeDualNode> m_dual_nodes;

	tcnn::GPUMemory<TriangleOctreeNode> m_nodes_gpu;
	tcnn::GPUMemory<TriangleOctreeDualNode> m_dual_nodes_gpu;

	uint32_t m_n_vertices = 0;
	uint32_t m_depth = 0;
};

NGP_NAMESPACE_END
