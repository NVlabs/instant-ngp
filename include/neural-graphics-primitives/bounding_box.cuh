/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   bounding_box.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 *  @brief  CUDA/C++ AABB implementation.
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>

NGP_NAMESPACE_BEGIN

template <int N_POINTS>
NGP_HOST_DEVICE inline void project(vec3 points[N_POINTS], const vec3& axis, float& min, float& max) {
	min = std::numeric_limits<float>::infinity();
	max = -std::numeric_limits<float>::infinity();

	NGP_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N_POINTS; ++i) {
		float val = dot(axis, points[i]);

		if (val < min) {
			min = val;
		}

		if (val > max) {
			max = val;
		}
	}
}

struct BoundingBox {
	NGP_HOST_DEVICE BoundingBox() {}

	NGP_HOST_DEVICE BoundingBox(const vec3& a, const vec3& b) : min{a}, max{b} {}

	NGP_HOST_DEVICE explicit BoundingBox(const Triangle& tri) {
		min = max = tri.a;
		enlarge(tri.b);
		enlarge(tri.c);
	}

	BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end) {
		min = max = begin->a;
		for (auto it = begin; it != end; ++it) {
			enlarge(*it);
		}
	}

	NGP_HOST_DEVICE void enlarge(const BoundingBox& other) {
		min = glm::min(min, other.min);
		max = glm::max(max, other.max);
	}

	NGP_HOST_DEVICE void enlarge(const Triangle& tri) {
		enlarge(tri.a);
		enlarge(tri.b);
		enlarge(tri.c);
	}

	NGP_HOST_DEVICE void enlarge(const vec3& point) {
		min = glm::min(min, point);
		max = glm::max(max, point);
	}

	NGP_HOST_DEVICE void inflate(float amount) {
		min -= vec3(amount);
		max += vec3(amount);
	}

	NGP_HOST_DEVICE vec3 diag() const {
		return max - min;
	}

	NGP_HOST_DEVICE vec3 relative_pos(const vec3& pos) const {
		return (pos - min) / diag();
	}

	NGP_HOST_DEVICE vec3 center() const {
		return 0.5f * (max + min);
	}

	NGP_HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
		BoundingBox result = *this;
		result.min = glm::max(result.min, other.min);
		result.max = glm::min(result.max, other.max);
		return result;
	}

	NGP_HOST_DEVICE bool intersects(const BoundingBox& other) const {
		return !intersection(other).is_empty();
	}

	// Based on the separating axis theorem
	// (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf)
	// Code adapted from a C# implementation at stack overflow
	// https://stackoverflow.com/a/17503268
	NGP_HOST_DEVICE bool intersects(const Triangle& triangle) const {
		float triangle_min, triangle_max;
		float box_min, box_max;

		// Test the box normals (x-, y- and z-axes)
		vec3 box_normals[3] = {
			vec3{1.0f, 0.0f, 0.0f},
			vec3{0.0f, 1.0f, 0.0f},
			vec3{0.0f, 0.0f, 1.0f},
		};

		vec3 triangle_normal = triangle.normal();
		vec3 triangle_verts[3];
		triangle.get_vertices(triangle_verts);

		for (int i = 0; i < 3; i++) {
			project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
			if (triangle_max < min[i] || triangle_min > max[i]) {
				return false; // No intersection possible.
			}
		}

		vec3 verts[8];
		get_vertices(verts);

		// Test the triangle normal
		float triangle_offset = dot(triangle_normal, triangle.a);
		project<8>(verts, triangle_normal, box_min, box_max);
		if (box_max < triangle_offset || box_min > triangle_offset) {
			return false; // No intersection possible.
		}

		// Test the nine edge cross-products
		vec3 edges[3] = {
			triangle.a - triangle.b,
			triangle.a - triangle.c,
			triangle.b - triangle.c,
		};

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				// The box normals are the same as it's edge tangents
				vec3 axis = cross(edges[i], box_normals[j]);
				project<8>(verts, axis, box_min, box_max);
				project<3>(triangle_verts, axis, triangle_min, triangle_max);
				if (box_max < triangle_min || box_min > triangle_max)
					return false; // No intersection possible
			}
		}

		// No separating axis found.
		return true;
	}

	NGP_HOST_DEVICE vec2 ray_intersect(const vec3& pos, const vec3& dir) const {
		float tmin = (min.x - pos.x) / dir.x;
		float tmax = (max.x - pos.x) / dir.x;

		if (tmin > tmax) {
			tcnn::host_device_swap(tmin, tmax);
		}

		float tymin = (min.y - pos.y) / dir.y;
		float tymax = (max.y - pos.y) / dir.y;

		if (tymin > tymax) {
			tcnn::host_device_swap(tymin, tymax);
		}

		if (tmin > tymax || tymin > tmax) {
			return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
		}

		if (tymin > tmin) {
			tmin = tymin;
		}

		if (tymax < tmax) {
			tmax = tymax;
		}

		float tzmin = (min.z - pos.z) / dir.z;
		float tzmax = (max.z - pos.z) / dir.z;

		if (tzmin > tzmax) {
			tcnn::host_device_swap(tzmin, tzmax);
		}

		if (tmin > tzmax || tzmin > tmax) {
			return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
		}

		if (tzmin > tmin) {
			tmin = tzmin;
		}

		if (tzmax < tmax) {
			tmax = tzmax;
		}

		return { tmin, tmax };
	}

	NGP_HOST_DEVICE bool is_empty() const {
		return any(lessThan(max, min));
	}

	NGP_HOST_DEVICE bool contains(const vec3& p) const {
		return
			p.x >= min.x && p.x <= max.x &&
			p.y >= min.y && p.y <= max.y &&
			p.z >= min.z && p.z <= max.z;
	}

	/// Calculate the squared point-AABB distance
	NGP_HOST_DEVICE float distance(const vec3& p) const {
		return sqrt(distance_sq(p));
	}

	NGP_HOST_DEVICE float distance_sq(const vec3& p) const {
		return length2(glm::max(glm::max(min - p, p - max), vec3(0.0f)));
	}

	NGP_HOST_DEVICE float signed_distance(const vec3& p) const {
		vec3 q = abs(p - min) - diag();
		return length(glm::max(q, vec3(0.0f))) + std::min(compMax(q), 0.0f);
	}

	NGP_HOST_DEVICE void get_vertices(vec3 v[8]) const {
		v[0] = {min.x, min.y, min.z};
		v[1] = {min.x, min.y, max.z};
		v[2] = {min.x, max.y, min.z};
		v[3] = {min.x, max.y, max.z};
		v[4] = {max.x, min.y, min.z};
		v[5] = {max.x, min.y, max.z};
		v[6] = {max.x, max.y, min.z};
		v[7] = {max.x, max.y, max.z};
	}

	vec3 min = vec3(std::numeric_limits<float>::infinity());
	vec3 max = vec3(-std::numeric_limits<float>::infinity());
};

inline std::ostream& operator<<(std::ostream& os, const ngp::BoundingBox& bb) {
	os << "[";
	os << "min=[" << bb.min.x << "," << bb.min.y << "," << bb.min.z << "], ";
	os << "max=[" << bb.max.x << "," << bb.max.y << "," << bb.max.z << "]";
	os << "]";
	return os;
}

NGP_NAMESPACE_END
