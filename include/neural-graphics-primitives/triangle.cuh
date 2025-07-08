/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   triangle_bvh.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 *  @brief  CUDA/C++ triangle implementation.
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>

namespace ngp {

inline NGP_HOST_DEVICE float normdot(const vec3 &a, const vec3 &b) {
	float div = length(a) * length(b);
	if (div == 0.0f) {
		return 1.0f;
	}

	return dot(a, b) / div;
}

inline NGP_HOST_DEVICE float angle(const vec3 &a, const vec3 &b) {
	return acosf(clamp(normdot(a, b), -1.0f, 1.0f));
}

struct Triangle {
	NGP_HOST_DEVICE vec3 sample_uniform_position(const vec2& sample) const {
		float sqrt_x = sqrt(sample.x);
		float factor0 = 1.0f - sqrt_x;
		float factor1 = sqrt_x * (1.0f - sample.y);
		float factor2 = sqrt_x * sample.y;

		return factor0 * a + factor1 * b + factor2 * c;
	}

	NGP_HOST_DEVICE float surface_area() const {
		return 0.5f * length(cross(b - a, c - a));
	}

	NGP_HOST_DEVICE vec3 normal() const {
		return normalize(cross(b - a, c - a));
	}

	NGP_HOST_DEVICE const vec3 &operator[](uint32_t i) const {
		return i == 0 ? a : (i == 1 ? b : c);
	}

	NGP_HOST_DEVICE float angle_at_vertex(uint32_t i) const {
		vec3 v1 = (*this)[i] - (*this)[(i + 1) % 3];
		vec3 v2 = (*this)[i] - (*this)[(i + 2) % 3];
		return angle(v1, v2);
	}

	NGP_HOST_DEVICE uint32_t closest_vertex_idx(const vec3 &pos) const {
		float mag1 = length2(pos - a);
		float mag2 = length2(pos - b);
		float mag3 = length2(pos - c);

		float minv = min(vec3{ mag1, mag2, mag3 });

		if (minv == mag1) {
			return 0;
		} else if (minv == mag2) {
			return 1;
		} else {
			return 2;
		}
	}

	NGP_HOST_DEVICE float angle_at_pos(const vec3 &pos) const {
		return angle_at_vertex(closest_vertex_idx(pos));
	}

	// based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
	NGP_HOST_DEVICE float ray_intersect(const vec3 &ro, const vec3 &rd, vec3& n) const {
		vec3 v1v0 = b - a;
		vec3 v2v0 = c - a;
		vec3 rov0 = ro - a;
		n = cross(v1v0, v2v0);
		vec3 q = cross(rov0, rd);
		float d = 1.0f / dot(rd, n);
		float u = d * -dot(q, v2v0);
		float v = d *  dot(q, v1v0);
		float t = d * -dot(n, rov0);
		if (u < 0.0f || u > 1.0f || v < 0.0f || (u+v) > 1.0f || t < 0.0f) {
			t = std::numeric_limits<float>::max();
		}
		return t;
	}

	NGP_HOST_DEVICE float ray_intersect(const vec3 &ro, const vec3 &rd) const {
		vec3 n;
		return ray_intersect(ro, rd, n);
	}

	// based on https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
	NGP_HOST_DEVICE float distance_sq(const vec3& pos) const {
		vec3 v21 = b - a; vec3 p1 = pos - a;
		vec3 v32 = c - b; vec3 p2 = pos - b;
		vec3 v13 = a - c; vec3 p3 = pos - c;
		vec3 nor = cross(v21, v13);

		return
			// inside/outside test
			(sign(dot(cross(v21, nor), p1)) + sign(dot(cross(v32, nor), p2)) + sign(dot(cross(v13, nor), p3)) < 2.0f)
			?
			// 3 edges
			min(vec3{
				length2(v21 * clamp(dot(v21, p1) / length2(v21), 0.0f, 1.0f)-p1),
				length2(v32 * clamp(dot(v32, p2) / length2(v32), 0.0f, 1.0f)-p2),
				length2(v13 * clamp(dot(v13, p3) / length2(v13), 0.0f, 1.0f)-p3),
			})
			:
			// 1 face
			dot(nor, p1) * dot(nor, p1) / length2(nor);
	}

	NGP_HOST_DEVICE float distance(const vec3& pos) const {
		return sqrt(distance_sq(pos));
	}

	NGP_HOST_DEVICE bool point_in_triangle(const vec3& p) const {
		// Move the triangle so that the point becomes the
		// triangles origin
		vec3 local_a = a - p;
		vec3 local_b = b - p;
		vec3 local_c = c - p;

		// The point should be moved too, so they are both
		// relative, but because we don't use p in the
		// equation anymore, we don't need it!
		// p -= p;

		// Compute the normal vectors for triangles:
		// u = normal of PBC
		// v = normal of PCA
		// w = normal of PAB

		vec3 u = cross(local_b, local_c);
		vec3 v = cross(local_c, local_a);
		vec3 w = cross(local_a, local_b);

		// Test to see if the normals are facing the same direction.
		// If yes, the point is inside, otherwise it isn't.
		return dot(u, v) >= 0.0f && dot(u, w) >= 0.0f;
	}

	NGP_HOST_DEVICE vec3 closest_point_to_line(const vec3& a, const vec3& b, const vec3& c) const {
		float t = dot(c - a, b - a) / dot(b - a, b - a);
		t = max(min(t, 1.0f), 0.0f);
		return a + t * (b - a);
	}

	NGP_HOST_DEVICE vec3 closest_point(vec3 point) const {
		point -= dot(normal(), point - a) * normal();

		if (point_in_triangle(point)) {
			return point;
		}

		vec3 c1 = closest_point_to_line(a, b, point);
		vec3 c2 = closest_point_to_line(b, c, point);
		vec3 c3 = closest_point_to_line(c, a, point);

		float mag1 = length2(point - c1);
		float mag2 = length2(point - c2);
		float mag3 = length2(point - c3);

		float min = tcnn::min(vec3{mag1, mag2, mag3});

		if (min == mag1) {
			return c1;
		} else if (min == mag2) {
			return c2;
		} else {
			return c3;
		}
	}

	NGP_HOST_DEVICE vec3 centroid() const {
		return (a + b + c) / 3.0f;
	}

	NGP_HOST_DEVICE float centroid(int axis) const {
		return (a[axis] + b[axis] + c[axis]) / 3;
	}

	NGP_HOST_DEVICE void get_vertices(vec3 v[3]) const {
		v[0] = a;
		v[1] = b;
		v[2] = c;
	}

	vec3 a, b, c;
};

}
