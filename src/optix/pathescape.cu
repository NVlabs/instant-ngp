/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   pathescape.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Minimal optix program.
 */

#include <neural-graphics-primitives/random_val.cuh>
#include <neural-graphics-primitives/common_device.cuh>
#include <optix.h>

#include "pathescape.h"

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

extern "C" {
	__constant__ PathEscape::Params params;
}

struct Onb {
	inline __device__ Onb(const Vector3f& normal) {
		m_normal = normal;

		if (fabs(m_normal.x()) > fabs(m_normal.z())) {
			m_binormal.x() = -m_normal.y();
			m_binormal.y() =  m_normal.x();
			m_binormal.z() =  0;
		} else {
			m_binormal.x() =  0;
			m_binormal.y() = -m_normal.z();
			m_binormal.z() =  m_normal.y();
		}

		m_binormal = m_binormal.normalized();
		m_tangent = m_binormal.cross(m_normal);
	}

	inline __device__ void inverse_transform(Vector3f& p) const {
		p = p.x()*m_tangent + p.y()*m_binormal + p.z()*m_normal;
	}

	Vector3f m_tangent;
	Vector3f m_binormal;
	Vector3f m_normal;
};

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	Vector3f query_point = params.ray_origins[idx.x];

	static constexpr uint32_t N_PATHS = 32;
	static constexpr uint32_t N_BOUNCES = 4;

	default_rng_t rng;
	rng.advance(idx.x * 4 * N_PATHS * N_BOUNCES);
	uint32_t n_escaped = 0;
	for (uint32_t i = 0; i < N_PATHS; ++i) {
		Vector3f ray_origin = query_point;
		Vector3f ray_direction = random_dir(rng);

		for (uint32_t j = 0; j < N_BOUNCES; ++j) {
			// Trace the stab ray against our scene hierarchy
			unsigned int p0;
			optixTrace(
				params.handle,
				to_float3(ray_origin),
				to_float3(ray_direction),
				0.0f,                // Min intersection distance
				1e16f,               // Max intersection distance
				0.0f,                // rayTime
				OptixVisibilityMask(255), // Specify always visible
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				0,                   // SBT offset
				1,                   // SBT stride
				0,                   // missSBTIndex
				p0
			);

			// If the ray didn't escape, p0 contains the index of the triangle that was hit.
			if ((int)p0 == -1) {
				// 2 rays escaped. We are definitely outside and the escape was likely not a numerical fluke.
				// Distance doesn't need to be signed.
				if (++n_escaped > 2) {
					return;
				}
				break;
			}

			Vector3f N_0;
			float t = params.triangles[p0].ray_intersect(ray_origin, ray_direction, N_0);
			const Vector3f N = faceforward(N_0, -ray_direction, N_0).normalized();

			// Prevent self-intersections by subtracting 1e-3f from the target distance.
			ray_origin += ray_direction * fmaxf(0.0f, t - 1e-3f);

			ray_direction = random_dir_cosine(rng);
			Onb onb(N);
			onb.inverse_transform(ray_direction);
		}
	}
	params.distances[idx.x] = -params.distances[idx.x];
}

extern "C" __global__ void __miss__ms() {
	optixSetPayload_0((uint32_t)-1);
}

extern "C" __global__ void __closesthit__ch() {
	optixSetPayload_0(optixGetPrimitiveIndex());
}

NGP_NAMESPACE_END
