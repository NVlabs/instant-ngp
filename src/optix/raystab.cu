/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   raystab.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Minimal optix program.
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/random_val.cuh>

#include <optix.h>

#include "raystab.h"

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

extern "C" {
	__constant__ Raystab::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	Vector3f ray_origin = params.ray_origins[idx.x];

	default_rng_t rng;
	rng.advance(idx.x * 2);
	Vector2f offset = random_val_2d(rng);

	static constexpr uint32_t N_STAB_RAYS = 32;
	for (uint32_t i = 0; i < N_STAB_RAYS; ++i) {
		// Use a Fibonacci lattice (with random offset) to regularly
		// distribute the stab rays over the sphere.
		Vector3f ray_direction = fibonacci_dir<N_STAB_RAYS>(i, offset);

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
			OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0,                   // SBT offset
			1,                   // SBT stride
			0,                   // missSBTIndex
			p0
		);

		if (p0 == 0) {
			// One ray escaped. We are outside. Distance doesn't need to be signed.
			return;
		}
	}

	params.distances[idx.x] = -params.distances[idx.x];
}

extern "C" __global__ void __miss__ms() {
	optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__ch() {
	optixSetPayload_0(1);
}

NGP_NAMESPACE_END
