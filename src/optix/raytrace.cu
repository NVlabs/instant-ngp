/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   raytrace.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Minimal optix program.
 */

#include <optix.h>

#include "raytrace.h"
#include <neural-graphics-primitives/common_device.cuh>

using namespace Eigen;

NGP_NAMESPACE_BEGIN

extern "C" {
	__constant__ Raytrace::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	Vector3f ray_origin = params.ray_origins[idx.x];
	Vector3f ray_direction = params.ray_directions[idx.x];

	unsigned int p0, p1;
	optixTrace(
		params.handle,
		to_float3(ray_origin),
		to_float3(ray_direction),
		0.0f,                // Min intersection distance
		1e16f,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,                   // SBT offset
		1,                   // SBT stride
		0,                   // missSBTIndex
		p0, p1
	);

	// Hit position
	float t = __int_as_float(p1);
	params.ray_origins[idx.x] = ray_origin + t * ray_direction;

	// If a triangle was hit, p0 is its index, otherwise p0 is -1.
	// Write out the triangle's normal if it (abuse the direction buffer).
	if ((int)p0 == -1) {
		return;
	}

	params.ray_directions[idx.x] = params.triangles[p0].normal();
}

extern "C" __global__ void __miss__ms() {
	optixSetPayload_0((uint32_t)-1);
	optixSetPayload_1(__float_as_int(optixGetRayTmax()));
}

extern "C" __global__ void __closesthit__ch() {
	optixSetPayload_0(optixGetPrimitiveIndex());
	optixSetPayload_1(__float_as_int(optixGetRayTmax()));
}

NGP_NAMESPACE_END
