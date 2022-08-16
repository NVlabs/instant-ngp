/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   pathescape.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Minimal optix program.
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/triangle.cuh>

#include <optix.h>

NGP_NAMESPACE_BEGIN

struct PathEscape {
	struct Params {
		const Eigen::Vector3f* ray_origins;
		const Triangle* triangles;
		float* distances;
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};

NGP_NAMESPACE_END
