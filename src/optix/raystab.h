/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   raystab.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Minimal optix program.
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <optix.h>

NGP_NAMESPACE_BEGIN

struct Raystab {
	struct Params {
		const Eigen::Vector3f* ray_origins;
		float* distances;
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};

NGP_NAMESPACE_END
