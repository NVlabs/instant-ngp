/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   sdf_device.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common_device.h>

namespace ngp {

struct SdfPayload {
	vec3 dir;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct BRDFParams {
	float metallic = 0.f;
	float subsurface = 0.f;
	float specular = 1.f;
	float roughness = 0.5f;
	float sheen = 0.f;
	float clearcoat = 0.f;
	float clearcoat_gloss = 0.f;
	vec3 basecolor = {0.8f, 0.8f, 0.8f};
	vec3 ambientcolor = {0.0f, 0.0f, 0.0f};
};

} // namespace ngp
