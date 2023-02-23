/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   common_device.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/tinyexr_wrapper.h>

// #include <unsupported/Eigen/MatrixFunctions>

#include <stb_image/stb_image.h>

using namespace tcnn;

NGP_NAMESPACE_BEGIN


mat4x3 camera_lerp(const mat4x3& a, const mat4x3& b, float t) {
	// mat4 A = a;
	// mat4 B = b;
	// mat4 log_space_a_to_b = log(B * inverse(A));
	// return exp(log_space_a_to_b * t) * A;

	mat3 rot = slerp(a, b, t);
	return {rot[0], rot[1], rot[2], mix(a[3], b[3], t)};
}

GPUMemory<float> load_exr_gpu(const fs::path& path, int* width, int* height) {
	float* out; // width * height * RGBA
	load_exr(&out, width, height, path.str().c_str());
	ScopeGuard mem_guard{[&]() { free(out); }};

	GPUMemory<float> result((*width) * (*height) * 4);
	result.copy_from_host(out);
	return result;
}

GPUMemory<float> load_stbi_gpu(const fs::path& path, int* width, int* height) {
	bool is_hdr = is_hdr_stbi(path);

	void* data; // width * height * RGBA
	int comp;
	if (is_hdr) {
		data = load_stbi_float(path, width, height, &comp, 4);
	} else {
		data = load_stbi(path, width, height, &comp, 4);
	}

	if (!data) {
		throw std::runtime_error{std::string{stbi_failure_reason()}};
	}

	ScopeGuard mem_guard{[&]() { stbi_image_free(data); }};

	if (*width == 0 || *height == 0) {
		throw std::runtime_error{"Image has zero pixels."};
	}

	GPUMemory<float> result((*width) * (*height) * 4);
	if (is_hdr) {
		result.copy_from_host((float*)data);
	} else {
		GPUMemory<uint8_t> bytes((*width) * (*height) * 4);
		bytes.copy_from_host((uint8_t*)data);
		linear_kernel(from_rgba32<float>, 0, nullptr, (*width) * (*height), bytes.data(), result.data(), false, false, 0);
	}

	return result;
}

NGP_NAMESPACE_END
