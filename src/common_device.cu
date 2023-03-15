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


mat3 so3_log(const mat3& m) {
	float tr = tcnn::clamp(m[0][0] + m[1][1] + m[2][2], -0.99999f, 1.0f);
	float radians = acosf((tr - 1.0f) / 2.0f);
	return radians / sqrt((1.0f + tr) * (3.0f - tr)) * (m - transpose(m));
}

mat3 so3_exp(const mat3& m) {
	vec3 axis = {-m[2][1], m[2][0], -m[1][0]};
	float radians_sq = length2(axis);
	if (radians_sq == 0.0f) {
		return mat3(1.0f);
	}

	float radians = sqrt(radians_sq);
	return mat3(1.0f) + (sin(radians) / radians) * m + ((1.0f - cos(radians)) / radians_sq) * (m * m);
}

mat4x3 se3_log(const mat4x3& m) {
	mat3 omega = so3_log(mat3(m));
	vec3 axis = {-omega[2][1], omega[2][0], -omega[1][0]};
	float radians_sq = length2(axis);
	mat3 inv_trans = mat3(1.0f);
	if (radians_sq > 0.0f) {
		float radians = sqrt(radians_sq);
		inv_trans += -0.5f * omega + ((1.0f - 0.5f * radians * cos(0.5f * radians) / sin(0.5f * radians)) / radians_sq) * (omega * omega);
	}

	return mat4x3(omega[0], omega[1], omega[2], inv_trans * m[3]);
}

mat4x3 se3_exp(const mat4x3& m) {
	mat3 omega{m};
	vec3 axis = {-omega[2][1], omega[2][0], -omega[1][0]};
	float radians_sq = length2(axis);
	mat3 trans = mat3(1.0f);
	if (radians_sq > 0.0f) {
		float radians = sqrt(radians_sq);
		trans += ((1.0f - cos(radians)) / radians_sq) * omega + ((radians - sin(radians)) / (radians * radians_sq)) * (omega * omega);
	}

	mat3 rot = so3_exp(omega);
	return mat4x3(rot[0], rot[1], rot[2], trans * m[3]);
}

mat4 se3_log(const mat4& m) {
	mat4 result = mat4(se3_log(mat4x3(m)));
	result[3][3] = 0.0f;
	return result;
}

mat4 se3_exp(const mat4& m) {
	return mat4(se3_exp(mat4x3(m)));
}

float frobenius_norm(const mat4& m) {
	return sqrt(length2(m[0]) + length2(m[1]) + length2(m[2]) + length2(m[3]));
}

mat4 mat_sqrt(const mat4& m, float eps = 1e-10f) {
	mat4 X = m, Y = mat4(1.0f);
	for (uint32_t i = 0; i < 32; ++i) {
		if (frobenius_norm(X * X - m) < eps) {
			return X;
		}

		mat4 iX = inverse(X);
		X = 0.5f * (X + inverse(Y));
		Y = 0.5f * (Y + iX);
	}

	return X;
}

mat4 mat_log_taylor(const mat4& m, uint32_t n_iters) {
	mat4 result = mat4(0.0f);
	mat4 cur = m - mat4(1.0f);
	float sign = 1.0f;
	for (uint32_t i = 1; i < n_iters; ++i) {
		result += (sign / (float)i) * cur;
		cur *= (m - mat4(1.0f));
		sign = -sign;
	}
	return result;
}

mat4 mat_log_hawkins(const mat4& m, float eps = 1e-10f) {
	mat4 A = m - mat4(1.0f), Z = A, X = A;
	for (uint32_t i = 2; i < 32; ++i) {
		if (frobenius_norm(Z) < eps) {
			return X;
		}

		Z = Z * A;
		X += (1.0f / (float)i) * Z;
	}

	return X;
}

mat4 mat_exp_power(const mat4& m, uint32_t n_iters) {
	mat4 result = mat4(1.0f);
	mat4 cur = m;
	float div = 1.0f;
	for (uint32_t i = 1; i < n_iters; ++i) {
		div *= (float)i;
		result += (1.0f / div) * cur;
		cur *= m;
	}
	return result;
}

mat4 mat_exp_pade(const mat4& m) {
	// Pade approximation with scaling; same as Matlab.
	// Pseudocode translated from Hawkins and Grimm [2007]
	mat4 X = mat4(1.0f), D = mat4(1.0f), N = mat4(1.0f);
	float c = 1.0f;
	constexpr uint32_t q = 6; // Matlab's default when using this algorithm

	float s = -1.0f;
	for (uint32_t k = 1; k <= q; ++k) {
		c = c * (q - k + 1) / (k * (2 * q - k + 1));
		X = m * X;
		auto cX = c * X;
		N = N + cX;
		D = D + s * cX;
		s = -s;
	}

	return inverse(D) * N;
}

mat4 mat_log(const mat4& m) {
	mat4 result(m);

	uint32_t j = 0;
	for (; j < 32; ++j) {
		if (frobenius_norm(result - mat4(1.0f)) < 1e-5f) {
			break;
		}

		result = mat_sqrt(result);
	}

	result = mat_log_hawkins(result);
	return scalbnf(1.0f, j) * result;
}

mat4 mat_exp(const mat4& m) {
	uint32_t N_SQUARING = max(0, 1 + (int)floor(log2(frobenius_norm(m))));

	mat4 result = scalbnf(1.0f, -N_SQUARING) * m;
	result = mat_exp_pade(result);

	for (uint32_t i = 0; i < N_SQUARING; ++i) {
		result *= result;
	}

	return result;
}

mat3 orthogonalize(const mat3& m) {
	return mat3{
		0.5f * (3.0f - dot(m[0], m[0])) * m[0],
		0.5f * (3.0f - dot(m[1], m[1])) * m[1],
		0.5f * (3.0f - dot(m[2], m[2])) * m[2],
	};
}

mat4x3 camera_log_lerp(const mat4x3& a, const mat4x3& b, float t) {
	return mat_exp(mat_log(mat4(b) * inverse(mat4(a))) * t) * mat4(a);
}

mat4x3 camera_slerp(const mat4x3& a, const mat4x3& b, float t) {
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
