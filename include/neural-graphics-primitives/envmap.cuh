/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   envmap.cuh
 *  @author Thomas Müller, NVIDIA
 *  @brief  An implementation of kernels for evaluating an envmap and splatting gradients into it.
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

NGP_NAMESPACE_BEGIN

template <typename T>
__device__ Eigen::Array4f read_envmap(const T* __restrict__ envmap_data, const Eigen::Vector2i envmap_resolution, const Eigen::Vector3f& dir) {
	auto dir_cyl = dir_to_spherical_unorm({dir.z(), -dir.x(), dir.y()});

	auto envmap_float = Eigen::Vector2f{dir_cyl.y() * (envmap_resolution.x()-1), dir_cyl.x() * (envmap_resolution.y()-1)};
	Eigen::Vector2i envmap_texel = envmap_float.cast<int>();

	auto weight = envmap_float - envmap_texel.cast<float>();

	auto read_val = [&](Eigen::Vector2i pos) {
		if (pos.x() < 0) {
			pos.x() += envmap_resolution.x();
		} else if (pos.x() >= envmap_resolution.x()) {
			pos.x() -= envmap_resolution.x();
		}
		pos.y() = std::max(std::min(pos.y(), envmap_resolution.y()-1), 0);

		Eigen::Array4f result;
		if (std::is_same<T, float>::value) {
			result = *(Eigen::Array4f*)&envmap_data[(pos.x() + pos.y() * envmap_resolution.x()) * 4];
		} else {
			auto val = *(tcnn::vector_t<T, 4>*)&envmap_data[(pos.x() + pos.y() * envmap_resolution.x()) * 4];
			result = {(float)val[0], (float)val[1], (float)val[2], (float)val[3]};
		}
		return result;
	};

	auto result = (
		(1 - weight.x()) * (1 - weight.y()) * read_val({envmap_texel.x(), envmap_texel.y()}) +
		(weight.x()) * (1 - weight.y()) * read_val({envmap_texel.x()+1, envmap_texel.y()}) +
		(1 - weight.x()) * (weight.y()) * read_val({envmap_texel.x(), envmap_texel.y()+1}) +
		(weight.x()) * (weight.y()) * read_val({envmap_texel.x()+1, envmap_texel.y()+1})
	);

	return result;
}

template <typename T, typename GRAD_T>
__device__ void deposit_envmap_gradient(const tcnn::vector_t<T, 4>& value, GRAD_T* __restrict__ envmap_gradient, const Eigen::Vector2i envmap_resolution, const Eigen::Vector3f& dir) {
	auto dir_cyl = dir_to_spherical_unorm({dir.z(), -dir.x(), dir.y()});

	auto envmap_float = Eigen::Vector2f{dir_cyl.y() * (envmap_resolution.x()-1), dir_cyl.x() * (envmap_resolution.y()-1)};
	Eigen::Vector2i envmap_texel = envmap_float.cast<int>();

	auto weight = envmap_float - envmap_texel.cast<float>();

	auto deposit_val = [&](const tcnn::vector_t<T, 4>& value, T weight, Eigen::Vector2i pos) {
		if (pos.x() < 0) {
			pos.x() += envmap_resolution.x();
		} else if (pos.x() >= envmap_resolution.x()) {
			pos.x() -= envmap_resolution.x();
		}
		pos.y() = std::max(std::min(pos.y(), envmap_resolution.y()-1), 0);

		Eigen::Array4f result;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (std::is_same<GRAD_T, __half>::value) {
			for (uint32_t c = 0; c < 4; c += 2) {
				atomicAdd((__half2*)&envmap_gradient[(pos.x() + pos.y() * envmap_resolution.x()) * 4 + c], {value[c] * weight, value[c+1] * weight});
			}
		} else
#endif
		{
			for (uint32_t c = 0; c < 4; ++c) {
				atomicAdd(&envmap_gradient[(pos.x() + pos.y() * envmap_resolution.x()) * 4 + c], (GRAD_T)(value[c] * weight));
			}
		}
	};

	deposit_val(value, (1 - weight.x()) * (1 - weight.y()), {envmap_texel.x(), envmap_texel.y()});
	deposit_val(value, (weight.x()) * (1 - weight.y()), {envmap_texel.x()+1, envmap_texel.y()});
	deposit_val(value, (1 - weight.x()) * (weight.y()), {envmap_texel.x(), envmap_texel.y()+1});
	deposit_val(value, (weight.x()) * (weight.y()), {envmap_texel.x()+1, envmap_texel.y()+1});
}

NGP_NAMESPACE_END
