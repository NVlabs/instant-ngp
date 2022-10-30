/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/gpu_memory.h>

NGP_NAMESPACE_BEGIN

// size of the density/occupancy grid in number of cells along an axis.
inline constexpr __device__ uint32_t NERF_GRIDSIZE() {
	return 128;
}

struct NerfPayload {
	Eigen::Vector3f origin;
	Eigen::Vector3f dir;
	float t;
	float max_weight;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct RaysNerfSoa {
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(const RaysNerfSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgba, other.rgba, size * sizeof(Eigen::Array4f), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.depth, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, size * sizeof(NerfPayload), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(Eigen::Array4f* rgba, float* depth, NerfPayload* payload, size_t size) {
		this->rgba = rgba;
		this->depth = depth;
		this->payload = payload;
		this->size = size;
	}

	Eigen::Array4f* rgba;
	float* depth;
	NerfPayload* payload;
	size_t size;
};

//#define TRIPLANAR_COMPATIBLE_POSITIONS   // if this is defined, then positions are stored as [x,y,z,x] so that it can be split as [x,y] [y,z] [z,x] by the input encoding

struct NerfPosition {
	NGP_HOST_DEVICE NerfPosition(const Eigen::Vector3f& pos, float dt)
	:
	p{pos}
#ifdef TRIPLANAR_COMPATIBLE_POSITIONS
	, x{pos.x()}
#endif
	{}
	Eigen::Vector3f p;
#ifdef TRIPLANAR_COMPATIBLE_POSITIONS
	float x;
#endif
};

struct NerfDirection {
	NGP_HOST_DEVICE NerfDirection(const Eigen::Vector3f& dir, float dt) : d{dir} {}
	Eigen::Vector3f d;
};

struct NerfCoordinate {
	NGP_HOST_DEVICE NerfCoordinate(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}
	NGP_HOST_DEVICE void set_with_optional_extra_dims(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir, float dt, const float* extra_dims, uint32_t stride_in_bytes) {
		this->dt = dt;
		this->pos = NerfPosition(pos, dt);
		this->dir = NerfDirection(dir, dt);
		copy_extra_dims(extra_dims, stride_in_bytes);
	}
	inline NGP_HOST_DEVICE const float* get_extra_dims() const { return (const float*)(this + 1); }
	inline NGP_HOST_DEVICE float* get_extra_dims() { return (float*)(this + 1); }

	NGP_HOST_DEVICE void copy(const NerfCoordinate& inp, uint32_t stride_in_bytes) {
		*this = inp;
		copy_extra_dims(inp.get_extra_dims(), stride_in_bytes);
	}
	NGP_HOST_DEVICE inline void copy_extra_dims(const float *extra_dims, uint32_t stride_in_bytes) {
		if (stride_in_bytes >= sizeof(NerfCoordinate)) {
			float* dst = get_extra_dims();
			const uint32_t n_extra = (stride_in_bytes - sizeof(NerfCoordinate)) / sizeof(float);
			for (uint32_t i = 0; i < n_extra; ++i) dst[i] = extra_dims[i];
		}
	}

	NerfPosition pos;
	float dt;
	NerfDirection dir;
};

NGP_NAMESPACE_END
