/*
* Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/** @file   common.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Shared functionality among multiple neural-graphics-primitives components.
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/random_val.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>

NGP_NAMESPACE_BEGIN

using precision_t = tcnn::network_precision_t;

inline __host__ __device__ float srgb_to_linear(float srgb) {
	if (srgb <= 0.04045f) {
		return srgb / 12.92f;
	} else {
		return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
	}
}

inline __host__ __device__ Eigen::Array3f srgb_to_linear(const Eigen::Array3f& x) {
	return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}

inline __host__ __device__ float srgb_to_linear_derivative(float srgb) {
	if (srgb <= 0.04045f) {
		return 1.0f / 12.92f;
	} else {
		return 2.4f / 1.055f * std::pow((srgb + 0.055f) / 1.055f, 1.4f);
	}
}

inline __host__ __device__ Eigen::Array3f srgb_to_linear_derivative(const Eigen::Array3f& x) {
	return {srgb_to_linear_derivative(x.x()), srgb_to_linear_derivative(x.y()), (srgb_to_linear_derivative(x.z()))};
}

inline __host__ __device__ float linear_to_srgb(float linear) {
	if (linear < 0.0031308f) {
		return 12.92f * linear;
	} else {
		return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
	}
}

inline __host__ __device__ Eigen::Array3f linear_to_srgb(const Eigen::Array3f& x) {
	return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline __host__ __device__ float linear_to_srgb_derivative(float linear) {
	if (linear < 0.0031308f) {
		return 12.92f;
	} else {
		return 1.055f * 0.41666f * std::pow(linear, 0.41666f - 1.0f);
	}
}

inline __host__ __device__ Eigen::Array3f linear_to_srgb_derivative(const Eigen::Array3f& x) {
	return {linear_to_srgb_derivative(x.x()), linear_to_srgb_derivative(x.y()), (linear_to_srgb_derivative(x.z()))};
}

template <uint32_t N_DIMS, typename T>
__host__ __device__ Eigen::Matrix<float, N_DIMS, 1> read_image(const T* __restrict__ data, const Eigen::Vector2i& resolution, const Eigen::Vector2f& pos) {
	auto pos_float = Eigen::Vector2f{pos.x() * (float)(resolution.x()-1), pos.y() * (float)(resolution.y()-1)};
	Eigen::Vector2i texel = pos_float.cast<int>();

	auto weight = pos_float - texel.cast<float>();

	auto read_val = [&](Eigen::Vector2i pos) {
		pos.x() = std::max(std::min(pos.x(), resolution.x()-1), 0);
		pos.y() = std::max(std::min(pos.y(), resolution.y()-1), 0);

		Eigen::Matrix<float, N_DIMS, 1> result;
		if (std::is_same<T, float>::value) {
			result = *(Eigen::Matrix<T, N_DIMS, 1>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];
		} else {
			auto val = *(tcnn::vector_t<T, N_DIMS>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];

			NGP_PRAGMA_UNROLL
			for (uint32_t i = 0; i < N_DIMS; ++i) {
				result[i] = (float)val[i];
			}
		}
		return result;
	};

	auto result = (
		(1 - weight.x()) * (1 - weight.y()) * read_val({texel.x(), texel.y()}) +
		(weight.x()) * (1 - weight.y()) * read_val({texel.x()+1, texel.y()}) +
		(1 - weight.x()) * (weight.y()) * read_val({texel.x(), texel.y()+1}) +
		(weight.x()) * (weight.y()) * read_val({texel.x()+1, texel.y()+1})
	);

	return result;
}

template <uint32_t N_DIMS, typename T>
__device__ void deposit_image_gradient(const Eigen::Matrix<float, N_DIMS, 1>& value, T* __restrict__ gradient, T* __restrict__ gradient_weight, const Eigen::Vector2i& resolution, const Eigen::Vector2f& pos) {
	auto pos_float = Eigen::Vector2f{pos.x() * (resolution.x()-1), pos.y() * (resolution.y()-1)};
	Eigen::Vector2i texel = pos_float.cast<int>();

	auto weight = pos_float - texel.cast<float>();

	auto deposit_val = [&](const Eigen::Matrix<float, N_DIMS, 1>& value, T weight, Eigen::Vector2i pos) {
		pos.x() = std::max(std::min(pos.x(), resolution.x()-1), 0);
		pos.y() = std::max(std::min(pos.y(), resolution.y()-1), 0);

#if TCNN_MIN_GPU_ARCH >= 60 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (std::is_same<T, __half>::value) {
			for (uint32_t c = 0; c < N_DIMS; c += 2) {
				atomicAdd((__half2*)&gradient[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], {(T)value[c] * weight, (T)value[c+1] * weight});
				atomicAdd((__half2*)&gradient_weight[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], {weight, weight});
			}
		} else
#endif
		{
			for (uint32_t c = 0; c < N_DIMS; ++c) {
				atomicAdd(&gradient[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], (T)value[c] * weight);
				atomicAdd(&gradient_weight[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], weight);
			}
		}
	};

	deposit_val(value, (1 - weight.x()) * (1 - weight.y()), {texel.x(), texel.y()});
	deposit_val(value, (weight.x()) * (1 - weight.y()), {texel.x()+1, texel.y()});
	deposit_val(value, (1 - weight.x()) * (weight.y()), {texel.x(), texel.y()+1});
	deposit_val(value, (weight.x()) * (weight.y()), {texel.x()+1, texel.y()+1});
}

template <typename T>
__device__ __host__ inline void camera_distortion(const T* extra_params, const T u, const T v, T* du, T* dv) {
	const T k1 = extra_params[0];
	const T k2 = extra_params[1];
	const T p1 = extra_params[2];
	const T p2 = extra_params[3];

	const T u2 = u * u;
	const T uv = u * v;
	const T v2 = v * v;
	const T r2 = u2 + v2;
	const T radial = k1 * r2 + k2 * r2 * r2;
	*du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
	*dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

template <typename T>
__device__ __host__ inline void iterative_camera_undistortion(const T* params, T* u, T* v) {
	// Parameters for Newton iteration using numerical differentiation with
	// central differences, 100 iterations should be enough even for complex
	// camera models with higher order terms.
	const uint32_t kNumIterations = 100;
	const float kMaxStepNorm = 1e-10f;
	const float kRelStepSize = 1e-6f;

	Eigen::Matrix2f J;
	const Eigen::Vector2f x0(*u, *v);
	Eigen::Vector2f x(*u, *v);
	Eigen::Vector2f dx;
	Eigen::Vector2f dx_0b;
	Eigen::Vector2f dx_0f;
	Eigen::Vector2f dx_1b;
	Eigen::Vector2f dx_1f;

	for (uint32_t i = 0; i < kNumIterations; ++i) {
		const float step0 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(0)));
		const float step1 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(1)));
		camera_distortion(params, x(0), x(1), &dx(0), &dx(1));
		camera_distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
		camera_distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
		camera_distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
		camera_distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
		J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
		J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
		J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
		J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
		const Eigen::Vector2f step_x = J.inverse() * (x + dx - x0);
		x -= step_x;
		if (step_x.squaredNorm() < kMaxStepNorm) {
			break;
		}
	}

	*u = x(0);
	*v = x(1);
}

inline __host__ __device__ Ray pixel_to_ray_pinhole(
	uint32_t spp,
	const Eigen::Vector2i& pixel,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center,
	float focus_z = 1.0f,
	float dof = 0.0f
) {
	auto uv = pixel.cast<float>().cwiseQuotient(resolution.cast<float>());

	Eigen::Vector3f dir = {
		(uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
		(uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y(),
		1.0f
	};

	dir = camera_matrix.block<3, 3>(0, 0) * dir;

	Eigen::Vector3f origin = camera_matrix.col(3);
	return {origin, dir};
}

inline __host__ __device__ Eigen::Matrix<float, 3, 4> get_xform_given_rolling_shutter(const TrainingXForm &training_xform, const Eigen::Vector4f &rolling_shutter, const Eigen::Vector2f &uv, float motionblur_time) {
	float pixel_t = rolling_shutter.x() + rolling_shutter.y() * uv.x() + rolling_shutter.z() * uv.y() + rolling_shutter.w() * motionblur_time;
	return training_xform.start + (training_xform.end - training_xform.start) * pixel_t;
}

inline __host__ __device__ Eigen::Vector3f f_theta_undistortion(const Eigen::Vector2f &uv, const Eigen::Vector2f &screen_center, const CameraDistortion& camera_distortion, const Eigen::Vector3f& error_direction) {
	// we take f_theta intrinsics to be: resx, resy, r0, r1, r2, r3; we rescale to whatever res the intrinsics specify.
	float xpix = (uv.x() - screen_center.x()) * camera_distortion.params[5];
	float ypix = (uv.y() - screen_center.y()) * camera_distortion.params[6];
	float norm = sqrtf(xpix*xpix + ypix*ypix);
	float alpha = camera_distortion.params[0] + norm * (camera_distortion.params[1] + norm * (camera_distortion.params[2] + norm * (camera_distortion.params[3] + norm * camera_distortion.params[4])));
	float sin_alpha, cos_alpha;
	sincosf(alpha, &sin_alpha, &cos_alpha);
	if (cos_alpha <= std::numeric_limits<float>::min() || norm == 0.f)
		return error_direction;
	sin_alpha *= 1.f/norm;
	return { sin_alpha * xpix, sin_alpha * ypix, cos_alpha };
}

inline __host__ __device__ Ray pixel_to_ray(
	uint32_t spp,
	const Eigen::Vector2i& pixel,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center,
	bool snap_to_pixel_centers = false,
	float focus_z = 1.0f,
	float dof = 0.0f,
	const CameraDistortion& camera_distortion = {},
	const float* __restrict__ distortion_data = nullptr,
	const Eigen::Vector2i distortion_resolution = Eigen::Vector2i::Zero()
) {
	Eigen::Vector2f offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : spp, pixel.x(), pixel.y());
	auto uv = (pixel.cast<float>() + offset).cwiseQuotient(resolution.cast<float>());

	Eigen::Vector3f dir;
	if (camera_distortion.mode == ECameraDistortionMode::FTheta) {
		dir = f_theta_undistortion(uv, screen_center, camera_distortion, {1000.f, 0.f, 0.f});
		if (dir.x() == 1000.f) {
			return {{1000.f, 0.f, 0.f}, {0.f, 0.f, 1.f}}; // return a point outside the aabb so the pixel is not rendered
		}
	} else {
		dir = {
			(uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
			(uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y(),
			1.0f
		};
		if (camera_distortion.mode == ECameraDistortionMode::Iterative) {
			iterative_camera_undistortion(camera_distortion.params, &dir.x(), &dir.y());
		}
	}
	if (distortion_data) {
		dir.head<2>() += read_image<2>(distortion_data, distortion_resolution, uv);
	}

	dir = camera_matrix.block<3, 3>(0, 0) * dir;

	Eigen::Vector3f origin = camera_matrix.col(3);
	if (dof == 0.0f) {
		return {origin, dir};
	}

	Eigen::Vector3f lookat = origin + dir * focus_z;
	Eigen::Vector2f blur = dof * square2disk_shirley(ld_random_val_2d(spp, (uint32_t)pixel.x() * 19349663 + (uint32_t)pixel.y() * 96925573) * 2.0f - Eigen::Vector2f::Ones());
	origin += camera_matrix.block<3, 2>(0, 0) * blur;
	dir = (lookat - origin) / focus_z;

	return {origin, dir};
}

inline __host__ __device__ float fov_to_focal_length(int resolution, float degrees) {
	return 0.5f * (float)resolution / tanf(0.5f * degrees*(float)PI()/180);
}

inline __host__ __device__ Eigen::Vector2f fov_to_focal_length(const Eigen::Vector2i& resolution, const Eigen::Vector2f& degrees) {
	return 0.5f * resolution.cast<float>().cwiseQuotient((0.5f * degrees * (float)PI()/180).array().tan().matrix());
}

inline __host__ __device__ float focal_length_to_fov(int resolution, float focal_length) {
	return 2.f * 180.f / PI() * atanf(float(resolution)/(focal_length*2.f));
}

inline __host__ __device__ Eigen::Vector2f focal_length_to_fov(const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length) {
	return 2.f * 180.f / PI() * resolution.cast<float>().cwiseQuotient(focal_length*2).array().atan().matrix();
}

inline __host__ __device__ float4 to_float4(const Eigen::Array4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline __host__ __device__ float4 to_float4(const Eigen::Vector4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline __host__ __device__ float3 to_float3(const Eigen::Array3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline __host__ __device__ float3 to_float3(const Eigen::Vector3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline __host__ __device__ float2 to_float2(const Eigen::Array2f& x) {
	return {x.x(), x.y()};
}

inline __host__ __device__ float2 to_float2(const Eigen::Vector2f& x) {
	return {x.x(), x.y()};
}

inline __host__ __device__ Eigen::Array4f to_array4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline __host__ __device__ Eigen::Vector4f to_vec4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline __host__ __device__ Eigen::Array3f to_array3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline __host__ __device__ Eigen::Vector3f to_vec3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline __host__ __device__ Eigen::Array2f to_array2(const float2& x) {
	return {x.x, x.y};
}

inline __host__ __device__ Eigen::Vector2f to_vec2(const float2& x) {
	return {x.x, x.y};
}

inline __host__ __device__ Eigen::Vector3f faceforward(const Eigen::Vector3f& n, const Eigen::Vector3f& i, const Eigen::Vector3f& nref) {
	return n * copysignf(1.0f, i.dot(nref));
}

template <typename T>
__global__ void from_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, T* __restrict__ out, bool white_2_transparent = false, bool black_2_transparent = false, uint32_t mask_color = 0) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	uint8_t rgba[4];
	*((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

	float alpha = rgba[3] * (1.0f/255.0f);
	// NSVF dataset has 'white = transparent' madness
	if (white_2_transparent && rgba[0]==255 && rgba[1]==255 && rgba[2]==255) {
		alpha=0.f;
	}
	if (black_2_transparent && rgba[0]==0 && rgba[1]==0 && rgba[2]==0) {
		alpha=0.f;
	}

	tcnn::vector_t<T, 4> rgba_out;
	rgba_out[0] = (T)(srgb_to_linear(rgba[0] * (1.0f/255.0f)) * alpha);
	rgba_out[1] = (T)(srgb_to_linear(rgba[1] * (1.0f/255.0f)) * alpha);
	rgba_out[2] = (T)(srgb_to_linear(rgba[2] * (1.0f/255.0f)) * alpha);
	rgba_out[3] = (T)alpha;

	if (mask_color != 0 && mask_color == *((uint32_t*)&rgba[0])) {
		rgba_out[0] = rgba_out[1] = rgba_out[2] = rgba_out[3] = (T)-1.0f;
	}

	*((tcnn::vector_t<T, 4>*)&out[i*4]) = rgba_out;
}

Eigen::Matrix<float, 3, 4> log_space_lerp(const Eigen::Matrix<float, 3, 4>& begin, const Eigen::Matrix<float, 3, 4>& end, float t);

tcnn::GPUMemory<float> load_exr(const std::string& filename, int& width, int& height);
tcnn::GPUMemory<float> load_stbi(const std::string& filename, int& width, int& height);

NGP_NAMESPACE_END
