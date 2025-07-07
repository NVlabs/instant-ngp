/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_device.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common_device.h>

namespace ngp {

// size of the density/occupancy grid in number of cells along an axis.
inline constexpr NGP_HOST_DEVICE uint32_t NERF_GRIDSIZE() { return 128; }
inline constexpr NGP_HOST_DEVICE uint32_t NERF_GRID_N_CELLS() { return NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE(); }

inline constexpr NGP_HOST_DEVICE float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr NGP_HOST_DEVICE uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
inline constexpr NGP_HOST_DEVICE uint32_t NERF_CASCADES() { return 8; }

inline constexpr NGP_HOST_DEVICE float SQRT3() { return 1.73205080757f; }
inline constexpr NGP_HOST_DEVICE float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
inline constexpr NGP_HOST_DEVICE float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr NGP_HOST_DEVICE float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr NGP_HOST_DEVICE uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 16; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr NGP_HOST_DEVICE float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

struct TrainingImageMetadata {
	// Camera intrinsics and additional data associated with a NeRF training image
	// the memory to back the pixels and rays is held by GPUMemory objects in the NerfDataset and copied here.
	const void* pixels = nullptr;
	EImageDataType image_data_type = EImageDataType::Half;

	const float* depth = nullptr;
	const Ray* rays = nullptr;

	Lens lens = {};
	ivec2 resolution = ivec2(0);
	vec2 principal_point = vec2(0.5f);
	vec2 focal_length = vec2(1000.f);
	vec4 rolling_shutter = vec4(0.0f);
	vec3 light_dir = vec3(0.f); // TODO: replace this with more generic float[] of task-specific metadata.
};

struct LossAndGradient {
	vec3 loss;
	vec3 gradient;

	NGP_HOST_DEVICE LossAndGradient operator*(float scalar) {
		return {loss * scalar, gradient * scalar};
	}

	NGP_HOST_DEVICE LossAndGradient operator/(float scalar) {
		return {loss / scalar, gradient / scalar};
	}
};

inline NGP_HOST_DEVICE LossAndGradient l2_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	return {
		difference * difference,
		2.0f * difference
	};
}

inline NGP_HOST_DEVICE LossAndGradient relative_l2_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 denom = prediction * prediction + 1e-2f;
	return {
		difference * difference / denom,
		2.0f * difference / denom
	};
}

inline NGP_HOST_DEVICE LossAndGradient l1_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	return {
		abs(difference),
		copysign(vec3(1.0f), difference),
	};
}

inline NGP_HOST_DEVICE LossAndGradient huber_loss(const vec3& target, const vec3& prediction, float alpha = 1) {
	vec3 difference = prediction - target;
	vec3 abs_diff = abs(difference);
	vec3 square = 0.5f/alpha * difference * difference;
	return {
		{
			abs_diff.x > alpha ? (abs_diff.x - 0.5f * alpha) : square.x,
			abs_diff.y > alpha ? (abs_diff.y - 0.5f * alpha) : square.y,
			abs_diff.z > alpha ? (abs_diff.z - 0.5f * alpha) : square.z,
		},
		{
			abs_diff.x > alpha ? (difference.x > 0 ? 1.0f : -1.0f) : (difference.x / alpha),
			abs_diff.y > alpha ? (difference.y > 0 ? 1.0f : -1.0f) : (difference.y / alpha),
			abs_diff.z > alpha ? (difference.z > 0 ? 1.0f : -1.0f) : (difference.z / alpha),
		},
	};
}

inline NGP_HOST_DEVICE LossAndGradient log_l1_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 divisor = abs(difference) + 1.0f;
	return {
		log(divisor),
		copysign(vec3(1.0f) / divisor, difference),
	};
}

inline NGP_HOST_DEVICE LossAndGradient smape_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 denom = 0.5f * (abs(prediction) + abs(target)) + 1e-2f;
	return {
		abs(difference) / denom,
		copysign(vec3(1.0f) / denom, difference),
	};
}

inline NGP_HOST_DEVICE LossAndGradient mape_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 denom = abs(prediction) + 1e-2f;
	return {
		abs(difference) / denom,
		copysign(vec3(1.0f) / denom, difference),
	};
}

struct NerfPayload {
	vec3 origin;
	vec3 dir;
	float t;
	float max_weight;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

//#define TRIPLANAR_COMPATIBLE_POSITIONS   // if this is defined, then positions are stored as [x,y,z,x] so that it can be split as [x,y] [y,z] [z,x] by the input encoding

struct NerfPosition {
	NGP_HOST_DEVICE NerfPosition(const vec3& pos, float dt)
	:
	p{pos}
#ifdef TRIPLANAR_COMPATIBLE_POSITIONS
	, x{pos.x}
#endif
	{}
	vec3 p;
#ifdef TRIPLANAR_COMPATIBLE_POSITIONS
	float x;
#endif
};

struct NerfDirection {
	NGP_HOST_DEVICE NerfDirection(const vec3& dir, float dt) : d{dir} {}
	vec3 d;
};

struct NerfCoordinate {
	NGP_HOST_DEVICE NerfCoordinate(const vec3& pos, const vec3& dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}
	NGP_HOST_DEVICE void set_with_optional_extra_dims(const vec3& pos, const vec3& dir, float dt, const float* extra_dims, uint32_t stride_in_bytes) {
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

inline NGP_HOST_DEVICE float network_to_rgb(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return logistic(val);
		case ENerfActivation::Exponential: return expf(clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

inline NGP_HOST_DEVICE float network_to_rgb_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return expf(clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

template <typename T>
NGP_HOST_DEVICE vec3 network_to_rgb_derivative_vec(const T& val, ENerfActivation activation) {
	return {
		network_to_rgb_derivative(float(val[0]), activation),
		network_to_rgb_derivative(float(val[1]), activation),
		network_to_rgb_derivative(float(val[2]), activation),
	};
}

inline NGP_HOST_DEVICE float network_to_density(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return logistic(val);
		case ENerfActivation::Exponential: return expf(val);
		default: assert(false);
	}
	return 0.0f;
}

inline NGP_HOST_DEVICE float network_to_density_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return expf(clamp(val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

template <typename T>
NGP_HOST_DEVICE vec3 network_to_rgb_vec(const T& val, ENerfActivation activation) {
	return {
		network_to_rgb(float(val[0]), activation),
		network_to_rgb(float(val[1]), activation),
		network_to_rgb(float(val[2]), activation),
	};
}

inline NGP_HOST_DEVICE vec3 warp_position(const vec3& pos, const BoundingBox& aabb) {
	// return {logistic(pos.x - 0.5f), logistic(pos.y - 0.5f), logistic(pos.z - 0.5f)};
	// return pos;

	return aabb.relative_pos(pos);
}

inline NGP_HOST_DEVICE vec3 unwarp_position(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
	// return pos;

	return aabb.min + pos * aabb.diag();
}

inline NGP_HOST_DEVICE vec3 unwarp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
	// return pos;

	return aabb.diag();
}

inline NGP_HOST_DEVICE vec3 warp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
	return vec3(1.0f) / unwarp_position_derivative(pos, aabb);
}

inline NGP_HOST_DEVICE vec3 warp_direction(const vec3& dir) {
	return (dir + 1.0f) * 0.5f;
}

inline NGP_HOST_DEVICE vec3 unwarp_direction(const vec3& dir) {
	return dir * 2.0f - 1.0f;
}

inline NGP_HOST_DEVICE vec3 warp_direction_derivative(const vec3& dir) {
	return vec3(0.5f);
}

inline NGP_HOST_DEVICE vec3 unwarp_direction_derivative(const vec3& dir) {
	return vec3(2.0f);
}

inline NGP_HOST_DEVICE float warp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

inline NGP_HOST_DEVICE float unwarp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

inline NGP_HOST_DEVICE uint32_t cascaded_grid_idx_at(vec3 pos, uint32_t mip) {
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= vec3(0.5f);
	pos *= mip_scale;
	pos += vec3(0.5f);

	ivec3 i = pos * (float)NERF_GRIDSIZE();
	if (i.x < 0 || i.x >= NERF_GRIDSIZE() || i.y < 0 || i.y >= NERF_GRIDSIZE() || i.z < 0 || i.z >= NERF_GRIDSIZE()) {
		return 0xFFFFFFFF;
	}

	return morton3D(i.x, i.y, i.z);
}

inline NGP_HOST_DEVICE uint32_t grid_mip_offset(uint32_t mip) {
	return NERF_GRID_N_CELLS() * mip;
}

inline NGP_HOST_DEVICE bool density_grid_occupied_at(const vec3& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return false;
	}
	return density_grid_bitfield[idx/8+grid_mip_offset(mip)/8] & (1<<(idx%8));
}

inline NGP_HOST_DEVICE float cascaded_grid_at(vec3 pos, const float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return 0.0f;
	}
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

inline NGP_HOST_DEVICE float& cascaded_grid_at(vec3 pos, float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		idx = 0;
		printf("WARNING: invalid cascaded grid access.");
	}
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

inline NGP_HOST_DEVICE float distance_to_next_voxel(const vec3& pos, const vec3& dir, const vec3& idir, float res) { // dda like step
	vec3 p = res * (pos - 0.5f);
	float tx = (floorf(p.x + 0.5f + 0.5f * sign(dir.x)) - p.x) * idir.x;
	float ty = (floorf(p.y + 0.5f + 0.5f * sign(dir.y)) - p.y) * idir.y;
	float tz = (floorf(p.z + 0.5f + 0.5f * sign(dir.z)) - p.z) * idir.z;
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline NGP_HOST_DEVICE float calc_cone_angle(float cosine, const vec2& focal_length, float cone_angle_constant) {
	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();

	return cone_angle_constant;
}

inline NGP_HOST_DEVICE float to_stepping_space(float t, float cone_angle) {
	if (cone_angle <= 1e-5f) {
		return t / MIN_CONE_STEPSIZE();
	}

	float log1p_c = logf(1.0f + cone_angle);

	float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
	float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

	float at = expf(a * log1p_c);
	float bt = expf(b * log1p_c);

	if (t <= at) {
		return (t - at) / MIN_CONE_STEPSIZE() + a;
	} else if (t <= bt) {
		return logf(t) / log1p_c;
	} else {
		return (t - bt) / MAX_CONE_STEPSIZE() + b;
	}
}

inline NGP_HOST_DEVICE float from_stepping_space(float n, float cone_angle) {
	if (cone_angle <= 1e-5f) {
		return n * MIN_CONE_STEPSIZE();
	}

	float log1p_c = logf(1.0f + cone_angle);

	float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
	float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

	float at = expf(a * log1p_c);
	float bt = expf(b * log1p_c);

	if (n <= a) {
		return (n - a) * MIN_CONE_STEPSIZE() + at;
	} else if (n <= b) {
		return expf(n * log1p_c);
	} else {
		return (n - b) * MAX_CONE_STEPSIZE() + bt;
	}
}

inline NGP_HOST_DEVICE float advance_n_steps(float t, float cone_angle, float n) {
	return from_stepping_space(to_stepping_space(t, cone_angle) + n, cone_angle);
}

inline NGP_HOST_DEVICE float calc_dt(float t, float cone_angle) {
	return advance_n_steps(t, cone_angle, 1.0f) - t;
}

inline NGP_HOST_DEVICE float advance_to_next_voxel(float t, float cone_angle, const vec3& pos, const vec3& dir, const vec3& idir, uint32_t mip) {
	float res = scalbnf(NERF_GRIDSIZE(), -(int)mip);

	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);

	// Analytic stepping in multiples of 1 in the "log-space" of our exponential stepping routine
	t = to_stepping_space(t, cone_angle);
	t_target = to_stepping_space(t_target, cone_angle);

	return from_stepping_space(t + ceilf(fmaxf(t_target - t, 0.5f)), cone_angle);
}

inline NGP_HOST_DEVICE uint32_t mip_from_pos(const vec3& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	int exponent;
	float maxval = max(abs(pos - 0.5f));
	frexpf(maxval, &exponent);
	return (uint32_t)clamp(exponent+1, 0, (int)max_cascade);
}

inline NGP_HOST_DEVICE uint32_t mip_from_dt(float dt, const vec3& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	uint32_t mip = mip_from_pos(pos, max_cascade);
	dt *= 2 * NERF_GRIDSIZE();
	if (dt < 1.0f) {
		return mip;
	}

	int exponent;
	frexpf(dt, &exponent);
	return (uint32_t)clamp((int)mip, exponent, (int)max_cascade);
}

template <bool MIP_FROM_DT=false>
NGP_HOST_DEVICE float if_unoccupied_advance_to_next_occupied_voxel(
	float t,
	float cone_angle,
	const Ray& ray,
	const vec3& idir,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	BoundingBox aabb,
	mat3 aabb_to_local = mat3::identity()
) {
	while (true) {
		vec3 pos = ray(t);
		if (t >= MAX_DEPTH() || !aabb.contains(aabb_to_local * pos)) {
			return MAX_DEPTH();
		}

		uint32_t mip = clamp(MIP_FROM_DT ? mip_from_dt(calc_dt(t, cone_angle), pos) : mip_from_pos(pos), min_mip, max_mip);

		if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
			return t;
		}

		// Find largest empty voxel surrounding us, such that we can advance as far as possible in the next step.
		// Other places that do voxel stepping don't need this, because they don't rely on thread coherence as
		// much as this one here.
		while (mip < max_mip && !density_grid_occupied_at(pos, density_grid, mip+1)) {
			++mip;
		}

		t = advance_to_next_voxel(t, cone_angle, pos, ray.d, idir, mip);
	}
}

static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;

inline NGP_HOST_DEVICE vec2 sample_cdf_2d(vec2 sample, uint32_t img, const ivec2& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, float* __restrict__ pdf) {
	if (sample.x < UNIFORM_SAMPLING_FRACTION) {
		sample.x /= UNIFORM_SAMPLING_FRACTION;
		return sample;
	}

	sample.x = (sample.x - UNIFORM_SAMPLING_FRACTION) / (1.0f - UNIFORM_SAMPLING_FRACTION);

	cdf_y += img * res.y;

	// First select row according to cdf_y
	uint32_t y = binary_search(sample.y, cdf_y, res.y);
	float prev = y > 0 ? cdf_y[y-1] : 0.0f;
	float pmf_y = cdf_y[y] - prev;
	sample.y = (sample.y - prev) / pmf_y;

	cdf_x_cond_y += img * res.y * res.x + y * res.x;

	// Then, select col according to x
	uint32_t x = binary_search(sample.x, cdf_x_cond_y, res.x);
	prev = x > 0 ? cdf_x_cond_y[x-1] : 0.0f;
	float pmf_x = cdf_x_cond_y[x] - prev;
	sample.x = (sample.x - prev) / pmf_x;

	if (pdf) {
		*pdf = pmf_x * pmf_y * product(res);
	}

	return {((float)x + sample.x) / (float)res.x, ((float)y + sample.y) / (float)res.y};
}

inline NGP_HOST_DEVICE float pdf_2d(vec2 sample, uint32_t img, const ivec2& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y) {
	ivec2 p = clamp(ivec2(sample * vec2(res)), 0, res - 1);

	cdf_y += img * res.y;
	cdf_x_cond_y += img * res.y * res.x + p.y * res.x;

	float pmf_y = cdf_y[p.y];
	if (p.y > 0) {
		pmf_y -= cdf_y[p.y-1];
	}

	float pmf_x = cdf_x_cond_y[p.x];
	if (p.x > 0) {
		pmf_x -= cdf_x_cond_y[p.x-1];
	}

	// Probability mass of picking the pixel
	float pmf = pmf_x * pmf_y;

	// To convert to probability density, divide by area of pixel
	return UNIFORM_SAMPLING_FRACTION + pmf * product(res) * (1.0f - UNIFORM_SAMPLING_FRACTION);
}

inline __device__ vec2 nerf_random_image_pos_training(default_rng_t& rng, const ivec2& resolution, bool snap_to_pixel_centers, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, const ivec2& cdf_res, uint32_t img, float* __restrict__ pdf = nullptr) {
	vec2 uv = random_val_2d(rng);

	if (cdf_x_cond_y) {
		uv = sample_cdf_2d(uv, img, cdf_res, cdf_x_cond_y, cdf_y, pdf);
	} else {
		// // Warp-coherent tile
		// uv.x = __shfl_sync(0xFFFFFFFF, uv.x, 0);
		// uv.y = __shfl_sync(0xFFFFFFFF, uv.y, 0);

		// const ivec2 TILE_SIZE = {8, 4};
		// uv = (uv * vec2(resolution - TILE_SIZE) + vec2(tcnn::lane_id() % TILE_SIZE.x, tcnn::lane_id() / threadIdx.x)) / vec2(resolution);

		if (pdf) {
			*pdf = 1.0f;
		}
	}

	if (snap_to_pixel_centers) {
		uv = (vec2(clamp(ivec2(uv * vec2(resolution)), 0, resolution - 1)) + 0.5f) / vec2(resolution);
	}

	return uv;
}

inline NGP_HOST_DEVICE uint32_t image_idx(uint32_t base_idx, uint32_t n_rays, uint32_t n_rays_total, uint32_t n_training_images, const float* __restrict__ cdf = nullptr, float* __restrict__ pdf = nullptr) {
	if (cdf) {
		float sample = ld_random_val(base_idx/* + n_rays_total*/, 0xdeadbeef);
		// float sample = random_val(base_idx/* + n_rays_total*/);
		uint32_t img = binary_search(sample, cdf, n_training_images);

		if (pdf) {
			float prev = img > 0 ? cdf[img-1] : 0.0f;
			*pdf = (cdf[img] - prev) * n_training_images;
		}

		return img;
	}

	// return ((base_idx/* + n_rays_total*/) * 56924617 + 96925573) % n_training_images;

	// Neighboring threads in the warp process the same image. Increases locality.
	if (pdf) {
		*pdf = 1.0f;
	}
	return (((base_idx/* + n_rays_total*/) * n_training_images) / n_rays) % n_training_images;
}

inline NGP_HOST_DEVICE LossAndGradient loss_and_gradient(const vec3& target, const vec3& prediction, ELossType loss_type) {
	switch (loss_type) {
		case ELossType::RelativeL2:  return relative_l2_loss(target, prediction); break;
		case ELossType::L1:          return l1_loss(target, prediction); break;
		case ELossType::Mape:        return mape_loss(target, prediction); break;
		case ELossType::Smape:       return smape_loss(target, prediction); break;
		// Note: we divide the huber loss by a factor of 5 such that its L2 region near zero
		// matches with the L2 loss and error numbers become more comparable. This allows reading
		// off dB numbers of ~converged models and treating them as approximate PSNR to compare
		// with other NeRF methods. Self-normalizing optimizers such as Adam are agnostic to such
		// constant factors; optimization is therefore unaffected.
		case ELossType::Huber:       return huber_loss(target, prediction, 0.1f) / 5.0f; break;
		case ELossType::LogL1:       return log_l1_loss(target, prediction); break;
		default: case ELossType::L2: return l2_loss(target, prediction); break;
	}
}

}
