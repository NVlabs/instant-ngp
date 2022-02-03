/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_nerf.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */


#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#ifdef copysign
#undef copysign
#endif

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;


NGP_NAMESPACE_BEGIN

inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
inline constexpr __device__ uint32_t NERF_CASCADES() { return 5; }

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;


inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
	return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}

inline __host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f& focal_length, float cone_angle_constant) {
	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();

	return cone_angle_constant;
}

inline __host__ __device__ float calc_dt(float t, float cone_angle) {
	return tcnn::clamp(t*cone_angle, MIN_CONE_STEPSIZE(), MAX_CONE_STEPSIZE());
}

struct LossAndGradient {
	Eigen::Array3f loss;
	Eigen::Array3f gradient;

	__host__ __device__ LossAndGradient operator*(float scalar) {
		return {loss * scalar, gradient * scalar};
	}

	__host__ __device__ LossAndGradient operator/(float scalar) {
		return {loss / scalar, gradient / scalar};
	}
};

inline __device__ Array3f copysign(const Array3f& a, const Array3f& b) {
	return {
		copysignf(a.x(), b.x()),
		copysignf(a.y(), b.y()),
		copysignf(a.z(), b.z()),
	};
}

inline __device__ LossAndGradient l2_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	return {
		difference * difference,
		2.0f * difference
	};
}

inline __device__ LossAndGradient relative_l2_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f factor = (prediction * prediction + Array3f::Constant(1e-2f)).inverse();
	return {
		difference * difference * factor,
		2.0f * difference * factor
	};
}

inline __device__ LossAndGradient l1_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	return {
		difference.abs(),
		copysign(Array3f::Ones(), difference),
	};
}

inline __device__ LossAndGradient huber_loss(const Array3f& target, const Array3f& prediction, float alpha = 1) {
	Array3f difference = prediction - target;
	Array3f abs_diff = difference.abs();
	Array3f square = 0.5f/alpha * difference * difference;
	return {
		{
			abs_diff.x() > alpha ? (abs_diff.x() - 0.5f * alpha) : square.x(),
			abs_diff.y() > alpha ? (abs_diff.y() - 0.5f * alpha) : square.y(),
			abs_diff.z() > alpha ? (abs_diff.z() - 0.5f * alpha) : square.z(),
		},
		{
			abs_diff.x() > alpha ? (difference.x() > 0 ? 1.0f : -1.0f) : (difference.x() / alpha),
			abs_diff.y() > alpha ? (difference.y() > 0 ? 1.0f : -1.0f) : (difference.y() / alpha),
			abs_diff.z() > alpha ? (difference.z() > 0 ? 1.0f : -1.0f) : (difference.z() / alpha),
		},
	};
}

inline __device__ LossAndGradient log_l1_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f divisor = difference.abs() + Array3f::Ones();
	return {
		divisor.log(),
		copysign(divisor.inverse(), difference),
	};
}

inline __device__ LossAndGradient smape_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f factor = (0.5f * (prediction.abs() + target.abs()) + Array3f::Constant(1e-2f)).inverse();
	return {
		difference.abs() * factor,
		copysign(factor, difference),
	};
}

inline __device__ LossAndGradient mape_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f factor = (prediction.abs() + Array3f::Constant(1e-2f)).inverse();
	return {
		difference.abs() * factor,
		copysign(factor, difference),
	};
}

inline __device__ float distance_to_next_voxel(const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) { // dda like step
	Vector3f p = res * pos;
	float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
	float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
	float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) {
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
	do {
		t += calc_dt(t, cone_angle);
	} while (t < t_target);
	return t;
}

__device__ float network_to_rgb(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_rgb_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_density(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_density_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, ENerfActivation activation) {
	return {
		network_to_rgb(float(local_network_output[0]), activation),
		network_to_rgb(float(local_network_output[1]), activation),
		network_to_rgb(float(local_network_output[2]), activation)
	};
}

__device__ Vector3f warp_position(const Vector3f& pos, const BoundingBox& aabb) {
	// return {tcnn::logistic(pos.x() - 0.5f), tcnn::logistic(pos.y() - 0.5f), tcnn::logistic(pos.z() - 0.5f)};
	// return pos;

	return aabb.relative_pos(pos);
}

__device__ Vector3f unwarp_position(const Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.min + pos.cwiseProduct(aabb.diag());
}

__device__ Vector3f unwarp_position_derivative(const Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.diag();
}

__device__ Vector3f warp_position_derivative(const Vector3f& pos, const BoundingBox& aabb) {
	return unwarp_position_derivative(pos, aabb).cwiseInverse();
}

__device__ Vector3f warp_direction(const Vector3f& dir) {
	return (dir + Vector3f::Ones()) * 0.5f;
}

__device__ Vector3f unwarp_direction(const Vector3f& dir) {
	return dir * 2.0f - Vector3f::Ones();
}

__device__ Vector3f warp_direction_derivative(const Vector3f& dir) {
	return Vector3f::Constant(0.5f);
}

__device__ Vector3f unwarp_direction_derivative(const Vector3f& dir) {
	return Vector3f::Constant(2.0f);
}

__device__ float warp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

__device__ float unwarp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

__device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip) {
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= Vector3f::Constant(0.5f);
	pos *= mip_scale;
	pos += Vector3f::Constant(0.5f);

	Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

	if (i.x() < -1 || i.x() > NERF_GRIDSIZE() || i.y() < -1 || i.y() > NERF_GRIDSIZE() || i.z() < -1 || i.z() > NERF_GRIDSIZE()) {
		printf("WTF %d %d %d\n", i.x(), i.y(), i.z());
	}

	uint32_t idx = tcnn::morton3D(
		tcnn::clamp(i.x(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.y(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.z(), 0, (int)NERF_GRIDSIZE()-1)
	);

	return idx;
}

__device__ bool density_grid_occupied_at(const Vector3f& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return density_grid_bitfield[idx/8+grid_mip_offset(mip)/8] & (1<<(idx%8));
}

__device__ float cascaded_grid_at(Vector3f pos, const float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__device__ float& cascaded_grid_at(Vector3f pos, float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__global__ void extract_srgb_with_activation(const uint32_t n_elements,	const uint32_t rgb_stride, const float* __restrict__ rgbd, float* __restrict__ rgb, ENerfActivation rgb_activation, bool from_linear) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	float c = network_to_rgb(rgbd[elem_idx*4 + dim_idx], rgb_activation);
	if (from_linear) {
		c = linear_to_srgb(c);
	}
	rgb[elem_idx*rgb_stride + dim_idx] = c;
}

__global__ void mark_untrained_density_grid(const uint32_t n_elements,  float* __restrict__ grid_out,
	const uint32_t n_training_images,
	const Vector2f* __restrict__ focal_lengths,
	const Matrix<float, 3, 4>* training_xforms,
	Vector2i resolution
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	uint32_t level = i / (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());
	uint32_t pos_idx = i % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	float half_resx=resolution.x()*0.5f;
	float half_resy=resolution.y()*0.5f;

	Vector3f pos = ((Vector3f{(float)x+0.5f, (float)y+0.5f, (float)z+0.5f}) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);
	float voxel_radius = 0.5f*SQRT3()*scalbnf(1.0f, level) / NERF_GRIDSIZE();
	int count=0;
	for (uint32_t j=0; j < n_training_images; ++j) {
		Matrix<float, 3, 4> xform = training_xforms[j];
		Vector3f ploc = pos-xform.col(3);
		float x=ploc.dot(xform.col(0));
		float y=ploc.dot(xform.col(1));
		float z=ploc.dot(xform.col(2));
		if (z>0.f) {
			auto focal = focal_lengths[j];
			// TODO - add a box / plane intersection to stop thomas from murdering me
			if (fabsf(x)-voxel_radius < z/focal.x()*half_resx && fabsf(y)-voxel_radius < z/focal.y()*half_resy) {
				count++;
				if (count > 0) break;
			}
		}
	}
	grid_out[i] = (count > 0) ? 0.f : -1.f;
}


__global__ void generate_grid_samples_nerf_uniform(Eigen::Vector3i res_3d, const uint32_t step, BoundingBox render_aabb, BoundingBox train_aabb, NerfPosition* __restrict__ out) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x>=res_3d.x() || y>=res_3d.y() || z>=res_3d.z())
		return;
	uint32_t i = x+ y*res_3d.x() + z*res_3d.x()*res_3d.y();
	Vector3f pos = Array3f{(float)x, (float)y, (float)z} * Array3f{1.f/res_3d.x(),1.f/res_3d.y(),1.f/res_3d.z()};
	pos = pos.cwiseProduct(render_aabb.max - render_aabb.min) + render_aabb.min;
	out[i] = { warp_position(pos, train_aabb), warp_dt(MIN_CONE_STEPSIZE()) };
}

// generate samples for uniform grid including constant ray direction
__global__ void generate_grid_samples_nerf_uniform_dir(Eigen::Vector3i res_3d, const uint32_t step, BoundingBox render_aabb, BoundingBox train_aabb, Eigen::Vector3f ray_dir, NerfCoordinate* __restrict__ network_input) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x>=res_3d.x() || y>=res_3d.y() || z>=res_3d.z())
		return;
	uint32_t i = x+ y*res_3d.x() + z*res_3d.x()*res_3d.y();
	Vector3f pos = Array3f{(float)x, (float)y, (float)z} * Array3f{1.f/res_3d.x(),1.f/res_3d.y(),1.f/res_3d.z()};
	pos = pos.cwiseProduct(render_aabb.max - render_aabb.min) + render_aabb.min;
	network_input[i] = { warp_position(pos, train_aabb), warp_direction(ray_dir), warp_dt(MIN_CONE_STEPSIZE()) };
}

inline __device__ int mip_from_pos(const Vector3f& pos) {
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(NERF_CASCADES()-1, max(0, exponent+1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f& pos) {
	int mip = mip_from_pos(pos);
	dt *= 2*NERF_GRIDSIZE();
	if (dt<1.f) return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(NERF_CASCADES()-1, max(exponent, mip));
}


__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements, default_rng_t rng, const uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, uint32_t* __restrict__ indices, uint32_t n_cascades, float thresh) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// 1 random number to select the level, 3 to select the position.
	rng.advance(i*4);
	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

	// Select grid cell that has density
	uint32_t idx;
	for (uint32_t j = 0; j < 10; ++j) {
		idx = ((i+step*n_elements) * 56924617 + j * 19349663 + 96925573) % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());
		idx += level * NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE();
		if (grid_in[idx] > thresh) {
			break;
		}
	}

	// Random position within that cellq
	uint32_t pos_idx = idx % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	Vector3f pos = ((Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);

	out[i] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
	indices[i] = idx;
}

__global__ void splat_grid_samples_nerf_max_nearest_neighbor(const uint32_t n_elements, const uint32_t* __restrict__ indices, int padded_output_width, const tcnn::network_precision_t* network_output, float* __restrict__ grid_out, ENerfActivation rgb_activation, ENerfActivation density_activation) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t local_idx = indices[i];

	// Current setting: optical thickness of the smallest possible stepsize.
	// Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
	uint32_t level = 0;//local_idx / (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());

	float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
	float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);

	// Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
	// uint atomicMax is thus perfectly acceptable.
	atomicMax((uint32_t*)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

__global__ void grid_samples_half_to_float(const uint32_t n_elements, BoundingBox aabb, float *dst, int padded_output_width, const tcnn::network_precision_t* network_output, ENerfActivation density_activation, const NerfPosition* __restrict__ coords_in, const float* __restrict__ grid_in) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// let's interpolate for marching cubes based on the raw MLP output, not the density (exponentiated) version
	//float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
	float mlp = float(network_output[i * padded_output_width]);

	if (grid_in) {
		Vector3f pos = unwarp_position(coords_in[i].p, aabb);
		float grid_density = cascaded_grid_at(pos, grid_in, mip_from_pos(pos));
		if (grid_density < NERF_MIN_OPTICAL_THICKNESS()) {
			mlp = -10000.f;
		}
	}
	dst[i] = mlp;
}

__global__ void ema_grid_samples_nerf(const uint32_t n_elements,
	float decay,
	const uint32_t count,
	float* __restrict__ grid_out,
	const float* __restrict__ grid_in
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float importance = grid_in[i];

	// float ema_debias_old = 1 - (float)powf(decay, count);
	// float ema_debias_new = 1 - (float)powf(decay, count+1);

	// float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
	// grid_out[i] = filtered_val;

	// Maximum instead of EMA allows capture of very thin features.
	// Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

	float prev_val = grid_out[i];
	float val = (prev_val<0.f) ? prev_val : fmaxf(prev_val * decay, importance);
	grid_out[i] = val;
}

__global__ void decay_sharpness_grid_nerf(const uint32_t n_elements, float decay, float* __restrict__ grid) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	grid[i] *= decay;
}

__global__ void grid_to_bitfield(const uint32_t n_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

	#pragma unroll
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(const uint32_t n_elements,
	const uint8_t* __restrict__ prev_level,
	uint8_t* __restrict__ next_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	#pragma unroll
	for (uint8_t j = 0; j < 8; ++j) {
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = tcnn::morton3D_invert(i>>0) + NERF_GRIDSIZE()/8;
	uint32_t y = tcnn::morton3D_invert(i>>1) + NERF_GRIDSIZE()/8;
	uint32_t z = tcnn::morton3D_invert(i>>2) + NERF_GRIDSIZE()/8;

	next_level[tcnn::morton3D(x, y, z)] |= bits;
}


__global__ void advance_pos_nerf(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	Vector3f camera_fwd,
	Vector2f focal_length,
	uint32_t spp,
	NerfPayload* __restrict__ payloads,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	Vector3f origin = payload.origin;
	Vector3f dir = payload.dir;
	Vector3f idir = dir.cwiseInverse();

	float cone_angle = calc_cone_angle(dir.dot(camera_fwd), focal_length, cone_angle_constant);

	float t = payload.t;
	float dt = calc_dt(t, cone_angle);
	t += ld_random_val(spp, i * 786433) * dt;
	Vector3f pos;

	while (1) {
		if (!render_aabb.contains(pos = origin + dir * t)) {
			payload.alive = false;
			break;
		}

		dt = calc_dt(t, cone_angle);
		uint32_t mip = max(min_mip, mip_from_dt(dt, pos));

		if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
			break;
		}

		uint32_t res = NERF_GRIDSIZE()>>mip;
		t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
	}

	payload.t = t;
}

__global__ void generate_nerf_network_inputs_from_positions(const uint32_t n_elements, BoundingBox aabb, const Vector3f* __restrict__ pos, NerfCoordinate* __restrict__ network_input) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Vector3f dir=(pos[i]-Vector3f::Constant(0.5f)).normalized(); // choose outward pointing directions, for want of a better choice
	network_input[i] = { warp_position(pos[i], aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()) };
}

__global__ void generate_nerf_network_inputs_at_current_position(const uint32_t n_elements, BoundingBox aabb, const NerfPayload* __restrict__ payloads, NerfCoordinate* __restrict__ network_input) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Vector3f dir = payloads[i].dir;
	network_input[i] = { warp_position(payloads[i].origin + dir * payloads[i].t, aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()) };
}

__global__ void compute_nerf_density(const uint32_t n_elements, Array4f* network_output, ENerfActivation rgb_activation, ENerfActivation density_activation) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Array4f rgba = network_output[i];
	rgba.w() = tcnn::clamp(1.f - __expf(-network_to_density(rgba.w(), density_activation) / 100.0f), 0.0f, 1.0f);
	rgba.x() = network_to_rgb(rgba.x(), rgb_activation) * rgba.w();
	rgba.y() = network_to_rgb(rgba.y(), rgb_activation) * rgba.w();
	rgba.z() = network_to_rgb(rgba.z(), rgb_activation) * rgba.w();

	network_output[i] = rgba;
}

__global__ void generate_next_nerf_network_inputs(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	BoundingBox train_aabb,
	Vector2f focal_length,
	Vector3f camera_fwd,
	NerfPayload* __restrict__ payloads,
	NerfCoordinate* __restrict__ network_input,
	uint32_t n_steps,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	Vector3f origin = payload.origin;
	Vector3f dir = payload.dir;
	Vector3f idir = dir.cwiseInverse();

	float cone_angle = calc_cone_angle(dir.dot(camera_fwd), focal_length, cone_angle_constant);

	float t = payload.t;

	for (uint32_t j = 0; j < n_steps; ++j) {
		Vector3f pos;
		float dt = 0.0f;
		while (1) {
			if (!render_aabb.contains(pos = origin + dir * t)) {
				payload.n_steps = j;
				return;
			}

			dt = calc_dt(t, cone_angle);
			uint32_t mip = max(min_mip, mip_from_dt(dt, pos));

			if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
				break;
			}

			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
		}

		network_input[i*n_steps + j] = { warp_position(pos, train_aabb), warp_direction(dir), warp_dt(dt) }; // XXXCONE
		t += dt;
	}

	payload.t = t;
	payload.n_steps = n_steps;
}


__global__ void composite_kernel_nerf(
	const uint32_t n_elements,
	const uint32_t current_step,
	BoundingBox aabb,
	const uint32_t n_training_images,
	const Matrix<float, 3, 4>* training_xforms,
	Matrix<float, 3, 4> camera_matrix,
	Vector2f focal_length,
	float depth_scale,
	Array4f* rgba,
	NerfPayload* payloads,
	const NerfCoordinate* network_input,
	const tcnn::network_precision_t* network_output,
	uint32_t padded_output_width,
	uint32_t n_steps,
	ERenderMode render_mode,
	const uint8_t* density_grid,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_alpha
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	Array4f local_rgba = rgba[i];
	Vector3f origin = payload.origin;
	Vector3f cam_fwd = camera_matrix.col(2);
	// Composite in the last n steps
	uint32_t actual_n_steps = payload.n_steps;
	uint32_t j = 0;

	for (; j < actual_n_steps; ++j) {
		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)&network_output[(i*n_steps+j)*padded_output_width];
		Vector3f warped_pos = network_input[i*n_steps+j].pos.p;
		Vector3f pos = unwarp_position(warped_pos, aabb);

		//Vector3f pos2 = pos-Vector3f::Constant(0.5f);
		//float fog_scale = 1.f/max(1.f,pos2.dot(pos2)*4.f);

		float T = 1.f - local_rgba.w();
		float dt = unwarp_dt(network_input[i*n_steps+j].dt);
		float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt /* * fog_scale*/);
		if (show_accel>=0)
		alpha=1.f;
		float weight = alpha * T;

		Array3f rgb = network_to_rgb(local_network_output, rgb_activation);

		if (render_mode == ERenderMode::Normals) {
			// Network input contains the gradient of the network output w.r.t. input.
			// So to compute density gradients, we need to apply the chain rule.
			// The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
			Vector3f normal = -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
			rgb = normal.normalized().array();
		} else if (render_mode == ERenderMode::Positions || render_mode == ERenderMode::EncodingVis) {
			if (show_accel>=0) {
				uint32_t mip = max(show_accel, mip_from_pos(pos));
				uint32_t res = NERF_GRIDSIZE() >> mip;
				int ix = pos.x()*(res);
				int iy = pos.y()*(res);
				int iz = pos.z()*(res);
				default_rng_t rng(ix+iy*232323+iz*727272);
				rgb.x() = 1.f-mip*(1.f/(NERF_CASCADES()-1));
				rgb.y() = rng.next_float();
				rgb.z() = rng.next_float();
			} else {
				rgb = pos.array();
			}
		} else if (render_mode == ERenderMode::Depth) {
			float z=cam_fwd.dot(pos-origin) * depth_scale;
			rgb = {z,z,z};
		} else if (render_mode == ERenderMode::Distance) {
			float z=(pos-origin).norm() * depth_scale;
			rgb = {z,z,z};
		} else if (render_mode == ERenderMode::Stepsize) {
			float warped_dt = warp_dt(dt);
			rgb = {warped_dt,warped_dt,warped_dt};
		} else if (render_mode == ERenderMode::AO) {
			rgb = Array3f::Constant(alpha);
		}

		local_rgba.head<3>() += rgb * weight;
		local_rgba.w() += weight;

		if (local_rgba.w() > (1.0f - min_alpha)) {
			rgba[i] = local_rgba / local_rgba.w();
			break;
		}
	}

	if (j < n_steps) {
		payload.alive = false;
		payload.n_steps = j + current_step;
	}

	rgba[i] = local_rgba;
}

static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;

inline __device__ Vector2f sample_cdf_2d(Vector2f sample, uint32_t img, const Vector2i& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, float* __restrict__ pdf) {
	if (sample.x() < UNIFORM_SAMPLING_FRACTION) {
		sample.x() /= UNIFORM_SAMPLING_FRACTION;
		return sample;
	}

	sample.x() = (sample.x() - UNIFORM_SAMPLING_FRACTION) / (1.0f - UNIFORM_SAMPLING_FRACTION);

	cdf_y += img * res.y();

	// First select row according to cdf_y
	uint32_t y = binary_search(sample.y(), cdf_y, res.y());
	float prev = y > 0 ? cdf_y[y-1] : 0.0f;
	float pmf_y = cdf_y[y] - prev;
	sample.y() = (sample.y() - prev) / pmf_y;

	cdf_x_cond_y += img * res.y() * res.x() + y * res.x();

	// Then, select col according to x
	uint32_t x = binary_search(sample.x(), cdf_x_cond_y, res.x());
	prev = x > 0 ? cdf_x_cond_y[x-1] : 0.0f;
	float pmf_x = cdf_x_cond_y[x] - prev;
	sample.x() = (sample.x() - prev) / pmf_x;

	if (pdf) {
		*pdf = pmf_x * pmf_y * res.prod();
	}

	return {((float)x + sample.x()) / (float)res.x(), ((float)y + sample.y()) / (float)res.y()};
}

inline __device__ float pdf_2d(Vector2f sample, uint32_t img, const Vector2i& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y) {
	Vector2i p = (sample.cwiseProduct(res.cast<float>())).cast<int>().cwiseMax(0).cwiseMin(res - Vector2i::Ones());

	cdf_y += img * res.y();
	cdf_x_cond_y += img * res.y() * res.x() + p.y() * res.x();

	float pmf_y = cdf_y[p.y()];
	if (p.y() > 0) {
		pmf_y -= cdf_y[p.y()-1];
	}

	float pmf_x = cdf_x_cond_y[p.x()];
	if (p.x() > 0) {
		pmf_x -= cdf_x_cond_y[p.x()-1];
	}

	// Probability mass of picking the pixel
	float pmf = pmf_x * pmf_y;

	// To convert to probability density, divide by area of pixel
	return UNIFORM_SAMPLING_FRACTION + pmf * res.prod() * (1.0f - UNIFORM_SAMPLING_FRACTION);
}

inline __device__ Vector2f nerf_random_image_pos_training(default_rng_t& rng, const Vector2i& resolution, bool snap_to_pixel_centers, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, const Vector2i& cdf_res, uint32_t img, float* __restrict__ pdf = nullptr) {
	Vector2f xy = random_val_2d(rng);

	if (cdf_x_cond_y) {
		xy = sample_cdf_2d(xy, img, cdf_res, cdf_x_cond_y, cdf_y, pdf);
	} else if (pdf) {
		*pdf = 1.0f;
	}

	if (snap_to_pixel_centers) {
		xy = (xy.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMax(0).cwiseMin(resolution - Vector2i::Ones()).cast<float>() + Vector2f::Constant(0.5f)).cwiseQuotient(resolution.cast<float>());
	}
	return xy;
}

inline __device__ uint32_t image_idx(uint32_t base_idx, uint32_t n_rays, uint32_t n_rays_total, uint32_t n_training_images, const float* __restrict__ cdf = nullptr, float* __restrict__ pdf = nullptr) {
	if (cdf) {
		float sample = ld_random_val(base_idx + n_rays_total, 0xdeadbeef);
		// float sample = random_val(base_idx + n_rays_total);
		uint32_t img = binary_search(sample, cdf, n_training_images);

		if (pdf) {
			float prev = img > 0 ? cdf[img-1] : 0.0f;
			*pdf = (cdf[img] - prev) * n_training_images;
		}

		return img;
	}

	// return ((base_idx + n_rays_total) * 56924617 + 96925573) % n_training_images;

	// Neighboring threads in the warp process the same image. Increases locality.
	if (pdf) {
		*pdf = 1.0f;
	}
	return (((base_idx + n_rays_total) * n_training_images) / n_rays) % n_training_images;
}

inline __device__ Vector2i image_pos(const Vector2f& pos, const Vector2i& resolution) {
	return pos.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMin(resolution - Vector2i::Constant(1)).cwiseMax(0);
}

inline __device__ uint64_t pixel_idx(const Vector2i& pos, const Vector2i& resolution, uint32_t img) {
	return pos.x() + pos.y() * resolution.x() + img * (uint64_t)resolution.x() * resolution.y();
}

inline __device__ uint64_t pixel_idx(const Vector2f& xy, const Vector2i& resolution, uint32_t img) {
	return pixel_idx(image_pos(xy, resolution), resolution, img);
}

__global__ void generate_training_samples_nerf(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const Ray* __restrict__ rays_in,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out,
	uint32_t* __restrict__ numsteps_out,
	NerfCoordinate* __restrict__ coords_out,
	Vector2i resolution,
	const uint32_t n_training_images,
	Vector2f principal_point,
	const Vector2f* __restrict__ focal_lengths,
	const Matrix<float, 3, 4>* training_xforms,
	CameraDistortion camera_distortion,
	const uint8_t* __restrict__ density_grid,
	bool max_level_rand_training,
	float* __restrict__ max_level_ptr,
	bool snap_to_pixel_centers,
	bool train_envmap,
	float cone_angle_constant,
	const float* __restrict__ distortion_data,
	const Vector2i distortion_resolution,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const Vector2i cdf_res,
	float near_distance,
	const __half* __restrict__ training_images
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	uint32_t img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	Vector2f xy = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

	// Negative values indicate masked-away regions
	if ((float)training_images[pixel_idx(xy, resolution, img)*4] < 0.0f) {
		return;
	}

	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	Matrix<float, 3, 4> xform = training_xforms[img];
	Vector2f focal_length = focal_lengths[img];

	Ray ray;

	if (rays_in) {
		// Rays have been explicitly supplied. Read them.
		ray = rays_in[pixel_idx(xy, resolution, img)];
	} else {
		// Rays need to be inferred from the camera matrix
		ray.o = xform.col(3);
		ray.d = {
			(xy.x()-principal_point.x())*resolution.x() / focal_length.x(),
			(xy.y()-principal_point.y())*resolution.y() / focal_length.y(),
			1.0f,
		};
		if (!camera_distortion.is_zero()) {
			iterative_camera_undistortion((float*)&camera_distortion, &ray.d.x(), &ray.d.y());
		}
		if (distortion_data) {
			ray.d.head<2>() += read_image<2>(distortion_data, distortion_resolution, xy);
		}

		ray.d = (xform.block<3, 3>(0, 0) * ray.d).normalized();
	}

	Vector2f tminmax = aabb.ray_intersect(ray.o, ray.d);
	float cone_angle = calc_cone_angle(ray.d.dot(xform.col(2)), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x() = fmaxf(tminmax.x(), near_distance);

	float startt = tminmax.x();
	startt += calc_dt(startt, cone_angle) * random_val(rng);
	Vector3f idir = ray.d.cwiseInverse();

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t=startt;
	Vector3f pos;

	while (aabb.contains(pos = ray.o + t * ray.d) && j < NERF_STEPS()) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray.d, idir, res);
		}
	}
	if (j == 0 && !train_envmap) {
		return;
	}
	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	if (base + numsteps > max_samples) {
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	ray_indices_out[ray_idx] = i;
	rays_out[ray_idx] = ray;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	Vector3f warped_dir = warp_direction(ray.d);
	t=startt;
	j=0;
	while (aabb.contains(pos = ray.o + t * ray.d) && j < numsteps) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			coords_out[j] = { warp_position(pos, aabb), warped_dir, warp_dt(dt) };
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray.d, idir, res);
		}
	}
	if (max_level_rand_training) {
		max_level_ptr += base;
		for (j = 0; j < numsteps; ++j) {
			max_level_ptr[j] = max_level;
		}
	}
}

__device__ LossAndGradient loss_and_gradient(const Vector3f& target, const Vector3f& prediction, ELossType loss_type) {
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

inline __device__ Array3f composit_and_lerp(Vector2f pos, const Vector2i& resolution, uint32_t img, const __half* training_images, const Array3f& background_color, const Array3f& exposure_scale = Array3f::Ones()) {
	pos = (pos.cwiseProduct(resolution.cast<float>()) - Vector2f::Constant(0.5f)).cwiseMax(0.0f).cwiseMin(resolution.cast<float>() - Vector2f::Constant(1.0f + 1e-4f));

	Vector2i pos_int = pos.cast<int>();
	auto weight = pos - pos_int.cast<float>();

	Vector2i idx = pos_int.cwiseMin(resolution - Vector2i::Constant(2)).cwiseMax(0);

	auto read_val = [&](const Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
	};

	Array3f result = (
		(1 - weight.x()) * (1 - weight.y()) * read_val({idx.x(), idx.y()}) +
		(weight.x()) * (1 - weight.y()) * read_val({idx.x()+1, idx.y()}) +
		(1 - weight.x()) * (weight.y()) * read_val({idx.x(), idx.y()+1}) +
		(weight.x()) * (weight.y()) * read_val({idx.x()+1, idx.y()+1})
	);

	return result;
}

inline __device__ Array3f composit(Vector2f pos, const Vector2i& resolution, uint32_t img, const __half* training_images, const Array3f& background_color, const Array3f& exposure_scale = Array3f::Ones()) {
	Vector2i idx = image_pos(pos, resolution);

	auto read_val = [&](const Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
	};

	return read_val(idx);
}

inline __device__ Array4f read_rgba(Vector2f pos, const Vector2i& resolution, uint32_t img, const __half* training_images) {
	Vector2i idx = image_pos(pos, resolution);

	auto read_val = [&](const Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Array4f{val[0], val[1], val[2], val[3]};
	};

	return read_val(idx);
}

__global__ void compute_loss_kernel_train_nerf(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const uint32_t max_samples_compacted,
	const uint32_t* __restrict__ rays_counter,
	float loss_scale,
	int padded_output_width,
	const float* __restrict__ envmap_data,
	float* __restrict__ envmap_gradient,
	const Vector2i envmap_resolution,
	ELossType envmap_loss_type,
	Array3f background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const __half* __restrict__ training_images,
	const uint32_t n_training_images,
	Vector2i resolution,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in,
	uint32_t* __restrict__ numsteps_in,
	const NerfCoordinate* __restrict__ coords_in,
	NerfCoordinate* __restrict__ coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ELossType loss_type,
	float* __restrict__ loss_output,
	bool max_level_rand_training,
	float* __restrict__ max_level_compacted_ptr,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	bool snap_to_pixel_centers,
	float* __restrict__ error_map,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const Vector2i error_map_res,
	const Vector2i error_map_cdf_res,
	const float* __restrict__ sharpness_data,
	Eigen::Vector2i sharpness_resolution,
	float* __restrict__ sharpness_grid,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	const Eigen::Array3f* __restrict__ exposure,
	Eigen::Array3f* __restrict__ exposure_gradient
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * 4;

	float T = 1.f;

	float EPSILON = 1e-4f;

	Array3f rgb_ray = Array3f::Zero();
	Vector3f hitpoint = Vector3f::Zero();

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T < EPSILON) {
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		const Vector3f pos = unwarp_position(coords_in->pos.p, aabb);
		const float dt = unwarp_dt(coords_in->dt);

		float density = network_to_density(float(local_network_output[3]), density_activation);


		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray += weight * rgb;
		hitpoint += weight * pos;
		T *= (1.f - alpha);

		network_output += 4;
		coords_in++;
	}
	hitpoint /= (1.0f - T);

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	float img_pdf = 1.0f;
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);

	float xy_pdf = 1.0f;
	Vector2f xy = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &xy_pdf);
	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	Array3f pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	Array4f envmap_value;
	Vector3f dir;
	if (envmap_data) {
		dir = rays_in[i].d;
		envmap_value = read_envmap(envmap_data, envmap_resolution, dir);
		background_color = envmap_value.head<3>() + background_color * (1.0f - envmap_value.w());
	}

	Array3f exposure_scale = (0.6931471805599453f * exposure[img]).exp();
	// Array3f rgbtarget = composit_and_lerp(xy, resolution, img, training_images, background_color, exposure_scale);
	// Array3f rgbtarget = composit(xy, resolution, img, training_images, background_color, exposure_scale);
	Array4f texsamp = read_rgba(xy, resolution, img, training_images);

	Array3f rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.head<3>() + (1.0f - texsamp.w()) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.w() > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.head<3>() / texsamp.w()) * texsamp.w() + (1.0f - texsamp.w()) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

	// Step again, this time computing loss
	network_output -= 4 * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_in[i*2+0] = compacted_numsteps;
	numsteps_in[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	max_level_compacted_ptr += compacted_base;
	coords_out += compacted_base;

	dloss_doutput += compacted_base * padded_output_width;

	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
	lg.loss /= img_pdf * xy_pdf;

	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * xy_pdf;

	float mean_loss = lg.loss.mean();
	if (loss_output) {
		loss_output[i] = mean_loss / (float)n_rays;
	}

	if (error_map) {
		Vector2f pos = (xy.cwiseProduct(error_map_res.cast<float>()) - Vector2f::Constant(0.5f)).cwiseMax(0.0f).cwiseMin(error_map_res.cast<float>() - Vector2f::Constant(1.0f + 1e-4f));
		Vector2i pos_int = pos.cast<int>();
		auto weight = pos - pos_int.cast<float>();

		Vector2i idx = pos_int.cwiseMin(resolution - Vector2i::Constant(2)).cwiseMax(0);

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[img * error_map_res.prod() + y * error_map_res.x() + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			Vector2i sharpness_pos = xy.cwiseProduct(sharpness_resolution.cast<float>()).cast<int>().cwiseMax(0).cwiseMin(sharpness_resolution - Vector2i::Constant(1));
			float sharp = sharpness_data[img * sharpness_resolution.prod() + sharpness_pos.y() * sharpness_resolution.x() + sharpness_pos.x()] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x(),   idx.y(),   (1 - weight.x()) * (1 - weight.y()) * mean_loss);
		deposit_val(idx.x()+1, idx.y(),        weight.x()  * (1 - weight.y()) * mean_loss);
		deposit_val(idx.x(),   idx.y()+1, (1 - weight.x()) *      weight.y()  * mean_loss);
		deposit_val(idx.x()+1, idx.y()+1,      weight.x()  *      weight.y()  * mean_loss);
	}

	loss_scale /= n_rays;

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
	Array3f rgb_ray2 = { 0.f,0.f,0.f };
	T = 1.f;
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {
		if (max_level_rand_training) {
			max_level_compacted_ptr[j] = max_level;
		}
		// Compact network inputs
		coords_out[j] = coords_in[j];
		float dt = unwarp_dt(coords_in[j].dt);
		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		const float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray2 += weight * rgb;
		T *= (1.f - alpha);

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const Array3f suffix = rgb_ray - rgb_ray2;
		const Array3f dloss_by_drgb = weight * lg.gradient;

		tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x() * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y() * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z() * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		float dloss_by_dmlp = density_derivative * (
			dt * lg.gradient.matrix().dot((T * rgb - suffix).matrix())
		);

		//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
		//dloss_by_dmlp += (texsamp.w()<0.001f) ? mask_supervision_strength * weight : 0.f ;

		local_dL_doutput[3] = loss_scale * dloss_by_dmlp + (float(local_network_output[3]) < 0 ? -output_l1_reg_density : 0.0f);

		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += 4;
	}

	if (exposure_gradient) {
		// Assume symmetric loss
		Array3f dloss_by_dgt = -lg.gradient / xy_pdf;

		if (!train_in_linear_colors) {
			dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
		}

		// 2^exposure * log(2)
		Array3f dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
		atomicAdd(&exposure_gradient[img].x(), dloss_by_dexposure.x());
		atomicAdd(&exposure_gradient[img].y(), dloss_by_dexposure.y());
		atomicAdd(&exposure_gradient[img].z(), dloss_by_dexposure.z());
	}

	if (compacted_numsteps == numsteps && envmap_gradient) {
		Array3f loss_gradient = lg.gradient;
		if (envmap_loss_type != loss_type) {
			loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
		}

		Array3f dloss_by_dbackground = T * loss_gradient;
		if (!train_in_linear_colors) {
			dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
		}

		tcnn::vector_t<tcnn::network_precision_t, 4> dL_denvmap;
		dL_denvmap[0] = loss_scale * dloss_by_dbackground.x();
		dL_denvmap[1] = loss_scale * dloss_by_dbackground.y();
		dL_denvmap[2] = loss_scale * dloss_by_dbackground.z();


		float dloss_by_denvmap_alpha = dloss_by_dbackground.matrix().dot(-pre_envmap_background_color.matrix());

		// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
		dL_denvmap[3] = (tcnn::network_precision_t)0;

		deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
	}
}

__global__ void compute_cam_gradient_train_nerf(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	const Matrix<float, 3, 4>* training_xforms,
	Vector2i resolution,
	bool snap_to_pixel_centers,
	Vector3f* cam_pos_gradient,
	Vector3f* cam_rot_gradient,
	const uint32_t n_training_images,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in,
	uint32_t* __restrict__ numsteps_in,
	const NerfCoordinate* __restrict__ coords,
	const NerfCoordinate* __restrict__ coords_gradient,
	float* __restrict__ distortion_gradient,
	float* __restrict__ distortion_gradient_weight,
	const Vector2i distortion_resolution,
	Vector2f* cam_focal_length_gradient,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const Vector2i error_map_res
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}

	uint32_t base = numsteps_in[i*2+1];
	coords += base;
	coords_gradient += base;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);

	const Matrix<float, 3, 4>& xform = training_xforms[img];

	Ray ray = rays_in[i];
	Ray ray_gradient = { Vector3f::Zero(), Vector3f::Zero() };

	// Compute ray gradient
	for (uint32_t j = 0; j < numsteps; ++j) {
		// pos = ray.o + t * ray.d;

		const Vector3f warped_pos = coords[j].pos.p;
		const Vector3f pos_gradient = coords_gradient[j].pos.p.cwiseProduct(warp_position_derivative(warped_pos, aabb));
		ray_gradient.o += pos_gradient;
		const Vector3f pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = (pos - ray.o).norm();
		const Vector3f dir_gradient = coords_gradient[j].dir.d.cwiseProduct(warp_direction_derivative(coords[j].dir.d));
		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

	// Projection of the raydir gradient onto the plane normal to raydir,
	// because that's the only degree of motion that the raydir has.
	ray_gradient.d -= ray.d * ray_gradient.d.dot(ray.d);

	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
	float xy_pdf = 1.0f;
	Vector2f xy = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, &xy_pdf);

	if (distortion_gradient) {
		// Rotate ray gradient to obtain image plane gradient.
		// This has the effect of projecting the (already projected) ray gradient from the
		// tangent plane of the sphere onto the image plane (which is correct!).
		Vector3f image_plane_gradient = xform.block<3,3>(0,0).inverse() * ray_gradient.d;

		// Splat the resulting 2D image plane gradient into the distortion params
		deposit_image_gradient<2>(image_plane_gradient.head<2>() / xy_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, xy);
	}

	if (cam_pos_gradient) {
		// Atomically reduce the ray gradient into the xform gradient
		#pragma unroll
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / xy_pdf);
		}
	}

	if (cam_rot_gradient) {
		// Rotation is averaged in log-space (i.e. by averaging angle-axes).
		// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
		// orthogonal, leading to the angle_axis magnitude to equal the magnitude
		// of ray_gradient.d.
		Vector3f angle_axis = ray.d.cross(ray_gradient.d);

		// Atomically reduce the ray gradient into the xform gradient
		#pragma unroll
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / xy_pdf);
		}
	}
}

__global__ void shade_kernel_nerf(const uint32_t n_elements, Array4f* rgba, NerfPayload* payloads, ERenderMode render_mode, bool train_in_linear_colors, Array4f* frame_buffer) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	NerfPayload& payload = payloads[i];

	Array4f tmp = rgba[i];

	if (render_mode == ERenderMode::Normals) {
		Array3f n = tmp.head<3>().matrix().normalized().array();
		tmp.head<3>() = (0.5f * n + Array3f::Constant(0.5f)) * tmp.w();
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	}

	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice)) {
		// Accumulate in linear colors
		tmp.head<3>() = srgb_to_linear(tmp.head<3>());
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.w());
}

__global__ void compact_kernel_nerf(
	const uint32_t n_elements,
	Array4f* src_rgba, NerfPayload* src_payloads,
	Array4f* dst_rgba, NerfPayload* dst_payloads,
	Array4f* dst_final_rgba, NerfPayload* dst_final_payloads,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_rgba[idx] = src_rgba[i];
	} else if (src_rgba[i].w() > 0.001f) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_rgba[idx] = src_rgba[i];
	}
}

__global__ void init_rays_with_payload_kernel_nerf(
	uint32_t spp,
	NerfPayload* __restrict__ payloads,
	Vector2i resolution,
	Vector2f focal_length,
	Matrix<float, 3, 4> camera_matrix0,
	Matrix<float, 3, 4> camera_matrix1,
	Vector2f screen_center,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float plane_z,
	float dof,
	CameraDistortion camera_distortion,
	const float* __restrict__ envmap_data,
	const Vector2i envmap_resolution,
	Array4f* __restrict__ framebuffer,
	const float* __restrict__ distortion_data,
	const Vector2i distortion_resolution,
	ERenderMode render_mode
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	if (plane_z < 0) {
		dof = 0.0;
	}

	float ray_time = ld_random_val(spp, idx*72239731);
	Ray ray = pixel_to_ray(
		spp,
		{x, y},
		resolution,
		focal_length,
		camera_matrix0 * ray_time + camera_matrix1 * (1.f - ray_time),
		screen_center,
		snap_to_pixel_centers,
		plane_z,
		dof,
		camera_distortion,
		distortion_data,
		distortion_resolution
	);

	if (plane_z < 0) {
		float n = ray.d.norm();
		NerfPayload& payload = payloads[idx];
		payload.origin = ray.o;
		payload.dir = (1.0f/n) * ray.d;
		payload.t = -plane_z*n;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		return;
	}

	ray.d = ray.d.normalized();

	if (envmap_data) {
		framebuffer[idx] = read_envmap(envmap_data, envmap_resolution, ray.d);
	}

	float t = fmaxf(aabb.ray_intersect(ray.o, ray.d).x(), NERF_RENDERING_NEAR_DISTANCE()) + 1e-6f;

	NerfPayload& payload = payloads[idx];
	if (!aabb.contains(ray.o + ray.d * t)) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (render_mode == ERenderMode::Distortion) {
		if (distortion_data) {
			framebuffer[idx].head<2>() = read_image<2>(distortion_data, distortion_resolution, Vector2f((float)x + 0.5f, (float)y + 0.5f).cwiseQuotient(resolution.cast<float>())) * 50.0f + Vector2f::Constant(0.5f);
		} else {
			framebuffer[idx].head<2>() = Vector2f{0.5f, 0.5f};
		}
		framebuffer[idx].tail<2>() = Vector2f{0.5f, 1.0f};
		payload.origin = ray.o + ray.d * 10000.0f;
		payload.alive = false;
		return;
	}

	payload.origin = ray.o;
	payload.dir = ray.d;
	payload.t = t;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = true;
}

static constexpr float MIN_PDF = 0.01f;

__global__ void construct_cdf_2d(
	uint32_t n_images,
	uint32_t height,
	uint32_t width,
	const float* __restrict__ data,
	float* __restrict__ cdf_x_cond_y,
	float* __restrict__ cdf_y
) {
	const uint32_t y = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t img = threadIdx.y + blockIdx.y * blockDim.y;
	if (y >= height || img >= n_images) return;

	const uint32_t offset_xy = img * height * width + y * width;
	data += offset_xy;
	cdf_x_cond_y += offset_xy;

	float cum = 0;
	for (uint32_t x = 0; x < width; ++x) {
		cum += data[x] + 1e-10f;
		cdf_x_cond_y[x] = cum;
	}

	cdf_y[img * height + y] = cum;
	float norm = __frcp_rn(cum);

	for (uint32_t x = 0; x < width; ++x) {
		cdf_x_cond_y[x] = (1.0f - MIN_PDF) * cdf_x_cond_y[x] * norm + MIN_PDF * (float)(x+1) / (float)width;
	}
}

__global__ void construct_cdf_1d(
	uint32_t n_images,
	uint32_t height,
	float* __restrict__ cdf_y,
	float* __restrict__ cdf_img
) {
	const uint32_t img = threadIdx.x + blockIdx.x * blockDim.x;
	if (img >= n_images) return;

	cdf_y += img * height;

	float cum = 0;
	for (uint32_t y = 0; y < height; ++y) {
		cum += cdf_y[y];
		cdf_y[y] = cum;
	}

	cdf_img[img] = cum;

	float norm = __frcp_rn(cum);
	for (uint32_t y = 0; y < height; ++y) {
		cdf_y[y] = (1.0f - MIN_PDF) * cdf_y[y] * norm + MIN_PDF * (float)(y+1) / (float)height;
	}
}

__global__ void safe_divide(const uint32_t num_elements, float* __restrict__ inout, const float* __restrict__ divisor) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	float local_divisor = divisor[i];
	inout[i] = local_divisor > 0.0f ? (inout[i] / local_divisor) : 0.0f;
}

void Testbed::NerfTracer::init_rays_from_camera(
	GPUMemory<char>& scratch_memory,
	uint32_t spp,
	uint32_t padded_output_width,
	const Vector2i& resolution,
	const Vector2f& focal_length,
	const Matrix<float, 3, 4>& camera_matrix0,
	const Matrix<float, 3, 4>& camera_matrix1,
	Vector2f screen_center,
	bool snap_to_pixel_centers,
	const BoundingBox& render_aabb,
	float plane_z,
	float dof,
	const CameraDistortion& camera_distortion,
	const float* envmap_data,
	const Vector2i& envmap_resolution,
	const float* distortion_data,
	const Vector2i& distortion_resolution,
	Eigen::Array4f* frame_buffer,
	uint8_t *grid,
	int show_accel,
	float cone_angle_constant,
	ERenderMode render_mode,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x() * resolution.y();
	enlarge(scratch_memory, n_pixels, padded_output_width);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x(), threads.x), div_round_up((uint32_t)resolution.y(), threads.y), 1 };
	init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, stream>>>(
		spp,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		screen_center,
		snap_to_pixel_centers,
		render_aabb,
		plane_z,
		dof,
		camera_distortion,
		envmap_data,
		envmap_resolution,
		frame_buffer,
		distortion_data,
		distortion_resolution,
		render_mode
	);

	m_n_rays_initialized = resolution.x() * resolution.y();

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(Array4f), stream));

	linear_kernel(advance_pos_nerf, 0, stream,
		m_n_rays_initialized,
		render_aabb,
		camera_matrix1.col(2),
		focal_length,
		spp,
		m_rays[0].payload,
		grid,
		(show_accel >= 0) ? show_accel : 0,
		cone_angle_constant
	);
}

uint32_t Testbed::NerfTracer::trace(
	NerfNetwork<network_precision_t>& network,
	const BoundingBox& render_aabb,
	const BoundingBox& train_aabb,
	const uint32_t n_training_images,
	const Eigen::Matrix<float, 3, 4>* training_xforms,
	const Vector2f& focal_length,
	float cone_angle_constant,
	const uint8_t* grid,
	ERenderMode render_mode,
	const Eigen::Matrix<float, 3, 4> &camera_matrix,
	float depth_scale,
	int visualized_layer,
	int visualized_dim,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_alpha,
	cudaStream_t stream
) {
	if (m_n_rays_initialized == 0) {
		return 0;
	}

	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter.data(), 0, sizeof(uint32_t), stream));

	uint32_t n_alive = m_n_rays_initialized;
	// m_n_rays_initialized = 0;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
		RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		++double_buffer_index;

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter.data(), 0, sizeof(uint32_t), stream));
			linear_kernel(compact_kernel_nerf, 0, stream,
				n_alive,
				rays_tmp.rgba, rays_tmp.payload,
				rays_current.rgba, rays_current.payload,
				m_rays_hit.rgba, m_rays_hit.payload,
				m_alive_counter.data(), m_hit_counter.data()
			);
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (n_alive == 0) {
			break;
		}

		uint32_t n_steps_between_compaction = tcnn::clamp(m_n_rays_initialized / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

		linear_kernel(generate_next_nerf_network_inputs, 0, stream,
			n_alive,
			render_aabb,
			train_aabb,
			focal_length,
			camera_matrix.col(2),
			rays_current.payload,
			m_network_input,
			n_steps_between_compaction,
			grid,
			(show_accel>=0) ? show_accel : 0,
			cone_angle_constant
		);
		uint32_t n_elements = next_multiple(n_alive*n_steps_between_compaction, BATCH_SIZE_MULTIPLE);
		GPUMatrix<float> positions_matrix((float*)m_network_input, sizeof(NerfCoordinate)/sizeof(float), n_elements);
		GPUMatrix<network_precision_t> rgbsigma_matrix((network_precision_t*)m_network_output, network.padded_output_width(), n_elements);
		network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

		if (render_mode == ERenderMode::Normals) {
			network.input_gradient(stream, 3, positions_matrix, positions_matrix);
		} else if (render_mode == ERenderMode::EncodingVis) {
			network.visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
		}

		linear_kernel(composite_kernel_nerf, 0, stream,
			n_alive,
			i,
			train_aabb,
			n_training_images,
			training_xforms,
			camera_matrix,
			focal_length,
			depth_scale,
			rays_current.rgba,
			rays_current.payload,
			m_network_input,
			m_network_output,
			network.padded_output_width(),
			n_steps_between_compaction,
			render_mode,
			grid,
			rgb_activation,
			density_activation,
			show_accel,
			min_alpha
		);

		i += n_steps_between_compaction;
	}

	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::NerfTracer::enlarge(tcnn::GPUMemory<char>& scratch_memory, size_t n_elements, uint32_t padded_output_width) {
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_MULTIPLE)); // network inference rounds n_elements up to 256, and uses these arrays, so we must do so also.

	auto scratch = scratch_memory.enlarge_and_distribute<
		Array4f, NerfPayload, // m_rays[0]
		Array4f, NerfPayload, // m_rays[1]
		Array4f, NerfPayload, // m_rays_hit

		network_precision_t,
		NerfCoordinate
	>(
		n_elements, n_elements,
		n_elements, n_elements,
		n_elements, n_elements,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), n_elements);
	m_rays[1].set(std::get<2>(scratch), std::get<3>(scratch), n_elements);
	m_rays_hit.set(std::get<4>(scratch), std::get<5>(scratch), n_elements);

	m_network_output = std::get<6>(scratch);
	m_network_input = std::get<7>(scratch);
}

void Testbed::render_nerf(CudaRenderBuffer& render_buffer, const Vector2i& max_res, const Vector2f& focal_length, const Matrix<float, 3, 4>& camera_matrix0, const Matrix<float, 3, 4>& camera_matrix1, const Vector2f& screen_center, cudaStream_t stream) {
	// Reserve the memory for max-res rendering to prevent stuttering
	m_nerf.tracer.enlarge(m_scratch_gpu_memory, max_res.x() * max_res.y(), m_network->padded_output_width());

	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}

	ERenderMode render_mode = m_visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

	m_nerf.tracer.init_rays_from_camera(
		m_scratch_gpu_memory,
		render_buffer.spp(),
		m_network->padded_output_width(),
		render_buffer.resolution(),
		focal_length,
		camera_matrix0,
		camera_matrix1,
		screen_center,
		m_snap_to_pixel_centers,
		m_render_aabb,
		plane_z,
		m_dof,
		m_nerf.render_with_camera_distortion ? m_nerf.training.dataset.camera_distortion : CameraDistortion{},
		m_envmap.envmap->params_inference(),
		m_envmap.resolution,
		m_nerf.render_with_camera_distortion ? m_distortion.map->params_inference() : nullptr,
		m_distortion.resolution,
		render_buffer.frame_buffer(),
		m_nerf.density_grid_bitfield.data(),
		m_nerf.show_accel,
		m_nerf.cone_angle_constant,
		render_mode,
		stream
	);

	uint32_t n_hit;
	if (m_render_mode == ERenderMode::Slice) {
		n_hit = m_nerf.tracer.n_rays_initialized();
	} else {
		float depth_scale = 1.f/m_nerf.training.dataset.scale;
		n_hit = m_nerf.tracer.trace(
			*m_nerf_network,
			m_render_aabb,
			m_aabb,
			m_nerf.training.n_images,
			m_nerf.training.transforms.data(),
			focal_length,
			m_nerf.cone_angle_constant,
			m_nerf.density_grid_bitfield.data(),
			render_mode, camera_matrix1, depth_scale, m_visualized_layer, m_visualized_dimension,
			m_nerf.rgb_activation, m_nerf.density_activation, m_nerf.show_accel, m_nerf.rendering_min_alpha,
			stream
		);
	}
	RaysNerfSoa& rays_hit = m_render_mode == ERenderMode::Slice ? m_nerf.tracer.rays_init() : m_nerf.tracer.rays_hit();

	if (m_render_mode == ERenderMode::Slice) {
		// Store colors in the normal buffer
		uint32_t n_elements = next_multiple(n_hit, BATCH_SIZE_MULTIPLE);

		m_nerf.vis_input.enlarge(n_elements);
		m_nerf.vis_rgba.enlarge(n_elements);
		linear_kernel(generate_nerf_network_inputs_at_current_position, 0, stream, n_hit, m_aabb, rays_hit.payload, m_nerf.vis_input.data());

		GPUMatrix<float> positions_matrix((float*)m_nerf.vis_input.data(), sizeof(NerfCoordinate)/sizeof(float), n_elements);
		GPUMatrix<float> rgbsigma_matrix((float*)m_nerf.vis_rgba.data(), 4, n_elements);

		if (m_visualized_dimension == -1) {
			m_network->inference(stream, positions_matrix, rgbsigma_matrix);
			linear_kernel(compute_nerf_density, 0, stream, n_hit, m_nerf.vis_rgba.data(), m_nerf.rgb_activation, m_nerf.density_activation);
		} else {
			m_network->visualize_activation(stream, m_visualized_layer, m_visualized_dimension, positions_matrix, rgbsigma_matrix);
		}

		linear_kernel(shade_kernel_nerf, 0, stream,
			n_hit,
			m_nerf.vis_rgba.data(),
			rays_hit.payload,
			m_render_mode,
			m_nerf.training.linear_colors,
			render_buffer.frame_buffer()
		);
		return;
	}

	linear_kernel(shade_kernel_nerf, 0, stream,
		n_hit,
		rays_hit.rgba,
		rays_hit.payload,
		m_render_mode,
		m_nerf.training.linear_colors,
		render_buffer.frame_buffer()
	);

	if (render_mode == ERenderMode::Cost) {
		std::vector<NerfPayload> payloads_final_cpu(n_hit);
		CUDA_CHECK_THROW(cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(NerfPayload), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		size_t total_n_steps = 0;
		for (uint32_t i = 0; i < n_hit; ++i) {
			total_n_steps += payloads_final_cpu[i].n_steps;
		}
		tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps/(float)n_hit);
	}
}

void Testbed::update_nerf_focal_lengths() {
	std::vector<Vector2f> updated_focal_lengths;
	updated_focal_lengths.reserve(m_nerf.training.n_images);
	for (uint32_t i = 0; i < m_nerf.training.n_images; ++i) {
		updated_focal_lengths.push_back(m_nerf.training.dataset.focal_lengths[i] + m_nerf.training.cam_focal_length_offset.variable());
	}
	m_nerf.training.focal_lengths_gpu.resize_and_copy_from_host(updated_focal_lengths);
}

void Testbed::update_nerf_transforms() {
	m_nerf.training.transforms.resize(m_nerf.training.n_images);
	for (uint32_t i = 0; i < m_nerf.training.n_images; ++i) {
		auto xform = m_nerf.training.dataset.xforms[i];

		Vector3f rot = m_nerf.training.cam_rot_offset[i].variable();
		float angle = rot.norm();
		rot /= angle;
		if (angle > 0) {
			xform.block<3,3>(0,0) = AngleAxisf(angle, rot) * xform.block<3,3>(0,0);
		}

		xform.col(3) += m_nerf.training.cam_pos_offset[i].variable();

		m_nerf.training.transforms[i] = xform;
	}

	m_nerf.training.transforms_gpu.resize_and_copy_from_host(m_nerf.training.transforms);
}

void Testbed::load_nerf() {
	if (!m_data_path.empty()) {
		std::vector<fs::path> json_paths;
		if (m_data_path.is_directory()) {
			for (const auto& path : fs::directory{m_data_path}) {
				if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
					json_paths.emplace_back(path);
				}
			}
		} else if (equals_case_insensitive(m_data_path.extension(), "msgpack")) {
			load_snapshot(m_data_path.str());
			set_train(false);
			return;
		} else if (equals_case_insensitive(m_data_path.extension(), "json")) {
			json_paths.emplace_back(m_data_path);
		} else {
			throw std::runtime_error{"NeRF data path must either be a json file or a directory containing json files."};
		}

		m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen);
	}

	m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	m_nerf.training.image_resolution = m_nerf.training.dataset.image_resolution;
	m_nerf.training.n_images = (int)m_nerf.training.dataset.n_images;

	m_nerf.training.focal_lengths = m_nerf.training.dataset.focal_lengths;
	m_nerf.training.focal_lengths_gpu.resize_and_copy_from_host(m_nerf.training.focal_lengths);

	m_nerf.training.cam_pos_gradient.resize(m_nerf.training.n_images, Vector3f::Zero());
	m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

	m_nerf.training.cam_pos_offset.resize(m_nerf.training.n_images, AdamOptimizer<Vector3f>(1e-4f));
	m_nerf.training.cam_rot_offset.resize(m_nerf.training.n_images, RotationAdamOptimizer(1e-4f));
	m_nerf.training.cam_focal_length_offset = AdamOptimizer<Vector2f>(1e-4f);

	m_nerf.training.cam_rot_gradient.resize(m_nerf.training.n_images, Vector3f::Zero());
	m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

	m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.n_images, Array3f::Zero());
	m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
	m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

	m_nerf.training.cam_focal_length_gradient = Vector2f::Zero();
	m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

	if (m_nerf.training.dataset.rays_data.data()) {
		m_nerf.training.near_distance = 0.0f;
		// m_nerf.training.optimize_exposure = true;
	}

	// Uncomment the following line to see how the network learns distortion from scratch rather than
	// starting from the distortion that's described by the training data.
	// m_nerf.training.dataset.camera_distortion = {};

	// Perturbation of the training cameras -- for debugging the online extrinsics learning code
	// for (uint32_t i = 0; i < m_nerf.training.n_images; ++i) {
	// 	Vector3f rot = random_val_3d(i+1) * 0.01f;
	// 	float angle = rot.norm();
	// 	rot /= angle;
	// 	m_nerf.training.dataset.xforms[i].block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].block<3,3>(0,0);
	// 	m_nerf.training.dataset.xforms[i].col(3) += random_val_3d(i+1+m_nerf.training.n_images) * 0.01f;
	// }

	update_nerf_transforms();

	m_aabb = BoundingBox{Vector3f::Constant(0.5f), Vector3f::Constant(0.5f)};
	m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_nerf.training.dataset.aabb_scale));
	m_raw_aabb = m_aabb;
	m_render_aabb = m_aabb;
	if (!m_nerf.training.dataset.render_aabb.is_empty()) {
		m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
	}

	m_nerf.max_cascade = 0;
	while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
		++m_nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = m_nerf.training.dataset.up;
}

void Testbed::update_density_grid_nerf(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES();

	m_nerf.density_grid.enlarge(n_elements);

	const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

	auto scratch = m_scratch_gpu_memory.enlarge_and_distribute<
		NerfPosition,       // positions at which the NN will be queried for density evaluation
		uint32_t,           // indices of corresponding density grid cells
		float,              // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
		network_precision_t // output of the MLP before being converted to densities.
	>(n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

	NerfPosition* density_grid_positions = std::get<0>(scratch);
	uint32_t* density_grid_indices = std::get<1>(scratch);
	float* density_grid_tmp = std::get<2>(scratch);
	network_precision_t* mlp_out = std::get<3>(scratch);

	if (m_training_step == 0) {
		m_nerf.density_grid_ema_step = 0;
		// Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
		if (!m_nerf.training.dataset.rays_data.data()) {
			linear_kernel(mark_untrained_density_grid, 0, stream, n_elements, m_nerf.density_grid.data(),
				m_nerf.training.n_images,
				m_nerf.training.focal_lengths_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
				m_nerf.training.image_resolution
			);
		} else {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
		}
	}

	uint32_t n_steps = 1;
	for (uint32_t i = 0; i < n_steps; ++i) {
		CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float)*n_elements, stream));

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_uniform_density_grid_samples,
			m_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions,
			density_grid_indices,
			m_nerf.max_cascade+1,
			-0.01f
		);
		m_rng.advance();

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_nonuniform_density_grid_samples,
			m_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions+n_uniform_density_grid_samples,
			density_grid_indices+n_uniform_density_grid_samples,
			m_nerf.max_cascade+1,
			NERF_MIN_OPTICAL_THICKNESS()
		);
		m_rng.advance();

		GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, n_density_grid_samples);
		m_nerf_network->density(stream, {(float*)density_grid_positions, sizeof(NerfPosition)/sizeof(float)}, rgbsigma_matrix, false);

		linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream, n_density_grid_samples, density_grid_indices, padded_output_width, mlp_out, density_grid_tmp, m_nerf.rgb_activation, m_nerf.density_activation);
		linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, decay, m_nerf.density_grid_ema_step, m_nerf.density_grid.data(), density_grid_tmp);

		++m_nerf.density_grid_ema_step;
	}

	update_density_grid_mean_and_bitfield(stream);
}

void Testbed::update_density_grid_mean_and_bitfield(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();

	size_t size_including_mips = grid_mip_offset(NERF_CASCADES())/8;
	m_nerf.density_grid_bitfield.enlarge(size_including_mips);
	m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
	reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

	linear_kernel(grid_to_bitfield, 0, stream, n_elements/8 * NERF_CASCADES(), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());

	for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
		linear_kernel(bitfield_max_pool, 0, stream, n_elements/64, m_nerf.get_density_grid_bitfield_mip(level-1), m_nerf.get_density_grid_bitfield_mip(level));
	}
}

void Testbed::train_nerf(uint32_t target_batch_size, uint32_t n_training_steps, cudaStream_t stream) {
	if (m_nerf.training.include_sharpness_in_error) {
		size_t n_cells = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES();
		if (m_nerf.training.sharpness_grid.size() < n_cells) {
			m_nerf.training.sharpness_grid.enlarge(NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES());
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		}

		if (m_training_step == 0) {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		} else {
			linear_kernel(decay_sharpness_grid_nerf, 0, stream, m_nerf.training.sharpness_grid.size(), 0.95f, m_nerf.training.sharpness_grid.data());
		}
	}

	// number of steps each ray took, and the first offset of samples
	m_nerf.training.numsteps_counter.enlarge(n_training_steps);
	m_nerf.training.numsteps_counter_compacted.enlarge(n_training_steps);

	m_nerf.training.loss.enlarge(m_nerf.training.rays_per_batch * n_training_steps);

	uint32_t* counter = m_nerf.training.numsteps_counter.data();
	uint32_t* compacted_counter = m_nerf.training.numsteps_counter_compacted.data();

	CUDA_CHECK_THROW(cudaMemsetAsync(counter, 0, sizeof(uint32_t)*n_training_steps, stream)); // clear the counter in the first slot
	CUDA_CHECK_THROW(cudaMemsetAsync(compacted_counter, 0, sizeof(uint32_t)*n_training_steps, stream)); // clear the counter in the first slot
	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.loss.data(), 0, sizeof(float)*m_nerf.training.rays_per_batch*n_training_steps, stream));

	if (m_nerf.training.n_steps_since_cam_update == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(), 0, m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradient_weights(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(), 0, m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), stream));
	}

	if (m_nerf.training.n_steps_since_error_map_update == 0) {
		uint32_t n_samples_per_image = (m_nerf.training.n_steps_between_error_map_updates * m_nerf.training.rays_per_batch) / m_nerf.training.dataset.n_images;
		m_nerf.training.error_map.resolution = Vector2i::Constant((int)(std::sqrt(std::sqrt((float)n_samples_per_image)) * 3.5f)).cwiseMin(m_nerf.training.image_resolution);
		m_nerf.training.error_map.data.resize(m_nerf.training.error_map.resolution.prod() * m_nerf.training.dataset.n_images);
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.data.data(), 0, m_nerf.training.error_map.data.get_bytes(), stream));
	}

	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
	if (envmap_gradient) {
		CUDA_CHECK_THROW(cudaMemsetAsync(envmap_gradient, 0, sizeof(float)*m_envmap.envmap->n_params(), stream));
	}

	for (uint32_t i = 0; i < n_training_steps; ++i) {
		train_nerf_step(target_batch_size, m_nerf.training.rays_per_batch, counter+i, compacted_counter+i, m_nerf.training.loss.data()+i*m_nerf.training.rays_per_batch, m_training_stream);
	}

	if (envmap_gradient) {
		m_envmap.trainer->optimizer_step(stream, LOSS_SCALE*(float)n_training_steps);
	}

	std::vector<uint32_t> counter_cpu(n_training_steps);
	std::vector<uint32_t> compacted_counter_cpu(n_training_steps);
	m_nerf.training.numsteps_counter.copy_to_host(counter_cpu, n_training_steps);
	m_nerf.training.numsteps_counter_compacted.copy_to_host(compacted_counter_cpu, n_training_steps);

	m_nerf.training.measured_batch_size = 0;
	m_nerf.training.measured_batch_size_before_compaction = 0;

	for (uint32_t i = 0; i < n_training_steps; ++i) {
		if (counter_cpu[i] == 0 || compacted_counter_cpu[i] == 0) {
			m_train = false;
			tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
			return;
		}

		m_nerf.training.measured_batch_size_before_compaction += counter_cpu[i];
		m_nerf.training.measured_batch_size += compacted_counter_cpu[i];
	}

	m_nerf.training.measured_batch_size_before_compaction /= n_training_steps;
	m_nerf.training.measured_batch_size /= n_training_steps;

	m_loss_scalar = reduce_sum(m_nerf.training.loss.data(), m_nerf.training.rays_per_batch * n_training_steps, stream) / (float)(n_training_steps);
	m_loss_scalar *= (float)m_nerf.training.measured_batch_size / (float)target_batch_size;
	update_loss_graph();

	m_nerf.training.rays_per_batch = (uint32_t)((float)m_nerf.training.rays_per_batch * (float)target_batch_size / (float)m_nerf.training.measured_batch_size);
	m_nerf.training.rays_per_batch = std::min(next_multiple(m_nerf.training.rays_per_batch, BATCH_SIZE_MULTIPLE), 1u << 18);

	// Compute CDFs from the error map
	m_nerf.training.n_steps_since_error_map_update += n_training_steps;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;
	if (accumulate_error && m_nerf.training.n_steps_since_error_map_update >= m_nerf.training.n_steps_between_error_map_updates) {
		m_nerf.training.error_map.cdf_resolution = m_nerf.training.error_map.resolution;
		m_nerf.training.error_map.cdf_x_cond_y.resize(m_nerf.training.error_map.cdf_resolution.prod() * m_nerf.training.dataset.n_images);
		m_nerf.training.error_map.cdf_y.resize(m_nerf.training.error_map.cdf_resolution.y() * m_nerf.training.dataset.n_images);
		m_nerf.training.error_map.cdf_img.resize(m_nerf.training.dataset.n_images);

		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_x_cond_y.data(), 0, m_nerf.training.error_map.cdf_x_cond_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_y.data(), 0, m_nerf.training.error_map.cdf_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_img.data(), 0, m_nerf.training.error_map.cdf_img.get_bytes(), stream));

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)m_nerf.training.error_map.cdf_resolution.y(), threads.x), div_round_up((uint32_t)m_nerf.training.dataset.n_images, threads.y), 1 };
		construct_cdf_2d<<<blocks, threads, 0, stream>>>(
			m_nerf.training.dataset.n_images, m_nerf.training.error_map.cdf_resolution.y(), m_nerf.training.error_map.cdf_resolution.x(),
			m_nerf.training.error_map.data.data(),
			m_nerf.training.error_map.cdf_x_cond_y.data(),
			m_nerf.training.error_map.cdf_y.data()
		);
		linear_kernel(construct_cdf_1d, 0, stream,
			m_nerf.training.dataset.n_images,
			m_nerf.training.error_map.cdf_resolution.y(),
			m_nerf.training.error_map.cdf_y.data(),
			m_nerf.training.error_map.cdf_img.data()
		);

		// Compute image CDF on the CPU. It's single-threaded anyway. No use parallelizing.
		m_nerf.training.error_map.pmf_img_cpu.resize(m_nerf.training.error_map.cdf_img.size());
		m_nerf.training.error_map.cdf_img.copy_to_host(m_nerf.training.error_map.pmf_img_cpu);
		std::vector<float> cdf_img_cpu = m_nerf.training.error_map.pmf_img_cpu; // Copy unnormalized PDF into CDF buffer
		float cum = 0;
		for (float& f : cdf_img_cpu) {
			cum += f;
			f = cum;
		}
		float norm = 1.0f / cum;
		for (size_t i = 0; i < cdf_img_cpu.size(); ++i) {
			constexpr float MIN_PMF = 0.1f;
			m_nerf.training.error_map.pmf_img_cpu[i] = (1.0f - MIN_PMF) * m_nerf.training.error_map.pmf_img_cpu[i] * norm + MIN_PMF / (float)m_nerf.training.dataset.n_images;
			cdf_img_cpu[i] = (1.0f - MIN_PMF) * cdf_img_cpu[i] * norm + MIN_PMF * (float)(i+1) / (float)m_nerf.training.dataset.n_images;
		}
		m_nerf.training.error_map.cdf_img.copy_from_host(cdf_img_cpu);

		// Reset counters and decrease update rate.
		m_nerf.training.n_steps_since_error_map_update = 0;
		m_nerf.training.n_rays_since_error_map_update = 0;
		m_nerf.training.error_map.is_cdf_valid = true;

		m_nerf.training.n_steps_between_error_map_updates = (uint32_t)(m_nerf.training.n_steps_between_error_map_updates * 1.5f);
	}

	// Get extrinsics gradients
	m_nerf.training.n_steps_since_cam_update += n_training_steps;

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length || m_nerf.training.optimize_exposure;
	if (train_camera && m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates) {
		float per_camera_loss_scale = (float)m_nerf.training.n_images / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;

		if (m_nerf.training.optimize_extrinsics) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// Optimization step
			for (uint32_t i = 0; i < m_nerf.training.n_images; ++i) {
				Vector3f pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
				Vector3f rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

				float l2_reg = 0.01f;
				pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
				rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

				m_nerf.training.cam_pos_offset[i].set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_training_step / 2048)), m_optimizer->learning_rate()/1000.0f));
				m_nerf.training.cam_rot_offset[i].set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_training_step / 2048)), m_optimizer->learning_rate()/1000.0f));

				m_nerf.training.cam_pos_offset[i].step(pos_gradient);
				m_nerf.training.cam_rot_offset[i].step(rot_gradient);
			}

			update_nerf_transforms();
		}

		if (m_nerf.training.optimize_distortion) {
			linear_kernel(safe_divide, 0, stream,
				m_distortion.map->n_params(),
				m_distortion.map->gradients(),
				m_distortion.map->gradient_weights()
			);
			m_distortion.trainer->optimizer_step(stream, LOSS_SCALE*(float)m_nerf.training.n_steps_between_cam_updates);
		}

		if (m_nerf.training.optimize_focal_length) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_focal_length_gradient.data(),m_nerf.training.cam_focal_length_gradient_gpu.data(),m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(),cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			Vector2f focal_length_gradient = m_nerf.training.cam_focal_length_gradient * per_camera_loss_scale;
			float l2_reg = 0.01f;
			focal_length_gradient += m_nerf.training.cam_focal_length_offset.variable() * l2_reg;
			m_nerf.training.cam_focal_length_offset.set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_training_step / 2048)),m_optimizer->learning_rate() / 1000.0f));
			m_nerf.training.cam_focal_length_offset.step(focal_length_gradient);
			update_nerf_focal_lengths();
		}

		if (m_nerf.training.optimize_exposure) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gradient.data(), m_nerf.training.cam_exposure_gradient_gpu.data(), m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			Array3f mean_exposure = Array3f::Constant(0.0f);

			// Optimization step
			for (uint32_t i = 0; i < m_nerf.training.n_images; ++i) {
				Array3f gradient = m_nerf.training.cam_exposure_gradient[i] * per_camera_loss_scale;

				float l2_reg = 0.00f;
				gradient += m_nerf.training.cam_exposure[i].variable() * l2_reg;

				m_nerf.training.cam_exposure[i].set_learning_rate(m_optimizer->learning_rate());
				m_nerf.training.cam_exposure[i].step(gradient);

				mean_exposure += m_nerf.training.cam_exposure[i].variable();
			}

			mean_exposure /= m_nerf.training.n_images;

			// Renormalize
			std::vector<Array3f> cam_exposures(m_nerf.training.n_images);
			for (uint32_t i = 0; i < m_nerf.training.n_images; ++i) {
				cam_exposures[i] = m_nerf.training.cam_exposure[i].variable() -= mean_exposure;
			}

			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gpu.data(), cam_exposures.data(), m_nerf.training.cam_exposure_gpu.get_bytes(), cudaMemcpyHostToDevice, stream));
		}

		m_nerf.training.n_steps_since_cam_update = 0;
	}
}

void Testbed::train_nerf_step(uint32_t target_batch_size, uint32_t n_rays_per_batch, uint32_t* counter, uint32_t* compacted_counter, float* loss, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case

	m_nerf.training.ray_counter.enlarge(1);

	auto scratch = m_scratch_gpu_memory.enlarge_and_distribute<
		uint32_t, // ray_indices
		Ray, // rays
		uint32_t, // numsteps
		NerfCoordinate, // coords
		float, // max_level
		network_precision_t, // mlp_out
		network_precision_t, // mlp_out_trimmed
		network_precision_t, // dloss_dmlp_out
		NerfCoordinate, // coords_compacted
		NerfCoordinate, // coords_gradient
		float // max_level_compacted
	>(
		n_rays_per_batch,
		n_rays_per_batch,
		n_rays_per_batch * 2,
		max_samples,
		max_samples,
		std::max(target_batch_size, max_samples) * padded_output_width,
		std::max(target_batch_size, max_samples) * 4,
		target_batch_size * padded_output_width,
		target_batch_size,
		target_batch_size,
		target_batch_size
	);

	// TODO: C++17 structured binding
	uint32_t* ray_indices = std::get<0>(scratch);
	Ray* rays = std::get<1>(scratch);
	uint32_t* numsteps = std::get<2>(scratch);
	NerfCoordinate* coords = std::get<3>(scratch);
	float* max_level = std::get<4>(scratch);
	network_precision_t* mlp_out = std::get<5>(scratch);
	network_precision_t* mlp_out_trimmed = std::get<6>(scratch);
	network_precision_t* dloss_dmlp_out = std::get<7>(scratch);
	NerfCoordinate* coords_compacted = std::get<8>(scratch);
	NerfCoordinate* coords_gradient = std::get<9>(scratch);
	float* max_level_compacted = std::get<10>(scratch);

	uint32_t max_inference;
	if (m_nerf.training.measured_batch_size_before_compaction == 0) {
		m_nerf.training.measured_batch_size_before_compaction = max_inference = max_samples;
	} else {
		max_inference = next_multiple(std::min(m_nerf.training.measured_batch_size_before_compaction, max_samples), BATCH_SIZE_MULTIPLE);
	}

	GPUMatrix<float> coords_matrix((float*)coords, sizeof(NerfCoordinate)/sizeof(float), max_inference);
	GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, sizeof(NerfCoordinate)/sizeof(float), target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_training_step == 0) {
		m_nerf.training.n_rays_total = 0;
	}

	uint32_t n_rays_total = m_nerf.training.n_rays_total;
	m_nerf.training.n_rays_total += n_rays_per_batch;
	m_nerf.training.n_rays_since_error_map_update += n_rays_per_batch;

	// If we have an envmap, prepare its gradient buffer
	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;

	bool sample_focal_plane_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_focal_plane_proportional_to_error;
	bool sample_image_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
	bool include_sharpness_in_error = m_nerf.training.include_sharpness_in_error;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.ray_counter.data(), 0, sizeof(uint32_t), stream));

	linear_kernel(generate_training_samples_nerf, 0, stream,
		n_rays_per_batch,
		m_aabb,
		max_inference,
		n_rays_total,
		m_rng,
		m_nerf.training.dataset.rays_data.data(),
		m_nerf.training.ray_counter.data(),
		counter,
		ray_indices,
		rays,
		numsteps,
		coords,
		m_nerf.training.image_resolution,
		m_nerf.training.n_images,
		m_nerf.training.dataset.principal_point,
		m_nerf.training.focal_lengths_gpu.data(),
		m_nerf.training.transforms_gpu.data(),
		m_nerf.training.dataset.camera_distortion,
		m_nerf.density_grid_bitfield.data(),
		m_max_level_rand_training,
		max_level,
		m_nerf.training.snap_to_pixel_centers,
		m_nerf.training.train_envmap,
		m_nerf.cone_angle_constant,
		m_distortion.map->params(),
		m_distortion.resolution,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
		sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
		m_nerf.training.error_map.cdf_resolution,
		m_nerf.training.near_distance,
		m_nerf.training.dataset.images_data.data()
	);

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
	}

	m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);
	if (padded_output_width != 4) {
		linear_kernel(tcnn::trim<network_precision_t>, 0, stream, max_inference*4, padded_output_width, 4, mlp_out, mlp_out_trimmed);
		mlp_out = mlp_out_trimmed;
	}

	if (hg_enc) {
		hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
	}

	linear_kernel(compute_loss_kernel_train_nerf, 0, stream,
		n_rays_per_batch,
		m_aabb,
		n_rays_total,
		m_rng,
		target_batch_size,
		m_nerf.training.ray_counter.data(),
		LOSS_SCALE,
		padded_output_width,
		m_envmap.envmap->params(),
		envmap_gradient,
		m_envmap.resolution,
		m_envmap.loss_type,
		m_background_color.head<3>(),
		m_color_space,
		m_nerf.training.random_bg_color,
		m_nerf.training.linear_colors,
		m_nerf.training.dataset.images_data.data(),
		m_nerf.training.n_images,
		m_nerf.training.image_resolution,
		mlp_out,
		compacted_counter,
		ray_indices,
		rays,
		numsteps,
		coords,
		coords_compacted,
		dloss_dmlp_out,
		m_nerf.training.loss_type,
		loss,
		m_max_level_rand_training,
		max_level_compacted,
		m_nerf.rgb_activation,
		m_nerf.density_activation,
		m_nerf.training.snap_to_pixel_centers,
		accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
		sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
		m_nerf.training.error_map.resolution,
		m_nerf.training.error_map.cdf_resolution,
		include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
		m_nerf.training.dataset.sharpness_resolution,
		m_nerf.training.sharpness_grid.data(),
		m_nerf.density_grid.data(),
		m_nerf.density_grid_mean.data(),
		m_nerf.training.cam_exposure_gpu.data(),
		m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr
	);

	fill_rollover_and_rescale<network_precision_t><<<n_blocks_linear(target_batch_size*padded_output_width), n_threads_linear, 0, stream>>>(
		target_batch_size, padded_output_width, compacted_counter, dloss_dmlp_out
	);
	fill_rollover<NerfCoordinate><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, compacted_counter, coords_compacted
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, compacted_counter, max_level_compacted
	);

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length;
	m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, train_camera);

	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, sizeof(NerfCoordinate)/sizeof(float), target_batch_size);
	m_network->backward(stream, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, train_camera ? &coords_gradient_matrix : nullptr);

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf, 0, stream,
			n_rays_per_batch,
			n_rays_total,
			m_rng,
			m_aabb,
			m_nerf.training.ray_counter.data(),
			m_nerf.training.transforms_gpu.data(),
			m_nerf.training.image_resolution,
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_pos_gradient_gpu.data() : nullptr,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_rot_gradient_gpu.data() : nullptr,
			m_nerf.training.n_images,
			ray_indices,
			rays,
			numsteps,
			coords_compacted,
			coords_gradient,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
			m_distortion.resolution,
			m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution
		);
	}

	m_rng.advance();
	m_trainer->optimizer_step(stream, LOSS_SCALE);

	++m_training_step;

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}
}

void Testbed::training_prep_nerf(uint32_t batch_size, uint32_t n_training_steps, cudaStream_t stream) {
	float alpha = std::pow(m_nerf.training.density_grid_decay, n_training_steps / 16.0f);
	uint32_t n_cascades = m_nerf.max_cascade+1;
	if (m_training_step < 256) {
		update_density_grid_nerf(alpha, NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE()*n_cascades, 0, stream);
	} else {
		update_density_grid_nerf(alpha, NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE()/4*n_cascades, NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE()/4*n_cascades, stream);
	}
}

void Testbed::optimise_mesh_step(uint32_t N_STEPS) {
	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	if (!n_verts) {
		return;
	}

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();
	GPUMemory<NerfCoordinate> coords(n_verts);
	GPUMemory<network_precision_t> mlp_out(n_verts * padded_output_width);

	GPUMatrix<float> positions_matrix((float*)coords.data(), sizeof(NerfCoordinate)/sizeof(float), n_verts);
	GPUMatrix<network_precision_t> density_matrix(mlp_out.data(), padded_output_width, n_verts);

	for (uint32_t i = 0; i < N_STEPS; ++i) {
		linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_inference_stream, n_verts, m_aabb, m_mesh.verts.data(), coords.data());

		// For each optimizer step, we need the density at the given pos...
		m_nerf_network->density(m_inference_stream, positions_matrix, density_matrix);
		// ...as well as the input gradient w.r.t. density, which we will store in the nerf coords.
		m_nerf_network->input_gradient(m_inference_stream, 3, positions_matrix, positions_matrix);
		// and the 1ring centroid for laplacian smoothing
		compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);

		// With these, we can compute a gradient that points towards the threshold-crossing of density...
		compute_mesh_opt_gradients(m_mesh.thresh, m_mesh.verts, m_mesh.vert_normals, m_mesh.verts_smoothed, padded_output_width, mlp_out.data(), sizeof(NerfCoordinate)/sizeof(float), (const float*)coords.data(), m_mesh.verts_gradient, m_mesh.smooth_amount, m_mesh.density_amount, m_mesh.inflate_amount);

		// ...that we can pass to the optimizer.
		m_mesh.verts_optimizer->step(m_inference_stream, 1.0f, (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
	}
}

void Testbed::compute_mesh_vertex_colors() {
	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	if (!n_verts) {
		return;
	}

	m_mesh.vert_colors.resize(n_verts);
	const uint32_t padded_output_width = m_network->padded_output_width();
	GPUMemory<NerfCoordinate> coords(n_verts);
	GPUMemory<float> mlp_out(n_verts* padded_output_width);

	GPUMatrix<float> positions_matrix((float*)coords.data(), sizeof(NerfCoordinate)/sizeof(float), n_verts);
	GPUMatrix<float> color_matrix(mlp_out.data(), padded_output_width, n_verts);
	linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_inference_stream, n_verts, m_aabb, m_mesh.verts.data(), coords.data());
	m_network->inference(m_inference_stream, positions_matrix, color_matrix);
	linear_kernel(extract_srgb_with_activation, 0, m_inference_stream, n_verts*3, 3, mlp_out.data(), (float*)m_mesh.vert_colors.data(), m_nerf.rgb_activation, m_nerf.training.linear_colors);
}

GPUMemory<float> Testbed::get_density_on_grid(Vector3i res3d, const BoundingBox& aabb) {
	const uint32_t n_elements = (res3d.x()*res3d.y()*res3d.z());
	GPUMemory<float> density(n_elements);

	const uint32_t batch_size = std::min(n_elements, 1u<<20);
	bool nerf_mode = m_testbed_mode == ETestbedMode::Nerf;

	const uint32_t padded_output_width = nerf_mode ? m_nerf_network->padded_density_output_width() : m_network->padded_output_width();

	auto scratch = m_scratch_gpu_memory.enlarge_and_distribute<
		NerfPosition,
		network_precision_t
	>(n_elements, batch_size * padded_output_width);

	NerfPosition* positions = std::get<0>(scratch);
	network_precision_t* mlp_out = std::get<1>(scratch);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res3d.x(), threads.x), div_round_up((uint32_t)res3d.y(), threads.y), div_round_up((uint32_t)res3d.z(), threads.z) };

	BoundingBox unit_cube = BoundingBox{Vector3f::Zero(), Vector3f::Ones()};
	generate_grid_samples_nerf_uniform<<<blocks, threads, 0, m_inference_stream>>>(res3d, m_nerf.density_grid_ema_step, aabb, nerf_mode ? m_aabb : unit_cube , positions);

	// Only process 1m elements at a time
	for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
		uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

		GPUMatrix<network_precision_t> density_matrix(mlp_out, padded_output_width, local_batch_size);
		if (nerf_mode)
			m_nerf_network->density(m_inference_stream, {(float*)(positions + offset), sizeof(NerfPosition)/sizeof(float)}, density_matrix);
		else {
			GPUMatrix<float> positions_matrix((float*)(positions + offset), sizeof(NerfPosition)/sizeof(float), local_batch_size);
			m_network->inference_mixed_precision(m_inference_stream, positions_matrix, density_matrix);
		}
		linear_kernel(grid_samples_half_to_float, 0, m_inference_stream,
			local_batch_size,
			m_aabb,
			density.data() + offset , //+ axis_step * n_elements,
			padded_output_width,
			mlp_out,
			m_nerf.density_activation,
			positions + offset,
			nerf_mode ? m_nerf.density_grid.data() : nullptr
		);
	}

	return density;
}

GPUMemory<Eigen::Array4f> Testbed::get_rgba_on_grid(Vector3i res3d, Eigen::Vector3f ray_dir) {
	const uint32_t n_elements = (res3d.x()*res3d.y()*res3d.z());
	GPUMemory<Eigen::Array4f> rgba(n_elements);
	GPUMemory<NerfCoordinate> positions(n_elements);
	const uint32_t batch_size = std::min(n_elements, 1u<<20);

	// generate inputs
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res3d.x(), threads.x), div_round_up((uint32_t)res3d.y(), threads.y), div_round_up((uint32_t)res3d.z(), threads.z) };
	generate_grid_samples_nerf_uniform_dir<<<blocks, threads, 0, m_inference_stream>>>(res3d, m_nerf.density_grid_ema_step, m_render_aabb, m_aabb, ray_dir, positions.data());

	// Only process 1m elements at a time
	for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
		uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

		// run network
		GPUMatrix<float> positions_matrix((float*) (positions.data() + offset), sizeof(NerfCoordinate)/sizeof(float), local_batch_size);
		GPUMatrix<float> rgbsigma_matrix((float*) (rgba.data() + offset), 4, local_batch_size);
		m_network->inference(m_inference_stream, positions_matrix, rgbsigma_matrix);

		// convert network output to RGBA (in place)
		linear_kernel(compute_nerf_density, 0, m_inference_stream, local_batch_size, rgba.data() + offset, m_nerf.rgb_activation, m_nerf.density_activation);
	}
	return rgba;
}

int Testbed::marching_cubes(Vector3i res3d, const BoundingBox& aabb, float thresh) {
	res3d.x() = next_multiple((unsigned int)res3d.x(), 16u);
	res3d.y() = next_multiple((unsigned int)res3d.y(), 16u);
	res3d.z() = next_multiple((unsigned int)res3d.z(), 16u);

	GPUMemory<float> density = get_density_on_grid(res3d, aabb);
	marching_cubes_gpu(m_scratch_gpu_memory, m_render_aabb, res3d, thresh, density, m_mesh.verts, m_mesh.indices);

	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	m_mesh.verts_gradient.resize(n_verts);

	m_mesh.trainable_verts = std::make_shared<TrainableBuffer<3, 1, float>>(Matrix<int, 1, 1>{(int)n_verts});
	m_mesh.verts_gradient.copy_from_device(m_mesh.verts); // Make sure the vertices don't get destroyed in the initialization

	pcg32 rnd{m_seed};
	m_mesh.trainable_verts->initialize_params(rnd, (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
	m_mesh.verts.copy_from_device(m_mesh.verts_gradient);

	m_mesh.verts_optimizer.reset(create_optimizer<float>({
		{"otype", "Adam"},
		{"learning_rate", 1e-4},
		{"beta1", 0.9f},
		{"beta2", 0.99f},
	}));

	m_mesh.verts_optimizer->allocate(m_mesh.trainable_verts);

	compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);
	compute_mesh_vertex_colors();
	return (int)(m_mesh.indices.size()/3);
}

uint8_t* Testbed::Nerf::get_density_grid_bitfield_mip(uint32_t mip) {
	return density_grid_bitfield.data() + grid_mip_offset(mip)/8;
}

int Testbed::find_best_training_view(int default_view) {
	int bestimage=default_view;
	float bestscore=1000.f;
	for (int i=0;i<m_nerf.training.dataset.n_images;++i) {
		float score=(m_nerf.training.dataset.xforms[i].col(3)-m_camera.col(3)).norm();
		score+=0.25f * (m_nerf.training.dataset.xforms[i].col(2)-m_camera.col(2)).norm();
		if (score<bestscore) { bestscore=score; bestimage=i; }
	}
	return bestimage;
}

NGP_NAMESPACE_END
