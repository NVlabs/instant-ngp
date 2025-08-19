/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   train_nerf.cuh
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/nerf_device.cuh>

using namespace ngp;

__global__ void train_nerf(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const uint32_t n_rays_total,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	float* __restrict__ loss_output,
	bool max_level_rand_training,
	float* __restrict__ max_level_ptr,
	uint32_t max_mip,
	bool snap_to_pixel_centers,
	bool train_envmap,
	float cone_angle_constant,
	Buffer2DView<const vec2> distortion,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 cdf_res,
	const float* __restrict__ extra_dims_gpu,
	const network_precision_t* __restrict__ params,
	ENerfActivation density_activation,
	ENerfActivation rgb_activation,

	float loss_scale,
	int padded_output_width,
	Buffer2DView<const vec4> envmap,
	float* __restrict__ envmap_gradient,
	const ivec2 envmap_resolution,
	ELossType envmap_loss_type,
	vec3 background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	network_precision_t* dloss_doutput,
	ELossType loss_type,
	ELossType depth_loss_type,

	float* __restrict__ error_map,
	const ivec2 error_map_res,
	const float* __restrict__ sharpness_data,
	ivec2 sharpness_resolution,
	float* __restrict__ sharpness_grid,

	const float* __restrict__ mean_density_ptr,
	const vec3* __restrict__ exposure,
	vec3* __restrict__ exposure_gradient,
	float depth_supervision_lambda,
	float near_distance,

	uint32_t training_step,
	ETrainMode training_mode
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	float img_pdf = 1.0f;
	uint32_t img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
	ivec2 resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());

	float uv_pdf = 1.0f;
	vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img, &uv_pdf);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(uv, resolution, 0);
	vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);
	bool valid = texsamp.x >= 0.0f;

	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level
	float motionblur_time = random_val(rng);

	const vec2 focal_length = metadata[img].focal_length;
	const vec2 principal_point = metadata[img].principal_point;
	const float* extra_dims = extra_dims_gpu + img * N_EXTRA_DIMS;
	const Lens lens = metadata[img].lens;

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);

	Ray ray;

	const Ray* rays_in_unnormalized = metadata[img].rays;
	if (rays_in_unnormalized) {
		// Rays have been explicitly supplied. Read them.
		ray = rays_in_unnormalized[pix_idx];
	} else {
		ray = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, lens, distortion);
		valid &= ray.is_valid();
	}

	float ray_length = length(ray.d);
	ray.d /= ray_length;

	vec2 tminmax = aabb.ray_intersect(ray.o, ray.d);
	float cone_angle = calc_cone_angle(dot(ray.d, xform[2]), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x = fmaxf(tminmax.x, 0.0f);

	const float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
	vec3 idir = vec3(1.0f) / ray.d;

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}

	vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	vec4 envmap_value;
	if (envmap) {
		envmap_value = read_envmap(envmap, ray.d);
		background_color = envmap_value.rgb() + background_color * (1.0f - envmap_value.a);
	}

	vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);

	// Need rgbtarget before the first pass in RFL training mode
	vec3 rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.rgb() + (1.0f - texsamp.a) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.a > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb() / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;

	constexpr float EPSILON = 1e-4f;

	vec4 color = vec4(0.0f);
	vec3 hitpoint = vec3(0.0f);
	vec3 loss_bg = vec3(0.0f);

	bool alive = valid;

	while (true) {
		// Advance to next occupied voxel
		vec3 pos = xform[3];

		if (alive) {
			t = if_unoccupied_advance_to_next_occupied_voxel<true>(t, cone_angle, ray, idir, density_grid, 0, max_mip, aabb);
			alive &= t < MAX_DEPTH();
			if (alive) {
				pos = ray(t);
			}
		}

		// Evaluate NeRF model
		float dt = calc_dt(t, cone_angle);
		constexpr uint32_t N_NERF_COORDS = sizeof(NerfCoordinate) / sizeof(float);
		vec<N_NERF_COORDS + N_EXTRA_DIMS> nerf_in;
		*(NerfCoordinate*)&nerf_in[0] = {warp_position(pos, aabb), warp_direction(ray.d), warp_dt(dt)};

		NGP_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_EXTRA_DIMS; ++i) {
			nerf_in[N_NERF_COORDS + i] = extra_dims[i];
		}

		// Implicit sync is fine here, because the following nerf model wants synchronized warps anyway
		if (__all_sync(0xFFFFFFFF, !alive)) {
			break;
		}

		vec4 nerf_out = eval_nerf(nerf_in, params);

		// All threads in the warp must execute the above MLPs for coherence reasons.
		// Starting from here, it's fine to skip computation.
		if (!alive) {
			continue;
		}

		t += dt;
		++j;

		// Composit color
		float alpha = 1.f - __expf(-network_to_density(nerf_out.w, density_activation) * dt);
		float weight = alpha * (1.0f - color.a);
		vec3 rgb = network_to_rgb_vec(nerf_out.rgb(), rgb_activation);
		color += vec4(rgb * weight, weight);

		loss_bg += weight * loss_and_gradient(rgbtarget, rgb, loss_type).loss;
		hitpoint += weight * pos;

		if (1.0f - color.a < EPSILON || j >= NERF_STEPS()) {
			alive = false;
		}
	}

	float depth = distance(hitpoint, ray.o);
	hitpoint /= color.a;

	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	numsteps = min(max_samples - min(max_samples, base), numsteps);
	bool can_write = numsteps > 0;

	coords_out += base;
	max_level_ptr += base;
	dloss_doutput += base * padded_output_width;

	uint32_t ray_idx = 0;
	if (can_write) {
		ray_idx = atomicAdd(ray_counter, 1);
		ray_indices_out[ray_idx] = i;
		numsteps_out[ray_idx*2+0] = numsteps;
		numsteps_out[ray_idx*2+1] = base;
	}

	if (1.0f - color.a >= EPSILON) {
		color.rgb() += (1.0f - color.a) * background_color;
		loss_bg += (1.0f - color.a) * loss_and_gradient(rgbtarget, background_color, loss_type).loss;
	}

	// Step again, this time computing loss
	LossAndGradient lg = loss_and_gradient(rgbtarget, color.rgb(), loss_type);
	lg.loss /= img_pdf * uv_pdf;

	float target_depth = ray_length * ((depth_supervision_lambda > 0.0f && metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);
	LossAndGradient lg_depth = loss_and_gradient(vec3(target_depth), vec3(depth), depth_loss_type);
	float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;

	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * uv_pdf;

	float mean_loss = mean(lg.loss);
	if (can_write && loss_output) {
		loss_output[ray_idx] = mean_loss / (float)n_rays;
	}

	if (can_write && error_map) {
		const vec2 pos = clamp(uv * vec2(error_map_res) - 0.5f, vec2(0.0f), vec2(error_map_res) - (1.0f + 1e-4f));
		const ivec2 pos_int = pos;
		const vec2 weight = pos - vec2(pos_int);

		ivec2 idx = clamp(pos_int, 0, resolution - 2);

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[img * product(error_map_res) + y * error_map_res.x + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), 0, sharpness_resolution - 1);
			float sharp = sharpness_data[img * product(sharpness_resolution) + sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x,   idx.y,   (1 - weight.x) * (1 - weight.y) * mean_loss);
		deposit_val(idx.x+1, idx.y,        weight.x  * (1 - weight.y) * mean_loss);
		deposit_val(idx.x,   idx.y+1, (1 - weight.x) *      weight.y  * mean_loss);
		deposit_val(idx.x+1, idx.y+1,      weight.x  *      weight.y  * mean_loss);
	}

	loss_scale /= n_rays;

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = 0.0f;// *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
	vec4 color2 = vec4(0.0f);
	vec3 loss_bg2 = vec3(0.0f);
	float depth2 = 0.0f;
	t = startt;
	j = 0;
	alive = valid;

	while (true) {
		// Advance to next occupied voxel
		vec3 pos = xform[3];

		if (alive) {
			t = if_unoccupied_advance_to_next_occupied_voxel<true>(t, cone_angle, ray, idir, density_grid, 0, max_mip, aabb);
			alive &= t < MAX_DEPTH();
			if (alive) {
				pos = ray(t);
			}
		}

		// Evaluate NeRF model
		float dt = calc_dt(t, cone_angle);
		constexpr uint32_t N_NERF_COORDS = sizeof(NerfCoordinate) / sizeof(float);
		vec<N_NERF_COORDS + N_EXTRA_DIMS> nerf_in;
		*(NerfCoordinate*)&nerf_in[0] = {warp_position(pos, aabb), warp_direction(ray.d), warp_dt(dt)};

		NGP_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_EXTRA_DIMS; ++i) {
			nerf_in[N_NERF_COORDS + i] = extra_dims[i];
		}

		// Implicit sync is fine here, because the following nerf model wants synchronized warps anyway
		if (__all_sync(0xFFFFFFFF, !alive)) {
			break;
		}

		vec4 local_network_output = eval_nerf(nerf_in, params);

		// All threads in the warp must execute the above MLPs for coherence reasons.
		// Starting from here, it's fine to skip computation.
		if (!alive) {
			continue;
		}

		// Composit color
		float density = network_to_density(local_network_output.w, density_activation);
		vec3 rgb = network_to_rgb_vec(local_network_output.rgb(), rgb_activation);

		float alpha = 1.f - __expf(-density * dt);
		float weight = alpha * (1.0f - color2.a);
		color2 += vec4{rgb * weight, weight};
		float local_depth = distance(pos, ray.o);
		depth2 += weight * local_depth;
		float T = 1.0f - color2.a;

		t += dt;
		++j;

		// Compute and write loss stuff. Only needed if we want to actually output anything
		if (j > numsteps) {
			continue;
		}

		coords_out(j-1)->copy(*(NerfCoordinate*)&nerf_in[0], coords_out.stride_in_bytes);
		if (max_level_rand_training) {
			max_level_ptr[j-1] = max_level;
		}

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const vec3 suffix = color.rgb() - color2.rgb();

		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		const float depth_suffix = depth - depth2;
		const float depth_supervision = depth_loss_gradient * (T * local_depth - depth_suffix);

		vec3 dloss_by_drgb;
		float dloss_by_dmlp;
		if (training_mode == ETrainMode::Rfl && training_step < 1000) {
			training_mode = ETrainMode::Nerf; // Warm up training
		}
		if (training_mode == ETrainMode::Rfl) {
			// Radiance field loss
			LossAndGradient local_lg = loss_and_gradient(rgbtarget, rgb, loss_type);
			loss_bg2 += weight * local_lg.loss;
			dloss_by_drgb = weight * local_lg.gradient;
			dloss_by_dmlp = density_derivative * (
				dt * sum(T * local_lg.loss - (loss_bg - loss_bg2) + depth_supervision)
			);
		} else if (training_mode == ETrainMode::RflRelax) {
			// In-between volume reconstruction and surface reconstruction.
			// This is different from the relaxation in the paper, but is much simpler and also promotes surfaces.
			const vec3 rgb_bg = suffix / fmaxf(1e-6f, T);
			const vec3 rgb_lerp = (1 - alpha) * rgb_bg + alpha * rgb;
			LossAndGradient local_lg = loss_and_gradient(rgbtarget, rgb_lerp, loss_type);

			dloss_by_drgb = weight * local_lg.gradient;
			dloss_by_dmlp = density_derivative * (
				dt * (dot(local_lg.gradient, T * rgb - suffix) + depth_supervision)
			);
		} else {
			// The original NeRF loss
			dloss_by_drgb = weight * lg.gradient;
			dloss_by_dmlp = density_derivative * (
				dt * (dot(lg.gradient, T * rgb - suffix) + depth_supervision)
			);
		}

		tvec<network_precision_t, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
		//dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

		local_dL_doutput[3] =
			loss_scale * dloss_by_dmlp +
			(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
			(float(local_network_output[3]) > -10.0f && local_depth < near_distance ? 1e-4f : 0.0f);
			;

		*(tvec<network_precision_t, 4>*)dloss_doutput = local_dL_doutput;
		dloss_doutput += padded_output_width;

		if (1.0f - color2.a < EPSILON || j >= NERF_STEPS()) {
			alive = false;
		}
	}

	if (j < numsteps) {
		printf("WTF %d vs %d; %d vs %d\n", j, numsteps, base, max_samples);
	}

	if (can_write && exposure_gradient) {
		// Assume symmetric loss
		vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

		if (!train_in_linear_colors) {
			dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
		}

		// 2^exposure * log(2)
		vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
		atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
		atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
		atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
	}

	if (can_write && 1.0f - color2.a >= EPSILON && envmap_gradient) {
		vec3 loss_gradient = lg.gradient;
		if (envmap_loss_type != loss_type) {
			loss_gradient = loss_and_gradient(rgbtarget, color.rgb(), envmap_loss_type).gradient;
		}

		vec3 dloss_by_dbackground = (1.0f - color2.a) * loss_gradient;
		if (!train_in_linear_colors) {
			dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
		}

		tvec<network_precision_t, 4> dL_denvmap;
		dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
		dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
		dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;

		float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

		// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
		dL_denvmap[3] = (network_precision_t)0;

		deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, ray.d);
	}
}
