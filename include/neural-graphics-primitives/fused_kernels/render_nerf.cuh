/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_nerf.cuh
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/nerf_device.cuh>

using namespace ngp;

__launch_bounds__(128, 4)
__global__ void render_nerf(
	uint32_t sample_index,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	BoundingBox train_aabb,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode,
	const network_precision_t* __restrict__ params,
	const float* __restrict__ extra_dims,
	ENerfActivation density_activation,
	ENerfActivation rgb_activation,
	float min_transmittance,
	bool train_in_linear_colors,
	bool surface_rendering,
	float surface_rendering_threshold
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	bool valid = x < resolution.x && y < resolution.y;
	uint32_t idx = x + resolution.x * y;

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	mat4x3 camera = get_xform_given_rolling_shutter({camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);

	bool alive = valid && ray.is_valid();

	ray.d = normalize(ray.d);
	float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

	alive &= render_aabb.contains(render_aabb_to_local * ray(t));

	vec3 idir = vec3(1.0f) / ray.d;
	vec4 color = vec4(0.0f);

	vec3 cam_fwd = camera[2];
	vec3 cam_pos = camera[3];
	float best_depth_candidate = MAX_DEPTH();

	float cone_angle = calc_cone_angle(dot(ray.d, cam_fwd), focal_length, cone_angle_constant);
	t = advance_n_steps(t, cone_angle, ld_random_val(sample_index, idx * 786433));

	float max_weight = 0.0f;

	while (true) {
		// Advance to next occupied voxel
		vec3 pos = cam_pos;

		if (alive) {
			t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle, ray, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
			alive &= t < MAX_DEPTH();
			if (alive) {
				pos = ray(t);
			}
		}

		// Evaluate NeRF model
		float dt = calc_dt(t, cone_angle);
		constexpr uint32_t N_NERF_COORDS = sizeof(NerfCoordinate) / sizeof(float);
		vec<N_NERF_COORDS + N_EXTRA_DIMS> nerf_in;
		*(NerfCoordinate*)&nerf_in[0] = {warp_position(pos, train_aabb), warp_direction(ray.d), warp_dt(dt)};

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

		// Composit color
		float alpha = 1.f - __expf(-network_to_density(nerf_out.w, density_activation) * dt);
		float weight = alpha * (1.0f - color.a);
		vec3 rgb = network_to_rgb_vec(nerf_out.xyz(), rgb_activation);
		color += vec4(rgb * weight, weight);

		if (weight > max_weight) {
			max_weight = weight;
			best_depth_candidate = lens.is_360() ? distance(pos, cam_pos) : dot(cam_fwd, pos - cam_pos);
		}

		if (surface_rendering && alpha >= surface_rendering_threshold) {
			// Surface rendering: return the first surface point that has a sufficient occupancy
			color.rgb() = rgb;
			color.a = 1.0f;
			best_depth_candidate = lens.is_360() ? distance(pos, cam_pos) : dot(cam_fwd, pos - cam_pos);
			alive = false;
		} else if (color.a > (1.0f - min_transmittance)) {
			color /= color.a;
			alive = false;
		}
	}

	if (!valid) {
		return;
	}

	if (!train_in_linear_colors) {
		color.rgb() = srgb_to_linear(color.rgb());
	}

	depth_buffer[idx] = color.a > 0.2f ? best_depth_candidate : MAX_DEPTH();

	if (envmap) {
		color += read_envmap(envmap, ray.d) * (1.0f - color.a);
	}

	frame_buffer[idx] = color;
}
