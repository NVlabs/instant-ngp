/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_sdf.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/tinyobj_loader_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

static constexpr uint32_t MARCH_ITER = 10000;

Testbed::NetworkDims Testbed::network_dims_sdf() const {
	NetworkDims dims;
	dims.n_input = 3;
	dims.n_output = 1;
	dims.n_pos = 3;
	return dims;
}

__device__ inline float square(float x) { return x * x; }
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline Vector3f mix(const Vector3f& a, const Vector3f& b, float t) { return a + (b - a) * t; }

__device__ inline float SchlickFresnel(float u) {
	float m = __saturatef(1.0 - u);
	return square(square(m)) * m;
}

__device__ inline float G1(float NdotH, float a) {
	if (a >= 1.0) { return 1.0 / PI(); }
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return (a2 - 1.0) / (PI() * log(a2) * t);
}

__device__ inline float G2(float NdotH, float a) {
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return a2 / (PI() * t * t);
}

__device__ inline float SmithG_GGX(float NdotV, float alphaG) {
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1.0 / (NdotV + sqrtf(a + b - a * b));
}

// this function largely based on:
// https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
// http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
__device__ Vector3f evaluate_shading(
	const Vector3f& base_color,
	const Vector3f& ambient_color, // :)
	const Vector3f& light_color, // :)
	float metallic,
	float subsurface,
	float specular,
	float roughness,
	float specular_tint,
	float sheen,
	float sheen_tint,
	float clearcoat,
	float clearcoat_gloss,
	Vector3f L,
	Vector3f V,
	Vector3f N
) {
	float NdotL = N.dot(L);
	float NdotV = N.dot(V);

	Vector3f H = (L + V).normalized();
	float NdotH = N.dot(H);
	float LdotH = L.dot(H);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
	Vector3f amb = (ambient_color * mix(0.2f, FV, metallic));
	amb = amb.array() * base_color.array();
	if (NdotL < 0.f || NdotV < 0.f) {
		return amb;
	}

	float luminance = base_color.dot(Vector3f(0.3f, 0.6f, 0.1f));

	// normalize luminance to isolate hue and saturation components
	Vector3f Ctint = base_color * (1.f/(luminance+0.00001f));
	Vector3f Cspec0 = mix(mix(Vector3f(1.0f,1.0f,1.0f), Ctint, specular_tint) * specular * 0.08f, base_color, metallic);
	Vector3f Csheen = mix(Vector3f(1.0f,1.0f,1.0f), Ctint, sheen_tint);

	float Fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
	float Fd = mix(1, Fd90, FL) * mix(1.f, Fd90, FV);

	// Based on Hanrahan-Krueger BRDF approximation of isotropic BSSRDF
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LdotH * LdotH * roughness;
	float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
	float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - 0.5f) + 0.5f);

	// Specular
	float a= std::max(0.001f, square(roughness));
	float Ds = G2(NdotH, a);
	float FH = SchlickFresnel(LdotH);
	Vector3f Fs = mix(Cspec0, Vector3f(1.0f,1.0f,1.0f), FH);
	float Gs = SmithG_GGX(NdotL, a) * SmithG_GGX(NdotV, a);

	// sheen
	Vector3f Fsheen = FH * sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = G1(NdotH, mix(0.1f, 0.001f, clearcoat_gloss));
	float Fr = mix(0.04f, 1.0f, FH);
	float Gr = SmithG_GGX(NdotL, 0.25f) * SmithG_GGX(NdotV, 0.25f);

	float CCs=0.25f * clearcoat * Gr * Fr * Dr;
	Vector3f brdf = (float(1.0f / PI()) * mix(Fd, ss, subsurface) * base_color + Fsheen) * (1.0f - metallic) +
		Gs * Fs * Ds + Vector3f(CCs,CCs,CCs);
	return Vector3f(brdf.array() * light_color.array()) * NdotL + amb;
}

__global__ void advance_pos_kernel_sdf(
	const uint32_t n_elements,
	const float zero_offset,
	Vector3f* __restrict__ positions,
	float* __restrict__ distances,
	SdfPayload* __restrict__ payloads,
	BoundingBox aabb,
	float floor_y,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	int max_depth,
	float distance_scale,
	float maximum_distance,
	float k,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	SdfPayload& payload = payloads[i];
	if (!payload.alive) {
		return;
	}

	float distance = distances[i] - zero_offset;

	distance *= distance_scale;

	// Advance by the predicted distance
	Vector3f pos = positions[i];
	pos += distance * payload.dir;

	// Skip over regions not covered by the octree
	if (octree_nodes && !TriangleOctree::contains(octree_nodes, max_depth, pos)) {
		float octree_distance = (TriangleOctree::ray_intersect(octree_nodes, max_depth, pos, payload.dir) + 1e-6f);
		distance += octree_distance;
		pos += octree_distance * payload.dir;
	}
	if (pos.y() < floor_y && payload.dir.y()<0.f) {
		float floor_dist = -(pos.y()-floor_y)/payload.dir.y();
		distance += floor_dist;
		pos += floor_dist * payload.dir;
		payload.alive=false;
	}

	positions[i] = pos;

	if (total_distances && distance > 0.0f) {
		// From https://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
		float total_distance = total_distances[i];
		float y = distance*distance / (2.0f * prev_distances[i]);
		float d = sqrtf(distance*distance - y*y);

		min_visibility[i] = fminf(min_visibility[i], k * d / fmaxf(0.0f, total_distance - y));
		prev_distances[i] = distance;
		total_distances[i] = total_distance + distance;
	}

	bool stay_alive = distance > maximum_distance && fabsf(distance / 2) > 3*maximum_distance;
	if (!stay_alive) {
		payload.alive = false;
		return;
	}

	if (!aabb.contains(pos)) {
		payload.alive = false;
		return;
	}

	payload.n_steps++;
}

__global__ void perturb_sdf_samples(uint32_t n_elements, const Vector3f* __restrict__ perturbations, Vector3f* __restrict__ positions, float* __restrict__ distances) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	Vector3f perturbation = perturbations[i];
	positions[i] += perturbation;

	// Small epsilon above 1 to ensure a triangle is always found.
	distances[i] = perturbation.norm()*1.001f;
}

__global__ void prepare_shadow_rays(const uint32_t n_elements,
	Vector3f sun_dir,
	Vector3f* __restrict__ positions,
	Vector3f* __restrict__ normals,
	float* __restrict__ distances,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility,
	SdfPayload* __restrict__ payloads,
	BoundingBox aabb,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	int max_depth
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	SdfPayload& payload = payloads[i];

	// Step back a little along the ray to prevent self-intersection
	Vector3f view_pos = positions[i] + faceforward(normals[i], -payload.dir, normals[i]).normalized() * 1e-3f;
	Vector3f dir = sun_dir.normalized();

	float t = fmaxf(aabb.ray_intersect(view_pos, dir).x() + 1e-6f, 0.0f);
	view_pos += t * dir;

	if (octree_nodes && !TriangleOctree::contains(octree_nodes, max_depth, view_pos)) {
		t = fmaxf(0.0f, TriangleOctree::ray_intersect(octree_nodes, max_depth, view_pos, dir) + 1e-6f);
		view_pos += t * dir;
	}

	positions[i] = view_pos;

	if (!aabb.contains(view_pos)) {
		distances[i] = 10000.0f;
		payload.alive = false;
		min_visibility[i] = 1.0f;
		return;
	}

	distances[i] = 10000.0f;
	payload.idx = i;
	payload.dir = dir;
	payload.n_steps = 0;
	payload.alive = true;

	if (prev_distances) {
		prev_distances[i] = 1e20f;
	}

	if (total_distances) {
		total_distances[i] = 0.0f;
	}

	if (min_visibility) {
		min_visibility[i] = 1.0f;
	}
}

__global__ void write_shadow_ray_result(const uint32_t n_elements, BoundingBox aabb, const Vector3f* __restrict__ positions, const SdfPayload* __restrict__ shadow_payloads, const float* __restrict__ min_visibility, float* __restrict__ shadow_factors) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	shadow_factors[shadow_payloads[i].idx] = aabb.contains(positions[i]) ? 0.0f : min_visibility[i];
}

__global__ void shade_kernel_sdf(
	const uint32_t n_elements,
	BoundingBox aabb,
	float floor_y,
	const ERenderMode mode,
	const BRDFParams brdf,
	Vector3f sun_dir,
	Vector3f up_dir,
	Matrix<float, 3, 4> camera_matrix,
	Vector3f* __restrict__ positions,
	Vector3f* __restrict__ normals,
	float* __restrict__ distances,
	SdfPayload* __restrict__ payloads,
	Array4f* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	SdfPayload& payload = payloads[i];
	if (!aabb.contains(positions[i])) {
		return;
	}

	// The normal in memory isn't normalized yet
	Vector3f normal = normals[i].normalized();

	Vector3f pos = positions[i];
	bool floor = false;
	if (pos.y() < floor_y+0.001f && payload.dir.y() < 0.f) {
		normal = Vector3f(0.f, 1.f, 0.f);
		floor = true;
	}
	Vector3f cam_pos = camera_matrix.col(3);
	Vector3f cam_fwd = camera_matrix.col(2);
	float ao = powf(0.92f, payload.n_steps * 0.5f) * (1.f / 0.92f);
	Array3f color;
	switch (mode) {
		case ERenderMode::AO: color = Array3f::Constant(powf(0.92f, payload.n_steps)); break;
		case ERenderMode::Shade: {
			float skyam = normal.dot(up_dir)*-0.5f+0.5f;
			Vector3f suncol = Array3f{255.f/255.0f, 225.f/255.0f, 195.f/255.0f} * 4.f * distances[i]; // Distance encodes shadow occlusion. 0=occluded, 1=no shadow
			const Vector3f skycol = Array3f{195.f/255.0f, 215.f/255.0f, 255.f/255.0f} * 4.f * skyam;
			float check_size = 8.f/aabb.diag().x();
			float check=((int(floorf(check_size*(pos.x()-aabb.min.x())))^int(floorf(check_size*(pos.z()-aabb.min.z())))) &1) ? 0.8f : 0.2f;
			const Vector3f floorcol = Array3f{check*check*check, check*check, check};
			Vector3f col = evaluate_shading(floor ? floorcol : brdf.basecolor.array() * brdf.basecolor.array(), brdf.ambientcolor.array() * skycol.array(), suncol, floor ? 0.f : brdf.metallic, floor ? 0.f : brdf.subsurface, floor ? 1.f : brdf.specular, floor ? 0.5f : brdf.roughness, 0.f, floor ? 0.f : brdf.sheen, 0.f, floor ? 0.f : brdf.clearcoat, brdf.clearcoat_gloss, sun_dir, -payload.dir.normalized(), normal);
			color = col.array();
		} break;
		case ERenderMode::Depth: color = Array3f::Constant(cam_fwd.dot(pos - cam_pos)); break;
		case ERenderMode::Positions: {
			color = (pos.array() - Array3f::Constant(0.5f)) / 2.0f + Array3f::Constant(0.5f);
		} break;
		case ERenderMode::Normals: color = 0.5f * normal.array() + Array3f::Constant(0.5f); break;
		case ERenderMode::Cost: color = Array3f::Constant((float)payload.n_steps / 30); break;
		case ERenderMode::EncodingVis: color = normals[i]; break;
	}

	frame_buffer[payload.idx] = {color.x(), color.y(), color.z(), 1.0f};
	depth_buffer[payload.idx] = cam_fwd.dot(pos - cam_pos);
}

__global__ void compact_kernel_shadow_sdf(
	const uint32_t n_elements,
	const float zero_offset,
	Vector3f* src_positions, float* src_distances, SdfPayload* src_payloads, float* src_prev_distances, float* src_total_distances, float* src_min_visibility,
	Vector3f* dst_positions, float* dst_distances, SdfPayload* dst_payloads, float* dst_prev_distances, float* dst_total_distances, float* dst_min_visibility,
	Vector3f* dst_final_positions, float* dst_final_distances, SdfPayload* dst_final_payloads, float* dst_final_prev_distances, float* dst_final_total_distances, float* dst_final_min_visibility,
	BoundingBox aabb,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	SdfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_positions[idx] = src_positions[i];
		dst_distances[idx] = src_distances[i];
		dst_prev_distances[idx] = src_prev_distances[i];
		dst_total_distances[idx] = src_total_distances[i];
		dst_min_visibility[idx] = src_min_visibility[i];
	} else { // For shadow rays, collect _all_ final samples to keep track of their partial visibility
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = src_distances[i];
		dst_final_prev_distances[idx] = src_prev_distances[i];
		dst_final_total_distances[idx] = src_total_distances[i];
		dst_final_min_visibility[idx] = aabb.contains(src_positions[i]) ? 0.0f : src_min_visibility[i];
	}
}

__global__ void compact_kernel_sdf(
	const uint32_t n_elements,
	const float zero_offset,
	Vector3f* src_positions, float* src_distances, SdfPayload* src_payloads,
	Vector3f* dst_positions, float* dst_distances, SdfPayload* dst_payloads,
	Vector3f* dst_final_positions, float* dst_final_distances, SdfPayload* dst_final_payloads,
	BoundingBox aabb,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	SdfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_positions[idx] = src_positions[i];
		dst_distances[idx] = src_distances[i];
	} else if (aabb.contains(src_positions[i])) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = 1.0f; // HACK: Distances encode shadowing factor when shading
	}
}

__global__ void uniform_octree_sample_kernel(
	const uint32_t num_elements,
	default_rng_t rng,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	uint32_t num_nodes,
	uint32_t depth,
	Vector3f* __restrict__ samples
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	rng.advance(i * (1<<8));

	// Samples random nodes until a leaf is picked
	uint32_t node;
	uint32_t child;
	do {
		node = umin((uint32_t)(random_val(rng) * num_nodes), num_nodes-1);
		child = umin((uint32_t)(random_val(rng) * 8), 8u-1);
	} while (octree_nodes[node].depth < depth-2 || octree_nodes[node].children[child] == -1);

	// Here it should be guaranteed that any child of the node is -1
	float size = scalbnf(1.0f, -depth+1);

	Vector3i16 pos = octree_nodes[node].pos*2;
	if (child&1) ++pos.x();
	if (child&2) ++pos.y();
	if (child&4) ++pos.z();
	samples[i] = size * (pos.cast<float>() + samples[i]);
}

__global__ void scale_to_aabb_kernel(uint32_t n_elements, BoundingBox aabb, Vector3f* __restrict__ inout) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	inout[i] = aabb.min + inout[i].cwiseProduct(aabb.diag());
}

__global__ void compare_signs_kernel(uint32_t n_elements, const Vector3f *positions, const float *distances_ref, const float *distances_model, uint32_t *counters, const TriangleOctreeNode* octree_nodes, int max_depth) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	bool inside1 = distances_ref[i]<=0.f;
	bool inside2 = distances_model[i]<=0.f;
	if (octree_nodes && !TriangleOctree::contains(octree_nodes, max_depth, positions[i])) {
		inside2=inside1; // assume, when using the octree, that the model is always correct outside the octree.
		atomicAdd(&counters[6],1); // outside the octree
	} else {
		atomicAdd(&counters[7],1); // inside the octree
	}
	atomicAdd(&counters[inside1 ? 0 : 1],1);
	atomicAdd(&counters[inside2 ? 2 : 3],1);
	if (inside1&&inside2)
		atomicAdd(&counters[4],1);
	if (inside1||inside2)
		atomicAdd(&counters[5],1);

}

__global__ void scale_iou_counters_kernel(uint32_t n_elements, uint32_t *counters, float scale) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	counters[i] = uint32_t(roundf(counters[i]*scale));
}

__global__ void assign_float(uint32_t n_elements, float value, float* __restrict__ out) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	out[i] = value;
}

__global__ void init_rays_with_payload_kernel_sdf(
	uint32_t sample_index,
	Vector3f* __restrict__ positions,
	float* __restrict__ distances,
	SdfPayload* __restrict__ payloads,
	Vector2i resolution,
	Vector2f focal_length,
	Matrix<float, 3, 4> camera_matrix,
	Vector2f screen_center,
	Vector3f parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	const float* __restrict__ envmap_data,
	const Vector2i envmap_resolution,
	Array4f* __restrict__ framebuffer,
	float* __restrict__ depthbuffer,
	const TriangleOctreeNode* __restrict__ octree_nodes = nullptr,
	int max_depth = 0
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	Ray ray = pixel_to_ray(sample_index, {x, y}, resolution, focal_length, camera_matrix, screen_center, parallax_shift, snap_to_pixel_centers, near_distance, plane_z, aperture_size);

	distances[idx] = 10000.0f;

	if (plane_z < 0) {
		float n = ray.d.norm();
		SdfPayload& payload = payloads[idx];
		payload.dir = (1.0f/n) * ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o - plane_z * ray.d;
		depthbuffer[idx] = -plane_z;
		return;
	}

	depthbuffer[idx] = 1e10f;

	ray.d = ray.d.normalized();
	float t = max(aabb.ray_intersect(ray.o, ray.d).x(), 0.0f);
	ray.o = ray.o + (t + 1e-6f) * ray.d;

	if (octree_nodes && !TriangleOctree::contains(octree_nodes, max_depth, ray.o)) {
		t = max(0.0f, TriangleOctree::ray_intersect(octree_nodes, max_depth, ray.o, ray.d));
		if (ray.o.y() > floor_y && ray.d.y() < 0.f) {
			float floor_dist = -(ray.o.y() - floor_y) / ray.d.y();
			if (floor_dist > 0.f) {
				t = min(t, floor_dist);
			}
		}

		ray.o = ray.o + (t + 1e-6f) * ray.d;
	}

	positions[idx] = ray.o;

	if (envmap_data) {
		framebuffer[idx] = read_envmap(envmap_data, envmap_resolution, ray.d);
	}

	SdfPayload& payload = payloads[idx];
	payload.dir = ray.d;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = aabb.contains(ray.o);
}

__host__ __device__ uint32_t sample_discrete(float uniform_sample, const float* __restrict__ cdf, int length) {
	return binary_search(uniform_sample, cdf, length);
}

__global__ void sample_uniform_on_triangle_kernel(uint32_t n_elements, const float* __restrict__ cdf, uint32_t length, const Triangle* __restrict__ triangles, Vector3f* __restrict__ sampled_positions) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	Vector3f sample = sampled_positions[i];
	uint32_t tri_idx = sample_discrete(sample.x(), cdf, length);

	sampled_positions[i] = triangles[tri_idx].sample_uniform_position(sample.tail<2>());
}

void Testbed::SphereTracer::init_rays_from_camera(
	uint32_t sample_index,
	const Vector2i& resolution,
	const Vector2f& focal_length,
	const Matrix<float, 3, 4>& camera_matrix,
	const Vector2f& screen_center,
	const Vector3f& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	const float* envmap_data,
	const Vector2i& envmap_resolution,
	Array4f* frame_buffer,
	float* depth_buffer,
	const TriangleOctree* octree,
	uint32_t n_octree_levels,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x() * resolution.y();
	enlarge(n_pixels, stream);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x(), threads.x), div_round_up((uint32_t)resolution.y(), threads.y), 1 };
	init_rays_with_payload_kernel_sdf<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].pos,
		m_rays[0].distance,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		aabb,
		floor_y,
		near_distance,
		plane_z,
		aperture_size,
		envmap_data,
		envmap_resolution,
		frame_buffer,
		depth_buffer,
		octree ? octree->nodes_gpu() : nullptr,
		octree ? n_octree_levels : 0
	);
	m_n_rays_initialized = (uint32_t)n_pixels;
}

void Testbed::SphereTracer::init_rays_from_data(uint32_t n_elements, const RaysSdfSoa& data, cudaStream_t stream) {
	enlarge(n_elements, stream);
	m_rays[0].copy_from_other_async(n_elements, data, stream);
	m_n_rays_initialized = n_elements;
}

uint32_t Testbed::SphereTracer::trace_bvh(TriangleBvh* bvh, const Triangle* triangles, cudaStream_t stream) {
	uint32_t n_alive = m_n_rays_initialized;
	m_n_rays_initialized = 0;

	if (!bvh) {
		return 0;
	}

	// Abuse the normal buffer to temporarily hold ray directions
	parallel_for_gpu(stream, n_alive, [payloads=m_rays[0].payload, normals=m_rays[0].normal] __device__ (size_t i) {
		normals[i] = payloads[i].dir;
	});

	bvh->ray_trace_gpu(n_alive, m_rays[0].pos, m_rays[0].normal, triangles, stream);
	return n_alive;
}

uint32_t Testbed::SphereTracer::trace(
	const distance_fun_t& distance_function,
	float zero_offset,
	float distance_scale,
	float maximum_distance,
	const BoundingBox& aabb,
	const float floor_y,
	const TriangleOctree* octree,
	const uint32_t n_octree_levels,
	cudaStream_t stream
) {
	if (m_n_rays_initialized == 0) {
		return 0;
	}

	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

	const uint32_t STEPS_INBETWEEN_COMPACTION = 4;

	uint32_t n_alive = m_n_rays_initialized;
	m_n_rays_initialized = 0;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		// Compact more frequently in the first couple of steps
		uint32_t step_size = std::min(i, STEPS_INBETWEEN_COMPACTION);

		RaysSdfSoa& rays_current = m_rays[(double_buffer_index+1)%2];
		RaysSdfSoa& rays_tmp = m_rays[double_buffer_index%2];
		++double_buffer_index;

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
			if (m_trace_shadow_rays) {
				linear_kernel(compact_kernel_shadow_sdf, 0, stream,
					n_alive,
					zero_offset,
					rays_tmp.pos, rays_tmp.distance, rays_tmp.payload, rays_tmp.prev_distance, rays_tmp.total_distance, rays_tmp.min_visibility,
					rays_current.pos, rays_current.distance, rays_current.payload, rays_current.prev_distance, rays_current.total_distance, rays_current.min_visibility,
					m_rays_hit.pos, m_rays_hit.distance, m_rays_hit.payload, m_rays_hit.prev_distance, m_rays_hit.total_distance, m_rays_hit.min_visibility,
					aabb,
					m_alive_counter, m_hit_counter
				);
			} else {
				linear_kernel(compact_kernel_sdf, 0, stream,
					n_alive,
					zero_offset,
					rays_tmp.pos, rays_tmp.distance, rays_tmp.payload,
					rays_current.pos, rays_current.distance, rays_current.payload,
					m_rays_hit.pos, m_rays_hit.distance, m_rays_hit.payload,
					aabb,
					m_alive_counter, m_hit_counter
				);
			}
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (n_alive == 0) {
			break;
		}

		for (uint32_t j = 0; j < step_size; ++j) {
			distance_function(n_alive, rays_current.pos, rays_current.distance, stream);
			linear_kernel(advance_pos_kernel_sdf, 0, stream,
				n_alive,
				zero_offset,
				rays_current.pos,
				rays_current.distance,
				rays_current.payload,
				aabb,
				floor_y,
				octree ? octree->nodes_gpu() : nullptr,
				octree ? n_octree_levels : 0,
				distance_scale,
				maximum_distance,
				m_shadow_sharpness,
				m_trace_shadow_rays ? rays_current.prev_distance : nullptr,
				m_trace_shadow_rays ? rays_current.total_distance : nullptr,
				m_trace_shadow_rays ? rays_current.min_visibility : nullptr
			);
		}

		i += step_size;
	}

	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::SphereTracer::enlarge(size_t n_elements, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
	auto scratch = allocate_workspace_and_distribute<
		Vector3f, Vector3f, float, float, float, float, SdfPayload, // m_rays[0]
		Vector3f, Vector3f, float, float, float, float, SdfPayload, // m_rays[1]
		Vector3f, Vector3f, float, float, float, float, SdfPayload, // m_rays_hit

		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), std::get<6>(scratch));
	m_rays[1].set(std::get<7>(scratch), std::get<8>(scratch), std::get<9>(scratch), std::get<10>(scratch), std::get<11>(scratch), std::get<12>(scratch), std::get<13>(scratch));
	m_rays_hit.set(std::get<14>(scratch), std::get<15>(scratch), std::get<16>(scratch), std::get<17>(scratch), std::get<18>(scratch), std::get<19>(scratch), std::get<20>(scratch));

	m_hit_counter = std::get<21>(scratch);
	m_alive_counter = std::get<22>(scratch);
}

void Testbed::FiniteDifferenceNormalsApproximator::enlarge(uint32_t n_elements, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, tcnn::batch_size_granularity);
	auto scratch = allocate_workspace_and_distribute<
		Vector3f, Vector3f, Vector3f,
		float, float, float,
		float, float, float
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements
	);

	dx = std::get<0>(scratch);
	dy = std::get<1>(scratch);
	dz = std::get<2>(scratch);

	dist_dx_pos = std::get<3>(scratch);
	dist_dy_pos = std::get<4>(scratch);
	dist_dz_pos = std::get<5>(scratch);

	dist_dx_neg = std::get<6>(scratch);
	dist_dy_neg = std::get<7>(scratch);
	dist_dz_neg = std::get<8>(scratch);
}

void Testbed::FiniteDifferenceNormalsApproximator::normal(uint32_t n_elements, const distance_fun_t& distance_function, const Vector3f* pos, Vector3f* normal, float epsilon, cudaStream_t stream) {
	enlarge(n_elements, stream);

	parallel_for_gpu(stream, n_elements, [pos=pos, dx=dx, dy=dy, dz=dz, epsilon] __device__ (size_t i) {
		Vector3f p = pos[i];
		dx[i] = Vector3f{p.x() + epsilon, p.y(), p.z()};
		dy[i] = Vector3f{p.x(), p.y() + epsilon, p.z()};
		dz[i] = Vector3f{p.x(), p.y(), p.z() + epsilon};
	});

	distance_function(n_elements, dx, dist_dx_pos, stream);
	distance_function(n_elements, dy, dist_dy_pos, stream);
	distance_function(n_elements, dz, dist_dz_pos, stream);

	parallel_for_gpu(stream, n_elements, [pos=pos, dx=dx, dy=dy, dz=dz, epsilon] __device__ (size_t i) {
		Vector3f p = pos[i];
		dx[i] = Vector3f{p.x() - epsilon, p.y(), p.z()};
		dy[i] = Vector3f{p.x(), p.y() - epsilon, p.z()};
		dz[i] = Vector3f{p.x(), p.y(), p.z() - epsilon};
	});

	distance_function(n_elements, dx, dist_dx_neg, stream);
	distance_function(n_elements, dy, dist_dy_neg, stream);
	distance_function(n_elements, dz, dist_dz_neg, stream);

	parallel_for_gpu(stream, n_elements, [normal=normal, dist_dx_pos=dist_dx_pos, dist_dx_neg=dist_dx_neg, dist_dy_pos=dist_dy_pos, dist_dy_neg=dist_dy_neg, dist_dz_pos=dist_dz_pos, dist_dz_neg=dist_dz_neg] __device__ (size_t i) {
		normal[i] = {dist_dx_pos[i] - dist_dx_neg[i], dist_dy_pos[i] - dist_dy_neg[i], dist_dz_pos[i] - dist_dz_neg[i]};
	});
}

void Testbed::render_sdf(
	const distance_fun_t& distance_function,
	const normals_fun_t& normals_function,
	CudaRenderBuffer& render_buffer,
	const Vector2i& max_res,
	const Vector2f& focal_length,
	const Matrix<float, 3, 4>& camera_matrix,
	const Vector2f& screen_center,
	cudaStream_t stream
) {
	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}
	auto* octree_ptr = m_sdf.uses_takikawa_encoding || m_sdf.use_triangle_octree ? m_sdf.triangle_octree.get() : nullptr;

	SphereTracer tracer;

	uint32_t n_octree_levels = octree_ptr ? octree_ptr->depth() : 0;
	if (m_render_ground_truth && m_sdf.groundtruth_mode == ESDFGroundTruthMode::SDFBricks) {
		n_octree_levels = std::min(n_octree_levels, m_sdf.brick_level);
	}

	BoundingBox sdf_bounding_box = m_aabb;
	sdf_bounding_box.inflate(m_sdf.zero_offset);
	tracer.init_rays_from_camera(
		render_buffer.spp(),
		render_buffer.in_resolution(),
		focal_length,
		camera_matrix,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		sdf_bounding_box,
		get_floor_y(),
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		m_envmap.envmap->inference_params(),
		m_envmap.resolution,
		render_buffer.frame_buffer(),
		render_buffer.depth_buffer(),
		octree_ptr,
		n_octree_levels,
		stream
	);

	bool gt_raytrace = m_render_ground_truth && m_sdf.groundtruth_mode == ESDFGroundTruthMode::RaytracedMesh;

	auto trace = [&](SphereTracer& tracer) {
		if (gt_raytrace) {
			return tracer.trace_bvh(m_sdf.triangle_bvh.get(), m_sdf.triangles_gpu.data(), stream);
		} else {
			return tracer.trace(
				distance_function,
				m_sdf.zero_offset,
				m_sdf.distance_scale,
				m_sdf.maximum_distance,
				sdf_bounding_box,
				get_floor_y(),
				octree_ptr,
				n_octree_levels,
				stream
			);
		}
	};

	uint32_t n_hit;
	if (m_render_mode == ERenderMode::Slice) {
		n_hit = tracer.n_rays_initialized();
	} else {
		n_hit = trace(tracer);
	}
	RaysSdfSoa& rays_hit = m_render_mode == ERenderMode::Slice || gt_raytrace ? tracer.rays_init() : tracer.rays_hit();

	if (m_render_mode == ERenderMode::Slice) {
		if (m_visualized_dimension == -1) {
			distance_function(n_hit, rays_hit.pos, rays_hit.distance, stream);
			extract_dimension_pos_neg_kernel<float><<<n_blocks_linear(n_hit*3), n_threads_linear, 0, stream>>>(n_hit*3, 0, 1, 3, rays_hit.distance, CM, (float*)rays_hit.normal);
		} else {
			// Store colors in the normal buffer
			uint32_t n_elements = next_multiple(n_hit, tcnn::batch_size_granularity);

			GPUMatrix<float> positions_matrix((float*)rays_hit.pos, 3, n_elements);
			GPUMatrix<float> colors_matrix((float*)rays_hit.normal, 3, n_elements);
			m_network->visualize_activation(stream, m_visualized_layer, m_visualized_dimension, positions_matrix, colors_matrix);
		}
	}

	ERenderMode render_mode = (m_visualized_dimension > -1 || m_render_mode == ERenderMode::Slice) ? ERenderMode::EncodingVis : m_render_mode;
	if (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Normals) {
		if (m_sdf.analytic_normals || gt_raytrace) {
			normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);
		} else {
			float fd_normals_epsilon = m_sdf.fd_normals_epsilon;
			if (m_render_ground_truth && m_sdf.groundtruth_mode == ESDFGroundTruthMode::SDFBricks && m_sdf.brick_smooth_normals) {
				fd_normals_epsilon = exp2f(-float(n_octree_levels)) * (1.f/(m_sdf.brick_res-1)); // in sdf brick mode, use one voxel as the normal central difference radius
			}

			FiniteDifferenceNormalsApproximator fd_normals;
			fd_normals.normal(n_hit, distance_function, rays_hit.pos, rays_hit.normal, fd_normals_epsilon, stream);
		}

		if (render_mode == ERenderMode::Shade && n_hit > 0) {
			// Shadow rays towards the sun
			SphereTracer shadow_tracer;

			shadow_tracer.init_rays_from_data(n_hit, rays_hit, stream);
			shadow_tracer.set_trace_shadow_rays(true);
			shadow_tracer.set_shadow_sharpness(m_sdf.shadow_sharpness);
			RaysSdfSoa& shadow_rays_init = shadow_tracer.rays_init();
			linear_kernel(prepare_shadow_rays, 0, stream,
				n_hit,
				m_sun_dir.normalized(),
				shadow_rays_init.pos,
				shadow_rays_init.normal,
				shadow_rays_init.distance,
				shadow_rays_init.prev_distance,
				shadow_rays_init.total_distance,
				shadow_rays_init.min_visibility,
				shadow_rays_init.payload,
				sdf_bounding_box,
				octree_ptr ? octree_ptr->nodes_gpu() : nullptr,
				n_octree_levels
			);
			uint32_t n_hit_shadow = trace(shadow_tracer);
			auto& shadow_rays_hit = gt_raytrace ? shadow_tracer.rays_init() : shadow_tracer.rays_hit();

			linear_kernel(write_shadow_ray_result, 0, stream,
				n_hit_shadow,
				sdf_bounding_box,
				shadow_rays_hit.pos,
				shadow_rays_hit.payload,
				shadow_rays_hit.min_visibility,
				rays_hit.distance
			);

			// todo: Reflection rays?
		}
	} else if (render_mode == ERenderMode::EncodingVis && m_render_mode != ERenderMode::Slice) {
		// HACK: Store colors temporarily in the normal buffer
		uint32_t n_elements = next_multiple(n_hit, tcnn::batch_size_granularity);

		GPUMatrix<float> positions_matrix((float*)rays_hit.pos, 3, n_elements);
		GPUMatrix<float> colors_matrix((float*)rays_hit.normal, 3, n_elements);
		m_network->visualize_activation(stream, m_visualized_layer, m_visualized_dimension, positions_matrix, colors_matrix);
	}

	linear_kernel(shade_kernel_sdf, 0, stream,
		n_hit,
		m_aabb,
		get_floor_y(),
		render_mode,
		m_sdf.brdf,
		m_sun_dir.normalized(),
		m_up_dir.normalized(),
		camera_matrix,
		rays_hit.pos,
		rays_hit.normal,
		rays_hit.distance,
		rays_hit.payload,
		render_buffer.frame_buffer(),
		render_buffer.depth_buffer()
	);

	if (render_mode == ERenderMode::Cost) {
		std::vector<SdfPayload> payloads_final_cpu(n_hit);
		CUDA_CHECK_THROW(cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(SdfPayload), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		size_t total_n_steps = 0;
		for (uint32_t i = 0; i < n_hit; ++i) {
			total_n_steps += payloads_final_cpu[i].n_steps;
		}
		tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps/(float)n_hit);
	}
}

void Testbed::load_mesh() {
	std::vector<Vector3f> vertices;
	if (equals_case_insensitive(m_data_path.extension(), "obj")) {
		vertices = load_obj(m_data_path.str());
	} else if (equals_case_insensitive(m_data_path.extension(), "stl")) {
		FILE* f = fopen(m_data_path.str().c_str(), "rb");
		if (!f) {
			throw std::runtime_error{"stl file not found"};
		}
		uint32_t buf[21]={};
		if (fread(buf, 4, 21, f) != 4*21) {
			throw std::runtime_error{"stl file too small for header"};
		}
		uint32_t nfaces = buf[20];
		if (memcmp(buf,"solid",5)==0 || buf[20]==0) {
			fclose(f);
			throw std::runtime_error{"ascii stl files are not supported"};
		}
		vertices.reserve(nfaces * 3);
		for (uint32_t i = 0; i < nfaces; ++i) {
			if (fread(buf, 1, 50, f) < 50) {
				nfaces = i;
				break;
			}
			vertices.push_back(*(Vector3f*)(buf + 3));
			vertices.push_back(*(Vector3f*)(buf + 6));
			vertices.push_back(*(Vector3f*)(buf + 9));
		}
		fclose(f);
	} else {
		throw std::runtime_error{"Sdf data path must be a mesh in ascii .obj or binary .stl format."};
	}
	// The expected format is
	// [v1.x][v1.y][v1.z][v2.x]...
	size_t n_vertices = vertices.size();
	size_t n_triangles = n_vertices/3;

	m_raw_aabb.min = Vector3f::Constant(std::numeric_limits<float>::infinity());
	m_raw_aabb.max = Vector3f::Constant(-std::numeric_limits<float>::infinity());
	for (size_t i = 0; i < n_vertices; ++i) {
		m_raw_aabb.enlarge(vertices[i]);
	}

	// Inflate AABB by 1% to give the network a little wiggle room.
	const float inflation = 0.005f;

	m_raw_aabb.inflate(m_raw_aabb.diag().norm() * inflation);
	m_sdf.mesh_scale = m_raw_aabb.diag().maxCoeff();

	// Normalize vertex coordinates to lie within [0,1]^3.
	// This way, none of the constants need to carry around
	// bounding box factors.
	for (size_t i = 0; i < n_vertices; ++i) {
		vertices[i] = (vertices[i] - m_raw_aabb.min - 0.5f * m_raw_aabb.diag()) / m_sdf.mesh_scale + Vector3f::Constant(0.5f);
	}

	m_aabb = {};
	for (size_t i = 0; i < n_vertices; ++i) {
		m_aabb.enlarge(vertices[i]);
	}
	m_aabb.inflate(m_aabb.diag().norm() * inflation);
	m_aabb = m_aabb.intersection(BoundingBox{Vector3f::Zero(), Vector3f::Ones()});
	m_render_aabb = m_aabb;
	m_render_aabb_to_local = Matrix3f::Identity();
	m_mesh.thresh = 0.f;

	m_sdf.triangles_cpu.resize(n_triangles);
	for (size_t i = 0; i < n_vertices; i += 3) {
		m_sdf.triangles_cpu[i/3] = {vertices[i+0], vertices[i+1], vertices[i+2]};
	}

	if (!m_sdf.triangle_bvh)
		m_sdf.triangle_bvh = TriangleBvh::make();
	m_sdf.triangle_bvh->build(m_sdf.triangles_cpu, 8);
	m_sdf.triangles_gpu.resize_and_copy_from_host(m_sdf.triangles_cpu);
	m_sdf.triangle_bvh->build_optix(m_sdf.triangles_gpu, m_stream.get());

	m_sdf.triangle_octree.reset(new TriangleOctree{});
	m_sdf.triangle_octree->build(*m_sdf.triangle_bvh, m_sdf.triangles_cpu, 10);
	m_sdf.brick_data.free_memory();

	m_bounding_radius = Vector3f::Constant(0.5f).norm();
	set_scale(m_bounding_radius * 1.5f);

	// Compute discrete probability distribution for later sampling of the mesh's surface
	m_sdf.triangle_weights.resize(n_triangles);
	for (size_t i = 0; i < n_triangles; ++i) {
		m_sdf.triangle_weights[i] = m_sdf.triangles_cpu[i].surface_area();
	}
	m_sdf.triangle_distribution.build(m_sdf.triangle_weights);

	// Move CDF to gpu
	m_sdf.triangle_cdf.resize_and_copy_from_host(m_sdf.triangle_distribution.cdf);

	// Clear training data as it's no longer representative
	// of the previously loaded mesh... but don't clear the network.
	// Perhaps it'll look interesting while morphing from one mesh to another.
	m_sdf.training.idx = 0;
	m_sdf.training.size = 0;

	tlog::success() << "Loaded mesh: triangles=" << n_triangles << " AABB=" << m_raw_aabb << " after scaling=" << m_aabb;
}

void Testbed::generate_training_samples_sdf(Vector3f* positions, float* distances, uint32_t n_to_generate, cudaStream_t stream, bool uniform_only) {
	uint32_t n_to_generate_base = n_to_generate / 8;
	const uint32_t n_to_generate_surface_exact = uniform_only ? 0 : n_to_generate_base*4;
	const uint32_t n_to_generate_surface_offset = uniform_only ? 0 : n_to_generate_base*3;
	const uint32_t n_to_generate_uniform = uniform_only ? n_to_generate : n_to_generate_base*1;

	const uint32_t n_to_generate_surface = n_to_generate_surface_exact + n_to_generate_surface_offset;

	// Generate uniform 3D samples. Some of these will be transformed to cover the surfaces uniformly. Others will be left as-is.
	generate_random_uniform<float>(stream, m_rng, n_to_generate*3, (float*)positions);

	linear_kernel(sample_uniform_on_triangle_kernel, 0, stream,
		n_to_generate_surface,
		m_sdf.triangle_cdf.data(),
		(uint32_t)m_sdf.triangle_cdf.size(),
		m_sdf.triangles_gpu.data(),
		positions
	);

	// The distances of points on the mesh are zero. Can immediately set.
	CUDA_CHECK_THROW(cudaMemsetAsync(distances, 0, n_to_generate_surface_exact*sizeof(float), stream));

	// If we have an octree, generate uniform samples within that octree.
	// Otherwise, at least confine uniform samples to the AABB.
	// (For the uniform_only case, we always use the AABB, then the IoU kernel checks against the octree later)
	float stddev = m_bounding_radius/1024.0f * m_sdf.training.surface_offset_scale;
	if (!uniform_only && (m_sdf.uses_takikawa_encoding || m_sdf.use_triangle_octree)) {
		linear_kernel(uniform_octree_sample_kernel, 0, stream,
			n_to_generate_uniform,
			m_rng,
			m_sdf.triangle_octree->nodes_gpu(),
			m_sdf.triangle_octree->n_nodes(),
			m_sdf.triangle_octree->depth(),
			positions+n_to_generate_surface
		);
		m_rng.advance();

		// If we know the finest discretization of the octree, we can concentrate
		// points MUCH closer to the mesh surface
		float leaf_size = scalbnf(1.0f, -m_sdf.triangle_octree->depth()+1);
		if (leaf_size < stddev) {
			tlog::warning() << "leaf_size < stddev";
			stddev = leaf_size;
		}

		linear_kernel(assign_float, 0, stream,
			n_to_generate_uniform,
			Vector3f::Constant(leaf_size).norm()*1.001f,
			distances+n_to_generate_surface
		);
	} else {
		BoundingBox sdf_aabb = m_aabb;
		sdf_aabb.inflate(m_sdf.zero_offset);
		linear_kernel(scale_to_aabb_kernel, 0, stream,
			n_to_generate_uniform, sdf_aabb,
			positions+n_to_generate_surface
		);

		linear_kernel(assign_float, 0, stream,
			n_to_generate_uniform,
			sdf_aabb.diag().norm()*1.001f,
			distances+n_to_generate_surface
		);
	}

	m_sdf.training.perturbations.enlarge(n_to_generate_surface_offset);
	generate_random_logistic<float>(stream, m_rng, n_to_generate_surface_offset*3, (float*)m_sdf.training.perturbations.data(), 0.0f, stddev);

	linear_kernel(perturb_sdf_samples, 0, stream,
		n_to_generate_surface_offset,
		m_sdf.training.perturbations.data(),
		positions+n_to_generate_surface_exact,
		distances+n_to_generate_surface_exact
	);

	// The following function expects `distances` to contain an upper bound on the
	// true distance. This accelerates lookups.
	m_sdf.triangle_bvh->signed_distance_gpu(
		n_to_generate_uniform+n_to_generate_surface_offset,
		m_sdf.mesh_sdf_mode,
		positions+n_to_generate_surface_exact,
		distances+n_to_generate_surface_exact,
		m_sdf.triangles_gpu.data(),
		true,
		stream
	);

	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

__global__ void generate_grid_samples_sdf_uniform(Eigen::Vector3i res_3d, BoundingBox aabb, const Matrix3f& render_aabb_to_local, Vector3f* __restrict__ out) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x>=res_3d.x() || y>=res_3d.y() || z>=res_3d.z())
		return;
	uint32_t i = x+ y*res_3d.x() + z*res_3d.x()*res_3d.y();
	Vector3f pos = Array3f{(float)x, (float)y, (float)z} * Array3f{1.f/res_3d.x(),1.f/res_3d.y(),1.f/res_3d.z()};
	pos = pos.cwiseProduct(aabb.max - aabb.min) + aabb.min;
	out[i] = render_aabb_to_local.transpose() * pos;
}

GPUMemory<float> Testbed::get_sdf_gt_on_grid(Vector3i res3d, const BoundingBox& aabb, const Matrix3f& render_aabb_to_local) {
	const uint32_t n_elements = (res3d.x()*res3d.y()*res3d.z());
	GPUMemory<float> density(n_elements);
	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		Vector3f
	>(m_stream.get(), &alloc, n_elements);
	Vector3f* positions = std::get<0>(scratch);
	float* sdf_out = density.data();
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res3d.x(), threads.x), div_round_up((uint32_t)res3d.y(), threads.y), div_round_up((uint32_t)res3d.z(), threads.z) };
	generate_grid_samples_sdf_uniform<<<blocks, threads, 0, m_stream.get()>>>(res3d, aabb, render_aabb_to_local, positions);
	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	m_sdf.triangle_bvh->signed_distance_gpu(
			n_elements,
			m_sdf.mesh_sdf_mode,
			positions,
			sdf_out,
			m_sdf.triangles_gpu.data(),
			false,
			m_stream.get()
		);
	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	/*
	std::vector<float> cpudensity(density.size());
	std::vector<Vector3f> cpupositions(n_elements);
	density.copy_to_host(cpudensity);
	cudaMemcpy(cpupositions.data(),positions,n_elements*12,cudaMemcpyDeviceToHost);
	for (int i=0;i<64;++i)
		printf("[%0.3f %0.3f %0.3f] -> %0.3f\n", cpupositions[i].x(),cpupositions[i].y(),cpupositions[i].z(),cpudensity[i]);
	*/
	return density;
}


void Testbed::train_sdf(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 1;
	const uint32_t n_input_dims = 3;

	if (m_sdf.training.size >= target_batch_size) {
		// Auxiliary matrices for training
		const uint32_t batch_size = (uint32_t)std::min(m_sdf.training.size, target_batch_size);

		// Permute all training records to de-correlate training data
		linear_kernel(shuffle<Vector3f>, 0, stream, m_sdf.training.size, 1, m_training_step, m_sdf.training.positions.data(), m_sdf.training.positions_shuffled.data());
		linear_kernel(shuffle<float>, 0, stream, m_sdf.training.size, 1, m_training_step, m_sdf.training.distances.data(), m_sdf.training.distances_shuffled.data());

		GPUMatrix<float> training_target_matrix(m_sdf.training.distances_shuffled.data(), n_output_dims, batch_size);
		GPUMatrix<float> training_batch_matrix((float*)(m_sdf.training.positions_shuffled.data()), n_input_dims, batch_size);

		auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

		m_training_step++;

		if (get_loss_scalar) {
			m_loss_scalar.update(m_trainer->loss(stream, *ctx));
		}
	}
}

void Testbed::training_prep_sdf(uint32_t batch_size, cudaStream_t stream) {
	if (m_sdf.training.generate_sdf_data_online) {
		m_sdf.training.size = batch_size;
		m_sdf.training.positions.enlarge(m_sdf.training.size);
		m_sdf.training.positions_shuffled.enlarge(m_sdf.training.size);
		m_sdf.training.distances.enlarge(m_sdf.training.size);
		m_sdf.training.distances_shuffled.enlarge(m_sdf.training.size);

		generate_training_samples_sdf(m_sdf.training.positions.data(), m_sdf.training.distances.data(), batch_size, stream, false);
	}
}

// set scale_existing_results_factor=0. to reset any existing results; set it to 1.0 to accumulate more samples onto existing results
// set it to a fraction near 1 to use a sliding EMA
// if blocking is false, then this returns the iou from the *last* call
double Testbed::calculate_iou(uint32_t n_samples, float scale_existing_results_factor, bool blocking, bool force_use_octree) {
	cudaStream_t stream = m_stream.get();
	uint32_t countercpu[8];
	m_sdf.iou_counter.enlarge(8);
	if (!blocking) // when not blocking, returns data from the *last* run, then kicks off work to accumulate some more samples
		cudaMemcpy(countercpu, m_sdf.iou_counter.data(), 8*4, cudaMemcpyDeviceToHost);

	if (scale_existing_results_factor < 1.f) {
		linear_kernel(scale_iou_counters_kernel, 0, stream, 8, m_sdf.iou_counter.data(), scale_existing_results_factor);
	}
	while (n_samples>0) {
		uint32_t batch_size = std::min(uint32_t(128*128*128), n_samples);
		m_sdf.training.size = batch_size;
		n_samples-=batch_size;
		m_sdf.training.positions.enlarge(m_sdf.training.size);
		m_sdf.training.distances.enlarge(m_sdf.training.size); // we use this buffer for the GT distances
		m_sdf.training.distances_shuffled.enlarge(m_sdf.training.size); // we use the shuffled output for the output of inference

		generate_training_samples_sdf(m_sdf.training.positions.data(), m_sdf.training.distances.data(), (uint32_t)(batch_size), stream, true);
		GPUMatrix<float> positions_matrix((float*)m_sdf.training.positions.data(), 3, batch_size);
		GPUMatrix<float> distances_matrix(m_sdf.training.distances_shuffled.data(), 1, batch_size);
		m_network->inference(stream, positions_matrix, distances_matrix);
		auto* octree_ptr = (m_sdf.uses_takikawa_encoding || m_sdf.use_triangle_octree || force_use_octree) ? m_sdf.triangle_octree.get() : nullptr;
		linear_kernel(compare_signs_kernel,0, stream, batch_size, m_sdf.training.positions.data(),
			m_sdf.training.distances.data(), //  ref
			m_sdf.training.distances_shuffled.data(), // model
			m_sdf.iou_counter.data(),
			octree_ptr ? octree_ptr->nodes_gpu() : nullptr, octree_ptr ? octree_ptr->depth() : 0);
	}
	if (blocking) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(countercpu, m_sdf.iou_counter.data(), 8*4, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	}

	return countercpu[4]/double(countercpu[5]);
}

NGP_NAMESPACE_END
