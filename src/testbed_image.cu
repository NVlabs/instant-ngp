/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_image.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/random_val.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/trainer.h>

#include <fstream>

namespace ngp {

Testbed::NetworkDims Testbed::network_dims_image() const {
	NetworkDims dims;
	dims.n_input = 2;
	dims.n_output = 3;
	dims.n_pos = 2;
	return dims;
}

__global__ void halton23_kernel(uint32_t n_elements, size_t base_idx, vec2* __restrict__ output) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	output[i] = {halton<2>(base_idx+i), halton<3>(base_idx+i)};
}

__global__ void sobol2_kernel(uint32_t n_elements, size_t base_idx, uint32_t seed, vec2* __restrict__ output) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	output[i] = ld_random_val_2d(base_idx + i, seed);
}

__global__ void zip_kernel(uint32_t n_elements, const float* __restrict__ in, vec2* __restrict__ output) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	output[i] = {in[i], in[i+n_elements]};
}

__global__ void stratify2_kernel(uint32_t n_elements, uint32_t log2_batch_size, vec2* __restrict__ inout) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t log2Size = log2_batch_size / 2;
	uint32_t size = 1 << log2Size;

	uint32_t in_batch_index = i & ((1 << log2_batch_size)-1);

	uint32_t x = in_batch_index & ((1 << log2Size)-1);
	uint32_t y = in_batch_index >> log2Size;

	vec2 val = inout[i];
	inout[i] = {val.x / size + ((float)x/size), val.y / size + ((float)y/size)};
}

__global__ void init_image_coords(
	uint32_t sample_index,
	vec2* __restrict__ positions,
	float* __restrict__ depth_buffer,
	ivec2 resolution,
	float aspect,
	vec2 focal_length,
	mat4x3 camera_matrix,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Buffer2DView<const uint8_t> hidden_area_mask
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	// The image is displayed on the plane [0.5, 0.5, 0.5] + [X, Y, 0] to facilitate
	// a top-down view by default, while permitting general camera movements (for
	// motion vectors and code sharing with 3D tasks).
	// Hence: generate rays and intersect that plane.
	Ray ray = pixel_to_ray(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		0.0f, // near distance
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask
	);

	// Intersect the Z=0.5 plane
	float t = ray.is_valid() ? (0.5f - ray.o.z) / ray.d.z : -1.0f;

	uint32_t idx = x + resolution.x * y;
	if (t <= 0.0f) {
		depth_buffer[idx] = MAX_DEPTH();
		positions[idx] = -vec2(1.0f);
		return;
	}

	vec2 uv = ray(t).xy();

	// Flip from world coordinates where Y goes up to image coordinates where Y goes down.
	// Also, multiply the x-axis by the image's aspect ratio to make it have the right proportions.
	uv = (uv - 0.5f) * vec2{aspect, -1.0f} + 0.5f;

	depth_buffer[idx] = t;
	positions[idx] = uv;
}

__global__ void shade_kernel_image(ivec2 resolution, const vec2* __restrict__ positions, const vec3* __restrict__ colors, vec4* __restrict__ frame_buffer, bool linear_colors) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	const vec2 uv = positions[idx];
	if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
		frame_buffer[idx] = vec4(0.0f);
		return;
	}

	vec3 color = colors[idx];

	if (!linear_colors) {
		color = srgb_to_linear(color);
	}

	frame_buffer[idx] = {color.x, color.y, color.z, 1.0f};
}

template <typename T, uint32_t stride>
__global__ void eval_image_kernel_and_snap(uint32_t n_elements, const T* __restrict__ texture, vec2* __restrict__ positions, ivec2 resolution, float* __restrict__ result, bool snap_to_pixel_centers, bool linear_colors) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t output_idx = i * stride;

	vec2 pos = positions[i];

	auto read_val = [&](int x, int y) {
		auto val = ((tvec<T, 4>*)texture)[y * resolution.x + x];
		vec4 result{(float)val[0], (float)val[1], (float)val[2], (float)val[3]};
		if (!linear_colors) {
			result.rgb() = linear_to_srgb(result.rgb());
		}
		return result;
	};

	vec4 val;
	if (snap_to_pixel_centers) {
		ivec2 pos_int = floor(pos * vec2(resolution));
		positions[i] = (vec2(pos_int) + 0.5f) / vec2(resolution);
		pos_int = clamp(pos_int, 0, resolution - 1);
		val = read_val(pos_int.x, pos_int.y);
	} else {
		pos = clamp(pos * vec2(resolution) - 0.5f, 0.0f, vec2(resolution) - (1.0f + 1e-4f));

		const ivec2 pos_int = pos;
		const vec2 weight = pos - vec2(pos_int);

		const ivec2 idx = clamp(pos_int, 0, resolution - 2);

		val =
			(1 - weight.x) * (1 - weight.y) * read_val(idx.x, idx.y) +
			(weight.x) * (1 - weight.y) * read_val(idx.x+1, idx.y) +
			(1 - weight.x) * (weight.y) * read_val(idx.x, idx.y+1) +
			(weight.x) * (weight.y) * read_val(idx.x+1, idx.y+1);
	}

	result[output_idx + 0] = val.x;
	result[output_idx + 1] = val.y;
	result[output_idx + 2] = val.z;

	for (uint32_t i = 3; i < stride; ++i) {
		result[output_idx + i] = 1;
	}
}

void Testbed::train_image(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 3;
	const uint32_t n_input_dims = 2;

	const uint32_t batch_size = (uint32_t)target_batch_size;

	const uint32_t n_elements = batch_size;
	m_image.training.positions.enlarge(n_elements);
	m_image.training.targets.enlarge(n_elements);

	auto generate_training_data = [&]() {
		if (m_image.random_mode == ERandomMode::Halton) {
			linear_kernel(halton23_kernel, 0, stream, n_elements, (size_t)batch_size * m_training_step, m_image.training.positions.data());
		} else if (m_image.random_mode == ERandomMode::Sobol) {
			linear_kernel(sobol2_kernel, 0, stream, n_elements, (size_t)batch_size * m_training_step, m_seed, m_image.training.positions.data());
		} else {
			generate_random_uniform<float>(stream, m_rng, n_elements * n_input_dims, (float*)m_image.training.positions.data());
			if (m_image.random_mode == ERandomMode::Stratified) {
				uint32_t log2_batch_size = 0;
				if (!is_pot(batch_size, &log2_batch_size)) {
					tlog::warning() << "Can't stratify a non-pot batch size";
				} else if (log2_batch_size % 2 != 0) {
					tlog::warning() << "Can't stratify a non-square batch size";
				} else {
					linear_kernel(stratify2_kernel, 0, stream, n_elements, log2_batch_size, m_image.training.positions.data());
				}
			}
		}

		if (m_image.type == EDataType::Float) {
			linear_kernel(eval_image_kernel_and_snap<float, 3>, 0, stream,
				n_elements,
				(float*)m_image.data.data(),
				m_image.training.positions.data(),
				m_image.resolution,
				(float*)m_image.training.targets.data(),
				m_image.training.snap_to_pixel_centers,
				m_image.training.linear_colors
			);
		} else {
			linear_kernel(eval_image_kernel_and_snap<__half, 3>, 0, stream,
				n_elements,
				(__half*)m_image.data.data(),
				m_image.training.positions.data(),
				m_image.resolution,
				(float*)m_image.training.targets.data(),
				m_image.training.snap_to_pixel_centers,
				m_image.training.linear_colors
			);
		}
	};

	generate_training_data();

	GPUMatrix<float> training_batch_matrix((float*)(m_image.training.positions.data()), n_input_dims, batch_size);
	GPUMatrix<float> training_target_matrix((float*)(m_image.training.targets.data()), n_output_dims, batch_size);

	auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);
	if (get_loss_scalar) {
		m_loss_scalar.update(m_trainer->loss(stream, *ctx));
	}

	m_training_step++;
}

void Testbed::render_image(
	cudaStream_t stream,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation,
	int visualized_dimension
) {
	auto res = render_buffer.resolution;

	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)res.x * res.y;
	uint32_t n_elements = next_multiple((uint32_t)n_pixels, BATCH_SIZE_GRANULARITY);
	m_image.render_coords.enlarge(n_elements);
	m_image.render_out.enlarge(n_elements);

	float plane_z = m_slice_plane_z + m_scale;
	float aspect = (float)m_image.resolution.y / (float)m_image.resolution.x;

	// Generate 2D coords at which to query the network
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };
	init_image_coords<<<blocks, threads, 0, stream>>>(
		render_buffer.spp,
		m_image.render_coords.data(),
		render_buffer.depth_buffer,
		res,
		aspect,
		focal_length,
		camera_matrix,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		plane_z,
		m_aperture_size,
		foveation,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{}
	);

	// Obtain colors for each 2D coord
	if (m_image.type == EDataType::Float) {
		linear_kernel(eval_image_kernel_and_snap<float, 3>, 0, stream,
			n_elements,
			(float*)m_image.data.data(),
			m_image.render_coords.data(),
			m_image.resolution,
			(float*)m_image.render_out.data(),
			m_image.training.snap_to_pixel_centers,
			m_image.training.linear_colors
		);
	} else {
		linear_kernel(eval_image_kernel_and_snap<__half, 3>, 0, stream,
			n_elements,
			(__half*)m_image.data.data(),
			m_image.render_coords.data(),
			m_image.resolution,
			(float*)m_image.render_out.data(),
			m_image.training.snap_to_pixel_centers,
			m_image.training.linear_colors
		);
	}

	if (!m_render_ground_truth) {
		if (visualized_dimension >= 0) {
			GPUMatrix<float> positions_matrix((float*)m_image.render_coords.data(), 2, n_elements);
			GPUMatrix<float> colors_matrix((float*)m_image.render_out.data(), 3, n_elements);
			m_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, colors_matrix);
		} else {
			GPUMatrix<float> positions_matrix((float*)m_image.render_coords.data(), 2, n_elements);
			GPUMatrix<float> colors_matrix((float*)m_image.render_out.data(), 3, n_elements);
			m_network->inference(stream, positions_matrix, colors_matrix);
		}
	}

	// Splat colors to render texture
	shade_kernel_image<<<blocks, threads, 0, stream>>>(
		res,
		m_image.render_coords.data(),
		m_image.render_out.data(),
		render_buffer.frame_buffer,
		m_image.training.linear_colors
	);
}

void Testbed::load_image(const fs::path& data_path) {
	if (equals_case_insensitive(data_path.extension(), "exr")) {
		load_exr_image(data_path);
	} else if (equals_case_insensitive(data_path.extension(), "bin")) {
		load_binary_image(data_path);
	} else {
		load_stbi_image(data_path);
	}

	m_aabb = m_render_aabb = BoundingBox{vec3(0.0f), vec3(1.0f)};
	m_render_aabb_to_local = mat3::identity();

	tlog::success()
		<< "Loaded a " << (m_image.type == EDataType::Half ? "half" : "full") << "-precision image with "
		<< m_image.resolution.x << "x" << m_image.resolution.y << " pixels.";
}

void Testbed::load_exr_image(const fs::path& data_path) {
	if (!data_path.exists()) {
		throw std::runtime_error{fmt::format("Image file '{}' does not exist.", data_path.str())};
	}

	tlog::info() << "Loading EXR image from " << data_path;

	// First step: load an image that we'd like to learn
	GPUMemory<float> image = load_exr_gpu(data_path, &m_image.resolution.x, &m_image.resolution.y);
	m_image.data.resize(image.size() * sizeof(float));
	CUDA_CHECK_THROW(cudaMemcpy(m_image.data.data(), image.data(), image.size() * sizeof(float), cudaMemcpyDeviceToDevice));

	m_image.type = EDataType::Float;
}

void Testbed::load_stbi_image(const fs::path& data_path) {
	if (!data_path.exists()) {
		throw std::runtime_error{fmt::format("Image file '{}' does not exist.", data_path.str())};
	}

	tlog::info() << "Loading STBI image from " << data_path;

	// First step: load an image that we'd like to learn
	GPUMemory<float> image = load_stbi_gpu(data_path, &m_image.resolution.x, &m_image.resolution.y);
	m_image.data.resize(image.size() * sizeof(float));
	CUDA_CHECK_THROW(cudaMemcpy(m_image.data.data(), image.data(), image.size() * sizeof(float), cudaMemcpyDeviceToDevice));

	m_image.type = EDataType::Float;
}


void Testbed::load_binary_image(const fs::path& data_path) {
	if (!data_path.exists()) {
		throw std::runtime_error{fmt::format("Image file '{}' does not exist.", data_path.str())};
	}

	tlog::info() << "Loading binary image from " << data_path;

	std::ifstream f{native_string(data_path), std::ios::in | std::ios::binary};
	f.read(reinterpret_cast<char*>(&m_image.resolution.y), sizeof(int));
	f.read(reinterpret_cast<char*>(&m_image.resolution.x), sizeof(int));

	size_t n_pixels = (size_t)m_image.resolution.x * m_image.resolution.y;
	m_image.data.resize(n_pixels * 4 * sizeof(__half));

	std::vector<__half> image(n_pixels * 4);
	f.read(reinterpret_cast<char*>(image.data()), sizeof(__half) * image.size());
	CUDA_CHECK_THROW(cudaMemcpy(m_image.data.data(), image.data(), image.size() * sizeof(__half), cudaMemcpyHostToDevice));
	m_image.type = EDataType::Half;
}

__global__ void image_coords_from_idx(const uint32_t n_elements, uint32_t offset, vec2* __restrict__ pos, ivec2 resolution) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	const uint32_t idx = i + offset;

	int x = idx % resolution.x;
	int y = idx / resolution.x;

	pos[i] = (vec2(clamp(ivec2{x, y}, 0, resolution - 1)) + 0.5f) / vec2(resolution);
}

__global__ void image_mse_kernel(const uint32_t n_elements, const vec3* __restrict__ target, const vec3* __restrict__ prediction, float* __restrict__ result, bool quantize_to_byte) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	vec3 pred = prediction[i];
	if (quantize_to_byte) {
		pred = vec3(clamp(ivec3(pred * 255.0f + 0.5f), 0, 255)) / 255.0f;
	}

	const vec3 diff = target[i] - pred;
	result[i] = dot(diff, diff) / 3.0f;
}

float Testbed::compute_image_mse(bool quantize_to_byte) {
	const uint32_t n_output_dims = 3;
	const uint32_t n_input_dims = 2;

	// Auxiliary matrices for training
	const uint32_t n_elements = product(m_image.resolution);
	const uint32_t max_batch_size = 1u<<20;

	GPUMemory<float> se(n_elements);
	GPUMemory<vec2> pos(max_batch_size);
	GPUMemory<vec3> targets(max_batch_size);
	GPUMemory<vec3> predictions(max_batch_size);
	const uint32_t n_batches = div_round_up(n_elements, max_batch_size);
	for (uint32_t i = 0; i < n_batches; ++i) {
		uint32_t offset = i * max_batch_size;
		uint32_t batch_size = (std::min(max_batch_size, n_elements - offset) + 255u ) & (~255u);

		GPUMatrix<float> pos_matrix((float*)(pos.data()), n_input_dims, batch_size);
		GPUMatrix<float> targets_matrix((float*)(targets.data()), n_output_dims, batch_size);
		GPUMatrix<float> predictions_matrix((float*)(predictions.data()), n_output_dims, batch_size);

		linear_kernel(image_coords_from_idx, 0, nullptr,
			batch_size,
			offset,
			pos.data(),
			m_image.resolution
		);

		if (m_image.type == EDataType::Float) {
			linear_kernel(eval_image_kernel_and_snap<float, 3>, 0, nullptr,
				batch_size,
				(float*)m_image.data.data(),
				pos.data(),
				m_image.resolution,
				(float*)targets.data(),
				true,
				m_image.training.linear_colors
			);
		} else {
			linear_kernel(eval_image_kernel_and_snap<__half, 3>, 0, nullptr,
				batch_size,
				(__half*)m_image.data.data(),
				pos.data(),
				m_image.resolution,
				(float*)targets.data(),
				true,
				m_image.training.linear_colors
			);
		}

		m_network->inference(pos_matrix, predictions_matrix);

		linear_kernel(image_mse_kernel, 0, nullptr,
			batch_size,
			targets.data(),
			predictions.data(),
			se.data() + offset,
			quantize_to_byte
		);
	}

	return reduce_sum(se.data(), n_elements, nullptr) / n_elements;
}

}
