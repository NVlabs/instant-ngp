/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_buffer.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common_host.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/dlss.h>

#include <tiny-cuda-nn/gpu_memory.h>

#include <memory>
#include <vector>

namespace ngp {

typedef unsigned int GLenum;
typedef int          GLint;
typedef unsigned int GLuint;

class SurfaceProvider {
public:
	virtual cudaSurfaceObject_t surface() = 0;
	virtual cudaArray_t array() = 0;
	virtual ivec2 resolution() const = 0;
	virtual void resize(const ivec2&, int n_channels = 4) = 0;
};

class CudaSurface2D : public SurfaceProvider {
public:
	CudaSurface2D() {
		m_array = nullptr;
		m_surface = 0;
	}

	~CudaSurface2D() {
		free();
	}

	void free();

	void resize(const ivec2& size, int n_channels) override;

	cudaSurfaceObject_t surface() override {
		return m_surface;
	}

	cudaArray_t array() override {
		return m_array;
	}

	ivec2 resolution() const override {
		return m_size;
	}

private:
	ivec2 m_size = ivec2(0);
	int m_n_channels = 0;
	cudaArray_t m_array;
	cudaSurfaceObject_t m_surface;
};

#ifdef NGP_GUI
class GLTexture : public SurfaceProvider {
public:
	GLTexture() = default;
	GLTexture(const std::string& texture_name)
	: m_texture_name(texture_name), m_texture_id(0)
	{ }

	GLTexture(const GLTexture& other) = delete;

	GLTexture(GLTexture&& other)
	: m_texture_name(move(other.m_texture_name)), m_texture_id(other.m_texture_id) {
		other.m_texture_id = 0;
	}

	GLTexture& operator=(GLTexture&& other) {
		m_texture_name = move(other.m_texture_name);
		std::swap(m_texture_id, other.m_texture_id);
		return *this;
	}

	~GLTexture();

	GLuint texture();

	cudaSurfaceObject_t surface() override;

	cudaArray_t array() override;

	void blit_from_cuda_mapping();

	const std::string& texture_name() const { return m_texture_name; }

	bool is_8bit() { return m_is_8bit; }

	void load(const fs::path& path);

	void load(const float* data, ivec2 new_size, int n_channels);

	void load(const uint8_t* data, ivec2 new_size, int n_channels);

	void resize(const ivec2& new_size, int n_channels, bool is_8bit);

	void resize(const ivec2& new_size, int n_channels) override {
		resize(new_size, n_channels, false);
	}

	ivec2 resolution() const override {
		return m_size;
	}

private:
	class CUDAMapping {
	public:
		CUDAMapping(GLuint texture_id, const ivec2& size, int n_channels);
		~CUDAMapping();

		cudaSurfaceObject_t surface() const { return m_cuda_surface ? m_cuda_surface->surface() : m_surface; }

		cudaArray_t array() const { return m_cuda_surface ? m_cuda_surface->array() : m_mapped_array; }

		bool is_interop() const { return !m_cuda_surface; }

		const float* data_cpu();

	private:
		cudaGraphicsResource_t m_graphics_resource = {};
		cudaArray_t m_mapped_array = {};
		cudaSurfaceObject_t m_surface = {};

		ivec2 m_size;
		int m_n_channels;
		std::vector<float> m_data_cpu;

		std::unique_ptr<CudaSurface2D> m_cuda_surface;
	};

	std::string m_texture_name;
	GLuint m_texture_id = 0;
	ivec2 m_size = ivec2(0);
	int m_n_channels = 0;
	GLint m_internal_format;
	GLenum m_format;
	bool m_is_8bit = false;
	std::unique_ptr<CUDAMapping> m_cuda_mapping;
};
#endif //NGP_GUI

struct CudaRenderBufferView {
	vec4* frame_buffer = nullptr;
	float* depth_buffer = nullptr;
	ivec2 resolution = ivec2(0);
	uint32_t spp = 0;

	std::shared_ptr<Buffer2D<uint8_t>> hidden_area_mask = nullptr;

	void clear(cudaStream_t stream) const;
};

class CudaRenderBuffer {
public:
	CudaRenderBuffer(const std::shared_ptr<SurfaceProvider>& rgba, const std::shared_ptr<SurfaceProvider>& depth = nullptr) : m_rgba_target{rgba}, m_depth_target{depth} {}

	CudaRenderBuffer(const CudaRenderBuffer& other) = delete;
	CudaRenderBuffer& operator=(const CudaRenderBuffer& other) = delete;
	CudaRenderBuffer(CudaRenderBuffer&& other) = default;
	CudaRenderBuffer& operator=(CudaRenderBuffer&& other) = default;

	cudaSurfaceObject_t surface() {
		return m_rgba_target->surface();
	}

	ivec2 in_resolution() const {
		return m_in_resolution;
	}

	ivec2 out_resolution() const {
		return m_rgba_target->resolution();
	}

	void resize(const ivec2& res);

	void reset_accumulation() {
		m_spp = 0;
	}

	uint32_t spp() const {
		return m_spp;
	}

	void set_spp(uint32_t value) {
		m_spp = value;
	}

	vec4* frame_buffer() const {
		return m_frame_buffer.data();
	}

	float* depth_buffer() const {
		return m_depth_buffer.data();
	}

	vec4* accumulate_buffer() const {
		return m_accumulate_buffer.data();
	}

	CudaRenderBufferView view() const {
		return {
			frame_buffer(),
			depth_buffer(),
			in_resolution(),
			spp(),
			hidden_area_mask(),
		};
	}

	void clear_frame(cudaStream_t stream);

	void accumulate(float exposure, cudaStream_t stream);

	void tonemap(float exposure, const vec4& background_color, EColorSpace output_color_space, float znear, float zfar, bool snap_to_pixel_centers, cudaStream_t stream);

	void overlay_image(
		float alpha,
		const vec3& exposure,
		const vec4& background_color,
		EColorSpace output_color_space,
		const void* __restrict__ image,
		EImageDataType image_data_type,
		const ivec2& resolution,
		int fov_axis,
		float zoom,
		const vec2& screen_center,
		cudaStream_t stream
	);

	void overlay_depth(
		float alpha,
		const float* __restrict__ depth,
		float depth_scale,
		float max_depth,
		const ivec2& resolution,
		int fov_axis,
		float zoom,
		const vec2& screen_center,
		cudaStream_t stream
	);

	void overlay_false_color(ivec2 training_resolution, bool to_srgb, int fov_axis, cudaStream_t stream, const float *error_map, ivec2 error_map_resolution, const float *average, float brightness, bool viridis);

	SurfaceProvider& surface_provider() {
		return *m_rgba_target;
	}

	void set_color_space(EColorSpace color_space) {
		if (color_space != m_color_space) {
			m_color_space = color_space;
			reset_accumulation();
		}
	}

	void set_tonemap_curve(ETonemapCurve tonemap_curve) {
		if (tonemap_curve != m_tonemap_curve) {
			m_tonemap_curve = tonemap_curve;
			reset_accumulation();
		}
	}

	void enable_dlss(IDlssProvider& dlss_provider, const ivec2& max_out_res);
	void disable_dlss();
	void set_dlss_sharpening(float value) {
		m_dlss_sharpening = value;
	}

	const std::unique_ptr<IDlss>& dlss() const {
		return m_dlss;
	}

	void set_hidden_area_mask(const std::shared_ptr<Buffer2D<uint8_t>>& hidden_area_mask) {
		m_hidden_area_mask = hidden_area_mask;
	}

	const std::shared_ptr<Buffer2D<uint8_t>>& hidden_area_mask() const {
		return m_hidden_area_mask;
	}

private:
	uint32_t m_spp = 0;
	EColorSpace m_color_space = EColorSpace::Linear;
	ETonemapCurve m_tonemap_curve = ETonemapCurve::Identity;

	std::unique_ptr<IDlss> m_dlss;
	float m_dlss_sharpening = 0.0f;

	ivec2 m_in_resolution = ivec2(0);

	GPUMemory<vec4> m_frame_buffer;
	GPUMemory<float> m_depth_buffer;
	GPUMemory<vec4> m_accumulate_buffer;

	std::shared_ptr<Buffer2D<uint8_t>> m_hidden_area_mask = nullptr;

	std::shared_ptr<SurfaceProvider> m_rgba_target;
	std::shared_ptr<SurfaceProvider> m_depth_target;
};

}
