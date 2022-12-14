/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_buffer.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>

#include <tiny-cuda-nn/gpu_memory.h>

#include <filesystem/path.h>

#ifdef NGP_GUI
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#  include <cuda_gl_interop.h>
#endif

#include <stb_image/stb_image.h>

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

extern std::atomic<size_t> g_total_n_bytes_allocated;

void CudaSurface2D::free() {
	if (m_surface) {
		cudaDestroySurfaceObject(m_surface);
	}
	m_surface = 0;
	if (m_array) {
		cudaFreeArray(m_array);
		g_total_n_bytes_allocated -= m_size.prod() * sizeof(float4);
	}
	m_array = nullptr;
}

void CudaSurface2D::resize(const Vector2i& size) {
	if (size == m_size) {
		return;
	}

	free();

	m_size = size;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
	CUDA_CHECK_THROW(cudaMallocArray(&m_array, &desc, size.x(), size.y(), cudaArraySurfaceLoadStore));

	g_total_n_bytes_allocated += m_size.prod() * sizeof(float4);

	struct cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(resource_desc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = m_array;
	CUDA_CHECK_THROW(cudaCreateSurfaceObject(&m_surface, &resource_desc));
}

#ifdef NGP_GUI
GLTexture::~GLTexture() {
	m_cuda_mapping.reset();
	if (m_texture_id) {
		glDeleteTextures(1, &m_texture_id);
	}
}

GLuint GLTexture::texture() {
	if (!m_texture_id) {
		glGenTextures(1, &m_texture_id);
	}

	return m_texture_id;
}

cudaSurfaceObject_t GLTexture::surface() {
	if (!m_cuda_mapping) {
		m_cuda_mapping = std::make_unique<CUDAMapping>(texture(), m_size);
	}
	return m_cuda_mapping->surface();
}

cudaArray_t GLTexture::array() {
	if (!m_cuda_mapping) {
		m_cuda_mapping = std::make_unique<CUDAMapping>(texture(), m_size);
	}
	return m_cuda_mapping->array();
}

void GLTexture::blit_from_cuda_mapping() {
	if (!m_cuda_mapping || m_cuda_mapping->is_interop()) {
		return;
	}

	if (m_internal_format != GL_RGBA32F || m_format != GL_RGBA || m_is_8bit) {
		throw std::runtime_error{"Can only blit from CUDA mapping if the texture is RGBA float."};
	}

	const float* data_cpu = m_cuda_mapping->data_cpu();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_size.x(), m_size.y(), 0, GL_RGBA, GL_FLOAT, data_cpu);
}

void GLTexture::load(const char* fname) {
	uint8_t* out; // width * height * RGBA
	int comp,width,height;
	out = stbi_load(fname, &width, &height, &comp, 4);
	if (!out) {
		throw std::runtime_error{std::string{stbi_failure_reason()}};
	}
	ScopeGuard mem_guard{[&]() { stbi_image_free(out); }};
	load(out, { width, height }, 4);
}

void GLTexture::load(const float* data, Vector2i new_size, int n_channels) {
	resize(new_size, n_channels, false);

	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexImage2D(GL_TEXTURE_2D, 0, m_internal_format, new_size.x(), new_size.y(), 0, m_format, GL_FLOAT, data);
}

void GLTexture::load(const uint8_t* data, Vector2i new_size, int n_channels) {
	resize(new_size, n_channels, true);

	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexImage2D(GL_TEXTURE_2D, 0, m_internal_format, new_size.x(), new_size.y(), 0, m_format, GL_UNSIGNED_BYTE, data);
}

void GLTexture::resize(const Vector2i& new_size, int n_channels, bool is_8bit) {
	if (m_size == new_size && m_n_channels == n_channels && m_is_8bit == is_8bit) {
		return;
	}

	if (m_texture_id) {
		m_cuda_mapping.reset();
		glDeleteTextures(1, &m_texture_id);
		m_texture_id = 0;
	}

	glGenTextures(1, &m_texture_id);
	glBindTexture(GL_TEXTURE_2D, m_texture_id);

	switch (n_channels) {
		case 1: m_internal_format = is_8bit ? GL_R8    : GL_R32F;    m_format = GL_RED;  break;
		case 2: m_internal_format = is_8bit ? GL_RG8   : GL_RG32F;   m_format = GL_RG;   break;
		case 3: m_internal_format = is_8bit ? GL_RGB8  : GL_RGB32F;  m_format = GL_RGB;  break;
		case 4: m_internal_format = is_8bit ? GL_RGBA8 : GL_RGBA32F; m_format = GL_RGBA; break;
		default: tlog::error() << "Unsupported number of channels: " << n_channels;
	}
	m_is_8bit = is_8bit;
	m_size = new_size;
	m_n_channels = n_channels;

	glTexImage2D(GL_TEXTURE_2D, 0, m_internal_format, new_size.x(), new_size.y(), 0, m_format, is_8bit ? GL_UNSIGNED_BYTE : GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

static bool is_wsl() {
#ifdef _WIN32
	return false;
#else
	fs::path path = "/proc/sys/kernel/osrelease";
	if (!path.exists()) {
		return false;
	}

	std::ifstream f{path.str()};
	std::string content((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	return content.find("microsoft") != std::string::npos;
#endif
}

GLTexture::CUDAMapping::CUDAMapping(GLuint texture_id, const Vector2i& size) : m_size{size} {
	static bool s_is_cuda_interop_supported = !is_wsl();
	if (s_is_cuda_interop_supported) {
		cudaError_t err = cudaGraphicsGLRegisterImage(&m_graphics_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		if (err != cudaSuccess) {
			s_is_cuda_interop_supported = false;
			cudaGetLastError(); // Reset error
		}
	}

	if (!s_is_cuda_interop_supported) {
		// falling back to a regular cuda surface + CPU copy of data
		m_cuda_surface = std::make_unique<CudaSurface2D>();
		m_cuda_surface->resize(size);
		m_data_cpu.resize(m_size.prod() * 4);
		return;
	}

	CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &m_graphics_resource));
	CUDA_CHECK_THROW(cudaGraphicsSubResourceGetMappedArray(&m_mapped_array, m_graphics_resource, 0, 0));

	struct cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(resource_desc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = m_mapped_array;

	CUDA_CHECK_THROW(cudaCreateSurfaceObject(&m_surface, &resource_desc));
}

GLTexture::CUDAMapping::~CUDAMapping() {
	if (m_surface) {
		cudaDestroySurfaceObject(m_surface);
		cudaGraphicsUnmapResources(1, &m_graphics_resource);
		cudaGraphicsUnregisterResource(m_graphics_resource);
	}
}

const float* GLTexture::CUDAMapping::data_cpu() {
	CUDA_CHECK_THROW(cudaMemcpy2DFromArray(m_data_cpu.data(), m_size.x() * sizeof(float) * 4, array(), 0, 0, m_size.x() * sizeof(float) * 4, m_size.y(), cudaMemcpyDeviceToHost));
	return m_data_cpu.data();
}
#endif //NGP_GUI

__global__ void accumulate_kernel(Vector2i resolution, Array4f* frame_buffer, Array4f* accumulate_buffer, float sample_count, EColorSpace color_space) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	Array4f color = frame_buffer[idx];
	Array4f tmp = accumulate_buffer[idx];

	switch (color_space) {
		case EColorSpace::VisPosNeg:
			{
				float val = color.x() - color.y();
				float tmp_val = tmp.x() - tmp.y();

				tmp_val = (tmp_val * sample_count + val) / (sample_count+1);

				tmp.x() = fmaxf(tmp_val, 0.0f);
				tmp.y() = fmaxf(-tmp_val, 0.0f);
				break;
			}
		case EColorSpace::SRGB:
			color.head<3>() = linear_to_srgb(color.head<3>());
			// fallthrough is intended!
		case EColorSpace::Linear:
			tmp.head<3>() = (tmp.head<3>() * sample_count + color.head<3>()) / (sample_count+1); break;
	}

	tmp.w() = (tmp.w() * sample_count + color.w()) / (sample_count+1);

	accumulate_buffer[idx] = tmp;
}

__device__ Array3f tonemap(Array3f x, ETonemapCurve curve) {
	if (curve == ETonemapCurve::Identity) {
		return x;
	}

	x = x.cwiseMax(0.f);

	float k0, k1, k2, k3, k4, k5;
	if (curve == ETonemapCurve::ACES) {
		// Source:  ACES approximation : https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
		// Include pre - exposure cancelation in constants
		k0 = 0.6f * 0.6f * 2.51f;
		k1 = 0.6f * 0.03f;
		k2 = 0.0f;
		k3 = 0.6f * 0.6f * 2.43f;
		k4 = 0.6f * 0.59f;
		k5 = 0.14f;
	} else if (curve == ETonemapCurve::Hable) {
		// Source: https://64.github.io/tonemapping/
		const float A = 0.15f;
		const float B = 0.50f;
		const float C = 0.10f;
		const float D = 0.20f;
		const float E = 0.02f;
		const float F = 0.30f;
		k0 = A * F - A * E;
		k1 = C * B * F - B * E;
		k2 = 0.0f;
		k3 = A * F;
		k4 = B * F;
		k5 = D * F * F;

		const float W = 11.2f;
		const float nom = k0 * (W*W) + k1 * W + k2;
		const float denom = k3 * (W*W) + k4 * W + k5;
		const float white_scale = denom / nom;

		// Include white scale and exposure bias in rational polynomial coefficients
		k0 = 4.0f * k0 * white_scale;
		k1 = 2.0f * k1 * white_scale;
		k2 = k2 * white_scale;
		k3 = 4.0f * k3;
		k4 = 2.0f * k4;
	} else { //if (curve == ETonemapCurve::Reinhard)
		const Vector3f luminance_coefficients = Vector3f(0.2126f, 0.7152f, 0.0722f);
		float Y = luminance_coefficients.dot(x.matrix());

		return x * (1.f / (Y + 1.0f));
	}

	Array3f color_sq = x * x;
	Array3f nom = color_sq * k0 + k1 * x + k2;
	Array3f denom = k3 * color_sq + k4 * x + k5;

	Array3f tonemapped_color = nom / denom;

	return tonemapped_color;
}

__device__ Array3f tonemap(Array3f col, const Array3f& exposure, ETonemapCurve tonemap_curve, EColorSpace color_space, EColorSpace output_color_space) {
	// Conversion to output by
	// 1. converting to linear. (VisPosNeg is treated as linear red/green)
	if (color_space == EColorSpace::SRGB) {
		col = srgb_to_linear(col);
	}

	// 2. applying exposure in linear space
	col *= Array3f::Constant(2.0f).pow(exposure);

	// 3. tonemapping in linear space according to the specified curve
	col = tonemap(col, tonemap_curve);

	// 4. converting to output color space.
	if (output_color_space == EColorSpace::SRGB) {
		col = linear_to_srgb(col);
	}

	return col;
}

__global__ void overlay_image_kernel(
	Vector2i resolution,
	float alpha,
	Array3f exposure,
	Array4f background_color,
	const void* __restrict__ image,
	EImageDataType image_data_type,
	Vector2i image_resolution,
	ETonemapCurve tonemap_curve,
	EColorSpace color_space,
	EColorSpace output_color_space,
	int fov_axis,
	float zoom,
	Eigen::Vector2f screen_center,
	cudaSurfaceObject_t surface
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	float scale = image_resolution[fov_axis] / float(resolution[fov_axis]);

	float fx = x+0.5f;
	float fy = y+0.5f;

	fx-=resolution.x()*0.5f; fx/=zoom; fx+=screen_center.x() * resolution.x();
	fy-=resolution.y()*0.5f; fy/=zoom; fy+=screen_center.y() * resolution.y();

	float u = (fx-resolution.x()*0.5f) * scale  + image_resolution.x()*0.5f;
	float v = (fy-resolution.y()*0.5f) * scale  + image_resolution.y()*0.5f;

	int srcx = floorf(u);
	int srcy = floorf(v);
	uint32_t idx = x + resolution.x() * y;
	uint32_t srcidx = srcx + image_resolution.x() * srcy;

	Array4f val;
	if (srcx >= image_resolution.x() || srcy >= image_resolution.y() || srcx < 0 || srcy < 0) {
		val = Array4f::Zero();
	} else {
		val = read_rgba(Vector2i{srcx, srcy}, image_resolution, image, image_data_type);
	}

	Array4f color = {val[0], val[1], val[2], val[3]};

	// The background color is represented in SRGB, so convert
	// to linear if that's not the space in which we're rendering.
	if (color_space != EColorSpace::SRGB) {
		background_color.head<3>() = srgb_to_linear(background_color.head<3>());
	} else {
		if (color.w() > 0) {
			color.head<3>() = linear_to_srgb(color.head<3>() / color.w()) * color.w();
		} else {
			color.head<3>() = Array3f::Zero();
		}
	}

	float weight = (1 - color.w()) * background_color.w();
	color.head<3>() += background_color.head<3>() * weight;
	color.w() += weight;

	color.head<3>() = tonemap(color.head<3>(), exposure, tonemap_curve, color_space, output_color_space);

	Array4f prev_color;
	surf2Dread((float4*)&prev_color, surface, x * sizeof(float4), y);
	color = color * alpha + prev_color * (1.f-alpha);
	surf2Dwrite(to_float4(color), surface, x * sizeof(float4), y);
}

__device__ Array3f colormap_turbo(float x) {
	const Vector4f kRedVec4 =   Vector4f(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
	const Vector4f kGreenVec4 = Vector4f(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
	const Vector4f kBlueVec4 =  Vector4f(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
	const Vector2f kRedVec2 =   Vector2f(-152.94239396f, 59.28637943f);
	const Vector2f kGreenVec2 = Vector2f(4.27729857f, 2.82956604f);
	const Vector2f kBlueVec2 =  Vector2f(-89.90310912f, 27.34824973f);

	x = __saturatef(x);
	Vector4f v4 = Vector4f{ 1.0f, x, x * x, x * x * x };
	Vector2f v2 = Vector2f{ v4.w() * x, v4.w() * v4.z() };
	return Array3f{
		v4.dot(kRedVec4)   + v2.dot(kRedVec2),
		v4.dot(kGreenVec4) + v2.dot(kGreenVec2),
		v4.dot(kBlueVec4)  + v2.dot(kBlueVec2)
	};
}

__global__ void overlay_depth_kernel(
	Vector2i resolution,
	float alpha,
	const float* __restrict__ depth,
	float depth_scale,
	Vector2i image_resolution,
	int fov_axis,
	float zoom, Eigen::Vector2f screen_center,
	cudaSurfaceObject_t surface
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	float scale = image_resolution[fov_axis] / float(resolution[fov_axis]);

	float fx = x+0.5f;
	float fy = y+0.5f;

	fx-=resolution.x()*0.5f; fx/=zoom; fx+=screen_center.x() * resolution.x();
	fy-=resolution.y()*0.5f; fy/=zoom; fy+=screen_center.y() * resolution.y();

	float u = (fx-resolution.x()*0.5f) * scale  + image_resolution.x()*0.5f;
	float v = (fy-resolution.y()*0.5f) * scale  + image_resolution.y()*0.5f;

	int srcx = floorf(u);
	int srcy = floorf(v);
	uint32_t idx = x + resolution.x() * y;
	uint32_t srcidx = srcx + image_resolution.x() * srcy;

	Array4f color;
	if (srcx >= image_resolution.x() || srcy >= image_resolution.y() || srcx < 0 || srcy < 0) {
		color = {0.0f, 0.0f, 0.0f, 0.0f};
	} else {
		float depth_value = depth[srcidx] * depth_scale;
		Array3f c = colormap_turbo(depth_value);
		color = {c[0], c[1], c[2], 1.0f};
	}

	Array4f prev_color;
	surf2Dread((float4*)&prev_color, surface, x * sizeof(float4), y);
	color = color * alpha + prev_color * (1.f-alpha);
	surf2Dwrite(to_float4(color), surface, x * sizeof(float4), y);
}

__device__ Array3f colormap_viridis(float x) {
	const Array3f c0 = Array3f{0.2777273272234177f, 0.005407344544966578f, 0.3340998053353061f};
	const Array3f c1 = Array3f{0.1050930431085774f, 1.404613529898575f, 1.384590162594685f};
	const Array3f c2 = Array3f{-0.3308618287255563f, 0.214847559468213f, 0.09509516302823659f};
	const Array3f c3 = Array3f{-4.634230498983486f, -5.799100973351585f, -19.33244095627987f};
	const Array3f c4 = Array3f{6.228269936347081f, 14.17993336680509f, 56.69055260068105f};
	const Array3f c5 = Array3f{4.776384997670288f, -13.74514537774601f, -65.35303263337234f};
	const Array3f c6 = Array3f{-5.435455855934631f, 4.645852612178535f, 26.3124352495832f};
	x = __saturatef(x);
	return (c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*c6))))));
}

__global__ void overlay_false_color_kernel(Vector2i resolution, Vector2i training_resolution, bool to_srgb, int fov_axis, cudaSurfaceObject_t surface, const float *error_map, Vector2i error_map_resolution, const float *average, float brightness, bool viridis) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	float error_map_scale = brightness/(0.0000001f+average[0]); // average maps to 1/16th

	float scale = training_resolution[fov_axis] / float(resolution[fov_axis]);
	float u = (x+0.5f-resolution.x()*0.5f) * scale + training_resolution.x()*0.5f;
	float v = (y+0.5f-resolution.y()*0.5f) * scale + training_resolution.y()*0.5f;
	int srcx = floorf(u * error_map_resolution.x() / float(max(1.f, (float)training_resolution.x())));
	int srcy = floorf(v * error_map_resolution.y() / float(max(1.f, (float)training_resolution.y())));

	uint32_t idx = x + resolution.x() * y;
	uint32_t srcidx = srcx + error_map_resolution.x() * srcy;

	if (srcx >= error_map_resolution.x() || srcy >= error_map_resolution.y() || srcx<0 || srcy<0) {
		return;
	}

	float err = error_map[srcidx] * error_map_scale;
	if (viridis) {
		err *= 1.f / (1.f+err);
	}
	Array4f color;
	surf2Dread((float4*)&color, surface, x * sizeof(float4), y);
	Array3f c = viridis ? colormap_viridis(err) : colormap_turbo(err);
	float grey = color.x() * 0.2126f + color.y() * 0.7152f + color.z() * 0.0722f;
	color.x() = grey*__saturatef(c.x());
	color.y() = grey*__saturatef(c.y());
	color.z() = grey*__saturatef(c.z());

	surf2Dwrite(to_float4(color), surface, x * sizeof(float4), y);
}

__global__ void tonemap_kernel(Vector2i resolution, float exposure, Array4f background_color, Array4f* accumulate_buffer, EColorSpace color_space, EColorSpace output_color_space, ETonemapCurve tonemap_curve, bool clamp_output_color, cudaSurfaceObject_t surface) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	// The background color is represented in SRGB, so convert
	// to linear if that's not the space in which we're rendering.
	if (color_space != EColorSpace::SRGB) {
		background_color.head<3>() = srgb_to_linear(background_color.head<3>());
	}

	Array4f color = accumulate_buffer[idx];
	float weight = (1 - color.w()) * background_color.w();
	color.head<3>() += background_color.head<3>() * weight;
	color.w() += weight;

	color.head<3>() = tonemap(color.head<3>(), Array3f::Constant(exposure), tonemap_curve, color_space, output_color_space);
	if (clamp_output_color) {
		color = color.cwiseMax(0.0f).cwiseMin(1.0f);
	}

	surf2Dwrite(to_float4(color), surface, x * sizeof(float4), y);
}

__global__ void dlss_splat_kernel(
	Vector2i resolution,
	cudaSurfaceObject_t dlss_surface,
	cudaSurfaceObject_t surface
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	float4 color;
	surf2Dread(&color, dlss_surface, x * sizeof(float4), y);
	surf2Dwrite(color, surface, x * sizeof(float4), y);
}

void CudaRenderBuffer::resize(const Vector2i& res) {
	m_in_resolution = res;
	m_frame_buffer.enlarge(res.x() * res.y());
	m_depth_buffer.enlarge(res.x() * res.y());
	m_accumulate_buffer.enlarge(res.x() * res.y());

	Vector2i out_res = m_dlss ? m_dlss->out_resolution() : res;
	auto prev_out_res = out_resolution();
	m_surface_provider->resize(out_res);

	if (out_resolution() != prev_out_res) {
		reset_accumulation();
	}
}

void CudaRenderBuffer::clear_frame(cudaStream_t stream) {
	CUDA_CHECK_THROW(cudaMemsetAsync(m_frame_buffer.data(), 0, m_frame_buffer.bytes(), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_depth_buffer.data(), 0, m_depth_buffer.bytes(), stream));
}

void CudaRenderBuffer::accumulate(float exposure, cudaStream_t stream) {
	Vector2i res = in_resolution();

	uint32_t accum_spp = m_dlss ? 0 : m_spp;

	if (accum_spp == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_accumulate_buffer.data(), 0, m_accumulate_buffer.bytes(), stream));
	}

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };
	accumulate_kernel<<<blocks, threads, 0, stream>>>(
		res,
		frame_buffer(),
		accumulate_buffer(),
		(float)accum_spp,
		m_color_space
	);

	++m_spp;
}

void CudaRenderBuffer::tonemap(float exposure, const Array4f& background_color, EColorSpace output_color_space, cudaStream_t stream) {
	assert(m_dlss || out_resolution() == in_resolution());

	auto res = m_dlss ? in_resolution() : out_resolution();
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };
	tonemap_kernel<<<blocks, threads, 0, stream>>>(
		res,
		exposure,
		background_color,
		accumulate_buffer(),
		m_color_space,
		output_color_space,
		m_tonemap_curve,
		m_dlss && output_color_space == EColorSpace::SRGB,
		m_dlss ? m_dlss->frame() : surface()
	);

	if (m_dlss) {
		assert(out_resolution() == m_dlss->out_resolution());

		assert(m_spp >= 1);
		uint32_t sample_index = m_spp - 1;

		m_dlss->run(
			res,
			output_color_space == EColorSpace::Linear, /* HDR mode */
			m_dlss_sharpening,
			Vector2f::Constant(0.5f) - ld_random_pixel_offset(sample_index), /* jitter offset in [-0.5, 0.5] */
			sample_index == 0 /* reset history */
		);

		auto out_res = out_resolution();
		const dim3 out_blocks = { div_round_up((uint32_t)out_res.x(), threads.x), div_round_up((uint32_t)out_res.y(), threads.y), 1 };
		dlss_splat_kernel<<<out_blocks, threads, 0, stream>>>(out_res, m_dlss->output(), surface());
	}
}

void CudaRenderBuffer::overlay_image(
	float alpha,
	const Eigen::Array3f& exposure,
	const Array4f& background_color,
	EColorSpace output_color_space,
	const void* __restrict__ image,
	EImageDataType image_data_type,
	const Vector2i& image_resolution,
	int fov_axis,
	float zoom,
	const Eigen::Vector2f& screen_center,
	cudaStream_t stream
) {
	auto res = out_resolution();
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };
	overlay_image_kernel<<<blocks, threads, 0, stream>>>(
		res,
		alpha,
		exposure,
		background_color,
		image,
		image_data_type,
		image_resolution,
		m_tonemap_curve,
		m_color_space,
		output_color_space,
		fov_axis,
		zoom,
		screen_center,
		surface()
	);
}

void CudaRenderBuffer::overlay_depth(
	float alpha,
	const float* __restrict__ depth,
	float depth_scale,
	const Vector2i& image_resolution,
	int fov_axis,
	float zoom,
	const Eigen::Vector2f& screen_center,
	cudaStream_t stream
) {
	auto res = out_resolution();
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };
	overlay_depth_kernel<<<blocks, threads, 0, stream>>>(
		res,
		alpha,
		depth,
		depth_scale,
		image_resolution,
		fov_axis,
		zoom,
		screen_center,
		surface()
	);
}

void CudaRenderBuffer::overlay_false_color(Vector2i training_resolution, bool to_srgb, int fov_axis, cudaStream_t stream, const float* error_map, Vector2i error_map_resolution, const float* average, float brightness, bool viridis) {
	auto res = out_resolution();
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };
	overlay_false_color_kernel<<<blocks, threads, 0, stream>>>(
		res,
		training_resolution,
		to_srgb,
		fov_axis,
		surface(),
		error_map,
		error_map_resolution,
		average,
		brightness,
		viridis
	);
}

void CudaRenderBuffer::enable_dlss(const Eigen::Vector2i& max_out_res) {
#ifdef NGP_VULKAN
	if (!m_dlss || m_dlss->max_out_resolution() != max_out_res) {
		m_dlss = dlss_init(max_out_res);
	}

	if (m_dlss) {
		resize(m_dlss->clamp_resolution(in_resolution()));
	}
#else
	throw std::runtime_error{"NGP was compiled without Vulkan/NGX/DLSS support."};
#endif
}

void CudaRenderBuffer::disable_dlss() {
	m_dlss = nullptr;
}

NGP_NAMESPACE_END
