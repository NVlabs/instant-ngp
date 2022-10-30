/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   tinyexr_wrapper.cpp
 *  @author Thomas Müller, NVIDIA
 *  @brief  Wrapper around the tinyexr library, providing a simple interface
 *          to load and store EXR images.
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/tinyexr_wrapper.h>

#include <tiny-cuda-nn/gpu_memory.h>

#ifdef __NVCC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress 174
#    pragma nv_diag_suppress 550
#  else
#    pragma diag_suppress 174
#    pragma diag_suppress 550
#  endif
#endif

#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

using namespace tcnn;

NGP_NAMESPACE_BEGIN

template <typename T>
__global__ void interleave_and_cast_kernel(const uint32_t num_pixels, bool has_alpha, const T* __restrict__ in, __half* __restrict__ out, bool fix_pre_mult) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	__half rgba_out[4];

	float alpha = has_alpha ? (float)in[3*num_pixels + i] : (float)1.0f;
	float fix = fix_pre_mult ? alpha : 1.0f;
	rgba_out[0] = (__half)(float(in[0*num_pixels + i]) * fix);
	rgba_out[1] = (__half)(float(in[1*num_pixels + i]) * fix);
	rgba_out[2] = (__half)(float(in[2*num_pixels + i]) * fix);
	rgba_out[3] = (__half)alpha;

	*((uint64_t*)&out[i*4]) = *((uint64_t*)&rgba_out[0]);
}

void save_exr(const float* data, int width, int height, int nChannels, int channelStride, const char* outfilename) {
	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);

	image.num_channels = nChannels;

	std::vector<std::vector<float>> images(nChannels);
	std::vector<float*> image_ptr(nChannels);
	for (int i = 0; i < nChannels; ++i) {
		images[i].resize(width * height);
	}

	for (int i = 0; i < nChannels; ++i) {
		image_ptr[i] = images[nChannels - i - 1].data();
	}

	for (size_t i = 0; i < (size_t)width * height; i++) {
		for (int c = 0; c < nChannels; ++c) {
			images[c][i] = data[channelStride*i+c];
		}
	}

	image.images = (unsigned char**)image_ptr.data();
	image.width = width;
	image.height = height;

	header.num_channels = nChannels;
	header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be (A)BGR order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
	if (nChannels > 1) {
		strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
	}
	if (nChannels > 2) {
		strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';
	}
	if (nChannels > 3) {
		strncpy(header.channels[3].name, "A", 255); header.channels[3].name[strlen("A")] = '\0';
	}

	header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	for (int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
	}

	const char* err = NULL; // or nullptr in C++11 or later.
	int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
	if (ret != TINYEXR_SUCCESS) {
		std::string error_message = std::string("Failed to save EXR image: ") + err;
		FreeEXRErrorMessage(err); // free's buffer for an error message
		throw std::runtime_error(error_message);
	}
	tlog::info() << "Saved exr file: " << outfilename;

	free(header.channels);
	free(header.pixel_types);
	free(header.requested_pixel_types);
}

void load_exr(float** data, int* width, int* height, const char* filename) {
	const char* err = nullptr;

	int ret = LoadEXR(data, width, height, filename, &err);

	if (ret != TINYEXR_SUCCESS) {
		if (err) {
			std::string error_message = std::string("Failed to load EXR image: ") + err;
			FreeEXRErrorMessage(err);
			throw std::runtime_error(error_message);
		} else {
			throw std::runtime_error("Failed to load EXR image");
		}
	}
}

__half* load_exr_to_gpu(int* width, int* height, const char* filename, bool fix_premult) {
	// 1. Read EXR version.
	EXRVersion exr_version;

	int ret = ParseEXRVersionFromFile(&exr_version, filename);
	if (ret != 0) {
		std::string error_message = std::string("Failed to parse EXR image version");
		throw std::runtime_error(error_message);
	}

	if (exr_version.multipart) {
		throw std::runtime_error("EXR file must be singlepart");
	}

	// 2. Read EXR header
	EXRHeader exr_header;
	InitEXRHeader(&exr_header);

	const char* err = NULL; // or `nullptr` in C++11 or later.
	ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, filename, &err);
	if (ret != 0) {
		std::string error_message = std::string("Failed to parse EXR image header: ") + err;
		FreeEXRErrorMessage(err); // free's buffer for an error message
		throw std::runtime_error(error_message);
	}

	bool full_precision = exr_header.pixel_types[0] == TINYEXR_PIXELTYPE_FLOAT;
	// Read FLOAT channel as HALF.
	for (int i = 0; i < exr_header.num_channels; i++) {
		bool local_fp = exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_FLOAT;
		if (local_fp != full_precision) {
			throw std::runtime_error("Can't handle EXR images with mixed channel types");
		}
	}

	EXRImage exr_image;
	InitEXRImage(&exr_image);

	ret = LoadEXRImageFromFile(&exr_image, &exr_header, filename, &err);
	if (ret != 0) {
		std::string error_message = std::string("Failed to load EXR image: ") + err;
		FreeEXRHeader(&exr_header);
		FreeEXRErrorMessage(err); // free's buffer for an error message
		throw std::runtime_error(error_message);
	}

	// 3. Access image data
	// `exr_image.images` will be filled when EXR is scanline format.
	// `exr_image.tiled` will be filled when EXR is tiled format.

	*width = exr_image.width;
	*height = exr_image.height;

	size_t n_pixels = exr_image.width * exr_image.height;

	size_t bytes_per_pixel = full_precision ? 4 : 2;

	GPUMemory<uint8_t> tmp{n_pixels*4*bytes_per_pixel};

	uint8_t* rawptr = nullptr;
	CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_pixels*4*bytes_per_pixel));
	__half* result = (__half*)rawptr;

	CUDA_CHECK_THROW(cudaMemset(tmp.data(), 0, bytes_per_pixel * n_pixels*4));

	bool has_alpha = false;
	for (int c = 0; c < exr_header.num_channels; c++) {
		if (strcmp(exr_header.channels[c].name, "R") == 0) {
			CUDA_CHECK_THROW(cudaMemcpy(tmp.data() + n_pixels*0*bytes_per_pixel, exr_image.images[c], bytes_per_pixel * n_pixels, cudaMemcpyHostToDevice));
		} else if (strcmp(exr_header.channels[c].name, "G") == 0) {
			CUDA_CHECK_THROW(cudaMemcpy(tmp.data() + n_pixels*1*bytes_per_pixel, exr_image.images[c], bytes_per_pixel * n_pixels, cudaMemcpyHostToDevice));
		} else if (strcmp(exr_header.channels[c].name, "B") == 0) {
			CUDA_CHECK_THROW(cudaMemcpy(tmp.data() + n_pixels*2*bytes_per_pixel, exr_image.images[c], bytes_per_pixel * n_pixels, cudaMemcpyHostToDevice));
		} else if (strcmp(exr_header.channels[c].name, "A") == 0) {
			has_alpha = true;
			CUDA_CHECK_THROW(cudaMemcpy(tmp.data() + n_pixels*3*bytes_per_pixel, exr_image.images[c], bytes_per_pixel * n_pixels, cudaMemcpyHostToDevice));
		}
	}

	if (full_precision) {
		linear_kernel(interleave_and_cast_kernel<float>, 0, nullptr, n_pixels, has_alpha, (float*)tmp.data(), result, fix_premult);
	} else {
		linear_kernel(interleave_and_cast_kernel<__half>, 0, nullptr, n_pixels, has_alpha, (__half*)tmp.data(), result, fix_premult);
	}

	// 4. Free image data
	FreeEXRImage(&exr_image);
	FreeEXRHeader(&exr_header);

	return result;
}

NGP_NAMESPACE_END
