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

#ifdef _WIN32
#  define NOMINMAX
#endif

#include <tiny-cuda-nn/common.h>
using namespace tcnn;


#if defined(__CUDA_ARCH__)
	#define NGP_PRAGMA_UNROLL _Pragma("unroll")
	#define NGP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
	#define NGP_PRAGMA_UNROLL
	#define NGP_PRAGMA_NO_UNROLL
#endif

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
#define NGP_HOST_DEVICE __host__ __device__
#else
#define NGP_HOST_DEVICE
#endif

namespace ngp {

enum class EMeshRenderMode : int {
	Off,
	VertexColors,
	VertexNormals,
	FaceIDs,
};

enum class EGroundTruthRenderMode : int {
	Shade,
	Depth,
	NumRenderModes,
};
static constexpr const char* GroundTruthRenderModeStr = "Shade\0Depth\0\0";

enum class ERenderMode : int {
	AO,
	Shade,
	Normals,
	Positions,
	Depth,
	Distortion,
	Cost,
	Slice,
	NumRenderModes,
	EncodingVis, // EncodingVis exists outside of the standard render modes
};
static constexpr const char* RenderModeStr = "AO\0Shade\0Normals\0Positions\0Depth\0Distortion\0Cost\0Slice\0\0";

enum class ERandomMode : int {
	Random,
	Halton,
	Sobol,
	Stratified,
	NumImageRandomModes,
};
static constexpr const char* RandomModeStr = "Random\0Halton\0Sobol\0Stratified\0\0";

enum class ELossType : int {
	L2,
	L1,
	Mape,
	Smape,
	Huber,
	LogL1,
	RelativeL2,
};
static constexpr const char* LossTypeStr = "L2\0L1\0MAPE\0SMAPE\0Huber\0LogL1\0RelativeL2\0\0";

enum class ENerfActivation : int {
	None,
	ReLU,
	Logistic,
	Exponential,
};
static constexpr const char* NerfActivationStr = "None\0ReLU\0Logistic\0Exponential\0\0";

enum class EMeshSdfMode : int {
	Watertight,
	Raystab,
	PathEscape,
};
static constexpr const char* MeshSdfModeStr = "Watertight\0Raystab\0PathEscape\0\0";

enum class EColorSpace : int {
	Linear,
	SRGB,
	VisPosNeg,
};
static constexpr const char* ColorSpaceStr = "Linear\0SRGB\0\0";

enum class ETonemapCurve : int {
	Identity,
	ACES,
	Hable,
	Reinhard
};
static constexpr const char* TonemapCurveStr = "Identity\0ACES\0Hable\0Reinhard\0\0";

enum class EDlssQuality : int {
	UltraPerformance,
	MaxPerformance,
	Balanced,
	MaxQuality,
	UltraQuality,
	NumDlssQualitySettings,
	None,
};
static constexpr const char* DlssQualityStr = "UltraPerformance\0MaxPerformance\0Balanced\0MaxQuality\0UltraQuality\0Invalid\0None\0\0";
static constexpr const char* DlssQualityStrArray[] = {"UltraPerformance", "MaxPerformance", "Balanced", "MaxQuality", "UltraQuality", "Invalid", "None"};

enum class ETestbedMode : int {
	Nerf,
	Sdf,
	Image,
	Volume,
	None,
};

enum class EMlpAlgorithm : int {
	MMA,
	FMA,
};

enum class ESDFGroundTruthMode : int {
	RaytracedMesh,
	SpheretracedMesh,
	SDFBricks,
};

struct Ray {
	vec3 o;
	vec3 d;

	NGP_HOST_DEVICE vec3 operator()(float t) const {
		return o + t * d;
	}

	NGP_HOST_DEVICE void advance(float t) {
		o += d * t;
	}

	NGP_HOST_DEVICE float distance_to(const vec3& p) const {
		vec3 nearest = p - o;
		nearest -= d * dot(nearest, d) / length2(d);
		return length(nearest);
	}

	NGP_HOST_DEVICE bool is_valid() const {
		return d != vec3(0.0f);
	}

	static NGP_HOST_DEVICE Ray invalid() {
		return {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
	}
};

struct TrainingXForm {
	NGP_HOST_DEVICE bool operator==(const TrainingXForm& other) const {
		return start == other.start && end == other.end;
	}

	mat4x3 start;
	mat4x3 end;
};

enum class ELensMode : int {
	Perspective,
	OpenCV,
	FTheta,
	LatLong,
	OpenCVFisheye,
	Equirectangular,
};
static constexpr const char* LensModeStr = "Perspective\0OpenCV\0F-Theta\0LatLong\0OpenCV Fisheye\0Equirectangular\0\0";

inline NGP_HOST_DEVICE bool supports_dlss(ELensMode mode) {
	return mode == ELensMode::Perspective || mode == ELensMode::OpenCV || mode == ELensMode::OpenCVFisheye;
}

struct Lens {
	ELensMode mode = ELensMode::Perspective;
	float params[7] = {};
};

inline NGP_HOST_DEVICE uint32_t binary_search(float val, const float* data, uint32_t length) {
	if (length == 0) {
		return 0;
	}

	uint32_t it;
	uint32_t count, step;
	count = length;

	uint32_t first = 0;
	while (count > 0) {
		it = first;
		step = count / 2;
		it += step;
		if (data[it] < val) {
			first = ++it;
			count -= step + 1;
		} else {
			count = step;
		}
	}

	return min(first, length-1);
}

template <typename T>
struct Buffer2DView {
	T* data = nullptr;
	ivec2 resolution = 0;

	// Lookup via integer pixel position (no bounds checking)
	NGP_HOST_DEVICE T at(const ivec2& px) const {
		return data[px.x + px.y * resolution.x];
	}

	// Lookup via UV coordinates in [0,1]^2
	NGP_HOST_DEVICE T at(const vec2& uv) const {
		ivec2 px = clamp(ivec2(vec2(resolution) * uv), 0, resolution - 1);
		return at(px);
	}

	// Lookup via UV coordinates in [0,1]^2 and LERP the nearest texels
	NGP_HOST_DEVICE T at_lerp(const vec2& uv) const {
		const vec2 px_float = vec2(resolution) * uv;
		const ivec2 px = ivec2(px_float);

		const vec2 weight = px_float - vec2(px);

		auto read_val = [&](ivec2 pos) {
			return at(clamp(pos, 0, resolution - 1));
		};

		return (
			(1 - weight.x) * (1 - weight.y) * read_val({px.x, px.y}) +
			(weight.x) * (1 - weight.y) * read_val({px.x+1, px.y}) +
			(1 - weight.x) * (weight.y) * read_val({px.x, px.y+1}) +
			(weight.x) * (weight.y) * read_val({px.x+1, px.y+1})
		);
	}

	NGP_HOST_DEVICE operator bool() const {
		return data;
	}
};

}
