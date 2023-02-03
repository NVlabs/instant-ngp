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


#include <tinylogger/tinylogger.h>

// Eigen uses __device__ __host__ on a bunch of defaulted constructors.
// This doesn't actually cause unwanted behavior, but does cause NVCC
// to emit this diagnostic.
// nlohmann::json produces a comparison with zero in one of its templates,
// which can also safely be ignored.
#ifdef __NVCC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#    pragma nv_diag_suppress = unsigned_compare_with_zero
#  else
#    pragma diag_suppress = esa_on_defaulted_function_ignored
#    pragma diag_suppress = unsigned_compare_with_zero
#  endif
#endif
#include <Eigen/Dense>

#define NGP_NAMESPACE_BEGIN namespace ngp {
#define NGP_NAMESPACE_END }

#if defined(__CUDA_ARCH__)
	#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
		#define NGP_PRAGMA_UNROLL _Pragma("unroll")
		#define NGP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
	#else
		#define NGP_PRAGMA_UNROLL #pragma unroll
		#define NGP_PRAGMA_NO_UNROLL #pragma unroll 1
	#endif
#else
	#define NGP_PRAGMA_UNROLL
	#define NGP_PRAGMA_NO_UNROLL
#endif

#include <filesystem/path.h>

#include <chrono>
#include <functional>

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define NGP_HOST_DEVICE __host__ __device__
#else
#define NGP_HOST_DEVICE
#endif

NGP_NAMESPACE_BEGIN

namespace fs = filesystem;

bool is_wsl();

fs::path get_executable_dir();
fs::path get_root_dir();

#ifdef _WIN32
std::string utf16_to_utf8(const std::wstring& utf16);
std::wstring utf8_to_utf16(const std::string& utf16);
std::wstring native_string(const fs::path& path);
#else
std::string native_string(const fs::path& path);
#endif

bool ends_with(const std::string& str, const std::string& ending);
bool ends_with_case_insensitive(const std::string& str, const std::string& ending);

using Vector2i32 = Eigen::Matrix<uint32_t, 2, 1>;
using Vector3i16 = Eigen::Matrix<uint16_t, 3, 1>;
using Vector4i16 = Eigen::Matrix<uint16_t, 4, 1>;
using Vector4i32 = Eigen::Matrix<uint32_t, 4, 1>;

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

ETestbedMode mode_from_scene(const std::string& scene);
ETestbedMode mode_from_string(const std::string& str);
std::string to_string(ETestbedMode);

enum class ESDFGroundTruthMode : int {
	RaytracedMesh,
	SpheretracedMesh,
	SDFBricks,
};

struct Ray {
	Eigen::Vector3f o;
	Eigen::Vector3f d;

	NGP_HOST_DEVICE Eigen::Vector3f operator()(float t) const {
		return o + t * d;
	}

	NGP_HOST_DEVICE void advance(float t) {
		o += d * t;
	}

	NGP_HOST_DEVICE float distance_to(const Eigen::Vector3f& p) const {
		Eigen::Vector3f nearest = p - o;
		nearest -= d * nearest.dot(d) / d.squaredNorm();
		return nearest.norm();
	}

	NGP_HOST_DEVICE bool is_valid() const {
		return d != Eigen::Vector3f::Zero();
	}

	static NGP_HOST_DEVICE Ray invalid() {
		return {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
	}
};

struct TrainingXForm {
	Eigen::Matrix<float, 3, 4> start;
	Eigen::Matrix<float, 3, 4> end;
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

inline bool supports_dlss(ELensMode mode) {
	return mode == ELensMode::Perspective || mode == ELensMode::OpenCV || mode == ELensMode::OpenCVFisheye;
}

struct Lens {
	ELensMode mode = ELensMode::Perspective;
	float params[7] = {};
};

inline NGP_HOST_DEVICE float sign(float x) {
	return copysignf(1.0, x);
}

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

	return std::min(first, length-1);
}

inline std::string replace_all(std::string str, const std::string& a, const std::string& b) {
	std::string::size_type n = 0;
	while ((n = str.find(a, n)) != std::string::npos) {
		str.replace(n, a.length(), b);
		n += b.length();
	}
	return str;
}

template <typename T>
std::string join(const T& components, const std::string& delim) {
	std::ostringstream s;
	for (const auto& component : components) {
		if (&components[0] != &component) {
			s << delim;
		}
		s << component;
	}

	return s.str();
}

enum class EEmaType {
	Time,
	Step,
};

class Ema {
public:
	Ema(EEmaType type, float half_life)
	: m_type{type}, m_decay{std::pow(0.5f, 1.0f / half_life)}, m_creation_time{std::chrono::steady_clock::now()} {}

	int64_t current_progress() {
		if (m_type == EEmaType::Time) {
			auto now = std::chrono::steady_clock::now();
			return std::chrono::duration_cast<std::chrono::milliseconds>(now - m_creation_time).count();
		} else {
			return m_last_progress + 1;
		}
	}

	void update(float val) {
		int64_t cur = current_progress();
		int64_t elapsed = cur - m_last_progress;
		m_last_progress = cur;

		float decay = std::pow(m_decay, elapsed);
		m_val = val;
		m_ema_val = decay * m_ema_val + (1.0f - decay) * val;
	}

	void set(float val) {
		m_last_progress = current_progress();
		m_val = m_ema_val = val;
	}

	float val() const {
		return m_val;
	}

	float ema_val() const {
		return m_ema_val;
	}

private:
	float m_val = 0.0f;
	float m_ema_val = 0.0f;
	EEmaType m_type;
	float m_decay;

	int64_t m_last_progress = 0;
	std::chrono::time_point<std::chrono::steady_clock> m_creation_time;
};

template <typename T>
struct Buffer2DView {
	T* data = nullptr;
	Eigen::Vector2i resolution = Eigen::Vector2i::Zero();

	// Lookup via integer pixel position (no bounds checking)
	NGP_HOST_DEVICE T at(const Eigen::Vector2i& xy) const {
		return data[xy.x() + xy.y() * resolution.x()];
	}

	// Lookup via UV coordinates in [0,1]^2
	NGP_HOST_DEVICE T at(const Eigen::Vector2f& uv) const {
		Eigen::Vector2i xy = resolution.cast<float>().cwiseProduct(uv).cast<int>().cwiseMax(0).cwiseMin(resolution - Eigen::Vector2i::Ones());
		return at(xy);
	}

	// Lookup via UV coordinates in [0,1]^2 and LERP the nearest texels
	NGP_HOST_DEVICE T at_lerp(const Eigen::Vector2f& uv) const {
		const Eigen::Vector2f xy_float = resolution.cast<float>().cwiseProduct(uv);
		const Eigen::Vector2i xy = xy_float.cast<int>();

		const Eigen::Vector2f weight = xy_float - xy.cast<float>();

		auto read_val = [&](Eigen::Vector2i pos) {
			pos = pos.cwiseMax(0).cwiseMin(resolution - Eigen::Vector2i::Ones());
			return at(pos);
		};

		return (
			(1 - weight.x()) * (1 - weight.y()) * read_val({xy.x(), xy.y()}) +
			(weight.x()) * (1 - weight.y()) * read_val({xy.x()+1, xy.y()}) +
			(1 - weight.x()) * (weight.y()) * read_val({xy.x(), xy.y()+1}) +
			(weight.x()) * (weight.y()) * read_val({xy.x()+1, xy.y()+1})
		);
	}

	NGP_HOST_DEVICE operator bool() const {
		return data;
	}
};

uint8_t* load_stbi(const fs::path& path, int* width, int* height, int* comp, int req_comp);
float* load_stbi_float(const fs::path& path, int* width, int* height, int* comp, int req_comp);
uint16_t* load_stbi_16(const fs::path& path, int* width, int* height, int* comp, int req_comp);
bool is_hdr_stbi(const fs::path& path);
int write_stbi(const fs::path& path, int width, int height, int comp, const uint8_t* pixels, int quality = 100);

FILE* native_fopen(const fs::path& path, const char* mode);

NGP_NAMESPACE_END
