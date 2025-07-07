/*
* Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/** @file   common_host.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Shared functionality among multiple neural-graphics-primitives components.
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <filesystem/path.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <tinylogger/tinylogger.h>

#include <chrono>
#include <functional>

namespace ngp {

namespace fs = filesystem;

bool is_wsl();

fs::path discover_executable_dir();
fs::path discover_root_dir();

#ifdef _WIN32
std::string utf16_to_utf8(const std::wstring& utf16);
std::wstring utf8_to_utf16(const std::string& utf16);
std::wstring native_string(const fs::path& path);
#else
std::string native_string(const fs::path& path);
#endif

bool ends_with(const std::string& str, const std::string& ending);
bool ends_with_case_insensitive(const std::string& str, const std::string& ending);

ETestbedMode mode_from_scene(const std::string& scene);
ETestbedMode mode_from_string(const std::string& str);
std::string to_string(ETestbedMode);

inline std::string replace_all(std::string str, const std::string& a, const std::string& b) {
	std::string::size_type n = 0;
	while ((n = str.find(a, n)) != std::string::npos) {
		str.replace(n, a.length(), b);
		n += b.length();
	}
	return str;
}

template <typename T>
T snap_to_nearest(T val, const std::vector<T>& candidates) {
	T best_dist = std::numeric_limits<T>::max();
	T result = candidates.empty() ? val : candidates[0];
	for (T c : candidates) {
		T dist = abs(val - c);
		if (dist < best_dist) {
			best_dist = dist;
			result = c;
		}
	}

	return result;
}

enum class EEmaType {
	Time,
	Step,
};

template <typename T>
class Ema {
public:
	Ema(EEmaType type, float half_life)
	: m_type{type}, m_decay{std::pow(0.5f, 1.0f / max(half_life, 0.000001f))}, m_creation_time{std::chrono::steady_clock::now()} {}

	int64_t current_progress() {
		if (m_type == EEmaType::Time) {
			auto now = std::chrono::steady_clock::now();
			return std::chrono::duration_cast<std::chrono::milliseconds>(now - m_creation_time).count();
		} else {
			return m_last_progress + 1;
		}
	}

	void update(const T& val) {
		int64_t cur = current_progress();
		int64_t elapsed = cur - m_last_progress;
		m_last_progress = cur;

		float decay = std::pow(m_decay, elapsed);
		m_val = val;
		m_ema_val = decay * m_ema_val + (1.0f - decay) * val;
	}

	void set(const T& val) {
		m_last_progress = current_progress();
		m_val = m_ema_val = val;
	}

	T val() const {
		return m_val;
	}

	T ema_val() const {
		return m_ema_val;
	}

private:
	T m_val = 0.0f;
	T m_ema_val = 0.0f;
	EEmaType m_type;
	float m_decay;

	int64_t m_last_progress = 0;
	std::chrono::time_point<std::chrono::steady_clock> m_creation_time;
};

class CameraPredictor {
public:
	virtual void sample_camera(const mat4x3& cam, float delta_t) = 0;
	virtual std::vector<mat4x3> predict_camera_sequence(float, size_t n) const = 0;
	virtual mat4x3 predict_camera(float delta_t) const = 0;
};

class StationaryCameraPredictor : public CameraPredictor {
public:
	void sample_camera(const mat4x3& cam, float) override {
		m_prev_cam = cam;
	}

	std::vector<mat4x3> predict_camera_sequence(float, size_t n) const override {
		return std::vector<mat4x3>(n, m_prev_cam);
	}

	mat4x3 predict_camera(float) const override {
		return m_prev_cam;
	}

private:
	mat4x3 m_prev_cam = mat4x3::identity();
};

class MatLogCameraPredictor : public CameraPredictor {
public:
	// If `is_quadratic` is false, will predict linearly in log-space instead
	// of quadratically.
	MatLogCameraPredictor(bool is_quadratic, bool is_se3, float half_life) :
		m_is_quadratic{is_quadratic},
		m_is_se3{is_se3},
		m_ema_vel{half_life == 0 ? EEmaType::Step : EEmaType::Time, half_life * 1000.0f},
		m_ema_acc{half_life == 0 ? EEmaType::Step : EEmaType::Time, half_life * 1000.0f}
	{
	}

	mat4 log(const mat4& mat) const {
		if (m_is_se3) {
			return se3_log(mat);
		} else {
			return mat_log(mat);
		}
	}

	mat4 exp(const mat4& mat) const {
		if (m_is_se3) {
			return se3_exp(mat);
		} else {
			return mat_exp(mat);
		}
	}

	void sample_camera(const mat4x3& cam, float delta_t) override {
		auto log_diff = log(mat4(cam) * inverse(mat4(m_prev_cam)));
		delta_t = max(0.000001f, delta_t);

		m_ema_vel.update(log_diff / delta_t);
		m_ema_acc.update((m_ema_vel.val() - m_prev_log_vel) / delta_t);

		m_prev_cam = cam;
		m_prev_log_vel = m_ema_vel.val();
	}

	mat4 predict_camera_quadratic(float delta_t, const mat4& pos, const mat4& vel, const mat4& acc) const {
		auto mean_v = vel + (0.5f * delta_t) * acc;
		return exp(mean_v * delta_t) * pos;
	}

	mat4 predict_camera_linear(float delta_t, const mat4& pos, const mat4& vel) const {
		return exp(vel * delta_t) * pos;
	}

	mat4x3 predict_camera(float delta_t, const mat4& pos, const mat4& vel, const mat4& acc) const {
		if (m_is_quadratic) {
			return predict_camera_quadratic(delta_t, pos, vel, acc);
		} else {
			return predict_camera_linear(delta_t, pos, vel);
		}
	}

	std::vector<mat4x3> predict_camera_sequence(float time, size_t sequence_size) const override {
		std::vector<mat4x3> result(sequence_size);
		result[0] = m_prev_cam;

		mat4 vel = m_ema_vel.ema_val();
		const mat4& acc = m_ema_acc.ema_val();

		float dt = time / (sequence_size - 1);
		for (size_t i = 1; i < sequence_size; ++i) {
			if (m_is_quadratic) {
				result[i] = predict_camera_quadratic(dt, result[i-1], vel, acc);
				vel += acc * dt;
			} else {
				result[i] = predict_camera_linear(i * dt, m_prev_cam, vel);
			}
		}

		return result;
	}

	mat4x3 predict_camera(float delta_t) const override {
		if (m_is_quadratic) {
			// There's no simple closed form quadratic integral over
			// SE(3), so let's perform Euler integration. The alternative
			// would be closed-form integration in matrix-log space
			// (or some other Lie algebra), but that's slightly erroneous.
			return predict_camera_sequence(delta_t, 64).back();
		} else {
			return mat4x3(exp(m_ema_vel.ema_val() * delta_t) * mat4(m_prev_cam));
		}
	}

private:
	mat4x3 m_prev_cam = mat4x3::identity();
	mat4 m_prev_log_vel = mat4::zero();

	Ema<mat4> m_ema_vel;
	Ema<mat4> m_ema_acc;

	bool m_is_quadratic = true;
	bool m_is_se3 = false;
};

uint8_t* load_stbi(const fs::path& path, int* width, int* height, int* comp, int req_comp);
float* load_stbi_float(const fs::path& path, int* width, int* height, int* comp, int req_comp);
uint16_t* load_stbi_16(const fs::path& path, int* width, int* height, int* comp, int req_comp);
bool is_hdr_stbi(const fs::path& path);
int write_stbi(const fs::path& path, int width, int height, int comp, const uint8_t* pixels, int quality = 100);

FILE* native_fopen(const fs::path& path, const char* mode);

GPUMemory<float> load_exr_gpu(const fs::path& path, int* width, int* height);
GPUMemory<float> load_stbi_gpu(const fs::path& path, int* width, int* height);

template <typename T>
class Buffer2D {
public:
	Buffer2D() = default;
	Buffer2D(const ivec2& resolution) {
		resize(resolution);
	}

	T* data() const {
		return m_data.data();
	}

	size_t bytes() const {
		return m_data.bytes();
	}

	void resize(const ivec2& resolution) {
		m_data.resize(product(resolution));
		m_resolution = resolution;
	}

	const ivec2& resolution() const {
		return m_resolution;
	}

	Buffer2DView<T> view() const {
		// Row major for now.
		return {data(), m_resolution};
	}

	Buffer2DView<const T> const_view() const {
		// Row major for now.
		return {data(), m_resolution};
	}

private:
	GPUMemory<T> m_data;
	ivec2 m_resolution;
};

template <typename T> struct GPUImage {
	GPUImage() : image{}, padding{0} {}

	GPUImage(ivec2 resolution, uint32_t padding, cudaStream_t stream) :
		image{resolution.y + padding * 2, resolution.x + padding * 2, stream}, padding{padding} {}

	GPUImage(ivec2 resolution, cudaStream_t stream) : GPUImage(resolution, 0, stream) {}

	MatrixView<T> view() const { return image.slice(padding, image.m() - 2 * padding, padding, image.n() - 2 * padding).view(); }

	T* data() const { return image.data(); }

	size_t n_elements_padded() const { return image.n_elements(); }
	size_t n_elements() const { return product(resolution()); }

	ivec2 resolution_padded() const { return {(int)image.n(), (int)image.m()}; }
	ivec2 resolution() const { return {(int)(image.n() - 2 * padding), (int)(image.m() - 2 * padding)}; }

	explicit operator bool() const { return image.data() != nullptr; }

	GPUMatrix<T, RM> image;
	uint32_t padding;
};

struct BoundingBox;
struct Triangle;
std::ostream& operator<<(std::ostream& os, const BoundingBox& triangle);
std::ostream& operator<<(std::ostream& os, const Triangle& triangle);

}
