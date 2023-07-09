/*
* Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

struct BoundingBox;
struct Triangle;
std::ostream& operator<<(std::ostream& os, const BoundingBox& triangle);
std::ostream& operator<<(std::ostream& os, const Triangle& triangle);

}
