/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   common_host.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common_host.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>
#include <neural-graphics-primitives/triangle.cuh>

#include <tiny-cuda-nn/common.h>

#include <filesystem/path.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#ifdef __CUDACC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress 550
#  else
#    pragma diag_suppress 550
#  endif
#endif
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>
#ifdef __CUDACC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_default 550
#  else
#    pragma diag_default 550
#  endif
#endif

#ifdef _WIN32
#  include <Windows.h>
#else
#  include <unistd.h>
#  include <linux/limits.h>
#endif

#undef min
#undef max
#undef near
#undef far

namespace ngp {

bool is_wsl() {
#ifdef _WIN32
	return false;
#else
	fs::path path = "/proc/sys/kernel/osrelease";
	if (!path.exists()) {
		return false;
	}

	std::ifstream f{native_string(path)};
	std::string content((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	return content.find("microsoft") != std::string::npos;
#endif
}

#ifdef _WIN32
std::string utf16_to_utf8(const std::wstring& utf16) {
	std::string utf8;
	if (!utf16.empty()) {
		int size = WideCharToMultiByte(CP_UTF8, 0, &utf16[0], (int)utf16.size(), NULL, 0, NULL, NULL);
		utf8.resize(size, 0);
		WideCharToMultiByte(CP_UTF8, 0, &utf16[0], (int)utf16.size(), &utf8[0], size, NULL, NULL);
	}
	return utf8;
}

std::wstring utf8_to_utf16(const std::string& utf8) {
	std::wstring utf16;
	if (!utf8.empty()) {
		int size = MultiByteToWideChar(CP_UTF8, 0, &utf8[0], (int)utf8.size(), NULL, 0);
		utf16.resize(size, 0);
		MultiByteToWideChar(CP_UTF8, 0, &utf8[0], (int)utf8.size(), &utf16[0], size);
	}
	return utf16;
}

std::wstring native_string(const fs::path& path) { return path.wstr(); }
#else
std::string native_string(const fs::path& path) { return path.str(); }
#endif

fs::path discover_executable_dir() {
#ifdef _WIN32
	WCHAR path[1024];
	if (GetModuleFileNameW(NULL, path, 1024) == 0) {
		return ".";
	}
	return fs::path{std::wstring{path}}.parent_path();
#else
	char path[PATH_MAX];
	ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
	if (count == -1) {
		return ".";
	}
	return fs::path{std::string{path}}.parent_path();
#endif
}

fs::path discover_root_dir() {
	auto executable_dir = discover_executable_dir();
	fs::path exists_in_root_dir = "scripts";
	for (const auto& candidate : {
		fs::path{"."}/exists_in_root_dir,
		fs::path{".."}/exists_in_root_dir,
		executable_dir/exists_in_root_dir,
		executable_dir/".."/exists_in_root_dir,
	}) {
		if (candidate.exists()) {
			return candidate.parent_path();
		}
	}

	tlog::warning() << "Could not find root directory.";
	return ".";
}

bool ends_with(const std::string& str, const std::string& ending) {
	if (ending.length() > str.length()) {
		return false;
	}
	return std::equal(std::rbegin(ending), std::rend(ending), std::rbegin(str));
}

bool ends_with_case_insensitive(const std::string& str, const std::string& ending) {
	return ends_with(to_lower(str), to_lower(ending));
}

ETestbedMode mode_from_scene(const std::string& scene) {
	fs::path scene_path = scene;
	if (!scene_path.exists()) {
		return ETestbedMode::None;
	}

	if (scene_path.is_directory() || equals_case_insensitive(scene_path.extension(), "json")) {
		return ETestbedMode::Nerf;
	} else if (equals_case_insensitive(scene_path.extension(), "obj") || equals_case_insensitive(scene_path.extension(), "stl")) {
		return ETestbedMode::Sdf;
	} else if (equals_case_insensitive(scene_path.extension(), "nvdb")) {
		return ETestbedMode::Volume;
	} else { // probably an image. Too bothersome to list all supported ones: exr, bin, jpg, png, tga, hdr, ...
		return ETestbedMode::Image;
	}
}

ETestbedMode mode_from_string(const std::string& str) {
	if (equals_case_insensitive(str, "nerf")) {
		return ETestbedMode::Nerf;
	} else if (equals_case_insensitive(str, "sdf")) {
		return ETestbedMode::Sdf;
	} else if (equals_case_insensitive(str, "image")) {
		return ETestbedMode::Image;
	} else if (equals_case_insensitive(str, "volume")) {
		return ETestbedMode::Volume;
	} else {
		return ETestbedMode::None;
	}
}

std::string to_string(ETestbedMode mode) {
	switch (mode) {
		case ETestbedMode::Nerf: return "nerf";
		case ETestbedMode::Sdf: return "sdf";
		case ETestbedMode::Image: return "image";
		case ETestbedMode::Volume: return "volume";
		case ETestbedMode::None: return "none";
		default: throw std::runtime_error{fmt::format("Can not convert mode {} to string.", (int)mode)};
	}
}

static const stbi_io_callbacks istream_stbi_callbacks = {
	// Read
	[](void* context, char* data, int size) {
		auto stream = reinterpret_cast<std::istream*>(context);
		stream->read(data, size);
		return (int)stream->gcount();
	},
	// Seek
	[](void* context, int size) {
		reinterpret_cast<std::istream*>(context)->seekg(size, std::ios_base::cur);
	},
	// EOF
	[](void* context) {
		return (int)!!(*reinterpret_cast<std::istream*>(context));
	},
};

void istream_stbi_write_func(void* context, void* data, int size) {
	reinterpret_cast<std::ostream*>(context)->write(reinterpret_cast<char*>(data), size);
}

uint8_t* load_stbi(const fs::path& path, int* width, int* height, int* comp, int req_comp) {
	std::ifstream f{native_string(path), std::ios::in | std::ios::binary};
	return stbi_load_from_callbacks(&istream_stbi_callbacks, &f, width, height, comp, req_comp);
}

float* load_stbi_float(const fs::path& path, int* width, int* height, int* comp, int req_comp) {
	std::ifstream f{native_string(path), std::ios::in | std::ios::binary};
	return stbi_loadf_from_callbacks(&istream_stbi_callbacks, &f, width, height, comp, req_comp);
}

uint16_t* load_stbi_16(const fs::path& path, int* width, int* height, int* comp, int req_comp) {
	std::ifstream f{native_string(path), std::ios::in | std::ios::binary};
	return stbi_load_16_from_callbacks(&istream_stbi_callbacks, &f, width, height, comp, req_comp);
}

bool is_hdr_stbi(const fs::path& path) {
	std::ifstream f{native_string(path), std::ios::in | std::ios::binary};
	return stbi_is_hdr_from_callbacks(&istream_stbi_callbacks, &f);
}

int write_stbi(const fs::path& path, int width, int height, int comp, const uint8_t* pixels, int quality) {
	std::ofstream f{native_string(path), std::ios::out | std::ios::binary};

	if (equals_case_insensitive(path.extension(), "jpg") || equals_case_insensitive(path.extension(), "jpeg")) {
		return stbi_write_jpg_to_func(istream_stbi_write_func, &f, width, height, comp, pixels, quality);
	} else if (equals_case_insensitive(path.extension(), "png")) {
		return stbi_write_png_to_func(istream_stbi_write_func, &f, width, height, comp, pixels, width * comp);
	} else if (equals_case_insensitive(path.extension(), "tga")) {
		return stbi_write_tga_to_func(istream_stbi_write_func, &f, width, height, comp, pixels);
	} else if (equals_case_insensitive(path.extension(), "bmp")) {
		return stbi_write_bmp_to_func(istream_stbi_write_func, &f, width, height, comp, pixels);
	} else {
		throw std::runtime_error{fmt::format("write_stbi: unknown image extension '{}'", path.extension())};
	}
}

FILE* native_fopen(const fs::path& path, const char* mode) {
#ifdef _WIN32
	return _wfopen(path.wstr().c_str(), utf8_to_utf16(mode).c_str());
#else
	return fopen(path.str().c_str(), mode);
#endif
}

GPUMemory<float> load_exr_gpu(const fs::path& path, int* width, int* height) {
	float* out; // width * height * RGBA
	load_exr(&out, width, height, path.str().c_str());
	ScopeGuard mem_guard{[&]() { free(out); }};

	GPUMemory<float> result((*width) * (*height) * 4);
	result.copy_from_host(out);
	return result;
}

GPUMemory<float> load_stbi_gpu(const fs::path& path, int* width, int* height) {
	bool is_hdr = is_hdr_stbi(path);

	void* data; // width * height * RGBA
	int comp;
	if (is_hdr) {
		data = load_stbi_float(path, width, height, &comp, 4);
	} else {
		data = load_stbi(path, width, height, &comp, 4);
	}

	if (!data) {
		throw std::runtime_error{std::string{stbi_failure_reason()}};
	}

	ScopeGuard mem_guard{[&]() { stbi_image_free(data); }};

	if (*width == 0 || *height == 0) {
		throw std::runtime_error{"Image has zero pixels."};
	}

	GPUMemory<float> result((*width) * (*height) * 4);
	if (is_hdr) {
		result.copy_from_host((float*)data);
	} else {
		GPUMemory<uint8_t> bytes((*width) * (*height) * 4);
		bytes.copy_from_host((uint8_t*)data);
		linear_kernel(from_rgba32<float>, 0, nullptr, (*width) * (*height), bytes.data(), result.data(), false, false, 0);
	}

	return result;
}

std::ostream& operator<<(std::ostream& os, const BoundingBox& bb) {
	os << "[";
	os << "min=[" << bb.min.x << "," << bb.min.y << "," << bb.min.z << "], ";
	os << "max=[" << bb.max.x << "," << bb.max.y << "," << bb.max.z << "]";
	os << "]";
	return os;
}

std::ostream& operator<<(std::ostream& os, const Triangle& triangle) {
	os << "[";
	os << "a=[" << triangle.a.x << "," << triangle.a.y << "," << triangle.a.z << "], ";
	os << "b=[" << triangle.b.x << "," << triangle.b.y << "," << triangle.b.z << "], ";
	os << "c=[" << triangle.c.x << "," << triangle.c.y << "," << triangle.c.z << "]";
	os << "]";
	return os;
}

}
