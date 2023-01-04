/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   common_device.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/common.h>

#include <filesystem/path.h>

#ifndef _WIN32
#  include <unistd.h>
#  include <linux/limits.h>
#endif

using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

fs::path get_executable_dir() {
#ifdef _WIN32
	WCHAR path[MAX_PATH];
	if (GetModuleFileNameW(NULL, path, MAX_PATH) == 0) {
		return ".";
	}
#else
	char path[PATH_MAX];
	ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
	if (count == -1) {
		return ".";
	}
#endif
	return fs::path{path}.parent_path();
}

filesystem::path get_root_dir() {
	auto executable_dir = get_executable_dir();
	fs::path exists_in_root_dir = "./scripts";
	for (const auto& candidate : {
		exists_in_root_dir,
		fs::path{"../"}/exists_in_root_dir,
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

NGP_NAMESPACE_END
