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

using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

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
