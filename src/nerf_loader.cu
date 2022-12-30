/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerfloader.cu
 *  @author Alex Evans & Thomas Müller, NVIDIA
 *  @brief  Loads a NeRF data set from NeRF's original format
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/thread_pool.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>

#include <json/json.hpp>

#include <filesystem/path.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION

#ifdef __NVCC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress 550
#  else
#    pragma diag_suppress 550
#  endif
#endif
#include <stb_image/stb_image.h>
#ifdef __NVCC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_default 550
#  else
#    pragma diag_default 550
#  endif
#endif

using namespace tcnn;
using namespace std::literals;
using namespace Eigen;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

__global__ void convert_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, uint8_t* __restrict__ out, bool white_2_transparent = false, bool black_2_transparent = false, uint32_t mask_color = 0) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	uint8_t rgba[4];
	*((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

	// NSVF dataset has 'white = transparent' madness
	if (white_2_transparent && rgba[0] == 255 && rgba[1] == 255 && rgba[2] == 255) {
		rgba[3] = 0;
	}

	if (black_2_transparent && rgba[0] == 0 && rgba[1] == 0 && rgba[2] == 0) {
		rgba[3] = 0;
	}

	if (mask_color != 0 && mask_color == *((uint32_t*)&rgba[0])) {
		// turn the mask into hot pink
		rgba[0] = 0xFF; rgba[1] = 0x00; rgba[2] = 0xFF; rgba[3] = 0x00;
	}

	*((uint32_t*)&out[i*4]) = *((uint32_t*)&rgba[0]);
}

__global__ void from_fullp(const uint64_t num_elements, const float* __restrict__ pixels, __half* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	out[i] = (__half)pixels[i];
}

template <typename T>
__global__ void copy_depth(const uint64_t num_elements, float* __restrict__ depth_dst, const T* __restrict__ depth_pixels, float depth_scale) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	if (depth_pixels == nullptr || depth_scale <= 0.f) {
		depth_dst[i] = 0.f; // no depth data for this entire image. zero it out
	} else {
		depth_dst[i] = depth_pixels[i] * depth_scale;
	}
}

template <typename T>
__global__ void sharpen(const uint64_t num_pixels, const uint32_t w, const T* __restrict__ pix, T* __restrict__ destpix, float center_w, float inv_totalw) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	float rgba[4] = {
		(float)pix[i*4+0]*center_w,
		(float)pix[i*4+1]*center_w,
		(float)pix[i*4+2]*center_w,
		(float)pix[i*4+3]*center_w
	};

	int64_t i2=i-1; if (i2<0) i2=0; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	i2=i-w; if (i2<0) i2=0; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	i2=i+1; if (i2>=num_pixels) i2-=num_pixels; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	i2=i+w; if (i2>=num_pixels) i2-=num_pixels; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	for (int j=0;j<4;++j) destpix[i*4+j]=(T)max(0.f, rgba[j] * inv_totalw);
}

__device__ inline float luma(const Array4f& c) {
	return c[0] * 0.2126f + c[1] * 0.7152f + c[2] * 0.0722f;
}

__global__ void compute_sharpness(Eigen::Vector2i sharpness_resolution, Eigen::Vector2i image_resolution, uint32_t n_images, const void* __restrict__ images_data, EImageDataType image_data_type, float* __restrict__ sharpness_data) {
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t i = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= sharpness_resolution.x() || y >= sharpness_resolution.y() || i>=n_images) return;
	const size_t sharp_size = sharpness_resolution.x() * sharpness_resolution.y();
	sharpness_data += sharp_size * i + x + y * sharpness_resolution.x();

	// overlap patches a bit
	int x_border = 0; // (image_resolution.x()/sharpness_resolution.x())/4;
	int y_border = 0; // (image_resolution.y()/sharpness_resolution.y())/4;

	int x1 = (x*image_resolution.x())/sharpness_resolution.x()-x_border, x2 = ((x+1)*image_resolution.x())/sharpness_resolution.x()+x_border;
	int y1 = (y*image_resolution.y())/sharpness_resolution.y()-y_border, y2 = ((y+1)*image_resolution.y())/sharpness_resolution.y()+y_border;
	// clamp to 1 pixel in from edge
	x1=max(x1,1); y1=max(y1,1);
	x2=min(x2,image_resolution.x()-2); y2=min(y2,image_resolution.y()-2);
	// yes, yes I know I should do a parallel reduction and shared memory and stuff. but we have so many tiles in flight, and this is load-time, meh.
	float tot_lap=0.f,tot_lap2=0.f,tot_lum=0.f;
	float scal=1.f/((x2-x1)*(y2-y1));
	for (int yy=y1;yy<y2;++yy) {
		for (int xx=x1; xx<x2; ++xx) {
			Array4f n, e, s, w, c;
			c = read_rgba(Vector2i{xx, yy}, image_resolution, images_data, image_data_type, i);
			n = read_rgba(Vector2i{xx, yy-1}, image_resolution, images_data, image_data_type, i);
			w = read_rgba(Vector2i{xx-1, yy}, image_resolution, images_data, image_data_type, i);
			s = read_rgba(Vector2i{xx, yy+1}, image_resolution, images_data, image_data_type, i);
			e = read_rgba(Vector2i{xx+1, yy}, image_resolution, images_data, image_data_type, i);
			float lum = luma(c);
			float lap = lum * 4.f - luma(n) - luma(e) - luma(s) - luma(w);
			tot_lap += lap;
			tot_lap2 += lap*lap;
			tot_lum += lum;
		}
	}
	tot_lap*=scal;
	tot_lap2*=scal;
	tot_lum*=scal;
	float variance_of_laplacian = tot_lap2 - tot_lap * tot_lap;
	*sharpness_data = (variance_of_laplacian) ; // / max(0.00001f,tot_lum*tot_lum); // var / (tot+0.001f);
}

NerfDataset create_empty_nerf_dataset(size_t n_images, int aabb_scale, bool is_hdr) {
	NerfDataset result{};
	result.n_images = n_images;
	result.sharpness_resolution = { 128, 72 };
	result.sharpness_data.enlarge( result.sharpness_resolution.x() * result.sharpness_resolution.y() *  result.n_images );
	result.xforms.resize(n_images);
	result.metadata.resize(n_images);
	result.pixelmemory.resize(n_images);
	result.depthmemory.resize(n_images);
	result.raymemory.resize(n_images);
	result.scale = NERF_SCALE;
	result.offset = {0.5f, 0.5f, 0.5f};
	result.aabb_scale = aabb_scale;
	result.is_hdr = is_hdr;
	result.paths = std::vector<std::string>(n_images, "");
	for (size_t i = 0; i < n_images; ++i) {
		result.xforms[i].start = Eigen::Matrix<float, 3, 4>::Identity();
		result.xforms[i].end = Eigen::Matrix<float, 3, 4>::Identity();
	}
	return result;
}

void read_lens(const nlohmann::json& json, Lens& lens, Vector2f& principal_point, Vector4f& rolling_shutter) {
	ELensMode mode = ELensMode::Perspective;

	ELensMode opencv_mode = json.value("is_fisheye", false) ? ELensMode::OpenCVFisheye : ELensMode::OpenCV;
	auto read_opencv_parameter = [&](const std::string& name, size_t idx) {
		if (json.contains(name)) {
			lens.params[idx] = json[name];
			if (lens.params[idx] != 0.f) {
				mode = opencv_mode;
			}
		}
	};

	read_opencv_parameter("k1", 0);
	read_opencv_parameter("k2", 1);
	read_opencv_parameter("k3", 2);
	read_opencv_parameter("k4", 3);

	read_opencv_parameter("p1", 2);
	read_opencv_parameter("p2", 3);

	if (json.contains("cx")) {
		principal_point.x() = (float)json["cx"] / (float)json["w"];
	}

	if (json.contains("cy")) {
		principal_point.y() = (float)json["cy"] / (float)json["h"];
	}

	if (json.contains("rolling_shutter")) {
		// The rolling shutter is a float4 of [A,B,C,D] where the time
		// for each pixel is t= A + B * u + C * v + D * motionblur_time,
		// where u and v are the pixel coordinates within (0-1).
		// The resulting t is used to interpolate between the start
		// and end transforms for each training xform.
		float motionblur_amount = 0.f;
		if (json["rolling_shutter"].size() >= 4) {
			motionblur_amount = float(json["rolling_shutter"][3]);
		}

		rolling_shutter = {float(json["rolling_shutter"][0]), float(json["rolling_shutter"][1]), float(json["rolling_shutter"][2]), motionblur_amount};
	}

	if (json.contains("ftheta_p0")) {
		lens.params[0] = json["ftheta_p0"];
		lens.params[1] = json["ftheta_p1"];
		lens.params[2] = json["ftheta_p2"];
		lens.params[3] = json["ftheta_p3"];
		lens.params[4] = json["ftheta_p4"];
		lens.params[5] = json["w"];
		lens.params[6] = json["h"];
		mode = ELensMode::FTheta;
	}

	if (json.contains("latlong")) {
		mode = ELensMode::LatLong;
	}

	// If there was an outer distortion mode, don't override it with nothing.
	if (mode != ELensMode::Perspective) {
		lens.mode = mode;
	}
}

bool read_focal_length(const nlohmann::json &json, Vector2f &focal_length, const Vector2i &res) {
	auto read_focal_length = [&](int resolution, const std::string& axis) {
		if (json.contains(axis + "_fov")) {
			return fov_to_focal_length(resolution, (float)json[axis + "_fov"]);
		} else if (json.contains("fl_"s + axis)) {
			return (float)json["fl_"s + axis];
		} else if (json.contains("camera_angle_"s + axis)) {
			return fov_to_focal_length(resolution, (float)json["camera_angle_"s + axis] * 180 / PI());
		} else {
			return 0.0f;
		}
	};

	// x_fov is in degrees, camera_angle_x in radians. Yes, it's silly.
	float x_fl = read_focal_length(res.x(), "x");
	float y_fl = read_focal_length(res.y(), "y");

	if (x_fl != 0) {
		focal_length = Vector2f::Constant(x_fl);
		if (y_fl != 0) {
			focal_length.y() = y_fl;
		}
	} else if (y_fl != 0) {
		focal_length = Vector2f::Constant(y_fl);
	} else {
		return false;
	}
	return true;
}

NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount) {
	if (jsonpaths.empty()) {
		throw std::runtime_error{"Cannot load NeRF data from an empty set of paths."};
	}

	tlog::info() << "Loading NeRF dataset from";

	NerfDataset result{};

	std::ifstream f{jsonpaths.front().str()};
	nlohmann::json transforms = nlohmann::json::parse(f, nullptr, true, true);

	ThreadPool pool;

	struct LoadedImageInfo {
		Eigen::Vector2i res = Eigen::Vector2i::Zero();
		bool image_data_on_gpu = false;
		EImageDataType image_type = EImageDataType::None;
		bool white_transparent = false;
		bool black_transparent = false;
		uint32_t mask_color = 0;
		void *pixels = nullptr;
		uint16_t *depth_pixels = nullptr;
		Ray *rays = nullptr;
		float depth_scale = -1.f;
	};
	std::vector<LoadedImageInfo> images;
	LoadedImageInfo info = {};

	if (transforms["camera"].is_array()) {
		throw std::runtime_error{"hdf5 is no longer supported. please use the hdf52nerf.py conversion script"};
	}

	// nerf original format
	std::vector<nlohmann::json> jsons;
	std::transform(
		jsonpaths.begin(), jsonpaths.end(),
		std::back_inserter(jsons), [](const auto& path) {
			return nlohmann::json::parse(std::ifstream{path.str()}, nullptr, true, true);
		}
	);

	std::vector<std::string> supported_image_formats = {
		"png", "jpg", "jpeg", "bmp", "gif", "tga", "pic", "pnm", "psd", "exr",
	};

	auto resolve_path = [&supported_image_formats](const fs::path& base_path, const fs::path& local_path) {
		fs::path path = local_path.is_absolute() ? local_path : (base_path / local_path);
		if (path.extension().empty() && !path.exists()) {
			for (const auto& format : supported_image_formats) {
				if (path.with_extension(format).exists()) {
					return path.with_extension(format);
				}
			}
		}

		return path;
	};

	result.n_images = 0;
	for (size_t i = 0; i < jsons.size(); ++i) {
		auto& json = jsons[i];
		fs::path base_path = jsonpaths[i].parent_path();


		if (!json.contains("frames") || !json["frames"].is_array()) {
			tlog::warning() << "  " << jsonpaths[i] << " does not contain any frames. Skipping.";
			continue;
		}
		tlog::info() << "  " << jsonpaths[i];
		auto& frames = json["frames"];

		float sharpness_discard_threshold = json.value("sharpness_discard_threshold", 0.0f); // Keep all by default

		std::sort(frames.begin(), frames.end(), [](const auto& frame1, const auto& frame2) {
			return frame1["file_path"] < frame2["file_path"];
		});

		for (auto&& frame : frames) {
			// Compatibility with Windows paths on Linux. (Breaks linux filenames with "\\" in them, which is acceptable for us.)
			frame["file_path"] = replace_all(frame["file_path"], "\\", "/");
		}

		if (json.contains("n_frames")) {
			size_t cull_idx = std::min(frames.size(), (size_t)json["n_frames"]);
			frames.get_ptr<nlohmann::json::array_t*>()->resize(cull_idx);
		}

		if (frames[0].contains("sharpness")) {
			auto frames_copy = frames;
			frames.clear();

			// Kill blurrier frames than their neighbors
			const int neighborhood_size = 3;
			for (int i = 0; i < (int)frames_copy.size(); ++i) {
				float mean_sharpness = 0.0f;
				int mean_start = std::max(0, i-neighborhood_size);
				int mean_end = std::min(i + neighborhood_size, (int)frames_copy.size() - 1);
				for (int j = mean_start; j < mean_end; ++j) {
					mean_sharpness += float(frames_copy[j].value("sharpness", 1.0));
				}

				mean_sharpness /= (mean_end - mean_start);

				if (resolve_path(base_path, frames_copy[i]["file_path"]).exists() && frames_copy[i].value("sharpness", 1.0) > sharpness_discard_threshold * mean_sharpness) {
					frames.emplace_back(frames_copy[i]);
				} else {
					// tlog::info() << "discarding frame " << frames_copy[i]["file_path"];
					// fs::remove(resolve_path(base_path, frames_copy[i]["file_path"]));
				}
			}
		}

		for (size_t i = 0; i < frames.size(); ++i) {
			result.paths.emplace_back(frames[i]["file_path"]);
		}

		result.n_images += frames.size();
	}

	images.resize(result.n_images);
	result.xforms.resize(result.n_images);
	result.metadata.resize(result.n_images);
	result.pixelmemory.resize(result.n_images);
	result.depthmemory.resize(result.n_images);
	result.raymemory.resize(result.n_images);

	result.scale = NERF_SCALE;
	result.offset = {0.5f, 0.5f, 0.5f};

	std::vector<std::future<void>> futures;

	size_t image_idx = 0;

	if (result.n_images == 0) {
		throw std::invalid_argument{"No training images were found for NeRF training!"};
	}

	auto progress = tlog::progress(result.n_images);

	result.from_mitsuba = false;
	bool fix_premult = false;
	bool enable_ray_loading = true;
	bool enable_depth_loading = true;
	std::atomic<int> n_loaded{0};
	BoundingBox cam_aabb;
	for (size_t i = 0; i < jsons.size(); ++i) {
		auto& json = jsons[i];

		fs::path base_path = jsonpaths[i].parent_path();
		std::string jp = jsonpaths[i].str();
		auto lastdot = jp.find_last_of('.'); if (lastdot==std::string::npos) lastdot = jp.length();
		auto lastunderscore = jp.find_last_of('_'); if (lastunderscore == std::string::npos) lastunderscore=lastdot; else lastunderscore++;
		std::string part_after_underscore(jp.begin()+lastunderscore,jp.begin()+lastdot);

		if (json.contains("enable_ray_loading")) {
			enable_ray_loading = bool(json["enable_ray_loading"]);
			tlog::info() << "enable_ray_loading=" << enable_ray_loading;
		}
		if (json.contains("enable_depth_loading")) {
			enable_depth_loading = bool(json["enable_depth_loading"]);
			tlog::info() << "enable_depth_loading is " << enable_depth_loading;
		}

		if (json.contains("normal_mts_args")) {
			result.from_mitsuba = true;
		}

		if (json.contains("fix_premult")) {
			fix_premult = (bool)json["fix_premult"];
		}

		if (result.from_mitsuba) {
			result.scale = 0.66f;
			result.offset = {0.25f * result.scale, 0.25f * result.scale, 0.25f * result.scale};
		}

		if (json.contains("render_aabb")) {
			result.render_aabb.min={float(json["render_aabb"][0][0]),float(json["render_aabb"][0][1]),float(json["render_aabb"][0][2])};
			result.render_aabb.max={float(json["render_aabb"][1][0]),float(json["render_aabb"][1][1]),float(json["render_aabb"][1][2])};
		}

		if (json.contains("sharpen")) {
			sharpen_amount = json["sharpen"];
		}

		if (json.contains("white_transparent")) {
			info.white_transparent = bool(json["white_transparent"]);
		}

		if (json.contains("black_transparent")) {
			info.black_transparent = bool(json["black_transparent"]);
		}

		if (json.contains("scale")) {
			result.scale = json["scale"];
		}

		if (json.contains("importance_sampling")) {
			result.wants_importance_sampling = json["importance_sampling"];
		}

		if (json.contains("n_extra_learnable_dims")) {
			result.n_extra_learnable_dims = json["n_extra_learnable_dims"];
		}

		Lens lens = {};
		Vector2f principal_point = Vector2f::Constant(0.5f);
		Vector4f rolling_shutter = Vector4f::Zero();

		if (json.contains("integer_depth_scale")) {
			info.depth_scale = json["integer_depth_scale"];
		}

		// Lens parameters
		read_lens(json, lens, principal_point, rolling_shutter);

		if (json.contains("aabb_scale")) {
			result.aabb_scale = json["aabb_scale"];
		}

		if (json.contains("offset")) {
			result.offset =
				json["offset"].is_array() ?
				Vector3f{float(json["offset"][0]), float(json["offset"][1]), float(json["offset"][2])} :
				Vector3f{float(json["offset"]), float(json["offset"]), float(json["offset"])};
		}

		if (json.contains("aabb")) {
			// map the given aabb of the form [[minx,miny,minz],[maxx,maxy,maxz]] via an isotropic scale and translate to fit in the (0,0,0)-(1,1,1) cube, with the given center at 0.5,0.5,0.5
			const auto& aabb=json["aabb"];
			float length = std::max(0.000001f,std::max(std::max(std::abs(float(aabb[1][0])-float(aabb[0][0])),std::abs(float(aabb[1][1])-float(aabb[0][1]))),std::abs(float(aabb[1][2])-float(aabb[0][2]))));
			result.scale = 1.f/length;
			result.offset = { ((float(aabb[1][0])+float(aabb[0][0]))*0.5f)*-result.scale + 0.5f , ((float(aabb[1][1])+float(aabb[0][1]))*0.5f)*-result.scale + 0.5f,((float(aabb[1][2])+float(aabb[0][2]))*0.5f)*-result.scale + 0.5f};
		}

		if (json.contains("frames") && json["frames"].is_array()) {
			for (int j = 0; j < json["frames"].size(); ++j) {
				auto& frame = json["frames"][j];
				nlohmann::json& jsonmatrix_start = frame.contains("transform_matrix_start") ? frame["transform_matrix_start"] : frame["transform_matrix"];
				nlohmann::json& jsonmatrix_end = frame.contains("transform_matrix_end") ? frame["transform_matrix_end"] : jsonmatrix_start;
				const Vector3f p = Vector3f{float(jsonmatrix_start[0][3]), float(jsonmatrix_start[1][3]), float(jsonmatrix_start[2][3])} * result.scale + result.offset;
				const Vector3f q = Vector3f{float(jsonmatrix_end[0][3]), float(jsonmatrix_end[1][3]), float(jsonmatrix_end[2][3])} * result.scale + result.offset;
				cam_aabb.enlarge(p);
				cam_aabb.enlarge(q);
			}
		}

		if (json.contains("up")) {
			// axes are permuted as for the xforms below
			result.up[0] = float(json["up"][1]);
			result.up[1] = float(json["up"][2]);
			result.up[2] = float(json["up"][0]);
		}

		if (json.contains("envmap") && result.envmap_resolution.isZero()) {
			fs::path envmap_path = resolve_path(base_path, json["envmap"]);
			if (!envmap_path.exists()) {
				throw std::runtime_error{fmt::format("Environment map {} does not exist.", envmap_path.str())};
			}

			if (equals_case_insensitive(envmap_path.extension(), "exr")) {
				result.envmap_data = load_exr(envmap_path.str(), result.envmap_resolution.x(), result.envmap_resolution.y());
				result.is_hdr = true;
			} else {
				result.envmap_data = load_stbi(envmap_path.str(), result.envmap_resolution.x(), result.envmap_resolution.y());
			}
		}


		if (json.contains("frames") && json["frames"].is_array()) pool.parallel_for_async<size_t>(0, json["frames"].size(), [&progress, &n_loaded, &result, &images, &json, &resolve_path, &supported_image_formats, base_path, image_idx, info, rolling_shutter, principal_point, lens, part_after_underscore, fix_premult, enable_depth_loading, enable_ray_loading](size_t i) {
			size_t i_img = i + image_idx;
			auto& frame = json["frames"][i];
			LoadedImageInfo& dst = images[i_img];
			dst = info; // copy defaults

			std::string json_provided_path = frame["file_path"];
			if (json_provided_path == "") {
				char buf[256];
				snprintf(buf, 256, "%s_%03d/rgba.png", part_after_underscore.c_str(), (int)i);
				json_provided_path = buf;
			}

			fs::path path = resolve_path(base_path, json_provided_path);
			if (!path.exists()) {
				throw std::runtime_error{fmt::format("Could not find image file '{}'.", path.str())};
			}

			int comp = 0;
			if (equals_case_insensitive(path.extension(), "exr")) {
				dst.pixels = load_exr_to_gpu(&dst.res.x(), &dst.res.y(), path.str().c_str(), fix_premult);
				dst.image_type = EImageDataType::Half;
				dst.image_data_on_gpu = true;
				result.is_hdr = true;
			} else {
				dst.image_data_on_gpu = false;
				uint8_t* img = stbi_load(path.str().c_str(), &dst.res.x(), &dst.res.y(), &comp, 4);
				if (!img) {
					throw std::runtime_error{"Could not open image file: "s + std::string{stbi_failure_reason()}};
				}

				fs::path alphapath = resolve_path(base_path, fmt::format("{}.alpha.{}", frame["file_path"], path.extension()));
				if (alphapath.exists()) {
					int wa = 0, ha = 0;
					uint8_t* alpha_img = stbi_load(alphapath.str().c_str(), &wa, &ha, &comp, 4);
					if (!alpha_img) {
						throw std::runtime_error{"Could not load alpha image "s + alphapath.str()};
					}

					ScopeGuard mem_guard{[&]() { stbi_image_free(alpha_img); }};
					if (wa != dst.res.x() || ha != dst.res.y()) {
						throw std::runtime_error{fmt::format("Alpha image {} has wrong resolution.", alphapath.str())};
					}

					tlog::success() << "Alpha loaded from " << alphapath;
					for (int i = 0; i < dst.res.prod(); ++i) {
						img[i*4+3] = (uint8_t)(255.0f*srgb_to_linear(alpha_img[i*4]*(1.f/255.f))); // copy red channel of alpha to alpha.png to our alpha channel
					}
				}

				fs::path maskpath = path.parent_path() / fmt::format("dynamic_mask_{}.png", path.basename());
				if (maskpath.exists()) {
					int wa = 0, ha = 0;
					uint8_t* mask_img = stbi_load(maskpath.str().c_str(), &wa, &ha, &comp, 4);
					if (!mask_img) {
						throw std::runtime_error{fmt::format("Dynamic mask {} could not be loaded.", maskpath.str())};
					}

					ScopeGuard mem_guard{[&]() { stbi_image_free(mask_img); }};
					if (wa != dst.res.x() || ha != dst.res.y()) {
						throw std::runtime_error{fmt::format("Dynamic mask {} has wrong resolution.", maskpath.str())};
					}

					dst.mask_color = 0x00FF00FF; // HOT PINK
					for (int i = 0; i < dst.res.prod(); ++i) {
						if (mask_img[i*4] != 0 || mask_img[i*4+1] != 0 || mask_img[i*4+2] != 0) {
							*(uint32_t*)&img[i*4] = dst.mask_color;
						}
					}
				}

				dst.pixels = img;
				dst.image_type = EImageDataType::Byte;
			}

			if (!dst.pixels) {
				throw std::runtime_error{fmt::format("Could not load image file '{}'.", path.str())};
			}

			if (enable_depth_loading && info.depth_scale > 0.f && frame.contains("depth_path")) {
				fs::path depthpath = resolve_path(base_path, frame["depth_path"]);
				if (depthpath.exists()) {
					int wa = 0, ha = 0;
					dst.depth_pixels = stbi_load_16(depthpath.str().c_str(), &wa, &ha, &comp, 1);
					if (!dst.depth_pixels) {
						throw std::runtime_error{fmt::format("Could not load depth image '{}'.", depthpath.str())};
					}

					if (wa != dst.res.x() || ha != dst.res.y()) {
						throw std::runtime_error{fmt::format("Depth image {} has wrong resolution.", depthpath.str())};
					}
				}
			}

			fs::path rayspath = path.parent_path() / fmt::format("rays_{}.dat", path.basename());
			if (enable_ray_loading && rayspath.exists()) {
				uint32_t n_pixels = dst.res.prod();
				dst.rays = (Ray*)malloc(n_pixels * sizeof(Ray));

				std::ifstream rays_file{rayspath.str(), std::ios::binary};
				rays_file.read((char*)dst.rays, n_pixels * sizeof(Ray));

				std::streampos fsize = 0;
				fsize = rays_file.tellg();
				rays_file.seekg(0, std::ios::end);
				fsize = rays_file.tellg() - fsize;

				if (fsize > 0) {
					tlog::warning() << fsize << " bytes remaining in rays file " << rayspath;
				}

				for (uint32_t px = 0; px < n_pixels; ++px) {
					result.nerf_ray_to_ngp(dst.rays[px]);
				}

				result.has_rays = true;
			}

			nlohmann::json& jsonmatrix_start = frame.contains("transform_matrix_start") ? frame["transform_matrix_start"] : frame["transform_matrix"];
			nlohmann::json& jsonmatrix_end = frame.contains("transform_matrix_end") ? frame["transform_matrix_end"] : jsonmatrix_start;

			if (frame.contains("driver_parameters")) {
				Eigen::Vector3f light_dir(
					frame["driver_parameters"].value("LightX", 0.f),
					frame["driver_parameters"].value("LightY", 0.f),
					frame["driver_parameters"].value("LightZ", 0.f)
				);
				result.metadata[i_img].light_dir = result.nerf_direction_to_ngp(light_dir.normalized());
				result.has_light_dirs = true;
				result.n_extra_learnable_dims = 0;
			}

			bool got_fl = read_focal_length(json, result.metadata[i_img].focal_length, dst.res);
			got_fl |= read_focal_length(frame, result.metadata[i_img].focal_length, dst.res);
			if (!got_fl) {
				throw std::runtime_error{"Couldn't read fov."};
			}

			for (int m = 0; m < 3; ++m) {
				for (int n = 0; n < 4; ++n) {
					result.xforms[i_img].start(m, n) = float(jsonmatrix_start[m][n]);
					result.xforms[i_img].end(m, n) = float(jsonmatrix_end[m][n]);
				}
			}

			// set these from the base settings
			result.metadata[i_img].rolling_shutter = rolling_shutter;
			result.metadata[i_img].principal_point = principal_point;
			result.metadata[i_img].lens = lens;
			// see if there is a per-frame override
			read_lens(frame, result.metadata[i_img].lens, result.metadata[i_img].principal_point, result.metadata[i_img].rolling_shutter);

			result.xforms[i_img].start = result.nerf_matrix_to_ngp(result.xforms[i_img].start);
			result.xforms[i_img].end = result.nerf_matrix_to_ngp(result.xforms[i_img].end);

			progress.update(++n_loaded);
		}, futures);

		if (json.contains("frames")) {
			image_idx += json["frames"].size();
		}

	}

	wait_all(futures);

	tlog::success() << "Loaded " << images.size() << " images after " << tlog::durationToString(progress.duration());
	tlog::info() << "  cam_aabb=" << cam_aabb;

	if (result.has_rays) {
		tlog::success() << "Loaded per-pixel rays.";
	}
	if (!images.empty() && images[0].mask_color) {
		tlog::success() << "Loaded dynamic masks.";
	}

	result.sharpness_resolution = { 128, 72 };
	result.sharpness_data.enlarge( result.sharpness_resolution.x() * result.sharpness_resolution.y() *  result.n_images );

	// copy / convert images to the GPU
	for (uint32_t i = 0; i < result.n_images; ++i) {
		const LoadedImageInfo& m = images[i];
		result.set_training_image(i, m.res, m.pixels, m.depth_pixels, m.depth_scale * result.scale, m.image_data_on_gpu, m.image_type, EDepthDataType::UShort, sharpen_amount, m.white_transparent, m.black_transparent, m.mask_color, m.rays);
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	// free memory
	for (uint32_t i = 0; i < result.n_images; ++i) {
		if (images[i].image_data_on_gpu) {
			CUDA_CHECK_THROW(cudaFree(images[i].pixels));
		} else {
			free(images[i].pixels);
		}
		free(images[i].rays);
		free(images[i].depth_pixels);
	}
	return result;
}

void NerfDataset::set_training_image(int frame_idx, const Eigen::Vector2i& image_resolution, const void* pixels, const void* depth_pixels, float depth_scale, bool image_data_on_gpu, EImageDataType image_type, EDepthDataType depth_type, float sharpen_amount, bool white_transparent, bool black_transparent, uint32_t mask_color, const Ray *rays) {
	if (frame_idx < 0 || frame_idx >= n_images) {
		throw std::runtime_error{"NerfDataset::set_training_image: invalid frame index"};
	}
	size_t n_pixels = image_resolution.prod();
	size_t img_size = n_pixels * 4; // 4 channels
	size_t image_type_stride = image_type_size(image_type);
	// copy to gpu if we need to do a conversion
	GPUMemory<uint8_t> images_data_gpu_tmp;
	GPUMemory<uint8_t> depth_tmp;
	if (!image_data_on_gpu && image_type == EImageDataType::Byte) {
		images_data_gpu_tmp.resize(img_size * image_type_stride);
		images_data_gpu_tmp.copy_from_host((uint8_t*)pixels);
		pixels = images_data_gpu_tmp.data();

		if (depth_pixels) {
			depth_tmp.resize(n_pixels * depth_type_size(depth_type));
			depth_tmp.copy_from_host((uint8_t*)depth_pixels);
			depth_pixels = depth_tmp.data();
		}

		image_data_on_gpu = true;
	}

	// copy or convert the pixels
	pixelmemory[frame_idx].resize(img_size * image_type_size(image_type));
	void* dst = pixelmemory[frame_idx].data();

	switch (image_type) {
		default: throw std::runtime_error{"unknown image type in set_training_image"};
		case EImageDataType::Byte: linear_kernel(convert_rgba32, 0, nullptr, n_pixels, (uint8_t*)pixels, (uint8_t*)dst, white_transparent, black_transparent, mask_color); break;
		case EImageDataType::Half: // fallthrough is intended
		case EImageDataType::Float: CUDA_CHECK_THROW(cudaMemcpy(dst, pixels, img_size * image_type_size(image_type), image_data_on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)); break;
	}

	// copy over depths if provided
	if (depth_scale >= 0.f) {
		depthmemory[frame_idx].resize(img_size);
		float* depth_dst = depthmemory[frame_idx].data();

		if (depth_pixels && !image_data_on_gpu) {
			depth_tmp.resize(n_pixels * depth_type_size(depth_type));
			depth_tmp.copy_from_host((uint8_t*)depth_pixels);
			depth_pixels = depth_tmp.data();
		}

		switch (depth_type) {
			default: throw std::runtime_error{"unknown depth type in set_training_image"};
			case EDepthDataType::UShort: linear_kernel(copy_depth<uint16_t>, 0, nullptr, n_pixels, depth_dst, (const uint16_t*)depth_pixels, depth_scale); break;
			case EDepthDataType::Float: linear_kernel(copy_depth<float>, 0, nullptr, n_pixels, depth_dst, (const float*)depth_pixels, depth_scale); break;
		}
	} else {
		depthmemory[frame_idx].free_memory();
	}

	// apply requested sharpening
	if (sharpen_amount > 0.f) {
		if (image_type == EImageDataType::Byte) {
			tcnn::GPUMemory<uint8_t> images_data_half(img_size * sizeof(__half));
			linear_kernel(from_rgba32<__half>, 0, nullptr, n_pixels, (uint8_t*)pixels, (__half*)images_data_half.data(), white_transparent, black_transparent, mask_color);
			pixelmemory[frame_idx] = std::move(images_data_half);
			dst = pixelmemory[frame_idx].data();
			image_type = EImageDataType::Half;
		}

		assert(image_type == EImageDataType::Half || image_type == EImageDataType::Float);

		tcnn::GPUMemory<uint8_t> images_data_sharpened(img_size * image_type_size(image_type));

		float center_w = 4.f + 1.f / sharpen_amount; // center_w ranges from 5 (strong sharpening) to infinite (no sharpening)
		if (image_type == EImageDataType::Half) {
			linear_kernel(sharpen<__half>, 0, nullptr, n_pixels, image_resolution.x(), (__half*)dst, (__half*)images_data_sharpened.data(), center_w, 1.f / (center_w - 4.f));
		} else {
			linear_kernel(sharpen<float>, 0, nullptr, n_pixels, image_resolution.x(), (float*)dst, (float*)images_data_sharpened.data(), center_w, 1.f / (center_w - 4.f));
		}

		pixelmemory[frame_idx] = std::move(images_data_sharpened);
		dst = pixelmemory[frame_idx].data();
	}

	if (sharpness_data.size() > 0) {
		// compute overall sharpness
		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)sharpness_resolution.x(), threads.x), div_round_up((uint32_t)sharpness_resolution.y(), threads.y), 1 };
		sharpness_data.enlarge(sharpness_resolution.x() * sharpness_resolution.y());
		compute_sharpness<<<blocks, threads, 0, nullptr>>>(sharpness_resolution, image_resolution, 1, dst, image_type, sharpness_data.data() + sharpness_resolution.x() * sharpness_resolution.y() * (size_t)frame_idx);
	}

	metadata[frame_idx].pixels = pixelmemory[frame_idx].data();
	metadata[frame_idx].depth = depthmemory[frame_idx].data();
	metadata[frame_idx].resolution = image_resolution;
	metadata[frame_idx].image_data_type = image_type;
	if (rays) {
		raymemory[frame_idx].resize(n_pixels);
		CUDA_CHECK_THROW(cudaMemcpy(raymemory[frame_idx].data(), rays, n_pixels * sizeof(Ray), cudaMemcpyHostToDevice));
	} else {
		raymemory[frame_idx].free_memory();
	}
	metadata[frame_idx].rays = raymemory[frame_idx].data();
	update_metadata(frame_idx, frame_idx + 1);
}

void NerfDataset::update_metadata(int first, int last) {
	if (last < 0) {
		last = n_images;
	}

	if (last > n_images) {
		last = n_images;
	}

	int n = last - first;
	if (n <= 0) {
		return;
	}

	metadata_gpu.enlarge(last);
	CUDA_CHECK_THROW(cudaMemcpy(metadata_gpu.data() + first, metadata.data() + first, n * sizeof(TrainingImageMetadata), cudaMemcpyHostToDevice));
}

NGP_NAMESPACE_END
