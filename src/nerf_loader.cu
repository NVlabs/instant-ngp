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


#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
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

#if defined(__NVCC__)
#if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#  pragma nv_diag_suppress 550
#else
#  pragma diag_suppress 550
#endif
#endif
#include <stb_image/stb_image.h>
#if defined(__NVCC__)
#if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#  pragma nv_diag_default 550
#else
#  pragma diag_default 550
#endif
#endif

using namespace tcnn;
using namespace std::literals;
using namespace Eigen;
namespace fs = filesystem;


NGP_NAMESPACE_BEGIN

// how much to scale the scene by vs the original nerf dataset; we want to fit the thing in the unit cube
static constexpr float NERF_SCALE = 0.33f;


__global__ void from_fullp(const uint64_t num_elements, const float* __restrict__ pixels, __half* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	out[i] = (__half)pixels[i];
}

__global__ void sharpen(const uint64_t num_pixels, const uint32_t w, const __half* __restrict__ pix,__half* __restrict__ destpix, float center_w, float inv_totalw) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;
	float rgba[4]={
		__half2float(pix[i*4+0])*center_w,
		__half2float(pix[i*4+1])*center_w,
		__half2float(pix[i*4+2])*center_w,
		__half2float(pix[i*4+3])*center_w};

	int64_t i2=i-1; if (i2<0) i2=0; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=__half2float(pix[i2++]);
	i2=i-w; if (i2<0) i2=0; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=__half2float(pix[i2++]);
	i2=i+1; if (i2>=num_pixels) i2-=num_pixels; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=__half2float(pix[i2++]);
	i2=i+w; if (i2>=num_pixels) i2-=num_pixels; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=__half2float(pix[i2++]);
	for (int j=0;j<4;++j) destpix[i*4+j]=(__half)max(0.f,rgba[j] * inv_totalw);
}

__device__ inline float luma(__half c[4]) {
	return float(c[0]) * 0.2126f + float(c[1]) * 0.7152f + float(c[2]) * 0.0722f;
}

__global__ void compute_sharpness(Eigen::Vector2i sharpness_resolution, Eigen::Vector2i image_resolution, uint32_t n_images, const __half* __restrict__ images_data, float* __restrict__ sharpness_data) {
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t i = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= sharpness_resolution.x() || y >= sharpness_resolution.y() || i>=n_images) return;
	const size_t sharp_size = sharpness_resolution.x() * sharpness_resolution.y();
	const size_t img_size = image_resolution.x() * image_resolution.y() * 4;
	sharpness_data += sharp_size * i + x + y * sharpness_resolution.x();
	images_data += img_size * i;

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
		uint32_t idx = x1+yy*image_resolution.x();
		for (int xx=x1;xx<x2;++xx, ++idx) {
			__half n[4],e[4],s[4],w[4],c[4];
			*(uint64_t*)&c[0] = ((const uint64_t*)images_data)[idx];
			*(uint64_t*)&n[0] = ((const uint64_t*)images_data)[idx-image_resolution.x()];
			*(uint64_t*)&e[0] = ((const uint64_t*)images_data)[idx-1];
			*(uint64_t*)&s[0] = ((const uint64_t*)images_data)[idx+image_resolution.x()];
			*(uint64_t*)&w[0] = ((const uint64_t*)images_data)[idx+1];
			float lum=luma(c);
			float lap=lum*4.f - luma(n) - luma(e) - luma(s) - luma(w);
			tot_lap+=lap;
			tot_lap2+=lap*lap;
			tot_lum+=lum;
		}
	}
	tot_lap*=scal;
	tot_lap2*=scal;
	tot_lum*=scal;
	float variance_of_laplacian = tot_lap2 - tot_lap * tot_lap;
	*sharpness_data = (variance_of_laplacian) ; // / max(0.00001f,tot_lum*tot_lum); // var / (tot+0.001f);
}

bool ends_with(const std::string& str, const std::string& suffix) {
	return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
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

	enum class ImageDataType {
		None,
		Float,
		Half,
		Byte,
	};
	bool image_data_on_gpu = false;
	bool has_rays = false;
	ImageDataType image_type = ImageDataType::None;
	bool white_transparent = false;
	bool black_transparent = false;
	uint32_t mask_color = 0;

	std::vector<void*> images;
	std::vector<Ray*> rays;

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

	result.n_images = 0;
	for (size_t i = 0; i < jsons.size(); ++i) {
		auto& json = jsons[i];
		fs::path basepath = jsonpaths[i].parent_path();
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
				int mean_end = std::min(i+neighborhood_size, (int)frames_copy.size()-1);
				for (int j = mean_start; j < mean_end; ++j) {
					mean_sharpness += float(frames_copy[j]["sharpness"]);
				}
				mean_sharpness /= (mean_end - mean_start);

				if ((basepath / fs::path(std::string(frames_copy[i]["file_path"]))).exists() && frames_copy[i]["sharpness"] > sharpness_discard_threshold * mean_sharpness) {
					frames.emplace_back(frames_copy[i]);
				} else {
					// tlog::info() << "discarding frame " << frames_copy[i]["file_path"];
					// fs::remove(basepath / fs::path(std::string(frames_copy[i]["file_path"])));
				}
			}
		}

		result.n_images += frames.size();
	}

	images.resize(result.n_images, nullptr);
	rays.resize(result.n_images, nullptr);
	result.xforms.resize(result.n_images);
	result.focal_lengths.resize(result.n_images);

	result.scale = NERF_SCALE;
	result.offset = {0.5f, 0.5f, 0.5f};

	std::vector<std::future<void>> futures;

	size_t image_idx = 0;

	if (result.n_images==0) {
		throw std::invalid_argument{"No training images were found for NeRF training!"};
	}

	auto progress = tlog::progress(result.n_images);

	result.from_mitsuba = false;
	bool fix_premult = false;
	std::atomic<int> n_loaded{0};
	BoundingBox cam_aabb;
	for (size_t i = 0; i < jsons.size(); ++i) {
		auto& json = jsons[i];
		if (!json.contains("frames") || !json["frames"].is_array()) {
			continue;
		}
		fs::path basepath = jsonpaths[i].parent_path();
		std::string jp = jsonpaths[i].str();
		auto lastdot=jp.find_last_of('.'); if (lastdot==std::string::npos) lastdot=jp.length();
		auto lastunderscore=jp.find_last_of('_'); if (lastunderscore==std::string::npos) lastunderscore=lastdot; else lastunderscore++;
		std::string part_after_underscore(jp.begin()+lastunderscore,jp.begin()+lastdot);

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

		if (json.contains("white_transparent"))
			white_transparent = bool(json["white_transparent"]);
		if (json.contains("black_transparent"))
			black_transparent = bool(json["black_transparent"]);

		if (json.contains("scale")) {
			result.scale = json["scale"];
		}

		if (json.contains("importance_sampling")) {
			result.wants_importance_sampling = json["importance_sampling"];
		}

		// Camera distortion
		{
			if (json.contains("k1")) {
				result.camera_distortion.params[0] = json["k1"];
			}

			if (json.contains("k2")) {
				result.camera_distortion.params[1] = json["k2"];
			}

			if (json.contains("p1")) {
				result.camera_distortion.params[2] = json["p1"];
			}

			if (json.contains("p2")) {
				result.camera_distortion.params[3] = json["p2"];
			}

			if (json.contains("cx")) {
				result.principal_point.x() = (float)json["cx"] / (float)json["w"];
			}

			if (json.contains("cy")) {
				result.principal_point.y() = (float)json["cy"] / (float)json["h"];
			}
		}

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

		for (int j = 0; j < json["frames"].size(); ++j) {
			nlohmann::json& jsonmatrix = json["frames"][j]["transform_matrix"];
			auto p = Vector3f{float(jsonmatrix[0][3]), float(jsonmatrix[1][3]), float(jsonmatrix[2][3])} * result.scale + result.offset;
			cam_aabb.enlarge(p);
		}

		if (json.contains("up")) {
			// axes are permuted as for the xforms below
			result.up[0] = float(json["up"][1]);
			result.up[1] = float(json["up"][2]);
			result.up[2] = float(json["up"][0]);
		}

		if (json.contains("envmap") && result.envmap_resolution.isZero()) {
			std::string json_provided_path = json["envmap"];
			fs::path envmap_path = basepath / json_provided_path;
			if (!envmap_path.exists()) {
				throw std::runtime_error{std::string{"Environment map path "} + envmap_path.str() + " does not exist."};
			}

			if (equals_case_insensitive(envmap_path.extension(), "exr")) {
				result.envmap_data = load_exr(envmap_path.str(), result.envmap_resolution.x(), result.envmap_resolution.y());
				result.is_hdr = true;
			} else {
				result.envmap_data = load_stbi(envmap_path.str(), result.envmap_resolution.x(), result.envmap_resolution.y());
			}
		}

		pool.parallelForAsync<size_t>(0, json["frames"].size(), [&, basepath, image_idx](size_t i) {
			size_t i_img = i + image_idx;
			auto& frame = json["frames"][i];

			std::string json_provided_path(frame["file_path"]);
			if (json_provided_path == "") {
				char buf[256];
				snprintf(buf,256,"%s_%03d/rgba.png", part_after_underscore.c_str(), (int) i);
				json_provided_path = buf;
			}
			fs::path path = basepath / json_provided_path;

			if (path.extension() == "") {
				path = path.with_extension("png");
				if (!path.exists()) {
					path = path.with_extension("exr");
				}
				if (!path.exists()) {
					throw std::runtime_error{ "Could not find image file: " + path.str()};
				}
			}
			Vector2i res = Vector2i::Zero();
			int comp = 0;
			if (equals_case_insensitive(path.extension(), "exr")) {
				images[i_img] = load_exr_to_gpu(&res.x(), &res.y(), path.str().c_str(), fix_premult);

				if (image_type != ImageDataType::None && image_type != ImageDataType::Half) {
					throw std::runtime_error{ "May not mix png and exr images." };
				}

				image_type = ImageDataType::Half;
				image_data_on_gpu = true;
				result.is_hdr = true;
			} else {
				uint8_t* img = stbi_load(path.str().c_str(), &res.x(), &res.y(), &comp, 4);

				fs::path alphapath = basepath / (std::string{frame["file_path"]} + ".alpha."s + path.extension());
				if (alphapath.exists()) {
					int wa=0,ha=0;
					uint8_t* alpha_img = stbi_load(alphapath.str().c_str(), &wa, &ha, &comp, 4);
					if (!alpha_img) {
						throw std::runtime_error{"Could not load alpha image "s + alphapath.str()};
					}
					ScopeGuard mem_guard{[&]() { stbi_image_free(alpha_img); }};
					if (wa != res.x() || ha != res.y()) {
						throw std::runtime_error{std::string{"Alpha image has wrong resolution: "} + alphapath.str()};
					}
					tlog::success() << "Alpha loaded from " << alphapath;
					for (int i=0;i<res.prod();++i) {
						img[i*4+3] = uint8_t(255.0f*srgb_to_linear(alpha_img[i*4]*(1.f/255.f))); // copy red channel of alpha to alpha.png to our alpha channel
					}
				}

				fs::path maskpath = path.parent_path()/(std::string{"dynamic_mask_"} + path.basename() + ".png");
				if (maskpath.exists()) {
					int wa=0,ha=0;
					uint8_t* mask_img = stbi_load(maskpath.str().c_str(), &wa, &ha, &comp, 4);
					if (!mask_img) {
						throw std::runtime_error{std::string{"Could not load mask image "} + maskpath.str()};
					}
					ScopeGuard mem_guard{[&]() { stbi_image_free(mask_img); }};
					if (wa != res.x() || ha != res.y()) {
						throw std::runtime_error{std::string{"Mask image has wrong resolution: "} + maskpath.str()};
					}
					mask_color = 0x00FF00FF; // HOT PINK
					for (int i = 0; i < res.prod(); ++i) {
						if (mask_img[i*4] != 0) {
							*(uint32_t*)&img[i*4] = mask_color;
						}
					}
				}

				images[i_img] = img;

				if (image_type != ImageDataType::None && image_type != ImageDataType::Byte) {
					throw std::runtime_error{ "May not mix png and exr images." };
				}
				image_type = ImageDataType::Byte;
			}
			if (!images[i_img]) {
				throw std::runtime_error{ "image not found: " + path.str() };
			}
			if (!result.image_resolution.isZero() && res != result.image_resolution) {
				throw std::runtime_error{ "training images are not all the same size" };
			}

			fs::path rayspath = path.parent_path()/(std::string{"rays_"} + path.basename() + ".dat");
			if (rayspath.exists()) {
				has_rays = true;
				uint32_t n_pixels = res.prod();
				rays[i_img] = (Ray*)malloc(n_pixels * sizeof(Ray));

				std::ifstream rays_file{rayspath.str(), std::ios::binary};
				rays_file.read((char*)rays[i_img], n_pixels * sizeof(Ray));

				std::streampos fsize = 0;
				fsize = rays_file.tellg();
				rays_file.seekg(0, std::ios::end);
				fsize = rays_file.tellg() - fsize;

				if (fsize > 0) {
					tlog::warning() << fsize << " bytes remaining in rays file " << rayspath;
				}

				for (uint32_t px = 0; px < n_pixels; ++px) {
					result.nerf_ray_to_ngp(rays[i_img][px]);
				}
			}

			nlohmann::json& jsonmatrix = frame["transform_matrix"];

			result.image_resolution = res;

			auto read_focal_length = [&](int resolution, const std::string& axis) {
				if (frame.contains(axis + "_fov")) {
					return fov_to_focal_length(resolution, (float)frame[axis + "_fov"]);
				} else if (json.contains("fl_"s + axis)) {
					return (float)json["fl_"s + axis];
				} else if (json.contains("camera_angle_"s + axis)) {
					return fov_to_focal_length(resolution, (float)json["camera_angle_"s + axis] * 180 / PI());
				} else {
					return 0.0f;
				}
			};

			// x_fov is in degrees, camera_angle_x in radians. Yes, it's silly.
			float x_fl = read_focal_length(result.image_resolution.x(), "x");
			float y_fl = read_focal_length(result.image_resolution.y(), "y");

			if (x_fl != 0) {
				result.focal_lengths[i_img] = Vector2f::Constant(x_fl);
				if (y_fl != 0) {
					result.focal_lengths[i_img].y() = y_fl;
				}
			} else if (y_fl != 0) {
				result.focal_lengths[i_img] = Vector2f::Constant(y_fl);
			} else {
				throw std::runtime_error{"Couldn't read fov."};
			}

			Matrix<float, 3, 4> xform;
			for (int m = 0; m < 3; ++m) {
				for (int n = 0; n < 4; ++n) {
					result.xforms[i_img](m, n) = float(jsonmatrix[m][n]);
				}
			}

			result.xforms[i_img] = result.nerf_matrix_to_ngp(result.xforms[i_img]);

			progress.update(++n_loaded);
		}, futures);

		image_idx += json["frames"].size();
	}

	waitAll(futures);

	tlog::success() << "Loaded " << images.size() << " images of size " << result.image_resolution.x() << "x" << result.image_resolution.y() << " after " << tlog::durationToString(progress.duration());
	tlog::info() << "  cam_aabb=" << cam_aabb;

	// concatenate all training images into a giant array
	size_t n_pixels = result.image_resolution.prod();
	size_t img_size = n_pixels * 4;

	assert(image_type != ImageDataType::None);

	// Copy loaded images to the GPU. If the type is Half, directly copy to the resulting buffer.
	// Otherwise, copy to a temporary buffer and cast on the GPU.
	size_t bytes_per_channel = image_type == ImageDataType::Byte ? 1 : (image_type == ImageDataType::Half ? 2 : 4);
	GPUMemory<uint8_t> images_data_gpu_tmp;

	if (image_type != ImageDataType::Half) {
		images_data_gpu_tmp.resize(img_size * images.size() * bytes_per_channel);
	}
	result.images_data.resize(img_size * images.size());

	if (has_rays) {
		tlog::success() << "Loaded per-pixel rays.";
		result.rays_data.resize(n_pixels * images.size());
	}

	uint8_t* dst = image_type == ImageDataType::Half ? (uint8_t*)result.images_data.data() : (uint8_t*)images_data_gpu_tmp.data();
	pool.parallelFor<size_t>(0, result.n_images, [&](size_t i) {
		CUDA_CHECK_THROW(cudaMemcpy(dst + img_size * i * bytes_per_channel, images[i], img_size * bytes_per_channel, image_data_on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
		if (image_data_on_gpu) {
			CUDA_CHECK_THROW(cudaFree(images[i]));
		} else {
			free(images[i]);
		}

		if (has_rays) {
			Ray* rays_dst = result.rays_data.data();
			CUDA_CHECK_THROW(cudaMemcpy(rays_dst + n_pixels * i, rays[i], n_pixels * sizeof(Ray), cudaMemcpyHostToDevice));
			free(rays[i]);
		}
	});

	if (image_type == ImageDataType::Byte) {
		linear_kernel(from_rgba32<__half>, 0, nullptr, n_pixels * result.n_images,
			(uint8_t*)images_data_gpu_tmp.data(), result.images_data.data(), white_transparent, black_transparent, mask_color
		);
	} else if (image_type == ImageDataType::Float) {
		linear_kernel(from_fullp, 0, nullptr, img_size * result.n_images,
			(float*)images_data_gpu_tmp.data(), result.images_data.data()
		);
	}
	if (sharpen_amount > 0.f) {
		tlog::info() << "sharpen=" << sharpen_amount;
		tcnn::GPUMemory<__half> images_data_2;
		images_data_2.resize(img_size * result.n_images);
		float center_w = 4.f + 1.f / sharpen_amount; // center_w ranges from 5 (strong sharpening) to infinite (no sharpening)
		linear_kernel(sharpen, 0, nullptr, n_pixels * result.n_images, result.image_resolution.x(),
			result.images_data.data(), images_data_2.data(), center_w, 1.f / (center_w - 4.f)
		);
		result.images_data.free_memory();
		result.images_data = std::move(images_data_2);
	}

	result.sharpness_resolution = { 128, 72 };
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)result.sharpness_resolution.x(), threads.x), div_round_up((uint32_t)result.sharpness_resolution.y(), threads.y), div_round_up((uint32_t)result.n_images, threads.z) };
	result.sharpness_data.enlarge( result.sharpness_resolution.x() * result.sharpness_resolution.y() *  result.n_images );
	compute_sharpness<<<blocks, threads, 0, nullptr>>>(result.sharpness_resolution, result.image_resolution, result.n_images, result.images_data.data(), result.sharpness_data.data());

	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	return result;
}

NGP_NAMESPACE_END
