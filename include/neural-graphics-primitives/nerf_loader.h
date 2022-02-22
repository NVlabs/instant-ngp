/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_loader.h
 *  @author Alex Evans, NVIDIA
 *  @brief  Ability to load nerf datasets.
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>

#include <filesystem/path.h>

#include <vector>

NGP_NAMESPACE_BEGIN

struct TrainingImageMetadata {
	// Camera intrinsics and additional data associated with a NeRF training image
	CameraDistortion camera_distortion = {};
	Eigen::Vector2f principal_point = Eigen::Vector2f::Constant(0.5f);
	Eigen::Vector2f focal_length = Eigen::Vector2f::Constant(1000.f);

	// TODO: replace this with more generic float[] of task-specific metadata.
	Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.f);
};

struct NerfDataset {
	std::vector<TrainingImageMetadata> metadata;
	std::vector<Eigen::Matrix<float, 3, 4>> xforms;
	tcnn::GPUMemory<__half> images_data;
	tcnn::GPUMemory<float> sharpness_data;
	Eigen::Vector2i sharpness_resolution = {0, 0};
	tcnn::GPUMemory<float> envmap_data;
	BoundingBox render_aabb = {};
	Eigen::Vector3f up = {0.0f, 1.0f, 0.0f};
	Eigen::Vector3f offset = {0.0f, 0.0f, 0.0f};
	size_t n_images = 0;
	Eigen::Vector2i image_resolution = {0, 0};
	Eigen::Vector2i envmap_resolution = {0, 0};
	float scale = 1.0f;
	int aabb_scale = 1;
	bool from_mitsuba = false;
	bool is_hdr = false;
	bool wants_importance_sampling = true;

	// TODO: replace this with more generic `uint32_t n_extra_metadata_dims;`
	bool has_light_dirs = false;

	void set_training_image(int frame_idx, const float *pixels);

	tcnn::GPUMemory<Ray> rays_data;

	Eigen::Vector3f nerf_direction_to_ngp(const Eigen::Vector3f& nerf_dir) {
		Eigen::Vector3f result = nerf_dir;
		if (from_mitsuba) {
			result *= -1;
		} else {
			result=Eigen::Vector3f(result.y(), result.z(), result.x());
		}
		return result;
	}

	Eigen::Matrix<float, 3, 4> nerf_matrix_to_ngp(const Eigen::Matrix<float, 3, 4>& nerf_matrix) {
		Eigen::Matrix<float, 3, 4> result = nerf_matrix;
		result.col(1) *= -1;
		result.col(2) *= -1;
		result.col(3) = result.col(3) * scale + offset;

		if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz<-yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(1);
			result.row(1) = (Eigen::Vector4f)result.row(2);
			result.row(2) = tmp;
		}

		return result;
	}

	Eigen::Matrix<float, 3, 4> ngp_matrix_to_nerf(const Eigen::Matrix<float, 3, 4>& ngp_matrix) {
		Eigen::Matrix<float, 3, 4> result = ngp_matrix;
		if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz->yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(2);
			result.row(2) = (Eigen::Vector4f)result.row(1);
			result.row(1) = tmp;
		}
		result.col(1) *= -1;
		result.col(2) *= -1;
		result.col(3) = (result.col(3) - offset) / scale;
		return result;
	}

	void nerf_ray_to_ngp(Ray& ray) {
		ray.o = ray.o * scale + offset;

		float tmp = ray.o[0];
		ray.o[0] = ray.o[1];
		ray.o[1] = ray.o[2];
		ray.o[2] = tmp;

		tmp = ray.d[0];
		ray.d[0] = ray.d[1];
		ray.d[1] = ray.d[2];
		ray.d[2] = tmp;
	}
};

NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount = 0.f);
NerfDataset create_empty_nerf_dataset(size_t n_images, Eigen::Vector2i image_resolution, int aabb_scale = 1, bool is_hdr = false);

NGP_NAMESPACE_END
