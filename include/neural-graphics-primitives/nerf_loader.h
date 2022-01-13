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

struct NerfDataset {
	std::vector<Eigen::Vector2f> focal_lengths;
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
	CameraDistortion camera_distortion = {};
	Eigen::Vector2f principal_point = Eigen::Vector2f::Constant(0.5f);
	bool from_mitsuba = false;
	bool is_hdr = false;
	bool wants_importance_sampling = true;

	tcnn::GPUMemory<Ray> rays_data;

	auto nerf_matrix_to_ngp(const Eigen::Matrix<float, 3, 4>& nerf_matrix) {
		Eigen::Matrix<float, 3, 4> result;
		int X=0,Y=1,Z=2;
		result.col(0) = Eigen::Vector3f{ nerf_matrix(X,0),  nerf_matrix(Y,0),  nerf_matrix(Z,0)};
		result.col(1) = Eigen::Vector3f{-nerf_matrix(X,1), -nerf_matrix(Y,1), -nerf_matrix(Z,1)};
		result.col(2) = Eigen::Vector3f{-nerf_matrix(X,2), -nerf_matrix(Y,2), -nerf_matrix(Z,2)};
		result.col(3) = Eigen::Vector3f{ nerf_matrix(X,3),  nerf_matrix(Y,3),  nerf_matrix(Z,3)} * scale + offset;

		if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz->yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(1);
			result.row(1) = (Eigen::Vector4f)result.row(2);
			result.row(2) = tmp;
		}

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

NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount=0.f);

NGP_NAMESPACE_END
