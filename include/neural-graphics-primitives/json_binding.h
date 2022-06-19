/*
* Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/** @file   json_binding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Conversion between eigen
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf_loader.h>

#include <json/json.hpp>

NGP_NAMESPACE_BEGIN

// Conversion between eigen and json
template <typename Derived>
void to_json(nlohmann::json& j, const Eigen::MatrixBase<Derived>& mat) {
	for (int row = 0; row < mat.rows(); ++row) {
		if (mat.cols() == 1) {
			j.push_back(mat(row));
		} else {
			nlohmann::json column = nlohmann::json::array();
			for (int col = 0; col < mat.cols(); ++col) {
				column.push_back(mat(row, col));
			}
			j.push_back(column);
		}
	}
}

template <typename Derived>
void from_json(const nlohmann::json& j, Eigen::MatrixBase<Derived>& mat) {
	for (std::size_t row = 0; row < j.size(); ++row) {
		const auto& jrow = j.at(row);
		if (jrow.is_array()) {
			for (std::size_t col = 0; col < jrow.size(); ++col) {
				const auto& value = jrow.at(col);
				mat(row, col) = value.get<typename Eigen::MatrixBase<Derived>::Scalar>();
			}
		} else {
			mat(row) = jrow.get<typename Eigen::MatrixBase<Derived>::Scalar>();
		}
	}
}

template <typename Derived>
void to_json(nlohmann::json& j, const Eigen::QuaternionBase<Derived>& q) {
	j.push_back(q.w());
	j.push_back(q.x());
	j.push_back(q.y());
	j.push_back(q.z());
}

template <typename Derived>
void from_json(const nlohmann::json& j, Eigen::QuaternionBase<Derived>& q) {
	using Scalar = typename Eigen::QuaternionBase<Derived>::Scalar;
	q.w() = j.at(0).get<Scalar>();
	q.x() = j.at(1).get<Scalar>();
	q.y() = j.at(2).get<Scalar>();
	q.z() = j.at(3).get<Scalar>();
}

inline void to_json(nlohmann::json& j, const BoundingBox& box) {
	to_json(j["min"], box.min);
	to_json(j["max"], box.max);
}

inline void from_json(const nlohmann::json& j, BoundingBox& box) {
	from_json(j.at("min"), box.min);
	from_json(j.at("max"), box.max);
}

inline void to_json(nlohmann::json& j, const CameraDistortion& dist) {
	if (dist.mode == ECameraDistortionMode::Iterative) {
		j["k1"] = dist.params[0];
		j["k2"] = dist.params[1];
		j["p1"] = dist.params[2];
		j["p2"] = dist.params[3];
	} else if (dist.mode == ECameraDistortionMode::FTheta) {
		j["ftheta_p0"] = dist.params[0];
		j["ftheta_p1"] = dist.params[1];
		j["ftheta_p2"] = dist.params[2];
		j["ftheta_p3"] = dist.params[3];
		j["ftheta_p4"] = dist.params[4];
		j["w"] = dist.params[5];
		j["h"] = dist.params[6];
	}
}

inline void from_json(const nlohmann::json& j, CameraDistortion& dist) {
	if (j.contains("k1")) {
		dist.mode = ECameraDistortionMode::Iterative;
		dist.params[0] = j.at("k1");
		dist.params[1] = j.at("k2");
		dist.params[2] = j.at("p1");
		dist.params[3] = j.at("p2");
	} else if (j.contains("ftheta_p0")) {
		dist.mode = ECameraDistortionMode::FTheta;
		dist.params[0] = j.at("ftheta_p0");
		dist.params[1] = j.at("ftheta_p1");
		dist.params[2] = j.at("ftheta_p2");
		dist.params[3] = j.at("ftheta_p3");
		dist.params[4] = j.at("ftheta_p4");
		dist.params[5] = j.at("w");
		dist.params[6] = j.at("h");
	} else {
		dist.mode = ECameraDistortionMode::None;
	}
}

inline void from_json(const nlohmann::json& j, TrainingXForm& x) {
	from_json(j.at("start"), x.start);
	from_json(j.at("end"), x.end);
}

inline void to_json(nlohmann::json& j, const TrainingXForm& x) {
	to_json(j["start"], x.start);
	to_json(j["end"], x.end);
}

inline void to_json(nlohmann::json& j, const NerfDataset& dataset) {
	j["n_images"] = dataset.n_images;
	for (size_t i = 0; i < dataset.n_images; ++i) {
		j["metadata"].emplace_back();
		j["xforms"].emplace_back();
		to_json(j["metadata"].at(i)["focal_length"], dataset.metadata[i].focal_length);
		to_json(j["metadata"].at(i)["camera_distortion"], dataset.metadata[i].camera_distortion);
		to_json(j["metadata"].at(i)["principal_point"], dataset.metadata[i].principal_point);
		to_json(j["metadata"].at(i)["rolling_shutter"], dataset.metadata[i].rolling_shutter);
		to_json(j["metadata"].at(i)["resolution"], dataset.metadata[i].resolution);
		to_json(j["xforms"].at(i), dataset.xforms[i]);
	}
	j["render_aabb"] = dataset.render_aabb;
	to_json(j["up"], dataset.up);
	to_json(j["offset"], dataset.offset);
	// global resolution removed
	//to_json(j["image_resolution"], dataset.image_resolution);
	to_json(j["envmap_resolution"], dataset.envmap_resolution);
	j["scale"] = dataset.scale;
	j["aabb_scale"] = dataset.aabb_scale;
	// global distortion removed
	//j["camera_distortion"] = dataset.camera_distortion;
	//to_json(j["principal_point"], dataset.principal_point);
	//to_json(j["rolling_shutter"], dataset.rolling_shutter);
	j["from_mitsuba"] = dataset.from_mitsuba;
	j["is_hdr"] = dataset.is_hdr;
	j["wants_importance_sampling"] = dataset.wants_importance_sampling;
}

inline void from_json(const nlohmann::json& j, NerfDataset& dataset) {
	dataset.n_images = j.at("n_images");
	dataset.metadata.resize(dataset.n_images);
	dataset.xforms.resize(dataset.n_images);

	for (size_t i = 0; i < dataset.n_images; ++i) {
		// read global defaults first
		if (j.contains("camera_distortion")) from_json(j.at("camera_distortion"), dataset.metadata[i].camera_distortion);
		if (j.contains("principal_point")) from_json(j.at("principal_point"), dataset.metadata[i].principal_point);
		if (j.contains("rolling_shutter")) from_json(j.at("rolling_shutter"), dataset.metadata[i].rolling_shutter);
		if (j.contains("focal_length")) from_json(j.at("focal_length"), dataset.metadata[i].focal_length);
		if (j.contains("image_resolution")) from_json(j.at("image_resolution"), dataset.metadata[i].resolution);

		from_json(j.at("xforms").at(i), dataset.xforms[i]);
		if (j.contains("focal_lengths"))
			from_json(j.at("focal_lengths").at(i), dataset.metadata[i].focal_length);
		if (j.contains("metadata")) {
			auto &ji = j["metadata"].at(i);
			from_json(ji.at("resolution"),dataset.metadata[i].resolution);
			from_json(ji.at("focal_length"),dataset.metadata[i].focal_length);
			from_json(ji.at("principal_point"),dataset.metadata[i].principal_point);
			from_json(ji.at("rolling_shutter"),dataset.metadata[i].rolling_shutter);
			from_json(ji.at("camera_distortion"),dataset.metadata[i].camera_distortion);
		}
	}
	dataset.render_aabb = j.at("render_aabb");
	from_json(j.at("up"), dataset.up);
	from_json(j.at("offset"), dataset.offset);
	// global resolution removed
	//from_json(j.at("image_resolution"), dataset.image_resolution);
	from_json(j.at("envmap_resolution"), dataset.envmap_resolution);
	dataset.scale = j.at("scale");
	dataset.aabb_scale = j.at("aabb_scale");
	dataset.from_mitsuba = j.at("from_mitsuba");
	dataset.is_hdr = j.value("is_hdr", false);
	if (j.contains("wants_importance_sampling")) {
		dataset.wants_importance_sampling = j.at("wants_importance_sampling");
	} else {
		dataset.wants_importance_sampling = true;
	}
}

NGP_NAMESPACE_END
