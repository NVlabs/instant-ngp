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

// Conversion between glm and json
namespace glm {
	template <typename T>
	void to_json(nlohmann::json& j, const tmat3x3<T>& mat) {
		for (int row = 0; row < 3; ++row) {
			nlohmann::json column = nlohmann::json::array();
			for (int col = 0; col < 3; ++col) {
				column.push_back(mat[col][row]);
			}
			j.push_back(column);
		}
	}

	template <typename T>
	void from_json(const nlohmann::json& j, tmat3x3<T>& mat) {
		for (std::size_t row = 0; row < 3; ++row) {
			const auto& jrow = j.at(row);
			for (std::size_t col = 0; col < 3; ++col) {
				const auto& value = jrow.at(col);
				mat[col][row] = value.get<T>();
			}
		}
	}

	template <typename T>
	void to_json(nlohmann::json& j, const tmat4x3<T>& mat) {
		for (int row = 0; row < 3; ++row) {
			nlohmann::json column = nlohmann::json::array();
			for (int col = 0; col < 4; ++col) {
				column.push_back(mat[col][row]);
			}
			j.push_back(column);
		}
	}

	template <typename T>
	void from_json(const nlohmann::json& j, tmat4x3<T>& mat) {
		for (std::size_t row = 0; row < 3; ++row) {
			const auto& jrow = j.at(row);
			for (std::size_t col = 0; col < 4; ++col) {
				const auto& value = jrow.at(col);
				mat[col][row] = value.get<T>();
			}
		}
	}

	template <typename T>
	void to_json(nlohmann::json& j, const tmat4x4<T>& mat) {
		for (int row = 0; row < 4; ++row) {
			nlohmann::json column = nlohmann::json::array();
			for (int col = 0; col < 4; ++col) {
				column.push_back(mat[col][row]);
			}
			j.push_back(column);
		}
	}

	template <typename T>
	void from_json(const nlohmann::json& j, tmat4x4<T>& mat) {
		for (std::size_t row = 0; row < 4; ++row) {
			const auto& jrow = j.at(row);
			for (std::size_t col = 0; col < 4; ++col) {
				const auto& value = jrow.at(col);
				mat[col][row] = value.get<T>();
			}
		}
	}

	template <typename T>
	void to_json(nlohmann::json& j, const tvec2<T>& v) {
		j.push_back(v.x);
		j.push_back(v.y);
	}

	template <typename T>
	void from_json(const nlohmann::json& j, tvec2<T>& v) {
		v.x = j.at(0).get<T>();
		v.y = j.at(1).get<T>();
	}

	template <typename T>
	void to_json(nlohmann::json& j, const tvec3<T>& v) {
		j.push_back(v.x);
		j.push_back(v.y);
		j.push_back(v.z);
	}

	template <typename T>
	void from_json(const nlohmann::json& j, tvec3<T>& v) {
		v.x = j.at(0).get<T>();
		v.y = j.at(1).get<T>();
		v.z = j.at(2).get<T>();
	}

	template <typename T>
	void to_json(nlohmann::json& j, const tvec4<T>& v) {
		j.push_back(v.x);
		j.push_back(v.y);
		j.push_back(v.z);
		j.push_back(v.w);
	}

	template <typename T>
	void from_json(const nlohmann::json& j, tvec4<T>& v) {
		v.x = j.at(0).get<T>();
		v.y = j.at(1).get<T>();
		v.z = j.at(2).get<T>();
		v.w = j.at(3).get<T>();
	}

	template <typename T>
	void to_json(nlohmann::json& j, const tquat<T>& q) {
		j.push_back(q.x);
		j.push_back(q.y);
		j.push_back(q.z);
		j.push_back(q.w);
	}

	template <typename T>
	void from_json(const nlohmann::json& j, tquat<T>& q) {
		q.x = j.at(0).get<T>();
		q.y = j.at(1).get<T>();
		q.z = j.at(2).get<T>();
		q.w = j.at(3).get<T>();
	}
}

NGP_NAMESPACE_BEGIN

inline void to_json(nlohmann::json& j, const BoundingBox& box) {
	j["min"] = box.min;
	j["max"] = box.max;
}

inline void from_json(const nlohmann::json& j, BoundingBox& box) {
	box.min = j.at("min");
	box.max = j.at("max");
}

inline void to_json(nlohmann::json& j, const Lens& lens) {
	if (lens.mode == ELensMode::OpenCV) {
		j["is_fisheye"] = false;
		j["k1"] = lens.params[0];
		j["k2"] = lens.params[1];
		j["p1"] = lens.params[2];
		j["p2"] = lens.params[3];
	} else if (lens.mode == ELensMode::OpenCVFisheye) {
		j["is_fisheye"] = true;
		j["k1"] = lens.params[0];
		j["k2"] = lens.params[1];
		j["k3"] = lens.params[2];
		j["k4"] = lens.params[3];
	} else if (lens.mode == ELensMode::FTheta) {
		j["ftheta_p0"] = lens.params[0];
		j["ftheta_p1"] = lens.params[1];
		j["ftheta_p2"] = lens.params[2];
		j["ftheta_p3"] = lens.params[3];
		j["ftheta_p4"] = lens.params[4];
		j["w"] = lens.params[5];
		j["h"] = lens.params[6];
	} else if (lens.mode == ELensMode::LatLong) {
		j["latlong"] = true;
	} else if (lens.mode == ELensMode::Equirectangular) {
		j["equirectangular"] = true;
	}
}

inline void from_json(const nlohmann::json& j, Lens& lens) {
	if (j.contains("k1")) {
		if (j.value("is_fisheye", false)) {
			lens.mode = ELensMode::OpenCVFisheye;
			lens.params[0] = j.at("k1");
			lens.params[1] = j.at("k2");
			lens.params[2] = j.at("k3");
			lens.params[3] = j.at("k4");
		} else {
			lens.mode = ELensMode::OpenCV;
			lens.params[0] = j.at("k1");
			lens.params[1] = j.at("k2");
			lens.params[2] = j.at("p1");
			lens.params[3] = j.at("p2");
		}
	} else if (j.contains("ftheta_p0")) {
		lens.mode = ELensMode::FTheta;
		lens.params[0] = j.at("ftheta_p0");
		lens.params[1] = j.at("ftheta_p1");
		lens.params[2] = j.at("ftheta_p2");
		lens.params[3] = j.at("ftheta_p3");
		lens.params[4] = j.at("ftheta_p4");
		lens.params[5] = j.at("w");
		lens.params[6] = j.at("h");
	} else if (j.contains("latlong")) {
		lens.mode = ELensMode::LatLong;
	} else if (j.contains("equirectangular")) {
		lens.mode = ELensMode::Equirectangular;
	} else {
		lens.mode = ELensMode::Perspective;
	}
}

inline void from_json(const nlohmann::json& j, TrainingXForm& x) {
	x.start = j.at("start");
	x.end = j.at("end");
}

inline void to_json(nlohmann::json& j, const TrainingXForm& x) {
	j["start"] = x.start;
	j["end"] = x.end;
}

inline void to_json(nlohmann::json& j, const NerfDataset& dataset) {
	j["n_images"] = dataset.n_images;
	j["paths"] = dataset.paths;
	for (size_t i = 0; i < dataset.n_images; ++i) {
		j["metadata"].emplace_back();
		j["xforms"].emplace_back();
		j["metadata"].at(i)["focal_length"] = dataset.metadata[i].focal_length;
		j["metadata"].at(i)["lens"] = dataset.metadata[i].lens;
		j["metadata"].at(i)["principal_point"] = dataset.metadata[i].principal_point;
		j["metadata"].at(i)["rolling_shutter"] = dataset.metadata[i].rolling_shutter;
		j["metadata"].at(i)["resolution"] = dataset.metadata[i].resolution;
		j["xforms"].at(i) = dataset.xforms[i];
	}
	j["render_aabb"] = dataset.render_aabb;
	j["render_aabb_to_local"] = dataset.render_aabb_to_local;
	j["up"] = dataset.up;
	j["offset"] = dataset.offset;
	j["envmap_resolution"] = dataset.envmap_resolution;
	j["scale"] = dataset.scale;
	j["aabb_scale"] = dataset.aabb_scale;
	j["from_mitsuba"] = dataset.from_mitsuba;
	j["is_hdr"] = dataset.is_hdr;
	j["wants_importance_sampling"] = dataset.wants_importance_sampling;
	j["n_extra_learnable_dims"] = dataset.n_extra_learnable_dims;
}

inline void from_json(const nlohmann::json& j, NerfDataset& dataset) {
	dataset.n_images = j.at("n_images");
	dataset.metadata.resize(dataset.n_images);
	dataset.xforms.resize(dataset.n_images);
	dataset.paths.resize(dataset.n_images, "");
	if (j.contains("paths")) dataset.paths = j["paths"].get<std::vector<std::string>>();

	for (size_t i = 0; i < dataset.n_images; ++i) {
		// read global defaults first
		if (j.contains("lens")) dataset.metadata[i].lens = j.at("lens");
		// Legacy: "lens" used to be called "camera_distortion"
		if (j.contains("camera_distortion")) dataset.metadata[i].lens = j.at("camera_distortion");
		if (j.contains("principal_point")) dataset.metadata[i].principal_point = j.at("principal_point");
		if (j.contains("rolling_shutter")) dataset.metadata[i].rolling_shutter = j.at("rolling_shutter");
		if (j.contains("focal_length")) dataset.metadata[i].focal_length = j.at("focal_length");
		if (j.contains("image_resolution")) dataset.metadata[i].resolution = j.at("image_resolution");

		dataset.xforms[i] = j.at("xforms").at(i);
		if (j.contains("focal_lengths")) dataset.metadata[i].focal_length = j.at("focal_lengths").at(i);
		if (j.contains("metadata")) {
			auto &ji = j["metadata"].at(i);
			dataset.metadata[i].resolution = ji.at("resolution");
			dataset.metadata[i].focal_length = ji.at("focal_length");
			dataset.metadata[i].principal_point = ji.at("principal_point");
			if (ji.contains("lens")) dataset.metadata[i].lens = ji.at("lens");
			// Legacy: "lens" used to be called "camera_distortion"
			if (ji.contains("camera_distortion")) dataset.metadata[i].lens = ji.at("camera_distortion");
		}
	}

	dataset.render_aabb = j.at("render_aabb");
	dataset.render_aabb_to_local = mat3(1.0f);
	if (j.contains("render_aabb_to_local")) dataset.render_aabb_to_local = j.at("render_aabb_to_local");

	dataset.up = j.at("up");
	dataset.offset = j.at("offset");
	dataset.envmap_resolution = j.at("envmap_resolution");
	dataset.scale = j.at("scale");
	dataset.aabb_scale = j.at("aabb_scale");
	dataset.from_mitsuba = j.at("from_mitsuba");
	dataset.is_hdr = j.value("is_hdr", false);

	if (j.contains("wants_importance_sampling")) {
		dataset.wants_importance_sampling = j.at("wants_importance_sampling");
	} else {
		dataset.wants_importance_sampling = true;
	}

	dataset.n_extra_learnable_dims = j.value("n_extra_learnable_dims", 0);
}

NGP_NAMESPACE_END
