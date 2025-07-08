/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   camera_path.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common_host.h>

#include <tiny-cuda-nn/common.h>

#ifdef NGP_GUI
#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>
#endif

#include <chrono>
#include <vector>

struct ImDrawList;

namespace ngp {

struct CameraKeyframe {
	quat R;
	vec3 T;
	float slice;
	float scale; // not a scale factor as in scaling the world, but the value of m_scale (setting the focal plane along with slice)
	float fov;
	float aperture_size;

	float timestamp = 0;

	mat4x3 m() const {
		auto rot = to_mat3(normalize(R));
		return mat4x3(rot[0], rot[1], rot[2], T);
	}

	void from_m(const mat4x3& rv) {
		T = rv[3];
		R = quat(mat3(rv));
	}

	CameraKeyframe() = default;
	CameraKeyframe(const quat& r, const vec3& t, float sl, float sc, float fv, float df, float time) : R(r), T(t), slice(sl), scale(sc), fov(fv), aperture_size(df), timestamp{time} {}
	CameraKeyframe(mat4x3 m, float sl, float sc, float fv, float df, float time) : slice(sl), scale(sc), fov(fv), aperture_size(df), timestamp(time) { from_m(m); }
	CameraKeyframe operator*(float f) const { return {R*f, T*f, slice*f, scale*f, fov*f, aperture_size*f, timestamp}; }
	CameraKeyframe operator+(const CameraKeyframe& rhs) const {
		quat Rr = rhs.R;
		if (dot(Rr, R) < 0.0f) Rr = -Rr;
		return {R+Rr, T+rhs.T, slice+rhs.slice, scale+rhs.scale, fov+rhs.fov, aperture_size+rhs.aperture_size, rhs.timestamp};
	}

	bool same_pos_as(const CameraKeyframe& rhs) const {
		return distance(T, rhs.T) < 0.0001f && fabsf(dot(R, rhs.R)) >= 0.999f;
	}
};

CameraKeyframe lerp(const CameraKeyframe& p0, const CameraKeyframe& p1, float t, float t0, float t1);
CameraKeyframe spline_cm(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2, const CameraKeyframe& p3);
CameraKeyframe spline_cubic(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2, const CameraKeyframe& p3);
CameraKeyframe spline_quadratic(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2);
CameraKeyframe spline_linear(float t, const CameraKeyframe& p0, const CameraKeyframe& p1);

enum class EEditingKernel : int {
	None,
	Gaussian,
	Quartic,
	Hat,
	Box,
};
static constexpr const char* EditingKernelStr = "None\0Gaussian\0Quartic\0Hat\0Box\0\0";

struct CameraPath {
	std::vector<CameraKeyframe> keyframes;
	bool update_cam_from_path = false;
	float play_time = 0.f;
	float auto_play_speed = 0.f;
	// If loop is set true, the last frame set will be more like "next to last,"
	// with animation then returning back to the first frame, making a continuous loop.
	// Note that the user does not have to (and should not normally) duplicate the first frame to be the last frame.
	bool loop = false;

	int keyframe_subsampling = 1;
	float duration_seconds = 0.0f;

	float camera_prediction_duration_seconds = 0.0f;
	float camera_prediction_dt = 0.005f;

	EEditingKernel editing_kernel_type = EEditingKernel::None;
	float editing_kernel_radius = 0.1f;

	// Cubic spline per default. Order 1 (p/w linear) is also supported.
	int spline_order = 3;

	struct RenderSettings {
		ivec2 resolution = {1920, 1080};
		int spp = 8;
		float fps = 60.0f;
		float duration_seconds = 5.0f;
		float shutter_fraction = 0.5f;
		int quality = 10;

		uint32_t n_frames() const {
			return (uint32_t)((double)duration_seconds * fps);
		}

		float frame_seconds() const {
			return 1.0f / (duration_seconds * fps);
		}

		float frame_milliseconds() const {
			return 1000.0f / (duration_seconds * fps);
		}

		std::string filename = "video.mp4";
	};

	RenderSettings render_settings;
	bool rendering = false;
	uint32_t render_frame_idx = 0;
	std::chrono::time_point<std::chrono::steady_clock> render_start_time;

	mat4x3 render_frame_end_camera;

	struct Pos {
		int kfidx;
		float t;
	};

	void clear() {
		keyframes.clear();
		play_time = 0.0f;
		duration_seconds = 0.0f;
	}

	bool empty() const {
		return keyframes.empty();
	}

	bool has_valid_timestamps() const;

	void make_keyframe_timestamps_equidistant();

	void sanitize_keyframes();

	Pos get_pos(float playtime);

	float get_playtime(int i) {
		if (i <= 0 || keyframes.size() < 2) {
			return 0.0f;
		}

		const auto& kf = keyframes[clamp(i-1, 0, (int)keyframes.size() - 1)];
		const float duration = loop ? keyframes.back().timestamp : keyframes[keyframes.size() - 2].timestamp;
		return kf.timestamp / duration;
	}

	const CameraKeyframe& get_keyframe(int i) const {
		if (loop) {
			int size = (int)keyframes.size();
			// add size to ensure no negative value is generated by modulo
			return keyframes[(i + size) % size];
		} else {
			return keyframes[clamp(i, 0, (int)keyframes.size()-1)];
		}
	}

	CameraKeyframe eval_camera_path(float t) {
		if (keyframes.empty())
			return {};

		auto p = get_pos(t);
		switch (spline_order) {
			case 0:
				return get_keyframe(p.kfidx + (int)round(p.t));
			case 1:
				return spline_linear(p.t, get_keyframe(p.kfidx), get_keyframe(p.kfidx+1));
			case 2:
				return spline_quadratic(p.t, get_keyframe(p.kfidx-1), get_keyframe(p.kfidx), get_keyframe(p.kfidx+1));
			case 3:
				return spline_cubic(p.t, get_keyframe(p.kfidx-1), get_keyframe(p.kfidx), get_keyframe(p.kfidx+1), get_keyframe(p.kfidx+2));
			default: throw std::runtime_error{fmt::format("Spline of order {} is not supported.", spline_order)};
		}
	}

	void save(const fs::path& path);
	void load(const fs::path& path, const mat4x3 &first_xform);

	void add_camera(const mat4x3& camera, float slice_plane_z, float scale, float fov, float aperture_size, float bounding_radius, float timestamp);

#ifdef NGP_GUI
	ImGuizmo::MODE m_gizmo_mode = ImGuizmo::LOCAL;
	ImGuizmo::OPERATION m_gizmo_op = ImGuizmo::TRANSLATE;
	int imgui(char path_filename_buf[1024], float frame_milliseconds, const mat4x3& camera, float slice_plane_z, float scale, float fov, float aperture_size, float bounding_radius, const mat4x3& first_xform);
	bool imgui_viz(ImDrawList* list, mat4& view2proj, mat4& world2proj, mat4& world2view, vec2 focal, float aspect, float znear, float zfar, CameraPredictor* cam_predictor);
#endif
};

#ifdef NGP_GUI
void add_debug_line(ImDrawList* list, const mat4&proj, vec3 a, vec3 b, uint32_t col = 0xffffffff, float thickness = 1.0f);
void visualize_cube(ImDrawList* list, const mat4& world2proj, const vec3& a, const vec3& b, const mat3& render_aabb_to_local);
void visualize_camera(ImDrawList* list, const mat4& world2proj, const mat4x3& xform, float aspect, uint32_t col = 0x80ffffff, float thickness = 1.0f);
#endif

}
