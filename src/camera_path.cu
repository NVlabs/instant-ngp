/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   camera_path.cpp
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/camera_path.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/json_binding.h>

#ifdef NGP_GUI
#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>
#endif

#include <json/json.hpp>
#include <fstream>

using namespace nlohmann;

namespace ngp {

CameraKeyframe lerp(const CameraKeyframe& p0, const CameraKeyframe& p1, float t, float t0, float t1) {
	t = (t - t0) / (t1 - t0);
	quat R1 = p1.R;

	// take the short path
	if (dot(R1, p0.R) < 0.0f)  {
		R1 = -R1;
	}

	return {
		normalize(slerp(p0.R, R1, t)),
		p0.T + (p1.T - p0.T) * t,
		p0.slice + (p1.slice - p0.slice) * t,
		p0.scale + (p1.scale - p0.scale) * t,
		p0.fov + (p1.fov - p0.fov) * t,
		p0.aperture_size + (p1.aperture_size - p0.aperture_size) * t,
		p0.timestamp + (p1.timestamp - p0.timestamp) * t,
	};
}

CameraKeyframe normalize(const CameraKeyframe& p0) {
	CameraKeyframe result = p0;
	result.R = normalize(result.R);
	return result;
}

CameraKeyframe spline_cm(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2, const CameraKeyframe& p3) {
	CameraKeyframe q0 = lerp(p0, p1, t, -1.f, 0.f);
	CameraKeyframe q1 = lerp(p1, p2, t,  0.f, 1.f);
	CameraKeyframe q2 = lerp(p2, p3, t,  1.f, 2.f);
	CameraKeyframe r0 = lerp(q0, q1, t, -1.f, 1.f);
	CameraKeyframe r1 = lerp(q1, q2, t,  0.f, 2.f);
	return lerp(r0, r1, t, 0.f, 1.f);
}

CameraKeyframe spline_cubic(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2, const CameraKeyframe& p3) {
	float tt = t*t;
	float ttt = t*t*t;
	float a = (1-t)*(1-t)*(1-t)*(1.f/6.f);
	float b = (3.f*ttt-6.f*tt+4.f)*(1.f/6.f);
	float c = (-3.f*ttt+3.f*tt+3.f*t+1.f)*(1.f/6.f);
	float d = ttt*(1.f/6.f);
	return normalize(p0 * a + p1 * b + p2 * c + p3 * d);
}

CameraKeyframe spline_quadratic(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2) {
	float tt = t*t;
	float a = (1-t)*(1-t)*0.5f;
	float b = (-2.f*tt + 2.f*t + 1.f)*0.5f;
	float c = tt*0.5f;
	return normalize(p0 * a + p1 * b + p2 * c);
}

CameraKeyframe spline_linear(float t, const CameraKeyframe& p0, const CameraKeyframe& p1) {
	return normalize(p0 * (1.0f - t) + p1 * t);
}

void to_json(json& j, const CameraKeyframe& p) {
	j = json{
		{"R", p.R},
		{"T", p.T},
		{"slice", p.slice},
		{"scale", p.scale},
		{"fov", p.fov},
		{"aperture_size", p.aperture_size},
		{"timestamp", p.timestamp},
	};
}

bool load_relative_to_first=false; // set to true when using a camera path that is aligned with the first training image, such that it is invariant to changes in the space of the training data

void from_json(bool is_first, const json& j, CameraKeyframe& p, const CameraKeyframe& first, const mat4x3& ref) {
	 if (is_first && load_relative_to_first) {
		p.from_m(ref);
	 } else {
		p.R = j.at("R");
		p.T = j.at("T");

		if (load_relative_to_first) {
			mat4 ref4 = {ref};
			mat4 first4 = {first.m()};
			mat4 p4 = {p.m()};
			p.from_m(mat4x3(ref4 * inverse(first4) * p4));
		}
	}
	j.at("slice").get_to(p.slice);
	j.at("scale").get_to(p.scale);
	j.at("fov").get_to(p.fov);
	if (j.contains("dof")) j.at("dof").get_to(p.aperture_size); else j.at("aperture_size").get_to(p.aperture_size);
	if (j.contains("timestamp")) j.at("timestamp").get_to(p.timestamp); else p.timestamp = 0.f;
}

void CameraPath::save(const fs::path& path) {
	json j = {
		{"loop", loop},
		{"time", play_time},
		{"path", keyframes},
		{"duration_seconds", duration_seconds},
		{"spline_order", spline_order},
	};
	std::ofstream f(native_string(path));
	f << j;
}

void CameraPath::load(const fs::path& path, const mat4x3& first_xform) {
	std::ifstream f{native_string(path)};
	if (!f) {
		throw std::runtime_error{fmt::format("Camera path {} does not exist.", path.str())};
	}

	json j;
	f >> j;

	CameraKeyframe first;

	keyframes.clear();
	if (j.contains("loop")) loop = j["loop"];
	if (j.contains("time")) play_time = j["time"];
	if (j.contains("path")) for (auto& el : j["path"]) {
		CameraKeyframe p;
		bool is_first = keyframes.empty();
		from_json(is_first, el, p, first, first_xform);
		if (is_first) {
			first = p;
		}
		keyframes.push_back(p);
	}

	duration_seconds = j.value("duration_seconds", 0.0f);
	spline_order = j.value("spline_order", 3);
	sanitize_keyframes();

	play_time = 0.0f;

	if (keyframes.size() >= 16) {
		keyframe_subsampling = keyframes.size() - 1;
		editing_kernel_type = EEditingKernel::Gaussian;
	}
}

void CameraPath::add_camera(
	const mat4x3& camera,
	float slice_plane_z,
	float scale,
	float fov,
	float aperture_size,
	float bounding_radius,
	float timestamp
) {
	int n = std::max(0, int(keyframes.size()) - 1);
	int i = (int)ceil(play_time * (float)n + 0.001f);
	if (i > keyframes.size()) { i = keyframes.size(); }
	if (i < 0) i = 0;
	keyframes.insert(keyframes.begin() + i, CameraKeyframe(camera, slice_plane_z, scale, fov, aperture_size, timestamp));
	update_cam_from_path = false;
	play_time = get_playtime(i);

	sanitize_keyframes();
}

float editing_kernel(float x, EEditingKernel kernel) {
	x = kernel == EEditingKernel::Gaussian ? x : clamp(x, -1.0f, 1.0f);
	switch (kernel) {
		case EEditingKernel::Gaussian: return expf(-2.0f * x * x);
		case EEditingKernel::Quartic: return (1.0f - x*x) * (1.0f - x*x);
		case EEditingKernel::Hat: return 1.0f - fabsf(x);
		case EEditingKernel::Box: return x > -1.0f && x < 1.0f ? 1.0f : 0.0f;
		case EEditingKernel::None: return fabs(x) < 0.0001f ? 1.0f : 0.0f;
		default: throw std::runtime_error{"Unknown editing kernel"};
	}
}

bool CameraPath::has_valid_timestamps() const {
	float prev_timestamp = 0.0f;
	for (size_t i = 0; i < keyframes.size(); ++i) {
		if (!(keyframes[i].timestamp > prev_timestamp)) {
			return false;
		}

		prev_timestamp = keyframes[i].timestamp;
	}

	return true;
}

void CameraPath::make_keyframe_timestamps_equidistant() {
	for (size_t i = 0; i < keyframes.size(); ++i) {
		keyframes[i].timestamp = (i+1) / (float)keyframes.size();
	}

	duration_seconds = 1.0f;
}

void CameraPath::sanitize_keyframes() {
	if (has_valid_timestamps()) {
		return;
	}

	// Timestamps are invalid. Best effort is to equally space all frames.
	make_keyframe_timestamps_equidistant();
}

CameraPath::Pos CameraPath::get_pos(float playtime) {
	if (keyframes.empty()) {
		return {-1, 0.0f};
	} else if (keyframes.size() == 1) {
		return {0, playtime};
	}

	float duration = loop ? keyframes.back().timestamp : keyframes[keyframes.size() - 2].timestamp;
	playtime *= duration;

	CameraKeyframe dummy;
	dummy.timestamp = playtime;

	// Binary search to obtain relevant keyframe in O(log(n_keyframes)) time
	auto it = std::upper_bound(keyframes.begin(), keyframes.end(), dummy, [](const auto& a, const auto& b) {
		return a.timestamp < b.timestamp;
	});

	int i = clamp((int)std::distance(keyframes.begin(), it), 0, (int)keyframes.size() - (loop ? 1 : 2));
	float prev_timestamp = i == 0 ? 0.0f : keyframes[i - 1].timestamp;

	return {
		i,
		(playtime - prev_timestamp) / (keyframes[i].timestamp - prev_timestamp),
	};
}

#ifdef NGP_GUI
int CameraPath::imgui(char path_filename_buf[1024], float frame_milliseconds, const mat4x3& camera, float slice_plane_z, float scale, float fov, float aperture_size, float bounding_radius, const mat4x3& first_xform) {
	int n = std::max(0, int(keyframes.size()) - 1);
	int read = 0; // 1=smooth, 2=hard

	ImGui::InputText("##PathFile", path_filename_buf, 1024);
	ImGui::SameLine();
	static std::string camera_path_load_error_string = "";

	if (rendering) { ImGui::BeginDisabled(); }

	if (ImGui::Button("Load")) {
		try {
			load(path_filename_buf, first_xform);
		} catch (const std::exception& e) {
			ImGui::OpenPopup("Camera path load error");
			camera_path_load_error_string = std::string{"Failed to load camera path: "} + e.what();
		}
	}

	if (rendering) { ImGui::EndDisabled(); }

	if (ImGui::BeginPopupModal("Camera path load error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("%s", camera_path_load_error_string.c_str());
		if (ImGui::Button("OK", ImVec2(120, 0))) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	if (!keyframes.empty()) {
		ImGui::SameLine();
		if (ImGui::Button("Save")) {
			save(path_filename_buf);
		}
	}

	if (rendering) { ImGui::BeginDisabled(); }

	if (ImGui::Button("Add from cam")) {
		add_camera(camera, slice_plane_z, scale, fov, aperture_size, bounding_radius, 0.0f);
		make_keyframe_timestamps_equidistant();
		read = 2;
	}

	auto p = get_pos(play_time);

	if (!keyframes.empty()) {
		ImGui::SameLine();
		if (ImGui::Button("Split")) {
			update_cam_from_path = false;
			int i = clamp(p.kfidx + 1, 0, (int)keyframes.size());
			keyframes.insert(keyframes.begin() + i, eval_camera_path(play_time));
			make_keyframe_timestamps_equidistant();
			play_time = get_playtime(i);
			read = 2;
		}
		ImGui::SameLine();

		int i = p.kfidx;
		if (!loop) {
			i += (int)round(p.t);
		}

		if (ImGui::Button("|<")) { play_time = 0.f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button("<")) { play_time = n ? (get_playtime(i-1)+0.0001f) : 0.f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button(update_cam_from_path ? "Stop" : "Read")) { update_cam_from_path = !update_cam_from_path; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button(">")) { play_time = n ? (get_playtime(i+1)+0.0001f) : 1.0f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button(">|")) { play_time = 1.0f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button("Dup")) {
			update_cam_from_path = false;
			keyframes.insert(keyframes.begin() + i, keyframes[i]);
			make_keyframe_timestamps_equidistant();
			play_time = get_playtime(i);
			read = 2;
		}
		ImGui::SameLine();
		if (ImGui::Button("Del")) {
			update_cam_from_path = false;
			keyframes.erase(keyframes.begin() + i);
			make_keyframe_timestamps_equidistant();
			play_time = get_playtime(i-1);
			read = 2;
		}
		ImGui::SameLine();
		if (ImGui::Button("Set")) {
			keyframes[i] = CameraKeyframe(
				camera,
				slice_plane_z,
				scale,
				fov,
				aperture_size,
				0.0f
			);
			read = 2;
			if (n) play_time = get_playtime(i);
		}

		ImGui::Combo("Editing kernel", (int*)&editing_kernel_type, EditingKernelStr);
		ImGui::SliderFloat("Editing kernel radius", &editing_kernel_radius, 0.001f, 10.0f, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);

		if (ImGui::RadioButton("Translate", m_gizmo_op == ImGuizmo::TRANSLATE)) { m_gizmo_op = ImGuizmo::TRANSLATE; }
		ImGui::SameLine();
		if (ImGui::RadioButton("Rotate", m_gizmo_op == ImGuizmo::ROTATE)) { m_gizmo_op = ImGuizmo::ROTATE; }
		ImGui::SameLine();
		if (ImGui::RadioButton("Local", m_gizmo_mode == ImGuizmo::LOCAL)) { m_gizmo_mode = ImGuizmo::LOCAL; }
		ImGui::SameLine();
		if (ImGui::RadioButton("World", m_gizmo_mode == ImGuizmo::WORLD)) { m_gizmo_mode = ImGuizmo::WORLD; }
		ImGui::SameLine();
		ImGui::Checkbox("Loop path", &loop);

		ImGui::SliderInt("Spline order", &spline_order, 0, 3);

		if (ImGui::SliderFloat("Camera path time", &play_time, 0.0f, 1.0f)) { read = 1; }

		if (ImGui::Button("Start") && !keyframes.empty()) { auto_play_speed = 0.0f; play_time = 0.0f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button("Rev") && !keyframes.empty()) { auto_play_speed = -1.0f / keyframes.back().timestamp; }
		ImGui::SameLine();
		if (ImGui::Button(auto_play_speed != 0 ? "Pause" : "Play") && !keyframes.empty()) {
			auto_play_speed = auto_play_speed == 0.0f ? (1.0f / keyframes.back().timestamp) : 0.0f;
		}
		ImGui::SameLine();
		if (ImGui::Button("End") && !keyframes.empty()) { auto_play_speed = 0.0f; play_time = 1.0f; read = 2; }

		ImGui::SliderFloat("Auto play speed", &auto_play_speed, -1.0f, 1.0f);
		if (auto_play_speed != 0.0f) {
			float prev = play_time;
			play_time = clamp(play_time + auto_play_speed * (frame_milliseconds / 1000.f), 0.0f, 1.0f);

			if (play_time != prev) {
				read = 1;
			}
		}

		ImGui::SliderInt("Keyframe subsampling", &keyframe_subsampling, 1, max((int)keyframes.size() - 1, 1));

		ImGui::Text("Current keyframe %d/%d:", i, n+1);
		ImGui::SameLine();
		if (ImGui::Button("Apply to all")) {
			for (auto& k : keyframes) {
				k.fov = keyframes[i].fov;
				k.aperture_size = keyframes[i].aperture_size;
				k.slice = keyframes[i].slice;
				k.scale = keyframes[i].scale;
			}
		}

		if (ImGui::SliderFloat("Field of view", &keyframes[i].fov, 0.0f, 120.0f)) read = 2;
		if (ImGui::SliderFloat("Aperture size", &keyframes[i].aperture_size, 0.0f, 0.1f)) read = 2;
		if (ImGui::SliderFloat("Slice Z", &keyframes[i].slice, -bounding_radius, bounding_radius)) read = 2;
		if (ImGui::SliderFloat("Scale", &keyframes[i].scale, 0.f,10.f)) read = 2;
	}

	if (rendering) { ImGui::EndDisabled(); }

	return keyframes.empty() ? 0 : read;
}

bool debug_project(const mat4& proj, vec3 p, ImVec2& o) {
	vec4 ph{p.x, p.y, p.z, 1.0f};
	vec4 pa = proj * ph;
	if (pa.w <= 0.f) {
		return false;
	}

	o.x = pa.x / pa.w;
	o.y = pa.y / pa.w;
	return true;
}

void add_debug_line(ImDrawList* list, const mat4& proj, vec3 a, vec3 b, uint32_t col, float thickness) {
	ImVec2 aa, bb;
	if (debug_project(proj, a, aa) && debug_project(proj, b, bb)) {
		list->AddLine(aa, bb, col, thickness * 2.0f);
	}
}

void visualize_cube(ImDrawList* list, const mat4& world2proj, const vec3& a, const vec3& b, const mat3& render_aabb_to_local) {
	mat3 m = transpose(render_aabb_to_local);
	add_debug_line(list, world2proj, m * vec3{a.x, a.y, a.z}, m * vec3{a.x, a.y, b.z}, 0xffff4040); // Z
	add_debug_line(list, world2proj, m * vec3{b.x, a.y, a.z}, m * vec3{b.x, a.y, b.z}, 0xffffffff);
	add_debug_line(list, world2proj, m * vec3{a.x, b.y, a.z}, m * vec3{a.x, b.y, b.z}, 0xffffffff);
	add_debug_line(list, world2proj, m * vec3{b.x, b.y, a.z}, m * vec3{b.x, b.y, b.z}, 0xffffffff);

	add_debug_line(list, world2proj, m * vec3{a.x, a.y, a.z}, m * vec3{b.x, a.y, a.z}, 0xff4040ff); // X
	add_debug_line(list, world2proj, m * vec3{a.x, b.y, a.z}, m * vec3{b.x, b.y, a.z}, 0xffffffff);
	add_debug_line(list, world2proj, m * vec3{a.x, a.y, b.z}, m * vec3{b.x, a.y, b.z}, 0xffffffff);
	add_debug_line(list, world2proj, m * vec3{a.x, b.y, b.z}, m * vec3{b.x, b.y, b.z}, 0xffffffff);

	add_debug_line(list, world2proj, m * vec3{a.x, a.y, a.z}, m * vec3{a.x, b.y, a.z}, 0xff40ff40); // Y
	add_debug_line(list, world2proj, m * vec3{b.x, a.y, a.z}, m * vec3{b.x, b.y, a.z}, 0xffffffff);
	add_debug_line(list, world2proj, m * vec3{a.x, a.y, b.z}, m * vec3{a.x, b.y, b.z}, 0xffffffff);
	add_debug_line(list, world2proj, m * vec3{b.x, a.y, b.z}, m * vec3{b.x, b.y, b.z}, 0xffffffff);
}

void visualize_camera(ImDrawList* list, const mat4& world2proj, const mat4x3& xform, float aspect, uint32_t col, float thickness) {
	const float axis_size = 0.025f;
	const vec3* xforms = (const vec3*)&xform;
	vec3 pos = xforms[3];
	add_debug_line(list, world2proj, pos, pos+axis_size*xforms[0], 0xff4040ff, thickness);
	add_debug_line(list, world2proj, pos, pos+axis_size*xforms[1], 0xff40ff40, thickness);
	add_debug_line(list, world2proj, pos, pos+axis_size*xforms[2], 0xffff4040, thickness);
	float xs = axis_size * aspect;
	float ys = axis_size;
	float zs = axis_size * 2.0f * aspect;
	vec3 a = pos + xs * xforms[0] + ys * xforms[1] + zs * xforms[2];
	vec3 b = pos - xs * xforms[0] + ys * xforms[1] + zs * xforms[2];
	vec3 c = pos - xs * xforms[0] - ys * xforms[1] + zs * xforms[2];
	vec3 d = pos + xs * xforms[0] - ys * xforms[1] + zs * xforms[2];
	add_debug_line(list, world2proj, pos, a, col, thickness);
	add_debug_line(list, world2proj, pos, b, col, thickness);
	add_debug_line(list, world2proj, pos, c, col, thickness);
	add_debug_line(list, world2proj, pos, d, col, thickness);
	add_debug_line(list, world2proj, a, b, col, thickness);
	add_debug_line(list, world2proj, b, c, col, thickness);
	add_debug_line(list, world2proj, c, d, col, thickness);
	add_debug_line(list, world2proj, d, a, col, thickness);
}

bool CameraPath::imgui_viz(
	ImDrawList* list,
	mat4 &view2proj,
	mat4 &world2proj,
	mat4 &world2view,
	vec2 focal,
	float aspect,
	float znear,
	float zfar,
	CameraPredictor* cam_predictor
) {
	bool changed = false;
	// float flx = focal.x;
	float fly = focal.y;
	mat4 view2proj_guizmo = transpose(mat4{
		fly * 2.0f / aspect, 0.0f, 0.0f, 0.0f,
		0.0f, -fly * 2.0f, 0.0f, 0.0f,
		0.0f, 0.0f, (zfar + znear) / (zfar - znear), -(2.0f * zfar * znear) / (zfar - znear),
		0.0f, 0.0f, 1.0f, 0.0f,
	});

	if (!update_cam_from_path && !keyframes.empty()) {
		auto p = get_pos(play_time);
		int cur_cam_i = p.kfidx;
		if (!loop) cur_cam_i += (int)round(p.t);

		vec3 prevp;
		for (int i = 0; i < keyframes.size(); i += max(min(keyframe_subsampling, (int)keyframes.size() - 1 - i), 1)) {
			visualize_camera(list, world2proj, keyframes[i].m(), aspect, (i==cur_cam_i) ? 0xff80c0ff : 0x8080c0ff);
			vec3 p = keyframes[i].T;
			if (i && keyframe_subsampling == 1) {
				add_debug_line(list, world2proj, prevp, p, 0xccffc040);
			}
			prevp = p;
		}

		ImGuiIO& io = ImGui::GetIO();
		mat4 matrix = keyframes[cur_cam_i].m();
		ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
		if (ImGuizmo::Manipulate((const float*)&world2view, (const float*)&view2proj_guizmo, (ImGuizmo::OPERATION)m_gizmo_op, (ImGuizmo::MODE)m_gizmo_mode, (float*)&matrix, NULL, NULL)) {
			// Find overlapping keypoints...
			int i0 = cur_cam_i; while (i0 > 0 && keyframes[cur_cam_i].same_pos_as(keyframes[i0 - 1])) i0--;
			int i1 = cur_cam_i; while (i1 < keyframes.size() - 1 && keyframes[cur_cam_i].same_pos_as(keyframes[i1 + 1])) i1++;

			vec3 tdiff = matrix[3].xyz() - keyframes[cur_cam_i].T;
			mat3 rdiff = mat_log(mat3(matrix) * inverse(to_mat3(normalize(keyframes[cur_cam_i].R))));

			for (int i = 0; i < keyframes.size(); ++i) {
				float x = (get_playtime(i) - get_playtime(cur_cam_i)) / editing_kernel_radius;
				float w = editing_kernel(x, editing_kernel_type);

				keyframes[i].T += w * tdiff;
				keyframes[i].R = quat(mat_exp(w * rdiff) * to_mat3(normalize(keyframes[i].R)));
			}

			// ...and ensure overlapping keypoints were edited exactly in tandem
			for (int i = i0; i <= i1; ++i) {
				keyframes[i].T = keyframes[cur_cam_i].T;
				keyframes[i].R = keyframes[cur_cam_i].R;
			}

			changed = true;
		}

		visualize_camera(list, world2proj, eval_camera_path(play_time).m(), aspect, 0xff80ff80);

		float dt = 0.001f;
		float total_length = 0.0f;
		for (float t = 0.0f;; t += dt) {
			if (t > 1.0f) t = 1.0f;
			vec3 p = eval_camera_path(t).T;
			if (t) {
				total_length += distance(prevp, p);
			}
			prevp = p;
			if (t >= 1.0f) break;
		}

		dt = 0.001f / total_length;
		static const uint32_t N_DASH_STEPS = 10;
		uint32_t i = 0;
		for (float t = 0.0f;; t += dt, ++i) {
			if (t > 1.0f) t = 1.0f;
			vec3 p = eval_camera_path(t).T;
			if (t && (i / N_DASH_STEPS) % 2 == 0) {
				float thickness = 1.0f;
				if (editing_kernel_type != EEditingKernel::None) {
					float x = (t + dt/2.0f - get_playtime(cur_cam_i)) / editing_kernel_radius;
					thickness += 4.0f * editing_kernel(x, editing_kernel_type);
				}

				add_debug_line(list, world2proj, prevp, p, 0xff80c0ff, thickness);
			}

			prevp = p;
			if (t >= 1.0f) break;
		}
	}

	return changed;
}
#endif //NGP_GUI

}
