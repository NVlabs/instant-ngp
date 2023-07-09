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
		slerp(p0.R, R1, t),
		p0.T + (p1.T - p0.T) * t,
		p0.slice + (p1.slice - p0.slice) * t,
		p0.scale + (p1.scale - p0.scale) * t,
		p0.fov + (p1.fov - p0.fov) * t,
		p0.aperture_size + (p1.aperture_size - p0.aperture_size) * t,
		// Note, the glow mode from the previous frame is used, since the modes cannot be interpolated
		p0.glow_mode,
		p0.glow_y_cutoff + (p1.glow_y_cutoff - p0.glow_y_cutoff) * t,
	};
}

CameraKeyframe spline(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2, const CameraKeyframe& p3) {
	if (0) {
		// catmull rom spline
		CameraKeyframe q0 = lerp(p0, p1, t, -1.f, 0.f);
		CameraKeyframe q1 = lerp(p1, p2, t,  0.f, 1.f);
		CameraKeyframe q2 = lerp(p2, p3, t,  1.f, 2.f);
		CameraKeyframe r0 = lerp(q0, q1, t, -1.f, 1.f);
		CameraKeyframe r1 = lerp(q1, q2, t,  0.f, 2.f);
		return lerp(r0, r1, t, 0.f, 1.f);
	} else {
		// cubic bspline
		float tt = t*t;
		float ttt = t*t*t;
		float a = (1-t)*(1-t)*(1-t)*(1.f/6.f);
		float b = (3.f*ttt-6.f*tt+4.f)*(1.f/6.f);
		float c = (-3.f*ttt+3.f*tt+3.f*t+1.f)*(1.f/6.f);
		float d = ttt*(1.f/6.f);
		return p0 * a + p1 * b + p2 * c + p3 * d;
	}
}

void to_json(json& j, const CameraKeyframe& p) {
	j = json{
		{"R", p.R},
		{"T", p.T},
		{"slice", p.slice},
		{"scale", p.scale},
		{"fov", p.fov},
		{"aperture_size", p.aperture_size},
		{"glow_mode", p.glow_mode},
		{"glow_y_cutoff", p.glow_y_cutoff},
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
	if (j.contains("glow_mode")) j.at("glow_mode").get_to(p.glow_mode); else p.glow_mode = 0;
	if (j.contains("glow_y_cutoff")) j.at("glow_y_cutoff").get_to(p.glow_y_cutoff); else p.glow_y_cutoff = 0.f;
}

void CameraPath::save(const fs::path& path) {
	json j = {
		{"loop", loop},
		{"time", play_time},
		{"path", keyframes},
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
}

#ifdef NGP_GUI
int CameraPath::imgui(char path_filename_buf[1024], float frame_milliseconds, mat4x3& camera, float slice_plane_z, float scale, float fov, float aperture_size, float bounding_radius, const mat4x3& first_xform, int glow_mode, float glow_y_cutoff) {
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
		int i = (int)ceil(play_time * (float)n + 0.001f);
		if (i > keyframes.size()) { i = keyframes.size(); }
		if (i < 0) i = 0;
		keyframes.insert(keyframes.begin() + i, CameraKeyframe(camera, slice_plane_z, scale, fov, aperture_size, glow_mode, glow_y_cutoff));
		update_cam_from_path = false;
		int n = std::max(0, int(keyframes.size()) - 1);
		play_time = n ? float(i) / float(n) : 1.0f;
		read = 2;
	}

	if (!keyframes.empty()) {
		ImGui::SameLine();
		if (ImGui::Button("Split")) {
			update_cam_from_path = false;
			int i = (int)ceil(play_time * (float)n + 0.001f);
			if (i > keyframes.size()) { i = (int)keyframes.size(); }
			if (i < 0) { i = 0; }
			keyframes.insert(keyframes.begin() + i, eval_camera_path(play_time));
			play_time = float(i) / float(n + 1);
			read = 2;
		}
		ImGui::SameLine();
		int i=(int)round(play_time * (float)n);
		if (ImGui::Button("|<")) { play_time = 0.f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button("<")) { play_time = n ? std::max(0.0f, floorf((play_time - 0.0001f) * (float)n) / (float)n) : 0.f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button(update_cam_from_path ? "Stop" : "Read")) { update_cam_from_path = !update_cam_from_path; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button(">")) { play_time = n ? std::min(1.0f, ceilf((play_time + 0.0001f) * (float)n) / (float)n) : 1.0f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button(">|")) { play_time = 1.0f; read = 2; }
		ImGui::SameLine();
		if (ImGui::Button("Dup")) { update_cam_from_path = false; keyframes.insert(keyframes.begin() + i, keyframes[i]); play_time = i / float(n + 1); read = 2; }
		ImGui::SameLine();
		if (ImGui::Button("Del")) { update_cam_from_path = false; keyframes.erase(keyframes.begin() + i); read = 2; }
		ImGui::SameLine();
		if (ImGui::Button("Set")) { keyframes[i] = CameraKeyframe(camera, slice_plane_z, scale, fov, aperture_size, glow_mode, glow_y_cutoff); read = 2; if (n) play_time = i / float(n); }

		if (ImGui::RadioButton("Translate", m_gizmo_op == ImGuizmo::TRANSLATE)) { m_gizmo_op = ImGuizmo::TRANSLATE; }
		ImGui::SameLine();
		if (ImGui::RadioButton("Rotate", m_gizmo_op == ImGuizmo::ROTATE)) { m_gizmo_op = ImGuizmo::ROTATE; }
		ImGui::SameLine();
		if (ImGui::RadioButton("Local", m_gizmo_mode == ImGuizmo::LOCAL)) { m_gizmo_mode = ImGuizmo::LOCAL; }
		ImGui::SameLine();
		if (ImGui::RadioButton("World", m_gizmo_mode == ImGuizmo::WORLD)) { m_gizmo_mode = ImGuizmo::WORLD; }
		ImGui::SameLine();
		ImGui::Checkbox("Loop path", &loop);

		if (ImGui::SliderFloat("Camera path time", &play_time, 0.0f, 1.0f)) { read = 1; }

		ImGui::SliderFloat("Auto play speed", &auto_play_speed, 0.0f, 1.0f);
		if (auto_play_speed > 0.0f && play_time < 1.0f) {
			play_time += auto_play_speed * (frame_milliseconds / 1000.f);
			if (play_time >= 1.0f) {
				play_time = 1.0f;
			}

			read = 1;
		}

		ImGui::Text("Current keyframe %d/%d:", i, n+1);
		if (ImGui::SliderFloat("Field of view", &keyframes[i].fov, 0.0f, 120.0f)) read = 2;
		if (ImGui::SliderFloat("Aperture size", &keyframes[i].aperture_size, 0.0f, 0.1f)) read = 2;
		if (ImGui::SliderFloat("Slice Z", &keyframes[i].slice, -bounding_radius, bounding_radius)) read = 2;
		if (ImGui::SliderFloat("Scale", &keyframes[i].scale, 0.f,10.f)) read = 2;
		if (ImGui::SliderInt("Glow Mode", &keyframes[i].glow_mode, 0,16)) read = 2;
		if (ImGui::SliderFloat("Glow Y Cutoff", &keyframes[i].glow_y_cutoff, -2.f,3.f)) read = 2;
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

void visualize_nerf_camera(ImDrawList* list, const mat4& world2proj, const mat4x3& xform, float aspect, uint32_t col, float thickness) {
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

bool CameraPath::imgui_viz(ImDrawList* list, mat4 &view2proj, mat4 &world2proj, mat4 &world2view, vec2 focal, float aspect, float znear, float zfar) {
	bool changed = false;
	// float flx = focal.x;
	float fly = focal.y;
	mat4 view2proj_guizmo = transpose(mat4{
		fly * 2.0f / aspect, 0.0f, 0.0f, 0.0f,
		0.0f, -fly * 2.0f, 0.0f, 0.0f,
		0.0f, 0.0f, (zfar + znear) / (zfar - znear), -(2.0f * zfar * znear) / (zfar - znear),
		0.0f, 0.0f, 1.0f, 0.0f,
	});

	if (!update_cam_from_path) {
		ImDrawList* list = ImGui::GetForegroundDrawList();
		int cur_cam_i=(int)round(play_time * (float)(keyframes.size()-1));
		vec3 prevp;
		for (int i = 0; i < keyframes.size(); ++i) {
			visualize_nerf_camera(list, world2proj, keyframes[i].m(), aspect, (i==cur_cam_i) ? 0xff80c0ff : 0x8080c0ff);
			vec3 p = keyframes[i].T;
			if (i) {
				add_debug_line(list, world2proj, prevp, p, 0xccffc040);
			}
			prevp = p;
		}
		if (!keyframes.empty()) {
			ImGuiIO& io = ImGui::GetIO();
			mat4 matrix = keyframes[cur_cam_i].m();
			ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
			if (ImGuizmo::Manipulate((const float*)&world2view, (const float*)&view2proj_guizmo, (ImGuizmo::OPERATION)m_gizmo_op, (ImGuizmo::MODE)m_gizmo_mode, (float*)&matrix, NULL, NULL)) {
				int i0 = cur_cam_i; while (i0 > 0 && keyframes[cur_cam_i].same_pos_as(keyframes[i0 - 1])) i0--;
				int i1 = cur_cam_i; while (i1 < keyframes.size() - 1 && keyframes[cur_cam_i].same_pos_as(keyframes[i1 + 1])) i1++;
				for (int i = i0; i <= i1; ++i) {
					keyframes[i].T = matrix[3].xyz();
					keyframes[i].R = quat(mat3(matrix));
				}
				changed=true;
			}

			visualize_nerf_camera(list, world2proj, eval_camera_path(play_time).m(), aspect, 0xff80ff80);
			float dt = 0.05f / (float)keyframes.size();
			vec3 prevp;
			for (float t = 0.0f;; t += dt) {
				if (t > 1.0f) t = 1.0f;
				vec3 p = eval_camera_path(t).T;
				if (t) {
					// draw a line
					add_debug_line(list, world2proj, (prevp+p) * 0.5f, p, 0xff80c0ff);
				}
				prevp = p;
				if (t >= 1.0f) break;
			}
		}
	}
	return changed;
}
#endif //NGP_GUI

}
