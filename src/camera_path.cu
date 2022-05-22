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

#ifdef NGP_GUI
#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>
#endif

#include <json/json.hpp>
#include <fstream>

using namespace Eigen;
using namespace nlohmann;

NGP_NAMESPACE_BEGIN

CameraKeyframe lerp(const CameraKeyframe& p0, const CameraKeyframe& p1, float t, float t0, float t1) {
	t = (t - t0) / (t1 - t0);
	Eigen::Vector4f R1 = p1.R;

	// take the short path
	if (R1.dot(p0.R) < 0.f)  {
		R1=-R1;
	}

	return {
		Eigen::Quaternionf(p0.R).slerp(t, Eigen::Quaternionf(R1)).coeffs(),
		p0.T + (p1.T - p0.T) * t,
		p0.slice + (p1.slice - p0.slice) * t,
		p0.scale + (p1.scale - p0.scale) * t,
		p0.fov + (p1.fov - p0.fov) * t,
		p0.dof + (p1.dof - p0.dof) * t,
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
		// cublic bspline
		float tt=t*t;
		float ttt=t*t*t;
		float a = (1-t)*(1-t)*(1-t)*(1.f/6.f);
		float b = (3.f*ttt-6.f*tt+4.f)*(1.f/6.f);
		float c = (-3.f*ttt+3.f*tt+3.f*t+1.f)*(1.f/6.f);
		float d = ttt*(1.f/6.f);
		return p0 * a + p1 * b + p2 * c + p3 * d;
	}
}

void to_json(json& j, const CameraKeyframe& p) {
	j = json{{"R", p.R}, {"T", p.T}, {"slice", p.slice}, {"scale", p.scale}, {"fov", p.fov}, {"dof", p.dof}};
}

bool load_relative_to_first=false; // set to true when using a camera path that is aligned with the first training image, such that it is invariant to changes in the space of the training data

void from_json(bool is_first, const json& j, CameraKeyframe& p, const CameraKeyframe& first, const Eigen::Matrix<float, 3, 4>& ref) {
	 if (is_first && load_relative_to_first) {
	 	p.from_m(ref);
	 } else {
		p.R=Eigen::Vector4f(j["R"][0],j["R"][1],j["R"][2],j["R"][3]);
		p.T=Eigen::Vector3f(j["T"][0],j["T"][1],j["T"][2]);

		if (load_relative_to_first) {
	 		Eigen::Matrix4f ref4 = Eigen::Matrix4f::Identity();
	 		ref4.block<3,4>(0,0) = ref;

	 		Eigen::Matrix4f first4 = Eigen::Matrix4f::Identity();
	 		first4.block<3,4>(0,0) = first.m();

	 		Eigen::Matrix4f p4 = Eigen::Matrix4f::Identity();
	 		p4.block<3,4>(0,0) = p.m();

	 		auto cur4 = ref4 * first4.inverse() * p4;
	 		p.from_m(cur4.block<3,4>(0,0));
		}
	}
	j.at("slice").get_to(p.slice);
	j.at("scale").get_to(p.scale);
	j.at("fov").get_to(p.fov);
	j.at("dof").get_to(p.dof);
}


void CameraPath::save(const std::string& filepath_string) {
	json j = {
		{"time", m_playtime},
		{"path", m_keyframes}
	};
	std::ofstream f(filepath_string);
	f << j;
}

void CameraPath::load(const std::string& filepath_string, const Eigen::Matrix<float, 3, 4> &first_xform) {
	std::ifstream f(filepath_string);
	if (!f) {
		throw std::runtime_error{std::string{"Camera path \""} + filepath_string + "\" does not exist."};
	}

	json j;
	f >> j;

	CameraKeyframe first;

	m_keyframes.clear();
	if (j.contains("time")) m_playtime=j["time"];
	if (j.contains("path")) for (auto &el : j["path"]) {
		CameraKeyframe p;
		bool is_first = m_keyframes.empty();
		from_json(is_first, el, p, first, first_xform);
		if (is_first) {
			first = p;
		}
		m_keyframes.push_back(p);
	}
}

#ifdef NGP_GUI
int CameraPath::imgui(char path_filename_buf[128], float frame_milliseconds, Matrix<float, 3, 4> &camera, float slice_plane_z, float scale, float fov, float dof, float bounding_radius, const Eigen::Matrix<float, 3, 4> &first_xform) {
	int n=std::max(0,int(m_keyframes.size())-1);
	int read= 0;					// 1=smooth, 2=hard
	if (!m_keyframes.empty()) {
		if (ImGui::SliderFloat("camera path time", &m_playtime, 0.f, 1.f)) read=1;
		ImGui::SliderFloat("auto play speed",&m_autoplayspeed, 0.f, 1.f);
		if (m_autoplayspeed>0.f && m_playtime<1.f) {
			m_playtime+=m_autoplayspeed*(frame_milliseconds/1000.f);
			if (m_playtime>1.f) m_playtime=1.f;
			read=1;
		}
	}
	if (ImGui::Button("Add from cam")) {
		int i=(int)ceil(m_playtime*(float)n+0.001f);
		if (i>m_keyframes.size()) i=m_keyframes.size();
		if (i<0) i=0;
		m_keyframes.insert(m_keyframes.begin()+i, CameraKeyframe(camera, slice_plane_z, scale, fov, dof));
		m_update_cam_from_path = false;
		int n=std::max(0,int(m_keyframes.size())-1);
		m_playtime = n ? float(i)/float(n) : 1.f;
		read = 2;
	}
	if (!m_keyframes.empty()) {
		ImGui::SameLine();
		if (ImGui::Button("split")) {
			m_update_cam_from_path=false;
			int i=(int)ceil(m_playtime*(float)n+0.001f);
			if (i>m_keyframes.size()) i=(int)m_keyframes.size();
			if (i<0) i=0;
			m_keyframes.insert(m_keyframes.begin()+i, eval_camera_path(m_playtime));
			m_playtime=float(i)/float(n+1);
			read=2;
		}
		ImGui::SameLine();
		int i=(int)round(m_playtime*(float)n);
		if (ImGui::Button("|<")) { m_playtime=0.f; read=2; }						ImGui::SameLine();
		if (ImGui::Button("<")) { m_playtime=n?std::max(0.f,floorf((m_playtime-0.0001f)*(float)n)/(float)n):0.f; read=2;}		ImGui::SameLine();
		if (ImGui::Button(m_update_cam_from_path ? "STOP" : "READ")) { m_update_cam_from_path=!m_update_cam_from_path; read=2; }				ImGui::SameLine();
		if (ImGui::Button(">")) { m_playtime=n?std::min(1.f,ceilf((m_playtime+0.0001f)*(float)n)/(float)n):1.f; read=2;}			ImGui::SameLine();
		if (ImGui::Button(">|")) { m_playtime=1.f; read=2;}						ImGui::SameLine();
		if (ImGui::Button("Dup")) { m_update_cam_from_path=false; m_keyframes.insert(m_keyframes.begin()+i, m_keyframes[i]); m_playtime=i/float(n+1); read=2;} ImGui::SameLine();
		if (ImGui::Button("Del")) { m_update_cam_from_path=false; m_keyframes.erase(m_keyframes.begin()+i); read=2;} ImGui::SameLine();
		if (ImGui::Button("Set")) { m_keyframes[i]=CameraKeyframe(camera, slice_plane_z, scale, fov, dof); read=2; if (n) m_playtime=i/float(n); }

		if (ImGui::RadioButton("Translate", m_gizmo_op == ImGuizmo::TRANSLATE))
			m_gizmo_op = ImGuizmo::TRANSLATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Rotate", m_gizmo_op == ImGuizmo::ROTATE))
			m_gizmo_op = ImGuizmo::ROTATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Local", m_gizmo_mode == ImGuizmo::LOCAL))
			m_gizmo_mode = ImGuizmo::LOCAL;
		ImGui::SameLine();
		if (ImGui::RadioButton("World", m_gizmo_mode == ImGuizmo::WORLD))
			m_gizmo_mode = ImGuizmo::WORLD;
	}

	ImGui::InputText("##PathFile", path_filename_buf, 128);
	ImGui::SameLine();
	static std::string camera_path_load_error_string = "";
	if (ImGui::Button("Load")) {
		try {
			load(path_filename_buf, first_xform);
		} catch (std::exception& e) {
			ImGui::OpenPopup("Camera path load error");
			camera_path_load_error_string = std::string{"Failed to load camera path: "} + e.what();
		}
	}
	if (ImGui::BeginPopupModal("Camera path load error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("%s", camera_path_load_error_string.c_str());
		if (ImGui::Button("OK", ImVec2(120, 0))) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	if (!m_keyframes.empty()) {
		ImGui::SameLine();
		if (ImGui::Button("Save"))
			save(path_filename_buf);
	}
	if (!m_keyframes.empty()) {
		int i=(int)round(m_playtime*(float)n);
		ImGui::Text("Current keyframe %d/%d:", i, n+1);
		if (ImGui::SliderFloat("fov", &m_keyframes[i].fov, 0.0f, 120.0f)) read=2;
		if (ImGui::SliderFloat("dof", &m_keyframes[i].dof, 0.0f, 0.1f)) read=2;
		if (ImGui::SliderFloat("slice Z", &m_keyframes[i].slice, -bounding_radius, bounding_radius)) read=2;
		if (ImGui::SliderFloat("scale", &m_keyframes[i].scale, 0.f,10.f)) read=2;
	}
	return m_keyframes.empty() ? 0 : read;
}

bool DebugProject(const Matrix<float, 4, 4>&proj, Vector3f p, ImVec2& o) {
	Vector4f ph; ph << p, 1.f;
	Vector4f pa = proj * ph;
	if (pa.w() <= 0.f) return false;
	o.x = pa.x() / pa.w();
	o.y = pa.y() / pa.w();
	return true;
}

void add_debug_line(ImDrawList* list, const Matrix<float, 4, 4>& proj, Vector3f a, Vector3f b, uint32_t col, float thickness) {
	ImVec2 aa, bb;
	if (DebugProject(proj, a, aa) && DebugProject(proj, b, bb)) {
		list->AddLine(aa, bb, col, thickness);
	}
}

void visualize_unit_cube(ImDrawList* list, const Matrix<float, 4, 4>& world2proj) {
	add_debug_line(list, world2proj, Vector3f{0.f,0.f,0.f}, Vector3f{0.f,0.f,1.f}, 0xffff4040); // Z
	add_debug_line(list, world2proj, Vector3f{1.f,0.f,0.f}, Vector3f{1.f,0.f,1.f}, 0xffffffff);
	add_debug_line(list, world2proj, Vector3f{0.f,1.f,0.f}, Vector3f{0.f,1.f,1.f}, 0xffffffff);
	add_debug_line(list, world2proj, Vector3f{1.f,1.f,0.f}, Vector3f{1.f,1.f,1.f}, 0xffffffff);

	add_debug_line(list, world2proj, Vector3f{0.f,0.f,0.f}, Vector3f{1.f,0.f,0.f}, 0xff4040ff); // X
	add_debug_line(list, world2proj, Vector3f{0.f,1.f,0.f}, Vector3f{1.f,1.f,0.f}, 0xffffffff);
	add_debug_line(list, world2proj, Vector3f{0.f,0.f,1.f}, Vector3f{1.f,0.f,1.f}, 0xffffffff);
	add_debug_line(list, world2proj, Vector3f{0.f,1.f,1.f}, Vector3f{1.f,1.f,1.f}, 0xffffffff);

	add_debug_line(list, world2proj, Vector3f{0.f,0.f,0.f}, Vector3f{0.f,1.f,0.f}, 0xff40ff40); // Y
	add_debug_line(list, world2proj, Vector3f{1.f,0.f,0.f}, Vector3f{1.f,1.f,0.f}, 0xffffffff);
	add_debug_line(list, world2proj, Vector3f{0.f,0.f,1.f}, Vector3f{0.f,1.f,1.f}, 0xffffffff);
	add_debug_line(list, world2proj, Vector3f{1.f,0.f,1.f}, Vector3f{1.f,1.f,1.f}, 0xffffffff);
}

void visualize_nerf_camera(ImDrawList* list, const Matrix<float, 4, 4>& world2proj, const Eigen::Matrix<float, 3, 4>& xform, float aspect, uint32_t col) {
	const float axis_size = 0.025f;
	const Vector3f *xforms = (const Vector3f*)&xform;
	Vector3f pos = xforms[3];
	add_debug_line(list, world2proj, pos, pos+axis_size*xforms[0], 0xff4040ff);
	add_debug_line(list, world2proj, pos, pos+axis_size*xforms[1], 0xff40ff40);
	add_debug_line(list, world2proj, pos, pos+axis_size*xforms[2], 0xffff4040);
	float xs=axis_size*aspect;
	float ys=axis_size;
	float zs=axis_size*2.f*aspect;
	Vector3f a=pos+xs*xforms[0]+ys*xforms[1]+zs*xforms[2];
	Vector3f b=pos-xs*xforms[0]+ys*xforms[1]+zs*xforms[2];
	Vector3f c=pos-xs*xforms[0]-ys*xforms[1]+zs*xforms[2];
	Vector3f d=pos+xs*xforms[0]-ys*xforms[1]+zs*xforms[2];
	add_debug_line(list, world2proj, pos, a, col);
	add_debug_line(list, world2proj, pos, b, col);
	add_debug_line(list, world2proj, pos, c, col);
	add_debug_line(list, world2proj, pos, d, col);
	add_debug_line(list, world2proj, a, b, col);
	add_debug_line(list, world2proj, b, c, col);
	add_debug_line(list, world2proj, c, d, col);
	add_debug_line(list, world2proj, d, a, col);
}

bool CameraPath::imgui_viz(ImDrawList* list, Matrix<float, 4, 4> &view2proj, Matrix<float, 4, 4> &world2proj, Matrix<float, 4, 4> &world2view, Vector2f focal, float aspect) {
	bool changed=false;
	float flx = focal.x();
	float fly = focal.y();
	Matrix<float, 4, 4> view2proj_guizmo;
	float zfar = 100.f;
	float znear = 0.1f;
	view2proj_guizmo <<
		fly*2.f/aspect, 0, 0, 0,
		0, -fly*2.f, 0, 0,
		0, 0, (zfar+znear)/(zfar-znear), -(2.f*zfar*znear) / (zfar-znear),
		0, 0, 1, 0;

	if (!m_update_cam_from_path) {
		ImDrawList* list = ImGui::GetForegroundDrawList();
		int cur_cam_i=(int)round(m_playtime * (float)(m_keyframes.size()-1));
		Eigen::Vector3f prevp;
		for (int i=0;i<m_keyframes.size();++i) {
			visualize_nerf_camera(list, world2proj, m_keyframes[i].m(), aspect, (i==cur_cam_i) ? 0xff80c0ff : 0x8080c0ff);
			Eigen::Vector3f p=m_keyframes[i].T;
			if (i) {
				add_debug_line(list, world2proj, prevp, p, 0xccffc040);
			}
			prevp=p;
		}
		if (!m_keyframes.empty()) {
			ImGuiIO& io = ImGui::GetIO();
			Eigen::Matrix4f matrix=Eigen::Matrix4f::Identity();
			matrix.block<3,4>(0,0) = m_keyframes[cur_cam_i].m();
			ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
			if (ImGuizmo::Manipulate((const float*)&world2view, (const float*)&view2proj_guizmo, (ImGuizmo::OPERATION)m_gizmo_op, (ImGuizmo::MODE)m_gizmo_mode, (float*)&matrix, NULL, NULL)) {
				int i0=cur_cam_i; while (i0>0 && m_keyframes[cur_cam_i].SamePosAs(m_keyframes[i0-1])) i0--;
				int i1=cur_cam_i; while (i1<m_keyframes.size()-1 && m_keyframes[cur_cam_i].SamePosAs(m_keyframes[i1+1])) i1++;
				for (int i=i0;i<=i1;++i) {
					m_keyframes[i].T=matrix.block<3,4>(0,0).col(3);
					m_keyframes[i].R=Eigen::Quaternionf(matrix.block<3,3>(0,0)).coeffs();
				}
				changed=true;
			}

			visualize_nerf_camera(list, world2proj, eval_camera_path(m_playtime).m(), aspect, 0xff80ff80);
			float dt = 0.05f / (float)m_keyframes.size();
			Eigen::Vector3f prevp;
			for (float t=0.f;;t+=dt) {
				if (t>1.f) t=1.f;
				Eigen::Vector3f p=eval_camera_path(t).T;
				if (t) {
					// draw a line
					add_debug_line(list, world2proj, (prevp+p)*0.5f, p, 0xff80c0ff);
				}
				prevp=p;
				if (t>=1.f) break;
			}
		}
	}
	return changed;
}
#endif //NGP_GUI

NGP_NAMESPACE_END
