/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <json/json.hpp>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#include <fstream>
#include <set>
#include <unordered_set>

#ifdef NGP_GUI
#  include <imgui/imgui.h>
#  include <imgui/backends/imgui_impl_glfw.h>
#  include <imgui/backends/imgui_impl_opengl3.h>
#  include <imguizmo/ImGuizmo.h>
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#  include <GLFW/glfw3native.h>
#  include <cuda_gl_interop.h>

#endif

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far


using namespace Eigen;
using namespace std::literals::chrono_literals;
using namespace tcnn;
namespace fs = filesystem;

int do_system(const std::string& cmd) {
#ifdef _WIN32
	tlog::info() << "> " << cmd;
#else
	tlog::info() << "$ " << cmd;
#endif
	return system(cmd.c_str());
}

NGP_NAMESPACE_BEGIN

std::atomic<size_t> g_total_n_bytes_allocated{0};

json merge_parent_network_config(const json &child, const fs::path &child_filename) {
	if (!child.contains("parent")) {
		return child;
	}
	fs::path parent_filename = child_filename.parent_path() / std::string(child["parent"]);
	tlog::info() << "Loading parent network config from: " << parent_filename.str();
	std::ifstream f{parent_filename.str()};
	json parent = json::parse(f, nullptr, true, true);
	parent = merge_parent_network_config(parent, parent_filename);
	parent.merge_patch(child);
	return parent;
}

void Testbed::load_training_data(const std::string& data_path_str) {
	fs::path data_path = data_path_str;
	if (!data_path.exists()) {
		throw std::runtime_error{fmt::format("Data path '{}' does not exist.", data_path.str())};
	}

	// Automatically determine the mode from the first scene that's loaded
	ETestbedMode scene_mode = mode_from_scene(data_path.str());
	if (scene_mode == ETestbedMode::None) {
		throw std::runtime_error{fmt::format("Unknown scene format for path '{}'.", data_path.str())};
	}

	set_mode(scene_mode);

	m_data_path = data_path;

	if (!m_data_path.exists()) {
		throw std::runtime_error{fmt::format("Data path '{}' does not exist.", m_data_path.str())};
	}

	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:   load_nerf(); break;
		case ETestbedMode::Sdf:    load_mesh(); break;
		case ETestbedMode::Image:  load_image(); break;
		case ETestbedMode::Volume: load_volume(); break;
		default: throw std::runtime_error{"Invalid testbed mode."};
	}

	m_training_data_available = true;
}

void Testbed::clear_training_data() {
	m_training_data_available = false;
	m_nerf.training.dataset.metadata.clear();
}

void Testbed::set_mode(ETestbedMode mode) {
	if (mode == m_testbed_mode) {
		return;
	}

	// Reset mode-specific members
	m_image = {};
	m_mesh = {};
	m_nerf = {};
	m_sdf = {};
	m_volume = {};

	// Kill training-related things
	m_encoding = {};
	m_loss = {};
	m_network = {};
	m_nerf_network = {};
	m_optimizer = {};
	m_trainer = {};
	m_envmap = {};
	m_distortion = {};
	m_training_data_available = false;

	// Reset paths that might be attached to the chosen mode
	m_data_path = {};

	m_testbed_mode = mode;

	reset_camera();
}

fs::path Testbed::find_network_config(const fs::path& network_config_path) {
	if (network_config_path.exists()) {
		return network_config_path;
	}

	// The following resolution steps do not work if the path is absolute. Treat it as nonexistent.
	if (network_config_path.is_absolute()) {
		return network_config_path;
	}

	fs::path candidate = get_root_dir()/"configs"/to_string(m_testbed_mode)/network_config_path;
	if (candidate.exists()) {
		return candidate;
	}

	return network_config_path;
}

json Testbed::load_network_config(const fs::path& network_config_path) {
	bool is_snapshot = equals_case_insensitive(network_config_path.extension(), "msgpack");
	if (network_config_path.empty() || !network_config_path.exists()) {
		throw std::runtime_error{fmt::format("Network {} '{}' does not exist.", is_snapshot ? "snapshot" : "config", network_config_path.str())};
	}

	tlog::info() << "Loading network " << (is_snapshot ? "snapshot" : "config") << " from: " << network_config_path;

	json result;
	if (equals_case_insensitive(network_config_path.extension(), "json")) {
		std::ifstream f{network_config_path.str()};
		result = json::parse(f, nullptr, true, true);
		result = merge_parent_network_config(result, network_config_path);
	} else if (equals_case_insensitive(network_config_path.extension(), "msgpack")) {
		std::ifstream f{network_config_path.str(), std::ios::in | std::ios::binary};
		result = json::from_msgpack(f);
		// we assume parent pointers are already resolved in snapshots.
	}

	return result;
}

void Testbed::reload_network_from_file(const std::string& network_config_path_string) {
	if (!network_config_path_string.empty()) {
		fs::path candidate = find_network_config(network_config_path_string);
		if (candidate.exists()) {
			// Store the path _argument_ in the member variable. E.g. for the base config,
			// it'll store `base.json`, even though the loaded config will be
			// config/<mode>/base.json. This has the benefit of switching to the
			// appropriate config when switching modes.
			m_network_config_path = network_config_path_string;
		}
	}

	// If the testbed mode hasn't been decided yet, don't load a network yet, but
	// still keep track of the requested config (see above).
	if (m_testbed_mode == ETestbedMode::None) {
		return;
	}

	fs::path full_network_config_path = find_network_config(m_network_config_path);
	bool is_snapshot = equals_case_insensitive(full_network_config_path.extension(), "msgpack");

	if (!full_network_config_path.exists()) {
		tlog::warning() << "Network " << (is_snapshot ? "snapshot" : "config") << " path '" << full_network_config_path << "' does not exist.";
	} else {
		m_network_config = load_network_config(full_network_config_path);
	}

	// Reset training if we haven't loaded a snapshot of an already trained model, in which case, presumably the network
	// configuration changed and the user is interested in seeing how it trains from scratch.
	if (!is_snapshot) {
		reset_network();
	}
}

void Testbed::reload_network_from_json(const json& json, const std::string& config_base_path) {
	// config_base_path is needed so that if the passed in json uses the 'parent' feature, we know where to look...
	// be sure to use a filename, or if a directory, end with a trailing slash
	m_network_config = merge_parent_network_config(json, config_base_path);
	reset_network();
}

void Testbed::load_file(const std::string& file_path) {
	if (!fs::path{file_path}.exists()) {
		// If the path doesn't exist, but a network config can be resolved, load that.
		if (ends_with_case_insensitive(file_path, ".json") && find_network_config(file_path).exists()) {
			reload_network_from_file(file_path);
			return;
		}

		tlog::error() << "File '" << file_path << "' does not exist.";
		return;
	}

	if (ends_with_case_insensitive(file_path, ".msgpack")) {
		load_snapshot(file_path);
		return;
	}

	// If we get a json file, we need to parse it to determine its purpose.
	if (ends_with_case_insensitive(file_path, ".json")) {
		json file;
		{
			std::ifstream f{file_path};
			file = json::parse(f, nullptr, true, true);
		}

		// Snapshot in json format... inefficient, but technically supported.
		if (file.contains("snapshot")) {
			load_snapshot(file_path);
			return;
		}

		// Regular network config
		if (file.contains("parent") || file.contains("network") || file.contains("encoding") || file.contains("loss") || file.contains("optimizer")) {
			reload_network_from_file(file_path);
			return;
		}

		// Camera path
		if (file.contains("path")) {
			load_camera_path(file_path);
			return;
		}
	}

	// If the dragged file isn't any of the above, assume that it's training data
	try {
		bool was_training_data_available = m_training_data_available;
		load_training_data(file_path);

		if (!was_training_data_available) {
			// If we previously didn't have any training data and only now dragged
			// some into the window, it is very unlikely that the user doesn't
			// want to immediately start training on that data. So: go for it.
			m_train = true;
		}
	} catch (std::runtime_error& e) {
		tlog::error() << "Failed to load training data: " << e.what();
	}
}

void Testbed::reset_accumulation(bool due_to_camera_movement, bool immediate_redraw) {
	if (immediate_redraw) {
		redraw_next_frame();
	}

	if (!due_to_camera_movement || !reprojection_available()) {
		m_windowless_render_surface.reset_accumulation();
		for (auto& tex : m_render_surfaces) {
			tex.reset_accumulation();
		}
	}
}

void Testbed::set_visualized_dim(int dim) {
	m_visualized_dimension = dim;
	reset_accumulation();
}

void Testbed::translate_camera(const Vector3f& rel) {
	m_camera.col(3) += m_camera.block<3, 3>(0, 0) * rel * m_bounding_radius;
	reset_accumulation(true);
}

void Testbed::set_nerf_camera_matrix(const Matrix<float, 3, 4>& cam) {
	m_camera = m_nerf.training.dataset.nerf_matrix_to_ngp(cam);
}

Vector3f Testbed::look_at() const {
	return view_pos() + view_dir() * m_scale;
}

void Testbed::set_look_at(const Vector3f& pos) {
	m_camera.col(3) += pos - look_at();
}

void Testbed::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera.col(3) = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void Testbed::set_view_dir(const Vector3f& dir) {
	auto old_look_at = look_at();
	m_camera.col(0) = dir.cross(m_up_dir).normalized();
	m_camera.col(1) = dir.cross(m_camera.col(0)).normalized();
	m_camera.col(2) = dir.normalized();
	set_look_at(old_look_at);
}

void Testbed::first_training_view() {
	m_nerf.training.view = 0;
	set_camera_to_training_view(m_nerf.training.view);
	reset_accumulation();
}

void Testbed::last_training_view() {
	m_nerf.training.view = m_nerf.training.dataset.n_images-1;
	set_camera_to_training_view(m_nerf.training.view);
	reset_accumulation();
}

void Testbed::previous_training_view() {
	if (m_nerf.training.view != 0) {
		m_nerf.training.view -= 1;
	}
	set_camera_to_training_view(m_nerf.training.view);
	reset_accumulation();
}

void Testbed::next_training_view() {
	if (m_nerf.training.view != m_nerf.training.dataset.n_images-1) {
		m_nerf.training.view += 1;
	}
	set_camera_to_training_view(m_nerf.training.view);
	reset_accumulation();
}

void Testbed::set_camera_to_training_view(int trainview) {
	auto old_look_at = look_at();
	m_camera = m_smoothed_camera = get_xform_given_rolling_shutter(m_nerf.training.transforms[trainview], m_nerf.training.dataset.metadata[trainview].rolling_shutter, Vector2f{0.5f, 0.5f}, 0.0f);
	m_relative_focal_length = m_nerf.training.dataset.metadata[trainview].focal_length / (float)m_nerf.training.dataset.metadata[trainview].resolution[m_fov_axis];
	m_scale = std::max((old_look_at - view_pos()).dot(view_dir()), 0.1f);
	m_nerf.render_with_lens_distortion = true;
	m_nerf.render_lens = m_nerf.training.dataset.metadata[trainview].lens;
	m_screen_center = Vector2f::Constant(1.0f) - m_nerf.training.dataset.metadata[0].principal_point;
}

void Testbed::reset_camera() {
	m_fov_axis = 1;
	set_fov(50.625f);
	m_zoom = 1.f;
	m_screen_center = Vector2f::Constant(0.5f);
	m_scale = m_testbed_mode == ETestbedMode::Image ? 1.0f : 1.5f;
	m_camera <<
		1.0f, 0.0f, 0.0f, 0.5f,
		0.0f, -1.0f, 0.0f, 0.5f,
		0.0f, 0.0f, -1.0f, 0.5f;
	m_camera.col(3) -= m_scale * view_dir();

	m_smoothed_camera = m_camera;
	m_up_dir = {0.0f, 1.0f, 0.0f};
	m_sun_dir = Vector3f::Ones().normalized();

	reset_accumulation();
}

void Testbed::set_train(bool mtrain) {
	if (m_train && !mtrain && m_max_level_rand_training) {
		set_max_level(1.f);
	}
	m_train = mtrain;
}

std::string get_filename_in_data_path_with_suffix(fs::path data_path, fs::path network_config_path, const char* suffix) {
	// use the network config name along with the data path to build a filename with the requested suffix & extension
	std::string default_name = network_config_path.basename();
	if (default_name == "") default_name = "base";
	if (data_path.empty())
		return default_name + std::string(suffix);
	if (data_path.is_directory())
		return (data_path / (default_name + std::string{suffix})).str();
	else
		return data_path.stem().str() + "_" + default_name + std::string(suffix);
}

void Testbed::compute_and_save_marching_cubes_mesh(const char* filename, Vector3i res3d , BoundingBox aabb, float thresh, bool unwrap_it) {
	Matrix3f render_aabb_to_local = Matrix3f::Identity();
	if (aabb.is_empty()) {
		aabb = m_testbed_mode == ETestbedMode::Nerf ? m_render_aabb : m_aabb;
		render_aabb_to_local = m_render_aabb_to_local;
	}
	marching_cubes(res3d, aabb, render_aabb_to_local, thresh);
	save_mesh(m_mesh.verts, m_mesh.vert_normals, m_mesh.vert_colors, m_mesh.indices, filename, unwrap_it, m_nerf.training.dataset.scale, m_nerf.training.dataset.offset);
}

Eigen::Vector3i Testbed::compute_and_save_png_slices(const char* filename, int res, BoundingBox aabb, float thresh, float density_range, bool flip_y_and_z_axes) {
	Matrix3f render_aabb_to_local = Matrix3f::Identity();
	if (aabb.is_empty()) {
		aabb = m_testbed_mode == ETestbedMode::Nerf ? m_render_aabb : m_aabb;
		render_aabb_to_local = m_render_aabb_to_local;
	}
	if (thresh == std::numeric_limits<float>::max()) {
		thresh = m_mesh.thresh;
	}
	float range = density_range;
	if (m_testbed_mode == ETestbedMode::Sdf) {
		auto res3d = get_marching_cubes_res(res, aabb);
		aabb.inflate(range * aabb.diag().x()/res3d.x());
	}
	auto res3d = get_marching_cubes_res(res, aabb);
	if (m_testbed_mode == ETestbedMode::Sdf)
		range *= -aabb.diag().x()/res3d.x(); // rescale the range to be in output voxels. ie this scale factor is mapped back to the original world space distances.
			// negated so that black = outside, white = inside
	char fname[128];
	snprintf(fname, sizeof(fname), ".density_slices_%dx%dx%d.png", res3d.x(), res3d.y(), res3d.z());
	GPUMemory<float> density = (m_render_ground_truth && m_testbed_mode == ETestbedMode::Sdf) ? get_sdf_gt_on_grid(res3d, aabb, render_aabb_to_local) : get_density_on_grid(res3d, aabb, render_aabb_to_local);
	save_density_grid_to_png(density, (std::string(filename) + fname).c_str(), res3d, thresh, flip_y_and_z_axes, range);
	return res3d;
}

inline float linear_to_db(float x) {
	return -10.f*logf(x)/logf(10.f);
}

template <typename T>
void Testbed::dump_parameters_as_images(const T* params, const std::string& filename_base) {
	size_t non_layer_params_width = 2048;

	size_t layer_params = 0;
	for (auto size : m_network->layer_sizes()) {
		layer_params += size.first * size.second;
	}

	size_t n_params = m_network->n_params();
	size_t n_non_layer_params = n_params - layer_params;

	std::vector<T> params_cpu_network_precision(layer_params + next_multiple(n_non_layer_params, non_layer_params_width));
	std::vector<float> params_cpu(params_cpu_network_precision.size(), 0.0f);
	CUDA_CHECK_THROW(cudaMemcpy(params_cpu_network_precision.data(), params, n_params * sizeof(T), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < n_params; ++i) {
		params_cpu[i] = (float)params_cpu_network_precision[i];
	}

	size_t offset = 0;
	size_t layer_id = 0;
	for (auto size : m_network->layer_sizes()) {
		save_exr(params_cpu.data() + offset, size.second, size.first, 1, 1, fmt::format("{}-layer-{}.exr", filename_base, layer_id).c_str());
		offset += size.first * size.second;
		++layer_id;
	}

	if (n_non_layer_params > 0) {
		std::string filename = fmt::format("{}-non-layer.exr", filename_base);
		save_exr(params_cpu.data() + offset, non_layer_params_width, n_non_layer_params / non_layer_params_width, 1, 1, filename.c_str());
	}
}

template void Testbed::dump_parameters_as_images<__half>(const __half*, const std::string&);
template void Testbed::dump_parameters_as_images<float>(const float*, const std::string&);

Eigen::Matrix<float, 3, 4> Testbed::crop_box(bool nerf_space) const {
	Eigen::Vector3f cen = m_render_aabb_to_local.transpose() * m_render_aabb.center();
	Eigen::Vector3f radius = m_render_aabb.diag() * 0.5f;
	Eigen::Vector3f x = m_render_aabb_to_local.row(0) * radius.x();
	Eigen::Vector3f y = m_render_aabb_to_local.row(1) * radius.y();
	Eigen::Vector3f z = m_render_aabb_to_local.row(2) * radius.z();
	Eigen::Matrix<float, 3, 4> rv;
	rv.col(0) = x;
	rv.col(1) = y;
	rv.col(2) = z;
	rv.col(3) = cen;
	if (nerf_space) {
		rv = m_nerf.training.dataset.ngp_matrix_to_nerf(rv, true);
	}
	return rv;
}

void Testbed::set_crop_box(Eigen::Matrix<float, 3, 4> m, bool nerf_space) {
	if (nerf_space) {
		m = m_nerf.training.dataset.nerf_matrix_to_ngp(m, true);
	}
	Eigen::Vector3f radius(m.col(0).norm(), m.col(1).norm(), m.col(2).norm());
	Eigen::Vector3f cen(m.col(3));
	m_render_aabb_to_local.row(0) = m.col(0) / radius.x();
	m_render_aabb_to_local.row(1) = m.col(1) / radius.y();
	m_render_aabb_to_local.row(2) = m.col(2) / radius.z();
	cen = m_render_aabb_to_local * cen;
	m_render_aabb.min = cen - radius;
	m_render_aabb.max = cen + radius;
}

std::vector<Eigen::Vector3f> Testbed::crop_box_corners(bool nerf_space) const {
	Eigen::Matrix<float, 3, 4> m = crop_box(nerf_space);
	std::vector<Eigen::Vector3f> rv(8);
	for (int i = 0; i < 8; ++i) {
		rv[i] = m * Eigen::Vector4f((i & 1) ? 1.f : -1.f, (i & 2) ? 1.f : -1.f, (i & 4) ? 1.f : -1.f, 1.f);
		/* debug print out corners to check math is all lined up */
		if (0) {
			tlog::info() << rv[i].x() << "," << rv[i].y() << "," << rv[i].z() << " [" << i << "]";
			Eigen::Vector3f mn = m_render_aabb.min;
			Eigen::Vector3f mx = m_render_aabb.max;
			Eigen::Matrix3f m = m_render_aabb_to_local.transpose();
			Eigen::Vector3f a;

			a.x() = (i&1) ? mx.x() : mn.x();
			a.y() = (i&2) ? mx.y() : mn.y();
			a.z() = (i&4) ? mx.z() : mn.z();
			a = m * a;
			if (nerf_space) {
				a = m_nerf.training.dataset.ngp_position_to_nerf(a);
			}
			tlog::info() << a.x() << "," << a.y() << "," << a.z() << " [" << i << "]";
		}
	}
	return rv;
}

#ifdef NGP_GUI
bool imgui_colored_button(const char *name, float hue) {
	ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(hue, 0.6f, 0.6f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, 0.7f, 0.7f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(hue, 0.8f, 0.8f));
	bool rv = ImGui::Button(name);
	ImGui::PopStyleColor(3);
	return rv;
}

void Testbed::imgui() {
	// If a GUI interaction causes an error, write that error to the following string and call
	//   ImGui::OpenPopup("Error");
	static std::string imgui_error_string = "";

	m_picture_in_picture_res = 0;
	if (ImGui::Begin("Camera path", 0, ImGuiWindowFlags_NoScrollbar)) {
		if (ImGui::CollapsingHeader("Path manipulation", ImGuiTreeNodeFlags_DefaultOpen)) {
			static char path_filename_buf[128] = "";
			if (path_filename_buf[0] == '\0') {
				snprintf(path_filename_buf, sizeof(path_filename_buf), "%s", get_filename_in_data_path_with_suffix(m_data_path, m_network_config_path, "_cam.json").c_str());
			}

			if (int read = m_camera_path.imgui(
				path_filename_buf,
				m_render_ms.val(),
				m_camera,
				m_slice_plane_z,
				m_scale,
				fov(),
				m_aperture_size,
				m_bounding_radius,
				!m_nerf.training.dataset.xforms.empty() ? m_nerf.training.dataset.xforms[0].start : Matrix<float, 3, 4>::Identity(),
				m_nerf.glow_mode,
				m_nerf.glow_y_cutoff
			)) {
				if (!m_camera_path.rendering) {
					reset_accumulation(true);

					if (m_camera_path.update_cam_from_path) {
						set_camera_from_time(m_camera_path.play_time);

						// A value of larger than 1 indicates that the camera path wants
						// to override camera smoothing.
						if (read > 1) {
							m_smoothed_camera = m_camera;
						}
					} else {
						m_pip_render_surface->reset_accumulation();
					}
				}
			}

			if (!m_camera_path.keyframes.empty()) {
				float w = ImGui::GetContentRegionAvail().x;
				if (m_camera_path.update_cam_from_path) {
					m_picture_in_picture_res = 0;
					ImGui::Image((ImTextureID)(size_t)m_render_textures.front()->texture(), ImVec2(w, w * 9.0f / 16.0f));
				} else {
					m_picture_in_picture_res = (float)std::min((int(w)+31)&(~31), 1920/4);
					ImGui::Image((ImTextureID)(size_t)m_pip_render_texture->texture(), ImVec2(w, w * 9.0f / 16.0f));
				}
			}
		}

		if (!m_camera_path.keyframes.empty() && ImGui::CollapsingHeader("Export video", ImGuiTreeNodeFlags_DefaultOpen)) {
			// Render a video
			if (imgui_colored_button(m_camera_path.rendering ? "Abort rendering" : "Render video", 0.4)) {
				m_camera_path.rendering = !m_camera_path.rendering;

				if (!clear_tmp_dir()) {
					imgui_error_string = "Failed to clear temporary directory 'tmp' to hold rendered images.";
					ImGui::OpenPopup("Error");

					m_camera_path.rendering = false;
				}

				if (m_camera_path.rendering) {
					m_camera_path.render_start_time = std::chrono::steady_clock::now();
					m_camera_path.update_cam_from_path = true;
					m_camera_path.play_time = 0.0f;
					m_camera_path.auto_play_speed = 1.0f;
					m_camera_path.render_frame_idx = 0;

					m_dlss = false;
					m_train = false;

					reset_accumulation(true);
					set_camera_from_time(m_camera_path.play_time);
					m_smoothed_camera = m_camera;
				} else {
					m_camera_path.update_cam_from_path = false;
					m_camera_path.play_time = 0.0f;
					m_camera_path.auto_play_speed = 0.0f;
				}
			}

			if (m_camera_path.rendering) {
				ImGui::SameLine();

				auto elapsed = std::chrono::steady_clock::now() - m_camera_path.render_start_time;

				uint32_t progress = m_camera_path.render_frame_idx * m_camera_path.render_settings.spp + m_render_surfaces.front().spp();
				uint32_t goal = m_camera_path.render_settings.n_frames() * m_camera_path.render_settings.spp;
				auto est_remaining = elapsed * (float)(goal - progress) / std::max(progress, 1u);

				ImGui::Text("%s", fmt::format(
					"Frame {}/{}, Elapsed: {}, Remaining: {}",
					m_camera_path.render_frame_idx+1,
					m_camera_path.render_settings.n_frames(),
					tlog::durationToString(std::chrono::steady_clock::now() - m_camera_path.render_start_time),
					tlog::durationToString(est_remaining)
				).c_str());
			}

			if (m_camera_path.rendering) { ImGui::BeginDisabled(); }

			static char video_filename_buf[1024] = "video.mp4";
			ImGui::InputText("File##Video file path", video_filename_buf, sizeof(video_filename_buf));
			m_camera_path.render_settings.filename = video_filename_buf;

			ImGui::InputInt2("Resolution", &m_camera_path.render_settings.resolution.x());
			ImGui::InputFloat("Duration (seconds)", &m_camera_path.render_settings.duration_seconds);
			ImGui::InputFloat("FPS (frames/second)", &m_camera_path.render_settings.fps);
			ImGui::InputInt("SPP (samples/pixel)", &m_camera_path.render_settings.spp);
			ImGui::SliderInt("Quality", &m_camera_path.render_settings.quality, 0, 10);

			ImGui::SliderFloat("Shutter fraction", &m_camera_path.render_settings.shutter_fraction, 0.0f, 1.0f);

			if (m_camera_path.rendering) { ImGui::EndDisabled(); }
		}
	}
	ImGui::End();


	ImGui::Begin("instant-ngp v" NGP_VERSION);

	size_t n_bytes = tcnn::total_n_bytes_allocated() + g_total_n_bytes_allocated + dlss_allocated_bytes();
	ImGui::Text("Frame: %.2f ms (%.1f FPS); Mem: %s", m_frame_ms.ema_val(), 1000.0f / m_frame_ms.ema_val(), bytes_to_string(n_bytes).c_str());
	bool accum_reset = false;

	if (!m_training_data_available) { ImGui::BeginDisabled(); }

	if (ImGui::CollapsingHeader("Training", m_training_data_available ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
		if (imgui_colored_button(m_train ? "Stop training" : "Start training", 0.4)) {
			set_train(!m_train);
		}
		ImGui::SameLine();
		ImGui::Checkbox("Train encoding", &m_train_encoding);
		ImGui::SameLine();
		ImGui::Checkbox("Train network", &m_train_network);
		ImGui::SameLine();
		ImGui::Checkbox("Random levels", &m_max_level_rand_training);
		if (m_testbed_mode == ETestbedMode::Nerf) {
			ImGui::Checkbox("Train envmap", &m_nerf.training.train_envmap);
			ImGui::SameLine();
			ImGui::Checkbox("Train extrinsics", &m_nerf.training.optimize_extrinsics);
			ImGui::SameLine();
			ImGui::Checkbox("Train exposure", &m_nerf.training.optimize_exposure);
			ImGui::SameLine();
			ImGui::Checkbox("Train distortion", &m_nerf.training.optimize_distortion);
			if (m_nerf.training.dataset.n_extra_learnable_dims) {
				ImGui::Checkbox("Train latent codes", &m_nerf.training.optimize_extra_dims);
			}
			static char opt_extr_filename_buf[1024] = "./trajectory.json";
			static bool export_extrinsics_in_quat_format = true;
			if (imgui_colored_button("Export extrinsics", 0.4f)) {
				m_nerf.training.export_camera_extrinsics(opt_extr_filename_buf, export_extrinsics_in_quat_format);
			}
			ImGui::SameLine();
			ImGui::PushItemWidth(400.f);
			ImGui::InputText("File##Extrinsics file path", opt_extr_filename_buf, sizeof(opt_extr_filename_buf));
			ImGui::PopItemWidth();
			ImGui::SameLine();
			ImGui::Checkbox("Quaternion format", &export_extrinsics_in_quat_format);
		}
		if (imgui_colored_button("Reset training", 0.f)) {
			reload_network_from_file();
		}
		ImGui::SameLine();
		ImGui::DragInt("Seed", (int*)&m_seed, 1.0f, 0, std::numeric_limits<int>::max());
		ImGui::SliderInt("Batch size", (int*)&m_training_batch_size, 1 << 12, 1 << 22, "%d", ImGuiSliderFlags_Logarithmic);
		m_training_batch_size = next_multiple(m_training_batch_size, batch_size_granularity);

		if (m_train) {
			std::vector<std::string> timings;
			if (m_testbed_mode == ETestbedMode::Nerf) {
				timings.emplace_back(fmt::format("Grid: {:.01f}ms", m_training_prep_ms.ema_val()));
			} else {
				timings.emplace_back(fmt::format("Datagen: {:.01f}ms", m_training_prep_ms.ema_val()));
			}

			timings.emplace_back(fmt::format("Training: {:.01f}ms", m_training_ms.ema_val()));
			ImGui::Text("%s", join(timings, ", ").c_str());
		} else {
			ImGui::Text("Training paused");
		}
		if (m_testbed_mode == ETestbedMode::Nerf) {
			ImGui::Text("Rays/batch: %d, Samples/ray: %.2f, Batch size: %d/%d", m_nerf.training.counters_rgb.rays_per_batch, (float)m_nerf.training.counters_rgb.measured_batch_size / (float)m_nerf.training.counters_rgb.rays_per_batch, m_nerf.training.counters_rgb.measured_batch_size, m_nerf.training.counters_rgb.measured_batch_size_before_compaction);
		}
		float elapsed_training = std::chrono::duration<float>(std::chrono::steady_clock::now() - m_training_start_time_point).count();
		ImGui::Text("Steps: %d, Loss: %0.6f (%0.2f dB), Elapsed: %.1fs", m_training_step, m_loss_scalar.ema_val(), linear_to_db(m_loss_scalar.ema_val()), elapsed_training);
		ImGui::PlotLines("loss graph", m_loss_graph.data(), std::min(m_loss_graph_samples, m_loss_graph.size()), (m_loss_graph_samples < m_loss_graph.size()) ? 0 : (m_loss_graph_samples % m_loss_graph.size()), 0, FLT_MAX, FLT_MAX, ImVec2(0, 50.f));

		if (m_testbed_mode == ETestbedMode::Nerf && ImGui::TreeNode("NeRF training options")) {
			ImGui::Checkbox("Random bg color", &m_nerf.training.random_bg_color);
			ImGui::SameLine();
			ImGui::Checkbox("Snap to pixel centers", &m_nerf.training.snap_to_pixel_centers);
			ImGui::SliderFloat("Near distance", &m_nerf.training.near_distance, 0.0f, 1.0f);
			accum_reset |= ImGui::Checkbox("Linear colors", &m_nerf.training.linear_colors);
			ImGui::Combo("Loss", (int*)&m_nerf.training.loss_type, LossTypeStr);
			ImGui::Combo("Depth Loss", (int*)&m_nerf.training.depth_loss_type, LossTypeStr);
			ImGui::Combo("RGB activation", (int*)&m_nerf.rgb_activation, NerfActivationStr);
			ImGui::Combo("Density activation", (int*)&m_nerf.density_activation, NerfActivationStr);
			ImGui::SliderFloat("Cone angle", &m_nerf.cone_angle_constant, 0.0f, 1.0f/128.0f);
			ImGui::SliderFloat("Depth supervision strength", &m_nerf.training.depth_supervision_lambda, 0.f, 1.f);

			// Importance sampling options, but still related to training
			ImGui::Checkbox("Sample focal plane ~error", &m_nerf.training.sample_focal_plane_proportional_to_error);
			ImGui::SameLine();
			ImGui::Checkbox("Sample focal plane ~sharpness", &m_nerf.training.include_sharpness_in_error);
			ImGui::Checkbox("Sample image ~error", &m_nerf.training.sample_image_proportional_to_error);
			ImGui::Text("%dx%d error res w/ %d steps between updates", m_nerf.training.error_map.resolution.x(), m_nerf.training.error_map.resolution.y(), m_nerf.training.n_steps_between_error_map_updates);
			ImGui::Checkbox("Display error overlay", &m_nerf.training.render_error_overlay);
			if (m_nerf.training.render_error_overlay) {
				ImGui::SliderFloat("Error overlay brightness", &m_nerf.training.error_overlay_brightness, 0.f, 1.f);
			}
			ImGui::SliderFloat("Density grid decay", &m_nerf.training.density_grid_decay, 0.f, 1.f,"%.4f");
			ImGui::SliderFloat("Extrinsic L2 reg.", &m_nerf.training.extrinsic_l2_reg, 1e-8f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::SliderFloat("Intrinsic L2 reg.", &m_nerf.training.intrinsic_l2_reg, 1e-8f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::SliderFloat("Exposure L2 reg.", &m_nerf.training.exposure_l2_reg, 1e-8f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::TreePop();
		}

		if (m_testbed_mode == ETestbedMode::Sdf && ImGui::TreeNode("SDF training options")) {
			accum_reset |= ImGui::Checkbox("Use octree for acceleration", &m_sdf.use_triangle_octree);
			accum_reset |= ImGui::Combo("Mesh SDF mode", (int*)&m_sdf.mesh_sdf_mode, MeshSdfModeStr);

			accum_reset |= ImGui::SliderFloat("Surface offset scale", &m_sdf.training.surface_offset_scale, 0.125f, 1024.0f, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);

			if (ImGui::Checkbox("Calculate IoU", &m_sdf.calculate_iou_online)) {
				m_sdf.iou_decay = 0;
			}

			ImGui::SameLine();
			ImGui::Text("%0.6f", m_sdf.iou);
			ImGui::TreePop();
		}

		if (m_testbed_mode == ETestbedMode::Image && ImGui::TreeNode("Image training options")) {
			ImGui::Combo("Training coords", (int*)&m_image.random_mode, RandomModeStr);
			ImGui::Checkbox("Snap to pixel centers", &m_image.training.snap_to_pixel_centers);
			accum_reset |= ImGui::Checkbox("Linear colors", &m_image.training.linear_colors);
			ImGui::TreePop();
		}

		if (m_testbed_mode == ETestbedMode::Volume && ImGui::TreeNode("Volume training options")) {
			accum_reset |= ImGui::SliderFloat("Albedo", &m_volume.albedo, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Scattering", &m_volume.scattering, -2.f, 2.f);
			accum_reset |= ImGui::SliderFloat("Distance scale", &m_volume.inv_distance_scale, 1.f, 100.f, "%.3g", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::TreePop();
		}
	}

	if (!m_training_data_available) { ImGui::EndDisabled(); }

	if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Render", &m_render);
		ImGui::SameLine();

		const auto& render_tex = m_render_surfaces.front();
		std::string spp_string = m_dlss ? std::string{""} : fmt::format("({} spp)", std::max(render_tex.spp(), 1u));
		ImGui::Text(": %.01fms for %dx%d %s", m_render_ms.ema_val(), render_tex.in_resolution().x(), render_tex.in_resolution().y(), spp_string.c_str());

		if (m_dlss_supported) {
			if (!m_single_view) {
				ImGui::BeginDisabled();
				m_dlss = false;
			}

			if (ImGui::Checkbox("DLSS", &m_dlss)) {
				accum_reset = true;
			}

			if (render_tex.dlss()) {
				ImGui::SameLine();
				ImGui::Text("(automatic quality setting: %s)", DlssQualityStrArray[(int)render_tex.dlss()->quality()]);
				ImGui::SliderFloat("DLSS sharpening", &m_dlss_sharpening, 0.0f, 1.0f, "%.02f");
			}

			if (!m_single_view) {
				ImGui::EndDisabled();
			}
		}

		ImGui::Checkbox("Dynamic resolution", &m_dynamic_res);
		if (ImGui::Checkbox("VSync", &m_vsync)) {
			glfwSwapInterval(m_vsync ? 1 : 0);
		}
		ImGui::SliderFloat("Target FPS", &m_dynamic_res_target_fps, 2.0f, 144.0f, "%.01f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
		ImGui::SliderInt("Max spp", &m_max_spp, 0, 1024, "%d", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);

		if (!m_dynamic_res) {
			ImGui::SliderInt("Fixed resolution factor", &m_fixed_res_factor, 8, 64);
		}

		if (m_testbed_mode == ETestbedMode::Nerf && m_nerf.training.dataset.has_light_dirs) {
			Vector3f light_dir = m_nerf.light_dir.normalized();
			if (ImGui::TreeNodeEx("Light Dir (Polar)", ImGuiTreeNodeFlags_DefaultOpen)) {
				float phi = atan2f(m_nerf.light_dir.x(), m_nerf.light_dir.z());
				float theta = asinf(m_nerf.light_dir.y());
				bool spin = ImGui::SliderFloat("Light Dir Theta", &theta, -PI() / 2.0f, PI() / 2.0f);
				spin |= ImGui::SliderFloat("Light Dir Phi", &phi, -PI(), PI());
				if (spin) {
					float sin_phi, cos_phi;
					sincosf(phi, &sin_phi, &cos_phi);
					float cos_theta=cosf(theta);
					m_nerf.light_dir = {sin_phi * cos_theta,sinf(theta),cos_phi * cos_theta};
					accum_reset = true;
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Light Dir (Cartesian)")) {
				accum_reset |= ImGui::SliderFloat("Light Dir X", ((float*)(&m_nerf.light_dir)) + 0, -1.0f, 1.0f);
				accum_reset |= ImGui::SliderFloat("Light Dir Y", ((float*)(&m_nerf.light_dir)) + 1, -1.0f, 1.0f);
				accum_reset |= ImGui::SliderFloat("Light Dir Z", ((float*)(&m_nerf.light_dir)) + 2, -1.0f, 1.0f);
				ImGui::TreePop();
			}
		}
		if (m_testbed_mode == ETestbedMode::Nerf && m_nerf.training.dataset.n_extra_learnable_dims) {
			accum_reset |= ImGui::SliderInt("training image latent code for inference", (int*)&m_nerf.extra_dim_idx_for_inference, 0, m_nerf.training.dataset.n_images-1);
		}
		accum_reset |= ImGui::Combo("Render mode", (int*)&m_render_mode, RenderModeStr);
		if (m_testbed_mode == ETestbedMode::Nerf)  {
			accum_reset |= ImGui::Combo("Groundtruth Render mode", (int*)&m_ground_truth_render_mode, GroundTruthRenderModeStr);
			accum_reset |= ImGui::SliderFloat("Groundtruth Alpha", &m_ground_truth_alpha, 0.0f, 1.0f, "%.02f", ImGuiSliderFlags_AlwaysClamp);
		}
		accum_reset |= ImGui::Combo("Color space", (int*)&m_color_space, ColorSpaceStr);
		accum_reset |= ImGui::Combo("Tonemap curve", (int*)&m_tonemap_curve, TonemapCurveStr);
		accum_reset |= ImGui::ColorEdit4("Background", &m_background_color[0]);
		if (ImGui::SliderFloat("Exposure", &m_exposure, -5.f, 5.f)) {
			set_exposure(m_exposure);
		}

		accum_reset |= ImGui::Checkbox("Snap to pixel centers", &m_snap_to_pixel_centers);

		float max_diam = (m_aabb.max-m_aabb.min).maxCoeff();
		float render_diam = (m_render_aabb.max-m_render_aabb.min).maxCoeff();
		float old_render_diam = render_diam;
		if (ImGui::SliderFloat("Crop size", &render_diam, 0.1f, max_diam, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat)) {
			accum_reset = true;
			if (old_render_diam > 0.f && render_diam > 0.f) {
				const Vector3f center = (m_render_aabb.max + m_render_aabb.min) * 0.5f;
				float scale = render_diam / old_render_diam;
				m_render_aabb.max = ((m_render_aabb.max-center) * scale + center).cwiseMin(m_aabb.max);
				m_render_aabb.min = ((m_render_aabb.min-center) * scale + center).cwiseMax(m_aabb.min);
			}
		}

		if (ImGui::TreeNode("Crop aabb")) {
			m_edit_render_aabb = true;
			accum_reset |= ImGui::SliderFloat("Min x", ((float*)&m_render_aabb.min)+0, m_aabb.min.x(), m_render_aabb.max.x(), "%.3f");
			accum_reset |= ImGui::SliderFloat("Min y", ((float*)&m_render_aabb.min)+1, m_aabb.min.y(), m_render_aabb.max.y(), "%.3f");
			accum_reset |= ImGui::SliderFloat("Min z", ((float*)&m_render_aabb.min)+2, m_aabb.min.z(), m_render_aabb.max.z(), "%.3f");
			ImGui::Separator();
			accum_reset |= ImGui::SliderFloat("Max x", ((float*)&m_render_aabb.max)+0, m_render_aabb.min.x(), m_aabb.max.x(), "%.3f");
			accum_reset |= ImGui::SliderFloat("Max y", ((float*)&m_render_aabb.max)+1, m_render_aabb.min.y(), m_aabb.max.y(), "%.3f");
			accum_reset |= ImGui::SliderFloat("Max z", ((float*)&m_render_aabb.max)+2, m_render_aabb.min.z(), m_aabb.max.z(), "%.3f");
			ImGui::Separator();
			Vector3f diag = m_render_aabb.diag();
			bool edit_diag = false;
			float max_diag = m_aabb.diag().maxCoeff();
			edit_diag |= ImGui::SliderFloat("Size x", ((float*)&diag)+0, 0.001f, max_diag, "%.3f");
			edit_diag |= ImGui::SliderFloat("Size y", ((float*)&diag)+1, 0.001f, max_diag, "%.3f");
			edit_diag |= ImGui::SliderFloat("Size z", ((float*)&diag)+2, 0.001f, max_diag, "%.3f");
			if (edit_diag) {
				accum_reset = true;
				Vector3f cen = m_render_aabb.center();
				m_render_aabb = BoundingBox(cen - diag * 0.5f, cen + diag * 0.5f);
			}
			if (ImGui::Button("Reset")) {
				accum_reset = true;
				m_render_aabb = m_aabb;
				m_render_aabb_to_local = Matrix3f::Identity();
			}
			ImGui::SameLine();
			if (ImGui::Button("Reset Rotation Only")) {
				accum_reset = true;
				Eigen::Vector3f world_cen = m_render_aabb_to_local.transpose() * m_render_aabb.center();
				m_render_aabb_to_local = Matrix3f::Identity();
				Eigen::Vector3f new_cen = m_render_aabb_to_local * world_cen;
				Eigen::Vector3f old_cen = m_render_aabb.center();
				m_render_aabb.min += new_cen - old_cen;
				m_render_aabb.max += new_cen - old_cen;
			}
			if (/*m_visualize_unit_cube*/ 1) {
				if (ImGui::RadioButton("Translate", m_camera_path.m_gizmo_op == ImGuizmo::TRANSLATE))
					m_camera_path.m_gizmo_op = ImGuizmo::TRANSLATE;
				ImGui::SameLine();
				if (ImGui::RadioButton("Rotate", m_camera_path.m_gizmo_op == ImGuizmo::ROTATE))
					m_camera_path.m_gizmo_op = ImGuizmo::ROTATE;
			}
			ImGui::TreePop();
		} else {
			m_edit_render_aabb = false;
		}

		if (m_testbed_mode == ETestbedMode::Nerf && ImGui::TreeNode("NeRF rendering options")) {
			accum_reset |= ImGui::Checkbox("Apply lens distortion", &m_nerf.render_with_lens_distortion);

			if (m_nerf.render_with_lens_distortion) {
				accum_reset |= ImGui::Combo("Lens mode", (int*)&m_nerf.render_lens.mode, LensModeStr);
				if (m_nerf.render_lens.mode == ELensMode::OpenCV) {
					accum_reset |= ImGui::InputFloat("k1", &m_nerf.render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k2", &m_nerf.render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("p1", &m_nerf.render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("p2", &m_nerf.render_lens.params[3], 0.f, 0.f, "%.5f");
				} else if (m_nerf.render_lens.mode == ELensMode::OpenCVFisheye) {
					accum_reset |= ImGui::InputFloat("k1", &m_nerf.render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k2", &m_nerf.render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k3", &m_nerf.render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k4", &m_nerf.render_lens.params[3], 0.f, 0.f, "%.5f");
				} else if (m_nerf.render_lens.mode == ELensMode::FTheta) {
					accum_reset |= ImGui::InputFloat("width", &m_nerf.render_lens.params[5], 0.f, 0.f, "%.0f");
					accum_reset |= ImGui::InputFloat("height", &m_nerf.render_lens.params[6], 0.f, 0.f, "%.0f");
					accum_reset |= ImGui::InputFloat("f_theta p0", &m_nerf.render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p1", &m_nerf.render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p2", &m_nerf.render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p3", &m_nerf.render_lens.params[3], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p4", &m_nerf.render_lens.params[4], 0.f, 0.f, "%.5f");
				}
			}

			accum_reset |= ImGui::SliderFloat("Min transmittance", &m_nerf.render_min_transmittance, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::TreePop();
		}

		if (m_testbed_mode == ETestbedMode::Sdf && ImGui::TreeNode("SDF rendering options")) {
			accum_reset |= ImGui::Combo("Ground Truth Rendering Mode", (int*)&m_sdf.groundtruth_mode,
				"Raytraced Mesh\0"
				"Sphere Traced Mesh\0"
				"SDF Bricks\0"
			);

			if (m_sdf.groundtruth_mode == ESDFGroundTruthMode::SDFBricks) {
				accum_reset |= ImGui::SliderInt("Brick octree Level", (int*)&m_sdf.brick_level, 1, 10);
				accum_reset |= ImGui::Checkbox("Brick normals track octree Level", &m_sdf.brick_smooth_normals);
				accum_reset |= ImGui::SliderInt("Brick quantize Bits", (int*)&m_sdf.brick_quantise_bits, 0, 16);
			}

			accum_reset |= ImGui::Checkbox("Analytic normals", &m_sdf.analytic_normals);

			accum_reset |= ImGui::SliderFloat("Normals epsilon", &m_sdf.fd_normals_epsilon, 0.00001f, 0.1f, "%.6g", ImGuiSliderFlags_Logarithmic);
			accum_reset |= ImGui::SliderFloat("Maximum distance", &m_sdf.maximum_distance, 0.00001f, 0.1f, "%.6g", ImGuiSliderFlags_Logarithmic);
			accum_reset |= ImGui::SliderFloat("Shadow sharpness", &m_sdf.shadow_sharpness, 0.1f, 2048.0f, "%.6g", ImGuiSliderFlags_Logarithmic);

			accum_reset |= ImGui::SliderFloat("Inflate (offset the zero set)", &m_sdf.zero_offset, -0.25f, 0.25f);
			accum_reset |= ImGui::SliderFloat("Distance scale", &m_sdf.distance_scale, 0.25f, 1.f);

			ImGui::TreePop();
		}

		if (m_testbed_mode == ETestbedMode::Image && ImGui::TreeNode("Image rendering options")) {
			static bool quantize_to_byte = false;
			static float mse = 0.0f;

			if (imgui_colored_button("Compute PSNR", 0.4)) {
				mse = compute_image_mse(quantize_to_byte);
			}

			float psnr = -10.0f * std::log(mse) / std::log(10.0f);

			ImGui::SameLine();
			ImGui::Text("%0.6f", psnr);
			ImGui::SameLine();
			ImGui::Checkbox("Quantize", &quantize_to_byte);

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Debug visualization")) {
			ImGui::Checkbox("Visualize unit cube", &m_visualize_unit_cube);
			if (m_testbed_mode == ETestbedMode::Nerf) {
				ImGui::SameLine();
				ImGui::Checkbox("Visualize cameras", &m_nerf.visualize_cameras);
				accum_reset |= ImGui::SliderInt("Show acceleration", &m_nerf.show_accel, -1, 7);
			}

			if (!m_single_view) { ImGui::BeginDisabled(); }
			if (ImGui::SliderInt("Visualized dimension", &m_visualized_dimension, -1, (int)network_width(m_visualized_layer)-1)) {
				set_visualized_dim(m_visualized_dimension);
			}
			if (!m_single_view) { ImGui::EndDisabled(); }

			if (ImGui::SliderInt("Visualized layer", &m_visualized_layer, 0, (int)network_num_forward_activations()-1)) {
				set_visualized_layer(m_visualized_layer);
			}
			if (ImGui::Checkbox("Single view", &m_single_view)) {
				set_visualized_dim(-1);
				accum_reset = true;
			}

			if (m_testbed_mode == ETestbedMode::Nerf) {
				if (ImGui::Button("First")) {
					first_training_view();
				}
				ImGui::SameLine();
				if (ImGui::Button("Previous")) {
					previous_training_view();
				}
				ImGui::SameLine();
				if (ImGui::Button("Next")) {
					next_training_view();
				}
				ImGui::SameLine();
				if (ImGui::Button("Last")) {
					last_training_view();
				}
				ImGui::SameLine();
				ImGui::Text("%s", m_nerf.training.dataset.paths.at(m_nerf.training.view).c_str());

				if (ImGui::SliderInt("Training view", &m_nerf.training.view, 0, (int)m_nerf.training.dataset.n_images-1)) {
					set_camera_to_training_view(m_nerf.training.view);
					accum_reset = true;
				}
				ImGui::PlotLines("Training view error", m_nerf.training.error_map.pmf_img_cpu.data(), m_nerf.training.error_map.pmf_img_cpu.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0, 60.f));

				if (m_nerf.training.optimize_exposure) {
					std::vector<float> exposures(m_nerf.training.dataset.n_images);
					for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
						exposures[i] = m_nerf.training.cam_exposure[i].variable().x();
					}

					ImGui::PlotLines("Training view exposures", exposures.data(), exposures.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 60.f));
				}

				if (ImGui::SliderInt("Glow mode", &m_nerf.glow_mode, 0, 16)) {
					accum_reset = true;
				}

				if (m_nerf.glow_mode && ImGui::SliderFloat("Glow height", &m_nerf.glow_y_cutoff, -2.f, 3.f)) {
					accum_reset = true;
				}
			}

			ImGui::TreePop();
		}
	}

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::SliderFloat("Aperture size", &m_aperture_size, 0.0f, 0.1f)) {
			m_dlss = false;
			accum_reset = true;
		}
		float local_fov = fov();
		if (ImGui::SliderFloat("Field of view", &local_fov, 0.0f, 120.0f)) {
			set_fov(local_fov);
			accum_reset = true;
		}
		accum_reset |= ImGui::SliderFloat("Zoom", &m_zoom, 1.f, 10.f);
		if (m_testbed_mode == ETestbedMode::Sdf) {
			accum_reset |= ImGui::Checkbox("Floor", &m_floor_enable);
			ImGui::SameLine();
		}

		ImGui::Checkbox("First person controls", &m_fps_camera);
		ImGui::Checkbox("Smooth camera motion", &m_camera_smoothing);
		ImGui::Checkbox("Autofocus", &m_autofocus);

		if (ImGui::TreeNode("Advanced camera settings")) {
			accum_reset |= ImGui::SliderFloat2("Screen center", &m_screen_center.x(), 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat2("Parallax shift", &m_parallax_shift.x(), -1.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Slice / focus depth", &m_slice_plane_z, -m_bounding_radius, m_bounding_radius);
			accum_reset |= ImGui::SliderFloat("Render near distance", &m_render_near_distance, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			char buf[2048];
			Vector3f v = view_dir();
			Vector3f p = look_at();
			Vector3f s = m_sun_dir;
			Vector3f u = m_up_dir;
			Array4f b = m_background_color;
			snprintf(buf, sizeof(buf),
				"testbed.background_color = [%0.3f, %0.3f, %0.3f, %0.3f]\n"
				"testbed.exposure = %0.3f\n"
				"testbed.sun_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.up_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.view_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.look_at = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.scale = %0.3f\n"
				"testbed.fov,testbed.aperture_size,testbed.slice_plane_z = %0.3f,%0.3f,%0.3f\n"
				"testbed.autofocus_target = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.autofocus = %s\n\n"
				, b.x(), b.y(), b.z(), b.w()
				, m_exposure
				, s.x(), s.y(), s.z()
				, u.x(), u.y(), u.z()
				, v.x(), v.y(), v.z()
				, p.x(), p.y(), p.z()
				, scale()
				, fov(), m_aperture_size, m_slice_plane_z
				, m_autofocus_target.x(), m_autofocus_target.y(), m_autofocus_target.z()
				, m_autofocus ? "True" : "False"
			);

			if (m_testbed_mode == ETestbedMode::Sdf) {
				size_t n = strlen(buf);
				snprintf(buf+n, sizeof(buf)-n,
					"testbed.sdf.shadow_sharpness = %0.3f\n"
					"testbed.sdf.analytic_normals = %s\n"
					"testbed.sdf.use_triangle_octree = %s\n\n"
					"testbed.sdf.brdf.metallic = %0.3f\n"
					"testbed.sdf.brdf.subsurface = %0.3f\n"
					"testbed.sdf.brdf.specular = %0.3f\n"
					"testbed.sdf.brdf.roughness = %0.3f\n"
					"testbed.sdf.brdf.sheen = %0.3f\n"
					"testbed.sdf.brdf.clearcoat = %0.3f\n"
					"testbed.sdf.brdf.clearcoat_gloss = %0.3f\n"
					"testbed.sdf.brdf.basecolor = [%0.3f,%0.3f,%0.3f]\n\n"
					, m_sdf.shadow_sharpness
					, m_sdf.analytic_normals ? "True" : "False"
					, m_sdf.use_triangle_octree ? "True" : "False"
					, m_sdf.brdf.metallic
					, m_sdf.brdf.subsurface
					, m_sdf.brdf.specular
					, m_sdf.brdf.roughness
					, m_sdf.brdf.sheen
					, m_sdf.brdf.clearcoat
					, m_sdf.brdf.clearcoat_gloss
					, m_sdf.brdf.basecolor.x()
					, m_sdf.brdf.basecolor.y()
					, m_sdf.brdf.basecolor.z()
				);
			}
			ImGui::InputTextMultiline("Params", buf, sizeof(buf));
			ImGui::TreePop();
		}
	}

	if (ImGui::CollapsingHeader("Snapshot")) {
		static char snapshot_filename_buf[128] = "";
		if (snapshot_filename_buf[0] == '\0') {
			snprintf(snapshot_filename_buf, sizeof(snapshot_filename_buf), "%s", get_filename_in_data_path_with_suffix(m_data_path, m_network_config_path, ".msgpack").c_str());
		}

		ImGui::Text("Snapshot");
		ImGui::SameLine();
		if (ImGui::Button("Save")) {
			save_snapshot(snapshot_filename_buf, m_include_optimizer_state_in_snapshot);
		}
		ImGui::SameLine();
		if (ImGui::Button("Load")) {
			try {
				load_snapshot(snapshot_filename_buf);
			} catch (std::exception& e) {
				imgui_error_string = fmt::format("Failed to load snapshot: {}", e.what());
				ImGui::OpenPopup("Error");
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Dump parameters as images")) {
			dump_parameters_as_images(m_trainer->params(), "params");
		}

		ImGui::SameLine();
		ImGui::Checkbox("w/ optimizer state", &m_include_optimizer_state_in_snapshot);
		ImGui::InputText("File##Snapshot file path", snapshot_filename_buf, sizeof(snapshot_filename_buf));
	}

	if (m_testbed_mode == ETestbedMode::Nerf || m_testbed_mode == ETestbedMode::Sdf) {
		if (ImGui::CollapsingHeader("Export mesh / volume / slices")) {
			static bool flip_y_and_z_axes = false;
			static float density_range = 4.f;
			BoundingBox aabb = (m_testbed_mode == ETestbedMode::Nerf) ? m_render_aabb : m_aabb;

			auto res3d = get_marching_cubes_res(m_mesh.res, aabb);

			// If we use an octree to fit the SDF only close to the surface, then marching cubes will not work (SDF not defined everywhere)
			bool disable_marching_cubes = m_testbed_mode == ETestbedMode::Sdf && (m_sdf.uses_takikawa_encoding || m_sdf.use_triangle_octree);
			if (disable_marching_cubes) { ImGui::BeginDisabled(); }

			if (imgui_colored_button("Mesh it!", 0.4f)) {
				marching_cubes(res3d, aabb, m_render_aabb_to_local, m_mesh.thresh);
				m_nerf.render_with_lens_distortion = false;
			}
			if (m_mesh.indices.size()>0) {
				ImGui::SameLine();
				if (imgui_colored_button("Clear Mesh", 0.f)) {
					m_mesh.clear();
				}
			}

			if (disable_marching_cubes) { ImGui::EndDisabled(); }

			ImGui::SameLine();

			if (imgui_colored_button("Save density PNG", -0.7f)) {
				Testbed::compute_and_save_png_slices(m_data_path.str().c_str(), m_mesh.res, {}, m_mesh.thresh, density_range, flip_y_and_z_axes);
			}

			if (m_testbed_mode == ETestbedMode::Nerf) {
				ImGui::SameLine();
				if (imgui_colored_button("Save RGBA PNG sequence", 0.2f)) {
					auto effective_view_dir = flip_y_and_z_axes ? Vector3f{0.0f, 1.0f, 0.0f} : Vector3f{0.0f, 0.0f, 1.0f};
					// Depth of 0.01f is arbitrarily chosen to produce a visually interpretable range of alpha values.
					// Alternatively, if the true transparency of a given voxel is desired, one could use the voxel size,
					// the voxel diagonal, or some form of expected ray length through the voxel, given random directions.
					GPUMemory<Array4f> rgba = get_rgba_on_grid(res3d, effective_view_dir, true, 0.01f);
					auto dir = m_data_path / "rgba_slices";
					if (!dir.exists()) {
						fs::create_directory(dir);
					}
					save_rgba_grid_to_png_sequence(rgba, dir.str().c_str(), res3d, flip_y_and_z_axes);
				}
				if (imgui_colored_button("Save raw volumes", 0.4f)) {
					auto effective_view_dir = flip_y_and_z_axes ? Vector3f{0.0f, 1.0f, 0.0f} : Vector3f{0.0f, 0.0f, 1.0f};
					auto old_local = m_render_aabb_to_local;
					auto old_aabb = m_render_aabb;
					m_render_aabb_to_local = Eigen::Matrix3f::Identity();
					auto dir = m_data_path / "volume_raw";
					if (!dir.exists()) {
						fs::create_directory(dir);
					}
					for (int cascade = 0; (1<<cascade)<= m_aabb.diag().x()+0.5f; ++cascade) {
						float radius = (1<<cascade) * 0.5f;
						m_render_aabb = BoundingBox(Eigen::Vector3f::Constant(0.5f-radius), Eigen::Vector3f::Constant(0.5f+radius));
						// Dump raw density values that the user can then convert to alpha as they please.
						GPUMemory<Array4f> rgba = get_rgba_on_grid(res3d, effective_view_dir, true, 0.0f, true);
						save_rgba_grid_to_raw_file(rgba, dir.str().c_str(), res3d, flip_y_and_z_axes, cascade);
					}
					m_render_aabb_to_local = old_local;
					m_render_aabb = old_aabb;
				}
			}

			ImGui::SameLine();
			ImGui::Checkbox("Swap Y&Z", &flip_y_and_z_axes);
			ImGui::SliderFloat("PNG Density Range", &density_range, 0.001f, 8.f);

			static char obj_filename_buf[128] = "";
			ImGui::SliderInt("Res:", &m_mesh.res, 16, 2048, "%d", ImGuiSliderFlags_Logarithmic);
			ImGui::SameLine();

			ImGui::Text("%dx%dx%d", res3d.x(), res3d.y(), res3d.z());
			if (obj_filename_buf[0] == '\0') {
				snprintf(obj_filename_buf, sizeof(obj_filename_buf), "%s", get_filename_in_data_path_with_suffix(m_data_path, m_network_config_path, ".obj").c_str());
			}
			float thresh_range = (m_testbed_mode == ETestbedMode::Sdf) ? 0.5f : 10.f;
			ImGui::SliderFloat("MC density threshold",&m_mesh.thresh, -thresh_range, thresh_range);
			ImGui::Combo("Mesh render mode", (int*)&m_mesh_render_mode, "Off\0Vertex Colors\0Vertex Normals\0Face IDs\0");
			ImGui::Checkbox("Unwrap mesh", &m_mesh.unwrap);
			if (uint32_t tricount = m_mesh.indices.size()/3) {
				ImGui::InputText("##OBJFile", obj_filename_buf, sizeof(obj_filename_buf));
				if (ImGui::Button("Save it!")) {
					save_mesh(m_mesh.verts, m_mesh.vert_normals, m_mesh.vert_colors, m_mesh.indices, obj_filename_buf, m_mesh.unwrap, m_nerf.training.dataset.scale, m_nerf.training.dataset.offset);
				}
				ImGui::SameLine();
				ImGui::Text("Mesh has %d triangles\n", tricount);
				ImGui::Checkbox("Optimize mesh", &m_mesh.optimize_mesh);
				ImGui::SliderFloat("Laplacian smoothing", &m_mesh.smooth_amount, 0.f, 2048.f);
				ImGui::SliderFloat("Density push", &m_mesh.density_amount, 0.f, 128.f);
				ImGui::SliderFloat("Inflate", &m_mesh.inflate_amount, 0.f, 128.f);
			}
		}
	}

	if (m_testbed_mode == ETestbedMode::Sdf) {
		if (ImGui::CollapsingHeader("BRDF parameters")) {
			accum_reset |= ImGui::ColorEdit3("Base color", (float*)&m_sdf.brdf.basecolor );
			accum_reset |= ImGui::SliderFloat("Roughness", &m_sdf.brdf.roughness, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Specular", &m_sdf.brdf.specular, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Metallic", &m_sdf.brdf.metallic, 0.f, 1.f);
			ImGui::Separator();
			accum_reset |= ImGui::SliderFloat("Subsurface", &m_sdf.brdf.subsurface, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Sheen", &m_sdf.brdf.sheen, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Clearcoat", &m_sdf.brdf.clearcoat, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Clearcoat gloss", &m_sdf.brdf.clearcoat_gloss, 0.f, 1.f);
		}
		m_sdf.brdf.ambientcolor = (m_background_color * m_background_color).head<3>();
	}

	if (ImGui::CollapsingHeader("Histograms of encoding parameters")) {
		ImGui::Checkbox("Gather histograms", &m_gather_histograms);

		static float minlevel = 0.f;
		static float maxlevel = 1.f;
		if (ImGui::SliderFloat("Max level", &maxlevel, 0.f, 1.f))
			set_max_level(maxlevel);
		if (ImGui::SliderFloat("##Min level", &minlevel, 0.f, 1.f))
			set_min_level(minlevel);
		ImGui::SameLine();
		ImGui::Text("%0.1f%% values snapped to 0", m_quant_percent);

		std::vector<float> f(m_num_levels);


		// Hashgrid statistics
		for (int i = 0; i < m_num_levels; ++i) {
			f[i] = m_level_stats[i].mean();
		}
		ImGui::PlotHistogram("Grid means", f.data(), m_num_levels, 0, "means", FLT_MAX, FLT_MAX, ImVec2(0, 60.f));
		for (int i = 0; i < m_num_levels; ++i) {
			f[i] = m_level_stats[i].sigma();
		}
		ImGui::PlotHistogram("Grid sigmas", f.data(), m_num_levels, 0, "sigma", FLT_MAX, FLT_MAX, ImVec2(0, 60.f));
		ImGui::Separator();


		// Histogram of trained hashgrid params
		ImGui::SliderInt("Show details for level", &m_histo_level, 0, m_num_levels - 1);
		if (m_histo_level < m_num_levels) {
			LevelStats& s = m_level_stats[m_histo_level];
			static bool excludezero = false;
			if (excludezero)
				m_histo[128] = 0.f;
			ImGui::PlotHistogram("Values histogram", m_histo, 257, 0, "", FLT_MAX, FLT_MAX, ImVec2(0, 120.f));
			ImGui::SliderFloat("Histogram horizontal scale", &m_histo_scale, 0.01f, 2.f);
			ImGui::Checkbox("Exclude 'zero' from histogram", &excludezero);
			ImGui::Text("Range: %0.5f - %0.5f", s.min, s.max);
			ImGui::Text("Mean: %0.5f Sigma: %0.5f", s.mean(), s.sigma());
			ImGui::Text("Num Zero: %d (%0.1f%%)", s.numzero, s.fraczero() * 100.f);
		}
	}

	if (ImGui::BeginPopupModal("Error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("%s", imgui_error_string.c_str());
		if (ImGui::Button("OK", ImVec2(120, 0))) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	if (accum_reset) {
		reset_accumulation();
	}

	if (ImGui::Button("Go to python REPL")) {
		m_want_repl = true;
	}

	ImGui::End();
}

void Testbed::visualize_nerf_cameras(ImDrawList* list, const Matrix<float, 4, 4>& world2proj) {
	for (int i = 0; i < m_nerf.training.n_images_for_training; ++i) {
		auto res = m_nerf.training.dataset.metadata[i].resolution;
		float aspect = float(res.x())/float(res.y());
		auto current_xform = get_xform_given_rolling_shutter(m_nerf.training.transforms[i], m_nerf.training.dataset.metadata[i].rolling_shutter, Vector2f{0.5f, 0.5f}, 0.0f);
		visualize_nerf_camera(list, world2proj, m_nerf.training.dataset.xforms[i].start, aspect, 0x40ffff40);
		visualize_nerf_camera(list, world2proj, m_nerf.training.dataset.xforms[i].end, aspect, 0x40ffff40);
		visualize_nerf_camera(list, world2proj, current_xform, aspect, 0x80ffffff);

		// Visualize near distance
		add_debug_line(list, world2proj, current_xform.col(3), current_xform.col(3) + current_xform.col(2) * m_nerf.training.near_distance, 0x20ffffff);
	}

}

void Testbed::draw_visualizations(ImDrawList* list, const Matrix<float, 3, 4>& camera_matrix) {
	// Visualize 3D cameras for SDF or NeRF use cases
	if (m_testbed_mode != ETestbedMode::Image) {
		Matrix<float, 4, 4> world2view, view2world, view2proj, world2proj;
		view2world.setIdentity();
		view2world.block<3,4>(0,0) = camera_matrix;

		auto focal = calc_focal_length(Vector2i::Ones(), m_fov_axis, m_zoom);
		float zscale = 1.0f / focal[m_fov_axis];

		float xyscale = (float)m_window_res[m_fov_axis];
		Vector2f screen_center = render_screen_center();
		view2proj <<
			xyscale, 0,       (float)m_window_res.x()*screen_center.x()*zscale, 0,
			0,       xyscale, (float)m_window_res.y()*screen_center.y()*zscale, 0,
			0,       0,       1,                                                0,
			0,       0,       zscale,                                           0;

		world2view = view2world.inverse();
		world2proj = view2proj * world2view;
		float aspect = (float)m_window_res.x() / (float)m_window_res.y();

		// Visualize NeRF training poses
		if (m_testbed_mode == ETestbedMode::Nerf) {
			if (m_nerf.visualize_cameras) {
				visualize_nerf_cameras(list, world2proj);
			}
		}

		if (m_visualize_unit_cube) {
			visualize_unit_cube(list, world2proj, Eigen::Vector3f::Constant(0.f), Eigen::Vector3f::Constant(1.f), Eigen::Matrix3f::Identity());
		}
		if (m_edit_render_aabb) {
			if (m_testbed_mode == ETestbedMode::Nerf) {
				visualize_unit_cube(list, world2proj, m_render_aabb.min, m_render_aabb.max, m_render_aabb_to_local);
				ImGuiIO& io = ImGui::GetIO();
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
				ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

				Eigen::Matrix4f matrix=Eigen::Matrix4f::Identity();
				matrix.block<3,3>(0,0) = m_render_aabb_to_local.transpose();
				Eigen::Vector3f cen = m_render_aabb_to_local.transpose() * m_render_aabb.center();
				matrix.block<3,4>(0,0).col(3) = cen;
				if (ImGuizmo::Manipulate((const float*)&world2view, (const float*)&view2proj_guizmo, m_camera_path.m_gizmo_op, ImGuizmo::LOCAL, (float*)&matrix, NULL, NULL)) {
					m_render_aabb_to_local = matrix.block<3,3>(0,0).transpose();
					Eigen::Vector3f new_cen = m_render_aabb_to_local * matrix.block<3,4>(0,0).col(3);
					Eigen::Vector3f old_cen = m_render_aabb.center();
					m_render_aabb.min += new_cen - old_cen;
					m_render_aabb.max += new_cen - old_cen;
					reset_accumulation();
				}
			}
		}

		if (m_camera_path.imgui_viz(list, view2proj, world2proj, world2view, focal, aspect)) {
			m_pip_render_surface->reset_accumulation();
		}
	}
}

void glfw_error_callback(int error, const char* description) {
	tlog::error() << "GLFW error #" << error << ": " << description;
}

bool Testbed::keyboard_event() {
	if (ImGui::GetIO().WantCaptureKeyboard) {
		return false;
	}

	if (m_keyboard_event_callback && m_keyboard_event_callback()) {
		return false;
	}

	for (int idx = 0; idx < std::min((int)ERenderMode::NumRenderModes, 10); ++idx) {
		char c[] = { "1234567890" };
		if (ImGui::IsKeyPressed(c[idx])) {
			m_render_mode = (ERenderMode)idx;
			reset_accumulation();
		}
	}

	bool shift = ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Shift;

	if (ImGui::IsKeyPressed('Z')) {
		m_camera_path.m_gizmo_op = ImGuizmo::TRANSLATE;
	}

	if (ImGui::IsKeyPressed('X')) {
		m_camera_path.m_gizmo_op = ImGuizmo::ROTATE;
	}

	if (ImGui::IsKeyPressed('E')) {
		set_exposure(m_exposure + (shift ? -0.5f : 0.5f));
		redraw_next_frame();
	}

	if (ImGui::IsKeyPressed('R')) {
		if (shift) {
			reset_camera();
		} else {
			reload_network_from_file();
		}
	}

	if (m_training_data_available) {
		if (ImGui::IsKeyPressed('O')) {
			m_nerf.training.render_error_overlay = !m_nerf.training.render_error_overlay;
		}

		if (ImGui::IsKeyPressed('G')) {
			m_render_ground_truth = !m_render_ground_truth;
			reset_accumulation();
			if (m_render_ground_truth) {
				m_nerf.training.view = find_best_training_view(m_nerf.training.view);
			}
		}

		if (ImGui::IsKeyPressed('T')) {
			set_train(!m_train);
		}
	}

	if (ImGui::IsKeyPressed('.')) {
		if (m_single_view) {
			if (m_visualized_dimension == m_network->width(m_visualized_layer)-1 && m_visualized_layer < m_network->num_forward_activations()-1) {
				set_visualized_layer(std::max(0, std::min((int)m_network->num_forward_activations()-1, m_visualized_layer+1)));
				set_visualized_dim(0);
			} else {
				set_visualized_dim(std::max(-1, std::min((int)m_network->width(m_visualized_layer)-1, m_visualized_dimension+1)));
			}
		} else {
			set_visualized_layer(std::max(0, std::min((int)m_network->num_forward_activations()-1, m_visualized_layer+1)));
		}
	}

	if (ImGui::IsKeyPressed(',')) {
		if (m_single_view) {
			if (m_visualized_dimension == 0 && m_visualized_layer > 0) {
				set_visualized_layer(std::max(0, std::min((int)m_network->num_forward_activations()-1, m_visualized_layer-1)));
				set_visualized_dim(m_network->width(m_visualized_layer)-1);
			} else {
				set_visualized_dim(std::max(-1, std::min((int)m_network->width(m_visualized_layer)-1, m_visualized_dimension-1)));
			}
		} else {
			set_visualized_layer(std::max(0, std::min((int)m_network->num_forward_activations()-1, m_visualized_layer-1)));
		}
	}

	if (ImGui::IsKeyPressed('M')) {
		m_single_view = !m_single_view;
		set_visualized_dim(-1);
		reset_accumulation();
	}


	if (ImGui::IsKeyPressed('N')) {
		m_sdf.analytic_normals = !m_sdf.analytic_normals;
		reset_accumulation();
	}

	if (ImGui::IsKeyPressed('[')) {
		if (shift) {
			first_training_view();
		} else {
			previous_training_view();
		}
	}

	if (ImGui::IsKeyPressed(']')) {
		if (shift) {
			last_training_view();
		} else {
			next_training_view();
		}
	}

	if (ImGui::IsKeyPressed('=') || ImGui::IsKeyPressed('+')) {
		m_camera_velocity *= 1.5f;
	}

	if (ImGui::IsKeyPressed('-') || ImGui::IsKeyPressed('_')) {
		m_camera_velocity /= 1.5f;
	}

	// WASD camera movement
	Vector3f translate_vec = Vector3f::Zero();
	if (ImGui::IsKeyDown('W')) {
		translate_vec.z() += 1.0f;
	}

	if (ImGui::IsKeyDown('A')) {
		translate_vec.x() += -1.0f;
	}

	if (ImGui::IsKeyDown('S')) {
		translate_vec.z() += -1.0f;
	}

	if (ImGui::IsKeyDown('D')) {
		translate_vec.x() += 1.0f;
	}

	if (ImGui::IsKeyDown(' ')) {
		translate_vec.y() += -1.0f;
	}

	if (ImGui::IsKeyDown('C')) {
		translate_vec.y() += 1.0f;
	}

	translate_vec *= m_camera_velocity * m_frame_ms.val() / 1000.0f;
	if (shift) {
		translate_vec *= 5;
	}

	if (translate_vec != Vector3f::Zero()) {
		m_fps_camera = true;
		translate_camera(translate_vec);
	}

	return false;
}

void Testbed::mouse_wheel(Vector2f m, float delta) {
	if (delta == 0) {
		return;
	}

	if (!ImGui::GetIO().WantCaptureMouse) {
		float scale_factor = pow(1.1f, -delta);
		m_image.pos = (m_image.pos - m) / scale_factor + m;
		set_scale(m_scale * scale_factor);
	}

	reset_accumulation(true);
}

void Testbed::mouse_drag(const Vector2f& rel, int button) {
	Vector3f up = m_up_dir;
	Vector3f side = m_camera.col(0);

	bool is_left_held = (button & 1) != 0;
	bool is_right_held = (button & 2) != 0;

	bool shift = ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Shift;
	if (is_left_held) {
		if (shift) {
			auto mouse = ImGui::GetMousePos();
			determine_autofocus_target_from_pixel({mouse.x, mouse.y});
			reset_accumulation();
		} else {
			float rot_sensitivity = m_fps_camera ? 0.35f : 1.0f;
			Matrix3f rot =
				(AngleAxisf(static_cast<float>(-rel.x() * 2 * PI() * rot_sensitivity), up) * // Scroll sideways around up vector
				AngleAxisf(static_cast<float>(-rel.y() * 2 * PI() * rot_sensitivity), side)).matrix(); // Scroll around side vector

			m_image.pos += rel;
			if (m_fps_camera) {
				m_camera.block<3, 3>(0, 0) = rot * m_camera.block<3, 3>(0, 0);
			} else {
				// Turntable
				auto old_look_at = look_at();
				set_look_at({0.0f, 0.0f, 0.0f});
				m_camera = rot * m_camera;
				set_look_at(old_look_at);
			}

			reset_accumulation(true);
		}
	}

	if (is_right_held) {
		Matrix3f rot =
			(AngleAxisf(static_cast<float>(-rel.x() * 2 * PI()), up) * // Scroll sideways around up vector
			AngleAxisf(static_cast<float>(-rel.y() * 2 * PI()), side)).matrix(); // Scroll around side vector

		if (m_render_mode == ERenderMode::Shade) {
			m_sun_dir = rot.transpose() * m_sun_dir;
		}

		m_slice_plane_z += -rel.y() * m_bounding_radius;
		reset_accumulation();
	}

	bool is_middle_held = (button & 4) != 0;
	if (is_middle_held) {
		translate_camera({-rel.x(), -rel.y(), 0.0f});
	}
}

bool Testbed::begin_frame_and_handle_user_input() {
	if (glfwWindowShouldClose(m_glfw_window) || ImGui::IsKeyDown(GLFW_KEY_ESCAPE) || ImGui::IsKeyDown(GLFW_KEY_Q)) {
		destroy_window();
		return false;
	}

	{
		auto now = std::chrono::steady_clock::now();
		auto elapsed = now - m_last_frame_time_point;
		m_last_frame_time_point = now;
		m_frame_ms.update(std::chrono::duration<float, std::milli>(elapsed).count());
	}

	glfwPollEvents();
	glfwGetFramebufferSize(m_glfw_window, &m_window_res.x(), &m_window_res.y());

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	if (ImGui::IsKeyPressed(GLFW_KEY_TAB) || ImGui::IsKeyPressed(GLFW_KEY_GRAVE_ACCENT)) {
		m_imgui_enabled = !m_imgui_enabled;
	}

	ImVec2 m = ImGui::GetMousePos();
	int mb = 0;
	float mw = 0.f;
	ImVec2 relm = {};
	if (!ImGui::IsAnyItemActive() && !ImGuizmo::IsUsing() && !ImGuizmo::IsOver()) {
		relm = ImGui::GetIO().MouseDelta;
		if (ImGui::GetIO().MouseDown[0]) mb |= 1;
		if (ImGui::GetIO().MouseDown[1]) mb |= 2;
		if (ImGui::GetIO().MouseDown[2]) mb |= 4;
		mw = ImGui::GetIO().MouseWheel;
		relm = {relm.x / (float)m_window_res.y(), relm.y / (float)m_window_res.y()};
	}

	if (m_testbed_mode == ETestbedMode::Nerf && (m_render_ground_truth || m_nerf.training.render_error_overlay)) {
		// find nearest training view to current camera, and set it
		int bestimage = find_best_training_view(-1);
		if (bestimage >= 0) {
			m_nerf.training.view = bestimage;
			if (mb == 0) {// snap camera to ground truth view on mouse up
				set_camera_to_training_view(m_nerf.training.view);
			}
		}
	}

	keyboard_event();
	mouse_wheel({m.x / (float)m_window_res.y(), m.y / (float)m_window_res.y()}, mw);
	mouse_drag({relm.x, relm.y}, mb);

	if (m_imgui_enabled) {
		imgui();
	}

	return true;
}

void Testbed::SecondWindow::draw(GLuint texture) {
	if (!window)
		return;
	int display_w, display_h;
	GLFWwindow *old_context = glfwGetCurrentContext();
	glfwMakeContextCurrent(window);
	glfwGetFramebufferSize(window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(0.f,0.f,0.f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindVertexArray(vao);
	if (program)
		glUseProgram(program);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	glUseProgram(0);
	glfwSwapBuffers(window);
	glfwMakeContextCurrent(old_context);
}

void Testbed::draw_gui() {
	// Make sure all the cuda code finished its business here
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	if (!m_render_textures.empty())
		m_second_window.draw((GLuint)m_render_textures.front()->texture());
	glfwMakeContextCurrent(m_glfw_window);
	int display_w, display_h;
	glfwGetFramebufferSize(m_glfw_window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	ImDrawList* list = ImGui::GetBackgroundDrawList();
	list->AddCallback([](const ImDrawList*, const ImDrawCmd*) {
		glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
		glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	}, nullptr);

	if (m_single_view) {
		list->AddImageQuad((ImTextureID)(size_t)m_render_textures.front()->texture(), ImVec2{0.f, 0.f}, ImVec2{(float)display_w, 0.f}, ImVec2{(float)display_w, (float)display_h}, ImVec2{0.f, (float)display_h}, ImVec2(0, 0), ImVec2(1, 0), ImVec2(1, 1), ImVec2(0, 1));
	} else {
		m_dlss = false;

		int i = 0;
		for (int y = 0; y < m_n_views.y(); ++y) {
			for (int x = 0; x < m_n_views.x(); ++x) {
				if (i >= m_render_surfaces.size()) {
					break;
				}

				Vector2f top_left{x * m_view_size.x(), y * m_view_size.y()};

				list->AddImageQuad(
					(ImTextureID)(size_t)m_render_textures[i]->texture(),
					ImVec2{top_left.x(),                          top_left.y()                         },
					ImVec2{top_left.x() + (float)m_view_size.x(), top_left.y()                         },
					ImVec2{top_left.x() + (float)m_view_size.x(), top_left.y() + (float)m_view_size.y()},
					ImVec2{top_left.x(),                          top_left.y() + (float)m_view_size.y()},
					ImVec2(0, 0),
					ImVec2(1, 0),
					ImVec2(1, 1),
					ImVec2(0, 1)
				);

				++i;
			}
		}
	}

	list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

	auto draw_mesh = [&]() {
		glClear(GL_DEPTH_BUFFER_BIT);
		Vector2i res(display_w, display_h);
		Vector2f focal_length = calc_focal_length(res, m_fov_axis, m_zoom);
		Vector2f screen_center = render_screen_center();
		draw_mesh_gl(m_mesh.verts, m_mesh.vert_normals, m_mesh.vert_colors, m_mesh.indices, res, focal_length, m_smoothed_camera, screen_center, (int)m_mesh_render_mode);
	};

	// Visualizations are only meaningful when rendering a single view
	if (m_single_view) {
		if (m_mesh.verts.size() != 0 && m_mesh.indices.size() != 0 && m_mesh_render_mode != EMeshRenderMode::Off) {
			list->AddCallback([](const ImDrawList*, const ImDrawCmd* cmd) {
				(*(decltype(draw_mesh)*)cmd->UserCallbackData)();
			}, &draw_mesh);
			list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
		}

		draw_visualizations(list, m_smoothed_camera);
	}

	if (m_render_ground_truth) {
		list->AddText(ImVec2(4.f, 4.f), 0xffffffff, "Ground Truth");
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_glfw_window);

	// Make sure all the OGL code finished its business here.
	// Any code outside of this function needs to be able to freely write to
	// textures without being worried about interfering with rendering.
	glFinish();
}
#endif //NGP_GUI

__global__ void to_8bit_color_kernel(
	Vector2i resolution,
	EColorSpace output_color_space,
	cudaSurfaceObject_t surface,
	uint8_t* result
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	Array4f color;
	surf2Dread((float4*)&color, surface, x * sizeof(float4), y);

	if (output_color_space == EColorSpace::Linear) {
		color.head<3>() = linear_to_srgb(color.head<3>());
	}

	for (uint32_t i = 0; i < 3; ++i) {
		result[(x + resolution.x() * y) * 3 + i] = (uint8_t)(tcnn::clamp(color[i], 0.0f, 1.0f) * 255.0f + 0.5f);
	}
}

void Testbed::prepare_next_camera_path_frame() {
	if (!m_camera_path.rendering) {
		return;
	}

	// If we're rendering a video, we'd like to accumulate multiple spp
	// for motion blur. Hence dump the frame once the target spp has been reached
	// and only reset _then_.
	if (m_render_surfaces.front().spp() == m_camera_path.render_settings.spp) {
		auto tmp_dir = fs::path{"tmp"};
		if (!tmp_dir.exists()) {
			if (!fs::create_directory(tmp_dir)) {
				m_camera_path.rendering = false;
				tlog::error() << "Failed to create temporary directory 'tmp' to hold rendered images.";
				return;
			}
		}

		Vector2i res = m_render_surfaces.front().out_resolution();
		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };

		GPUMemory<uint8_t> image_data(res.prod() * 3);
		to_8bit_color_kernel<<<blocks, threads>>>(
			res,
			EColorSpace::SRGB, // the GUI always renders in SRGB
			m_render_surfaces.front().surface(),
			image_data.data()
		);

		m_render_futures.emplace_back(m_thread_pool.enqueue_task([image_data=std::move(image_data), frame_idx=m_camera_path.render_frame_idx++, res, tmp_dir] {
			std::vector<uint8_t> cpu_image_data(image_data.size());
			CUDA_CHECK_THROW(cudaMemcpy(cpu_image_data.data(), image_data.data(), image_data.bytes(), cudaMemcpyDeviceToHost));
			stbi_write_jpg(fmt::format("{}/{:06d}.jpg", tmp_dir.str(), frame_idx).c_str(), res.x(), res.y(), 3, cpu_image_data.data(), 100);
		}));

		reset_accumulation(true);

		if (m_camera_path.render_frame_idx == m_camera_path.render_settings.n_frames()) {
			m_camera_path.rendering = false;

			wait_all(m_render_futures);
			m_render_futures.clear();

			tlog::success() << "Finished rendering '.jpg' video frames to '" << tmp_dir << "'. Assembling them into a video next.";

			fs::path ffmpeg = "ffmpeg";

#ifdef _WIN32
			// Under Windows, try automatically downloading FFmpeg binaries if they don't exist
			if (system(fmt::format("where {} >nul 2>nul", ffmpeg.str()).c_str()) != 0) {
				fs::path root_dir = get_root_dir();
				if ((root_dir/"external"/"ffmpeg").exists()) {
					for (const auto& path : fs::directory{root_dir/"external"/"ffmpeg"}) {
						ffmpeg = path/"bin"/"ffmpeg.exe";
					}
				}

				if (!ffmpeg.exists()) {
					tlog::info() << "FFmpeg not found. Downloading FFmpeg...";
					do_system((root_dir/"scripts"/"download_ffmpeg.bat").str());
				}

				for (const auto& path : fs::directory{root_dir/"external"/"ffmpeg"}) {
					ffmpeg = path/"bin"/"ffmpeg.exe";
				}

				if (!ffmpeg.exists()) {
					tlog::warning() << "FFmpeg download failed. Trying system-wide FFmpeg.";
				}
			}
#endif

			auto ffmpeg_command = fmt::format(
				"{} -loglevel error -y -framerate {} -i tmp/%06d.jpg -c:v libx264 -preset slow -crf {} -pix_fmt yuv420p {}",
				ffmpeg.str(),
				m_camera_path.render_settings.fps,
				// Quality goes from 0 to 10. This conversion to CRF means a quality of 10
				// is a CRF of 17 and a quality of 0 a CRF of 27, which covers the "sane"
				// range of x264 quality settings according to the FFmpeg docs:
				// https://trac.ffmpeg.org/wiki/Encode/H.264
				27 - m_camera_path.render_settings.quality,
				m_camera_path.render_settings.filename
			);
			int ffmpeg_result = do_system(ffmpeg_command);
			if (ffmpeg_result == 0) {
				tlog::success() << "Saved video '" << m_camera_path.render_settings.filename << "'";
			} else if (ffmpeg_result == -1) {
				tlog::error() << "Video could not be assembled: FFmpeg not found.";
			} else {
				tlog::error() << "Video could not be assembled: FFmpeg failed";
			}

			clear_tmp_dir();
		}
	}

	const auto& rs = m_camera_path.render_settings;
	m_camera_path.play_time = (float)((double)m_camera_path.render_frame_idx / (double)rs.n_frames());

	if (m_render_surfaces.front().spp() == 0) {
		set_camera_from_time(m_camera_path.play_time);
		apply_camera_smoothing(rs.frame_milliseconds());

		auto smoothed_camera_backup = m_smoothed_camera;

		// Compute the camera for the next frame in order to be able to compute motion blur
		// between it and the current one.
		set_camera_from_time(m_camera_path.play_time + 1.0f / rs.n_frames());
		apply_camera_smoothing(rs.frame_milliseconds());

		m_camera_path.render_frame_end_camera = m_smoothed_camera;

		// Revert camera such that the next frame will be computed correctly
		// (Start camera of next frame should be the same as end camera of this frame)
		set_camera_from_time(m_camera_path.play_time);
		m_smoothed_camera = smoothed_camera_backup;
	}
}

void Testbed::train_and_render(bool skip_rendering) {
	if (m_train) {
		train(m_training_batch_size);
	}

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (m_testbed_mode != ETestbedMode::None && !m_network) {
		reload_network_from_file();
		if (!m_network) {
			throw std::runtime_error{"Unable to reload neural network."};
		}
	}

	if (m_mesh.optimize_mesh) {
		optimise_mesh_step(1);
	}

	// Don't do any smoothing here if a camera path is being rendered. It'll take care
	// of the smoothing on its own.
	float frame_ms = m_camera_path.rendering ? 0.0f : m_frame_ms.val();
	apply_camera_smoothing(frame_ms);

	if (!m_render_window || !m_render || skip_rendering) {
		return;
	}

	auto start = std::chrono::steady_clock::now();
	ScopeGuard timing_guard{[&]() {
		m_render_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count());
	}};

	if ((m_smoothed_camera - m_camera).norm() < 0.001f) {
		m_smoothed_camera = m_camera;
	} else if (!m_camera_path.rendering) {
		reset_accumulation(true);
	}

	if (m_autofocus) {
		autofocus();
	}

	if (m_single_view) {
		// Should have been created when the window was created.
		assert(!m_render_surfaces.empty());

		auto& render_buffer = m_render_surfaces.front();

		{
			// Don't count the time being spent allocating buffers and resetting DLSS as part of the frame time.
			// Otherwise the dynamic resolution calculations for following frames will be thrown out of whack
			// and may even start oscillating.
			auto skip_start = std::chrono::steady_clock::now();
			ScopeGuard skip_timing_guard{[&]() {
				start += std::chrono::steady_clock::now() - skip_start;
			}};
			if (m_dlss) {
				render_buffer.enable_dlss(m_window_res);
				m_aperture_size = 0.0f;
			} else {
				render_buffer.disable_dlss();
			}

			auto render_res = render_buffer.in_resolution();
			if (render_res.isZero() || (m_train && m_training_step == 0)) {
				render_res = m_window_res/16;
			} else {
				render_res = render_res.cwiseMin(m_window_res);
			}

			float render_time_per_fullres_frame = m_render_ms.val() / (float)render_res.x() / (float)render_res.y() * (float)m_window_res.x() * (float)m_window_res.y();

			// Make sure we don't starve training with slow rendering
			float factor = std::sqrt(1000.0f / m_dynamic_res_target_fps / render_time_per_fullres_frame);
			if (!m_dynamic_res) {
				factor = 8.f/(float)m_fixed_res_factor;
			}

			factor = tcnn::clamp(factor, 1.0f/16.0f, 1.0f);

			if (factor > m_last_render_res_factor * 1.2f || factor < m_last_render_res_factor * 0.8f || factor == 1.0f || !m_dynamic_res) {
				render_res = (m_window_res.cast<float>() * factor).cast<int>().cwiseMin(m_window_res).cwiseMax(m_window_res/16);
				m_last_render_res_factor = factor;
			}

			if (m_camera_path.rendering) {
				render_res = m_camera_path.render_settings.resolution;
				m_last_render_res_factor = 1.0f;
			}

			if (render_buffer.dlss()) {
				render_res = render_buffer.dlss()->clamp_resolution(render_res);
				render_buffer.dlss()->update_feature(render_res, render_buffer.dlss()->is_hdr(), render_buffer.dlss()->sharpen());
			}

			render_buffer.resize(render_res);
		}

		render_frame(
			m_smoothed_camera,
			m_camera_path.rendering ? log_space_lerp(m_smoothed_camera, m_camera_path.render_frame_end_camera, m_camera_path.render_settings.shutter_fraction) : m_smoothed_camera,
			{0.0f, 0.0f, 0.0f, 1.0f},
			render_buffer
		);

#ifdef NGP_GUI
		m_render_textures.front()->blit_from_cuda_mapping();

		if (m_picture_in_picture_res > 0) {
			Vector2i res(m_picture_in_picture_res, m_picture_in_picture_res * 9/16);
			m_pip_render_surface->resize(res);
			if (m_pip_render_surface->spp() < 8) {
				// a bit gross, but let's copy the keyframe's state into the global state in order to not have to plumb through the fov etc to render_frame.
				CameraKeyframe backup = copy_camera_to_keyframe();
				CameraKeyframe pip_kf = m_camera_path.eval_camera_path(m_camera_path.play_time);
				set_camera_from_keyframe(pip_kf);
				render_frame(pip_kf.m(), pip_kf.m(), Eigen::Vector4f::Zero(), *m_pip_render_surface);
				set_camera_from_keyframe(backup);

				m_pip_render_texture->blit_from_cuda_mapping();
			}
		}
#endif
	} else {
#ifdef NGP_GUI
		// Don't do DLSS when multi-view rendering
		m_dlss = false;
		m_render_surfaces.front().disable_dlss();

		int n_views = n_dimensions_to_visualize()+1;

		float d = std::sqrt((float)m_window_res.x() * (float)m_window_res.y() / (float)n_views);

		int nx = (int)std::ceil((float)m_window_res.x() / d);
		int ny = (int)std::ceil((float)n_views / (float)nx);

		m_n_views = {nx, ny};
		m_view_size = {m_window_res.x() / nx, m_window_res.y() / ny};

		while (m_render_surfaces.size() > n_views) {
			m_render_surfaces.pop_back();
		}

		m_render_textures.resize(n_views);
		while (m_render_surfaces.size() < n_views) {
			size_t idx = m_render_surfaces.size();
			m_render_textures[idx] = std::make_shared<GLTexture>();
			m_render_surfaces.emplace_back(m_render_textures[idx]);
		}

		int i = 0;
		for (int y = 0; y < ny; ++y) {
			for (int x = 0; x < nx; ++x) {
				if (i >= n_views) {
					return;
				}

				m_visualized_dimension = i-1;
				m_render_surfaces[i].resize(m_view_size);

				render_frame(m_smoothed_camera, m_smoothed_camera, Eigen::Vector4f::Zero(), m_render_surfaces[i]);

				m_render_textures[i]->blit_from_cuda_mapping();
				++i;
			}
		}
#else
		throw std::runtime_error{"Multi-view rendering is only supported when compiling with NGP_GUI."};
#endif
	}
}


#ifdef NGP_GUI
void Testbed::create_second_window() {
	if (m_second_window.window) {
		return;
	}
	bool frameless = false;
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, !frameless);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CENTER_CURSOR, false);
	glfwWindowHint(GLFW_DECORATED, !frameless);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, frameless);
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, true);
	// get the window size / coordinates
	int win_w=0,win_h=0,win_x=0,win_y=0;
	GLuint ps=0,vs=0;
	{
		win_w = 1920;
		win_h = 1080;
		win_x = 0x40000000;
		win_y = 0x40000000;
		static const char* copy_shader_vert = "\
			layout (location = 0)\n\
			in vec2 vertPos_data;\n\
			out vec2 texCoords;\n\
			void main(){\n\
				gl_Position = vec4(vertPos_data.xy, 0.0, 1.0);\n\
				texCoords = (vertPos_data.xy + 1.0) * 0.5; texCoords.y=1.0-texCoords.y;\n\
			}";
		static const char* copy_shader_frag = "\
			in vec2 texCoords;\n\
			out vec4 fragColor;\n\
			uniform sampler2D screenTex;\n\
			void main(){\n\
				fragColor = texture(screenTex, texCoords.xy);\n\
			}";
		vs = compile_shader(false, copy_shader_vert);
		ps = compile_shader(true, copy_shader_frag);
	}
	m_second_window.window = glfwCreateWindow(win_w, win_h, "Fullscreen Output", NULL, m_glfw_window);
	if (win_x!=0x40000000) glfwSetWindowPos(m_second_window.window, win_x, win_y);
	glfwMakeContextCurrent(m_second_window.window);
	m_second_window.program = glCreateProgram();
	glAttachShader(m_second_window.program, vs);
	glAttachShader(m_second_window.program, ps);
	glLinkProgram(m_second_window.program);
	if (!check_shader(m_second_window.program, "shader program", true)) {
		glDeleteProgram(m_second_window.program);
		m_second_window.program = 0;
	}
	// vbo and vao
	glGenVertexArrays(1, &m_second_window.vao);
	glGenBuffers(1, &m_second_window.vbo);
	glBindVertexArray(m_second_window.vao);
	const float fsquadVerts[] = {
		-1.0f, -1.0f,
		-1.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, -1.0f,
		-1.0f, -1.0f};
	glBindBuffer(GL_ARRAY_BUFFER, m_second_window.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fsquadVerts), fsquadVerts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
#endif //NGP_GUI

void Testbed::init_window(int resw, int resh, bool hidden, bool second_window) {
#ifndef NGP_GUI
	throw std::runtime_error{"init_window failed: NGP was built without GUI support"};
#else
	m_window_res = {resw, resh};

	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		throw std::runtime_error{"GLFW could not be initialized."};
	}

#ifdef NGP_VULKAN
	try {
		vulkan_and_ngx_init();
		m_dlss_supported = true;
		if (m_testbed_mode == ETestbedMode::Nerf) {
			m_dlss = true;
		}
	} catch (const std::runtime_error& e) {
		tlog::warning() << "Could not initialize Vulkan and NGX. DLSS not supported. (" << e.what() << ")";
	}
#else
	m_dlss_supported = false;
#endif

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	glfwWindowHint(GLFW_VISIBLE, hidden ? GLFW_FALSE : GLFW_TRUE);
	std::string title = "Instant Neural Graphics Primitives";
	m_glfw_window = glfwCreateWindow(m_window_res.x(), m_window_res.y(), title.c_str(), NULL, NULL);
	if (m_glfw_window == NULL) {
		throw std::runtime_error{"GLFW window could not be created."};
	}
	glfwMakeContextCurrent(m_glfw_window);
#ifdef _WIN32
	if (gl3wInit()) {
		throw std::runtime_error{"GL3W could not be initialized."};
	}
#else
	glewExperimental = 1;
	if (glewInit()) {
		throw std::runtime_error{"GLEW could not be initialized."};
	}
#endif
	glfwSwapInterval(0); // Disable vsync

	glfwSetWindowUserPointer(m_glfw_window, this);
	glfwSetDropCallback(m_glfw_window, [](GLFWwindow* window, int count, const char** paths) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (!testbed) {
			return;
		}

		testbed->redraw_gui_next_frame();
		for (int i = 0; i < count; i++) {
			testbed->load_file(paths[i]);
		}
	});

	glfwSetKeyCallback(m_glfw_window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetCursorPosCallback(m_glfw_window, [](GLFWwindow* window, double xpos, double ypos) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetMouseButtonCallback(m_glfw_window, [](GLFWwindow* window, int button, int action, int mods) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetScrollCallback(m_glfw_window, [](GLFWwindow* window, double xoffset, double yoffset) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetWindowSizeCallback(m_glfw_window, [](GLFWwindow* window, int width, int height) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_next_frame();
		}
	});

	glfwSetFramebufferSizeCallback(m_glfw_window, [](GLFWwindow* window, int width, int height) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_next_frame();
		}
	});

	float xscale, yscale;
	glfwGetWindowContentScale(m_glfw_window, &xscale, &yscale);

	// IMGUI init
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigInputTrickleEventQueue = false; // new ImGui event handling seems to make camera controls laggy if this is true.
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_glfw_window, true);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	ImGui::GetStyle().ScaleAllSizes(xscale);
	ImFontConfig font_cfg;
	font_cfg.SizePixels = 13.0f * xscale;
	io.Fonts->AddFontDefault(&font_cfg);

	// Make sure there's at least one usable render texture
	m_render_textures = { std::make_shared<GLTexture>() };

	m_render_surfaces.clear();
	m_render_surfaces.emplace_back(m_render_textures.front());
	m_render_surfaces.front().resize(m_window_res);

	m_pip_render_texture = std::make_shared<GLTexture>();
	m_pip_render_surface = std::make_unique<CudaRenderBuffer>(m_pip_render_texture);

	m_render_window = true;

	if (m_second_window.window == nullptr && second_window) {
		create_second_window();
	}
#endif // NGP_GUI
}

void Testbed::destroy_window() {
#ifndef NGP_GUI
	throw std::runtime_error{"destroy_window failed: NGP was built without GUI support"};
#else
	if (!m_render_window) {
		throw std::runtime_error{"Window must be initialized to be destroyed."};
	}

	m_render_surfaces.clear();
	m_render_textures.clear();

	m_pip_render_surface.reset();
	m_pip_render_texture.reset();

#ifdef NGP_VULKAN
	m_dlss_supported = m_dlss = false;
	vulkan_and_ngx_destroy();
#endif

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();

	m_glfw_window = nullptr;
	m_render_window = false;
#endif //NGP_GUI
}

bool Testbed::frame() {
#ifdef NGP_GUI
	if (m_render_window) {
		if (!begin_frame_and_handle_user_input()) {
			return false;
		}
	}
#endif

	// Render against the trained neural network. If we're training and already close to convergence,
	// we can skip rendering if the scene camera doesn't change
	uint32_t n_to_skip = m_train ? tcnn::clamp(m_training_step / 16u, 15u, 255u) : 0;
	if (m_render_skip_due_to_lack_of_camera_movement_counter > n_to_skip) {
		m_render_skip_due_to_lack_of_camera_movement_counter = 0;
	}
	bool skip_rendering = m_render_skip_due_to_lack_of_camera_movement_counter++ != 0;

	if (!m_dlss && m_max_spp > 0 && !m_render_surfaces.empty() && m_render_surfaces.front().spp() >= m_max_spp) {
		skip_rendering = true;
		if (!m_train) {
			std::this_thread::sleep_for(1ms);
		}
	}

	if (m_camera_path.rendering) {
		prepare_next_camera_path_frame();
		skip_rendering = false;
	}

	if (!skip_rendering || (std::chrono::steady_clock::now() - m_last_gui_draw_time_point) > 25ms) {
		redraw_gui_next_frame();
	}

	try {
		while (true) {
			(*m_task_queue.tryPop())();
		}
	} catch (SharedQueueEmptyException&) {}


	train_and_render(skip_rendering);
	if (m_testbed_mode == ETestbedMode::Sdf && m_sdf.calculate_iou_online) {
		m_sdf.iou = calculate_iou(m_train ? 64*64*64 : 128*128*128, m_sdf.iou_decay, false, true);
		m_sdf.iou_decay = 0.f;
	}

#ifdef NGP_GUI
	if (m_render_window) {
		if (m_gui_redraw) {
			if (m_gather_histograms) {
				gather_histograms();
			}

			draw_gui();
			m_gui_redraw = false;

			m_last_gui_draw_time_point = std::chrono::steady_clock::now();
		}

		ImGui::EndFrame();
	}
#endif

	return true;
}

fs::path Testbed::training_data_path() const {
	return m_data_path.with_extension("training");
}

bool Testbed::want_repl() {
	bool b = m_want_repl;
	m_want_repl = false;
	return b;
}

void Testbed::apply_camera_smoothing(float elapsed_ms) {
	if (m_camera_smoothing) {
		float decay = std::pow(0.02f, elapsed_ms/1000.0f);
		m_smoothed_camera = log_space_lerp(m_smoothed_camera, m_camera, 1.0f - decay);
	} else {
		m_smoothed_camera = m_camera;
	}
}

CameraKeyframe Testbed::copy_camera_to_keyframe() const {
	return CameraKeyframe(m_camera, m_slice_plane_z, m_scale, fov(), m_aperture_size, m_nerf.glow_mode, m_nerf.glow_y_cutoff);
}

void Testbed::set_camera_from_keyframe(const CameraKeyframe& k) {
	m_camera = k.m();
	m_slice_plane_z = k.slice;
	m_scale = k.scale;
	set_fov(k.fov);
	m_aperture_size = k.aperture_size;
	m_nerf.glow_mode = k.glow_mode;
	m_nerf.glow_y_cutoff = k.glow_y_cutoff;
}

void Testbed::set_camera_from_time(float t) {
	if (m_camera_path.keyframes.empty())
		return;
	set_camera_from_keyframe(m_camera_path.eval_camera_path(t));
}

void Testbed::update_loss_graph() {
	m_loss_graph[m_loss_graph_samples++ % m_loss_graph.size()] = std::log(m_loss_scalar.val());
}

uint32_t Testbed::n_dimensions_to_visualize() const {
	return m_network->width(m_visualized_layer);
}

float Testbed::fov() const {
	return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]);
}

void Testbed::set_fov(float val) {
	m_relative_focal_length = Vector2f::Constant(fov_to_focal_length(1, val));
}

Vector2f Testbed::fov_xy() const {
	return focal_length_to_fov(Vector2i::Ones(), m_relative_focal_length);
}

void Testbed::set_fov_xy(const Vector2f& val) {
	m_relative_focal_length = fov_to_focal_length(Vector2i::Ones(), val);
}

size_t Testbed::n_params() {
	return m_network->n_params();
}

size_t Testbed::n_encoding_params() {
	return m_network->n_params() - first_encoder_param();
}

size_t Testbed::first_encoder_param() {
	auto layer_sizes = m_network->layer_sizes();
	size_t first_encoder = 0;
	for (auto size : layer_sizes) {
		first_encoder += size.first * size.second;
	}
	return first_encoder;
}

uint32_t Testbed::network_width(uint32_t layer) const {
	return m_network->width(layer);
}

uint32_t Testbed::network_num_forward_activations() const {
	return m_network->num_forward_activations();
}

void Testbed::set_max_level(float maxlevel) {
	if (!m_network) return;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level(maxlevel);
	}
	reset_accumulation();
}

void Testbed::set_min_level(float minlevel) {
	if (!m_network) return;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_quantize_threshold(powf(minlevel, 4.f) * 0.2f);
	}
	reset_accumulation();
}

void Testbed::set_visualized_layer(int layer) {
	m_visualized_layer = layer;
	m_visualized_dimension = std::max(-1, std::min(m_visualized_dimension, (int)m_network->width(layer)-1));
	reset_accumulation();
}

ELossType Testbed::string_to_loss_type(const std::string& str) {
	if (equals_case_insensitive(str, "L2")) {
		return ELossType::L2;
	} else if (equals_case_insensitive(str, "RelativeL2")) {
		return ELossType::RelativeL2;
	} else if (equals_case_insensitive(str, "L1")) {
		return ELossType::L1;
	} else if (equals_case_insensitive(str, "Mape")) {
		return ELossType::Mape;
	} else if (equals_case_insensitive(str, "Smape")) {
		return ELossType::Smape;
	} else if (equals_case_insensitive(str, "Huber") || equals_case_insensitive(str, "SmoothL1")) {
		// Legacy: we used to refer to the Huber loss (L2 near zero, L1 further away) as "SmoothL1".
		return ELossType::Huber;
	} else if (equals_case_insensitive(str, "LogL1")) {
		return ELossType::LogL1;
	} else {
		throw std::runtime_error{"Unknown loss type."};
	}
}

Testbed::NetworkDims Testbed::network_dims() const {
	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:   return network_dims_nerf(); break;
		case ETestbedMode::Sdf:    return network_dims_sdf(); break;
		case ETestbedMode::Image:  return network_dims_image(); break;
		case ETestbedMode::Volume: return network_dims_volume(); break;
		default: throw std::runtime_error{"Invalid mode."};
	}
}

void Testbed::reset_network(bool clear_density_grid) {
	m_sdf.iou_decay = 0;

	m_rng = default_rng_t{m_seed};

	// Start with a low rendering resolution and gradually ramp up
	m_render_ms.set(10000);

	reset_accumulation();
	m_nerf.training.counters_rgb.rays_per_batch = 1 << 12;
	m_nerf.training.counters_rgb.measured_batch_size_before_compaction = 0;
	m_nerf.training.n_steps_since_cam_update = 0;
	m_nerf.training.n_steps_since_error_map_update = 0;
	m_nerf.training.n_rays_since_error_map_update = 0;
	m_nerf.training.n_steps_between_error_map_updates = 128;
	m_nerf.training.error_map.is_cdf_valid = false;
	m_nerf.training.density_grid_rng = default_rng_t{m_rng.next_uint()};

	m_nerf.training.reset_camera_extrinsics();

	if (clear_density_grid) {
		m_nerf.density_grid.memset(0);
		m_nerf.density_grid_bitfield.memset(0);
	}

	m_loss_graph_samples = 0;

	// Default config
	json config = m_network_config;

	json& encoding_config = config["encoding"];
	json& loss_config = config["loss"];
	json& optimizer_config = config["optimizer"];
	json& network_config = config["network"];

	auto dims = network_dims();

	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.loss_type = string_to_loss_type(loss_config.value("otype", "L2"));

		// Some of the Nerf-supported losses are not supported by tcnn::Loss,
		// so just create a dummy L2 loss there. The NeRF code path will bypass
		// the tcnn::Loss in any case.
		loss_config["otype"] = "L2";
	}

	// Automatically determine certain parameters if we're dealing with the (hash)grid encoding
	if (to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
		encoding_config["n_pos_dims"] = dims.n_pos;

		const uint32_t n_features_per_level = encoding_config.value("n_features_per_level", 2u);

		if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
			m_num_levels = (uint32_t)encoding_config["n_features"] / n_features_per_level;
		} else {
			m_num_levels = encoding_config.value("n_levels", 16u);
		}

		m_level_stats.resize(m_num_levels);
		m_first_layer_column_stats.resize(m_num_levels);

		const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

		m_base_grid_resolution = encoding_config.value("base_resolution", 0);
		if (!m_base_grid_resolution) {
			m_base_grid_resolution = 1u << ((log2_hashmap_size) / dims.n_pos);
			encoding_config["base_resolution"] = m_base_grid_resolution;
		}

		float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube
		if (m_testbed_mode == ETestbedMode::Image) {
			desired_resolution = m_image.resolution.maxCoeff() / 2.0f;
		} else if (m_testbed_mode == ETestbedMode::Volume) {
			desired_resolution = m_volume.world2index_scale;
		}

		// Automatically determine suitable per_level_scale
		m_per_level_scale = encoding_config.value("per_level_scale", 0.0f);
		if (m_per_level_scale <= 0.0f && m_num_levels > 1) {
			m_per_level_scale = std::exp(std::log(desired_resolution * (float)m_nerf.training.dataset.aabb_scale / (float)m_base_grid_resolution) / (m_num_levels-1));
			encoding_config["per_level_scale"] = m_per_level_scale;
		}

		tlog::info()
			<< "GridEncoding: "
			<< " Nmin=" << m_base_grid_resolution
			<< " b=" << m_per_level_scale
			<< " F=" << n_features_per_level
			<< " T=2^" << log2_hashmap_size
			<< " L=" << m_num_levels
			;
	}

	m_loss.reset(create_loss<precision_t>(loss_config));
	m_optimizer.reset(create_optimizer<precision_t>(optimizer_config));

	size_t n_encoding_params = 0;
	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Array3f>(1e-3f, Array3f::Zero()));
		m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Vector3f>(1e-4f, Vector3f::Zero()));
		m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
		m_nerf.training.cam_focal_length_offset = AdamOptimizer<Vector2f>(1e-5f);

		m_nerf.training.reset_extra_dims(m_rng);

		json& dir_encoding_config = config["dir_encoding"];
		json& rgb_network_config = config["rgb_network"];

		uint32_t n_dir_dims = 3;
		uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
		m_network = m_nerf_network = std::make_shared<NerfNetwork<precision_t>>(
			dims.n_pos,
			n_dir_dims,
			n_extra_dims,
			dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
			encoding_config,
			dir_encoding_config,
			network_config,
			rgb_network_config
		);

		m_encoding = m_nerf_network->encoding();
		n_encoding_params = m_encoding->n_params() + m_nerf_network->dir_encoding()->n_params();

		tlog::info()
			<< "Density model: " << dims.n_pos
			<< "--[" << std::string(encoding_config["otype"])
			<< "]-->" << m_nerf_network->encoding()->padded_output_width()
			<< "--[" << std::string(network_config["otype"])
			<< "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << 1
			;

		tlog::info()
			<< "Color model:   " << n_dir_dims
			<< "--[" << std::string(dir_encoding_config["otype"])
			<< "]-->" << m_nerf_network->dir_encoding()->padded_output_width() << "+" << network_config.value("n_output_dims", 16u)
			<< "--[" << std::string(rgb_network_config["otype"])
			<< "(neurons=" << (int)rgb_network_config["n_neurons"] << ",layers=" << ((int)rgb_network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << 3
			;

		// Create distortion map model
		{
			json& distortion_map_optimizer_config =  config.contains("distortion_map") && config["distortion_map"].contains("optimizer") ? config["distortion_map"]["optimizer"] : optimizer_config;

			m_distortion.resolution = Vector2i::Constant(32);
			if (config.contains("distortion_map") && config["distortion_map"].contains("resolution")) {
				from_json(config["distortion_map"]["resolution"], m_distortion.resolution);
			}
			m_distortion.map = std::make_shared<TrainableBuffer<2, 2, float>>(m_distortion.resolution);
			m_distortion.optimizer.reset(create_optimizer<float>(distortion_map_optimizer_config));
			m_distortion.trainer = std::make_shared<Trainer<float, float>>(m_distortion.map, m_distortion.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(loss_config)}, m_seed);
		}
	} else {
		uint32_t alignment = network_config.contains("otype") && (equals_case_insensitive(network_config["otype"], "FullyFusedMLP") || equals_case_insensitive(network_config["otype"], "MegakernelMLP")) ? 16u : 8u;

		if (encoding_config.contains("otype") && equals_case_insensitive(encoding_config["otype"], "Takikawa")) {
			if (m_sdf.octree_depth_target == 0) {
				m_sdf.octree_depth_target = encoding_config["n_levels"];
			}

			if (!m_sdf.triangle_octree || m_sdf.triangle_octree->depth() != m_sdf.octree_depth_target) {
				m_sdf.triangle_octree.reset(new TriangleOctree{});
				m_sdf.triangle_octree->build(*m_sdf.triangle_bvh, m_sdf.triangles_cpu, m_sdf.octree_depth_target);
				m_sdf.octree_depth_target = m_sdf.triangle_octree->depth();
				m_sdf.brick_data.free_memory();
			}

			m_encoding.reset(new TakikawaEncoding<precision_t>(
				encoding_config["starting_level"],
				m_sdf.triangle_octree,
				tcnn::string_to_interpolation_type(encoding_config.value("interpolation", "linear"))
			));

			m_sdf.uses_takikawa_encoding = true;
		} else {
			m_encoding.reset(create_encoding<precision_t>(dims.n_input, encoding_config));

			m_sdf.uses_takikawa_encoding = false;
			if (m_sdf.octree_depth_target == 0 && encoding_config.contains("n_levels")) {
				m_sdf.octree_depth_target = encoding_config["n_levels"];
			}
		}

		m_network = std::make_shared<NetworkWithInputEncoding<precision_t>>(m_encoding, dims.n_output, network_config);
		n_encoding_params = m_encoding->n_params();

		tlog::info()
			<< "Model:         " << dims.n_input
			<< "--[" << std::string(encoding_config["otype"])
			<< "]-->" << m_encoding->padded_output_width()
			<< "--[" << std::string(network_config["otype"])
			<< "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << dims.n_output;
	}

	size_t n_network_params = m_network->n_params() - n_encoding_params;

	tlog::info() << "  total_encoding_params=" << n_encoding_params << " total_network_params=" << n_network_params;

	m_trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(m_network, m_optimizer, m_loss, m_seed);
	m_training_step = 0;
	m_training_start_time_point = std::chrono::steady_clock::now();

	// Create envmap model
	{
		json& envmap_loss_config = config.contains("envmap") && config["envmap"].contains("loss") ? config["envmap"]["loss"] : loss_config;
		json& envmap_optimizer_config =  config.contains("envmap") && config["envmap"].contains("optimizer") ? config["envmap"]["optimizer"] : optimizer_config;

		m_envmap.loss_type = string_to_loss_type(envmap_loss_config.value("otype", "L2"));

		m_envmap.resolution = m_nerf.training.dataset.envmap_resolution;
		m_envmap.envmap = std::make_shared<TrainableBuffer<4, 2, float>>(m_envmap.resolution);
		m_envmap.optimizer.reset(create_optimizer<float>(envmap_optimizer_config));
		m_envmap.trainer = std::make_shared<Trainer<float, float, float>>(m_envmap.envmap, m_envmap.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(envmap_loss_config)}, m_seed);

		if (m_nerf.training.dataset.envmap_data.data()) {
			m_envmap.trainer->set_params_full_precision(m_nerf.training.dataset.envmap_data.data(), m_nerf.training.dataset.envmap_data.size());
		}
	}

}

Testbed::Testbed(ETestbedMode mode) {
	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		throw std::runtime_error{"Testbed requires CUDA 10.2 or later."};
	}

#ifdef NGP_GUI
	// Ensure we're running on the GPU that'll host our GUI. To do so, try creating a dummy
	// OpenGL context, figure out the GPU it's running on, and then kill that context again.
	if (glfwInit()) {
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);

		if (offscreen_context) {
			glfwMakeContextCurrent(offscreen_context);

			int gl_device = -1;
			unsigned int device_count = 0;
			if (cudaGLGetDevices(&device_count, &gl_device, 1, cudaGLDeviceListAll) == cudaSuccess) {
				if (device_count > 0 && gl_device != -1) {
					set_cuda_device(gl_device);
				}
			}

			glfwDestroyWindow(offscreen_context);
		}

		glfwTerminate();
	}
#endif

	int active_device = cuda_device();
	int active_compute_capability = cuda_compute_capability();
	tlog::success() << "Initialized CUDA. Active GPU is #" << active_device << ": " << cuda_device_name() << " [" << active_compute_capability << "]";

	if (active_compute_capability < MIN_GPU_ARCH) {
		tlog::warning() << "Insufficient compute capability " << active_compute_capability << " detected.";
		tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
	}

	m_network_config = {
		{"loss", {
			{"otype", "L2"}
		}},
		{"optimizer", {
			{"otype", "Adam"},
			{"learning_rate", 1e-3},
			{"beta1", 0.9f},
			{"beta2", 0.99f},
			{"epsilon", 1e-15f},
			{"l2_reg", 1e-6f},
		}},
		{"encoding", {
			{"otype", "HashGrid"},
			{"n_levels", 16},
			{"n_features_per_level", 2},
			{"log2_hashmap_size", 19},
			{"base_resolution", 16},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"n_neurons", 64},
			{"n_layers", 2},
			{"activation", "ReLU"},
			{"output_activation", "None"},
		}},
	};

	set_mode(mode);
	set_exposure(0);
	set_min_level(0.f);
	set_max_level(1.f);

	reset_camera();
}

Testbed::~Testbed() {

	// If any temporary file was created, make sure it's deleted
	clear_tmp_dir();

	if (m_render_window) {
		destroy_window();
	}
}

bool Testbed::clear_tmp_dir() {
	wait_all(m_render_futures);
	m_render_futures.clear();

	bool success = true;
	auto tmp_dir = fs::path{"tmp"};
	if (tmp_dir.exists()) {
		if (tmp_dir.is_directory()) {
			for (const auto& path : fs::directory{tmp_dir}) {
				if (path.is_file()) {
					success &= path.remove_file();
				}
			}
		}

		success &= tmp_dir.remove_file();
	}

	return success;
}

void Testbed::train(uint32_t batch_size) {
	if (!m_training_data_available || m_camera_path.rendering) {
		m_train = false;
		return;
	}

	if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Cannot train without a mode."};
	}

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (!m_trainer) {
		reload_network_from_file();
		if (!m_trainer) {
			throw std::runtime_error{"Unable to create a neural network trainer."};
		}
	}

	if (!m_dlss) {
		// No immediate redraw necessary
		reset_accumulation(false, false);
	}

	uint32_t n_prep_to_skip = m_testbed_mode == ETestbedMode::Nerf ? tcnn::clamp(m_training_step / 16u, 1u, 16u) : 1u;
	if (m_training_step % n_prep_to_skip == 0) {
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_prep_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count() / n_prep_to_skip);
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf: training_prep_nerf(batch_size, m_stream.get()); break;
			case ETestbedMode::Sdf: training_prep_sdf(batch_size, m_stream.get()); break;
			case ETestbedMode::Image: training_prep_image(batch_size, m_stream.get()); break;
			case ETestbedMode::Volume: training_prep_volume(batch_size, m_stream.get()); break;
			default: throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	// Find leaf optimizer and update its settings
	json* leaf_optimizer_config = &m_network_config["optimizer"];
	while (leaf_optimizer_config->contains("nested")) {
		leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
	}
	(*leaf_optimizer_config)["optimize_matrix_params"] = m_train_network;
	(*leaf_optimizer_config)["optimize_non_matrix_params"] = m_train_encoding;
	m_optimizer->update_hyperparams(m_network_config["optimizer"]);

	bool get_loss_scalar = m_training_step % 16 == 0;

	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count());
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf:   train_nerf(batch_size, get_loss_scalar, m_stream.get()); break;
			case ETestbedMode::Sdf:    train_sdf(batch_size, get_loss_scalar, m_stream.get()); break;
			case ETestbedMode::Image:  train_image(batch_size, get_loss_scalar, m_stream.get()); break;
			case ETestbedMode::Volume: train_volume(batch_size, get_loss_scalar, m_stream.get()); break;
			default: throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	if (get_loss_scalar) {
		update_loss_graph();
	}
}

Vector2f Testbed::calc_focal_length(const Vector2i& resolution, int fov_axis, float zoom) const {
	return m_relative_focal_length * resolution[fov_axis] * zoom;
}

Vector2f Testbed::render_screen_center() const {
	// see pixel_to_ray for how screen center is used; 0.5,0.5 is 'normal'. we flip so that it becomes the point in the original image we want to center on.
	auto screen_center = m_screen_center;
	return {(0.5f-screen_center.x())*m_zoom + 0.5f, (0.5-screen_center.y())*m_zoom + 0.5f};
}

__global__ void dlss_prep_kernel(
	ETestbedMode mode,
	Vector2i resolution,
	uint32_t sample_index,
	Vector2f focal_length,
	Vector2f screen_center,
	Vector3f parallax_shift,
	bool snap_to_pixel_centers,
	float* depth_buffer,
	Matrix<float, 3, 4> camera,
	Matrix<float, 3, 4> prev_camera,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface,
	Lens lens,
	const float view_dist,
	const float prev_view_dist,
	const Vector2f image_pos,
	const Vector2f prev_image_pos,
	const Vector2i image_resolution,
	const Vector2i quilting_dims
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;

	if (quilting_dims != Vector2i::Ones()) {
		apply_quilting(&x, &y, resolution, parallax_shift, quilting_dims);
	}

	const float depth = depth_buffer[idx];
	Vector2f mvec = mode == ETestbedMode::Image ? motion_vector_2d(
		sample_index,
		{x, y},
		resolution.cwiseQuotient(quilting_dims),
		image_resolution,
		screen_center,
		view_dist,
		prev_view_dist,
		image_pos,
		prev_image_pos,
		snap_to_pixel_centers
	) : motion_vector_3d(
		sample_index,
		{x, y},
		resolution.cwiseQuotient(quilting_dims),
		focal_length,
		camera,
		prev_camera,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		depth,
		lens
	);

	surf2Dwrite(make_float2(mvec.x(), mvec.y()), mvec_surface, x_orig * sizeof(float2), y_orig);

	// Scale depth buffer to be guaranteed in [0,1].
	surf2Dwrite(std::min(std::max(depth / 128.0f, 0.0f), 1.0f), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

void Testbed::render_frame(const Matrix<float, 3, 4>& camera_matrix0, const Matrix<float, 3, 4>& camera_matrix1, const Vector4f& nerf_rolling_shutter, CudaRenderBuffer& render_buffer, bool to_srgb) {
	Vector2i max_res = m_window_res.cwiseMax(render_buffer.in_resolution());

	render_buffer.clear_frame(m_stream.get());

	Vector2f focal_length = calc_focal_length(render_buffer.in_resolution(), m_fov_axis, m_zoom);
	Vector2f screen_center = render_screen_center();

	if (!m_network) {
		return;
	}

	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:
			if (!m_render_ground_truth || m_ground_truth_alpha < 1.0f) {
				render_nerf(render_buffer, max_res, focal_length, camera_matrix0, camera_matrix1, nerf_rolling_shutter, screen_center, m_stream.get());
			}
			break;
		case ETestbedMode::Sdf:
			{
				if (m_render_ground_truth && m_sdf.groundtruth_mode == ESDFGroundTruthMode::SDFBricks) {
					if (m_sdf.brick_data.size() == 0) {
						tlog::info() << "Building voxel brick positions for " << m_sdf.triangle_octree->n_dual_nodes() << " dual nodes.";
						m_sdf.brick_res = 5;
						std::vector<Eigen::Vector3f> positions = m_sdf.triangle_octree->build_brick_voxel_position_list(m_sdf.brick_res);
						GPUMemory<Eigen::Vector3f> positions_gpu;
						positions_gpu.resize_and_copy_from_host(positions);
						m_sdf.brick_data.resize(positions.size());
						tlog::info() << positions_gpu.size() << " voxel brick positions. Computing SDFs.";
						m_sdf.triangle_bvh->signed_distance_gpu(
							positions.size(),
							EMeshSdfMode::Watertight, //m_sdf.mesh_sdf_mode, // watertight seems to be the best method for 'one off' SDF signing
							positions_gpu.data(),
							m_sdf.brick_data.data(),
							m_sdf.triangles_gpu.data(),
							false,
							m_stream.get()
						);
					}
				}
				distance_fun_t distance_fun =
					m_render_ground_truth ? (distance_fun_t)[&](uint32_t n_elements, const Vector3f* positions, float* distances, cudaStream_t stream) {
						if (n_elements == 0) {
							return;
						}
						if (m_sdf.groundtruth_mode == ESDFGroundTruthMode::SDFBricks) {
							// linear_kernel(sdf_brick_kernel, 0, stream,
							// 	n_elements,
							// 	positions.data(),
							// 	distances.data(),
							// 	m_sdf.triangle_octree->nodes_gpu(),
							// 	m_sdf.triangle_octree->dual_nodes_gpu(),
							// 	std::max(1u,std::min(m_sdf.triangle_octree->depth(), m_sdf.brick_level)),
							// 	m_sdf.brick_data.data(),
							// 	m_sdf.brick_res,
							// 	m_sdf.brick_quantise_bits
							// );
						} else {
							m_sdf.triangle_bvh->signed_distance_gpu(
								n_elements,
								m_sdf.mesh_sdf_mode,
								(Vector3f*)positions,
								distances,
								m_sdf.triangles_gpu.data(),
								false,
								m_stream.get()
							);
						}
					} : (distance_fun_t)[&](uint32_t n_elements, const Vector3f* positions, float* distances, cudaStream_t stream) {
						if (n_elements == 0) {
							return;
						}
						n_elements = next_multiple(n_elements, tcnn::batch_size_granularity);
						GPUMatrix<float> positions_matrix((float*)positions, 3, n_elements);
						GPUMatrix<float, RM> distances_matrix(distances, 1, n_elements);
						m_network->inference(stream, positions_matrix, distances_matrix);
					};

				normals_fun_t normals_fun =
					m_render_ground_truth ? (normals_fun_t)[&](uint32_t n_elements, const Vector3f* positions, Vector3f* normals, cudaStream_t stream) {
						// NO-OP. Normals will automatically be populated by raytrace
					} : (normals_fun_t)[&](uint32_t n_elements, const Vector3f* positions, Vector3f* normals, cudaStream_t stream) {
						if (n_elements == 0) {
							return;
						}

						n_elements = next_multiple(n_elements, tcnn::batch_size_granularity);

						GPUMatrix<float> positions_matrix((float*)positions, 3, n_elements);
						GPUMatrix<float> normals_matrix((float*)normals, 3, n_elements);
						m_network->input_gradient(stream, 0, positions_matrix, normals_matrix);
					};

				render_sdf(
					distance_fun,
					normals_fun,
					render_buffer,
					max_res,
					focal_length,
					camera_matrix0,
					screen_center,
					m_stream.get()
				);
			}
			break;
		case ETestbedMode::Image:
			render_image(render_buffer, m_stream.get());
			break;
		case ETestbedMode::Volume:
			render_volume(render_buffer, focal_length, camera_matrix0, screen_center, m_stream.get());
			break;
		default:
			throw std::runtime_error{"Invalid render mode."};
	}

	render_buffer.set_color_space(m_color_space);
	render_buffer.set_tonemap_curve(m_tonemap_curve);

	// Prepare DLSS data: motion vectors, scaled depth, exposure
	if (render_buffer.dlss()) {
		auto res = render_buffer.in_resolution();

		bool distortion = m_testbed_mode == ETestbedMode::Nerf && m_nerf.render_with_lens_distortion;

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };

		dlss_prep_kernel<<<blocks, threads, 0, m_stream.get()>>>(
			m_testbed_mode,
			res,
			render_buffer.spp(),
			focal_length,
			screen_center,
			m_parallax_shift,
			m_snap_to_pixel_centers,
			render_buffer.depth_buffer(),
			camera_matrix0,
			m_prev_camera,
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure(),
			distortion ? m_nerf.render_lens : Lens{},
			m_scale,
			m_prev_scale,
			m_image.pos,
			m_image.prev_pos,
			m_image.resolution,
			m_quilting_dims
		);

		render_buffer.set_dlss_sharpening(m_dlss_sharpening);
	}

	m_prev_camera = camera_matrix0;
	m_prev_scale = m_scale;
	m_image.prev_pos = m_image.pos;

	render_buffer.accumulate(m_exposure, m_stream.get());
	render_buffer.tonemap(m_exposure, m_background_color, to_srgb ? EColorSpace::SRGB : EColorSpace::Linear, m_stream.get());

	if (m_testbed_mode == ETestbedMode::Nerf) {
		// Overlay the ground truth image if requested
		if (m_render_ground_truth) {
			auto const& metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			if (m_ground_truth_render_mode == EGroundTruthRenderMode::Shade) {
				render_buffer.overlay_image(
					m_ground_truth_alpha,
					Array3f::Constant(m_exposure) + m_nerf.training.cam_exposure[m_nerf.training.view].variable(),
					m_background_color,
					to_srgb ? EColorSpace::SRGB : EColorSpace::Linear,
					metadata.pixels,
					metadata.image_data_type,
					metadata.resolution,
					m_fov_axis,
					m_zoom,
					Vector2f::Constant(0.5f),
					m_stream.get()
				);
			} else if (m_ground_truth_render_mode == EGroundTruthRenderMode::Depth && metadata.depth) {
				render_buffer.overlay_depth(
					m_ground_truth_alpha,
					metadata.depth,
					1.0f/m_nerf.training.dataset.scale,
					metadata.resolution,
					m_fov_axis,
					m_zoom,
					Vector2f::Constant(0.5f),
					m_stream.get()
				);
			}
		}

		// Visualize the accumulated error map if requested
		if (m_nerf.training.render_error_overlay) {
			const float* err_data = m_nerf.training.error_map.data.data();
			Vector2i error_map_res = m_nerf.training.error_map.resolution;
			if (m_render_ground_truth) {
				err_data = m_nerf.training.dataset.sharpness_data.data();
				error_map_res = m_nerf.training.dataset.sharpness_resolution;
			}
			size_t emap_size = error_map_res.x() * error_map_res.y();
			err_data += emap_size * m_nerf.training.view;
			static GPUMemory<float> average_error;
			average_error.enlarge(1);
			average_error.memset(0);
			const float* aligned_err_data_s = (const float*)(((size_t)err_data)&~15);
			const float* aligned_err_data_e = (const float*)(((size_t)(err_data+emap_size))&~15);
			size_t reduce_size = aligned_err_data_e - aligned_err_data_s;
			reduce_sum(aligned_err_data_s, [reduce_size] __device__ (float val) { return max(val,0.f) / (reduce_size); }, average_error.data(), reduce_size, m_stream.get());
			auto const &metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			render_buffer.overlay_false_color(metadata.resolution, to_srgb, m_fov_axis, m_stream.get(), err_data, error_map_res, average_error.data(), m_nerf.training.error_overlay_brightness, m_render_ground_truth);
		}
	}

	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}

void Testbed::determine_autofocus_target_from_pixel(const Vector2i& focus_pixel) {
	float depth;

	const auto& surface = m_render_surfaces.front();
	if (surface.depth_buffer()) {
		auto res = surface.in_resolution();
		Vector2i depth_pixel = focus_pixel.cast<float>().cwiseProduct(res.cast<float>()).cwiseQuotient(m_window_res.cast<float>()).cast<int>();
		depth_pixel = depth_pixel.cwiseMin(res).cwiseMax(0);

		CUDA_CHECK_THROW(cudaMemcpy(&depth, surface.depth_buffer() + depth_pixel.x() + depth_pixel.y() * res.x(), sizeof(float), cudaMemcpyDeviceToHost));
	} else {
		depth = m_scale;
	}

	auto ray = pixel_to_ray_pinhole(0, focus_pixel, m_window_res, calc_focal_length(m_window_res, m_fov_axis, m_zoom), m_smoothed_camera, render_screen_center());

	m_autofocus_target = ray.o + ray.d * depth;
	m_autofocus = true; // If someone shift-clicked, that means they want the AUTOFOCUS
}

void Testbed::autofocus() {
	float new_slice_plane_z = std::max(view_dir().dot(m_autofocus_target - view_pos()), 0.1f) - m_scale;
	if (new_slice_plane_z != m_slice_plane_z) {
		m_slice_plane_z = new_slice_plane_z;
		if (m_aperture_size != 0.0f) {
			reset_accumulation();
		}
	}
}

Testbed::LevelStats compute_level_stats(const float* params, size_t n_params) {
	Testbed::LevelStats s = {};
	for (size_t i = 0; i < n_params; ++i) {
		float v = params[i];
		float av = fabsf(v);
		if (av < 0.00001f) {
			s.numzero++;
		} else {
			if (s.count == 0) s.min = s.max = v;
			s.count++;
			s.x += v;
			s.xsquared += v * v;
			s.min = min(s.min, v);
			s.max = max(s.max, v);
		}
	}
	return s;
}

void Testbed::gather_histograms() {
	int n_params = (int)m_network->n_params();
	int first_encoder = first_encoder_param();
	int n_encoding_params = n_params - first_encoder;

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc && m_trainer->params()) {
		std::vector<float> grid(n_encoding_params);

		uint32_t m = m_network->layer_sizes().front().first;
		uint32_t n = m_network->layer_sizes().front().second;
		std::vector<float> first_layer_rm(m * n);

		CUDA_CHECK_THROW(cudaMemcpyAsync(grid.data(), m_trainer->params() + first_encoder, grid.size() * sizeof(float), cudaMemcpyDeviceToHost, m_stream.get()));
		CUDA_CHECK_THROW(cudaMemcpyAsync(first_layer_rm.data(), m_trainer->params(), first_layer_rm.size() * sizeof(float), cudaMemcpyDeviceToHost, m_stream.get()));
		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));


		for (int l = 0; l < m_num_levels; ++l) {
			m_level_stats[l] = compute_level_stats(grid.data() + hg_enc->level_params_offset(l), hg_enc->level_n_params(l));
		}

		int numquant = 0;
		m_quant_percent = float(numquant * 100) / (float)n_encoding_params;
		if (m_histo_level < m_num_levels) {
			size_t nperlevel = hg_enc->level_n_params(m_histo_level);
			const float* d = grid.data() + hg_enc->level_params_offset(m_histo_level);
			float scale = 128.f / (m_histo_scale); // fixed scale for now to make it more comparable between levels
			memset(m_histo, 0, sizeof(m_histo));
			for (int i = 0; i < nperlevel; ++i) {
				float v = *d++;
				if (v == 0.f) {
					continue;
				}
				int bin = (int)floor(v * scale + 128.5f);
				if (bin >= 0 && bin <= 256) {
					m_histo[bin]++;
				}
			}
		}
	}
}

// Increment this number when making a change to the snapshot format
static const size_t SNAPSHOT_FORMAT_VERSION = 1;

void Testbed::save_snapshot(const std::string& filepath_string, bool include_optimizer_state) {
	fs::path filepath = filepath_string;
	m_network_config["snapshot"] = m_trainer->serialize(include_optimizer_state);

	auto& snapshot = m_network_config["snapshot"];
	snapshot["version"] = SNAPSHOT_FORMAT_VERSION;
	snapshot["mode"] = to_string(m_testbed_mode);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		snapshot["density_grid_size"] = NERF_GRIDSIZE();

		GPUMemory<__half> density_grid_fp16(m_nerf.density_grid.size());
		parallel_for_gpu(density_grid_fp16.size(), [density_grid=m_nerf.density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
			density_grid_fp16[i] = (__half)density_grid[i];
		});

		snapshot["density_grid_binary"] = density_grid_fp16;
		snapshot["nerf"]["aabb_scale"] = m_nerf.training.dataset.aabb_scale;
	}

	snapshot["training_step"] = m_training_step;
	snapshot["loss"] = m_loss_scalar.val();
	snapshot["aabb"] = m_aabb;
	snapshot["bounding_radius"] = m_bounding_radius;
	to_json(snapshot["render_aabb_to_local"], m_render_aabb_to_local);
	snapshot["render_aabb"] = m_render_aabb;

	if (m_testbed_mode == ETestbedMode::Nerf) {
		snapshot["nerf"]["rgb"]["rays_per_batch"] = m_nerf.training.counters_rgb.rays_per_batch;
		snapshot["nerf"]["rgb"]["measured_batch_size"] = m_nerf.training.counters_rgb.measured_batch_size;
		snapshot["nerf"]["rgb"]["measured_batch_size_before_compaction"] = m_nerf.training.counters_rgb.measured_batch_size_before_compaction;
		snapshot["nerf"]["dataset"] = m_nerf.training.dataset;
	}

	m_network_config_path = filepath;
	std::ofstream f(m_network_config_path.str(), std::ios::out | std::ios::binary);
	json::to_msgpack(m_network_config, f);
}

void Testbed::load_snapshot(const std::string& filepath_string) {
	auto config = load_network_config(filepath_string);
	if (!config.contains("snapshot")) {
		throw std::runtime_error{fmt::format("File '{}' does not contain a snapshot.", filepath_string)};
	}

	m_network_config_path = filepath_string;

	const auto& snapshot = config["snapshot"];

	if (snapshot.value("version", 0) < SNAPSHOT_FORMAT_VERSION) {
		throw std::runtime_error{"Snapshot uses an old format and can not be loaded."};
	}

	if (snapshot.contains("mode")) {
		set_mode(mode_from_string(snapshot["mode"]));
	} else if (snapshot.contains("nerf")) {
		// To be able to load old NeRF snapshots that don't specify their mode yet
		set_mode(ETestbedMode::Nerf);
	} else if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Unknown snapshot mode. Snapshot must be regenerated with a new version of instant-ngp."};
	}

	m_aabb = snapshot.value("aabb", m_aabb);
	m_bounding_radius = snapshot.value("bounding_radius", m_bounding_radius);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (snapshot["density_grid_size"] != NERF_GRIDSIZE()) {
			throw std::runtime_error{"Incompatible grid size."};
		}

		m_nerf.training.counters_rgb.rays_per_batch = snapshot["nerf"]["rgb"]["rays_per_batch"];
		m_nerf.training.counters_rgb.measured_batch_size = snapshot["nerf"]["rgb"]["measured_batch_size"];
		m_nerf.training.counters_rgb.measured_batch_size_before_compaction = snapshot["nerf"]["rgb"]["measured_batch_size_before_compaction"];

		// If we haven't got a nerf dataset loaded, load dataset metadata from the snapshot
		// and render using just that.
		if (m_data_path.empty() && snapshot["nerf"].contains("dataset")) {
			m_nerf.training.dataset = snapshot["nerf"]["dataset"];
			load_nerf();
		} else {
			if (snapshot["nerf"].contains("aabb_scale")) {
				m_nerf.training.dataset.aabb_scale = snapshot["nerf"]["aabb_scale"];
			}
		}

		load_nerf_post();

		GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
		m_nerf.density_grid.resize(density_grid_fp16.size());

		parallel_for_gpu(density_grid_fp16.size(), [density_grid=m_nerf.density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
			density_grid[i] = (float)density_grid_fp16[i];
		});

		if (m_nerf.density_grid.size() == NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * (m_nerf.max_cascade + 1)) {
			update_density_grid_mean_and_bitfield(nullptr);
		} else if (m_nerf.density_grid.size() != 0) {
			// A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
			throw std::runtime_error{"Incompatible number of grid cascades."};
		}
	}

	// Needs to happen after `load_nerf_post()`
	if (snapshot.contains("render_aabb_to_local")) from_json(snapshot.at("render_aabb_to_local"), m_render_aabb_to_local);
	m_render_aabb = snapshot.value("render_aabb", m_render_aabb);

	m_network_config_path = filepath_string;
	m_network_config = config;

	reset_network(false);

	m_training_step = m_network_config["snapshot"]["training_step"];
	m_loss_scalar.set(m_network_config["snapshot"]["loss"]);

	m_trainer->deserialize(m_network_config["snapshot"]);
}

void Testbed::load_camera_path(const std::string& filepath_string) {
	m_camera_path.load(filepath_string, Matrix<float, 3, 4>::Identity());
}

bool Testbed::loop_animation() {
	return m_camera_path.loop;
}

void Testbed::set_loop_animation(bool value) {
	m_camera_path.loop = value;
}

NGP_NAMESPACE_END

