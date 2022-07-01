/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   python_api.cpp
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/thread_pool.h>

#include <json/json.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>

#include <filesystem/path.h>

#ifdef NGP_GUI
#  include <imgui/imgui.h>
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#endif

using namespace tcnn;
using namespace Eigen;
using namespace nlohmann;
namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

NGP_NAMESPACE_BEGIN


void Testbed::Nerf::Training::set_image(int frame_idx, pybind11::array_t<float> img, pybind11::array_t<float> depth_img, float depth_scale) {
	if (frame_idx < 0 || frame_idx >= dataset.n_images) {
		throw std::runtime_error{"Invalid frame index"};
	}

	py::buffer_info img_buf = img.request();

	if (img_buf.ndim != 3) {
		throw std::runtime_error{"image should be (H,W,C) where C=4"};
	}

	if (img_buf.shape[2] != 4) {
		throw std::runtime_error{"image should be (H,W,C) where C=4"};
	}

	py::buffer_info depth_buf = depth_img.request();

	dataset.set_training_image(frame_idx, {img_buf.shape[1], img_buf.shape[0]}, (const void*)img_buf.ptr, (const float*)depth_buf.ptr, depth_scale, false, EImageDataType::Float, EDepthDataType::Float);
}

void Testbed::override_sdf_training_data(py::array_t<float> points, py::array_t<float> distances) {
	py::buffer_info points_buf = points.request();
	py::buffer_info distances_buf = distances.request();

	if (points_buf.ndim != 2 || distances_buf.ndim != 1 || points_buf.shape[0] != distances_buf.shape[0] || points_buf.shape[1] != 3) {
		tlog::error() << "Invalid Points<->Distances data";
		return;
	}

	std::vector<Vector3f> points_cpu(points_buf.shape[0]);
	std::vector<float> distances_cpu(distances_buf.shape[0]);

	for (size_t i = 0; i < points_cpu.size(); ++i) {
		Vector3f pos = *((Vector3f*)points_buf.ptr + i);
		float dist = *((float*)distances_buf.ptr + i);

		pos = (pos - m_raw_aabb.min) / m_sdf.mesh_scale + 0.5f * (Vector3f::Ones() - (m_raw_aabb.max - m_raw_aabb.min) / m_sdf.mesh_scale);
		dist /= m_sdf.mesh_scale;

		points_cpu[i] = pos;
		distances_cpu[i] = dist;
	}

	CUDA_CHECK_THROW(cudaMemcpyAsync(m_sdf.training.positions.data(), points_cpu.data(), points_buf.shape[0] * points_buf.shape[1] * sizeof(float), cudaMemcpyHostToDevice, m_training_stream));
	CUDA_CHECK_THROW(cudaMemcpyAsync(m_sdf.training.distances.data(), distances_cpu.data(), distances_buf.shape[0] * sizeof(float), cudaMemcpyHostToDevice, m_training_stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(m_training_stream));
	m_sdf.training.size = points_buf.shape[0];
	m_sdf.training.idx = 0;
	m_sdf.training.max_size = m_sdf.training.size;
	m_sdf.training.generate_sdf_data_online = false;
}

pybind11::dict Testbed::compute_marching_cubes_mesh(Eigen::Vector3i res3d, BoundingBox aabb, float thresh) {
	Matrix3f render_aabb_to_local = Matrix3f::Identity();
	if (aabb.is_empty()) {
		aabb = m_testbed_mode == ETestbedMode::Nerf ? m_render_aabb : m_aabb;
		render_aabb_to_local = m_render_aabb_to_local;
	}

	marching_cubes(res3d, aabb, render_aabb_to_local, thresh);

	py::array_t<float> cpuverts({(int)m_mesh.verts.size(), 3});
	py::array_t<float> cpunormals({(int)m_mesh.vert_normals.size(), 3});
	py::array_t<float> cpucolors({(int)m_mesh.vert_colors.size(), 3});
	py::array_t<int> cpuindices({(int)m_mesh.indices.size()/3, 3});
	CUDA_CHECK_THROW(cudaMemcpy(cpuverts.request().ptr, m_mesh.verts.data() , m_mesh.verts.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_THROW(cudaMemcpy(cpunormals.request().ptr, m_mesh.vert_normals.data() , m_mesh.vert_normals.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_THROW(cudaMemcpy(cpucolors.request().ptr, m_mesh.vert_colors.data() , m_mesh.vert_colors.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_THROW(cudaMemcpy(cpuindices.request().ptr, m_mesh.indices.data() , m_mesh.indices.size() * sizeof(int), cudaMemcpyDeviceToHost));

	Eigen::Vector3f* ns = (Eigen::Vector3f*)cpunormals.request().ptr;
	for (size_t i = 0; i < m_mesh.vert_normals.size(); ++i) {
		ns[i].normalize();
	}

	return py::dict("V"_a=cpuverts, "N"_a=cpunormals, "C"_a=cpucolors, "F"_a=cpuindices);
}

py::array_t<float> Testbed::render_to_cpu(int width, int height, int spp, bool linear, float start_time, float end_time, float fps, float shutter_fraction) {
	m_windowless_render_surface.resize({width, height});
	m_windowless_render_surface.reset_accumulation();

	if (end_time < 0.f) {
		end_time = start_time;
	}

	auto start_cam_matrix = m_smoothed_camera;

	if (start_time >= 0.f) {
		set_camera_from_time(end_time);
		apply_camera_smoothing(1000.f / fps);
	} else {
		start_cam_matrix = m_smoothed_camera = m_camera;
	}

	auto end_cam_matrix = m_smoothed_camera;

	for (int i = 0; i < spp; ++i) {
		float start_alpha = ((float)i)/(float)spp * shutter_fraction;
		float end_alpha = ((float)i + 1.0f)/(float)spp * shutter_fraction;

		auto sample_start_cam_matrix = log_space_lerp(start_cam_matrix, end_cam_matrix, start_alpha);
		auto sample_end_cam_matrix = log_space_lerp(start_cam_matrix, end_cam_matrix, end_alpha);

		if (start_time >= 0.f) {
			set_camera_from_time(start_time + (end_time-start_time) * (start_alpha + end_alpha) / 2.0f);
			m_smoothed_camera = m_camera;
		}

		if (m_autofocus) {
			autofocus();
		}

		render_frame(sample_start_cam_matrix, sample_end_cam_matrix, Eigen::Vector4f::Zero(), m_windowless_render_surface, !linear);
	}

	// For cam smoothing when rendering the next frame.
	m_smoothed_camera = end_cam_matrix;

	py::array_t<float> result({height, width, 4});
	py::buffer_info buf = result.request();

	CUDA_CHECK_THROW(cudaMemcpy2DFromArray(buf.ptr, width * sizeof(float) * 4, m_windowless_render_surface.surface_provider().array(), 0, 0, width * sizeof(float) * 4, height, cudaMemcpyDeviceToHost));
	return result;
}

py::array_t<float> Testbed::render_with_rolling_shutter_to_cpu(const Eigen::Matrix<float, 3, 4>& camera_transform_start, const Eigen::Matrix<float, 3, 4>& camera_transform_end, const Eigen::Vector4f& rolling_shutter, int width, int height, int spp, bool linear) {
	m_windowless_render_surface.resize({width, height});
	m_windowless_render_surface.reset_accumulation();
	for (int i = 0; i < spp; ++i) {
		if (m_autofocus) {
			autofocus();
		}
		render_frame(m_nerf.training.dataset.nerf_matrix_to_ngp(camera_transform_start), m_nerf.training.dataset.nerf_matrix_to_ngp(camera_transform_end), rolling_shutter, m_windowless_render_surface, !linear);
	}
	py::array_t<float> result({height, width, 4});
	py::buffer_info buf = result.request();
	CUDA_CHECK_THROW(cudaMemcpy2DFromArray(buf.ptr, width * sizeof(float) * 4, m_windowless_render_surface.surface_provider().array(), 0, 0, width * sizeof(float) * 4, height, cudaMemcpyDeviceToHost));
	return result;
}

#ifdef NGP_GUI
py::array_t<float> Testbed::screenshot(bool linear) const {
	std::vector<float> tmp(m_window_res.prod() * 4);
	glReadPixels(0, 0, m_window_res.x(), m_window_res.y(), GL_RGBA, GL_FLOAT, tmp.data());

	py::array_t<float> result({m_window_res.y(), m_window_res.x(), 4});
	py::buffer_info buf = result.request();
	float* data = (float*)buf.ptr;

	// Linear, alpha premultiplied, Y flipped
	ThreadPool pool;
	pool.parallelFor<size_t>(0, m_window_res.y(), [&](size_t y) {
		size_t base = y * m_window_res.x();
		size_t base_reverse = (m_window_res.y() - y - 1) * m_window_res.x();
		for (uint32_t x = 0; x < m_window_res.x(); ++x) {
			size_t px = base + x;
			size_t px_reverse = base_reverse + x;
			data[px_reverse*4+0] = linear ? srgb_to_linear(tmp[px*4+0]) : tmp[px*4+0];
			data[px_reverse*4+1] = linear ? srgb_to_linear(tmp[px*4+1]) : tmp[px*4+1];
			data[px_reverse*4+2] = linear ? srgb_to_linear(tmp[px*4+2]) : tmp[px*4+2];
			data[px_reverse*4+3] = tmp[px*4+3];
		}
	});

	return result;
}
#endif

PYBIND11_MODULE(pyngp, m) {
	m.doc() = "Instant neural graphics primitives";

	m.def("free_temporary_memory", &tcnn::free_all_gpu_memory_arenas);

	py::enum_<ETestbedMode>(m, "TestbedMode")
		.value("Nerf", ETestbedMode::Nerf)
		.value("Sdf", ETestbedMode::Sdf)
		.value("Image", ETestbedMode::Image)
		.value("Volume", ETestbedMode::Volume)
		.export_values();

	py::enum_<ERenderMode>(m, "RenderMode")
		.value("AO", ERenderMode::AO)
		.value("Shade", ERenderMode::Shade)
		.value("Normals", ERenderMode::Normals)
		.value("Positions", ERenderMode::Positions)
		.value("Depth", ERenderMode::Depth)
		.value("Distortion", ERenderMode::Distortion)
		.value("Cost", ERenderMode::Cost)
		.value("Slice", ERenderMode::Slice)
		.export_values();

	py::enum_<ERandomMode>(m, "RandomMode")
		.value("Random", ERandomMode::Random)
		.value("Halton", ERandomMode::Halton)
		.value("Sobol", ERandomMode::Sobol)
		.value("Stratified", ERandomMode::Stratified)
		.export_values();

	py::enum_<ELossType>(m, "LossType")
		.value("L2", ELossType::L2)
		.value("L1", ELossType::L1)
		.value("Mape", ELossType::Mape)
		.value("Smape", ELossType::Smape)
		.value("Huber", ELossType::Huber)
		// Legacy: we used to refer to the Huber loss
		// (L2 near zero, L1 further away) as "SmoothL1".
		.value("SmoothL1", ELossType::Huber)
		.value("LogL1", ELossType::LogL1)
		.value("RelativeL2", ELossType::RelativeL2)
		.export_values();

	py::enum_<ESDFGroundTruthMode>(m, "SDFGroundTruthMode")
		.value("RaytracedMesh", ESDFGroundTruthMode::RaytracedMesh)
		.value("SpheretracedMesh", ESDFGroundTruthMode::SpheretracedMesh)
		.value("SDFBricks", ESDFGroundTruthMode::SDFBricks)
		.export_values();

	py::enum_<ENerfActivation>(m, "NerfActivation")
		.value("None", ENerfActivation::None)
		.value("ReLU", ENerfActivation::ReLU)
		.value("Logistic", ENerfActivation::Logistic)
		.value("Exponential", ENerfActivation::Exponential)
		.export_values();

	py::enum_<EMeshSdfMode>(m, "MeshSdfMode")
		.value("Watertight", EMeshSdfMode::Watertight)
		.value("Raystab", EMeshSdfMode::Raystab)
		.value("PathEscape", EMeshSdfMode::PathEscape)
		.export_values();

	py::enum_<EColorSpace>(m, "ColorSpace")
		.value("Linear", EColorSpace::Linear)
		.value("SRGB", EColorSpace::SRGB)
		.export_values();

	py::enum_<ETonemapCurve>(m, "TonemapCurve")
		.value("Identity", ETonemapCurve::Identity)
		.value("ACES", ETonemapCurve::ACES)
		.value("Hable", ETonemapCurve::Hable)
		.value("Reinhard", ETonemapCurve::Reinhard)
		.export_values();

	py::enum_<ECameraDistortionMode>(m, "ECameraDistortionMode")
		.value("None", ECameraDistortionMode::None)
		.value("Iterative", ECameraDistortionMode::Iterative)
		.value("FTheta", ECameraDistortionMode::FTheta)
		.value("LatLong", ECameraDistortionMode::LatLong)
		.export_values();

	py::class_<BoundingBox>(m, "BoundingBox")
		.def(py::init<>())
		.def(py::init<const Vector3f&, const Vector3f&>())
		.def("center", &BoundingBox::center)
		.def("contains", &BoundingBox::contains)
		.def("diag", &BoundingBox::diag)
		.def("distance", &BoundingBox::distance)
		.def("distance_sq", &BoundingBox::distance_sq)
		.def("enlarge", py::overload_cast<const Vector3f&>(&BoundingBox::enlarge))
		.def("enlarge", py::overload_cast<const BoundingBox&>(&BoundingBox::enlarge))
		.def("get_vertices", &BoundingBox::get_vertices)
		.def("inflate", &BoundingBox::inflate)
		.def("intersection", &BoundingBox::intersection)
		.def("intersects", py::overload_cast<const BoundingBox&>(&BoundingBox::intersects, py::const_))
		.def("ray_intersect", &BoundingBox::ray_intersect)
		.def("relative_pos", &BoundingBox::relative_pos)
		.def("signed_distance", &BoundingBox::signed_distance)
		.def_readwrite("min", &BoundingBox::min)
		.def_readwrite("max", &BoundingBox::max)
		;

	py::class_<Testbed> testbed(m, "Testbed");
	testbed
		.def(py::init<ETestbedMode>())
		.def(py::init<ETestbedMode, const std::string&, const std::string&>())
		.def(py::init<ETestbedMode, const std::string&, const json&>())
		.def("create_empty_nerf_dataset", &Testbed::create_empty_nerf_dataset, "Allocate memory for a nerf dataset with a given size", py::arg("n_images"), py::arg("aabb_scale")=1, py::arg("is_hdr")=false)
		.def("load_training_data", &Testbed::load_training_data, py::call_guard<py::gil_scoped_release>(), "Load training data from a given path.")
		.def("clear_training_data", &Testbed::clear_training_data, "Clears training data to free up GPU memory.")
		// General control
#ifdef NGP_GUI
		.def("init_window", &Testbed::init_window, "Init a GLFW window that shows real-time progress and a GUI. 'second_window' creates a second copy of the output in its own window",
			py::arg("width"),
			py::arg("height"),
			py::arg("hidden") = false,
			py::arg("second_window") = false
		)
		.def_readwrite("keyboard_event_callback", &Testbed::m_keyboard_event_callback)
		.def("is_key_pressed", [](py::object& obj, char key) { return ImGui::IsKeyPressed(key); })
		.def("is_key_down", [](py::object& obj, char key) { return ImGui::IsKeyDown(key); })
		.def("is_alt_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Alt; })
		.def("is_ctrl_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Ctrl; })
		.def("is_shift_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Shift; })
		.def("is_super_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Super; })
		.def("screenshot", &Testbed::screenshot, "Takes a screenshot of the current window contents.", py::arg("linear")=true)
#endif
		.def("want_repl", &Testbed::want_repl, "returns true if the user clicked the 'I want a repl' button")
		.def("frame", &Testbed::frame, py::call_guard<py::gil_scoped_release>(), "Process a single frame. Renders if a window was previously created.")
		.def("render", &Testbed::render_to_cpu, "Renders an image at the requested resolution. Does not require a window.",
			py::arg("width") = 1920,
			py::arg("height") = 1080,
			py::arg("spp") = 1,
			py::arg("linear") = true,
			py::arg("start_t") = -1.f,
			py::arg("end_t") = -1.f,
			py::arg("fps") = 30.f,
			py::arg("shutter_fraction") = 1.0f
		)
		.def("render_with_rolling_shutter", &Testbed::render_with_rolling_shutter_to_cpu, "Renders an image at the requested resolution. Does not require a window. Supports rolling shutter, with per ray time being computed as A+B*u+C*v+D*t for [A,B,C,D]",
			py::arg("transform_matrix_start"),
			py::arg("transform_matrix_end"),
			py::arg("rolling_shutter") = Eigen::Vector4f::Zero(),
			py::arg("width") = 1920,
			py::arg("height") = 1080,
			py::arg("spp") = 1,
			py::arg("linear") = true
		)
		.def("destroy_window", &Testbed::destroy_window, "Destroy the window again.")
		.def("train", &Testbed::train, py::call_guard<py::gil_scoped_release>(), "Perform a specified number of training steps.")
		.def("reset", &Testbed::reset_network, "Reset training.")
		.def("reset_accumulation", &Testbed::reset_accumulation, "Reset rendering accumulation.")
		.def("reload_network_from_file", &Testbed::reload_network_from_file, py::arg("path")="", "Reload the network from a config file.")
		.def("reload_network_from_json", &Testbed::reload_network_from_json, "Reload the network from a json object.")
		.def("override_sdf_training_data", &Testbed::override_sdf_training_data, "Override the training data for learning a signed distance function")
		.def("calculate_iou", &Testbed::calculate_iou, "Calculate the intersection over union error value",
			py::arg("n_samples") = 128*1024*1024,
			py::arg("scale_existing_results_factor") = 0.0f,
			py::arg("blocking") = true,
			py::arg("force_use_octree") = true
		)
		.def("n_params", &Testbed::n_params, "Number of trainable parameters")
		.def("n_encoding_params", &Testbed::n_encoding_params, "Number of trainable parameters in the encoding")
		.def("save_snapshot", &Testbed::save_snapshot, py::arg("path"), py::arg("include_optimizer_state")=false, "Save a snapshot of the currently trained model")
		.def("load_snapshot", &Testbed::load_snapshot, py::arg("path"), "Load a previously saved snapshot")
		.def("load_camera_path", &Testbed::load_camera_path, "Load a camera path", py::arg("path"))
		.def("compute_and_save_png_slices", &Testbed::compute_and_save_png_slices,
			py::arg("filename"),
			py::arg("resolution") = Eigen::Vector3i::Constant(256),
			py::arg("aabb") = BoundingBox{},
			py::arg("thresh") = std::numeric_limits<float>::max(),
			py::arg("density_range") = 4.f,
			py::arg("flip_y_and_z_axes") = false,
			"Compute & save a PNG file representing the 3D density or distance field from the current SDF or NeRF model. "
		)
		.def("compute_and_save_marching_cubes_mesh", &Testbed::compute_and_save_marching_cubes_mesh,
			py::arg("filename"),
			py::arg("resolution") = Eigen::Vector3i::Constant(256),
			py::arg("aabb") = BoundingBox{},
			py::arg("thresh") = std::numeric_limits<float>::max(),
			py::arg("generate_uvs_for_obj_file") = false,
			"Compute & save a marching cubes mesh from the current SDF or NeRF model. "
			"Supports OBJ and PLY format. Note that UVs are only supported by OBJ files. "
			"`thresh` is the density threshold; use 0 for SDF; 2.5 works well for NeRF. "
			"If the aabb parameter specifies an inside-out (\"empty\") box (default), the current render_aabb bounding box is used."
		)
		.def("compute_marching_cubes_mesh", &Testbed::compute_marching_cubes_mesh,
			py::arg("resolution") = Eigen::Vector3i::Constant(256),
			py::arg("aabb") = BoundingBox{},
			py::arg("thresh") = std::numeric_limits<float>::max(),
			"Compute a marching cubes mesh from the current SDF or NeRF model. "
			"Returns a python dict with numpy arrays V (vertices), N (vertex normals), C (vertex colors), and F (triangular faces). "
			"`thresh` is the density threshold; use 0 for SDF; 2.5 works well for NeRF. "
			"If the aabb parameter specifies an inside-out (\"empty\") box (default), the current render_aabb bounding box is used."
		)
		;

	// Interesting members.
	testbed
		.def_readwrite("dynamic_res", &Testbed::m_dynamic_res)
		.def_readwrite("dynamic_res_target_fps", &Testbed::m_dynamic_res_target_fps)
		.def_readwrite("fixed_res_factor", &Testbed::m_fixed_res_factor)
		.def_readwrite("background_color", &Testbed::m_background_color)
		.def_readwrite("shall_train", &Testbed::m_train)
		.def_readwrite("shall_train_encoding", &Testbed::m_train_encoding)
		.def_readwrite("shall_train_network", &Testbed::m_train_network)
		.def_readwrite("render_groundtruth", &Testbed::m_render_ground_truth)
		.def_readwrite("render_mode", &Testbed::m_render_mode)
		.def_readwrite("slice_plane_z", &Testbed::m_slice_plane_z)
		.def_readwrite("dof", &Testbed::m_dof)
		.def_readwrite("autofocus", &Testbed::m_autofocus)
		.def_readwrite("autofocus_target", &Testbed::m_autofocus_target)
		.def_readwrite("floor_enable", &Testbed::m_floor_enable)
		.def_readwrite("exposure", &Testbed::m_exposure)
		.def_property("scale", &Testbed::scale, &Testbed::set_scale)
		.def_readonly("bounding_radius", &Testbed::m_bounding_radius)
		.def_readwrite("render_aabb", &Testbed::m_render_aabb)
		.def_readwrite("render_aabb_to_local", &Testbed::m_render_aabb_to_local)
		.def_readwrite("aabb", &Testbed::m_aabb)
		.def_readwrite("raw_aabb", &Testbed::m_raw_aabb)
		.def_property("fov", &Testbed::fov, &Testbed::set_fov)
		.def_property("fov_xy", &Testbed::fov_xy, &Testbed::set_fov_xy)
		.def_readwrite("fov_axis", &Testbed::m_fov_axis)
		.def_readwrite("zoom", &Testbed::m_zoom)
		.def_readwrite("screen_center", &Testbed::m_screen_center)
		.def("set_nerf_camera_matrix", &Testbed::set_nerf_camera_matrix)
		.def("set_camera_to_training_view", &Testbed::set_camera_to_training_view)
		.def("compute_image_mse", &Testbed::compute_image_mse,
			py::arg("quantize") = false
		)
		.def_readwrite("camera_matrix", &Testbed::m_camera)
		.def_readwrite("up_dir", &Testbed::m_up_dir)
		.def_readwrite("sun_dir", &Testbed::m_sun_dir)
		.def_property("look_at", &Testbed::look_at, &Testbed::set_look_at)
		.def_property("view_dir", &Testbed::view_dir, &Testbed::set_view_dir)
		.def_readwrite("max_level_rand_training", &Testbed::m_max_level_rand_training)
		.def_readwrite("visualized_dimension", &Testbed::m_visualized_dimension)
		.def_readwrite("visualized_layer", &Testbed::m_visualized_layer)
		.def_property_readonly("loss", [](py::object& obj) { return obj.cast<Testbed&>().m_loss_scalar.val(); })
		.def_readonly("training_step", &Testbed::m_training_step)
		.def_readonly("nerf", &Testbed::m_nerf)
		.def_readonly("sdf", &Testbed::m_sdf)
		.def_readonly("image", &Testbed::m_image)
		.def_readwrite("camera_smoothing", &Testbed::m_camera_smoothing)
		.def_readwrite("display_gui", &Testbed::m_imgui_enabled)
		.def_readwrite("visualize_unit_cube", &Testbed::m_visualize_unit_cube)
		.def_readwrite("snap_to_pixel_centers", &Testbed::m_snap_to_pixel_centers)
		.def_readwrite("parallax_shift", &Testbed::m_parallax_shift)
		.def_readwrite("color_space", &Testbed::m_color_space)
		.def_readwrite("tonemap_curve", &Testbed::m_tonemap_curve)
		.def_property("dlss",
			[](py::object& obj) { return obj.cast<Testbed&>().m_dlss; },
			[](const py::object& obj, bool value) {
				if (value && !obj.cast<Testbed&>().m_dlss_supported) {
					if (obj.cast<Testbed&>().m_render_window) {
						throw std::runtime_error{"DLSS not supported."};
					} else {
						throw std::runtime_error{"DLSS requires a Window to be initialized via `init_window`."};
					}
				}

				obj.cast<Testbed&>().m_dlss = value;
			}
		)
		.def_readwrite("dlss_sharpening", &Testbed::m_dlss_sharpening)
		;

	py::class_<CameraDistortion> camera_distortion(m, "CameraDistortion");
	camera_distortion
		.def_readwrite("mode", &CameraDistortion::mode)
		.def_property_readonly("params", [](py::object& obj) {
			CameraDistortion& o = obj.cast<CameraDistortion&>();
			return py::array{sizeof(o.params)/sizeof(o.params[0]), o.params, obj};
		})
		;

	py::class_<Testbed::Nerf> nerf(testbed, "Nerf");
	nerf
		.def_readonly("training", &Testbed::Nerf::training)
		.def_readwrite("rgb_activation", &Testbed::Nerf::rgb_activation)
		.def_readwrite("density_activation", &Testbed::Nerf::density_activation)
		.def_readwrite("sharpen", &Testbed::Nerf::sharpen)
		.def_readwrite("render_with_camera_distortion", &Testbed::Nerf::render_with_camera_distortion)
		.def_readwrite("render_distortion", &Testbed::Nerf::render_distortion)
		.def_readwrite("rendering_min_transmittance", &Testbed::Nerf::rendering_min_transmittance)
		.def_readwrite("cone_angle_constant", &Testbed::Nerf::cone_angle_constant)
		.def_readwrite("visualize_cameras", &Testbed::Nerf::visualize_cameras)
		;

	py::class_<BRDFParams> brdfparams(m, "BRDFParams");
	brdfparams
		.def_readwrite("metallic", &BRDFParams::metallic)
		.def_readwrite("subsurface", &BRDFParams::subsurface)
		.def_readwrite("specular", &BRDFParams::specular)
		.def_readwrite("roughness", &BRDFParams::roughness)
		.def_readwrite("sheen", &BRDFParams::sheen)
		.def_readwrite("clearcoat", &BRDFParams::clearcoat)
		.def_readwrite("clearcoat_gloss", &BRDFParams::clearcoat_gloss)
		.def_readwrite("basecolor", &BRDFParams::basecolor)
		.def_readwrite("ambientcolor", &BRDFParams::ambientcolor)
		;

	py::class_<TrainingImageMetadata> metadata(m, "TrainingImageMetadata");
	metadata
		.def_readwrite("focal_length", &TrainingImageMetadata::focal_length)
		.def_readwrite("camera_distortion", &TrainingImageMetadata::camera_distortion)
		.def_readwrite("principal_point", &TrainingImageMetadata::principal_point)
		.def_readwrite("rolling_shutter", &TrainingImageMetadata::rolling_shutter)
		.def_readwrite("light_dir", &TrainingImageMetadata::light_dir)
		;

	py::class_<NerfDataset> nerfdataset(m, "NerfDataset");
	nerfdataset
		.def_readonly("metadata", &NerfDataset::metadata)
		.def_readonly("transforms", &NerfDataset::xforms)
		.def_readonly("render_aabb", &NerfDataset::render_aabb)
		.def_readonly("render_aabb_to_local", &NerfDataset::render_aabb_to_local)
		.def_readonly("up", &NerfDataset::up)
		.def_readonly("offset", &NerfDataset::offset)
		.def_readonly("n_images", &NerfDataset::n_images)
		.def_readonly("envmap_resolution", &NerfDataset::envmap_resolution)
		.def_readonly("scale", &NerfDataset::scale)
		.def_readonly("aabb_scale", &NerfDataset::aabb_scale)
		.def_readonly("from_mitsuba", &NerfDataset::from_mitsuba)
		.def_readonly("is_hdr", &NerfDataset::is_hdr)
		;

	py::class_<Testbed::Nerf::Training>(nerf, "Training")
		.def_readwrite("random_bg_color", &Testbed::Nerf::Training::random_bg_color)
		.def_readwrite("n_images_for_training", &Testbed::Nerf::Training::n_images_for_training)
		.def_readwrite("linear_colors", &Testbed::Nerf::Training::linear_colors)
		.def_readwrite("loss_type", &Testbed::Nerf::Training::loss_type)
		.def_readwrite("depth_loss_type", &Testbed::Nerf::Training::depth_loss_type)
		.def_readwrite("snap_to_pixel_centers", &Testbed::Nerf::Training::snap_to_pixel_centers)
		.def_readwrite("optimize_extrinsics", &Testbed::Nerf::Training::optimize_extrinsics)
		.def_readwrite("optimize_extra_dims", &Testbed::Nerf::Training::optimize_extra_dims)
		.def_readwrite("optimize_exposure", &Testbed::Nerf::Training::optimize_exposure)
		.def_readwrite("optimize_distortion", &Testbed::Nerf::Training::optimize_distortion)
		.def_readwrite("optimize_focal_length", &Testbed::Nerf::Training::optimize_focal_length)
		.def_readwrite("n_steps_between_cam_updates", &Testbed::Nerf::Training::n_steps_between_cam_updates)
		.def_readwrite("sample_focal_plane_proportional_to_error", &Testbed::Nerf::Training::sample_focal_plane_proportional_to_error)
		.def_readwrite("sample_image_proportional_to_error", &Testbed::Nerf::Training::sample_image_proportional_to_error)
		.def_readwrite("include_sharpness_in_error", &Testbed::Nerf::Training::include_sharpness_in_error)
		.def_readonly("transforms", &Testbed::Nerf::Training::transforms)
		//.def_readonly("focal_lengths", &Testbed::Nerf::Training::focal_lengths) // use training.dataset.metadata instead
		.def_readwrite("near_distance", &Testbed::Nerf::Training::near_distance)
		.def_readwrite("density_grid_decay", &Testbed::Nerf::Training::density_grid_decay)
		.def_readwrite("extrinsic_l2_reg", &Testbed::Nerf::Training::extrinsic_l2_reg)
		.def_readwrite("extrinsic_learning_rate", &Testbed::Nerf::Training::extrinsic_learning_rate)
		.def_readwrite("intrinsic_l2_reg", &Testbed::Nerf::Training::intrinsic_l2_reg)
		.def_readwrite("exposure_l2_reg", &Testbed::Nerf::Training::exposure_l2_reg)
		.def_readwrite("depth_supervision_lambda", &Testbed::Nerf::Training::depth_supervision_lambda)
		.def_readonly("dataset", &Testbed::Nerf::Training::dataset)
		.def("set_camera_intrinsics", &Testbed::Nerf::Training::set_camera_intrinsics,
			py::arg("frame_idx"),
			py::arg("fx")=0.f, py::arg("fy")=0.f,
			py::arg("cx")=-0.5f, py::arg("cy")=-0.5f,
			py::arg("k1")=0.f, py::arg("k2")=0.f,
			py::arg("p1")=0.f, py::arg("p2")=0.f,
			"Set up the camera intrinsics for the given training image index."
		)
		.def("set_camera_extrinsics", &Testbed::Nerf::Training::set_camera_extrinsics,
			py::arg("frame_idx"),
			py::arg("camera_to_world"),
			py::arg("convert_to_ngp")=true,
			"Set up the camera extrinsics for the given training image index, from the given 3x4 transformation matrix."
		)
		.def("get_camera_extrinsics", &Testbed::Nerf::Training::get_camera_extrinsics, py::arg("frame_idx"), "return the 3x4 transformation matrix of given training frame")
		.def("set_image", &Testbed::Nerf::Training::set_image,
			py::arg("frame_idx"),
			py::arg("img"),
			py::arg("depth_img"),
			py::arg("depth_scale")=1.0f,
			"set one of the training images. must be a floating point numpy array of (H,W,C) with 4 channels; linear color space; W and H must match image size of the rest of the dataset"
		)
		;

	py::class_<Testbed::Sdf> sdf(testbed, "Sdf");
	sdf
		.def_readonly("training", &Testbed::Sdf::training)
		.def_readwrite("mesh_sdf_mode", &Testbed::Sdf::mesh_sdf_mode)
		.def_readwrite("mesh_scale", &Testbed::Sdf::mesh_scale)
		.def_readwrite("analytic_normals", &Testbed::Sdf::analytic_normals)
		.def_readwrite("shadow_sharpness", &Testbed::Sdf::shadow_sharpness)
		.def_readwrite("fd_normals_epsilon", &Testbed::Sdf::fd_normals_epsilon)
		.def_readwrite("use_triangle_octree", &Testbed::Sdf::use_triangle_octree)
		.def_readwrite("zero_offset", &Testbed::Sdf::zero_offset)
		.def_readwrite("distance_scale", &Testbed::Sdf::distance_scale)
		.def_readwrite("calculate_iou_online", &Testbed::Sdf::calculate_iou_online)
		.def_readwrite("groundtruth_mode", &Testbed::Sdf::groundtruth_mode)
		.def_readwrite("brick_level", &Testbed::Sdf::brick_level)
		.def_readonly("brick_res", &Testbed::Sdf::brick_res)
		.def_readwrite("brdf", &Testbed::Sdf::brdf)
		;

	py::class_<Testbed::Sdf::Training>(sdf, "Training")
		.def_readwrite("generate_sdf_data_online", &Testbed::Sdf::Training::generate_sdf_data_online)
		.def_readwrite("surface_offset_scale", &Testbed::Sdf::Training::surface_offset_scale)
		;

	py::class_<Testbed::Image> image(testbed, "Image");
	image
		.def_readonly("training", &Testbed::Image::training)
		.def_readwrite("random_mode", &Testbed::Image::random_mode)
		.def_readwrite("pos", &Testbed::Image::pos)
		;

	py::class_<Testbed::Image::Training>(image, "Training")
		.def_readwrite("snap_to_pixel_centers", &Testbed::Image::Training::snap_to_pixel_centers)
		.def_readwrite("linear_colors", &Testbed::Image::Training::linear_colors)
		;
}

NGP_NAMESPACE_END
