/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/camera_path.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/sdf.h>
#include <neural-graphics-primitives/shared_queue.h>
#include <neural-graphics-primitives/thread_pool.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>

#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <json/json.hpp>

#include <filesystem/path.h>

#ifdef NGP_PYTHON
#  include <pybind11/pybind11.h>
#  include <pybind11/numpy.h>
#endif

#include <thread>

struct GLFWwindow;

TCNN_NAMESPACE_BEGIN
template <typename T> class Loss;
template <typename T> class Optimizer;
template <typename T> class Encoding;
template <typename T, typename PARAMS_T> class Network;
template <typename T, typename PARAMS_T, typename COMPUTE_T> class Trainer;
template <uint32_t N_DIMS, uint32_t RANK, typename T> class TrainableBuffer;
TCNN_NAMESPACE_END

NGP_NAMESPACE_BEGIN

template <typename T> class NerfNetwork;
class TriangleOctree;
class TriangleBvh;
struct Triangle;
class GLTexture;

class Testbed {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Testbed(ETestbedMode mode = ETestbedMode::None);
	~Testbed();
	Testbed(ETestbedMode mode, const std::string& data_path) : Testbed(mode) { load_training_data(data_path); }
	Testbed(ETestbedMode mode, const std::string& data_path, const std::string& network_config_path) : Testbed(mode, data_path) { reload_network_from_file(network_config_path); }
	Testbed(ETestbedMode mode, const std::string& data_path, const nlohmann::json& network_config) : Testbed(mode, data_path) { reload_network_from_json(network_config); }

	bool clear_tmp_dir();
	void load_training_data(const std::string& data_path);
	void clear_training_data();

	void set_mode(ETestbedMode mode);

	using distance_fun_t = std::function<void(uint32_t, const Eigen::Vector3f*, float*, cudaStream_t)>;
	using normals_fun_t = std::function<void(uint32_t, const Eigen::Vector3f*, Eigen::Vector3f*, cudaStream_t)>;

	class SphereTracer {
	public:
		SphereTracer() {}

		void init_rays_from_camera(
			uint32_t spp,
			const Eigen::Vector2i& resolution,
			const Eigen::Vector2f& focal_length,
			const Eigen::Matrix<float, 3, 4>& camera_matrix,
			const Eigen::Vector2f& screen_center,
			const Eigen::Vector3f& parallax_shift,
			bool snap_to_pixel_centers,
			const BoundingBox& aabb,
			float floor_y,
			float near_distance,
			float plane_z,
			float aperture_size,
			const float* envmap_data,
			const Eigen::Vector2i& envmap_resolution,
			Eigen::Array4f* frame_buffer,
			float* depth_buffer,
			const TriangleOctree* octree,
			uint32_t n_octree_levels,
			cudaStream_t stream
		);

		void init_rays_from_data(uint32_t n_elements, const RaysSdfSoa& data, cudaStream_t stream);
		uint32_t trace_bvh(TriangleBvh* bvh, const Triangle* triangles, cudaStream_t stream);
		uint32_t trace(
			const distance_fun_t& distance_function,
			float zero_offset,
			float distance_scale,
			float maximum_distance,
			const BoundingBox& aabb,
			const float floor_y,
			const TriangleOctree* octree,
			uint32_t n_octree_levels,
			cudaStream_t stream
		);
		void enlarge(size_t n_elements, cudaStream_t stream);
		RaysSdfSoa& rays_hit() { return m_rays_hit; }
		RaysSdfSoa& rays_init() { return m_rays[0];	}
		uint32_t n_rays_initialized() const { return m_n_rays_initialized; }
		void set_trace_shadow_rays(bool val) { m_trace_shadow_rays = val; }
		void set_shadow_sharpness(float val) { m_shadow_sharpness = val; }
	private:
		RaysSdfSoa m_rays[2];
		RaysSdfSoa m_rays_hit;
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;

		uint32_t m_n_rays_initialized = 0;
		float m_shadow_sharpness = 2048.f;
		bool m_trace_shadow_rays = false;

		tcnn::GPUMemoryArena::Allocation m_scratch_alloc;
	};

	class NerfTracer {
	public:
		NerfTracer() {}

		void init_rays_from_camera(
			uint32_t spp,
			uint32_t padded_output_width,
			uint32_t n_extra_dims,
			const Eigen::Vector2i& resolution,
			const Eigen::Vector2f& focal_length,
			const Eigen::Matrix<float, 3, 4>& camera_matrix0,
			const Eigen::Matrix<float, 3, 4>& camera_matrix1,
			const Eigen::Vector4f& rolling_shutter,
			const Eigen::Vector2f& screen_center,
			const Eigen::Vector3f& parallax_shift,
			const Eigen::Vector2i& quilting_dims,
			bool snap_to_pixel_centers,
			const BoundingBox& render_aabb,
			const Eigen::Matrix3f& render_aabb_to_local,
			float near_distance,
			float plane_z,
			float aperture_size,
			const Lens& lens,
			const float* envmap_data,
			const Eigen::Vector2i& envmap_resolution,
			const float* distortion_data,
			const Eigen::Vector2i& distortion_resolution,
			Eigen::Array4f* frame_buffer,
			float* depth_buffer,
			uint8_t* grid,
			int show_accel,
			float cone_angle_constant,
			ERenderMode render_mode,
			cudaStream_t stream
		);

		uint32_t trace(
			NerfNetwork<precision_t>& network,
			const BoundingBox& render_aabb,
			const Eigen::Matrix3f& render_aabb_to_local,
			const BoundingBox& train_aabb,
			const uint32_t n_training_images,
			const TrainingXForm* training_xforms,
			const Eigen::Vector2f& focal_length,
			float cone_angle_constant,
			const uint8_t* grid,
			ERenderMode render_mode,
			const Eigen::Matrix<float, 3, 4> &camera_matrix,
			float depth_scale,
			int visualized_layer,
			int visualized_dim,
			ENerfActivation rgb_activation,
			ENerfActivation density_activation,
			int show_accel,
			float min_transmittance,
			float glow_y_cutoff,
			int glow_mode,
			const float* extra_dims_gpu,
			cudaStream_t stream
		);

		void enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream);
		RaysNerfSoa& rays_hit() { return m_rays_hit; }
		RaysNerfSoa& rays_init() { return m_rays[0]; }
		uint32_t n_rays_initialized() const { return m_n_rays_initialized; }

	private:
		RaysNerfSoa m_rays[2];
		RaysNerfSoa m_rays_hit;
		precision_t* m_network_output;
		float* m_network_input;
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;
		uint32_t m_n_rays_initialized = 0;
		tcnn::GPUMemoryArena::Allocation m_scratch_alloc;
	};

	class FiniteDifferenceNormalsApproximator {
	public:
		void enlarge(uint32_t n_elements, cudaStream_t stream);
		void normal(uint32_t n_elements, const distance_fun_t& distance_function, const Eigen::Vector3f* pos, Eigen::Vector3f* normal, float epsilon, cudaStream_t stream);

	private:
		Eigen::Vector3f* dx;
		Eigen::Vector3f* dy;
		Eigen::Vector3f* dz;

		float* dist_dx_pos;
		float* dist_dy_pos;
		float* dist_dz_pos;

		float* dist_dx_neg;
		float* dist_dy_neg;
		float* dist_dz_neg;

		tcnn::GPUMemoryArena::Allocation m_scratch_alloc;
	};

	struct LevelStats {
		float mean() { return count ? (x / (float)count) : 0.f; }
		float variance() { return count ? (xsquared - (x * x) / (float)count) / (float)count : 0.f; }
		float sigma() { return sqrtf(variance()); }
		float fraczero() { return (float)numzero / float(count + numzero); }
		float fracquant() { return (float)numquant / float(count); }

		float x;
		float xsquared;
		float min;
		float max;
		int numzero;
		int numquant;
		int count;
	};

	static constexpr float LOSS_SCALE = 128.f;

	struct NetworkDims {
		uint32_t n_input;
		uint32_t n_output;
		uint32_t n_pos;
	};

	NetworkDims network_dims_volume() const;
	NetworkDims network_dims_sdf() const;
	NetworkDims network_dims_image() const;
	NetworkDims network_dims_nerf() const;

	NetworkDims network_dims() const;

	void render_volume(CudaRenderBuffer& render_buffer,
		const Eigen::Vector2f& focal_length,
		const Eigen::Matrix<float, 3, 4>& camera_matrix,
		const Eigen::Vector2f& screen_center,
		cudaStream_t stream
	);
	void train_volume(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	void training_prep_volume(uint32_t batch_size, cudaStream_t stream) {}
	void load_volume();

	void render_sdf(
		const distance_fun_t& distance_function,
		const normals_fun_t& normals_function,
		CudaRenderBuffer& render_buffer,
		const Eigen::Vector2i& max_res,
		const Eigen::Vector2f& focal_length,
		const Eigen::Matrix<float, 3, 4>& camera_matrix,
		const Eigen::Vector2f& screen_center,
		cudaStream_t stream
	);
	const float* get_inference_extra_dims(cudaStream_t stream) const;
	void render_nerf(CudaRenderBuffer& render_buffer, const Eigen::Vector2i& max_res, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix0, const Eigen::Matrix<float, 3, 4>& camera_matrix1, const Eigen::Vector4f& rolling_shutter, const Eigen::Vector2f& screen_center, cudaStream_t stream);
	void render_image(CudaRenderBuffer& render_buffer, cudaStream_t stream);
	void render_frame(const Eigen::Matrix<float, 3, 4>& camera_matrix0, const Eigen::Matrix<float, 3, 4>& camera_matrix1, const Eigen::Vector4f& nerf_rolling_shutter, CudaRenderBuffer& render_buffer, bool to_srgb = true) ;
	void visualize_nerf_cameras(ImDrawList* list, const Eigen::Matrix<float, 4, 4>& world2proj);
	filesystem::path find_network_config(const filesystem::path& network_config_path);
	nlohmann::json load_network_config(const filesystem::path& network_config_path);
	void reload_network_from_file(const std::string& network_config_path = "");
	void reload_network_from_json(const nlohmann::json& json, const std::string& config_base_path=""); // config_base_path is needed so that if the passed in json uses the 'parent' feature, we know where to look... be sure to use a filename, or if a directory, end with a trailing slash
	void reset_accumulation(bool due_to_camera_movement = false, bool immediate_redraw = true);
	void redraw_next_frame() {
		m_render_skip_due_to_lack_of_camera_movement_counter = 0;
	}
	bool reprojection_available() { return m_dlss; }
	static ELossType string_to_loss_type(const std::string& str);
	void reset_network(bool clear_density_grid = true);
	void create_empty_nerf_dataset(size_t n_images, int aabb_scale = 1, bool is_hdr = false);
	void load_nerf();
	void load_nerf_post();
	void load_mesh();
	void set_exposure(float exposure) { m_exposure = exposure; }
	void set_max_level(float maxlevel);
	void set_min_level(float minlevel);
	void set_visualized_dim(int dim);
	void set_visualized_layer(int layer);
	void translate_camera(const Eigen::Vector3f& rel);
	void mouse_drag(const Eigen::Vector2f& rel, int button);
	void mouse_wheel(Eigen::Vector2f m, float delta);
	void load_file(const std::string& file);
	void set_nerf_camera_matrix(const Eigen::Matrix<float, 3, 4>& cam);
	Eigen::Vector3f look_at() const;
	void set_look_at(const Eigen::Vector3f& pos);
	float scale() const { return m_scale; }
	void set_scale(float scale);
	Eigen::Vector3f view_pos() const { return m_camera.col(3); }
	Eigen::Vector3f view_dir() const { return m_camera.col(2); }
	Eigen::Vector3f view_up() const { return m_camera.col(1); }
	Eigen::Vector3f view_side() const { return m_camera.col(0); }
	void set_view_dir(const Eigen::Vector3f& dir);
	void first_training_view();
	void last_training_view();
	void previous_training_view();
	void next_training_view();
	void set_camera_to_training_view(int trainview);
	void reset_camera();
	bool keyboard_event();
	void generate_training_samples_sdf(Eigen::Vector3f* positions, float* distances, uint32_t n_to_generate, cudaStream_t stream, bool uniform_only);
	void update_density_grid_nerf(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream);
	void update_density_grid_mean_and_bitfield(cudaStream_t stream);

	struct NerfCounters {
		tcnn::GPUMemory<uint32_t> numsteps_counter; // number of steps each ray took
		tcnn::GPUMemory<uint32_t> numsteps_counter_compacted; // number of steps each ray took
		tcnn::GPUMemory<float> loss;

		uint32_t rays_per_batch = 1<<12;
		uint32_t n_rays_total = 0;
		uint32_t measured_batch_size = 0;
		uint32_t measured_batch_size_before_compaction = 0;

		void prepare_for_training_steps(cudaStream_t stream);
		float update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	};

	void train_nerf(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	void train_nerf_step(uint32_t target_batch_size, NerfCounters& counters, cudaStream_t stream);
	void train_sdf(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	void train_image(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	void set_train(bool mtrain);

	template <typename T>
	void dump_parameters_as_images(const T* params, const std::string& filename_base);

	void prepare_next_camera_path_frame();
	void imgui();
	void training_prep_nerf(uint32_t batch_size, cudaStream_t stream);
	void training_prep_sdf(uint32_t batch_size, cudaStream_t stream);
	void training_prep_image(uint32_t batch_size, cudaStream_t stream) {}
	void train(uint32_t batch_size);
	Eigen::Vector2f calc_focal_length(const Eigen::Vector2i& resolution, int fov_axis, float zoom) const ;
	Eigen::Vector2f render_screen_center() const ;
	void optimise_mesh_step(uint32_t N_STEPS);
	void compute_mesh_vertex_colors();
	tcnn::GPUMemory<float> get_density_on_grid(Eigen::Vector3i res3d, const BoundingBox& aabb, const Eigen::Matrix3f& render_aabb_to_local); // network version (nerf or sdf)
	tcnn::GPUMemory<float> get_sdf_gt_on_grid(Eigen::Vector3i res3d, const BoundingBox& aabb, const Eigen::Matrix3f& render_aabb_to_local); // sdf gt version (sdf only)
	tcnn::GPUMemory<Eigen::Array4f> get_rgba_on_grid(Eigen::Vector3i res3d, Eigen::Vector3f ray_dir, bool voxel_centers, float depth, bool density_as_alpha = false);
	int marching_cubes(Eigen::Vector3i res3d, const BoundingBox& render_aabb, const Eigen::Matrix3f& render_aabb_to_local, float thresh);

	// Determines the 3d focus point by rendering a little 16x16 depth image around
	// the mouse cursor and picking the median depth.
	void determine_autofocus_target_from_pixel(const Eigen::Vector2i& focus_pixel);
	void autofocus();
	size_t n_params();
	size_t first_encoder_param();
	size_t n_encoding_params();

#ifdef NGP_PYTHON
	pybind11::dict compute_marching_cubes_mesh(Eigen::Vector3i res3d = Eigen::Vector3i::Constant(128), BoundingBox aabb = BoundingBox{Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones()}, float thresh=2.5f);
	pybind11::array_t<float> render_to_cpu(int width, int height, int spp, bool linear, float start_t, float end_t, float fps, float shutter_fraction);
	pybind11::array_t<float> render_with_rolling_shutter_to_cpu(const Eigen::Matrix<float, 3, 4>& camera_transform_start, const Eigen::Matrix<float, 3, 4>& camera_transform_end, const Eigen::Vector4f& rolling_shutter, int width, int height, int spp, bool linear);
	pybind11::array_t<float> screenshot(bool linear) const;
	void override_sdf_training_data(pybind11::array_t<float> points, pybind11::array_t<float> distances);
#endif

	double calculate_iou(uint32_t n_samples=128*1024*1024, float scale_existing_results_factor=0.0, bool blocking=true, bool force_use_octree = true);
	void draw_visualizations(ImDrawList* list, const Eigen::Matrix<float, 3, 4>& camera_matrix);
	void train_and_render(bool skip_rendering);
	filesystem::path training_data_path() const;
	void init_window(int resw, int resh, bool hidden = false, bool second_window = false);
	void destroy_window();
	void apply_camera_smoothing(float elapsed_ms);
	int find_best_training_view(int default_view);
	bool begin_frame_and_handle_user_input();
	void gather_histograms();
	void draw_gui();
	bool frame();
	bool want_repl();
	void load_image();
	void load_exr_image();
	void load_stbi_image();
	void load_binary_image();
	uint32_t n_dimensions_to_visualize() const;
	float fov() const ;
	void set_fov(float val) ;
	Eigen::Vector2f fov_xy() const ;
	void set_fov_xy(const Eigen::Vector2f& val);
	void save_snapshot(const std::string& filepath_string, bool include_optimizer_state);
	void load_snapshot(const std::string& filepath_string);
	CameraKeyframe copy_camera_to_keyframe() const;
	void set_camera_from_keyframe(const CameraKeyframe& k);
	void set_camera_from_time(float t);
	void update_loss_graph();
	void load_camera_path(const std::string& filepath_string);
	bool loop_animation();
	void set_loop_animation(bool value);

	float compute_image_mse(bool quantize_to_byte);

	void compute_and_save_marching_cubes_mesh(const char* filename, Eigen::Vector3i res3d = Eigen::Vector3i::Constant(128), BoundingBox aabb = {}, float thresh = 2.5f, bool unwrap_it = false);
	Eigen::Vector3i compute_and_save_png_slices(const char* filename, int res, BoundingBox aabb = {}, float thresh = 2.5f, float density_range = 4.f, bool flip_y_and_z_axes = false);

	////////////////////////////////////////////////////////////////
	// marching cubes related state
	struct MeshState {
		float thresh = 2.5f;
		int res = 256;
		bool unwrap = false;
		float smooth_amount = 2048.f;
		float density_amount = 128.f;
		float inflate_amount = 1.f;
		bool optimize_mesh = false;
		tcnn::GPUMemory<Eigen::Vector3f> verts;
		tcnn::GPUMemory<Eigen::Vector3f> vert_normals;
		tcnn::GPUMemory<Eigen::Vector3f> vert_colors;
		tcnn::GPUMemory<Eigen::Vector4f> verts_smoothed; // homogenous
		tcnn::GPUMemory<uint32_t> indices;
		tcnn::GPUMemory<Eigen::Vector3f> verts_gradient;
		std::shared_ptr<TrainableBuffer<3, 1, float>> trainable_verts;
		std::shared_ptr<tcnn::Optimizer<float>> verts_optimizer;

		void clear() {
			indices={};
			verts={};
			vert_normals={};
			vert_colors={};
			verts_smoothed={};
			verts_gradient={};
			trainable_verts=nullptr;
			verts_optimizer=nullptr;
		}
	};
	MeshState m_mesh;
	bool m_want_repl = false;

	bool m_render_window = false;
	bool m_gather_histograms = false;

	bool m_include_optimizer_state_in_snapshot = false;
	bool m_render_ground_truth = false;
	EGroundTruthRenderMode m_ground_truth_render_mode = EGroundTruthRenderMode::Shade;
	float m_ground_truth_alpha = 1.0f;

	bool m_train = false;
	bool m_training_data_available = false;
	bool m_render = true;
	int m_max_spp = 0;
	ETestbedMode m_testbed_mode = ETestbedMode::None;
	bool m_max_level_rand_training = false;

	// Rendering stuff
	Eigen::Vector2i m_window_res = Eigen::Vector2i::Constant(0);
	bool m_dynamic_res = true;
	float m_dynamic_res_target_fps = 20.0f;
	int m_fixed_res_factor = 8;
	float m_last_render_res_factor = 1.0f;
	float m_scale = 1.0;
	float m_prev_scale = 1.0;
	float m_aperture_size = 0.0f;
	Eigen::Vector2f m_relative_focal_length = Eigen::Vector2f::Ones();
	uint32_t m_fov_axis = 1;
	float m_zoom = 1.f; // 2d zoom factor (for insets?)
	Eigen::Vector2f m_screen_center = Eigen::Vector2f::Constant(0.5f); // center of 2d zoom

	Eigen::Matrix<float, 3, 4> m_camera = Eigen::Matrix<float, 3, 4>::Zero();
	Eigen::Matrix<float, 3, 4> m_smoothed_camera = Eigen::Matrix<float, 3, 4>::Zero();
	Eigen::Matrix<float, 3, 4> m_prev_camera = Eigen::Matrix<float, 3, 4>::Zero();
	size_t m_render_skip_due_to_lack_of_camera_movement_counter = 0;

	bool m_fps_camera = false;
	bool m_camera_smoothing = false;
	bool m_autofocus = false;
	Eigen::Vector3f m_autofocus_target = Eigen::Vector3f::Constant(0.5f);

	CameraPath m_camera_path = {};

	Eigen::Vector3f m_up_dir = {0.0f, 1.0f, 0.0f};
	Eigen::Vector3f m_sun_dir = Eigen::Vector3f::Ones().normalized();
	float m_bounding_radius = 1;
	float m_exposure = 0.f;

	Eigen::Vector2i m_quilting_dims = Eigen::Vector2i::Ones();

	ERenderMode m_render_mode = ERenderMode::Shade;
	EMeshRenderMode m_mesh_render_mode = EMeshRenderMode::VertexNormals;

	uint32_t m_seed = 1337;
#ifdef NGP_GUI
	GLFWwindow* m_glfw_window = nullptr;
	struct SecondWindow {
		GLFWwindow* window = nullptr;
		GLuint program = 0;
		GLuint vao = 0, vbo = 0;
		void draw(GLuint texture);
	} m_second_window;

	void create_second_window();

	std::function<bool()> m_keyboard_event_callback;

	std::shared_ptr<GLTexture> m_pip_render_texture;
	std::vector<std::shared_ptr<GLTexture>> m_render_textures;
#endif

	ThreadPool m_thread_pool;
	std::vector<std::future<void>> m_render_futures;

	std::vector<CudaRenderBuffer> m_render_surfaces;
	std::unique_ptr<CudaRenderBuffer> m_pip_render_surface;

	SharedQueue<std::unique_ptr<ICallable>> m_task_queue;

	void redraw_gui_next_frame() {
		m_gui_redraw = true;
	}

	bool m_gui_redraw = true;

	struct Nerf {
		struct Training {
			NerfDataset dataset;
			int n_images_for_training = 0; // how many images to train from, as a high watermark compared to the dataset size
			int n_images_for_training_prev = 0; // how many images we saw last time we updated the density grid

			struct ErrorMap {
				tcnn::GPUMemory<float> data;
				tcnn::GPUMemory<float> cdf_x_cond_y;
				tcnn::GPUMemory<float> cdf_y;
				tcnn::GPUMemory<float> cdf_img;
				std::vector<float> pmf_img_cpu;
				Eigen::Vector2i resolution = {16, 16};
				Eigen::Vector2i cdf_resolution = {16, 16};
				bool is_cdf_valid = false;
			} error_map;

			std::vector<TrainingXForm> transforms;
			tcnn::GPUMemory<TrainingXForm> transforms_gpu;

			std::vector<Eigen::Vector3f> cam_pos_gradient;
			tcnn::GPUMemory<Eigen::Vector3f> cam_pos_gradient_gpu;

			std::vector<Eigen::Vector3f> cam_rot_gradient;
			tcnn::GPUMemory<Eigen::Vector3f> cam_rot_gradient_gpu;

			tcnn::GPUMemory<Eigen::Array3f> cam_exposure_gpu;
			std::vector<Eigen::Array3f> cam_exposure_gradient;
			tcnn::GPUMemory<Eigen::Array3f> cam_exposure_gradient_gpu;

			Eigen::Vector2f cam_focal_length_gradient = Eigen::Vector2f::Zero();
			tcnn::GPUMemory<Eigen::Vector2f> cam_focal_length_gradient_gpu;

			std::vector<AdamOptimizer<Eigen::Array3f>> cam_exposure;
			std::vector<AdamOptimizer<Eigen::Vector3f>> cam_pos_offset;
			std::vector<RotationAdamOptimizer> cam_rot_offset;
			AdamOptimizer<Eigen::Vector2f> cam_focal_length_offset = AdamOptimizer<Eigen::Vector2f>(0.f);

			tcnn::GPUMemory<float> extra_dims_gpu; // if the model demands a latent code per training image, we put them in here.
			tcnn::GPUMemory<float> extra_dims_gradient_gpu;
			std::vector<AdamOptimizer<Eigen::ArrayXf>> extra_dims_opt;

			void reset_extra_dims(default_rng_t &rng);

			float extrinsic_l2_reg = 1e-4f;
			float extrinsic_learning_rate = 1e-3f;

			float intrinsic_l2_reg = 1e-4f;
			float exposure_l2_reg = 0.0f;

			NerfCounters counters_rgb;

			bool random_bg_color = true;
			bool linear_colors = false;
			ELossType loss_type = ELossType::L2;
			ELossType depth_loss_type = ELossType::L1;
			bool snap_to_pixel_centers = true;
			bool train_envmap = false;

			bool optimize_distortion = false;
			bool optimize_extrinsics = false;
			bool optimize_extra_dims = false;
			bool optimize_focal_length = false;
			bool optimize_exposure = false;
			bool render_error_overlay = false;
			float error_overlay_brightness = 0.125f;
			uint32_t n_steps_between_cam_updates = 16;
			uint32_t n_steps_since_cam_update = 0;

			bool sample_focal_plane_proportional_to_error = false;
			bool sample_image_proportional_to_error = false;
			bool include_sharpness_in_error = false;
			uint32_t n_steps_between_error_map_updates = 128;
			uint32_t n_steps_since_error_map_update = 0;
			uint32_t n_rays_since_error_map_update = 0;

			float near_distance = 0.1f;
			float density_grid_decay = 0.95f;
			default_rng_t density_grid_rng;
			int view = 0;

			float depth_supervision_lambda = 0.f;

			tcnn::GPUMemory<float> sharpness_grid;

			void set_camera_intrinsics(int frame_idx, float fx, float fy = 0.0f, float cx = -0.5f, float cy = -0.5f, float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f, float k3 = 0.0f, float k4 = 0.0f, bool is_fisheye = false);
			void set_camera_extrinsics_rolling_shutter(int frame_idx, Eigen::Matrix<float, 3, 4> camera_to_world_start, Eigen::Matrix<float, 3, 4> camera_to_world_end, const Eigen::Vector4f& rolling_shutter, bool convert_to_ngp = true);
			void set_camera_extrinsics(int frame_idx, Eigen::Matrix<float, 3, 4> camera_to_world, bool convert_to_ngp = true);
			Eigen::Matrix<float, 3, 4> get_camera_extrinsics(int frame_idx);
			void update_transforms(int first = 0, int last = -1);

#ifdef NGP_PYTHON
			void set_image(int frame_idx, pybind11::array_t<float> img, pybind11::array_t<float> depth_img, float depth_scale);
#endif

			void reset_camera_extrinsics();
			void export_camera_extrinsics(const std::string& filename, bool export_extrinsics_in_quat_format = true);
		} training = {};

		tcnn::GPUMemory<float> density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
		tcnn::GPUMemory<uint8_t> density_grid_bitfield;
		uint8_t* get_density_grid_bitfield_mip(uint32_t mip);
		tcnn::GPUMemory<float> density_grid_mean;
		uint32_t density_grid_ema_step = 0;

		uint32_t max_cascade = 0;

		ENerfActivation rgb_activation = ENerfActivation::Exponential;
		ENerfActivation density_activation = ENerfActivation::Exponential;

		Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.5f);
		uint32_t extra_dim_idx_for_inference = 0; // which training image's latent code should be presented at inference time

		int show_accel = -1;

		float sharpen = 0.f;

		float cone_angle_constant = 1.f/256.f;

		bool visualize_cameras = false;
		bool render_with_lens_distortion = false;
		Lens render_lens = {};

		float render_min_transmittance = 0.01f;

		float glow_y_cutoff = 0.f;
		int glow_mode = 0;
	} m_nerf;

	struct Sdf {
		float shadow_sharpness = 2048.0f;
		float maximum_distance = 0.00005f;
		float fd_normals_epsilon = 0.0005f;

		ESDFGroundTruthMode groundtruth_mode = ESDFGroundTruthMode::RaytracedMesh;

		BRDFParams brdf;

		// Mesh data
		EMeshSdfMode mesh_sdf_mode = EMeshSdfMode::Raystab;
		float mesh_scale;

		tcnn::GPUMemory<Triangle> triangles_gpu;
		std::vector<Triangle> triangles_cpu;
		std::vector<float> triangle_weights;
		DiscreteDistribution triangle_distribution;
		tcnn::GPUMemory<float> triangle_cdf;
		std::shared_ptr<TriangleBvh> triangle_bvh; // unique_ptr

		bool uses_takikawa_encoding = false;
		bool use_triangle_octree = false;
		int octree_depth_target = 0; // we duplicate this state so that you can waggle the slider without triggering it immediately
		std::shared_ptr<TriangleOctree> triangle_octree;

		tcnn::GPUMemory<float> brick_data;
		uint32_t brick_res = 0;
		uint32_t brick_level = 10;
		uint32_t brick_quantise_bits = 0;
		bool brick_smooth_normals = false; // if true, then we space the central difference taps by one voxel

		bool analytic_normals = false;
		float zero_offset = 0;
		float distance_scale = 0.95f;

		double iou = 0.0;
		float iou_decay = 0.0f;
		bool calculate_iou_online = false;
		tcnn::GPUMemory<uint32_t> iou_counter;
		struct Training {
			size_t idx = 0;
			size_t size = 0;
			size_t max_size = 1 << 24;
			bool did_generate_more_training_data = false;
			bool generate_sdf_data_online = true;
			float surface_offset_scale = 1.0f;
			tcnn::GPUMemory<Eigen::Vector3f> positions;
			tcnn::GPUMemory<Eigen::Vector3f> positions_shuffled;
			tcnn::GPUMemory<float> distances;
			tcnn::GPUMemory<float> distances_shuffled;
			tcnn::GPUMemory<Eigen::Vector3f> perturbations;
		} training = {};
	} m_sdf;

	enum EDataType {
		Float,
		Half,
	};

	struct Image {
		Eigen::Vector2f pos = Eigen::Vector2f::Constant(0.0f);
		Eigen::Vector2f prev_pos = Eigen::Vector2f::Constant(0.0f);
		tcnn::GPUMemory<char> data;

		EDataType type = EDataType::Float;
		Eigen::Vector2i resolution = Eigen::Vector2i::Constant(0.0f);

		tcnn::GPUMemory<Eigen::Vector2f> render_coords;
		tcnn::GPUMemory<Eigen::Array3f> render_out;

		struct Training {
			tcnn::GPUMemory<float> positions_tmp;
			tcnn::GPUMemory<Eigen::Vector2f> positions;
			tcnn::GPUMemory<Eigen::Array3f> targets;

			bool snap_to_pixel_centers = true;
			bool linear_colors = false;
		} training  = {};

		ERandomMode random_mode = ERandomMode::Stratified;
	} m_image;

	struct VolPayload {
		Eigen::Vector3f dir;
		Eigen::Array4f col;
		uint32_t pixidx;
	};

	struct Volume {
		float albedo = 0.95f;
		float scattering = 0.f;
		float inv_distance_scale = 100.f;
		tcnn::GPUMemory<char> nanovdb_grid;
		tcnn::GPUMemory<uint8_t> bitgrid;
		float global_majorant = 1.f;
		Eigen::Vector3f world2index_offset = {0, 0, 0};
		float world2index_scale = 1.f;

		struct Training {
			tcnn::GPUMemory<Eigen::Vector3f> positions = {};
			tcnn::GPUMemory<Eigen::Array4f> targets = {};
		} training = {};

		// tracing state
		tcnn::GPUMemory<Eigen::Vector3f> pos[2] = {};
		tcnn::GPUMemory<VolPayload> payload[2] = {};
		tcnn::GPUMemory<uint32_t> hit_counter = {};
		tcnn::GPUMemory<Eigen::Array4f> radiance_and_density;
	} m_volume;

	float m_camera_velocity = 1.0f;
	EColorSpace m_color_space = EColorSpace::Linear;
	ETonemapCurve m_tonemap_curve = ETonemapCurve::Identity;
	bool m_dlss = false;
	bool m_dlss_supported = false;
	float m_dlss_sharpening = 0.0f;

	// 3D stuff
	float m_render_near_distance = 0.0f;
	float m_slice_plane_z = 0.0f;
	bool m_floor_enable = false;
	inline float get_floor_y() const { return m_floor_enable ? m_aabb.min.y() + 0.001f : -10000.f; }
	BoundingBox m_raw_aabb;
	BoundingBox m_aabb;
	BoundingBox m_render_aabb;
	Eigen::Matrix3f m_render_aabb_to_local;

	Eigen::Matrix<float, 3, 4> crop_box(bool nerf_space) const;
	std::vector<Eigen::Vector3f> crop_box_corners(bool nerf_space) const;
	void set_crop_box(Eigen::Matrix<float, 3, 4> m, bool nerf_space);

	// Rendering/UI bookkeeping
	Ema m_training_prep_ms = {EEmaType::Time, 100};
	Ema m_training_ms = {EEmaType::Time, 100};
	Ema m_render_ms = {EEmaType::Time, 100};
	// The frame contains everything, i.e. training + rendering + GUI and buffer swapping
	Ema m_frame_ms = {EEmaType::Time, 100};
	std::chrono::time_point<std::chrono::steady_clock> m_last_frame_time_point;
	std::chrono::time_point<std::chrono::steady_clock> m_last_gui_draw_time_point;
	std::chrono::time_point<std::chrono::steady_clock> m_training_start_time_point;
	Eigen::Array4f m_background_color = {0.0f, 0.0f, 0.0f, 1.0f};

	bool m_vsync = false;

	// Visualization of neuron activations
	int m_visualized_dimension = -1;
	int m_visualized_layer = 0;
	Eigen::Vector2i m_n_views = {1, 1};
	Eigen::Vector2i m_view_size = {1, 1};
	bool m_single_view = true; // Whether a single neuron is visualized, or all in a tiled grid
	float m_picture_in_picture_res = 0.f; // if non zero, requests a small second picture :)

	bool m_imgui_enabled = true; // tab to toggle
	bool m_visualize_unit_cube = false;
	bool m_snap_to_pixel_centers = false;
	bool m_edit_render_aabb = false;

	Eigen::Vector3f m_parallax_shift = {0.0f, 0.0f, 0.0f}; // to shift the viewer's origin by some amount in camera space

	// CUDA stuff
	tcnn::StreamAndEvent m_stream;

	// Hashgrid encoding analysis
	float m_quant_percent = 0.f;
	std::vector<LevelStats> m_level_stats;
	std::vector<LevelStats> m_first_layer_column_stats;
	int m_num_levels = 0;
	int m_histo_level = 0; // collect a histogram for this level
	uint32_t m_base_grid_resolution;
	float m_per_level_scale;
	float m_histo[257] = {};
	float m_histo_scale = 1.f;

	uint32_t m_training_step = 0;
	uint32_t m_training_batch_size = 1 << 18;
	Ema m_loss_scalar = {EEmaType::Time, 100};
	std::vector<float> m_loss_graph = std::vector<float>(256, 0.0f);
	size_t m_loss_graph_samples = 0;

	bool m_train_encoding = true;
	bool m_train_network = true;

	filesystem::path m_data_path;
	filesystem::path m_network_config_path = "base.json";

	nlohmann::json m_network_config;


	default_rng_t m_rng;

	CudaRenderBuffer m_windowless_render_surface{std::make_shared<CudaSurface2D>()};

	uint32_t network_width(uint32_t layer) const;
	uint32_t network_num_forward_activations() const;

	std::shared_ptr<tcnn::Loss<precision_t>> m_loss;
	// Network & training stuff
	std::shared_ptr<tcnn::Optimizer<precision_t>> m_optimizer;
	std::shared_ptr<tcnn::Encoding<precision_t>> m_encoding;
	std::shared_ptr<tcnn::Network<float, precision_t>> m_network;
	std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> m_trainer;

	struct TrainableEnvmap {
		std::shared_ptr<tcnn::Optimizer<float>> optimizer;
		std::shared_ptr<TrainableBuffer<4, 2, float>> envmap;
		std::shared_ptr<tcnn::Trainer<float, float, float>> trainer;

		Eigen::Vector2i resolution;
		ELossType loss_type;
	} m_envmap;

	struct TrainableDistortionMap {
		std::shared_ptr<tcnn::Optimizer<float>> optimizer;
		std::shared_ptr<TrainableBuffer<2, 2, float>> map;
		std::shared_ptr<tcnn::Trainer<float, float, float>> trainer;
		Eigen::Vector2i resolution;
	} m_distortion;
	std::shared_ptr<NerfNetwork<precision_t>> m_nerf_network;
};

NGP_NAMESPACE_END
