#include <synerfgine/camera.h>
#include <synerfgine/input.h>

#include <tiny-cuda-nn/common.h>
#include <neural-graphics-primitives/bounding_box.cuh>

namespace sng {

__global__ void init_rays_cam(
	uint32_t n_elements,
	uint32_t sample_index,
	vec3* __restrict__ positions,
	vec3* __restrict__ directions,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	// BoundingBox render_aabb,
	// mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	// Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode
);

void Camera::handle_mouse_wheel() {
	float delta = ImGui::GetIO().MouseWheel;
	if (delta == 0) {
		return;
	}

	float scale_factor = pow(1.1f, -delta);
	set_scale(m_scale * scale_factor);
}

void Camera::handle_mouse_drag() {
	vec2 rel = vec2{ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y} / (float)m_resolution[m_fov_axis];
	vec2 mouse = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};

	vec3 side = m_camera[0];

	// Left held
	if (ImGui::GetIO().MouseDown[0]) {
        float rot_sensitivity = m_fps_camera ? 0.35f : 1.0f;
        mat3 rot = rotation_from_angles(-rel * 2.0f * PI() * rot_sensitivity);

        if (m_fps_camera) {
            rot *= mat3(m_camera);
            m_camera = mat4x3(rot[0], rot[1], rot[2], m_camera[3]);
        } else {
            // Turntable
            auto old_look_at = look_at();
            set_look_at({0.0f, 0.0f, 0.0f});
            m_camera = rot * m_camera;
            set_look_at(old_look_at);
        }
	}

	// Right held
	if (ImGui::GetIO().MouseDown[1]) {
		mat3 rot = rotation_from_angles(-rel * 2.0f * PI());
		// m_slice_plane_z += -rel.y * m_bounding_radius;
	}

	// Middle pressed
	// if (ImGui::GetIO().MouseClicked[2]) {
    //     if (m_render_buffer.has_value()) {
    //         m_drag_depth = get_depth_from_renderbuffer(m_render_buffer.value(), mouse / vec2(m_resolution));
    //     } else {
    //         m_drag_depth = 1.0f;
    //     }
	// }

	// Middle held
	if (ImGui::GetIO().MouseDown[2]) {
		vec3 translation = vec3{-rel.x, -rel.y, 0.0f} / m_zoom;

		// If we have a valid depth value, scale the scene translation by it such that the
		// hovered point in 3D space stays under the cursor.
		if (m_drag_depth < 256.0f) {
			translation *= m_drag_depth / m_relative_focal_length[m_fov_axis];
		}

		translate_camera(translation, mat3(m_camera));
	}
}

void Camera::generate_rays_async(CudaDevice& device) {
	handle_mouse_drag();
	handle_mouse_wheel();
    cudaStream_t stream = device.stream();
    uint32_t n_elements = m_resolution.x * m_resolution.y;
    // {
    //     gpu_camera.check_guards();
    //     gpu_camera.resize_and_copy_from_host({m_camera});
    // }
	auto buf_size = sizeof(vec3) * n_elements;
    if (m_gpu_positions.size() != buf_size) {
        m_gpu_positions.check_guards();
        m_gpu_positions.resize(buf_size);
    }
    if (m_gpu_directions.size() != buf_size) {
        m_gpu_directions.check_guards();
        m_gpu_directions.resize(buf_size);
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    // generate rays

    auto& render_buf = device.render_buffer_view();

    linear_kernel(init_rays_cam, 0, stream, n_elements,
        render_buf.spp,
        m_gpu_positions.data(),
        m_gpu_directions.data(),
        m_resolution,
        m_relative_focal_length,
		m_camera,
		m_camera,
		vec4(0.0f), // rolling_shutter
        m_screen_center,
		vec3(0.0),
		true,
		// render_aabb,
		// render_aabb_to_local,
		m_ndc_znear,
		1.0f, // plane_z
		1.0f, // aperture_size
		ngp::Foveation{},
		ngp::Lens{}, // default perspective lens
		// envmap,
		render_buf.frame_buffer,
		render_buf.depth_buffer,
		Buffer2DView<const uint8_t>{}, // hidden_area_mask
		Buffer2DView<const vec2>{}, // distortion
		ERenderMode::Shade
    );

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void Camera::translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down) {
	vec3 movement = rot * rel;
	if (!allow_up_down) {
		movement -= dot(movement, m_up_dir) * m_up_dir;
	}

	m_camera[3] += movement;
}

mat3 Camera::rotation_from_angles(const vec2& angles) const {
	vec3 up = m_up_dir;
	vec3 side = m_camera[0];
	return rotmat(angles.x, up) * rotmat(angles.y, side);
}

vec3 Camera::look_at() const {
	return view_pos() + view_dir() * m_scale;
}

void Camera::set_look_at(const vec3& pos) {
	m_camera[3] += pos - look_at();
}

void Camera::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera[3] = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void Camera::set_view_dir(const vec3& dir) {
	auto old_look_at = look_at();
	m_camera[0] = normalize(cross(dir, m_up_dir));
	m_camera[1] = normalize(cross(dir, m_camera[0]));
	m_camera[2] = normalize(dir);
	set_look_at(old_look_at);
}

void Camera::reset_camera() {
	m_fov_axis = camera_default::fov_axis;
	m_zoom = camera_default::zoom;
	m_screen_center = camera_default::screen_center;

    set_fov(50.625f);
    m_scale = camera_default::scale;

	m_camera = transpose(mat3x4{
		1.0f, 0.0f, 0.0f, 0.5f,
		0.0f, -1.0f, 0.0f, 0.5f,
		0.0f, 0.0f, -1.0f, 0.5f
	});

	m_camera[3] -= m_scale * view_dir();
}

float Camera::fov() const {
	return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]);
}

void Camera::set_fov(float val) {
	m_relative_focal_length = vec2(fov_to_focal_length(1, val));
}

vec2 Camera::fov_xy() const {
	return focal_length_to_fov(ivec2(1), m_relative_focal_length);
}

void Camera::set_fov_xy(const vec2& val) {
	m_relative_focal_length = fov_to_focal_length(ivec2(1), val);
}

__global__ void init_rays_cam(
	uint32_t n_elements,
	uint32_t sample_index,
	vec3* __restrict__ positions,
	vec3* __restrict__ directions,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	// BoundingBox render_aabb,
	// mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	// Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;
	frame_buffer[idx] = vec4(0.3, 0.0, 0.0, 1.0);


	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	mat4x3 camera = get_xform_given_rolling_shutter({camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

	if (idx == 0) { printf("UV: %f %ff\n", uv.x, uv.y); }

	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);

	if (idx == 0) { printf("DIR: %f %f %f\n", ray.d.x, ray.d.y, ray.d.z); }

	// NerfPayload& payload = payloads[idx];
	// payload.max_weight = 0.0f;

	depth_buffer[idx] = MAX_DEPTH();

	if (!ray.is_valid()) {
        positions[idx] = ray.o;
        directions[idx] = vec3(0.0);
		// payload.origin = ray.o;
		// payload.alive = false;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
        positions[idx] = ray.o;
        directions[idx] = (1.0f/n) * ray.d;
		// payload.origin = ray.o;
		// payload.dir = (1.0f/n) * ray.d;
		// payload.t = -plane_z*n;
		// payload.idx = idx;
		// payload.n_steps = 0;
		// payload.alive = false;
		depth_buffer[idx] = -plane_z;
		return;
	}

	if (render_mode == ERenderMode::Distortion) {
		vec2 uv_after_distortion = pos_to_uv(ray(1.0f), resolution, focal_length, camera, screen_center, parallax_shift, foveation);

		frame_buffer[idx].rgb() = to_rgb((uv_after_distortion - uv) * 64.0f);
		frame_buffer[idx].a = 1.0f;
		depth_buffer[idx] = 1.0f;
        positions[idx] = ray(MAX_DEPTH());
		// payload.origin = ray(MAX_DEPTH());
		// payload.alive = false;
		return;
	}

	ray.d = normalize(ray.d);

	// if (envmap) {
	// 	frame_buffer[idx] = read_envmap(envmap, ray.d);
	// }

	// float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

	// if (!render_aabb.contains(render_aabb_to_local * ray(t))) {
    //     positions[idx] = ray.o;
	// 	// payload.origin = ray.o;
	// 	// payload.alive = false;
	// 	return;
	// }

    positions[idx] = ray.o;
    directions[idx] = ray.d;
	frame_buffer[idx] = vec4(abs(ray.d), 1.0);
	// payload.origin = ray.o;
	// payload.dir = ray.d;
	// payload.t = t;
	// payload.idx = idx;
	// payload.n_steps = 0;
	// payload.alive = true;
}

}