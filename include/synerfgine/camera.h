#pragma once

#include <synerfgine/light.cuh>
#include <synerfgine/cuda_helpers.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <optional>

namespace sng {

using namespace ngp;
using namespace tcnn;

namespace camera_default {
const vec3 position{0.0f, 0.0f, 0.2f};
const vec3 up_dir{0.0f, 1.0f, 0.0f};
const vec2 screen_center{0.5f};
// const vec2 relative_focal_length{1.0f};
const float scale = 1.0f;
const uint32_t fov_axis = 1;
const float zoom = 1.0f;
const float bounding_radius = 1.0f;
const float exposure = 0.0f;
}

struct HitInfo;

class Camera {
public:
    Camera() {}
    ~Camera() {
        m_gpu_positions.check_guards();
        m_gpu_positions.free_memory();
        m_gpu_directions.check_guards();
        m_gpu_directions.free_memory();
    }

    void generate_rays_async(CudaDevice& device);
	void translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down = true);
	mat3 rotation_from_angles(const vec2& angles) const;
    // move out later
	bool handle_user_input() {
		bool is_dragged = handle_mouse_drag();
		bool is_wheeled = handle_mouse_wheel();
		return is_dragged || is_wheeled;
	}
	vec3 look_at() const;
	void set_look_at(const vec3& pos);
	float scale() const { return m_scale; }
	void set_scale(float scale);
	vec3 view_pos() const { return m_camera[3]; }
	vec3 view_dir() const { return m_camera[2]; }
	vec3 view_up() const { return m_camera[1]; }
	vec3 view_side() const { return m_camera[0]; }
	vec3 sun_pos() const { return m_sun.pos; }
	Light sun() const { return m_sun; }
	void set_view_dir(const vec3& dir);
	float fov() const ;
	void set_fov(float val) ;
	vec2 fov_xy() const ;
	void set_fov_xy(const vec2& val);
    void reset_camera();

    void set_resolution(const ivec2& resolution) { m_resolution = resolution; }
    vec3* gpu_positions() { return m_gpu_positions.data(); }
    vec3* gpu_directions() { return m_gpu_directions.data(); }

	vec2 calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const;
	vec2 render_screen_center(const vec2& screen_center) const;
    // void set_render_buffer(const CudaRenderBuffer& render_buffer) { m_render_buffer = {render_buffer};}

private:
	bool handle_mouse_drag();
	bool handle_mouse_wheel();
	float get_depth_from_renderbuffer(const CudaRenderBuffer& render_buffer, const vec2& uv);

	// mat4x3 m_camera = transpose(mat3x4{
	// 	1.0f, 0.0f, 0.0f, 0.5f,
	// 	0.0f, -1.0f, 0.0f, 0.5f,
	// 	0.0f, 0.0f, -1.0f, 0.5f
	// });
	mat4x3 m_camera = transpose(mat3x4{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 2.0f,
		0.0f, 0.0f, 1.0f, 0.0f
	});
    vec3 m_up_dir = camera_default::up_dir;
    float m_scale = camera_default::scale;
	uint32_t m_fov_axis = camera_default::fov_axis;
	float m_zoom = camera_default::zoom;
	vec2 m_screen_center = camera_default::screen_center;
	float m_bounding_radius = camera_default::bounding_radius;
	float m_exposure = camera_default::exposure;

    // GPUMemory<mat4x3> gpu_camera;
	vec2 m_relative_focal_length{1.0f, 1.0f};
	float m_drag_depth = 1.0f;

	const float m_ndc_znear = 1.0f / 32.0f;
	const float m_ndc_zfar = 128.0f;
    const uint32_t m_fps_camera = 12;

    // CudaRenderBuffer* p_render_buffer;

    Light m_sun;

    ivec2 m_resolution;
    GPUMemory<vec3> m_gpu_positions;
    GPUMemory<vec3> m_gpu_directions;
};


}