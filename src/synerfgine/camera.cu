#include <synerfgine/camera.h>

namespace sng {

__global__ void g_set_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
 const float focal_length, const mat4x3* cam_mat, vec3* __restrict__ positions, vec3* __restrict__ directions);

Camera::Camera(const vec3& eye, const vec3& at, const vec3& up) : m_camera(mat3::identity()), 
    m_eye(0.0), m_dir(0.0, 0.0, 1.0), m_up(up), m_scale(1.0) {
    translate_camera(eye, mat3::identity(), true);
    set_look_at(at);
    set_scale(m_scale);
    g_camera.allocate_memory(sizeof(mat4x3));
    g_camera.copy_from_host({m_camera});
}

Camera::~Camera() {
    g_camera.check_guards();
    g_camera.free_memory();
    g_positions.check_guards();
    g_positions.free_memory();
    g_directions.check_guards();
    g_directions.free_memory();
}

void Camera::translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down) {
    vec3 movement = rot * rel;
    if (!allow_up_down) {
        movement -= dot(movement, m_up) * m_up;
    }

    m_eye += movement;
    m_camera[3] += movement;
}

void Camera::set_position(const vec3& pos) {
    m_eye = pos;
    m_camera[3] = pos;
}

vec3 Camera::look_at() const {
    return m_eye + m_dir * m_scale;
}

void Camera::set_look_at(const vec3& pos) {
    auto full = pos - m_eye;
    m_dir = normalize(full);
    m_camera[0] = normalize(cross(m_dir, m_up));
    m_camera[1] = normalize(cross(m_dir, m_camera[0]));
    m_camera[2] = m_dir;
    m_camera[3] = full;
}

void Camera::set_scale(float scale) {
    auto prev_look_at = look_at();
    m_camera[3] = (m_eye - prev_look_at) * (scale / m_scale) + prev_look_at;
    m_scale = scale;
}

void Camera::set_resolution(const ivec2& res) {
    if (res == m_resolution) return;
    size_t n_elements = res.x * res.y;
    g_positions.check_guards();
    g_positions.allocate_memory(n_elements * sizeof(vec3));
    g_directions.check_guards();
    g_directions.allocate_memory(n_elements * sizeof(vec3));
    m_resolution = res;
}

void Camera::generate_rays_async(CudaDevice& device) {
    size_t n_elements = m_resolution.x * m_resolution.y;
    g_camera.check_guards();
    g_camera.resize_and_copy_from_host({m_camera});
    linear_kernel(g_set_rays, 0, device.stream(), n_elements, 
        m_resolution.x, 
        m_resolution.y,
        m_scale,
        g_camera.data(),
        g_positions.data(),
        g_directions.data()
    );
}

__global__ void g_set_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    const float focal_length, const mat4x3* p_camera, vec3* __restrict__ positions, vec3* __restrict__ directions) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    float x = (float)(i % width) / (float)width - 0.5;
    x *= width / height;
    float y = (float)(i / height) / (float)height - 0.5;
    positions[i] = (*p_camera)[3];
    vec4 dir = vec4(vec3(x, y, focal_length), 1.0);
    dir = (*p_camera) * dir;
    directions[i] = normalize(dir.xyz());
    // if (i == 10234) {
    //     printf("%i-1 %f %f %f\n", i, directions[i].x, directions[i].y, directions[i].z);
        // printf("%f %f %f\n%f %f %f\n%f %f %f\n", 
        //     (*p_camera)[0].x, (*p_camera)[0].y, (*p_camera)[0].z,
        //     (*p_camera)[1].x, (*p_camera)[1].y, (*p_camera)[1].z,
        //     (*p_camera)[2].x, (*p_camera)[2].y, (*p_camera)[2].z
        // );
    // }
}

}