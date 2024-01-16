#include <synerfgine/camera.h>

namespace sng {

__global__ void g_set_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
 const float focal_length, const mat4* cam_mat, vec3* __restrict__ positions, vec3* __restrict__ directions);

Camera::Camera(const vec3& eye, const vec3& at, const vec3& up) : 
    m_cam_to_world(mat4::identity()), m_world_to_cam(mat4::identity()), 
    m_eye(0.0), m_dir(0.0, 0.0, 1.0), m_up(up), m_scale(1.0) {
    // translate_camera(eye);
    set_position(eye);
    set_look_at(at);
    set_scale(m_scale);
    g_camera.allocate_memory(sizeof(mat4));
    g_camera.copy_from_host({m_cam_to_world});
}

Camera::~Camera() {
    g_camera.check_guards();
    g_camera.free_memory();
    g_positions.check_guards();
    g_positions.free_memory();
    g_directions.check_guards();
    g_directions.free_memory();
}

// void Camera::translate_camera(const vec3& rel, const mat3& rot) {
//     vec3 movement = rot * rel;

//     m_eye += movement;
//     m_camera[3] += vec4(movement, 0.0);
// }

void Camera::set_position(const vec3& pos) {
    m_eye = pos;
    m_world_to_cam[3] = vec4(m_eye, 1.0);
    m_cam_to_world = inverse(m_world_to_cam);
}

vec3 Camera::look_at() const {
    return m_eye + m_dir * m_scale;
}

void Camera::set_look_at(const vec3& pos) {
    auto full = m_eye - pos;
    m_world_to_cam[0] = vec4(normalize(cross(m_up, full)), 0.0);
    m_world_to_cam[1] = vec4(normalize(cross(full, m_world_to_cam[0].xyz())), 0.0);
    m_world_to_cam[2] = vec4(normalize(full));
    m_world_to_cam[3] = vec4(m_eye, 1.0);
    m_cam_to_world = inverse(m_world_to_cam);
}

void Camera::set_scale(float scale) {
    auto at = look_at();
    // mat4 translate = mat4::identity();
    // m_scale *= scale;
    // translate[3] = at - m_dir * m_scale;
    // m_camera = translate * m_rotate;
    m_scale = scale;
    m_eye = at - m_dir * m_scale;
    m_world_to_cam[3] = vec4(m_eye, 1.0);
    m_cam_to_world = inverse(m_world_to_cam);
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
    g_camera.resize_and_copy_from_host({m_cam_to_world});
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
    const float focal_length, const mat4* p_camera, vec3* __restrict__ positions, vec3* __restrict__ directions) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    float x = (float)(i % width) / (float)width - 0.5;
    x *= width / height;
    float y = (float)(i / height) / (float)height - 0.5;
    positions[i] = (*p_camera)[3];
    vec4 dir = vec4(vec3(x, y, focal_length), 1.0);
    directions[i] = ((*p_camera) * dir).xyz();
    // if (i == 10234) {
    //     printf("%i-1 %f %f %f\n", i, directions[i].x, directions[i].y, directions[i].z);
    //     printf("%i-2 %f %f %f\n", i, dir.x, dir.y, dir.z);
    //     printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", 
    //         (*p_camera)[0].x, (*p_camera)[0].y, (*p_camera)[0].z, (*p_camera)[0].w,
    //         (*p_camera)[1].x, (*p_camera)[1].y, (*p_camera)[1].z, (*p_camera)[1].w,
    //         (*p_camera)[2].x, (*p_camera)[2].y, (*p_camera)[2].z, (*p_camera)[2].w,
    //         (*p_camera)[3].x, (*p_camera)[3].y, (*p_camera)[3].z, (*p_camera)[3].w
    //     );
    // }
    // directions[i] = normalize(directions[i]);
}

}