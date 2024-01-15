#pragma once

#include <synerfgine/light.cuh>
#include <synerfgine/cuda_helpers.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

namespace sng {

using namespace tcnn;

namespace camera_default {
const vec3 position{0.0, 0.0, -2.0};
const vec3 lookat{0.0, 0.0, 0.0};
}

class Camera {
public:
    Camera(const vec3& eye, const vec3& at, const vec3& up);

    ~Camera();

    void translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down);
    vec3 look_at() const;
    void set_look_at(const vec3& pos);
    void set_scale(float scale);
    void set_view_dir(const vec3& dir);
    void set_position(const vec3& pos);

    void set_resolution(const ivec2& res);
    void generate_rays_async(CudaDevice& device);

    vec3* gpu_positions() { return g_positions.data(); }

    vec3* gpu_directions() { return g_directions.data(); }

private:
    vec3 m_eye{};
    vec3 m_dir{};
    vec3 m_up{};
    float m_scale{};
    mat4x3 m_camera{};
    ivec2 m_resolution{};

    GPUMemory<mat4x3> g_camera;
    GPUMemory<vec3> g_positions;
    GPUMemory<vec3> g_directions;
};


}