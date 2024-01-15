#pragma once

#include <synerfgine/camera.h>
#include <synerfgine/cuda_helpers.h>
#include <synerfgine/light.cuh>
#include <synerfgine/virtual_object.h>

#include <neural-graphics-primitives/render_buffer.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>

#include <vector>

namespace sng {

using ngp::CudaRenderBuffer;
using ngp::GLTexture;

class SyntheticWorld {
public:
    bool handle(CudaDevice& device, const ivec2& resolution);
    void create_object(const std::string& filename);
    void create_camera(const vec3& eye, const vec3& at, const vec3& up = vec3(0.0, 1.0, 0.0));
    void camera_position(const vec3& eye);
    void camera_look_at(const vec3& at);
    std::unordered_map<std::string, VirtualObject>& objects() { return m_objects; }
    void delete_object(const std::string& name) {
        m_objects.erase(name);
    }
private:
    ivec2 m_resolution;
    void draw_object_async(CudaDevice& device, VirtualObject& vo);
    std::vector<Light> m_lights;
    std::unordered_map<std::string, VirtualObject> m_objects;
    std::vector<Camera> m_cameras;
};

}