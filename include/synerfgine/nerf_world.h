#pragma once

#include <neural-graphics-primitives/testbed.h>
#include <synerfgine/cuda_helpers.h>
#include <synerfgine/camera.h>

namespace sng {
using ngp::Testbed;

class Display;

class NerfWorld {
public:
    NerfWorld();
    void init(Testbed* testbed);
    bool handle(CudaDevice& device, const Camera& cam, const ivec2& view_res);
    std::shared_ptr<CudaRenderBuffer> render_buffer() { return m_render_buffer; }

private:
    friend class sng::Display;
    mat4x3 m_last_camera;

    Testbed *m_testbed;
    std::shared_ptr<GLTexture> m_rgba_render_textures;
    std::shared_ptr<GLTexture> m_depth_render_textures;
    std::shared_ptr<CudaRenderBuffer> m_render_buffer;
    CudaRenderBufferView m_render_buffer_view;
    ivec2 m_resolution;
};

}