#pragma once

#include <neural-graphics-primitives/testbed.h>
#include <synerfgine/cuda_helpers.h>

namespace sng {
using ngp::Testbed;

class NerfWorld {
public:
    // NerfWorld();
    void init(int res_width, int res_height);
    // bool handle(CudaDevice& device, Testbed* nerf);
    bool handle(CudaDevice& device, Testbed* nerf, const ivec2& view_res);
private:
	std::shared_ptr<CudaRenderBuffer> m_render_buffer;
	ivec2 m_view_res = ivec2(0);
};

}