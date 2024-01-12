#pragma once

#include <synerfgine/camera.h>
#include <synerfgine/cuda_helpers.h>
#include <synerfgine/display.h>
#include <synerfgine/input.h>
#include <synerfgine/nerf_world.h>
#include <synerfgine/syn_world.h>

#include <neural-graphics-primitives/render_buffer.h>

#include <vector>

namespace sng {

class Engine {
public:
    // Engine();
    void init(int res_width, int res_height);
    bool frame() { return true; }

    Display display;
    std::vector<CudaDevice> m_devices;
};

}