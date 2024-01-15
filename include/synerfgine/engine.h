#pragma once

#include <synerfgine/cuda_helpers.h>
#include <synerfgine/display.h>
#include <synerfgine/input.h>
#include <synerfgine/nerf_world.h>
#include <synerfgine/syn_world.h>

#include <neural-graphics-primitives/render_buffer.h>

#include <tiny-cuda-nn/multi_stream.h>

#include <vector>

namespace sng {

class Engine {
public:
    Engine();
    void init(int res_width, int res_height);
    bool frame();
    void redraw_next_frame() { is_dirty = true; }
    ~Engine();

private:
    bool is_dirty = true;
    SyntheticWorld m_syn_world;
    Display m_display;
    std::vector<CudaDevice> m_devices;
    StreamAndEvent m_stream;
};

}