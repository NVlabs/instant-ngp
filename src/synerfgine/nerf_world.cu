#include <synerfgine/nerf_world.h>

#include <tiny-cuda-nn/common.h>

namespace sng {

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::Testbed;
using ngp::GLTexture;

bool NerfWorld::handle(Testbed* testbed) {
    testbed->train_and_render(true);
    mat4x3 cam0 = testbed->m_camera;
    mat4x3 cam1 = testbed->m_camera;
    auto& device = testbed->primary_device();
    // testbed->render_nerf(device.stream(), device, )
}

}