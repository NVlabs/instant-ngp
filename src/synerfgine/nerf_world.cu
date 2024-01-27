#include <synerfgine/cuda_helpers.h>
#include <synerfgine/nerf_world.h>

#include <tiny-cuda-nn/common.h>

namespace sng {
constexpr bool TRAIN_WITHOUT_RENDER = true;
constexpr float SCALE = 0.0707f;

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::Testbed;
using ngp::GLTexture;

__global__ void clear_nerf(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth);

inline ivec2 downscale_resolution(const ivec2& resolution) {
    return clamp(ivec2(vec2(resolution) * SCALE), resolution / 16, resolution);
}

NerfWorld::NerfWorld() {
	m_rgba_render_textures = std::make_shared<GLTexture>();
	m_depth_render_textures = std::make_shared<GLTexture>();

	m_render_buffer = std::make_shared<CudaRenderBuffer>(m_rgba_render_textures, m_depth_render_textures);
    m_render_buffer_view = m_render_buffer->view();
	m_render_buffer->disable_dlss();
}

void NerfWorld::init(Testbed* testbed) {
    m_testbed = testbed;
}

bool NerfWorld::handle(sng::CudaDevice& device, const Camera& cam, const ivec2& resolution) {
    if (!m_testbed) return false;
    auto cam_matrix = cam.get_matrix();
    if (m_last_camera == cam_matrix) return false;
    auto new_resolution = downscale_resolution(resolution);
    if (new_resolution != m_resolution) {
        m_resolution = new_resolution;
        m_rgba_render_textures->resize(m_resolution, 4);
        m_depth_render_textures->resize(m_resolution, 1);
        m_render_buffer->resize(m_resolution);
        m_render_buffer_view = m_render_buffer->view();
    }
    auto stream = device.stream();
    auto device_guard = use_device(stream, *m_render_buffer, device);
    // m_testbed->train_and_render(TRAIN_WITHOUT_RENDER);
    auto& testbed_device = m_testbed->m_devices.front();
    testbed_device.set_render_buffer_view(m_render_buffer_view);
	vec2 screen_center = cam.render_screen_center(camera_default::screen_center);
    int visualized_dimension = -1;
    {
        auto n_elements = product(m_resolution);
        linear_kernel(clear_nerf, 0, stream, n_elements, m_render_buffer_view.frame_buffer, m_render_buffer_view.depth_buffer);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
    // cam_matrix = m_testbed->m_camera;
    m_testbed->render_nerf(stream, testbed_device, testbed_device.render_buffer_view(), testbed_device.nerf_network(), 
        testbed_device.data().density_grid_bitfield_ptr, cam.get_focal_length(m_resolution), cam_matrix, cam_matrix, {}, screen_center, {}, visualized_dimension);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    m_render_buffer->set_color_space(ngp::EColorSpace::SRGB);
    m_render_buffer->set_tonemap_curve(ngp::ETonemapCurve::Identity);
    m_testbed->render_frame_epilogue(stream, cam.get_matrix(), m_last_camera, screen_center, 
        cam.relative_focal_length(), {}, {}, *m_render_buffer, true);
    m_rgba_render_textures->blit_from_cuda_mapping();
    m_depth_render_textures->blit_from_cuda_mapping();
    m_last_camera = cam_matrix;
    // m_last_camera = cam.get_matrix();
    return true;
}

__global__ void clear_nerf(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    rgba[i] = vec4(0.0, 0.0, 0.0, 1.0);
    depth[i] = 0.0f;
}

}