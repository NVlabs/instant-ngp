#include <synerfgine/cuda_helpers.h>
#include <synerfgine/nerf_world.h>

#include <tiny-cuda-nn/common.h>

#ifdef NGP_GUI
#	include <imgui/backends/imgui_impl_glfw.h>
#	include <imgui/backends/imgui_impl_opengl3.h>
#	include <imgui/imgui.h>
#	include <imguizmo/ImGuizmo.h>
#	ifdef _WIN32
#		include <GL/gl3w.h>
#	else
#		include <GL/glew.h>
#	endif
#	include <GLFW/glfw3.h>
#	include <GLFW/glfw3native.h>
#	include <cuda_gl_interop.h>
#endif

namespace sng {
constexpr bool TRAIN_WITHOUT_RENDER = true;

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::Testbed;
using ngp::GLTexture;

__global__ void clear_nerf(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth);

inline ivec2 downscale_resolution(const ivec2& resolution, float scale) {
    return clamp(ivec2(vec2(resolution) * scale), resolution / 16, resolution);
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

    constexpr float pixel_ratio = 1.0f;
    float factor = std::sqrt(pixel_ratio / m_render_ms * 1000.0f / m_dynamic_res_target_fps);
    factor = 8.f / (float)m_fixed_res_factor;
    factor = clamp(factor, 1.0f / 16.0f, 1.0f);

    auto new_resolution = downscale_resolution(resolution, factor);
    if (new_resolution != m_resolution) {
        m_testbed->m_nerf.training.dataset.scale = 1.0;
        m_resolution = new_resolution;
        m_rgba_render_textures->resize(m_resolution, 4);
        m_depth_render_textures->resize(m_resolution, 1);
        m_render_buffer->resize(m_resolution);
        m_render_buffer->set_hidden_area_mask(nullptr);
        m_render_buffer->disable_dlss();
        m_render_buffer_view = m_render_buffer->view();
    }
    auto stream = device.stream();
    m_render_buffer->reset_accumulation();
    m_render_buffer->clear_frame(stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    auto device_guard = use_device(stream, *m_render_buffer, device);
    auto& testbed_device = m_testbed->m_devices.front();
    testbed_device.set_render_buffer_view(m_render_buffer_view);
	vec2 screen_center = cam.render_screen_center(m_testbed->m_screen_center);
    int visualized_dimension = -1;
    // m_testbed->train_and_render(TRAIN_WITHOUT_RENDER);
    m_testbed->render_frame(stream, cam_matrix, cam_matrix, cam_matrix, screen_center, m_testbed->m_relative_focal_length,
        vec4(vec3(0.0f), 1.0f), {}, {}, visualized_dimension, *m_render_buffer);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    m_last_camera = cam_matrix;
    return true;
}

void NerfWorld::imgui(float frame_time) {
    m_render_ms = frame_time;
	if (ImGui::Begin("Nerf Settings")) {
        ImGui::Text("FPS: %.3f", 1000.0 / frame_time);
        ImGui::SliderFloat("Target FPS: ", &m_dynamic_res_target_fps, 1, 25, "%.3f", 1.0f);
        ImGui::SliderInt("Fixed res factor: ", &m_fixed_res_factor, 8, 64);
    }
    ImGui::End();
}

__global__ void clear_nerf(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    rgba[i] = vec4(0.0, 0.0, 0.0, 1.0);
    depth[i] = 0.0f;
}

}