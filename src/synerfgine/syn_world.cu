#include <synerfgine/syn_world.h>

#include <tiny-cuda-nn/common.h>
#include <filesystem>

namespace sng {

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::GLTexture;

__global__ void debug_paint(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

bool SyntheticWorld::handle(CudaDevice& device, const ivec2& resolution) {
    auto stream = device.stream();
    device.render_buffer_view().clear(stream);
    m_resolution = resolution;
    if (m_cameras.size() != 1) {
        m_cameras.clear();
        create_camera({0.0, 2.0, -2.0}, {0.0});
    }
    auto& cam = m_cameras.front();
    cam.set_resolution(m_resolution);
    cam.generate_rays_async(device);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    auto n_elements = m_resolution.x * m_resolution.y;
    linear_kernel(debug_draw_rays, 0, stream, n_elements,
        m_resolution.x, 
        m_resolution.y, 
        cam.gpu_positions(),
        cam.gpu_directions(),
        device.render_buffer_view().frame_buffer, 
        device.render_buffer_view().depth_buffer);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return true;
}

void SyntheticWorld::create_object(const std::string& filename) {
    size_t k = 0;
    fs::path fp = fs::path(filename.c_str());
    std::string name = fp.filename().string();
    while (m_objects.count(name)) {
        ++k;
        name = fp.filename().string() + " " + std::to_string(k);
    }
    m_objects.insert({name, load_virtual_obj(filename.c_str(), name)});
}

void SyntheticWorld::draw_object(CudaDevice& device, VirtualObject& virtual_object) {
    // if (m_lights.size() != 1) {
    //     m_lights.clear();
    //     m_lights.push_back({vec3(2.0, 3.0, 3.0), vec3(1.0), 1.0});
    // }
    // auto n_elements = m_resolution.x * m_resolution.y;
    // auto stream = device.stream();
    // // auto ddd = device.render_buffer_view().frame_buffer();
    // linear_kernel(gpu_draw_object, 0, stream, n_elements, 
    //     m_resolution.x, 
    //     m_resolution.y, 
    //     cam.gpu_positions(),
    //     cam.gpu_directions(),
    //     device.render_buffer_view().frame_buffer, 
    //     device.render_buffer_view().depth_buffer);
    // CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void SyntheticWorld::create_camera(const vec3& eye, const vec3& at, const vec3& up) {
    m_cameras.emplace_back(eye, at, up);
}

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    vec3 dir = ray_directions[i];
    rgba[i] = vec4(normalize(dir), 1.0);
}

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    vec3 dir = ray_directions[i];
    if (i == 10234) {
        printf("%i-2 %f %f %f\n", i, dir.x, dir.y, dir.z);
    }
    rgba[i] = vec4(abs(dir.x), abs(dir.y), abs(dir.z), 1.0);
}

__global__ void debug_paint(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    float x = (float)(i % width) / (float)width;
    float y = (float)(i / height) / (float)height;
    rgba[i] = vec4(x, y, 0.0, 1.0);
    depth[i] = 0.5f;
}

}