#include <synerfgine/syn_world.h>

#include <tiny-cuda-nn/common.h>
#include <filesystem>

namespace sng {

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::GLTexture;

static bool is_first = true;

__global__ void debug_paint(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, const uint32_t tri_count,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, const Triangle* __restrict__ triangles,
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void debug_triangle_vertices(const uint32_t n_elements, const Triangle* __restrict__ triangles,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions);

__global__ void debug_rt(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    const vec3 ro, const mat4 obj_to_world, const mat4 world_to_cam, 
    const uint32_t tri_count, const Triangle* __restrict__ triangles, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

bool SyntheticWorld::handle(CudaDevice& device, const ivec2& resolution) {
    auto stream = device.stream();
    device.render_buffer_view().clear(stream);
    m_resolution = resolution;
    // TEMPORARY
    if (m_cameras.size() != 1) {
        m_cameras.clear();
        create_camera(camera_default::position, camera_default::lookat);
    }
    if (m_lights.size() != 1) {
        m_lights.clear();
        m_lights.push_back({vec3(2.0, 3.0, 3.0), vec3(1.0), 1.0});
    }
    // END TEMPORARY
    auto& cam = m_cameras.front();
    cam.set_resolution(m_resolution);
    cam.generate_rays_async(device);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for (auto& vo_kv : m_objects) {
        auto& vo = vo_kv.second;
        if (is_first) {
            const std::string& name = vo_kv.first;
            uint32_t tri_count = static_cast<uint32_t>(m_objects.at(name).cpu_triangles().size());
            cudaStream_t one_timer;
            CUDA_CHECK_THROW(cudaStreamCreate(&one_timer));
            linear_kernel(debug_triangle_vertices, 0, one_timer, tri_count, 
                m_objects.at(name).gpu_triangles(), cam.gpu_positions(), cam.gpu_directions());
            CUDA_CHECK_THROW(cudaStreamSynchronize(one_timer));
            is_first = false;
        }
        // draw_object_async(device, vo);
        // CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        {
            const std::string& name = vo_kv.first;
            uint32_t tri_count = static_cast<uint32_t>(m_objects.at(name).cpu_triangles().size());
            auto n_elements = m_resolution.x * m_resolution.y;
            linear_kernel(debug_rt, 0, stream, n_elements,
                m_resolution.x, 
                m_resolution.y, 
                cam.m_world_to_cam[3],
                mat4::identity(),
                cam.m_world_to_cam,
                tri_count,
                m_objects.at(name).gpu_triangles(),
                device.render_buffer_view().frame_buffer, 
                device.render_buffer_view().depth_buffer);
        }
    }
    // {
    //     auto n_elements = m_resolution.x * m_resolution.y;
    //     linear_kernel(debug_draw_rays, 0, stream, n_elements,
    //         m_resolution.x, 
    //         m_resolution.y, 
    //         cam.gpu_positions(),
    //         cam.gpu_directions(),
    //         device.render_buffer_view().frame_buffer, 
    //         device.render_buffer_view().depth_buffer);
    // }
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

void SyntheticWorld::draw_object_async(CudaDevice& device, VirtualObject& virtual_object) {
    auto& cam = m_cameras.front();
    auto stream = device.stream();
    auto n_elements = m_resolution.x * m_resolution.y;
    uint32_t tri_count = static_cast<uint32_t>(virtual_object.cpu_triangles().size());
    linear_kernel(gpu_draw_object, 0, stream, n_elements,
        m_resolution.x, 
        m_resolution.y, 
        tri_count,
        cam.gpu_positions(),
        cam.gpu_directions(),
        virtual_object.gpu_triangles(),
        device.render_buffer_view().frame_buffer, 
        device.render_buffer_view().depth_buffer);
}

void SyntheticWorld::create_camera(const vec3& eye, const vec3& at, const vec3& up) {
    m_cameras.emplace_back(eye, at, up);
}

void SyntheticWorld::camera_position(const vec3& eye) {
    if (m_cameras.empty()) return;
    m_cameras[0].set_position(eye);
}

void SyntheticWorld::camera_look_at(const vec3& at) {
    if (m_cameras.empty()) return;
    m_cameras[0].set_look_at(at);
}

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, const uint32_t tri_count,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, const Triangle* __restrict__ triangles,
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    rgba[i] = vec4(vec3(0.0), 1.0);
    depth[i] = ngp::MAX_RT_DIST;
    vec3 rd = ray_directions[i];
    vec3 ro = ray_directions[i];
    float dt = ngp::MAX_RT_DIST;
    for (size_t k = 0; k < tri_count; ++k) {
        float t = triangles[k].ray_intersect(ro, rd);
        if (t < dt && t > ngp::MIN_RT_DIST) {
            dt = t;
            ro += rd * t;
        }
    }
    depth[i] = max(10.0 - dt, 0.0);
    if (dt < ngp::MAX_RT_DIST) {
        rgba[i] = vec4(vec3(depth[i] / 10.), 1.0);
    } else {
        rgba[i] = vec4(vec3(0.0), 1.0);
    }
}

__global__ void debug_rt(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    const vec3 ro, const mat4 obj_to_world, const mat4 world_to_cam, 
    const uint32_t tri_count, const Triangle* __restrict__ triangles, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    float x = (float)(i % width) / (float)width - 0.5;
    float y = (float)(i / height) / (float)height - 0.5;
    vec3 rd = normalize(vec3(x, y, 1.0));
    float dt = ngp::MAX_RT_DIST;
    vec3 new_ro = ro;
    for (size_t k = 0; k < tri_count; ++k) {
        Triangle tri = triangles[k];
        tri.a = (world_to_cam * obj_to_world * vec4(tri.a, 1.0)).xyz();
        tri.b = (world_to_cam * obj_to_world * vec4(tri.b, 1.0)).xyz();
        tri.c = (world_to_cam * obj_to_world * vec4(tri.c, 1.0)).xyz();
        float t = tri.ray_intersect(ro, rd);
        if (t < dt && t > ngp::MIN_RT_DIST) {
            dt = t;
            new_ro = ro + rd * t;
        }
    }
    depth[i] = max(10.0 - dt, 0.0);
    if (dt < ngp::MAX_RT_DIST) {
        rgba[i] = vec4(vec3(depth[i] / 50.), 1.0);
    } else {
        rgba[i] = vec4(vec3(0.0), 1.0);
    }

}

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    vec3 dir = ray_directions[i];
    rgba[i] = vec4(abs(dir.x), abs(dir.y), 0.0, 1.0);
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

__global__ void debug_triangle_vertices(const uint32_t n_elements, const Triangle* __restrict__ triangles, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    const Triangle* tri = &triangles[i];
    printf("%i: [%f %f %f], [%f %f %f], [%f %f %f]\n", i, 
        tri->a.r, tri->a.g, tri->a.b,
        tri->b.r, tri->b.g, tri->b.b,
        tri->c.r, tri->c.g, tri->c.b);
    printf("%i: pos [%f %f %f], dir [%f %f %f]\n", i, 
        ray_origins[i].r, ray_origins[i].g, ray_origins[i].b, 
        ray_directions[i].r, ray_directions[i].g, ray_directions[i].b
    );
}

}