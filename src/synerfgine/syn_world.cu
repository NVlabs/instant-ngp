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
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, const Triangle* __restrict__ triangles, const Light sun,
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void debug_triangle_vertices(const uint32_t n_elements, const Triangle* __restrict__ triangles,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions);

bool SyntheticWorld::handle(CudaDevice& device, const ivec2& resolution) {
    auto stream = device.stream();
    device.render_buffer_view().clear(stream);
    m_resolution = resolution;

    auto& cam = m_camera;
    cam.set_resolution(m_resolution);
    cam.generate_rays_async(device);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for (auto& vo_kv : m_objects) {
        auto& vo = vo_kv.second;
        // if (is_first) {
        //     const std::string& name = vo_kv.first;
        //     uint32_t tri_count = static_cast<uint32_t>(m_objects.at(name).cpu_triangles().size());
        //     cudaStream_t one_timer;
        //     CUDA_CHECK_THROW(cudaStreamCreate(&one_timer));
        //     linear_kernel(debug_triangle_vertices, 0, one_timer, tri_count, 
        //         m_objects.at(name).gpu_triangles(), cam.gpu_positions(), cam.gpu_directions());
        //     CUDA_CHECK_THROW(cudaStreamSynchronize(one_timer));
        //     is_first = false;
        // }
        draw_object_async(device, vo);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        {
            const std::string& name = vo_kv.first;
            uint32_t tri_count = static_cast<uint32_t>(m_objects.at(name).cpu_triangles().size());
            auto n_elements = m_resolution.x * m_resolution.y;
            // linear_kernel(debug_rt, 0, stream, n_elements,
            //     m_resolution.x, 
            //     m_resolution.y, 
            //     cam.m_world_to_cam[3],
            //     mat4::identity(),
            //     cam.m_world_to_cam,
            //     tri_count,
            //     m_objects.at(name).gpu_triangles(),
            //     device.render_buffer_view().frame_buffer, 
            //     device.render_buffer_view().depth_buffer);
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
    //     CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
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
    auto& cam = m_camera;
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
        cam.sun(),
        device.render_buffer_view().frame_buffer, 
        device.render_buffer_view().depth_buffer);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, const uint32_t tri_count,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, const Triangle* __restrict__ triangles, const Light sun,
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    rgba[i] = vec4(vec3(0.0), 1.0);
    depth[i] = ngp::MAX_RT_DIST;
    vec3 rd = ray_directions[i];
    vec3 ro = ray_origins[i];
    float dt = ngp::MAX_RT_DIST;
    vec3 normal;
    for (size_t k = 0; k < tri_count; ++k) {
        float t = triangles[k].ray_intersect(ro, rd);
        if (t < dt && t > ngp::MIN_RT_DIST) {
            dt = t;
            normal = triangles[k].normal();
        }
        // if (i == n_elements / 2) {
        //     printf("ro: [%f, %f, %f]; rd: [%f, %f, %f]; t: [%f]\n", ro.r, ro.b, ro.g, rd.r, rd.b, rd.g, t);
        //     printf("TRI: [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", 
        //         triangles[k].a.r, triangles[k].a.b, triangles[k].a.g,
        //         triangles[k].b.r, triangles[k].b.b, triangles[k].b.g,
        //         triangles[k].c.r, triangles[k].c.b, triangles[k].c.g);
        // }
    }

    // depth[i] = max(10.0 - dt, 0.0);
    if (dt < ngp::MAX_RT_DIST) {
        ro += rd * dt;
        vec3 to_sun = normalize(sun.pos - ro);
        // FOR DIFFUSE ONLY, NO AMBIENT / SPEC
        float ndotv = dot(normal, to_sun);
        rgba[i] = vec4(ndotv * vec3(1.0, 0.2, 0.0), 1.0);
    } else {
        rgba[i] = vec4(vec3(0.0), 1.0);
    }
}


__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	// if (i == 0) { printf("DIR: %f %f %f\n", ray_directions[i].x, ray_directions[i].y, ray_directions[i].z); }
    rgba[i] = vec4(abs(ray_directions[i]), 1.0);
	// if (i % 100000 == 0) { printf("COL %i: %f %f %f %f\n", i, rgba[i].x, rgba[i].y, rgba[i].z, rgba[i].w); }
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
    // printf("%i: pos [%f %f %f], dir [%f %f %f]\n", i, 
    //     ray_origins[i].r, ray_origins[i].g, ray_origins[i].b, 
    //     ray_directions[i].r, ray_directions[i].g, ray_directions[i].b
    // );
}

}