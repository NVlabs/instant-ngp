#include <synerfgine/cuda_helpers.h>
#include <synerfgine/syn_world.h>

#include <tiny-cuda-nn/common.h>
#include <filesystem>

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

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::GLTexture;

static bool is_first = true;

__global__ void debug_syn_depth(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth);

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

SyntheticWorld::SyntheticWorld() {
	m_rgba_render_textures = std::make_shared<GLTexture>();
	m_depth_render_textures = std::make_shared<GLTexture>();

	m_render_buffer = std::make_shared<CudaRenderBuffer>(m_rgba_render_textures, m_depth_render_textures);
    m_render_buffer_view = m_render_buffer->view();
	m_render_buffer->disable_dlss();
}

bool SyntheticWorld::handle(CudaDevice& device, const ivec2& resolution) {
    auto stream = device.stream();
    device.render_buffer_view().clear(stream);
    if (resolution != m_resolution) {
        m_resolution = resolution;
        m_rgba_render_textures->resize(resolution, 4);
        m_depth_render_textures->resize(resolution, 1);
        m_render_buffer->resize(resolution);
        m_render_buffer_view = m_render_buffer->view();
    }

    auto& cam = m_camera;
    auto cam_matrix = cam.get_matrix();
    cam.set_resolution(m_resolution);
    
    auto device_guard = use_device(stream, *m_render_buffer, device);
    cam.generate_rays_async(device);
    bool changed_depth = cam_matrix == m_last_camera;
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for (auto& vo_kv : m_objects) {
        auto& vo = vo_kv.second;
        changed_depth = changed_depth && vo.update_triangles(stream);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        draw_object_async(device, vo);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
    if (changed_depth && !m_objects.empty()) {
        // const std::string& name = vo_kv.first;
        // uint32_t tri_count = static_cast<uint32_t>(m_objects.at(name).cpu_triangles().size());
        auto n_elements = m_resolution.x * m_resolution.y;
        linear_kernel(debug_syn_depth, 0, stream, n_elements, m_render_buffer_view.frame_buffer, m_render_buffer_view.depth_buffer);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
    m_last_camera = cam_matrix;
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
        // device.render_buffer_view().frame_buffer, 
        // device.render_buffer_view().depth_buffer
        m_render_buffer_view.frame_buffer, 
        m_render_buffer_view.depth_buffer
        );
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
    depth[i] = dt;
    // if (dt != ngp::MAX_RT_DIST && i % 100000 == 0) {
    //     printf("SYN: %d: %f\n", i, dt);
    // }
}

void SyntheticWorld::imgui(float frame_time) {
	static std::string imgui_error_string = "";

	if (ImGui::Begin("Load Virtual Object")) {
		ImGui::Text("Add Virtual Object (.obj only)");
		ImGui::InputText("##PathFile", sng::virtual_object_fp, 1024);
		ImGui::SameLine();
		static std::string vo_path_load_error_string = "";
		if (ImGui::Button("Load")) {
			try {
				create_object(sng::virtual_object_fp);
			} catch (const std::exception& e) {
				ImGui::OpenPopup("Virtual object path load error");
				vo_path_load_error_string = std::string{"Failed to load object path: "} + e.what();
			}
		}
		if (ImGui::BeginPopupModal("Virtual object path load error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			ImGui::Text("%s", vo_path_load_error_string.c_str());
			if (ImGui::Button("OK", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
		if (ImGui::CollapsingHeader("Virtual Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
			auto& objs = objects();
			std::string to_remove;
			size_t k = 0;
			for (auto& vo : objs) {
				std::string delete_button_name = std::to_string(k) + ". Delete ";
				delete_button_name.append(vo.first);
				if (ImGui::Button(delete_button_name.c_str())) {
					to_remove = vo.first;
					break;
				}
				vo.second.imgui();
				++k;
			}
			if (!to_remove.empty()) {
				delete_object(to_remove);
			}
		}
	}
	ImGui::End();
	if (ImGui::Begin("Camera")) {
		auto rd = camera().view_pos();
		ImGui::Text("View Pos: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = camera().view_dir();
		ImGui::Text("View Dir: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = camera().look_at();
		ImGui::Text("Look At: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = camera().sun_pos();
		ImGui::Text("Sun Pos: %f, %f, %f", rd.r, rd.g, rd.b);
		float fps = !frame_time ? std::numeric_limits<float>::max() : (1000.0f / frame_time);
		ImGui::Text("Frame: %.2f ms (%.1f FPS)", frame_time, fps);
		if (ImGui::Button("Reset Camera")) {
			mut_camera().reset_camera();
		}
	}
	ImGui::End();
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

__global__ void debug_syn_depth(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    if (i == n_elements * 4 / 7) {
        printf("SYN DEPTH: %.5f\n", depth[i]);
    }
}

}