#include <synerfgine/virtual_object.h>

#include <tinylogger/tinylogger.h>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#ifdef NGP_GUI
#	include <imgui/backends/imgui_impl_glfw.h>
#	include <imgui/backends/imgui_impl_opengl3.h>
#	include <imgui/imgui.h>
#	include <imguizmo/ImGuizmo.h>
#endif

namespace sng {
__global__ void debug_set_screen(uint32_t n_elements, vec4* __restrict__ render_buffer, uint32_t width, uint32_t height);
__global__ void transform_triangles(uint32_t n_elements, const Triangle* __restrict__ orig, 
	Triangle* __restrict tris, const mat4x3* camera_matrix, const mat4* world_matrix);

VirtualObject sng::load_virtual_obj(const char* fp, const std::string& name) {
    return VirtualObject(fp, name);
}

// void sng::reset_final_views(size_t n_views, std::vector<RTView>& rt_views, ivec2 resolution) {
//     while (rt_views.size() > n_views) {
//         rt_views.pop_back();
//     }
// 	rt_views.resize(n_views, RTView(resolution));
// }

// void RTView::debug_paint(cudaStream_t stream) {
// 	if (render_buffer->frame_buffer() != nullptr) {
// 		uint32_t width_res = static_cast<uint32_t>(render_buffer->in_resolution().r);
// 		uint32_t height_res = static_cast<uint32_t>(render_buffer->in_resolution().g);
// 		linear_kernel(debug_set_screen, 0, stream, width_res * height_res, 
// 			render_buffer->frame_buffer(), width_res, height_res);
// 	}
// }

VirtualObject::VirtualObject(const char* fp, const std::string& name) 
    : file_path(fp), name(name), pos(0.0), rot(0.0) {
	std::string warn;
	std::string err;

	std::ifstream f{file_path.string(), std::ios::in | std::ios::binary};
	if (f.fail()) {
		err = "File not found";
	}
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, &f);

	if (!warn.empty()) {
		tlog::warning() << warn << " while loading '" << file_path.c_str() << "'";
	}

	if (!err.empty()) {
		std::string error = "Error loading: {";
		error.append(file_path.string());
		error.append("} : {");
		error.append(err);
		error.append("}");
		throw std::runtime_error{error};
	}

	std::vector<vec3> result;

	tlog::success() << "Loaded mesh \"" << file_path.c_str() << "\" file with " << shapes.size() << " shapes.";
	
    for(auto& shape : shapes) {
		auto& idxs = shape.mesh.indices;
		auto& verts = attrib.vertices;
		auto get_vec = [verts=verts, idxs=idxs](size_t i) {
			return vec3(
				verts[idxs[i].vertex_index * 3], 
				verts[idxs[i].vertex_index * 3 + 1], 
				verts[idxs[i].vertex_index * 3 + 2]
			);
		};
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            Triangle triangle {
                get_vec(i), get_vec(i+1), get_vec(i+2)
            };
			// std::cout << triangle << std::endl
            triangles_cpu.push_back(triangle);
        }
    }
	// bvh = TriangleBvh::make();
	triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	cam_triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	// bvh->build(triangles_cpu, 3);
	// TODO: build a bvh implementation that can be updated
}

mat4 VirtualObject::get_transform() {
	vec4 last = vec4(vec3(0.0), 1.0);
	mat4 rx = mat4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, cos(rot.x), sin(rot.x), 0.0), vec4(0.0, -sin(rot.x), cos(rot.x), 0.0), last);
	mat4 ry = mat4(vec4(cos(rot.y), 0.0, -sin(rot.y), 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(sin(rot.x), 0, cos(rot.x), 0.0), last);
	mat4 rz = mat4(vec4(cos(rot.y), sin(rot.y), 0.0, 0.0), vec4(-sin(rot.x), cos(rot.x), 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), last);
	mat4 rmat = rz * ry * rx;
	return mat4(rmat[0], rmat[1], rmat[2], vec4(pos, 1.0));
}

// GPUMemory<Triangle>& VirtualObject::transform_tris(const mat4x3& camera_matrix, cudaStream_t stream) {
//     mat4x3* gpu_camera_matrix;
//     CUDA_CHECK_THROW(cudaMallocAsync(&gpu_camera_matrix, sizeof(mat4x3), stream));
//     CUDA_CHECK_THROW(cudaMemcpyAsync(gpu_camera_matrix, &camera_matrix, sizeof(mat4x3), cudaMemcpyHostToDevice, stream));
// 	linear_kernel(transform_triangles, 0, stream, triangles_cpu.size(), triangles_gpu.data(), cam_triangles_gpu.data(), gpu_camera_matrix);
// 	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
// 	return cam_triangles_gpu;
// }

VirtualObject::~VirtualObject() {
	triangles_gpu.free_memory();
}

void VirtualObject::imgui() {
    std::string pos_name = "Position: " + name;
    std::string rot_name = "Rotation: " + name;
    ImGui::Text(name.c_str());
	// update the bvh when the values change
    if (ImGui::SliderFloat3(pos_name.c_str(), pos.data(), sng::MIN_DIST, sng::MAX_DIST)) {}
    if (ImGui::SliderFloat3(rot_name.c_str(), rot.data(), sng::MIN_DIST, sng::MAX_DIST)) {}
    ImGui::Separator();
}

__global__ void debug_set_screen(uint32_t n_elements, vec4* __restrict__ render_buffer, uint32_t width, uint32_t height) {
	uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t x = tidx % width;
	uint32_t y = tidx / width;
	if (x < width && y < height) {
		vec4 data_v = vec4((float)x / (float) width, (float)y / (float)height, 0.0, 1.0);
		render_buffer[tidx] = data_v;
	}
}

__global__ void transform_triangles(uint32_t n_elements, const Triangle* __restrict__ orig, 
	Triangle* __restrict tris, const mat4x3* camera_matrix, const mat4* world_matrix) {
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n_elements) {
        return;
    }
	tris[i].a = *camera_matrix * vec4(orig[i].a, 1.0);
	tris[i].b = *camera_matrix * vec4(orig[i].b, 1.0);
	tris[i].c = *camera_matrix * vec4(orig[i].c, 1.0);
}

}