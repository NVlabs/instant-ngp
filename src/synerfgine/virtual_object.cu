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
__global__ void transform_triangles(uint32_t n_elements, const Triangle* __restrict__ orig, 
	Triangle* __restrict__ tris, const mat4 world_matrix);

VirtualObject sng::load_virtual_obj(const char* fp, const std::string& name) {
    return VirtualObject(fp, name);
}

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
	
	vec3 center{0.0};
	uint32_t tri_count = 0;
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
			center += triangle.a + triangle.b + triangle.c;
			tri_count += 3;
			// std::cout << triangle << std::endl
            triangles_cpu.push_back(triangle);
        }
    }
	if (tri_count) center = center / (float)tri_count;
	// bvh = TriangleBvh::make();
	orig_triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	// cam_triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	// bvh->build(triangles_cpu, 3);
	// TODO: build a bvh implementation that can be updated
}

const std::vector<Triangle>& VirtualObject::cpu_triangles() {
	return triangles_cpu;
}

Triangle* VirtualObject::gpu_triangles() {
	return triangles_gpu.data();
}

bool VirtualObject::update_triangles(cudaStream_t stream) {
	if (needs_update) {
		mat4 world_mat = get_transform();
		linear_kernel(transform_triangles, 0, stream, triangles_cpu.size(), 
			orig_triangles_gpu.data(), triangles_gpu.data(), world_mat);
		needs_update = false;
		return true;
	}
	return false;
}

mat4 VirtualObject::get_transform() {
	vec4 last = vec4(vec3(0.0), 1.0);
	mat4 rx = mat4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, cos(rot.x), sin(rot.x), 0.0), vec4(0.0, -sin(rot.x), cos(rot.x), 0.0), last);
	mat4 ry = mat4(vec4(cos(rot.y), 0.0, -sin(rot.y), 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(sin(rot.y), 0, cos(rot.y), 0.0), last);
	mat4 rz = mat4(vec4(cos(rot.z), sin(rot.z), 0.0, 0.0), vec4(-sin(rot.z), cos(rot.z), 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), last);
	mat4 sc = mat4::identity() * scale;
	sc[3].a = 1.0f;
	mat4 rmat = sc * rz * ry * rx;
	return mat4(rmat[0], rmat[1], rmat[2], vec4(pos, 1.0));
}

VirtualObject::~VirtualObject() {
	triangles_gpu.free_memory();
}

void VirtualObject::imgui() {
    std::string scl_name = "Scale " + name;
    std::string pos_name = "Position " + name;
    std::string rot_name = "Rotation " + name;
    ImGui::Text(name.c_str());
	// update the bvh when the values change
    if (ImGui::SliderFloat(scl_name.c_str(), &scale, 0.001f, 5.0f)) {
		needs_update = true;
	}
    if (ImGui::SliderFloat3(pos_name.c_str(), pos.data(), sng::MIN_DIST, sng::MAX_DIST)) {
		needs_update = true;
	}
    if (ImGui::SliderFloat3(rot_name.c_str(), rot.data(), sng::MIN_DIST, sng::MAX_DIST)) {
		needs_update = true;
	}
    ImGui::Separator();
}

__global__ void transform_triangles(uint32_t n_elements, const Triangle* __restrict__ orig, 
	Triangle* __restrict__ tris, const mat4 world_matrix) {
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n_elements) {
        return;
    }
	tris[i].a = world_matrix * vec4(orig[i].a, 1.0);
	tris[i].b = world_matrix * vec4(orig[i].b, 1.0);
	tris[i].c = world_matrix * vec4(orig[i].c, 1.0);
}

}