#pragma once
#include <fstream>
#include <filesystem>
#include <string>
#include <memory>
#include <vector>

#include <tiny-cuda-nn/gpu_memory.h>

// #define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <neural-graphics-primitives/triangle.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/vec.h>

namespace sng {

using namespace tcnn;
namespace fs = std::filesystem;
using ngp::Triangle;

constexpr float MIN_DIST = -25.0;
constexpr float MAX_DIST = 25.0;

// struct RTView;

static char virtual_object_fp[1024] = "../data/obj/smallbox.obj";

class VirtualObject {
public:
    VirtualObject(const char* fp, const std::string& name);
    ~VirtualObject();
    void update_triangles(cudaStream_t stream);
    mat4 get_transform();
    Triangle* gpu_triangles();
    const std::vector<Triangle>& cpu_triangles();
    void imgui();

private:
    bool needs_update{true};
    std::string name;
    fs::path file_path;
    vec3 pos;
    vec3 rot;
    vec3 center;
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
    
    std::vector<Triangle> triangles_cpu;
	GPUMemory<Triangle> orig_triangles_gpu;
	GPUMemory<Triangle> triangles_gpu;
};

VirtualObject load_virtual_obj(const char* fp, const std::string& name);
// void reset_final_views(size_t n_views, std::vector<RTView>& rt_views, ivec2 resolution);

}