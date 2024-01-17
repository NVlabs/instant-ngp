#pragma once 

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>

namespace sng {

using namespace tcnn;

struct Light {
	Light(): pos(3.0f), col(1.0f), intensity(1.0f) {}
    Light(const vec3& pos, const vec3& col, float intensity) : pos(pos), col(col), intensity(intensity) {}
    vec3 pos;
    vec3 col;
    float intensity;
};

}