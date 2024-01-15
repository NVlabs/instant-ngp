#pragma once 

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>

namespace sng {

using namespace tcnn;

struct Ray {
	vec3 o;
	vec3 d;

	NGP_HOST_DEVICE vec3 operator()(float t) const {
		return o + t * d;
	}

	NGP_HOST_DEVICE void advance(float t) {
		o += d * t;
	}

	NGP_HOST_DEVICE float distance_to(const vec3& p) const {
		vec3 nearest = p - o;
		nearest -= d * dot(nearest, d) / length2(d);
		return length(nearest);
	}

	NGP_HOST_DEVICE bool is_valid() const {
		return d != vec3(0.0f);
	}

	static NGP_HOST_DEVICE Ray invalid() {
		return {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
	}
};

struct Light {
    Light(const vec3& pos, const vec3& col, float intensity) : pos(pos), col(col), intensity(intensity) {}
    vec3 pos;
    vec3 col;
    float intensity;
};

}