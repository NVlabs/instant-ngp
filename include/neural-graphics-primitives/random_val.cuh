/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   random_val.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 *  @brief  An application that learns and renders neural graphics primitives in
 *          real time. Such primitives include images, signed distance functions,
 *          and volumetric density/radiance representations (like NeRF).
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/random.h>

#include <Eigen/Dense>

NGP_NAMESPACE_BEGIN

using default_rng_t = tcnn::default_rng_t;

inline constexpr float PI() { return 3.14159265358979323846f; }

template <typename RNG>
inline __host__ __device__ float random_val(RNG& rng) {
	return rng.next_float();
}

template <typename RNG>
inline __host__ __device__ uint32_t random_uint(RNG& rng) {
	return rng.next_uint();
}

template <typename RNG>
inline __host__ __device__ Eigen::Vector2f random_val_2d(RNG& rng) {
	return {rng.next_float(), rng.next_float()};
}

inline __host__ __device__ Eigen::Vector3f cylindrical_to_dir(const Eigen::Vector2f& p) {
	const float cos_theta = -2.0f * p.x() + 1.0f;
	const float phi = 2.0f * PI() * (p.y() - 0.5f);

	const float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
	float sin_phi, cos_phi;
	sincosf(phi, &sin_phi, &cos_phi);

	return {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

inline __host__ __device__ Eigen::Vector2f dir_to_cylindrical(const Eigen::Vector3f& d) {
	const float cos_theta = fminf(fmaxf(-d.z(), -1.0f), 1.0f);
	float phi = std::atan2(d.y(), d.x());
	return {(cos_theta + 1.0f) / 2.0f, (phi / (2.0f * PI())) + 0.5f};
}

inline __host__ __device__ Eigen::Vector2f dir_to_spherical_unorm(const Eigen::Vector3f& d) {
	const float cos_theta = fminf(fmaxf(d.z(), -1.0f), 1.0f);
	const float theta = acosf(cos_theta);
	float phi = std::atan2(d.y(), d.x());
	return {theta / PI(), (phi / (2.0f * PI()) + 0.5f)};
}

template <typename RNG>
inline __host__ __device__ Eigen::Vector3f random_dir(RNG& rng) {
	return cylindrical_to_dir(random_val_2d(rng));
}

inline __host__ __device__ float fractf(float x) {
	return x - floorf(x);
}

template <uint32_t N_DIRS>
__device__ __host__ Eigen::Vector3f fibonacci_dir(uint32_t i, const Eigen::Vector2f& offset) {
	// Fibonacci lattice with offset
	float epsilon;
	if (N_DIRS >= 11000) {
		epsilon = 27;
	} else if (N_DIRS >= 890) {
		epsilon = 10;
	} else if (N_DIRS >= 177) {
		epsilon = 3.33;
	} else if (N_DIRS >= 24) {
		epsilon = 1.33;
	} else {
		epsilon = 0.33;
	}

	static constexpr float GOLDEN_RATIO = 1.6180339887498948482045868343656f;
	return cylindrical_to_dir(Eigen::Vector2f{fractf((i+epsilon) / (N_DIRS-1+2*epsilon) + offset.x()), fractf(i / GOLDEN_RATIO + offset.y())});
}

template <typename RNG>
inline __host__ __device__ Eigen::Vector2f random_uniform_disc(RNG& rng) {
	Eigen::Vector2f sample = random_val_2d(rng);
	float r = sqrtf(sample.x());
	float sin_phi, cos_phi;
	sincosf(2.0f * PI() * sample.y(), &sin_phi, &cos_phi);
	return Eigen::Vector2f{ r * sin_phi, r * cos_phi };
}

inline __host__ __device__ Eigen::Vector2f square2disk_shirley(const Eigen::Vector2f& square) {
	float phi, r;
	float a = square.x();
	float b = square.y();
	if (a*a > b*b) { // use squares instead of absolute values
		r = a;
		phi = (PI()/4.0f) * (b/a);
	} else {
		r = b;
		phi = (PI()/2.0f) - (PI()/4.0f) * (a/b);
	}

	float sin_phi, cos_phi;
	sincosf(phi, &sin_phi, &cos_phi);

	return {r*cos_phi, r*sin_phi};
}

inline __host__ __device__ __device__ Eigen::Vector3f cosine_hemisphere(const Eigen::Vector2f& u) {
	// Uniformly sample disk
	const float r   = sqrtf(u.x());
	const float phi = 2.0f * PI() * u.y();

	Eigen::Vector3f p;
	p.x() = r * cosf(phi);
	p.y() = r * sinf(phi);

	// Project up to hemisphere
	p.z() = sqrtf(fmaxf(0.0f, 1.0f - p.x()*p.x() - p.y()*p.y()));

	return p;
}

template <typename RNG>
inline __host__ __device__ Eigen::Vector3f random_dir_cosine(RNG& rng) {
	return cosine_hemisphere(random_val_2d(rng));
}

template <typename RNG>
inline __host__ __device__ Eigen::Vector3f random_val_3d(RNG& rng) {
	return {rng.next_float(), rng.next_float(), rng.next_float()};
}

template <typename RNG>
inline __host__ __device__ Eigen::Vector4f random_val_4d(RNG& rng) {
	return {rng.next_float(), rng.next_float(), rng.next_float(), rng.next_float()};
}

// The below code has been adapted from Burley [2019] https://www.jcgt.org/published/0009/04/01/paper.pdf

inline __host__ __device__ uint32_t sobol(uint32_t index, uint32_t dim) {
	static constexpr uint32_t directions[5][32] = {
		0x80000000, 0x40000000, 0x20000000, 0x10000000,
		0x08000000, 0x04000000, 0x02000000, 0x01000000,
		0x00800000, 0x00400000, 0x00200000, 0x00100000,
		0x00080000, 0x00040000, 0x00020000, 0x00010000,
		0x00008000, 0x00004000, 0x00002000, 0x00001000,
		0x00000800, 0x00000400, 0x00000200, 0x00000100,
		0x00000080, 0x00000040, 0x00000020, 0x00000010,
		0x00000008, 0x00000004, 0x00000002, 0x00000001,

		0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
		0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
		0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
		0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
		0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
		0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
		0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
		0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,

		0x80000000, 0xc0000000, 0x60000000, 0x90000000,
		0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
		0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
		0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
		0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
		0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
		0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
		0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,

		0x80000000, 0xc0000000, 0x20000000, 0x50000000,
		0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
		0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
		0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
		0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
		0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
		0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
		0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,

		0x80000000, 0x40000000, 0x20000000, 0xb0000000,
		0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
		0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000,
		0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
		0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000,
		0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
		0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0,
		0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d,
	};

	uint32_t X = 0;

	NGP_PRAGMA_UNROLL
	for (uint32_t bit = 0; bit < 32; bit++) {
		uint32_t mask = (index >> bit) & 1;
		X ^= mask * directions[dim][bit];
	}

	return X;
}

inline __host__ __device__ Vector2i32 sobol2d(uint32_t index) {
	return {sobol(index, 0), sobol(index, 1)};
}

inline __host__ __device__ Vector4i32 sobol4d(uint32_t index) {
	return {sobol(index, 0), sobol(index, 1), sobol(index, 2), sobol(index, 3)};
}

inline __host__ __device__ uint32_t hash_combine(uint32_t seed, uint32_t v) {
	return seed ^ (v + (seed << 6) + (seed >> 2));
}

inline __host__ __device__ uint32_t reverse_bits(uint32_t x) {
	x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
	x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
	x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
	x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
	return ((x >> 16) | (x << 16));
}

inline __host__ __device__ uint32_t laine_karras_permutation(uint32_t x, uint32_t seed) {
	x += seed;
	x ^= x * 0x6c50b47cu;
	x ^= x * 0xb82f1e52u;
	x ^= x * 0xc7afe638u;
	x ^= x * 0x8d22f6e6u;
	return x;
}

inline __host__ __device__ uint32_t nested_uniform_scramble_base2(uint32_t x, uint32_t seed) {
	x = reverse_bits(x);
	x = laine_karras_permutation(x, seed);
	x = reverse_bits(x);
	return x;
}

inline __host__ __device__ Vector4i32 shuffled_scrambled_sobol4d(uint32_t index, uint32_t seed) {
	index = nested_uniform_scramble_base2(index, seed);
	auto X = sobol4d(index);
	for (uint32_t i = 0; i < 4; i++) {
		X[i] = nested_uniform_scramble_base2(X[i], hash_combine(seed, i));
	}
	return X;
}

inline __host__ __device__ Vector2i32 shuffled_scrambled_sobol2d(uint32_t index, uint32_t seed) {
	index = nested_uniform_scramble_base2(index, seed);
	auto X = sobol2d(index);
	for (uint32_t i = 0; i < 2; ++i) {
		X[i] = nested_uniform_scramble_base2(X[i], hash_combine(seed, i));
	}
	return X;
}

inline __host__ __device__ Eigen::Vector4f ld_random_val_4d(uint32_t index, uint32_t seed) {
	constexpr float S = float(1.0/(1ull<<32));
	Vector4i32 x = shuffled_scrambled_sobol4d(index, seed);
	return {(float)x.x() * S, (float)x.y() * S, (float)x.z() * S, (float)x.w() * S};
}

inline __host__ __device__ Eigen::Vector2f ld_random_val_2d(uint32_t index, uint32_t seed) {
	constexpr float S = float(1.0/(1ull<<32));
	Vector2i32 x = shuffled_scrambled_sobol2d(index, seed);
	return {(float)x.x() * S, (float)x.y() * S};
}

inline __host__ __device__ float ld_random_val(uint32_t index, uint32_t seed, uint32_t dim = 0) {
	constexpr float S = float(1.0/(1ull<<32));
	index = nested_uniform_scramble_base2(index, seed);
	return (float)nested_uniform_scramble_base2(sobol(index, dim), hash_combine(seed, dim)) * S;
}

template <uint32_t base>
__host__ __device__ float halton(size_t idx) {
	float f = 1;
	float result = 0;

	while (idx > 0) {
		f /= base;
		result += f * (idx % base);
		idx /= base;
	}

	return result;
}

inline __host__ __device__ Eigen::Vector2f halton23(size_t idx) {
	return {halton<2>(idx), halton<3>(idx)};
}

// Halton
// inline __host__ __device__ Eigen::Vector2f ld_random_pixel_offset(const uint32_t spp) {
// 	Eigen::Vector2f offset = Eigen::Vector2f::Constant(0.5f) - halton23(0) + halton23(spp);
// 	offset.x() = fractf(offset.x());
// 	offset.y() = fractf(offset.y());
// 	return offset;
// }

// Scrambled Sobol
inline __host__ __device__ Eigen::Vector2f ld_random_pixel_offset(const uint32_t spp) {
	Eigen::Vector2f offset = Eigen::Vector2f::Constant(0.5f) - ld_random_val_2d(0, 0xdeadbeef) + ld_random_val_2d(spp, 0xdeadbeef);
	offset.x() = fractf(offset.x());
	offset.y() = fractf(offset.y());
	return offset;
}

NGP_NAMESPACE_END

