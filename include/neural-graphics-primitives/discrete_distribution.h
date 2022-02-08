/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   discrete_distribution.h
 *  @author Thomas Müller, NVIDIA
 */

#pragma once

NGP_NAMESPACE_BEGIN

struct DiscreteDistribution {
	void build(std::vector<float> weights) {
		float total_weight = 0;
		for (float w : weights) {
			total_weight += w;
		}
		float inv_total_weight = 1 / total_weight;

		float cdf_accum = 0;
		cdf.clear();
		for (float w : weights) {
			float norm = w * inv_total_weight;
			cdf_accum += norm;
			pmf.emplace_back(norm);
			cdf.emplace_back(cdf_accum);
		}
		cdf.back() = 1.0f; // Prevent precision problems from causing overruns in the end
	}

	uint32_t sample(float val) {
		return std::min(binary_search(val, cdf.data(), (uint32_t)cdf.size()), (uint32_t)cdf.size()-1);
	}

	std::vector<float> pmf;
	std::vector<float> cdf;
};

NGP_NAMESPACE_END
