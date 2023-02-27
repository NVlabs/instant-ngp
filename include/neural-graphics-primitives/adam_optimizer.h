/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   adam_optimizer.h
 *  @author Thomas Müller, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/json_binding.h>

#include <json/json.hpp>

NGP_NAMESPACE_BEGIN

class VarAdamOptimizer {
public:
	VarAdamOptimizer(size_t size = 0, float learning_rate = 1e-3, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.99f) : m_state{size} {
		m_hparams = { learning_rate, epsilon, beta1, beta2 };
	}

	VarAdamOptimizer& operator=(const VarAdamOptimizer& arg) {
		m_state = arg.m_state;
		m_hparams = arg.m_hparams;
		return *this;
	}

	VarAdamOptimizer(const VarAdamOptimizer& arg) {
		*this = arg;
	}

	void step(const std::vector<float>& gradient) {
		++m_state.iter;

		float actual_learning_rate = m_hparams.learning_rate * std::sqrt(1.0f - std::pow(m_hparams.beta2, (float)m_state.iter)) / (1.0f - std::pow(m_hparams.beta1, (float)m_state.iter));

		for (size_t i = 0; i < m_state.first_moment.size(); ++i) {
			m_state.first_moment[i] = m_hparams.beta1 * m_state.first_moment[i] + (1.0f - m_hparams.beta1) * gradient[i];
			m_state.second_moment[i] = m_hparams.beta2 * m_state.second_moment[i] + (1.0f - m_hparams.beta2) * gradient[i] * gradient[i];
			m_state.variable[i] -= actual_learning_rate * m_state.first_moment[i] / (std::sqrt(m_state.second_moment[i]) + m_hparams.epsilon);
		}
	}

	uint32_t step() const {
		return m_state.iter;
	}

	void set_learning_rate(float lr) {
		m_hparams.learning_rate = lr;
	}

	std::vector<float>& variable() {
		return m_state.variable;
	}

	const std::vector<float>& variable() const {
		return m_state.variable;
	}

	void reset_state() {
		m_state = State{m_state.first_moment.size()};
	}

	void to_json(nlohmann::json& j) const {
		j["iter"] = m_state.iter;
		j["first_moment"] = m_state.first_moment;
		j["second_moment"] = m_state.second_moment;
		j["variable"] = m_state.variable;
		j["learning_rate"] = m_hparams.learning_rate;
		j["epsilon"] = m_hparams.epsilon;
		j["beta1"] = m_hparams.beta1;
		j["beta2"] = m_hparams.beta2;
	}

	void from_json(const nlohmann::json& j) {
		m_state.iter = j.at("iter");
		m_state.first_moment = j.at("first_moment").get<std::vector<float>>();
		m_state.second_moment = j.at("second_moment").get<std::vector<float>>();
		m_state.variable = j.at("variable").get<std::vector<float>>();
		m_hparams.learning_rate = j.at("learning_rate");
		m_hparams.epsilon = j.at("epsilon");
		m_hparams.beta1 = j.at("beta1");
		m_hparams.beta2 = j.at("beta2");
	}

private:
	struct State {
		State() = default;
		State(const State&) = default;
		State(size_t size) {
			iter = 0;
			first_moment = std::vector<float>(size, 0.0f);
			second_moment = std::vector<float>(size, 0.0f);
			variable = std::vector<float>(size, 0.0f);
		}

		uint32_t iter;
		std::vector<float> first_moment;
		std::vector<float> second_moment;
		std::vector<float> variable;
	} m_state;

	struct Hyperparameters {
		float learning_rate;
		float epsilon;
		float beta1;
		float beta2;
	} m_hparams;
};

inline void to_json(nlohmann::json& j, const VarAdamOptimizer& opt) {
	opt.to_json(j);
}

inline void from_json(const nlohmann::json& j, VarAdamOptimizer& opt) {
	opt.from_json(j);
}

template <typename T>
class AdamOptimizer {
public:
	AdamOptimizer(float learning_rate = 1e-3, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.99f) {
		m_hparams = { learning_rate, epsilon, beta1, beta2 };
	}

	AdamOptimizer& operator=(const AdamOptimizer& arg) {
		m_state = arg.m_state;
		m_hparams = arg.m_hparams;
		return *this;
	}

	AdamOptimizer(const AdamOptimizer& arg) {
		*this = arg;
	}

	void step(const T& gradient) {
		++m_state.iter;

		float actual_learning_rate = m_hparams.learning_rate * std::sqrt(1.0f - std::pow(m_hparams.beta2, (float)m_state.iter)) / (1.0f - std::pow(m_hparams.beta1, (float)m_state.iter));
		m_state.first_moment = m_hparams.beta1 * m_state.first_moment + (1.0f - m_hparams.beta1) * gradient;
		m_state.second_moment = m_hparams.beta2 * m_state.second_moment + (1.0f - m_hparams.beta2) * gradient * gradient;
		m_state.variable -= actual_learning_rate * m_state.first_moment / (sqrt(m_state.second_moment) + T(m_hparams.epsilon));
	}

	uint32_t step() const {
		return m_state.iter;
	}

	void set_learning_rate(float lr) {
		m_hparams.learning_rate = lr;
	}

	T& variable() {
		return m_state.variable;
	}

	const T& variable() const {
		return m_state.variable;
	}

	void reset_state() {
		m_state = {};
	}

	void to_json(nlohmann::json& j) const {
		j["iter"] = m_state.iter;
		j["first_moment"] = m_state.first_moment;
		j["second_moment"] = m_state.second_moment;
		j["variable"] = m_state.variable;
		j["learning_rate"] = m_hparams.learning_rate;
		j["epsilon"] = m_hparams.epsilon;
		j["beta1"] = m_hparams.beta1;
		j["beta2"] = m_hparams.beta2;
	}

	void from_json(const nlohmann::json& j) {
		m_state.iter = j.at("iter");
		m_state.first_moment = j.at("first_moment");
		m_state.second_moment = j.at("second_moment");
		m_state.variable = j.at("variable");
		m_hparams.learning_rate = j.at("learning_rate");
		m_hparams.epsilon = j.at("epsilon");
		m_hparams.beta1 = j.at("beta1");
		m_hparams.beta2 = j.at("beta2");
	}

private:
	struct State {
		uint32_t iter = 0;
		T first_moment = T(0.0f);
		T second_moment = T(0.0f);
		T variable = T(0.0f);
	} m_state = {};

	struct Hyperparameters {
		float learning_rate;
		float epsilon;
		float beta1;
		float beta2;
	} m_hparams = {};
};

template <typename T>
inline void to_json(nlohmann::json& j, const AdamOptimizer<T>& opt) {
	opt.to_json(j);
}

template <typename T>
inline void from_json(const nlohmann::json& j, AdamOptimizer<T>& opt) {
	opt.from_json(j);
}

class RotationAdamOptimizer {
public:
	RotationAdamOptimizer(float learning_rate = 1e-3, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.99f) {
		m_hparams = { learning_rate, epsilon, beta1, beta2 };
	}

	RotationAdamOptimizer& operator=(const RotationAdamOptimizer& arg) {
		m_state = arg.m_state;
		m_hparams = arg.m_hparams;
		return *this;
	}

	RotationAdamOptimizer(const RotationAdamOptimizer& arg) {
		*this = arg;
	}

	void step(const vec3& gradient) {
		++m_state.iter;

		float actual_learning_rate = m_hparams.learning_rate * std::sqrt(1 - std::pow(m_hparams.beta2, m_state.iter)) / (1 - std::pow(m_hparams.beta1, m_state.iter));
		m_state.first_moment = m_hparams.beta1 * m_state.first_moment + (1 - m_hparams.beta1) * gradient;
		m_state.second_moment = m_hparams.beta2 * m_state.second_moment + (1 - m_hparams.beta2) * gradient * gradient;
		vec3 rot = actual_learning_rate * m_state.first_moment / (sqrt(m_state.second_moment) + vec3(m_hparams.epsilon));

		m_state.variable = rotvec(rotmat(-rot) * rotmat(variable()));
	}

	uint32_t step() const {
		return m_state.iter;
	}

	void set_learning_rate(float lr) {
		m_hparams.learning_rate = lr;
	}

	const vec3& variable() const {
		return m_state.variable;
	}

	void reset_state() {
		m_state = {};
	}

	void to_json(nlohmann::json& j) const {
		j["iter"] = m_state.iter;
		j["first_moment"] = m_state.first_moment;
		j["second_moment"] = m_state.second_moment;
		j["variable"] = m_state.variable;
		j["learning_rate"] = m_hparams.learning_rate;
		j["epsilon"] = m_hparams.epsilon;
		j["beta1"] = m_hparams.beta1;
		j["beta2"] = m_hparams.beta2;
	}

	void from_json(const nlohmann::json& j) {
		m_state.iter = j.at("iter");
		m_state.first_moment = j.at("first_moment");
		m_state.second_moment = j.at("second_moment");
		m_state.variable = j.at("variable");
		m_hparams.learning_rate = j.at("learning_rate");
		m_hparams.epsilon = j.at("epsilon");
		m_hparams.beta1 = j.at("beta1");
		m_hparams.beta2 = j.at("beta2");
	}

private:
	struct State {
		uint32_t iter = 0;
		vec3 first_moment = vec3(0.0f);
		vec3 second_moment = vec3(0.0f);
		vec3 variable = vec3(0.0f);
	} m_state;

	struct Hyperparameters {
		float learning_rate;
		float epsilon;
		float beta1;
		float beta2;
	} m_hparams;
};

inline void to_json(nlohmann::json& j, const RotationAdamOptimizer& opt) {
	opt.to_json(j);
}

inline void from_json(const nlohmann::json& j, RotationAdamOptimizer& opt) {
	opt.from_json(j);
}

NGP_NAMESPACE_END
