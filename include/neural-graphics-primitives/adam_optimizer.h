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


NGP_NAMESPACE_BEGIN

template <typename T>
class AdamOptimizer {
public:
	AdamOptimizer(float learning_rate, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.99f) {
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

		float actual_learning_rate = m_hparams.learning_rate * std::sqrt(1 - std::pow(m_hparams.beta2, (float)m_state.iter)) / (1 - std::pow(m_hparams.beta1, (float)m_state.iter));
		m_state.first_moment = m_hparams.beta1 * m_state.first_moment + (1 - m_hparams.beta1) * gradient;
		m_state.second_moment = m_hparams.beta2 * m_state.second_moment + (1 - m_hparams.beta2) * gradient.cwiseProduct(gradient);
		m_state.variable -= actual_learning_rate * m_state.first_moment.cwiseQuotient(m_state.second_moment.cwiseSqrt() + T::Constant(m_hparams.epsilon));
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

	private:
	struct State {
		int iter = 0;
		T first_moment = T::Zero();
		T second_moment = T::Zero();
		T variable = T::Zero();
	} m_state;

	struct Hyperparameters {
		float learning_rate;
		float epsilon;
		float beta1;
		float beta2;
	} m_hparams;
};



class RotationAdamOptimizer {
public:
	RotationAdamOptimizer(float learning_rate, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.99f) {
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

	void step(const Eigen::Vector3f& gradient) {
		++m_state.iter;

		float actual_learning_rate = m_hparams.learning_rate * std::sqrt(1 - std::pow(m_hparams.beta2, m_state.iter)) / (1 - std::pow(m_hparams.beta1, m_state.iter));
		m_state.first_moment = m_hparams.beta1 * m_state.first_moment + (1 - m_hparams.beta1) * gradient;
		m_state.second_moment = m_hparams.beta2 * m_state.second_moment + (1 - m_hparams.beta2) * gradient.cwiseProduct(gradient);
		Eigen::Vector3f rot = actual_learning_rate * m_state.first_moment.cwiseQuotient(m_state.second_moment.cwiseSqrt() + Eigen::Vector3f::Constant(m_hparams.epsilon));
		float rot_len = rot.norm();
		float var_len = variable().norm();

		Eigen::AngleAxisf result;
		Eigen::Matrix3f mat = Eigen::AngleAxisf(rot_len, rot).toRotationMatrix() * Eigen::AngleAxisf(var_len, variable()/var_len).toRotationMatrix();
		result.fromRotationMatrix(mat);
		m_state.variable = result.axis() * result.angle();
	}

	void set_learning_rate(float lr) {
		m_hparams.learning_rate = lr;
	}

	const Eigen::Vector3f& variable() const {
		return m_state.variable;
	}

	private:
	struct State {
		int iter = 0;
		Eigen::Vector3f first_moment = Eigen::Vector3f::Zero();
		Eigen::Vector3f second_moment = Eigen::Vector3f::Zero();
		Eigen::Vector3f variable = Eigen::Vector3f::Zero();
	} m_state;

	struct Hyperparameters {
		float learning_rate;
		float epsilon;
		float beta1;
		float beta2;
	} m_hparams;
};

NGP_NAMESPACE_END
