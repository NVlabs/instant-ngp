/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_nerf_kernel.cuh
 *  @author Thomas Müller, NVIDIA
 *  @brief  pybind11 bindings for NGP's vector and matrix types. Adapted from
 *          Patrik Huber's glm binding code per the BSD license of pybind11.
 */

#pragma once

#include <neural-graphics-primitives/vec.h>

#include <cstddef>

#include <pybind11/numpy.h>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

/**
 * @file utils/pybind11_glm.hpp
 * @brief Transparent conversion to and from Python for glm vector and matrix types.
 *
 * All converters for matrices assume col-major storage of glm, the default.
 * Things will likely break if non-default storage order is used.
 */

template <typename T, uint32_t N>
struct type_caster<ngp::tvec<T, N>> {
	using vector_type = ngp::tvec<T, N>;
	using Scalar = T;
	static constexpr std::size_t num_elements = N;

	bool load(handle src, bool)
	{
		array_t<Scalar> buf(src, true);
		if (!buf.check())
			return false;

		if (buf.ndim() == 1) // a 1-dimensional vector
		{
			if (buf.shape(0) != num_elements) {
				return false; // not a 2-elements vector
			}
			if (buf.strides(0) != sizeof(Scalar))
			{
				std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
				return false;
			}
			value = vector_type(buf.mutable_data()); // make_vec* copies the data (unnecessarily)
		}
		else { // buf.ndim() != 1
			return false;
		}
		return true;
	}

	static handle cast(const vector_type& src, return_value_policy /* policy */, handle /* parent */)
	{
		return array(
			num_elements,			// shape
			src.data()		// data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("vec"));
};

template <typename T, uint32_t N, uint32_t M>
struct type_caster<ngp::tmat<T, N, M>> {
	using matrix_type = ngp::tmat<T, N, M>;
	using Scalar = T;
	static constexpr std::size_t num_rows = M;
	static constexpr std::size_t num_cols = N;

	bool load(handle src, bool)
	{
		array_t<Scalar> buf(src, true);
		if (!buf.check())
			return false;

		if (buf.ndim() == 2) // a 2-dimensional matrix
		{
			if (buf.shape(0) != num_rows || buf.shape(1) != num_cols) {
				return false; // not a 4x4 matrix
			}
			if (buf.strides(0) / sizeof(Scalar) != num_cols || buf.strides(1) != sizeof(Scalar))
			{
				std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
				return false;
			}
			// What we get from Python is laid out in row-major memory order, while GLM's
			// storage is col-major, thus, we transpose.
			value = ngp::transpose(matrix_type(buf.mutable_data())); // make_mat*() copies the data (unnecessarily)
		}
		else { // buf.ndim() != 2
			return false;
		}
		return true;
	}

	static handle cast(const matrix_type& src, return_value_policy /* policy */, handle /* parent */)
	{
		return array(
			{ num_rows, num_cols }, // shape
			{ sizeof(Scalar), sizeof(Scalar) * num_rows }, // strides - flip the row/col layout!
			src.data()                            // data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(matrix_type, _("mat"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
