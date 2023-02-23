/*
    utils/pybind11_glm.hpp: Transparent conversion for glm types to NumPy arrays.
                            This header is based on pybind11/eigen.h.

    Copyright (c) 2016 Patrik Huber

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in pybind11's LICENSE file.
*/
#pragma once

#include <cstddef>

#include <pybind11/numpy.h>

#include <glm/gtc/type_ptr.hpp> // includes all vector and matrix types too

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

template <typename T, glm::precision P>
struct type_caster<glm::tvec2<T, P>>
{
	using vector_type = glm::tvec2<T, P>;
	using Scalar = T;
	static constexpr std::size_t num_elements = 2;

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
			value = glm::make_vec2(buf.mutable_data()); // make_vec* copies the data (unnecessarily)
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
			glm::value_ptr(src)		// data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("vec2"));
};

template<typename T, glm::precision P>
struct type_caster<glm::tvec3<T, P>>
{
	using vector_type = glm::tvec3<T, P>;
	using Scalar = T;
	static constexpr std::size_t num_elements = 3;

	bool load(handle src, bool)
	{
		array_t<Scalar> buf(src, true);
		if (!buf.check())
			return false;

		if (buf.ndim() == 1) // a 1-dimensional vector
		{
			if (buf.shape(0) != num_elements) {
				return false; // not a 3-elements vector
			}
			if (buf.strides(0) != sizeof(Scalar))
			{
				std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
				return false;
			}
			value = glm::make_vec3(buf.mutable_data()); // make_vec* copies the data (unnecessarily)
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
			glm::value_ptr(src)		// data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("vec3"));
};

template<typename T, glm::precision P>
struct type_caster<glm::tvec4<T, P>>
{
	using vector_type = glm::tvec4<T, P>;
	using Scalar = T;
	static constexpr std::size_t num_elements = 4;

	bool load(handle src, bool)
	{
		array_t<Scalar> buf(src, true);
		if (!buf.check())
			return false;

		if (buf.ndim() == 1) // a 1-dimensional vector
		{
			if (buf.shape(0) != num_elements) {
				return false; // not a 4-elements vector
			}
			if (buf.strides(0) != sizeof(Scalar))
			{
				std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
				return false;
			}
			value = glm::make_vec4(buf.mutable_data()); // make_vec* copies the data (unnecessarily)
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
			glm::value_ptr(src)		// data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("vec4"));
};

template<typename T, glm::precision P>
struct type_caster<glm::tmat4x3<T, P>>
{
	using matrix_type = glm::tmat4x3<T, P>;
	using Scalar = T;
	static constexpr std::size_t num_rows = 3;
	static constexpr std::size_t num_cols = 4;

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
			value = glm::transpose(glm::make_mat3x4(buf.mutable_data())); // make_mat*() copies the data (unnecessarily)
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
			glm::value_ptr(src)                            // data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(matrix_type, _("mat4x3"));
};

template<typename T, glm::precision P>
struct type_caster<glm::tmat4x4<T, P>>
{
	using matrix_type = glm::tmat4x4<T, P>;
	using Scalar = T;
	static constexpr std::size_t num_rows = 4;
	static constexpr std::size_t num_cols = 4;

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
			value = glm::transpose(glm::make_mat4x4(buf.mutable_data())); // make_mat*() copies the data (unnecessarily)
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
			glm::value_ptr(src)                            // data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(matrix_type, _("mat4x4"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
