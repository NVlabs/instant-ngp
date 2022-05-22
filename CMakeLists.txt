# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.18)

PROJECT(instant-ngp
	VERSION 1.0
	DESCRIPTION "Instant neural graphics primitives"
	LANGUAGES C CXX CUDA
)
set(NGP_VERSION "${CMAKE_PROJECT_VERSION}")

if (NOT NGP_DEPLOY)
	set(NGP_VERSION "${NGP_VERSION}dev")
endif()

option(NGP_BUILD_WITH_GUI "Build with GUI support (requires GLFW and GLEW)?" ON)
option(NGP_BUILD_WITH_VULKAN "Build with Vulkan to enable DLSS support?" ON)
option(NGP_BUILD_WITH_OPTIX "Build with OptiX to enable hardware ray tracing?" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/CMakeLists.txt")
	message(FATAL_ERROR
		"Some instant-ngp dependencies are missing. "
		"If you forgot the \"--recursive\" flag when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

###############################################################################
# Dependencies
###############################################################################

################
# tiny-cuda-nn #
################
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(dependencies/tiny-cuda-nn)
include_directories("dependencies/tiny-cuda-nn/include")
include_directories("dependencies/tiny-cuda-nn/dependencies")

if (NGP_BUILD_WITH_GUI)
	##########
	# Vulkan #
	##########
	find_package(Vulkan)
	if (Vulkan_FOUND AND NGP_BUILD_WITH_VULKAN)
		set(NGP_VULKAN ON)
		list(APPEND NGP_DEFINITIONS -DNGP_VULKAN -DGLFW_INCLUDE_VULKAN)
		include_directories("${Vulkan_INCLUDE_DIRS}")
		set(NGP_LIBRARIES ${NGP_LIBRARIES} ${Vulkan_LIBRARIES})

		set(GUI_SOURCES
			${GUI_SOURCES}
			src/dlss.cu
		)

		#############################################
		# DLSS (depends on vulkan, so appears here) #
		#############################################
		include_directories("dependencies/dlss/include")
		if (MSVC)
			link_directories("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Windows_x86_64/x86_64")
			if (CMAKE_BUILD_TYPE STREQUAL "Debug")
				set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx_d_dbg)
			else()
				set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx_d)
			endif()
		else()
			link_directories("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Linux_x86_64")
			set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx)
		endif()
	else()
		set(NGP_VULKAN OFF)
		if (NGP_BUILD_WITH_VULKAN)
			message(WARNING
				"Vulkan was not found. Neural graphics primitives will still compile "
				"and run correctly, but DLSS will not be supported."
			)
		endif()
	endif()

	########
	# GLFW #
	########
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
	set(GLFW_VULKAN_STATIC OFF CACHE BOOL " " FORCE)
	set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

	add_subdirectory(dependencies/glfw)

	set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

	mark_as_advanced(
		GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
		GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
		GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
		BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
	)

	include_directories("dependencies/glfw/include")
	include_directories("dependencies/imgui")

	#############
	# GL3W/GLEW #
	#############
	if (MSVC)
		include_directories("dependencies/gl3w")
		link_directories("dependencies")
		set(GUI_SOURCES ${GUI_SOURCES} "dependencies/gl3w/GL/gl3w.c")
		set(NGP_LIBRARIES ${NGP_LIBRARIES} opengl32 $<TARGET_OBJECTS:glfw_objects>)
	else()
		find_package(GLEW REQUIRED)
		include_directories(${GLEW_INCLUDE_DIRS})
		set(NGP_LIBRARIES ${NGP_LIBRARIES} GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects>)
	endif()

	set(GUI_SOURCES
		${GUI_SOURCES}
		dependencies/imguizmo/ImGuizmo.cpp
		dependencies/imgui/imgui.cpp
		dependencies/imgui/backends/imgui_impl_glfw.cpp
		dependencies/imgui/backends/imgui_impl_opengl3.cpp
		dependencies/imgui/imgui_draw.cpp
		dependencies/imgui/imgui_tables.cpp
		dependencies/imgui/imgui_widgets.cpp
	)

	list(APPEND NGP_DEFINITIONS -DNGP_GUI)
endif(NGP_BUILD_WITH_GUI)

include_directories("dependencies")

#########
# Eigen #
#########
include_directories("dependencies/eigen")
include_directories("dependencies/filesystem")

###########
# nanovdb #
###########
include_directories("dependencies/nanovdb")

##########
# OpenMP #
##########
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#########
# OPTIX #
#########
find_package(OptiX)
# set OptiX_INSTALL_DIR via your environment if it's not found automatically

if ((OptiX_FOUND OR OptiX_INCLUDE) AND NGP_BUILD_WITH_OPTIX)
	set(NGP_OPTIX ON)
	include_directories("${OptiX_INCLUDE}")
	list(APPEND NGP_DEFINITIONS -DNGP_OPTIX)
else()
	set(NGP_OPTIX OFF)
	if (NGP_BUILD_WITH_OPTIX)
		message(WARNING
			"OptiX was not found. Neural graphics primitives will still compile "
			"and run correctly. However, SDF training in 'raystab' and 'pathescape' "
			"modes will be significantly slower."
		)
	endif()
endif()

##########
# Python #
##########
find_package(Python 3.7 COMPONENTS Interpreter Development)
if (Python_FOUND)
	add_subdirectory("dependencies/pybind11")
endif()

##############
# tinylogger #
##############
include_directories("dependencies/tinylogger")


###############################################################################
# Program
###############################################################################

list(APPEND NGP_DEFINITIONS -DNGP_VERSION="${NGP_VERSION}")

include_directories("include")

set(SOURCES
	${GUI_SOURCES}
	src/camera_path.cu
	src/common_device.cu
	src/marching_cubes.cu
	src/nerf_loader.cu
	src/render_buffer.cu
	src/testbed.cu
	src/testbed_image.cu
	src/testbed_nerf.cu
	src/testbed_sdf.cu
	src/testbed_volume.cu
	src/thread_pool.cpp
	src/tinyexr_wrapper.cu
	src/tinyobj_loader_wrapper.cpp
	src/triangle_bvh.cu
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)

if (NGP_OPTIX)
	add_library(optix_program OBJECT
		src/optix/pathescape.cu
		src/optix/raystab.cu
		src/optix/raytrace.cu
	)

	set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON)
	set_target_properties(optix_program PROPERTIES CUDA_ARCHITECTURES OFF)
	target_compile_definitions(optix_program PUBLIC ${NGP_DEFINITIONS} -DTCNN_MIN_GPU_ARCH=0)
	target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")

	# OptiX programs will be compiled as PTX and packaged
	# as headers to be included from the binary dir.
	include_directories("${CMAKE_CURRENT_BINARY_DIR}")
	set(OPTIX_PTX_HEADER ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)

	find_program(bin_to_c NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
	if (NOT bin_to_c)
		message(FATAL_ERROR
			"bin2c not found:\n"
			"  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
			"  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
		)
	endif()

	add_custom_command(
		OUTPUT "${OPTIX_PTX_HEADER}"
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS=$<TARGET_OBJECTS:optix_program>"
		"-DOUTPUT=${OPTIX_PTX_HEADER}"
		-P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS optix_program
		COMMENT "Converting PTX files to a C header"
	)

	list(APPEND SOURCES ${OPTIX_PTX_HEADER})
endif()

add_library(ngp STATIC ${SOURCES})

set_target_properties(ngp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(ngp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ngp PUBLIC ${NGP_LIBRARIES} tiny-cuda-nn)
target_compile_options(ngp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(ngp PUBLIC ${NGP_DEFINITIONS} ${TCNN_DEFINITIONS})

add_executable(testbed src/main.cu)
target_link_libraries(testbed PUBLIC ngp)
target_compile_options(testbed PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(testbed PUBLIC ${NGP_DEFINITIONS} ${TCNN_DEFINITIONS})

# Copy DLSS shared libraries
if (NGP_VULKAN)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		set(NGX_BUILD_DIR "dev")
	else()
		set(NGX_BUILD_DIR "rel")
	endif()

	if (MSVC)
		add_custom_command(TARGET testbed POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Windows_x86_64/${NGX_BUILD_DIR}/nvngx_dlss.dll" $<TARGET_FILE_DIR:testbed>
			COMMAND_EXPAND_LISTS
		)
	else()
		file(GLOB DLSS_SOS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Linux_x86_64/${NGX_BUILD_DIR}/libnvidia-ngx-dlss.so.*")
		add_custom_command(TARGET testbed POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy ${DLSS_SOS} $<TARGET_FILE_DIR:testbed>
			COMMAND_EXPAND_LISTS
		)
	endif()
endif()


if (Python_FOUND)
	add_library(pyngp SHARED src/python_api.cu)
	set_target_properties(pyngp PROPERTIES CXX_VISIBILITY_PRESET "hidden")
	set_target_properties(pyngp PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
	target_link_libraries(pyngp PUBLIC ngp ${PYTHON_LIBRARIES} pybind11::module)
	target_compile_options(pyngp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
	target_compile_definitions(pyngp PUBLIC -DNGP_PYTHON ${NGP_DEFINITIONS} ${TCNN_DEFINITIONS})
	pybind11_extension(pyngp)

	if (MSVC)
		# Copy dlls needed at runtime
		file(GLOB CUBLAS_DLLS "${CUDA_COMPILER_BIN}/cublas*.dll")
		if (CUBLAS_DLLS)
			add_custom_command(TARGET pyngp POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy ${CUBLAS_DLLS} $<TARGET_FILE_DIR:pyngp>
				COMMAND_EXPAND_LISTS
			)
		endif()
	endif()
endif()
