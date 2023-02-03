/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   openxr_hmd.h
 *  @author Thomas Müller & Ingo Esser & Robert Menzel, NVIDIA
 *  @brief  Wrapper around the OpenXR API, providing access to
 *          per-eye framebuffers, lens parameters, visible area,
 *          view, hand, and eye poses, as well as controller inputs.
 */

#pragma once

#ifdef _WIN32
#  include <GL/gl3w.h>
#else
#  include <GL/glew.h>
#endif

#define XR_USE_GRAPHICS_API_OPENGL

#include <neural-graphics-primitives/common_device.cuh>

#include <openxr/openxr.h>
#include <xr_linear.h>
#include <xr_dependencies.h>
#include <openxr/openxr_platform.h>

#include <Eigen/Dense>

#include <tiny-cuda-nn/gpu_memory.h>

#include <array>
#include <memory>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers" //TODO: XR struct are uninitiaized apart from their type
#endif

NGP_NAMESPACE_BEGIN

enum class EEnvironmentBlendMode {
	Opaque = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
	Additive = XR_ENVIRONMENT_BLEND_MODE_ADDITIVE,
	AlphaBlend = XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND,
};

inline std::string to_string(EEnvironmentBlendMode mode) {
	switch (mode) {
		case EEnvironmentBlendMode::Opaque: return "Opaque";
		case EEnvironmentBlendMode::Additive: return "Additive";
		case EEnvironmentBlendMode::AlphaBlend: return "Blend";
		default: throw std::runtime_error{"Invalid blend mode."};
	}
}

class OpenXRHMD {
public:
	enum class EControlFlow {
		Continue,
		Restart,
		Quit,
	};

	struct FrameInfo {
		struct View {
			GLuint framebuffer;
			XrCompositionLayerProjectionView view{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
			XrCompositionLayerDepthInfoKHR depth_info{XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR};
			std::shared_ptr<Buffer2D<uint8_t>> hidden_area_mask = nullptr;
			Eigen::Matrix<float, 3, 4> pose;
		};
		struct Hand {
			Eigen::Matrix<float, 3, 4> pose;
			bool pose_active = false;
			Eigen::Vector2f thumbstick = Eigen::Vector2f::Zero();
			float grab_strength = 0.0f;
			bool grabbing = false;
			bool pressing = false;
			Eigen::Vector3f grab_pos;
			Eigen::Vector3f prev_grab_pos;
			Eigen::Vector3f drag() const {
				return grab_pos - prev_grab_pos;
			}
		};
		std::vector<View> views;
		Hand hands[2];
	};
	using FrameInfoPtr = std::shared_ptr<FrameInfo>;

	// RAII OpenXRHMD with OpenGL
#if defined(XR_USE_PLATFORM_WIN32)
	OpenXRHMD(HDC hdc, HGLRC hglrc);
#elif defined(XR_USE_PLATFORM_XLIB)
	OpenXRHMD(Display* xDisplay, uint32_t visualid, GLXFBConfig glxFBConfig, GLXDrawable glxDrawable, GLXContext glxContext);
#elif defined(XR_USE_PLATFORM_WAYLAND)
	OpenXRHMD(wl_display* display);
#endif

	virtual ~OpenXRHMD();

	// disallow copy / move
	OpenXRHMD(const OpenXRHMD&) = delete;
	OpenXRHMD& operator=(const OpenXRHMD&) = delete;
	OpenXRHMD(OpenXRHMD&&) = delete;
	OpenXRHMD& operator=(OpenXRHMD&&) = delete;

	void clear();

	// poll events, handle state changes, return control flow information
	EControlFlow poll_events();

	// begin OpenXR frame, return views to render
	FrameInfoPtr begin_frame();
	// must be called for each begin_frame
	void end_frame(FrameInfoPtr frame_info, float znear, float zfar);

	void set_environment_blend_mode(EEnvironmentBlendMode mode) {
		m_environment_blend_mode = mode;
	}

	EEnvironmentBlendMode environment_blend_mode() const {
		return m_environment_blend_mode;
	}

	const std::vector<EEnvironmentBlendMode>& supported_environment_blend_modes() const {
		return m_supported_environment_blend_modes;
	}

	const char* supported_environment_blend_modes_imgui_string() const {
		return m_supported_environment_blend_modes_imgui_string.data();
	}

	// if true call begin_frame and end_frame - does not imply visibility
	bool must_run_frame_loop() const {
		return
			m_session_state == XR_SESSION_STATE_READY ||
			m_session_state == XR_SESSION_STATE_SYNCHRONIZED ||
			m_session_state == XR_SESSION_STATE_VISIBLE ||
			m_session_state == XR_SESSION_STATE_FOCUSED;
	}

	// if true, VR is being rendered to the HMD
	bool is_visible() const {
		// XR_SESSION_STATE_VISIBLE -> app content is shown in HMD
		// XR_SESSION_STATE_FOCUSED -> VISIBLE + input is send to app
		return m_session_state == XR_SESSION_STATE_VISIBLE || m_session_state == XR_SESSION_STATE_FOCUSED;
	}

private:
	// steps of the init process, called from the constructor
	void init_create_xr_instance();
	void init_get_xr_system();
	void init_configure_xr_views();
	void init_check_for_xr_blend_mode();
	void init_xr_actions();

#if defined(XR_USE_PLATFORM_WIN32)
	void init_open_gl(HDC hdc, HGLRC hglrc);
#elif defined(XR_USE_PLATFORM_XLIB)
	void init_open_gl(Display* xDisplay, uint32_t visualid, GLXFBConfig glxFBConfig, GLXDrawable glxDrawable, GLXContext glxContext);
#elif defined(XR_USE_PLATFORM_WAYLAND)
	void init_open_gl(wl_display* display);
#endif

	void init_xr_session();
	void init_xr_spaces();
	void init_xr_swapchain_open_gl();
	void init_open_gl_shaders();

	// session state change
	void session_state_change(XrSessionState state, EControlFlow& flow);

	std::shared_ptr<Buffer2D<uint8_t>> rasterize_hidden_area_mask(uint32_t view_index, const XrCompositionLayerProjectionView& view);
	// system/instance
	XrInstance m_instance{XR_NULL_HANDLE};
	XrSystemId m_system_id = {};
	XrInstanceProperties m_instance_properties = {XR_TYPE_INSTANCE_PROPERTIES};
	XrSystemProperties m_system_properties = {XR_TYPE_SYSTEM_PROPERTIES};
	std::vector<XrApiLayerProperties> m_api_layer_properties;
	std::vector<XrExtensionProperties> m_instance_extension_properties;

	// view and blending
	XrViewConfigurationType m_view_configuration_type = {};
	XrViewConfigurationProperties m_view_configuration_properties = {XR_TYPE_VIEW_CONFIGURATION_PROPERTIES};
	std::vector<XrViewConfigurationView> m_view_configuration_views;
	std::vector<EEnvironmentBlendMode> m_supported_environment_blend_modes;
	std::vector<char> m_supported_environment_blend_modes_imgui_string;
	EEnvironmentBlendMode m_environment_blend_mode = EEnvironmentBlendMode::Opaque;

	// actions
	std::array<XrPath, 2> m_hand_paths;
	std::array<XrSpace, 2> m_hand_spaces;
	XrAction m_pose_action{XR_NULL_HANDLE};
	XrAction m_press_action{XR_NULL_HANDLE};
	XrAction m_grab_action{XR_NULL_HANDLE};

	// Two separate actions for Xbox controller support
	std::array<XrAction, 2> m_thumbstick_actions;

	XrActionSet m_action_set{XR_NULL_HANDLE};

#if defined(XR_USE_PLATFORM_WIN32)
	XrGraphicsBindingOpenGLWin32KHR m_graphics_binding{XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR};
#elif defined(XR_USE_PLATFORM_XLIB)
	XrGraphicsBindingOpenGLXlibKHR m_graphics_binding{XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR};
#elif defined(XR_USE_PLATFORM_WAYLAND)
	XrGraphicsBindingOpenGLWaylandKHR m_graphics_binding{XR_TYPE_GRAPHICS_BINDING_OPENGL_WAYLAND_KHR};
#endif

	XrSession m_session{XR_NULL_HANDLE};
	XrSessionState m_session_state{XR_SESSION_STATE_UNKNOWN};

	// reference space
	std::vector<XrReferenceSpaceType> m_reference_spaces;
	XrSpace m_space{XR_NULL_HANDLE};
	XrExtent2Df m_bounds;

	// swap chains
	struct Swapchain {
		Swapchain(XrSwapchainCreateInfo& rgba_create_info, XrSwapchainCreateInfo& depth_create_info, XrSession& session, XrInstance& xr_instance);
		Swapchain(const Swapchain&) = delete;
		Swapchain& operator=(const Swapchain&) = delete;
		Swapchain(Swapchain&& other) {
			*this = std::move(other);
		}
		Swapchain& operator=(Swapchain&& other) {
			std::swap(handle, other.handle);
			std::swap(depth_handle, other.depth_handle);
			std::swap(width, other.width);
			std::swap(height, other.height);
			images_gl = std::move(other.images_gl);
			depth_images_gl = std::move(other.depth_images_gl);
			framebuffers_gl = std::move(other.framebuffers_gl);
			return *this;
		}
		virtual ~Swapchain();

		void clear();

		XrSwapchain handle{XR_NULL_HANDLE};
		XrSwapchain depth_handle{XR_NULL_HANDLE};

		int32_t width = 0;
		int32_t height = 0;
		std::vector<XrSwapchainImageOpenGLKHR> images_gl;
		std::vector<XrSwapchainImageOpenGLKHR> depth_images_gl;
		std::vector<GLuint> framebuffers_gl;
	};

	int64_t m_swapchain_rgba_format = 0;
	std::vector<Swapchain> m_swapchains;

	bool m_supports_composition_layer_depth = false;
	int64_t m_swapchain_depth_format = 0;

	bool m_supports_hidden_area_mask = false;
	std::vector<std::shared_ptr<Buffer2D<uint8_t>>> m_hidden_area_masks;

	bool m_supports_eye_tracking = false;

	// frame data
	XrFrameState m_frame_state{XR_TYPE_FRAME_STATE};
	FrameInfoPtr m_previous_frame_info;

	GLuint m_hidden_area_mask_program = 0;

	// print more debug info during OpenXRs init:
	const bool m_print_api_layers = false;
	const bool m_print_extensions = false;
	const bool m_print_system_properties = false;
	const bool m_print_instance_properties = false;
	const bool m_print_view_configuration_types = false;
	const bool m_print_view_configuration_properties = false;
	const bool m_print_view_configuration_view = false;
	const bool m_print_environment_blend_modes = false;
	const bool m_print_available_swapchain_formats = false;
	const bool m_print_reference_spaces = false;
};

NGP_NAMESPACE_END

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
