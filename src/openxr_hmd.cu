/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   openxr_hmd.cu
 *  @author Thomas Müller & Ingo Esser & Robert Menzel, NVIDIA
 *  @brief  Wrapper around the OpenXR API, providing access to
 *          per-eye framebuffers, lens parameters, visible area,
 *          view, hand, and eye poses, as well as controller inputs.
 */

#define NOMINMAX

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/openxr_hmd.h>
#include <neural-graphics-primitives/render_buffer.h>

#include <openxr/openxr_reflection.h>

#include <fmt/format.h>

#include <imgui/imgui.h>

#include <tinylogger/tinylogger.h>

#include <tiny-cuda-nn/common.h>

#include <string>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers" //TODO: XR struct are uninitiaized apart from their type
#endif

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

// function XrEnumStr turns enum into string for printing
// uses expansion macro and data provided in openxr_reflection.h
#define XR_ENUM_CASE_STR(name, val) \
	case name:                      \
		return #name;
#define XR_ENUM_STR(enum_type)                                                     \
	constexpr const char* XrEnumStr(enum_type e) {                                 \
		switch (e) {                                                               \
			XR_LIST_ENUM_##enum_type(XR_ENUM_CASE_STR) default : return "Unknown"; \
		}                                                                          \
	}

XR_ENUM_STR(XrViewConfigurationType)
XR_ENUM_STR(XrEnvironmentBlendMode)
XR_ENUM_STR(XrReferenceSpaceType)
XR_ENUM_STR(XrStructureType)
XR_ENUM_STR(XrSessionState)

/// Checks the result of a xrXXXXXX call and throws an error on failure
#define XR_CHECK_THROW(x)                                                                                   \
	do {                                                                                                              \
		XrResult result = x;                                                                                          \
		if (XR_FAILED(result)) {                                                                                      \
			char buffer[XR_MAX_RESULT_STRING_SIZE];                                                                   \
			XrResult result_to_string_result = xrResultToString(m_instance, result, buffer);                            \
			if (XR_FAILED(result_to_string_result)) {                                                                 \
				throw std::runtime_error{std::string(FILE_LINE " " #x " failed, but could not obtain error string")}; \
			} else {                                                                                                  \
				throw std::runtime_error{std::string(FILE_LINE " " #x " failed with error ") + buffer};               \
			}                                                                                                         \
		}                                                                                                             \
	} while(0)

OpenXRHMD::Swapchain::Swapchain(XrSwapchainCreateInfo& rgba_create_info, XrSwapchainCreateInfo& depth_create_info, XrSession& session, XrInstance& m_instance) {
	ScopeGuard cleanup_guard{[&]() { clear(); }};

	XR_CHECK_THROW(xrCreateSwapchain(session, &rgba_create_info, &handle));

	width = rgba_create_info.width;
	height = rgba_create_info.height;

	{
		uint32_t size;
		XR_CHECK_THROW(xrEnumerateSwapchainImages(handle, 0, &size, nullptr));

		images_gl.resize(size, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
		XR_CHECK_THROW(xrEnumerateSwapchainImages(handle, size, &size, (XrSwapchainImageBaseHeader*)images_gl.data()));

		// One framebuffer per swapchain image
		framebuffers_gl.resize(size);
	}

	if (depth_create_info.format != 0) {
		XR_CHECK_THROW(xrCreateSwapchain(session, &depth_create_info, &depth_handle));

		uint32_t depth_size;
		XR_CHECK_THROW(xrEnumerateSwapchainImages(depth_handle, 0, &depth_size, nullptr));

		depth_images_gl.resize(depth_size, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
		XR_CHECK_THROW(xrEnumerateSwapchainImages(depth_handle, depth_size, &depth_size, (XrSwapchainImageBaseHeader*)depth_images_gl.data()));

		// We might have a different number of depth swapchain images as we have framebuffers,
		// so we will need to bind an acquired depth image to the current framebuffer on the
		// fly later on.
	}

	glGenFramebuffers(framebuffers_gl.size(), framebuffers_gl.data());

	cleanup_guard.disarm();
}

OpenXRHMD::Swapchain::~Swapchain() {
	clear();
}

void OpenXRHMD::Swapchain::clear() {
	if (!framebuffers_gl.empty()) {
		glDeleteFramebuffers(framebuffers_gl.size(), framebuffers_gl.data());
	}

	if (depth_handle != XR_NULL_HANDLE) {
		xrDestroySwapchain(depth_handle);
		depth_handle = XR_NULL_HANDLE;
	}

	if (handle != XR_NULL_HANDLE) {
		xrDestroySwapchain(handle);
		handle = XR_NULL_HANDLE;
	}
}

#if defined(XR_USE_PLATFORM_WIN32)
OpenXRHMD::OpenXRHMD(HDC hdc, HGLRC hglrc) {
#elif defined(XR_USE_PLATFORM_XLIB)
OpenXRHMD::OpenXRHMD(Display* xDisplay, uint32_t visualid, GLXFBConfig glxFBConfig, GLXDrawable glxDrawable, GLXContext glxContext) {
#elif defined(XR_USE_PLATFORM_WAYLAND)
OpenXRHMD::OpenXRHMD(wl_display* display) {
#endif
	ScopeGuard cleanup_guard{[&]() { clear(); }};

	init_create_xr_instance();
	init_get_xr_system();
	init_configure_xr_views();
	init_check_for_xr_blend_mode();
#if defined(XR_USE_PLATFORM_WIN32)
	init_open_gl(hdc, hglrc);
#elif defined(XR_USE_PLATFORM_XLIB)
	init_open_gl(xDisplay, visualid, glxFBConfig, glxDrawable, glxContext);
#elif defined(XR_USE_PLATFORM_WAYLAND)
	init_open_gl(display);
#endif
	init_xr_session();
	init_xr_actions();
	init_xr_spaces();
	init_xr_swapchain_open_gl();
	init_open_gl_shaders();

	cleanup_guard.disarm();
	tlog::success() << "Initialized OpenXR for " << m_system_properties.systemName;
	// tlog::success() << " "
	// 	<< " depth=" << (m_supports_composition_layer_depth ? "true" : "false")
	// 	<< " mask=" << (m_supports_hidden_area_mask ? "true" : "false")
	// 	<< " eye=" << (m_supports_hidden_area_mask ? "true" : "false")
	// 	;
}

OpenXRHMD::~OpenXRHMD() {
	clear();
}

void OpenXRHMD::clear() {
	auto xr_destroy = [&](auto& handle, auto destroy_fun) {
		if (handle != XR_NULL_HANDLE) {
			destroy_fun(handle);
			handle = XR_NULL_HANDLE;
		}
	};

	xr_destroy(m_pose_action, xrDestroyAction);
	xr_destroy(m_thumbstick_actions[0], xrDestroyAction);
	xr_destroy(m_thumbstick_actions[1], xrDestroyAction);
	xr_destroy(m_press_action, xrDestroyAction);
	xr_destroy(m_grab_action, xrDestroyAction);

	xr_destroy(m_action_set, xrDestroyActionSet);

	m_swapchains.clear();
	xr_destroy(m_space, xrDestroySpace);
	xr_destroy(m_session, xrDestroySession);
	xr_destroy(m_instance, xrDestroyInstance);
}

void OpenXRHMD::init_create_xr_instance() {
	std::vector<const char*> layers = {};
	std::vector<const char*> extensions = {
		XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
	};

	auto print_extension_properties = [](const char* layer_name) {
		uint32_t size;
		xrEnumerateInstanceExtensionProperties(layer_name, 0, &size, nullptr);
		std::vector<XrExtensionProperties> props(size, {XR_TYPE_EXTENSION_PROPERTIES});
		xrEnumerateInstanceExtensionProperties(layer_name, size, &size, props.data());
		tlog::info() << fmt::format("Extensions ({}):", props.size());
		for (XrExtensionProperties extension : props) {
			tlog::info() << fmt::format("\t{} (Version {})", extension.extensionName, extension.extensionVersion);
		}
	};

	uint32_t size;
	xrEnumerateApiLayerProperties(0, &size, nullptr);
	m_api_layer_properties.clear();
	m_api_layer_properties.resize(size, {XR_TYPE_API_LAYER_PROPERTIES});
	xrEnumerateApiLayerProperties(size, &size, m_api_layer_properties.data());

	if (m_print_api_layers) {
		tlog::info() << fmt::format("API Layers ({}):", m_api_layer_properties.size());
		for (auto p : m_api_layer_properties) {
			tlog::info() << fmt::format(
				"{} (v {}.{}.{}, {}) {}",
				p.layerName,
				XR_VERSION_MAJOR(p.specVersion),
				XR_VERSION_MINOR(p.specVersion),
				XR_VERSION_PATCH(p.specVersion),
				p.layerVersion,
				p.description
			);
			print_extension_properties(p.layerName);
		}
	}

	if (layers.size() != 0) {
		for (const auto& e : layers) {
			bool found = false;
			for (XrApiLayerProperties layer : m_api_layer_properties) {
				if (strcmp(e, layer.layerName) == 0) {
					found = true;
					break;
				}
			}

			if (!found) {
				throw std::runtime_error{fmt::format("OpenXR API layer {} not found", e)};
			}
		}
	}

	xrEnumerateInstanceExtensionProperties(nullptr, 0, &size, nullptr);
	m_instance_extension_properties.clear();
	m_instance_extension_properties.resize(size, {XR_TYPE_EXTENSION_PROPERTIES});
	xrEnumerateInstanceExtensionProperties(nullptr, size, &size, m_instance_extension_properties.data());

	if (m_print_extensions) {
		tlog::info() << fmt::format("Instance extensions ({}):", m_instance_extension_properties.size());
		for (XrExtensionProperties extension : m_instance_extension_properties) {
			tlog::info() << fmt::format("\t{} (Version {})", extension.extensionName, extension.extensionVersion);
		}
	}

	auto has_extension = [&](const char* e) {
		for (XrExtensionProperties extension : m_instance_extension_properties) {
			if (strcmp(e, extension.extensionName) == 0) {
				return true;
			}
		}

		return false;
	};

	for (const auto& e : extensions) {
		if (!has_extension(e)) {
			throw std::runtime_error{fmt::format("Required OpenXR extension {} not found", e)};
		}
	}

	auto add_extension_if_supported = [&](const char* extension) {
		if (has_extension(extension)) {
			extensions.emplace_back(extension);
			return true;
		}

		return false;
	};

	if (add_extension_if_supported(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME)) {
		m_supports_composition_layer_depth = true;
	}

	if (add_extension_if_supported(XR_KHR_VISIBILITY_MASK_EXTENSION_NAME)) {
		m_supports_hidden_area_mask = true;
	}

	if (add_extension_if_supported(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME)) {
		m_supports_eye_tracking = true;
	}

	XrInstanceCreateInfo instance_create_info = {XR_TYPE_INSTANCE_CREATE_INFO};
	instance_create_info.applicationInfo = {};
	strncpy(instance_create_info.applicationInfo.applicationName, "Instant Neural Graphics Primitives v" NGP_VERSION, XR_MAX_APPLICATION_NAME_SIZE);
	instance_create_info.applicationInfo.applicationVersion = 1;
	strncpy(instance_create_info.applicationInfo.engineName, "Instant Neural Graphics Primitives v" NGP_VERSION, XR_MAX_ENGINE_NAME_SIZE);
	instance_create_info.applicationInfo.engineVersion = 1;
	instance_create_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
	instance_create_info.enabledExtensionCount = (uint32_t)extensions.size();
	instance_create_info.enabledExtensionNames = extensions.data();
	instance_create_info.enabledApiLayerCount = (uint32_t)layers.size();
	instance_create_info.enabledApiLayerNames = layers.data();

	if (XR_FAILED(xrCreateInstance(&instance_create_info, &m_instance))) {
		throw std::runtime_error{"Failed to create OpenXR instance"};
	}

	XR_CHECK_THROW(xrGetInstanceProperties(m_instance, &m_instance_properties));
	if (m_print_instance_properties) {
		tlog::info() << "Instance Properties";
		tlog::info() << fmt::format("\t        runtime name: '{}'", m_instance_properties.runtimeName);
		const auto& v = m_instance_properties.runtimeVersion;
		tlog::info() << fmt::format(
			"\t     runtime version: {}.{}.{}",
			XR_VERSION_MAJOR(v),
			XR_VERSION_MINOR(v),
			XR_VERSION_PATCH(v)
		);
	}
}

void OpenXRHMD::init_get_xr_system() {
	XrSystemGetInfo system_get_info = {XR_TYPE_SYSTEM_GET_INFO, nullptr, XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY};
	XR_CHECK_THROW(xrGetSystem(m_instance, &system_get_info, &m_system_id));

	XR_CHECK_THROW(xrGetSystemProperties(m_instance, m_system_id, &m_system_properties));
	if (m_print_system_properties) {
		tlog::info() << "System Properties";
		tlog::info() << fmt::format("\t                name: '{}'", m_system_properties.systemName);
		tlog::info() << fmt::format("\t            vendorId: {:#x}", m_system_properties.vendorId);
		tlog::info() << fmt::format("\t            systemId: {:#x}", m_system_properties.systemId);
		tlog::info() << fmt::format("\t     max layer count: {}", m_system_properties.graphicsProperties.maxLayerCount);
		tlog::info() << fmt::format("\t       max img width: {}", m_system_properties.graphicsProperties.maxSwapchainImageWidth);
		tlog::info() << fmt::format("\t      max img height: {}", m_system_properties.graphicsProperties.maxSwapchainImageHeight);
		tlog::info() << fmt::format("\torientation tracking: {}", m_system_properties.trackingProperties.orientationTracking ? "YES" : "NO");
		tlog::info() << fmt::format("\t   position tracking: {}", m_system_properties.trackingProperties.orientationTracking ? "YES" : "NO");
	}
}

void OpenXRHMD::init_configure_xr_views() {
	uint32_t size;
	XR_CHECK_THROW(xrEnumerateViewConfigurations(m_instance, m_system_id, 0, &size, nullptr));
	std::vector<XrViewConfigurationType> view_config_types(size);
	XR_CHECK_THROW(xrEnumerateViewConfigurations(m_instance, m_system_id, size, &size, view_config_types.data()));

	if (m_print_view_configuration_types) {
		tlog::info() << fmt::format("View Configuration Types ({}):", view_config_types.size());
		for (const auto& t : view_config_types) {
			tlog::info() << fmt::format("\t{}", XrEnumStr(t));
		}
	}

	// view configurations we support, in descending preference
	const std::vector<XrViewConfigurationType> preferred_view_config_types = {
		//XR_VIEW_CONFIGURATION_TYPE_PRIMARY_QUAD_VARJO,
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO
	};

	bool found = false;
	for (const auto& p : preferred_view_config_types) {
		for (const auto& t : view_config_types) {
			if (p == t) {
				found = true;
				m_view_configuration_type = t;
			}
		}
	}

	if (!found) {
		throw std::runtime_error{"Could not find a suitable OpenXR view configuration type"};
	}

	// get view configuration properties
	XR_CHECK_THROW(xrGetViewConfigurationProperties(m_instance, m_system_id, m_view_configuration_type, &m_view_configuration_properties));
	if (m_print_view_configuration_properties) {
		tlog::info() << "View Configuration Properties:";
		tlog::info() << fmt::format("\t         Type: {}", XrEnumStr(m_view_configuration_type));
		tlog::info() << fmt::format("\t         FOV Mutable: {}", m_view_configuration_properties.fovMutable ? "YES" : "NO");
	}

	// enumerate view configuration views
	XR_CHECK_THROW(xrEnumerateViewConfigurationViews(m_instance, m_system_id, m_view_configuration_type, 0, &size, nullptr));
	m_view_configuration_views.clear();
	m_view_configuration_views.resize(size, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
	XR_CHECK_THROW(xrEnumerateViewConfigurationViews(
		m_instance,
		m_system_id,
		m_view_configuration_type,
		size,
		&size,
		m_view_configuration_views.data()
	));

	if (m_print_view_configuration_view) {
		tlog::info() << "View Configuration Views, Width x Height x Samples";
		for (size_t i = 0; i < m_view_configuration_views.size(); ++i) {
			const auto& view = m_view_configuration_views[i];
			tlog::info() << fmt::format(
				"\tView {}\tRecommended: {}x{}x{}  Max: {}x{}x{}",
				i,
				view.recommendedImageRectWidth,
				view.recommendedImageRectHeight,
				view.recommendedSwapchainSampleCount,
				view.maxImageRectWidth,
				view.maxImageRectHeight,
				view.maxSwapchainSampleCount
			);
		}
	}
}

void OpenXRHMD::init_check_for_xr_blend_mode() {
	// enumerate environment blend modes
	uint32_t size;
	XR_CHECK_THROW(xrEnumerateEnvironmentBlendModes(m_instance, m_system_id, m_view_configuration_type, 0, &size, nullptr));
	std::vector<XrEnvironmentBlendMode> supported_blend_modes(size);
	XR_CHECK_THROW(xrEnumerateEnvironmentBlendModes(
		m_instance,
		m_system_id,
		m_view_configuration_type,
		size,
		&size,
		supported_blend_modes.data()
	));

	if (supported_blend_modes.empty()) {
		throw std::runtime_error{"No OpenXR environment blend modes found"};
	}

	std::sort(std::begin(supported_blend_modes), std::end(supported_blend_modes));
	if (m_print_environment_blend_modes) {
		tlog::info() << fmt::format("Environment Blend Modes ({}):", supported_blend_modes.size());
	}

	m_supported_environment_blend_modes.resize(supported_blend_modes.size());
	m_supported_environment_blend_modes_imgui_string.clear();
	for (size_t i = 0; i < supported_blend_modes.size(); ++i) {
		if (m_print_environment_blend_modes) {
			tlog::info() << fmt::format("\t{}", XrEnumStr(supported_blend_modes[i]));
		}

		auto b = (EEnvironmentBlendMode)supported_blend_modes[i];
		m_supported_environment_blend_modes[i] = b;

		auto b_str = to_string(b);
		std::copy(std::begin(b_str), std::end(b_str), std::back_inserter(m_supported_environment_blend_modes_imgui_string));
		m_supported_environment_blend_modes_imgui_string.emplace_back('\0');
	}

	m_supported_environment_blend_modes_imgui_string.emplace_back('\0');
	m_environment_blend_mode = m_supported_environment_blend_modes.front();
}

void OpenXRHMD::init_xr_actions() {
	// paths for left (0) and right (1) hands
	XR_CHECK_THROW(xrStringToPath(m_instance, "/user/hand/left", &m_hand_paths[0]));
	XR_CHECK_THROW(xrStringToPath(m_instance, "/user/hand/right", &m_hand_paths[1]));

	// create action set
	XrActionSetCreateInfo action_set_create_info{XR_TYPE_ACTION_SET_CREATE_INFO, nullptr, "actionset", "actionset", 0};
	XR_CHECK_THROW(xrCreateActionSet(m_instance, &action_set_create_info, &m_action_set));

	{
		XrActionCreateInfo action_create_info{
			XR_TYPE_ACTION_CREATE_INFO,
			nullptr,
			"hand_pose",
			XR_ACTION_TYPE_POSE_INPUT,
			(uint32_t)m_hand_paths.size(),
			m_hand_paths.data(),
			"Hand pose"
		};
		XR_CHECK_THROW(xrCreateAction(m_action_set, &action_create_info, &m_pose_action));
	}

	{
		XrActionCreateInfo action_create_info{
			XR_TYPE_ACTION_CREATE_INFO,
			nullptr,
			"thumbstick_left",
			XR_ACTION_TYPE_VECTOR2F_INPUT,
			0,
			nullptr,
			"Left thumbstick"
		};
		XR_CHECK_THROW(xrCreateAction(m_action_set, &action_create_info, &m_thumbstick_actions[0]));
	}

	{
		XrActionCreateInfo action_create_info{
			XR_TYPE_ACTION_CREATE_INFO,
			nullptr,
			"thumbstick_right",
			XR_ACTION_TYPE_VECTOR2F_INPUT,
			0,
			nullptr,
			"Right thumbstick"
		};
		XR_CHECK_THROW(xrCreateAction(m_action_set, &action_create_info, &m_thumbstick_actions[1]));
	}

	{
		XrActionCreateInfo action_create_info{
			XR_TYPE_ACTION_CREATE_INFO,
			nullptr,
			"press",
			XR_ACTION_TYPE_BOOLEAN_INPUT,
			(uint32_t)m_hand_paths.size(),
			m_hand_paths.data(),
			"Press"
		};
		XR_CHECK_THROW(xrCreateAction(m_action_set, &action_create_info, &m_press_action));
	}

	{
		XrActionCreateInfo action_create_info{
			XR_TYPE_ACTION_CREATE_INFO,
			nullptr,
			"grab",
			XR_ACTION_TYPE_FLOAT_INPUT,
			(uint32_t)m_hand_paths.size(),
			m_hand_paths.data(),
			"Grab"
		};
		XR_CHECK_THROW(xrCreateAction(m_action_set, &action_create_info, &m_grab_action));
	}

	auto create_binding = [&](XrAction action, const std::string& binding_path_str) {
		XrPath binding;
		XR_CHECK_THROW(xrStringToPath(m_instance, binding_path_str.c_str(), &binding));
		return XrActionSuggestedBinding{action, binding};
	};

	auto suggest_bindings = [&](const std::string& interaction_profile_path_str, const std::vector<XrActionSuggestedBinding>& bindings) {
		XrPath interaction_profile;
		XR_CHECK_THROW(xrStringToPath(m_instance, interaction_profile_path_str.c_str(), &interaction_profile));
		XrInteractionProfileSuggestedBinding suggested_binding{
			XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING,
			nullptr,
			interaction_profile,
			(uint32_t)bindings.size(),
			bindings.data()
		};
		XR_CHECK_THROW(xrSuggestInteractionProfileBindings(m_instance, &suggested_binding));
	};

	suggest_bindings("/interaction_profiles/khr/simple_controller", {
		create_binding(m_pose_action, "/user/hand/left/input/grip/pose"),
		create_binding(m_pose_action, "/user/hand/right/input/grip/pose"),
	});

	auto suggest_controller_bindings = [&](const std::string& xy, const std::string& press, const std::string& grab, const std::string& squeeze, const std::string& interaction_profile_path_str) {
		std::vector<XrActionSuggestedBinding> bindings = {
			create_binding(m_pose_action, "/user/hand/left/input/grip/pose"),
			create_binding(m_pose_action, "/user/hand/right/input/grip/pose"),
			create_binding(m_thumbstick_actions[0], std::string{"/user/hand/left/input/"} + xy),
			create_binding(m_thumbstick_actions[1], std::string{"/user/hand/right/input/"} + xy),
			create_binding(m_press_action, std::string{"/user/hand/left/input/"} + press),
			create_binding(m_press_action, std::string{"/user/hand/right/input/"} + press),
			create_binding(m_grab_action, std::string{"/user/hand/left/input/"} + grab),
			create_binding(m_grab_action, std::string{"/user/hand/right/input/"} + grab),
		};

		if (!squeeze.empty()) {
			bindings.emplace_back(create_binding(m_grab_action, std::string{"/user/hand/left/input/"} + squeeze));
			bindings.emplace_back(create_binding(m_grab_action, std::string{"/user/hand/right/input/"} + squeeze));
		}

		suggest_bindings(interaction_profile_path_str, bindings);
	};

	suggest_controller_bindings("trackpad",   "select/click",     "trackpad/click", "",                  "/interaction_profiles/google/daydream_controller");
	suggest_controller_bindings("trackpad",   "trackpad/click",   "trigger/click",  "squeeze/click",     "/interaction_profiles/htc/vive_controller");
	suggest_controller_bindings("thumbstick", "thumbstick/click", "trigger/value",  "squeeze/click",     "/interaction_profiles/microsoft/motion_controller");
	suggest_controller_bindings("trackpad",   "trackpad/click",   "trigger/click",  "",                  "/interaction_profiles/oculus/go_controller");
	suggest_controller_bindings("thumbstick", "thumbstick/click", "trigger/value",  "squeeze/value",     "/interaction_profiles/oculus/touch_controller");

	// Valve Index force squeeze is very sensitive and can cause unwanted grabbing. Only permit trigger-grabbing for now.
	suggest_controller_bindings("thumbstick", "thumbstick/click", "trigger/value",  ""/*squeeze/force*/, "/interaction_profiles/valve/index_controller");

	// Xbox controller (currently not functional)
	suggest_bindings("/interaction_profiles/microsoft/xbox_controller", {
		create_binding(m_thumbstick_actions[0], std::string{"/user/gamepad/input/thumbstick_left"}),
		create_binding(m_thumbstick_actions[1], std::string{"/user/gamepad/input/thumbstick_right"}),
	});
}

#if defined(XR_USE_PLATFORM_WIN32)
void OpenXRHMD::init_open_gl(HDC hdc, HGLRC hglrc) {
#elif defined(XR_USE_PLATFORM_XLIB)
void OpenXRHMD::init_open_gl(Display* xDisplay, uint32_t visualid, GLXFBConfig glxFBConfig, GLXDrawable glxDrawable, GLXContext glxContext) {
#elif defined(XR_USE_PLATFORM_WAYLAND)
void OpenXRHMD::init_open_gl(wl_display* display) {
#endif
	// GL graphics requirements
	PFN_xrGetOpenGLGraphicsRequirementsKHR xrGetOpenGLGraphicsRequirementsKHR = nullptr;
	XR_CHECK_THROW(xrGetInstanceProcAddr(
		m_instance,
		"xrGetOpenGLGraphicsRequirementsKHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&xrGetOpenGLGraphicsRequirementsKHR)
	));

	XrGraphicsRequirementsOpenGLKHR graphics_requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR};
	xrGetOpenGLGraphicsRequirementsKHR(m_instance, m_system_id, &graphics_requirements);
	XrVersion min_version = graphics_requirements.minApiVersionSupported;
	GLint major = 0;
	GLint minor = 0;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);
	const XrVersion have_version = XR_MAKE_VERSION(major, minor, 0);

	if (have_version < min_version) {
		tlog::info() << fmt::format(
			"Required OpenGL version: {}.{}, found OpenGL version: {}.{}",
			XR_VERSION_MAJOR(min_version),
			XR_VERSION_MINOR(min_version),
			major,
			minor
		);

		throw std::runtime_error{"Insufficient graphics API support"};
	}

#if defined(XR_USE_PLATFORM_WIN32)
	m_graphics_binding.hDC = hdc;
	m_graphics_binding.hGLRC = hglrc;
#elif defined(XR_USE_PLATFORM_XLIB)
	m_graphics_binding.xDisplay = xDisplay;
	m_graphics_binding.visualid = visualid;
	m_graphics_binding.glxFBConfig = glxFBConfig;
	m_graphics_binding.glxDrawable = glxDrawable;
	m_graphics_binding.glxContext = glxContext;
#elif defined(XR_USE_PLATFORM_WAYLAND)
	m_graphics_binding.display = display;
#endif
}

void OpenXRHMD::init_xr_session() {
	// create session
	XrSessionCreateInfo create_info{
		XR_TYPE_SESSION_CREATE_INFO,
		reinterpret_cast<const XrBaseInStructure*>(&m_graphics_binding),
		0,
		m_system_id
	};

	XR_CHECK_THROW(xrCreateSession(m_instance, &create_info, &m_session));

	// tlog::info() << fmt::format("Created session {}", fmt::ptr(m_session));
}

void OpenXRHMD::init_xr_spaces() {
	// reference space
	uint32_t size;
	XR_CHECK_THROW(xrEnumerateReferenceSpaces(m_session, 0, &size, nullptr));
	m_reference_spaces.clear();
	m_reference_spaces.resize(size);
	XR_CHECK_THROW(xrEnumerateReferenceSpaces(m_session, size, &size, m_reference_spaces.data()));

	if (m_print_reference_spaces) {
		tlog::info() << fmt::format("Reference spaces ({}):", m_reference_spaces.size());
		for (const auto& r : m_reference_spaces) {
			tlog::info() << fmt::format("\t{}", XrEnumStr(r));
		}
	}

	XrReferenceSpaceCreateInfo reference_space_create_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
	reference_space_create_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
	reference_space_create_info.poseInReferenceSpace = XrPosef{};
	reference_space_create_info.poseInReferenceSpace.orientation.w = 1.0f;
	XR_CHECK_THROW(xrCreateReferenceSpace(m_session, &reference_space_create_info, &m_space));
	XR_CHECK_THROW(xrGetReferenceSpaceBoundsRect(m_session, reference_space_create_info.referenceSpaceType, &m_bounds));

	if (m_print_reference_spaces) {
		tlog::info() << fmt::format("Using reference space {}", XrEnumStr(reference_space_create_info.referenceSpaceType));
		tlog::info() << fmt::format("Reference space boundaries: {} x {}", m_bounds.width, m_bounds.height);
	}

	// action space
	XrActionSpaceCreateInfo action_space_create_info{XR_TYPE_ACTION_SPACE_CREATE_INFO};
	action_space_create_info.action = m_pose_action;
	action_space_create_info.poseInActionSpace.orientation.w = 1.0f;
	action_space_create_info.subactionPath = m_hand_paths[0];
	XR_CHECK_THROW(xrCreateActionSpace(m_session, &action_space_create_info, &m_hand_spaces[0]));
	action_space_create_info.subactionPath = m_hand_paths[1];
	XR_CHECK_THROW(xrCreateActionSpace(m_session, &action_space_create_info, &m_hand_spaces[1]));

	// attach action set
	XrSessionActionSetsAttachInfo attach_info{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
	attach_info.countActionSets = 1;
	attach_info.actionSets = &m_action_set;
	XR_CHECK_THROW(xrAttachSessionActionSets(m_session, &attach_info));
}

void OpenXRHMD::init_xr_swapchain_open_gl() {
	// swap chains
	uint32_t size;
	XR_CHECK_THROW(xrEnumerateSwapchainFormats(m_session, 0, &size, nullptr));
	std::vector<int64_t> swapchain_formats(size);
	XR_CHECK_THROW(xrEnumerateSwapchainFormats(m_session, size, &size, swapchain_formats.data()));

	if (m_print_available_swapchain_formats) {
		tlog::info() << fmt::format("Swapchain formats ({}):", swapchain_formats.size());
		for (const auto& f : swapchain_formats) {
			tlog::info() << fmt::format("\t{:#x}", f);
		}
	}

	auto find_compatible_swapchain_format = [&](const std::vector<int64_t>& candidates) {
		for (auto format : candidates) {
			if (std::find(std::begin(swapchain_formats), std::end(swapchain_formats), format) != std::end(swapchain_formats)) {
				return format;
			}
		}

		throw std::runtime_error{"No compatible OpenXR swapchain format found"};
	};

	m_swapchain_rgba_format = find_compatible_swapchain_format({
		GL_SRGB8_ALPHA8,
		GL_SRGB8,
		GL_RGBA8,
	});

	if (m_supports_composition_layer_depth) {
		m_swapchain_depth_format = find_compatible_swapchain_format({
			GL_DEPTH_COMPONENT32F,
			GL_DEPTH_COMPONENT24,
			GL_DEPTH_COMPONENT16,
		});
	}

	// tlog::info() << fmt::format("Chosen swapchain format: {:#x}", m_swapchain_rgba_format);
	for (const auto& vcv : m_view_configuration_views) {
		XrSwapchainCreateInfo rgba_swapchain_create_info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
		rgba_swapchain_create_info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
		rgba_swapchain_create_info.format = m_swapchain_rgba_format;
		rgba_swapchain_create_info.sampleCount = 1;
		rgba_swapchain_create_info.width = vcv.recommendedImageRectWidth;
		rgba_swapchain_create_info.height = vcv.recommendedImageRectHeight;
		rgba_swapchain_create_info.faceCount = 1;
		rgba_swapchain_create_info.arraySize = 1;
		rgba_swapchain_create_info.mipCount = 1;

		XrSwapchainCreateInfo depth_swapchain_create_info = rgba_swapchain_create_info;
		depth_swapchain_create_info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		depth_swapchain_create_info.format = m_swapchain_depth_format;

		m_swapchains.emplace_back(rgba_swapchain_create_info, depth_swapchain_create_info, m_session, m_instance);
	}
}

void OpenXRHMD::init_open_gl_shaders() {
	// Hidden area mask program
	{
		static const char* shader_vert = R"(#version 140
			in vec2 pos;
			uniform mat4 project;
			void main() {
				vec4 pos = project * vec4(pos, -1.0, 1.0);
				pos.xyz /= pos.w;
				pos.y *= -1.0;
				gl_Position = pos;
			})";

		static const char* shader_frag = R"(#version 140
			out vec4 frag_color;
			void main() {
				frag_color = vec4(0.0, 0.0, 0.0, 1.0);
			})";

		GLuint vert = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vert, 1, &shader_vert, NULL);
		glCompileShader(vert);
		check_shader(vert, "OpenXR hidden area mask vertex shader", false);

		GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(frag, 1, &shader_frag, NULL);
		glCompileShader(frag);
		check_shader(frag, "OpenXR hidden area mask fragment shader", false);

		m_hidden_area_mask_program = glCreateProgram();
		glAttachShader(m_hidden_area_mask_program, vert);
		glAttachShader(m_hidden_area_mask_program, frag);
		glLinkProgram(m_hidden_area_mask_program);
		check_shader(m_hidden_area_mask_program, "OpenXR hidden area mask shader program", true);

		glDeleteShader(vert);
		glDeleteShader(frag);
	}
}

void OpenXRHMD::session_state_change(XrSessionState state, EControlFlow& flow) {
	//tlog::info() << fmt::format("New session state {}", XrEnumStr(state));
	switch (state) {
		case XR_SESSION_STATE_READY: {
			XrSessionBeginInfo sessionBeginInfo {XR_TYPE_SESSION_BEGIN_INFO};
			sessionBeginInfo.primaryViewConfigurationType = m_view_configuration_type;
			XR_CHECK_THROW(xrBeginSession(m_session, &sessionBeginInfo));
			break;
		}
		case XR_SESSION_STATE_STOPPING: {
			XR_CHECK_THROW(xrEndSession(m_session));
			break;
		}
		case XR_SESSION_STATE_EXITING: {
			flow = EControlFlow::Quit;
			break;
		}
		case XR_SESSION_STATE_LOSS_PENDING: {
			flow = EControlFlow::Restart;
			break;
		}
		default: {
			break;
		}
	}
}

OpenXRHMD::EControlFlow OpenXRHMD::poll_events() {
	bool more = true;
	EControlFlow flow = EControlFlow::Continue;
	while (more) {
		// poll events
		XrEventDataBuffer event {XR_TYPE_EVENT_DATA_BUFFER, nullptr};
		XrResult result = xrPollEvent(m_instance, &event);

		if (XR_FAILED(result)) {
			tlog::error() << "xrPollEvent failed";
		} else if (XR_SUCCESS == result) {
			switch (event.type) {
				case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
					const XrEventDataSessionStateChanged& e = *reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
					//tlog::info() << "Session state change";
					//tlog::info() << fmt::format("\t from {}\t   to {}", XrEnumStr(m_session_state), XrEnumStr(e.state));
					//tlog::info() << fmt::format("\t session {}, time {}", fmt::ptr(e.session), e.time);
					m_session_state = e.state;
					session_state_change(e.state, flow);
					break;
				}

				case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
					flow = EControlFlow::Restart;
					break;
				}

				case XR_TYPE_EVENT_DATA_VISIBILITY_MASK_CHANGED_KHR: {
					m_hidden_area_masks.clear();
					break;
				}

				case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED: {
					break; // Can ignore
				}

				default: {
					tlog::info() << fmt::format("Unhandled event type {}", XrEnumStr(event.type));
					break;
				}
			}
		} else if (XR_EVENT_UNAVAILABLE == result) {
			more = false;
		}
	}
	return flow;
}

__global__ void read_hidden_area_mask_kernel(const Vector2i resolution, cudaSurfaceObject_t surface, uint8_t* __restrict__ mask) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;
	surf2Dread(&mask[idx], surface, x, y);
}

std::shared_ptr<Buffer2D<uint8_t>> OpenXRHMD::rasterize_hidden_area_mask(uint32_t view_index, const XrCompositionLayerProjectionView& view) {
	if (!m_supports_hidden_area_mask) {
		return {};
	}

	PFN_xrGetVisibilityMaskKHR xrGetVisibilityMaskKHR = nullptr;
	XR_CHECK_THROW(xrGetInstanceProcAddr(
		m_instance,
		"xrGetVisibilityMaskKHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&xrGetVisibilityMaskKHR)
	));

	XrVisibilityMaskKHR visibility_mask{XR_TYPE_VISIBILITY_MASK_KHR};
	XR_CHECK_THROW(xrGetVisibilityMaskKHR(m_session, m_view_configuration_type, view_index, XR_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH_KHR, &visibility_mask));

	if (visibility_mask.vertexCountOutput == 0 || visibility_mask.indexCountOutput == 0) {
		return nullptr;
	}

	std::vector<XrVector2f> vertices(visibility_mask.vertexCountOutput);
	std::vector<uint32_t> indices(visibility_mask.indexCountOutput);

	visibility_mask.vertices = vertices.data();
	visibility_mask.indices = indices.data();

	visibility_mask.vertexCapacityInput = visibility_mask.vertexCountOutput;
	visibility_mask.indexCapacityInput = visibility_mask.indexCountOutput;

	XR_CHECK_THROW(xrGetVisibilityMaskKHR(m_session, m_view_configuration_type, view_index, XR_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH_KHR, &visibility_mask));

	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	Vector2i size = {view.subImage.imageRect.extent.width, view.subImage.imageRect.extent.height};

	bool tex = glIsEnabled(GL_TEXTURE_2D);
	bool depth = glIsEnabled(GL_DEPTH_TEST);
	bool cull = glIsEnabled(GL_CULL_FACE);
	GLint previous_texture_id;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous_texture_id);

	if (!tex) glEnable(GL_TEXTURE_2D);
	if (depth) glDisable(GL_DEPTH_TEST);
	if (cull) glDisable(GL_CULL_FACE);

	// Generate texture to hold hidden area mask. Single channel, value of 1 means visible and 0 means masked away
	ngp::GLTexture mask_texture;
	mask_texture.resize(size, 1, true);
	glBindTexture(GL_TEXTURE_2D, mask_texture.texture());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	GLuint framebuffer = 0;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mask_texture.texture(), 0);

	GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, draw_buffers);

	glViewport(0, 0, size.x(), size.y());

	// Draw hidden area mask
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint vertex_buffer;
	glGenBuffers(1, &vertex_buffer);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(XrVector2f) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

	GLuint index_buffer;
	glGenBuffers(1, &index_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * indices.size(), indices.data(), GL_STATIC_DRAW);

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glUseProgram(m_hidden_area_mask_program);

	XrMatrix4x4f proj;
	XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_OPENGL, view.fov, 1.0f / 128.0f, 128.0f);

	GLuint project_id = glGetUniformLocation(m_hidden_area_mask_program, "project");
	glUniformMatrix4fv(project_id, 1, GL_FALSE, &proj.m[0]);

	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, (void*)0);
	glFinish();

	glDisableVertexAttribArray(0);
	glDeleteBuffers(1, &vertex_buffer);
	glDeleteBuffers(1, &index_buffer);
	glDeleteVertexArrays(1, &vao);
	glDeleteFramebuffers(1, &framebuffer);

	glBindVertexArray(0);
	glUseProgram(0);

	// restore old state
	if (!tex) glDisable(GL_TEXTURE_2D);
	if (depth) glEnable(GL_DEPTH_TEST);
	if (cull) glEnable(GL_CULL_FACE);
	glBindTexture(GL_TEXTURE_2D, previous_texture_id);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	std::shared_ptr<Buffer2D<uint8_t>> mask = std::make_shared<Buffer2D<uint8_t>>(size);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)size.x(), threads.x), div_round_up((uint32_t)size.y(), threads.y), 1 };

	read_hidden_area_mask_kernel<<<blocks, threads>>>(size, mask_texture.surface(), mask->data());
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	return mask;
}

Matrix<float, 3, 4> convert_xr_matrix_to_eigen(const XrMatrix4x4f& m) {
	Matrix<float, 3, 4> out;

	for (size_t i = 0; i < 3; ++i) {
		for (size_t j = 0; j < 4; ++j) {
			out(i, j) = m.m[i + j * 4];
		}
	}

	// Flip Y and Z axes to match NGP conventions
	out(0, 1) *= -1.f;
	out(1, 0) *= -1.f;

	out(0, 2) *= -1.f;
	out(2, 0) *= -1.f;

	out(1, 3) *= -1.f;
	out(2, 3) *= -1.f;

	return out;
}

Matrix<float, 3, 4> convert_xr_pose_to_eigen(const XrPosef& pose) {
	XrMatrix4x4f matrix;
	XrVector3f unit_scale{1.0f, 1.0f, 1.0f};
	XrMatrix4x4f_CreateTranslationRotationScale(&matrix, &pose.position, &pose.orientation, &unit_scale);
	return convert_xr_matrix_to_eigen(matrix);
}

OpenXRHMD::FrameInfoPtr OpenXRHMD::begin_frame() {
	XrFrameWaitInfo frame_wait_info{XR_TYPE_FRAME_WAIT_INFO};
	XR_CHECK_THROW(xrWaitFrame(m_session, &frame_wait_info, &m_frame_state));

	XrFrameBeginInfo frame_begin_info{XR_TYPE_FRAME_BEGIN_INFO};
	XR_CHECK_THROW(xrBeginFrame(m_session, &frame_begin_info));

	if (!m_frame_state.shouldRender) {
		return std::make_shared<FrameInfo>();
	}

	uint32_t num_views = (uint32_t)m_swapchains.size();
	// TODO assert m_view_configuration_views.size() == m_swapchains.size()

	// locate views
	std::vector<XrView> views(num_views, {XR_TYPE_VIEW});

	XrViewState viewState{XR_TYPE_VIEW_STATE};

	XrViewLocateInfo view_locate_info{XR_TYPE_VIEW_LOCATE_INFO};
	view_locate_info.viewConfigurationType = m_view_configuration_type;
	view_locate_info.displayTime = m_frame_state.predictedDisplayTime;
	view_locate_info.space = m_space;

	XR_CHECK_THROW(xrLocateViews(m_session, &view_locate_info, &viewState, uint32_t(views.size()), &num_views, views.data()));

	if (!(viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) || !(viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)) {
		return std::make_shared<FrameInfo>();
	}

	m_hidden_area_masks.resize(num_views);

	// Fill frame information
	if (!m_previous_frame_info) {
		m_previous_frame_info = std::make_shared<FrameInfo>();
	}

	FrameInfoPtr frame_info = std::make_shared<FrameInfo>(*m_previous_frame_info);
	frame_info->views.resize(m_swapchains.size());

	for (size_t i = 0; i < m_swapchains.size(); ++i) {
		const auto& sc = m_swapchains[i];

		XrSwapchainImageAcquireInfo image_acquire_info{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
		XrSwapchainImageWaitInfo image_wait_info{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, nullptr, XR_INFINITE_DURATION};

		uint32_t image_index;
		XR_CHECK_THROW(xrAcquireSwapchainImage(sc.handle, &image_acquire_info, &image_index));
		XR_CHECK_THROW(xrWaitSwapchainImage(sc.handle, &image_wait_info));

		FrameInfo::View& v = frame_info->views[i];
		v.framebuffer = sc.framebuffers_gl[image_index];
		v.view.pose = views[i].pose;
		v.view.fov = views[i].fov;
		v.view.subImage.imageRect = XrRect2Di{{0, 0}, {sc.width, sc.height}};
		v.view.subImage.imageArrayIndex = 0;
		v.view.subImage.swapchain = sc.handle;

		glBindFramebuffer(GL_FRAMEBUFFER, sc.framebuffers_gl[image_index]);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sc.images_gl.at(image_index).image, 0);

		if (sc.depth_handle != XR_NULL_HANDLE) {
			uint32_t depth_image_index;
			XR_CHECK_THROW(xrAcquireSwapchainImage(sc.depth_handle, &image_acquire_info, &depth_image_index));
			XR_CHECK_THROW(xrWaitSwapchainImage(sc.depth_handle, &image_wait_info));

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sc.depth_images_gl.at(depth_image_index).image, 0);

			v.depth_info.subImage.imageRect = XrRect2Di{{0, 0}, {sc.width, sc.height}};
			v.depth_info.subImage.imageArrayIndex = 0;
			v.depth_info.subImage.swapchain = sc.depth_handle;
			v.depth_info.minDepth = 0.0f;
			v.depth_info.maxDepth = 1.0f;

			// To be overwritten with actual near and far planes by end_frame
			v.depth_info.nearZ = 1.0f / 128.0f;
			v.depth_info.farZ = 128.0f;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		if (!m_hidden_area_masks.at(i)) {
			m_hidden_area_masks.at(i) = rasterize_hidden_area_mask(i, v.view);
		}

		v.hidden_area_mask = m_hidden_area_masks.at(i);
		v.pose = convert_xr_pose_to_eigen(v.view.pose);
	}

	XrActiveActionSet active_action_set{m_action_set, XR_NULL_PATH};
	XrActionsSyncInfo sync_info{XR_TYPE_ACTIONS_SYNC_INFO};
	sync_info.countActiveActionSets = 1;
	sync_info.activeActionSets = &active_action_set;
	XR_CHECK_THROW(xrSyncActions(m_session, &sync_info));

	for (size_t i = 0; i < 2; ++i) {
		// Hand pose
		{
			XrActionStatePose pose_state{XR_TYPE_ACTION_STATE_POSE};
			XrActionStateGetInfo get_info{XR_TYPE_ACTION_STATE_GET_INFO};
			get_info.action = m_pose_action;
			get_info.subactionPath = m_hand_paths[i];
			XR_CHECK_THROW(xrGetActionStatePose(m_session, &get_info, &pose_state));

			frame_info->hands[i].pose_active = pose_state.isActive;
			if (frame_info->hands[i].pose_active) {
				XrSpaceLocation space_location{XR_TYPE_SPACE_LOCATION};
				XR_CHECK_THROW(xrLocateSpace(m_hand_spaces[i], m_space, m_frame_state.predictedDisplayTime, &space_location));
				frame_info->hands[i].pose = convert_xr_pose_to_eigen(space_location.pose);
			}
		}

		// Stick
		{
			XrActionStateVector2f thumbstick_state{XR_TYPE_ACTION_STATE_VECTOR2F};
			XrActionStateGetInfo get_info{XR_TYPE_ACTION_STATE_GET_INFO};
			get_info.action = m_thumbstick_actions[i];
			XR_CHECK_THROW(xrGetActionStateVector2f(m_session, &get_info, &thumbstick_state));

			if (thumbstick_state.isActive) {
				frame_info->hands[i].thumbstick.x() = thumbstick_state.currentState.x;
				frame_info->hands[i].thumbstick.y() = thumbstick_state.currentState.y;
			} else {
				frame_info->hands[i].thumbstick = Vector2f::Zero();
			}
		}

		// Press
		{
			XrActionStateBoolean press_state{XR_TYPE_ACTION_STATE_BOOLEAN};
			XrActionStateGetInfo get_info{XR_TYPE_ACTION_STATE_GET_INFO};
			get_info.action = m_press_action;
			get_info.subactionPath = m_hand_paths[i];
			XR_CHECK_THROW(xrGetActionStateBoolean(m_session, &get_info, &press_state));

			if (press_state.isActive) {
				frame_info->hands[i].pressing = press_state.currentState;
			} else {
				frame_info->hands[i].pressing = 0.0f;
			}
		}

		// Grab
		{
			XrActionStateFloat grab_state{XR_TYPE_ACTION_STATE_FLOAT};
			XrActionStateGetInfo get_info{XR_TYPE_ACTION_STATE_GET_INFO};
			get_info.action = m_grab_action;
			get_info.subactionPath = m_hand_paths[i];
			XR_CHECK_THROW(xrGetActionStateFloat(m_session, &get_info, &grab_state));

			if (grab_state.isActive) {
				frame_info->hands[i].grab_strength = grab_state.currentState;
			} else {
				frame_info->hands[i].grab_strength = 0.0f;
			}

			bool was_grabbing = frame_info->hands[i].grabbing;
			frame_info->hands[i].grabbing = frame_info->hands[i].grab_strength >= 0.5f;

			if (frame_info->hands[i].grabbing) {
				frame_info->hands[i].prev_grab_pos = was_grabbing ? frame_info->hands[i].grab_pos : frame_info->hands[i].pose.col(3);
				frame_info->hands[i].grab_pos = frame_info->hands[i].pose.col(3);
			}
		}
	}

	m_previous_frame_info = frame_info;
	return frame_info;
}

void OpenXRHMD::end_frame(FrameInfoPtr frame_info, float znear, float zfar) {
	std::vector<XrCompositionLayerProjectionView> layer_projection_views(frame_info->views.size());
	for (size_t i = 0; i < layer_projection_views.size(); ++i) {
		auto& v = frame_info->views[i];
		auto& view = layer_projection_views[i];

		view = v.view;

		// release swapchain image
		XrSwapchainImageReleaseInfo release_info{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
		XR_CHECK_THROW(xrReleaseSwapchainImage(v.view.subImage.swapchain, &release_info));

		if (v.depth_info.subImage.swapchain != XR_NULL_HANDLE) {
			XR_CHECK_THROW(xrReleaseSwapchainImage(v.depth_info.subImage.swapchain, &release_info));
			v.depth_info.nearZ = znear;
			v.depth_info.farZ = zfar;
			// The following line being commented means that our provided depth buffer
			// _isn't_ actually passed to the runtime for reprojection. So far,
			// experimentation has shown that runtimes do a better job at reprojection
			// without getting a depth buffer from us, so we leave it disabled for now.
			// view.next = &v.depth_info;
		}
	}

	XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
	layer.space = m_space;
	if (m_environment_blend_mode != EEnvironmentBlendMode::Opaque) {
		layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
	}

	layer.viewCount = uint32_t(layer_projection_views.size());
	layer.views = layer_projection_views.data();

	std::vector<XrCompositionLayerBaseHeader*> layers;
	if (layer.viewCount) {
		layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));
	}

	XrFrameEndInfo frame_end_info{XR_TYPE_FRAME_END_INFO};
	frame_end_info.displayTime = m_frame_state.predictedDisplayTime;
	frame_end_info.environmentBlendMode = (XrEnvironmentBlendMode)m_environment_blend_mode;
	frame_end_info.layerCount = (uint32_t)layers.size();
	frame_end_info.layers = layers.data();
	XR_CHECK_THROW(xrEndFrame(m_session, &frame_end_info));
}

NGP_NAMESPACE_END

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
