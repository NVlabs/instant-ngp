/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   dlss.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/dlss.h>

#include <tiny-cuda-nn/common.h>

#include <filesystem/path.h>

#if !defined(NGP_VULKAN) || !defined(NGP_GUI)
static_assert(false, "DLSS can only be compiled when both Vulkan and GUI support is enabled.")
#endif

#ifdef _WIN32
#  include <GL/gl3w.h>
#else
#  include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>

#ifdef _WIN32
#  include <vulkan/vulkan_win32.h>
#endif

// NGX's macro `NVSDK_NGX_FAILED` results in a change of sign, which does not affect correctness.
// Thus, suppress the corresponding warning.
#ifdef __NVCC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress = integer_sign_change
#  else
#    pragma diag_suppress = integer_sign_change
#  endif
#endif
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <codecvt>
#include <locale>

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

extern std::atomic<size_t> g_total_n_bytes_allocated;

/// Checks the result of a vkXXXXXX call and throws an error on failure
#define VK_CHECK_THROW(x)                                                       \
	do {                                                                        \
		VkResult result = x;                                                    \
		if (result != VK_SUCCESS)                                               \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed"));  \
	} while(0)

std::string ngx_error_string(NVSDK_NGX_Result result) {
	std::wstring wstr = GetNGXResultAsString(result);
	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
	return converter.to_bytes(wstr);
};

/// Checks the result of a NVSDK_NGX_XXXXXX call and throws an error on failure
#define NGX_CHECK_THROW(x)                                                                                            \
	do {                                                                                                              \
		NVSDK_NGX_Result result = x;                                                                                  \
		if (NVSDK_NGX_FAILED(result))                                                                                 \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + ngx_error_string(result)); \
	} while(0)

static VkInstance vk_instance = VK_NULL_HANDLE;
static VkDebugUtilsMessengerEXT vk_debug_messenger = VK_NULL_HANDLE;
static VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
static VkDevice vk_device = VK_NULL_HANDLE;
static VkQueue vk_queue = VK_NULL_HANDLE;
static VkCommandPool vk_command_pool = VK_NULL_HANDLE;
static VkCommandBuffer vk_command_buffer = VK_NULL_HANDLE;

static bool ngx_initialized = false;
static NVSDK_NGX_Parameter* ngx_parameters = nullptr;

uint32_t vk_find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &mem_properties);

	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
		if (type_filter & (1 << i) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error{"Failed to find suitable memory type."};
}

static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
	VkDebugUtilsMessageTypeFlagsEXT message_type,
	const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
	void* user_data
) {
	if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
		tlog::warning() << "Vulkan error: " << callback_data->pMessage;
	} else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		tlog::warning() << "Vulkan: " << callback_data->pMessage;
	} else {
		tlog::info() << "Vulkan: " << callback_data->pMessage;
	}

	return VK_FALSE;
}

void vulkan_and_ngx_init() {
	static bool already_initialized = false;

	if (already_initialized) {
		return;
	}

	already_initialized = true;

	if (!glfwVulkanSupported()) {
		throw std::runtime_error{"!glfwVulkanSupported()"};
	}

	// -------------------------------
	// Vulkan Instance
	// -------------------------------
	VkApplicationInfo app_info{};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = "NGP";
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName = "No engine";
	app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo instance_create_info = {};
	instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instance_create_info.pApplicationInfo = &app_info;

	uint32_t available_layer_count;
	vkEnumerateInstanceLayerProperties(&available_layer_count, nullptr);

	std::vector<VkLayerProperties> available_layers(available_layer_count);
	vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers.data());

	std::vector<const char*> layers;
	auto try_add_layer = [&](const char* layer) {
		for (const auto& props : available_layers) {
			if (strcmp(layer, props.layerName) == 0) {
				layers.emplace_back(layer);
				return true;
			}
		}

		return false;
	};

	bool validation_layer_enabled = try_add_layer("VK_LAYER_KHRONOS_validation");
	if (!validation_layer_enabled) {
		tlog::warning() << "Vulkan validation layer is not available. Vulkan errors will be difficult to diagnose.";
	}

	instance_create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
	instance_create_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

	std::vector<const char*> instance_extensions;
	std::vector<const char*> device_extensions;

	uint32_t n_ngx_instance_extensions = 0;
	const char** ngx_instance_extensions;

	uint32_t n_ngx_device_extensions = 0;
	const char** ngx_device_extensions;

	NVSDK_NGX_VULKAN_RequiredExtensions(&n_ngx_instance_extensions, &ngx_instance_extensions, &n_ngx_device_extensions, &ngx_device_extensions);

	for (uint32_t i = 0; i < n_ngx_instance_extensions; ++i) {
		instance_extensions.emplace_back(ngx_instance_extensions[i]);
	}

	instance_extensions.emplace_back(VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME);
	instance_extensions.emplace_back(VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);
	instance_extensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	instance_extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

	if (validation_layer_enabled) {
		instance_extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	for (uint32_t i = 0; i < n_ngx_device_extensions; ++i) {
		device_extensions.emplace_back(ngx_device_extensions[i]);
	}

	device_extensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef _WIN32
	device_extensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
	device_extensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
	device_extensions.emplace_back(VK_KHR_DEVICE_GROUP_EXTENSION_NAME);

	instance_create_info.enabledExtensionCount = (uint32_t)instance_extensions.size();
	instance_create_info.ppEnabledExtensionNames = instance_extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info = {};
	debug_messenger_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debug_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debug_messenger_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	debug_messenger_create_info.pfnUserCallback = vk_debug_callback;
	debug_messenger_create_info.pUserData = nullptr;

	if (validation_layer_enabled) {
		instance_create_info.pNext = &debug_messenger_create_info;
	}

	VK_CHECK_THROW(vkCreateInstance(&instance_create_info, nullptr, &vk_instance));

	if (validation_layer_enabled) {
		auto CreateDebugUtilsMessengerEXT = [](VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func != nullptr) {
				return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
			} else {
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
		};

		if (CreateDebugUtilsMessengerEXT(vk_instance, &debug_messenger_create_info, nullptr, &vk_debug_messenger) != VK_SUCCESS) {
			tlog::warning() << "Vulkan: could not initialize debug messenger.";
		}
	}

	// -------------------------------
	// Vulkan Physical Device
	// -------------------------------
	uint32_t n_devices = 0;
	vkEnumeratePhysicalDevices(vk_instance, &n_devices, nullptr);

	if (n_devices == 0) {
		throw std::runtime_error{"Failed to find GPUs with Vulkan support."};
	}

	std::vector<VkPhysicalDevice> devices(n_devices);
	vkEnumeratePhysicalDevices(vk_instance, &n_devices, devices.data());

	struct QueueFamilyIndices {
		int graphics_family = -1;
		int compute_family = -1;
		int transfer_family = -1;
		int all_family = -1;
	};

	auto find_queue_families = [](VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

		std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

		int i = 0;
		for (const auto& queue_family : queue_families) {
			if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphics_family = i;
			}

			if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
				indices.compute_family = i;
			}

			if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) {
				indices.transfer_family = i;
			}

			if ((queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) && (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT)) {
				indices.all_family = i;
			}

			i++;
		}

		return indices;
	};

	cudaDeviceProp cuda_device_prop;
	CUDA_CHECK_THROW(cudaGetDeviceProperties(&cuda_device_prop, tcnn::cuda_device()));

	auto is_same_as_cuda_device = [&](VkPhysicalDevice device) {
		VkPhysicalDeviceIDProperties physical_device_id_properties = {};
		physical_device_id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
		physical_device_id_properties.pNext = NULL;

		VkPhysicalDeviceProperties2 physical_device_properties = {};
		physical_device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		physical_device_properties.pNext = &physical_device_id_properties;

		vkGetPhysicalDeviceProperties2(device, &physical_device_properties);

		return !memcmp(&cuda_device_prop.uuid, physical_device_id_properties.deviceUUID, VK_UUID_SIZE) && find_queue_families(device).all_family >= 0;
	};

	uint32_t device_id = 0;
	for (uint32_t i = 0; i < n_devices; ++i) {
		if (is_same_as_cuda_device(devices[i])) {
			vk_physical_device = devices[i];
			device_id = i;
			break;
		}
	}

	if (vk_physical_device == VK_NULL_HANDLE) {
		throw std::runtime_error{"Failed to find Vulkan device corresponding to CUDA device."};
	}

	// -------------------------------
	// Vulkan Logical Device
	// -------------------------------
	VkPhysicalDeviceProperties physical_device_properties;
	vkGetPhysicalDeviceProperties(vk_physical_device, &physical_device_properties);

	QueueFamilyIndices indices = find_queue_families(vk_physical_device);

	VkDeviceQueueCreateInfo queue_create_info{};
	queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queue_create_info.queueFamilyIndex = indices.all_family;
	queue_create_info.queueCount = 1;

	float queue_priority = 1.0f;
	queue_create_info.pQueuePriorities = &queue_priority;

	VkPhysicalDeviceFeatures device_features = {};
	device_features.shaderStorageImageWriteWithoutFormat = true;

	VkDeviceCreateInfo device_create_info = {};
	device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_create_info.pQueueCreateInfos = &queue_create_info;
	device_create_info.queueCreateInfoCount = 1;
	device_create_info.pEnabledFeatures = &device_features;
	device_create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
	device_create_info.ppEnabledExtensionNames = device_extensions.data();
	device_create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
	device_create_info.ppEnabledLayerNames = layers.data();

#ifdef VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME
	VkPhysicalDeviceBufferDeviceAddressFeaturesEXT buffer_device_address_feature = {};
	buffer_device_address_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT;
	buffer_device_address_feature.bufferDeviceAddress = VK_TRUE;
	device_create_info.pNext = &buffer_device_address_feature;
#else
	throw std::runtime_error{"Buffer device address extension not available."};
#endif

	VK_CHECK_THROW(vkCreateDevice(vk_physical_device, &device_create_info, nullptr, &vk_device));

	// -----------------------------------------------
	// Vulkan queue / command pool / command buffer
	// -----------------------------------------------
	vkGetDeviceQueue(vk_device, indices.all_family, 0, &vk_queue);

	VkCommandPoolCreateInfo command_pool_info = {};
	command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	command_pool_info.queueFamilyIndex = indices.all_family;

	VK_CHECK_THROW(vkCreateCommandPool(vk_device, &command_pool_info, nullptr, &vk_command_pool));

	VkCommandBufferAllocateInfo command_buffer_alloc_info = {};
	command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_alloc_info.commandPool = vk_command_pool;
	command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	command_buffer_alloc_info.commandBufferCount = 1;

	VK_CHECK_THROW(vkAllocateCommandBuffers(vk_device, &command_buffer_alloc_info, &vk_command_buffer));

	// -------------------------------
	// NGX init
	// -------------------------------
	std::wstring path;
#ifdef _WIN32
	path = fs::path::getcwd().wstr();
#else
	std::string tmp = fs::path::getcwd().str();
	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
	path = converter.from_bytes(tmp);
#endif

	NGX_CHECK_THROW(NVSDK_NGX_VULKAN_Init_with_ProjectID("ea75345e-5a42-4037-a5c9-59bf94dee157", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0.0", path.c_str(), vk_instance, vk_physical_device, vk_device));
	ngx_initialized = true;

	// -------------------------------
	// Ensure DLSS capability
	// -------------------------------
	NGX_CHECK_THROW(NVSDK_NGX_VULKAN_GetCapabilityParameters(&ngx_parameters));

	int needs_updated_driver = 0;
	unsigned int min_driver_version_major = 0;
	unsigned int min_driver_version_minor = 0;
	NVSDK_NGX_Result result_updated_driver = ngx_parameters->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_updated_driver);
	NVSDK_NGX_Result result_min_driver_version_major = ngx_parameters->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &min_driver_version_major);
	NVSDK_NGX_Result result_min_driver_version_minor = ngx_parameters->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &min_driver_version_minor);
	if (result_updated_driver == NVSDK_NGX_Result_Success && result_min_driver_version_major == NVSDK_NGX_Result_Success && result_min_driver_version_minor == NVSDK_NGX_Result_Success) {
		if (needs_updated_driver) {
			throw std::runtime_error{fmt::format("Driver too old. Minimum version required is {}.{}", min_driver_version_major, min_driver_version_minor)};
		}
	}

	int dlss_available  = 0;
	NVSDK_NGX_Result ngx_result = ngx_parameters->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlss_available);
	if (ngx_result != NVSDK_NGX_Result_Success || !dlss_available) {
		ngx_result = NVSDK_NGX_Result_Fail;
		NVSDK_NGX_Parameter_GetI(ngx_parameters, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, (int*)&ngx_result);
		throw std::runtime_error{fmt::format("DLSS not available: {}", ngx_error_string(ngx_result))};
	}

	tlog::success() << "Initialized Vulkan and NGX on GPU #" << device_id << ": " << physical_device_properties.deviceName;
}

size_t dlss_allocated_bytes() {
	unsigned long long allocated_bytes = 0;
	if (!ngx_parameters) {
		return 0;
	}

	try {
		NGX_CHECK_THROW(NGX_DLSS_GET_STATS(ngx_parameters, &allocated_bytes));
	} catch (...) {
		return 0;
	}

	return allocated_bytes;
}

void vk_command_buffer_begin() {
	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	begin_info.pInheritanceInfo = nullptr;

	VK_CHECK_THROW(vkBeginCommandBuffer(vk_command_buffer, &begin_info));
}

void vk_command_buffer_end() {
	VK_CHECK_THROW(vkEndCommandBuffer(vk_command_buffer));
}

void vk_command_buffer_submit() {
	VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &vk_command_buffer;

	VK_CHECK_THROW(vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE));
}

void vk_synchronize() {
	VK_CHECK_THROW(vkDeviceWaitIdle(vk_device));
}

void vk_command_buffer_submit_sync() {
	vk_command_buffer_submit();
	vk_synchronize();
}

void vk_command_buffer_end_and_submit_sync() {
	vk_command_buffer_end();
	vk_command_buffer_submit_sync();
}

class VulkanTexture {
public:
	VulkanTexture(const Vector2i& size, uint32_t n_channels) : m_size{size}, m_n_channels{n_channels} {
		VkImageCreateInfo image_info{};
		image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_info.imageType = VK_IMAGE_TYPE_2D;
		image_info.extent.width = static_cast<uint32_t>(m_size.x());
		image_info.extent.height = static_cast<uint32_t>(m_size.y());
		image_info.extent.depth = 1;
		image_info.mipLevels = 1;
		image_info.arrayLayers = 1;

		switch (n_channels) {
			case 1: image_info.format = VK_FORMAT_R32_SFLOAT; break;
			case 2: image_info.format = VK_FORMAT_R32G32_SFLOAT; break;
			case 3: image_info.format = VK_FORMAT_R32G32B32_SFLOAT; break;
			case 4: image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT; break;
			default: throw std::runtime_error{"VulkanTexture only supports 1, 2, 3, or 4 channels."};
		}

		image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		image_info.samples = VK_SAMPLE_COUNT_1_BIT;
		image_info.flags = 0;

		VkExternalMemoryImageCreateInfoKHR ext_image_info = {};
		ext_image_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;

#ifdef _WIN32
		ext_image_info.handleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
		ext_image_info.handleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

		image_info.pNext = &ext_image_info;

		VK_CHECK_THROW(vkCreateImage(vk_device, &image_info, nullptr, &m_vk_image));

		// Create device memory to back up the image
		VkMemoryRequirements mem_requirements = {};

		vkGetImageMemoryRequirements(vk_device, m_vk_image, &mem_requirements);

		VkMemoryAllocateInfo mem_alloc_info = {};
		mem_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		mem_alloc_info.allocationSize = mem_requirements.size;
		mem_alloc_info.memoryTypeIndex = vk_find_memory_type(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		VkExportMemoryAllocateInfoKHR export_info = {};
		export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
		export_info.handleTypes = ext_image_info.handleTypes;

		mem_alloc_info.pNext = &export_info;

		VK_CHECK_THROW(vkAllocateMemory(vk_device, &mem_alloc_info, nullptr, &m_vk_device_memory));
		VK_CHECK_THROW(vkBindImageMemory(vk_device, m_vk_image, m_vk_device_memory, 0));

		vk_command_buffer_begin();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = m_vk_image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		vkCmdPipelineBarrier(
			vk_command_buffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		vk_command_buffer_end_and_submit_sync();

		// Image view
		VkImageViewCreateInfo view_info = {};
		view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_info.image = m_vk_image;
		view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view_info.format = image_info.format;
		view_info.subresourceRange = barrier.subresourceRange;

		VK_CHECK_THROW(vkCreateImageView(vk_device, &view_info, nullptr, &m_vk_image_view));

		// Map to NGX
		m_ngx_resource = NVSDK_NGX_Create_ImageView_Resource_VK(m_vk_image_view, m_vk_image, view_info.subresourceRange, image_info.format, m_size.x(), m_size.y(), true);

		// Map to CUDA memory: VkDeviceMemory->FD/HANDLE->cudaExternalMemory->CUDA pointer
#ifdef _WIN32
		HANDLE handle = nullptr;
		VkMemoryGetWin32HandleInfoKHR handle_info = {};
		handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		handle_info.memory = m_vk_device_memory;
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		auto pfn_vkGetMemory = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(vk_device, "vkGetMemoryWin32HandleKHR");
#else
		int handle = -1;
		VkMemoryGetFdInfoKHR handle_info = {};
		handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
		handle_info.memory = m_vk_device_memory;
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
		auto pfn_vkGetMemory = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vk_device, "vkGetMemoryFdKHR");
#endif

		if (!pfn_vkGetMemory) {
			throw std::runtime_error{"Failed to locate pfn_vkGetMemory."};
		}

		VK_CHECK_THROW(pfn_vkGetMemory(vk_device, &handle_info, &handle));

		// Map handle to CUDA memory
		cudaExternalMemoryHandleDesc external_memory_handle_desc = {};
		memset(&external_memory_handle_desc, 0, sizeof(external_memory_handle_desc));

#ifdef _WIN32
		external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
		external_memory_handle_desc.handle.win32.handle = handle;
#else
		external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
		external_memory_handle_desc.handle.fd = handle;
#endif
		external_memory_handle_desc.size = mem_requirements.size;

		CUDA_CHECK_THROW(cudaImportExternalMemory(&m_cuda_external_memory, &external_memory_handle_desc));

		cudaExternalMemoryBufferDesc external_memory_buffer_desc = {};
		memset(&external_memory_buffer_desc, 0, sizeof(external_memory_buffer_desc));
		external_memory_buffer_desc.offset = 0;
		external_memory_buffer_desc.size = mem_requirements.size;

		void* ptr;
		CUDA_CHECK_THROW(cudaExternalMemoryGetMappedBuffer(&ptr, m_cuda_external_memory, &external_memory_buffer_desc));
		m_cuda_data = (float*)ptr;

		// ----------------
		// Also get a surface object array, as the above buffer might be too cumbersome to deal with
		// ----------------
		cudaExternalMemoryMipmappedArrayDesc external_memory_mipmapped_array_desc = {};
		memset(&external_memory_mipmapped_array_desc, 0, sizeof(external_memory_mipmapped_array_desc));

		cudaChannelFormatDesc channel_format = {};
		channel_format.f = cudaChannelFormatKindFloat;
		switch (n_channels) {
			case 1: channel_format.x = 32; channel_format.y = 0;  channel_format.z = 0;  channel_format.w = 0;  break;
			case 2: channel_format.x = 32; channel_format.y = 32; channel_format.z = 0;  channel_format.w = 0;  break;
			case 3: channel_format.x = 32; channel_format.y = 32; channel_format.z = 32; channel_format.w = 0;  break;
			case 4: channel_format.x = 32; channel_format.y = 32; channel_format.z = 32; channel_format.w = 32; break;
			default: throw std::runtime_error{"VulkanTexture only supports 1, 2, 3, or 4 channels."};
		}

		cudaExtent extent = {};
		extent.width = m_size.x();
		extent.height = m_size.y();
		extent.depth = 0;

		external_memory_mipmapped_array_desc.offset = 0;
		external_memory_mipmapped_array_desc.formatDesc = channel_format;
		external_memory_mipmapped_array_desc.extent = extent;
		external_memory_mipmapped_array_desc.flags = cudaArraySurfaceLoadStore;
		external_memory_mipmapped_array_desc.numLevels = 1;

		cudaExternalMemoryGetMappedMipmappedArray(&m_cuda_mipmapped_array, m_cuda_external_memory, &external_memory_mipmapped_array_desc);

		cudaArray_t first_level_array;
		CUDA_CHECK_THROW(cudaGetMipmappedArrayLevel(&first_level_array, m_cuda_mipmapped_array, 0));

		struct cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(resource_desc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = first_level_array;

		CUDA_CHECK_THROW(cudaCreateSurfaceObject(&m_cuda_surface_object, &resource_desc));

		m_n_bytes = mem_requirements.size;
		g_total_n_bytes_allocated += m_n_bytes;
	}

	virtual ~VulkanTexture() {
		g_total_n_bytes_allocated -= m_n_bytes;

		if (m_cuda_data) {
			cudaFree(m_cuda_data);
		}

		if (m_cuda_surface_object) {
			cudaDestroySurfaceObject(m_cuda_surface_object);
		}

		if (m_cuda_mipmapped_array) {
			cudaFreeMipmappedArray(m_cuda_mipmapped_array);
		}

		if (m_cuda_external_memory) {
			cudaDestroyExternalMemory(m_cuda_external_memory);
		}

		if (m_vk_image_view) {
			vkDestroyImageView(vk_device, m_vk_image_view, nullptr);
		}

		if (m_vk_image) {
			vkDestroyImage(vk_device, m_vk_image, nullptr);
		}

		if (m_vk_device_memory) {
			vkFreeMemory(vk_device, m_vk_device_memory, nullptr);
		}
	}

	float* data() {
		return m_cuda_data;
	}

	cudaSurfaceObject_t surface() {
		return m_cuda_surface_object;
	}

	NVSDK_NGX_Resource_VK& ngx_resource() {
		return m_ngx_resource;
	}

	size_t bytes() const {
		return m_size.x() * (size_t)m_size.y() * sizeof(float) * m_n_channels;
	}

	Vector2i size() const {
		return m_size;
	}

private:
	Vector2i m_size;
	uint32_t m_n_channels;

	size_t m_n_bytes = 0;

	VkImage m_vk_image = {};
	VkImageView m_vk_image_view = {};
	VkDeviceMemory m_vk_device_memory = {};

	cudaExternalMemory_t m_cuda_external_memory = {};
	cudaMipmappedArray_t m_cuda_mipmapped_array = {};
	cudaSurfaceObject_t m_cuda_surface_object = {};
	float* m_cuda_data;

	NVSDK_NGX_Resource_VK m_ngx_resource = {};
};

NVSDK_NGX_PerfQuality_Value ngx_dlss_quality(EDlssQuality quality) {
	switch (quality) {
		case EDlssQuality::UltraPerformance: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
		case EDlssQuality::MaxPerformance: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
		case EDlssQuality::Balanced: return NVSDK_NGX_PerfQuality_Value_Balanced;
		case EDlssQuality::MaxQuality: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
		case EDlssQuality::UltraQuality: return NVSDK_NGX_PerfQuality_Value_UltraQuality;
		default: throw std::runtime_error{"Unknown DLSS quality setting."};
	}
}

struct DlssFeatureSpecs {
	EDlssQuality quality;
	Vector2i out_resolution;
	Vector2i optimal_in_resolution;
	Vector2i min_in_resolution;
	Vector2i max_in_resolution;
	float optimal_sharpness;

	float distance(const Vector2i& resolution) const {
		return (min_in_resolution - resolution).cwiseMax(resolution - max_in_resolution).cwiseMax(0).cast<float>().norm();
	}

	Vector2i clamp_resolution(const Vector2i& resolution) const {
		return resolution.cwiseMax(min_in_resolution).cwiseMin(max_in_resolution);
	}
};

DlssFeatureSpecs dlss_feature_specs(const Eigen::Vector2i& out_resolution, EDlssQuality quality) {
	DlssFeatureSpecs specs;
	specs.quality = quality;
	specs.out_resolution = out_resolution;

	NGX_CHECK_THROW(NGX_DLSS_GET_OPTIMAL_SETTINGS(
		ngx_parameters,
		specs.out_resolution.x(), specs.out_resolution.y(),
		ngx_dlss_quality(quality),
		(uint32_t*)&specs.optimal_in_resolution.x(), (uint32_t*)&specs.optimal_in_resolution.y(),
		(uint32_t*)&specs.max_in_resolution.x(), (uint32_t*)&specs.max_in_resolution.y(),
		(uint32_t*)&specs.min_in_resolution.x(), (uint32_t*)&specs.min_in_resolution.y(),
		&specs.optimal_sharpness
	));

	// Don't permit input resolutions larger than the output. (Just in case DLSS allows it.)
	specs.optimal_in_resolution = specs.optimal_in_resolution.cwiseMin(out_resolution);
	specs.max_in_resolution = specs.max_in_resolution.cwiseMin(out_resolution);
	specs.min_in_resolution = specs.min_in_resolution.cwiseMin(out_resolution);

	return specs;
}

class DlssFeature {
public:
	DlssFeature(const DlssFeatureSpecs& specs, bool is_hdr, bool sharpen) : m_specs{specs}, m_is_hdr{is_hdr}, m_sharpen{sharpen} {
		// Initialize DLSS
		unsigned int creation_node_mask = 1;
		unsigned int visibility_node_mask = 1;

		int dlss_create_feature_flags = NVSDK_NGX_DLSS_Feature_Flags_None;
		dlss_create_feature_flags |= true ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0;
		dlss_create_feature_flags |= false ? NVSDK_NGX_DLSS_Feature_Flags_MVJittered : 0;
		dlss_create_feature_flags |= is_hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
		dlss_create_feature_flags |= false ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;
		dlss_create_feature_flags |= sharpen ? NVSDK_NGX_DLSS_Feature_Flags_DoSharpening : 0;
		dlss_create_feature_flags |= false ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0;

		NVSDK_NGX_DLSS_Create_Params dlss_create_params;

		memset(&dlss_create_params, 0, sizeof(dlss_create_params));

		dlss_create_params.Feature.InWidth = m_specs.optimal_in_resolution.x();
		dlss_create_params.Feature.InHeight = m_specs.optimal_in_resolution.y();
		dlss_create_params.Feature.InTargetWidth = m_specs.out_resolution.x();
		dlss_create_params.Feature.InTargetHeight = m_specs.out_resolution.y();
		dlss_create_params.Feature.InPerfQualityValue = ngx_dlss_quality(m_specs.quality);
		dlss_create_params.InFeatureCreateFlags = dlss_create_feature_flags;

		{
			vk_command_buffer_begin();
			ScopeGuard command_buffer_guard{[&]() { vk_command_buffer_end_and_submit_sync(); }};

			NGX_CHECK_THROW(NGX_VULKAN_CREATE_DLSS_EXT(vk_command_buffer, creation_node_mask, visibility_node_mask, &m_ngx_dlss, ngx_parameters, &dlss_create_params));
		}
	}

	DlssFeature(const Eigen::Vector2i& out_resolution, bool is_hdr, bool sharpen, EDlssQuality quality)
	: DlssFeature{dlss_feature_specs(out_resolution, quality), is_hdr, sharpen} {}

	~DlssFeature() {
		cudaDeviceSynchronize();

		if (m_ngx_dlss) {
			NVSDK_NGX_VULKAN_ReleaseFeature(m_ngx_dlss);
		}

		vk_synchronize();
	}

	void run(
		const Vector2i& in_resolution,
		const Vector2f& jitter_offset,
		float sharpening,
		bool shall_reset,
		NVSDK_NGX_Resource_VK& frame,
		NVSDK_NGX_Resource_VK& depth,
		NVSDK_NGX_Resource_VK& mvec,
		NVSDK_NGX_Resource_VK& exposure,
		NVSDK_NGX_Resource_VK& output
	) {
		if (!m_sharpen && sharpening != 0.0f) {
			throw std::runtime_error{"May only specify non-zero sharpening, when DlssFeature has been created with sharpen option."};
		}

		vk_command_buffer_begin();

		NVSDK_NGX_VK_DLSS_Eval_Params dlss_params;
		memset(&dlss_params, 0, sizeof(dlss_params));

		dlss_params.Feature.pInColor = &frame;
		dlss_params.Feature.pInOutput = &output;
		dlss_params.pInDepth = &depth;
		dlss_params.pInMotionVectors = &mvec;
		dlss_params.pInExposureTexture = &exposure;
		dlss_params.InJitterOffsetX = jitter_offset.x();
		dlss_params.InJitterOffsetY = jitter_offset.y();
		dlss_params.Feature.InSharpness = sharpening;
		dlss_params.InReset = shall_reset;
		dlss_params.InMVScaleX = 1.0f;
		dlss_params.InMVScaleY = 1.0f;
		dlss_params.InRenderSubrectDimensions = {(uint32_t)in_resolution.x(), (uint32_t)in_resolution.y()};

		NGX_CHECK_THROW(NGX_VULKAN_EVALUATE_DLSS_EXT(vk_command_buffer, m_ngx_dlss, ngx_parameters, &dlss_params));

		vk_command_buffer_end_and_submit_sync();
	}

	bool is_hdr() const {
		return m_is_hdr;
	}

	bool sharpen() const {
		return m_sharpen;
	}

	EDlssQuality quality() const {
		return m_specs.quality;
	}

	Vector2i out_resolution() const {
		return m_specs.out_resolution;
	}

	Vector2i clamp_resolution(const Vector2i& resolution) const {
		return m_specs.clamp_resolution(resolution);
	}

	Vector2i optimal_in_resolution() const {
		return m_specs.optimal_in_resolution;
	}

private:
	NVSDK_NGX_Handle* m_ngx_dlss = {};
	DlssFeatureSpecs m_specs;
	bool m_is_hdr;
	bool m_sharpen;
};

class Dlss : public IDlss {
public:
	Dlss(const Eigen::Vector2i& max_out_resolution)
	:
	m_max_out_resolution{max_out_resolution},
	// Allocate all buffers at output resolution and use dynamic sub-rects
	// to use subsets of them. This avoids re-allocations when using DLSS
	// with dynamically changing input resolution.
	m_frame_buffer{max_out_resolution, 4},
	m_depth_buffer{max_out_resolution, 1},
	m_mvec_buffer{max_out_resolution, 2},
	m_exposure_buffer{{1, 1}, 1},
	m_output_buffer{max_out_resolution, 4}
	{
		// Various quality modes of DLSS
		for (int i = 0; i < (int)EDlssQuality::NumDlssQualitySettings; ++i) {
			try {
				auto specs = dlss_feature_specs(max_out_resolution, (EDlssQuality)i);

				// Only emplace the specs if the feature can be created in practice!
				DlssFeature{specs, true, true};
				DlssFeature{specs, true, false};
				DlssFeature{specs, false, true};
				DlssFeature{specs, false, false};
				m_dlss_specs.emplace_back(specs);
			} catch (...) {}
		}

		// For super insane performance requirements (more than 3x upscaling) try UltraPerformance
		// with reduced output resolutions for 4.5x, 6x, 9x.
		std::vector<Vector2i> reduced_out_resolutions = {
			max_out_resolution / 3 * 2,
			max_out_resolution / 2,
			max_out_resolution / 3,
			// max_out_resolution / 4,
		};

		for (const auto& out_resolution : reduced_out_resolutions) {
			try {
				auto specs = dlss_feature_specs(out_resolution, EDlssQuality::UltraPerformance);

				// Only emplace the specs if the feature can be created in practice!
				DlssFeature{specs, true, true};
				DlssFeature{specs, true, false};
				DlssFeature{specs, false, true};
				DlssFeature{specs, false, false};
				m_dlss_specs.emplace_back(specs);
			} catch (...) {}
		}
	}

	virtual ~Dlss() {
		// Destroy DLSS feature prior to killing underlying buffers.
		m_dlss_feature = nullptr;
	}

	void update_feature(const Vector2i& in_resolution, bool is_hdr, bool sharpen) override {
		CUDA_CHECK_THROW(cudaDeviceSynchronize());

		DlssFeatureSpecs specs;
		bool found = false;
		for (const auto& s : m_dlss_specs) {
			if (s.distance(in_resolution) == 0.0f) {
				specs = s;
				found = true;
			}
		}

		if (!found) {
			throw std::runtime_error{"Dlss::run called with invalid input resolution."};
		}

		if (!m_dlss_feature || m_dlss_feature->is_hdr() != is_hdr || m_dlss_feature->sharpen() != sharpen || m_dlss_feature->quality() != specs.quality || m_dlss_feature->out_resolution() != specs.out_resolution) {
			m_dlss_feature.reset(new DlssFeature{specs.out_resolution, is_hdr, sharpen, specs.quality});
		}
	}

	void run(
		const Vector2i& in_resolution,
		bool is_hdr,
		float sharpening,
		const Vector2f& jitter_offset,
		bool shall_reset
	) override {
		CUDA_CHECK_THROW(cudaDeviceSynchronize());

		update_feature(in_resolution, is_hdr, sharpening != 0.0f);

		m_dlss_feature->run(
			in_resolution,
			jitter_offset,
			sharpening,
			shall_reset,
			m_frame_buffer.ngx_resource(),
			m_depth_buffer.ngx_resource(),
			m_mvec_buffer.ngx_resource(),
			m_exposure_buffer.ngx_resource(),
			m_output_buffer.ngx_resource()
		);
	}

	cudaSurfaceObject_t frame() override {
		return m_frame_buffer.surface();
	}

	cudaSurfaceObject_t depth() override {
		return m_depth_buffer.surface();
	}

	cudaSurfaceObject_t mvec() override {
		return m_mvec_buffer.surface();
	}

	cudaSurfaceObject_t exposure() override {
		return m_exposure_buffer.surface();
	}

	cudaSurfaceObject_t output() override {
		return m_output_buffer.surface();
	}

	Vector2i clamp_resolution(const Vector2i& resolution) const {
		float min_distance = std::numeric_limits<float>::infinity();
		DlssFeatureSpecs min_distance_specs = {};
		for (const auto& specs : m_dlss_specs) {
			float distance = specs.distance(resolution);
			if (distance <= min_distance) {
				min_distance = distance;
				min_distance_specs = specs;
			}
		}

		return min_distance_specs.clamp_resolution(resolution);
	}

	Vector2i out_resolution() const override {
		return m_dlss_feature ? m_dlss_feature->out_resolution() : m_max_out_resolution;
	}

	Vector2i max_out_resolution() const override {
		return m_max_out_resolution;
	}

	bool is_hdr() const override {
		return m_dlss_feature && m_dlss_feature->is_hdr();
	}

	bool sharpen() const override {
		return m_dlss_feature && m_dlss_feature->sharpen();
	}

	EDlssQuality quality() const override {
		return m_dlss_feature ? m_dlss_feature->quality() : EDlssQuality::None;
	}

private:
	std::unique_ptr<DlssFeature> m_dlss_feature;
	std::vector<DlssFeatureSpecs> m_dlss_specs;

	VulkanTexture m_frame_buffer;
	VulkanTexture m_depth_buffer;
	VulkanTexture m_mvec_buffer;
	VulkanTexture m_exposure_buffer;
	VulkanTexture m_output_buffer;

	Vector2i m_max_out_resolution;
};

std::shared_ptr<IDlss> dlss_init(const Eigen::Vector2i& out_resolution) {
	return std::make_shared<Dlss>(out_resolution);
}

void vulkan_and_ngx_destroy() {
	if (ngx_parameters) {
		NVSDK_NGX_VULKAN_DestroyParameters(ngx_parameters);
		ngx_parameters = nullptr;
	}

	if (ngx_initialized) {
		NVSDK_NGX_VULKAN_Shutdown();
		ngx_initialized = false;
	}

	if (vk_command_pool) {
		vkDestroyCommandPool(vk_device, vk_command_pool, nullptr);
		vk_command_pool = VK_NULL_HANDLE;
	}

	if (vk_device) {
		vkDestroyDevice(vk_device, nullptr);
		vk_device = VK_NULL_HANDLE;
	}

	if (vk_debug_messenger) {
		auto DestroyDebugUtilsMessengerEXT = [](VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr) {
				func(instance, debugMessenger, pAllocator);
			}
		};

		DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nullptr);
		vk_debug_messenger = VK_NULL_HANDLE;
	}

	if (vk_instance) {
		vkDestroyInstance(vk_instance, nullptr);
		vk_instance = VK_NULL_HANDLE;
	}
}

NGP_NAMESPACE_END
