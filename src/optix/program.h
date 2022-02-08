/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   program.h
 *  @author Thomas Müller, NVIDIA
 */

#pragma once

NGP_NAMESPACE_BEGIN

#define OPTIX_CHECK_THROW(x)                                                                                 \
	do {                                                                                                     \
		OptixResult res = x;                                                                                 \
		if (res != OPTIX_SUCCESS) {                                                                          \
			throw std::runtime_error(std::string("Optix call '" #x "' failed."));                            \
		}                                                                                                    \
	} while(0)

#define OPTIX_CHECK_THROW_LOG(x)                                                                                                                          \
	do {                                                                                                                                                  \
		OptixResult res = x;                                                                                                                              \
		const size_t sizeof_log_returned = sizeof_log;                                                                                                    \
		sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */                                                                               \
		if (res != OPTIX_SUCCESS) {                                                                                                                       \
			throw std::runtime_error(std::string("Optix call '" #x "' failed. Log:\n") + log + (sizeof_log_returned == sizeof_log ? "" : "<truncated>")); \
		}                                                                                                                                                 \
	} while(0)


namespace optix {
	template <typename T>
	struct SbtRecord {
		__align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	template <typename T>
	class Program {
	public:
		Program(const char* data, size_t size, OptixDeviceContext optix) {
			char log[2048]; // For error reporting from OptiX creation functions
			size_t sizeof_log = sizeof(log);

			// Module from PTX
			OptixModule optix_module = nullptr;
			OptixPipelineCompileOptions pipeline_compile_options = {};
			{
				// Default options for our module.
				OptixModuleCompileOptions module_compile_options = {};

				// Pipeline options must be consistent for all modules used in a
				// single pipeline
				pipeline_compile_options.usesMotionBlur = false;

				// This option is important to ensure we compile code which is optimal
				// for our scene hierarchy. We use a single GAS � no instancing or
				// multi-level hierarchies
				pipeline_compile_options.traversableGraphFlags =
					OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

				// Our device code uses 3 payload registers (r,g,b output value)
				pipeline_compile_options.numPayloadValues = 3;

				// This is the name of the param struct variable in our device code
				pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

				OPTIX_CHECK_THROW_LOG(optixModuleCreateFromPTX(
					optix,
					&module_compile_options,
					&pipeline_compile_options,
					data,
					size,
					log,
					&sizeof_log,
					&optix_module
				));
			}

			// Program groups
			OptixProgramGroup raygen_prog_group   = nullptr;
			OptixProgramGroup miss_prog_group     = nullptr;
			OptixProgramGroup hitgroup_prog_group = nullptr;
			{
				OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

				OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
				raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
				raygen_prog_group_desc.raygen.module            = optix_module;
				raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
				OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
					optix,
					&raygen_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					log,
					&sizeof_log,
					&raygen_prog_group
				));

				OptixProgramGroupDesc miss_prog_group_desc  = {};
				miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
				miss_prog_group_desc.miss.module            = optix_module;
				miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
				OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
					optix,
					&miss_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					log,
					&sizeof_log,
					&miss_prog_group
				));

				OptixProgramGroupDesc hitgroup_prog_group_desc = {};
				hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				hitgroup_prog_group_desc.hitgroup.moduleCH            = optix_module;
				hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
				OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
					optix,
					&hitgroup_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					log,
					&sizeof_log,
					&hitgroup_prog_group
				));
			}

			// Linking
			{
				const uint32_t max_trace_depth = 1;
				OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

				OptixPipelineLinkOptions pipeline_link_options = {};
				pipeline_link_options.maxTraceDepth = max_trace_depth;
				pipeline_link_options.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

				OPTIX_CHECK_THROW_LOG(optixPipelineCreate(
					optix,
					&pipeline_compile_options,
					&pipeline_link_options,
					program_groups,
					sizeof(program_groups) / sizeof(program_groups[0]),
					log,
					&sizeof_log,
					&m_pipeline
				));

				OptixStackSizes stack_sizes = {};
				for (auto& prog_group : program_groups) {
					OPTIX_CHECK_THROW(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
				}

				uint32_t direct_callable_stack_size_from_traversal;
				uint32_t direct_callable_stack_size_from_state;
				uint32_t continuation_stack_size;
				OPTIX_CHECK_THROW(optixUtilComputeStackSizes(
					&stack_sizes, max_trace_depth,
					0,  // maxCCDepth
					0,  // maxDCDEpth
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state, &continuation_stack_size
				));
				OPTIX_CHECK_THROW(optixPipelineSetStackSize(
					m_pipeline, direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state, continuation_stack_size,
					1  // maxTraversableDepth
				));
			}

			// Shader binding table
			{
				CUdeviceptr raygen_record;
				const size_t raygen_record_size = sizeof(SbtRecord<typename T::RayGenData>);
				CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
				SbtRecord<typename T::RayGenData> rg_sbt;
				OPTIX_CHECK_THROW(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
				CUDA_CHECK_THROW(cudaMemcpy(
					reinterpret_cast<void*>(raygen_record),
					&rg_sbt,
					raygen_record_size,
					cudaMemcpyHostToDevice
				));

				CUdeviceptr miss_record;
				size_t miss_record_size = sizeof(SbtRecord<typename T::MissData>);
				CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
				SbtRecord<typename T::MissData> ms_sbt;
				OPTIX_CHECK_THROW(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
				CUDA_CHECK_THROW(cudaMemcpy(
					reinterpret_cast<void*>(miss_record),
					&ms_sbt,
					miss_record_size,
					cudaMemcpyHostToDevice
				));

				CUdeviceptr hitgroup_record;
				size_t hitgroup_record_size = sizeof(SbtRecord<typename T::HitGroupData>);
				CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
				SbtRecord<typename T::HitGroupData> hg_sbt;
				OPTIX_CHECK_THROW(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
				CUDA_CHECK_THROW(cudaMemcpy(
					reinterpret_cast<void*>(hitgroup_record),
					&hg_sbt,
					hitgroup_record_size,
					cudaMemcpyHostToDevice
				));

				m_sbt.raygenRecord                = raygen_record;
				m_sbt.missRecordBase              = miss_record;
				m_sbt.missRecordStrideInBytes     = sizeof(SbtRecord<typename T::MissData>);
				m_sbt.missRecordCount             = 1;
				m_sbt.hitgroupRecordBase          = hitgroup_record;
				m_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<typename T::HitGroupData>);
				m_sbt.hitgroupRecordCount         = 1;
			}
		}

		void invoke(const typename T::Params& params, const uint3& dim, cudaStream_t stream) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_params_gpu.data(), &params, sizeof(typename T::Params), cudaMemcpyHostToDevice, stream));
			OPTIX_CHECK_THROW(optixLaunch(m_pipeline, stream, (CUdeviceptr)(uintptr_t)m_params_gpu.data(), sizeof(typename T::Params), &m_sbt, dim.x, dim.y, dim.z));
		}

	private:
		OptixShaderBindingTable m_sbt = {};
		OptixPipeline m_pipeline = nullptr;
		tcnn::GPUMemory<typename T::Params> m_params_gpu = tcnn::GPUMemory<typename T::Params>(1);
	};
}

NGP_NAMESPACE_END
