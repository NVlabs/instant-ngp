/*
 * Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header
///
/// OptiX host include file -- includes the host api if compiling host code.
/// For the math library routines include optix_math.h

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_7_host.h is an internal header file and must not be used directly.  Please use optix_host.h or optix.h instead.")
#endif

#ifndef __optix_optix_7_host_h__
#define __optix_optix_7_host_h__

#include "optix_7_types.h"
#if !defined( OPTIX_DONT_INCLUDE_CUDA )
// If OPTIX_DONT_INCLUDE_CUDA is defined, cuda driver types must be defined through other
// means before including optix headers.
#include <cuda.h>
#endif



#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup optix_host_api Host API
/// \brief OptiX Host API

/// \defgroup optix_host_api_error_handling Error handling
/// \ingroup optix_host_api
//@{

/// Returns a string containing the name of an error code in the enum.
///
/// Output is a string representation of the enum.  For example "OPTIX_SUCCESS" for
/// OPTIX_SUCCESS and "OPTIX_ERROR_INVALID_VALUE" for OPTIX_ERROR_INVALID_VALUE.
///
/// If the error code is not recognized, "Unrecognized OptixResult code" is returned.
///
/// \param[in] result  OptixResult enum to generate string name for
///
/// \see #optixGetErrorString
const char* optixGetErrorName( OptixResult result );

/// Returns the description string for an error code.
///
/// Output is a string description of the enum.  For example "Success" for
/// OPTIX_SUCCESS and "Invalid value" for OPTIX_ERROR_INVALID_VALUE.
///
/// If the error code is not recognized, "Unrecognized OptixResult code" is returned.
///
/// \param[in] result  OptixResult enum to generate string description for
///
/// \see #optixGetErrorName
const char* optixGetErrorString( OptixResult result );

//@}
/// \defgroup optix_host_api_device_context Device context
/// \ingroup optix_host_api
//@{

/// Create a device context associated with the CUDA context specified with 'fromContext'.
///
/// If zero is specified for 'fromContext', OptiX will use the current CUDA context. The
/// CUDA context should be initialized before calling optixDeviceContextCreate.
///
/// \param[in] fromContext
/// \param[in] options
/// \param[out] context
/// \return
/// - OPTIX_ERROR_CUDA_NOT_INITIALIZED
///   If using zero for 'fromContext' and CUDA has not been initialized yet on the calling
///   thread.
/// - OPTIX_ERROR_CUDA_ERROR
///   CUDA operation failed.
/// - OPTIX_ERROR_HOST_OUT_OF_MEMORY
///   Heap allocation failed.
/// - OPTIX_ERROR_INTERNAL_ERROR
///   Internal error
OptixResult optixDeviceContextCreate( CUcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context );

/// Destroys all CPU and GPU state associated with the device.
///
/// It will attempt to block on CUDA streams that have launch work outstanding.
///
/// Any API objects, such as OptixModule and OptixPipeline, not already destroyed will be
/// destroyed.
///
/// Thread safety: A device context must not be destroyed while it is still in use by concurrent API calls in other threads.
OptixResult optixDeviceContextDestroy( OptixDeviceContext context );

/// Query properties of a device context.
///
/// \param[in] context     the device context to query the property for
/// \param[in] property    the property to query
/// \param[out] value      pointer to the returned
/// \param[in] sizeInBytes size of output
OptixResult optixDeviceContextGetProperty( OptixDeviceContext context, OptixDeviceProperty property, void* value, size_t sizeInBytes );

/// Sets the current log callback method.
///
/// See #OptixLogCallback for more details.
///
/// Thread safety: It is guaranteed that the callback itself (callbackFunction and callbackData) are updated atomically.
/// It is not guaranteed that the callback itself (callbackFunction and callbackData) and the callbackLevel are updated
/// atomically. It is unspecified when concurrent API calls using the same context start to make use of the new
/// callback method.
///
/// \param[in] context          the device context
/// \param[in] callbackFunction the callback function to call
/// \param[in] callbackData     pointer to data passed to callback function while invoking it
/// \param[in] callbackLevel    callback level
OptixResult optixDeviceContextSetLogCallback( OptixDeviceContext context,
                                              OptixLogCallback   callbackFunction,
                                              void*              callbackData,
                                              unsigned int       callbackLevel );

/// Enables or disables the disk cache.
///
/// If caching was previously disabled, enabling it will attempt to initialize
/// the disk cache database using the currently configured cache location. An
/// error will be returned if initialization fails.
///
/// Note that no in-memory cache is used, so no caching behavior will be observed if the disk cache
/// is disabled.
///
/// The cache can be disabled by setting the environment variable OPTIX_CACHE_MAXSIZE=0.
/// The environment variable takes precedence over this setting.
/// See #optixDeviceContextSetCacheDatabaseSizes for additional information.
///
/// Note that the disk cache can be disabled by the environment variable, but it cannot be enabled
/// via the environment if it is disabled via the API.
///
/// \param[in] context the device context
/// \param[in] enabled 1 to enabled, 0 to disable
OptixResult optixDeviceContextSetCacheEnabled( OptixDeviceContext context,
                                               int                enabled );

/// Sets the location of the disk cache.
///
/// The location is specified by a directory. This directory should not be used for other purposes
/// and will be created if it does not exist. An error will be returned if is not possible to
/// create the disk cache at the specified location for any reason (e.g., the path is invalid or
/// the directory is not writable). Caching will be disabled if the disk cache cannot be
/// initialized in the new location. If caching is disabled, no error will be returned until caching
/// is enabled. If the disk cache is located on a network file share, behavior is undefined.
///
/// The location of the disk cache can be overridden with the environment variable OPTIX_CACHE_PATH.
/// The environment variable takes precedence over this setting.
///
/// The default location depends on the operating system:
/// - Windows: %LOCALAPPDATA%\\NVIDIA\\OptixCache
/// - Linux:   /var/tmp/OptixCache_\<username\> (or /tmp/OptixCache_\<username\> if the first choice is not usable),
///            the underscore and username suffix are omitted if the username cannot be obtained
/// - MacOS X: /Library/Application Support/NVIDIA/OptixCache
///
/// \param[in] context  the device context
/// \param[in] location directory of disk cache
OptixResult optixDeviceContextSetCacheLocation( OptixDeviceContext context, const char* location );

/// Sets the low and high water marks for disk cache garbage collection.
///
/// Garbage collection is triggered when a new entry is written to the cache and
/// the current cache data size plus the size of the cache entry that is about
/// to be inserted exceeds the high water mark. Garbage collection proceeds until
/// the size reaches the low water mark. Garbage collection will always free enough
/// space to insert the new entry without exceeding the low water mark. Setting
/// either limit to zero will disable garbage collection. An error will be returned
/// if both limits are non-zero and the high water mark is smaller than the low water mark.
///
/// Note that garbage collection is performed only on writes to the disk cache. No garbage
/// collection is triggered on disk cache initialization or immediately when calling this function,
/// but on subsequent inserting of data into the database.
///
/// If the size of a compiled module exceeds the value configured for the high water
/// mark and garbage collection is enabled, the module will not be added to the cache
/// and a warning will be added to the log.
///
/// The high water mark can be overridden with the environment variable OPTIX_CACHE_MAXSIZE.
/// The environment variable takes precedence over the function parameters.  The low water mark
/// will be set to half the value of OPTIX_CACHE_MAXSIZE.  Setting OPTIX_CACHE_MAXSIZE to 0 will
/// disable the disk cache, but will not alter the contents of the cache.  Negative and non-integer
/// values will be ignored.
///
/// \param[in] context       the device context
/// \param[in] lowWaterMark  the low water mark
/// \param[in] highWaterMark the high water mark
OptixResult optixDeviceContextSetCacheDatabaseSizes( OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark );

/// Indicates whether the disk cache is enabled or disabled.
///
/// \param[in] context   the device context
/// \param[out] enabled  1 if enabled, 0 if disabled
OptixResult optixDeviceContextGetCacheEnabled( OptixDeviceContext context, int* enabled );
/// Returns the location of the disk cache.  If the cache has been disabled by setting the environment
/// variable OPTIX_CACHE_MAXSIZE=0, this function will return an empy string.
///
/// \param[in] context      the device context
/// \param[out] location    directory of disk cache, null terminated if locationSize > 0
/// \param[in] locationSize locationSize
OptixResult optixDeviceContextGetCacheLocation( OptixDeviceContext context, char* location, size_t locationSize );

/// Returns the low and high water marks for disk cache garbage collection.  If the cache has been disabled by
/// setting the environment variable OPTIX_CACHE_MAXSIZE=0, this function will return 0 for the low and high
/// water marks.
///
/// \param[in] context        the device context
/// \param[out] lowWaterMark  the low water mark
/// \param[out] highWaterMark the high water mark
OptixResult optixDeviceContextGetCacheDatabaseSizes( OptixDeviceContext context, size_t* lowWaterMark, size_t* highWaterMark );

//@}
/// \defgroup optix_host_api_pipelines Pipelines
/// \ingroup optix_host_api
//@{

/// logString is an optional buffer that contains compiler feedback and errors.  This
/// information is also passed to the context logger (if enabled), however it may be
/// difficult to correlate output to the logger to specific API invocations when using
/// multiple threads.  The output to logString will only contain feedback for this specific
/// invocation of this API call.
///
/// logStringSize as input should be a pointer to the number of bytes backing logString.
/// Upon return it contains the length of the log message (including the null terminator)
/// which may be greater than the input value.  In this case, the log message will be
/// truncated to fit into logString.
///
/// If logString or logStringSize are NULL, no output is written to logString.  If
/// logStringSize points to a value that is zero, no output is written.  This does not
/// affect output to the context logger if enabled.
///
/// \param[in] context
/// \param[in] pipelineCompileOptions
/// \param[in] pipelineLinkOptions
/// \param[in] programGroups          array of ProgramGroup objects
/// \param[in] numProgramGroups       number of ProgramGroup objects
/// \param[out] logString             Information will be written to this string. If logStringSize > 0 logString will be null terminated.
/// \param[in,out] logStringSize
/// \param[out] pipeline
OptixResult optixPipelineCreate( OptixDeviceContext                 context,
                                 const OptixPipelineCompileOptions* pipelineCompileOptions,
                                 const OptixPipelineLinkOptions*    pipelineLinkOptions,
                                 const OptixProgramGroup*           programGroups,
                                 unsigned int                       numProgramGroups,
                                 char*                              logString,
                                 size_t*                            logStringSize,
                                 OptixPipeline*                     pipeline );

/// Thread safety: A pipeline must not be destroyed while it is still in use by concurrent API calls in other threads.
OptixResult optixPipelineDestroy( OptixPipeline pipeline );

/// Sets the stack sizes for a pipeline.
///
/// Users are encouraged to see the programming guide and the implementations of the helper functions
/// to understand how to construct the stack sizes based on their particular needs.
///
/// If this method is not used, an internal default implementation is used. The default implementation is correct (but
/// not necessarily optimal) as long as the maximum depth of call trees of CC and DC programs is at most 2 and no motion transforms are used.
///
/// The maxTraversableGraphDepth responds to the maximal number of traversables visited when calling trace.
/// Every acceleration structure and motion transform count as one level of traversal.
/// E.g., for a simple IAS (instance acceleration structure) -> GAS (geometry acceleration structure)
/// traversal graph, the maxTraversableGraphDepth is two.
/// For IAS -> MT (motion transform) -> GAS, the maxTraversableGraphDepth is three.
/// Note that it does not matter whether a IAS or GAS has motion or not, it always counts as one.
/// Launching optix with exceptions turned on (see #OPTIX_EXCEPTION_FLAG_TRACE_DEPTH) will throw an exception
/// if the specified maxTraversableGraphDepth is too small.
///
/// \param[in] pipeline                             The pipeline to configure the stack size for.
/// \param[in] directCallableStackSizeFromTraversal The direct stack size requirement for direct callables invoked from IS or AH.
/// \param[in] directCallableStackSizeFromState     The direct stack size requirement for direct callables invoked from RG, MS, or CH.
/// \param[in] continuationStackSize                The continuation stack requirement.
/// \param[in] maxTraversableGraphDepth             The maximum depth of a traversable graph passed to trace.
OptixResult optixPipelineSetStackSize( OptixPipeline pipeline,
                                       unsigned int  directCallableStackSizeFromTraversal,
                                       unsigned int  directCallableStackSizeFromState,
                                       unsigned int  continuationStackSize,
                                       unsigned int  maxTraversableGraphDepth );

//@}
/// \defgroup optix_host_api_modules Modules
/// \ingroup optix_host_api
//@{

/// logString is an optional buffer that contains compiler feedback and errors.  This
/// information is also passed to the context logger (if enabled), however it may be
/// difficult to correlate output to the logger to specific API invocations when using
/// multiple threads.  The output to logString will only contain feedback for this specific
/// invocation of this API call.
///
/// logStringSize as input should be a pointer to the number of bytes backing logString.
/// Upon return it contains the length of the log message (including the null terminator)
/// which may be greater than the input value.  In this case, the log message will be
/// truncated to fit into logString.
///
/// If logString or logStringSize are NULL, no output is written to logString.  If
/// logStringSize points to a value that is zero, no output is written.  This does not
/// affect output to the context logger if enabled.
///
/// \param[in] context
/// \param[in] moduleCompileOptions
/// \param[in] pipelineCompileOptions All modules in a pipeline need to use the same values for the pipeline compile options.
/// \param[in] PTX                    Pointer to the PTX input string.
/// \param[in] PTXsize                Parsing proceeds up to PTXsize characters, or the first NUL byte, whichever occurs first.
/// \param[out] logString             Information will be written to this string. If logStringSize > 0 logString will be null terminated.
/// \param[in,out] logStringSize
/// \param[out] module
///
/// \return OPTIX_ERROR_INVALID_VALUE - context is 0, moduleCompileOptions is 0, pipelineCompileOptions is 0, PTX is 0, module is 0.
OptixResult optixModuleCreateFromPTX( OptixDeviceContext                 context,
                                      const OptixModuleCompileOptions*   moduleCompileOptions,
                                      const OptixPipelineCompileOptions* pipelineCompileOptions,
                                      const char*                        PTX,
                                      size_t                             PTXsize,
                                      char*                              logString,
                                      size_t*                            logStringSize,
                                      OptixModule*                       module );

/// This function is designed to do just enough work to create the OptixTask return
/// parameter and is expected to be fast enough run without needing parallel execution. A
/// single thread could generate all the OptixTask objects for further processing in a
/// work pool.
///
/// Options are similar to #optixModuleCreateFromPTX(), aside from the return parameter,
/// firstTask.
///
/// The memory used to hold the PTX should be live until all tasks are finished.
///
/// It is illegal to call #optixModuleDestroy() if any OptixTask objects are currently
/// being executed. In that case OPTIX_ERROR_ILLEGAL_DURING_TASK_EXECUTE will be returned.
///
/// If an invocation of optixTaskExecute fails, the OptixModule will be marked as
/// OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE if there are outstanding tasks or
/// OPTIX_MODULE_COMPILE_STATE_FAILURE if there are no outstanding tasks. Subsequent calls
/// to #optixTaskExecute() may execute additional work to collect compilation errors
/// generated from the input. Currently executing tasks will not necessarily be terminated
/// immediately but at the next opportunity.

/// Logging will continue to be directed to the logger installed with the
/// OptixDeviceContext. If logString is provided to #optixModuleCreateFromPTXWithTasks(),
/// it will contain all the compiler feedback from all executed tasks. The lifetime of the
/// memory pointed to by logString should extend from calling
/// #optixModuleCreateFromPTXWithTasks() to when the compilation state is either
/// OPTIX_MODULE_COMPILE_STATE_FAILURE or OPTIX_MODULE_COMPILE_STATE_COMPLETED. OptiX will
/// not write to the logString outside of execution of
/// #optixModuleCreateFromPTXWithTasks() or #optixTaskExecute(). If the compilation state
/// is OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE and no further execution of
/// #optixTaskExecute() is performed the logString may be reclaimed by the application
/// before calling #optixModuleDestroy(). The contents of logString will contain output
/// from currently completed tasks.

/// All OptixTask objects associated with a given OptixModule will be cleaned up when
/// #optixModuleDestroy() is called regardless of whether the compilation was successful
/// or not. If the compilation state is OPTIX_MODULE_COMPILE_STATE_IMPENDIND_FAILURE, any
/// unstarted OptixTask objects do not need to be executed though there is no harm doing
/// so.
///
/// \see #optixModuleCreateFromPTX
OptixResult optixModuleCreateFromPTXWithTasks( OptixDeviceContext                 context,
                                               const OptixModuleCompileOptions*   moduleCompileOptions,
                                               const OptixPipelineCompileOptions* pipelineCompileOptions,
                                               const char*                        PTX,
                                               size_t                             PTXsize,
                                               char*                              logString,
                                               size_t*                            logStringSize,
                                               OptixModule*                       module,
                                               OptixTask*                         firstTask );

/// When creating a module with tasks, the current state of the module can be queried
/// using this function.
///
/// Thread safety: Safe to call from any thread until optixModuleDestroy is called.
///
/// \see #optixModuleCreateFromPTXWithTasks
OptixResult optixModuleGetCompilationState( OptixModule module, OptixModuleCompileState* state );

/// Call for OptixModule objects created with optixModuleCreateFromPTX and optixModuleDeserialize.
///
/// Modules must not be destroyed while they are still used by any program group.
///
/// Thread safety: A module must not be destroyed while it is still in use by concurrent API calls in other threads.
OptixResult optixModuleDestroy( OptixModule module );

/// Returns a module containing the intersection program for the built-in primitive type specified
/// by the builtinISOptions.  This module must be used as the moduleIS for the OptixProgramGroupHitgroup
/// in any SBT record for that primitive type.  (The entryFunctionNameIS should be null.)
OptixResult optixBuiltinISModuleGet( OptixDeviceContext                 context,
                                     const OptixModuleCompileOptions*   moduleCompileOptions,
                                     const OptixPipelineCompileOptions* pipelineCompileOptions,
                                     const OptixBuiltinISOptions*       builtinISOptions,
                                     OptixModule*                       builtinModule );

//@}
/// \defgroup optix_host_api_tasks Tasks
/// \ingroup optix_host_api
//@{

/// Each OptixTask should be executed with #optixTaskExecute(). If additional parallel
/// work is found, new OptixTask objects will be returned in additionalTasks along with
/// the number of additional tasks in numAdditionalTasksCreated. The parameter
/// additionalTasks should point to a user allocated array of minimum size
/// maxNumAdditionalTasks. OptiX can generate upto maxNumAdditionalTasks additional tasks.
///
/// Each task can be executed in parallel and in any order.
///
/// Thread safety: Safe to call from any thread until #optixModuleDestroy() is called for
/// any associated task.
///
/// \see #optixModuleCreateFromPTXWithTasks
///
/// \param[in] task the OptixTask to execute
/// \param[in] additionalTasks pointer to array of OptixTask objects to be filled in
/// \param[in] maxNumAdditionalTasks maximum number of additional OptixTask objects
/// \param[out] numAdditionalTasksCreated number of OptixTask objects created by OptiX and written into #additionalTasks
OptixResult optixTaskExecute( OptixTask task, OptixTask* additionalTasks, unsigned int maxNumAdditionalTasks, unsigned int* numAdditionalTasksCreated );

//@}
/// \defgroup optix_host_api_program_groups Program groups
/// \ingroup optix_host_api
//@{

/// Returns the stack sizes for the given program group.
///
/// \param[in] programGroup the program group
/// \param[out] stackSizes  the corresponding stack sizes
OptixResult optixProgramGroupGetStackSize( OptixProgramGroup programGroup, OptixStackSizes* stackSizes );

/// logString is an optional buffer that contains compiler feedback and errors.  This
/// information is also passed to the context logger (if enabled), however it may be
/// difficult to correlate output to the logger to specific API invocations when using
/// multiple threads.  The output to logString will only contain feedback for this specific
/// invocation of this API call.
///
/// logStringSize as input should be a pointer to the number of bytes backing logString.
/// Upon return it contains the length of the log message (including the null terminator)
/// which may be greater than the input value.  In this case, the log message will be
/// truncated to fit into logString.
///
/// If logString or logStringSize are NULL, no output is written to logString.  If
/// logStringSize points to a value that is zero, no output is written.  This does not
/// affect output to the context logger if enabled.
///
/// Creates numProgramGroups OptiXProgramGroup objects from the specified
/// OptixProgramGroupDesc array.  The size of the arrays must match.
///
/// \param[in] context
/// \param[in] programDescriptions    N * OptixProgramGroupDesc
/// \param[in] numProgramGroups       N
/// \param[in] options
/// \param[out] logString             Information will be written to this string. If logStringSize > 0 logString will be null terminated.
/// \param[in,out] logStringSize
/// \param[out] programGroups
OptixResult optixProgramGroupCreate( OptixDeviceContext              context,
                                     const OptixProgramGroupDesc*    programDescriptions,
                                     unsigned int                    numProgramGroups,
                                     const OptixProgramGroupOptions* options,
                                     char*                           logString,
                                     size_t*                         logStringSize,
                                     OptixProgramGroup*              programGroups );

/// Thread safety: A program group must not be destroyed while it is still in use by concurrent API calls in other threads.
OptixResult optixProgramGroupDestroy( OptixProgramGroup programGroup );

//@}
/// \defgroup optix_host_api_launches Launches
/// \ingroup optix_host_api
//@{

/// Where the magic happens.
///
/// The stream and pipeline must belong to the same device context.  Multiple launches
/// may be issues in parallel from multiple threads to different streams.
///
/// pipelineParamsSize number of bytes are copied from the device memory pointed to by
/// pipelineParams before launch.  It is an error if pipelineParamsSize is greater than the
/// size of the variable declared in modules and identified by
/// OptixPipelineCompileOptions::pipelineLaunchParamsVariableName. If the launch params
/// variable was optimized out or not found in the modules linked to the pipeline then
/// the pipelineParams and pipelineParamsSize parameters are ignored.
///
/// sbt points to the shader binding table, which defines shader
/// groupings and their resources. See the SBT spec.
///
/// \param[in] pipeline
/// \param[in] stream
/// \param[in] pipelineParams
/// \param[in] pipelineParamsSize
/// \param[in] sbt
/// \param[in] width              number of elements to compute
/// \param[in] height             number of elements to compute
/// \param[in] depth              number of elements to compute
///
/// Thread safety: In the current implementation concurrent launches to the same pipeline are not
/// supported.  Concurrent launches require separate OptixPipeline objects.
OptixResult optixLaunch( OptixPipeline                  pipeline,
                         CUstream                       stream,
                         CUdeviceptr                    pipelineParams,
                         size_t                         pipelineParamsSize,
                         const OptixShaderBindingTable* sbt,
                         unsigned int                   width,
                         unsigned int                   height,
                         unsigned int                   depth );

/// \param[in]  programGroup               the program group containing the program(s)
/// \param[out] sbtRecordHeaderHostPointer  the result sbt record header
OptixResult optixSbtRecordPackHeader( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer );

//@}
/// \defgroup optix_host_api_acceleration_structures Acceleration structures
/// \ingroup optix_host_api
//@{

/// \param[in] context
/// \param[in] accelOptions   options for the accel build
/// \param[in] buildInputs    an array of OptixBuildInput objects
/// \param[in] numBuildInputs number of elements in buildInputs (must be at least 1)
/// \param[out] bufferSizes   fills in buffer sizes
OptixResult optixAccelComputeMemoryUsage( OptixDeviceContext            context,
                                          const OptixAccelBuildOptions* accelOptions,
                                          const OptixBuildInput*        buildInputs,
                                          unsigned int                  numBuildInputs,
                                          OptixAccelBufferSizes*        bufferSizes );

/// \param[in] context
/// \param[in] stream
/// \param[in] accelOptions             accel options
/// \param[in] buildInputs              an array of OptixBuildInput objects
/// \param[in] numBuildInputs           must be >= 1 for GAS, and == 1 for IAS
/// \param[in] tempBuffer               must be a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT
/// \param[in] tempBufferSizeInBytes
/// \param[in] outputBuffer             must be a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT
/// \param[in] outputBufferSizeInBytes
/// \param[out] outputHandle
/// \param[in] emittedProperties         types of requested properties and output buffers
/// \param[in] numEmittedProperties      number of post-build properties to populate (may be zero)
OptixResult optixAccelBuild( OptixDeviceContext            context,
                             CUstream                      stream,
                             const OptixAccelBuildOptions* accelOptions,
                             const OptixBuildInput*        buildInputs,
                             unsigned int                  numBuildInputs,
                             CUdeviceptr                   tempBuffer,
                             size_t                        tempBufferSizeInBytes,
                             CUdeviceptr                   outputBuffer,
                             size_t                        outputBufferSizeInBytes,
                             OptixTraversableHandle*       outputHandle,
                             const OptixAccelEmitDesc*     emittedProperties,
                             unsigned int                  numEmittedProperties );

/// Obtain relocation information, stored in OptixRelocationInfo, for a given context
/// and acceleration structure's traversable handle.
///
/// The relocation information can be passed to optixCheckRelocationCompatibility to
/// determine if an acceleration structure, referenced by 'handle', can be relocated to a
/// different device's memory space (see #optixCheckRelocationCompatibility).
///
/// When used with optixAccelRelocate, it provides data necessary for doing the relocation.
///
/// If the acceleration structure data associated with 'handle' is copied multiple times,
/// the same OptixRelocationInfo can also be used on all copies.
///
/// \param[in] context
/// \param[in] handle
/// \param[out] info
/// \return OPTIX_ERROR_INVALID_VALUE will be returned for traversable handles that are not from
/// acceleration structure builds.
OptixResult optixAccelGetRelocationInfo( OptixDeviceContext context, OptixTraversableHandle handle, OptixRelocationInfo* info );

/// Checks if an optix data structure built using another OptixDeviceContext (that was
/// used to fill in 'info') is compatible with the OptixDeviceContext specified in the
/// 'context' parameter.
///
/// Any device is always compatible with itself.
///
/// \param[in] context
/// \param[in] info
/// \param[out] compatible If OPTIX_SUCCESS is returned 'compatible' will have the value of either:
/// - 0: This context is not compatible with the optix data structure associated with 'info'.
/// - 1: This context is compatible.
OptixResult optixCheckRelocationCompatibility( OptixDeviceContext context, const OptixRelocationInfo* info, int* compatible );

/// optixAccelRelocate is called to update the acceleration structure after it has been
/// relocated.  Relocation is necessary when the acceleration structure's location in device
/// memory has changed.  optixAccelRelocate does not copy the memory.  This function only
/// operates on the relocated memory whose new location is specified by 'targetAccel'.
/// optixAccelRelocate also returns the new OptixTraversableHandle associated with
/// 'targetAccel'.  The original memory (source) is not required to be valid, only the
/// OptixRelocationInfo.
///
/// Before calling optixAccelRelocate, optixCheckRelocationCompatibility should be 
/// called to ensure the copy will be compatible with the destination device context.
///
/// The memory pointed to by 'targetAccel' should be allocated with the same size as the
/// source acceleration.  Similar to the 'outputBuffer' used in optixAccelBuild, this
/// pointer must be a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.
///
/// The memory in 'targetAccel' must be allocated as long as the accel is in use.
///
/// The instance traversables referenced by an IAS and the
/// micromaps referenced by a triangle GAS may themselves require relocation.
/// 'relocateInputs' and 'numRelocateInputs' should be used to specify the relocated
/// traversables and micromaps. After relocation, the relocated accel will reference 
/// these relocated traversables and micromaps instead of their sources.
/// The number of relocate inputs 'numRelocateInputs' must match the number of build 
/// inputs 'numBuildInputs' used to build the source accel. Relocation inputs 
/// correspond with build inputs used to build the source accel and should appear in 
/// the same order (see #optixAccelBuild).
/// 'relocateInputs' and 'numRelocateInputs' may be zero, preserving any references
/// to traversables and micromaps from the source accel.
///
/// \param[in] context
/// \param[in] stream
/// \param[in] info
/// \param[in] relocateInputs
/// \param[in] numRelocateInputs
/// \param[in] targetAccel
/// \param[in] targetAccelSizeInBytes
/// \param[out] targetHandle
OptixResult optixAccelRelocate( OptixDeviceContext         context,
                                CUstream                   stream,
                                const OptixRelocationInfo* info,
                                const OptixRelocateInput*  relocateInputs,
                                size_t                     numRelocateInputs,
                                CUdeviceptr                targetAccel,
                                size_t                     targetAccelSizeInBytes,
                                OptixTraversableHandle*    targetHandle );

/// After building an acceleration structure, it can be copied in a compacted form to reduce
/// memory.  In order to be compacted, OPTIX_BUILD_FLAG_ALLOW_COMPACTION must be supplied in
/// OptixAccelBuildOptions::buildFlags passed to optixAccelBuild.
///
/// 'outputBuffer' is the pointer to where the compacted acceleration structure will be
/// written.  This pointer must be a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.
///
/// The size of the memory specified in 'outputBufferSizeInBytes' should be at least the
/// value computed using the OPTIX_PROPERTY_TYPE_COMPACTED_SIZE that was reported during
/// optixAccelBuild.
///
/// \param[in] context
/// \param[in] stream
/// \param[in] inputHandle
/// \param[in] outputBuffer
/// \param[in] outputBufferSizeInBytes
/// \param[out] outputHandle
OptixResult optixAccelCompact( OptixDeviceContext      context,
                               CUstream                stream,
                               OptixTraversableHandle  inputHandle,
                               CUdeviceptr             outputBuffer,
                               size_t                  outputBufferSizeInBytes,
                               OptixTraversableHandle* outputHandle );

/// \param[in] onDevice
/// \param[in] pointer            pointer to traversable allocated in OptixDeviceContext. This pointer must be a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT
/// \param[in] traversableType    Type of OptixTraversableHandle to create
/// \param[out] traversableHandle traversable handle. traversableHandle must be in host memory
OptixResult optixConvertPointerToTraversableHandle( OptixDeviceContext      onDevice,
                                                    CUdeviceptr             pointer,
                                                    OptixTraversableType    traversableType,
                                                    OptixTraversableHandle* traversableHandle );


/// Determine the amount of memory necessary for a Opacity Micromap Array build.
///
/// \param[in] context
/// \param[in] buildInput
/// \param[out] bufferSizes
OptixResult optixOpacityMicromapArrayComputeMemoryUsage( OptixDeviceContext                         context,
                                                         const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                         OptixMicromapBufferSizes*                  bufferSizes );

/// Construct an array of Opacity Micromaps.
///
/// Each triangle within an instance/GAS may reference one opacity micromap to give finer 
/// control over alpha behavior. A opacity micromap consists of a set of 4^N micro-triangles 
/// in a triangular uniform barycentric grid. Multiple opacity micromaps are collected (built) 
/// into a opacity micromap array with this function. Each geometry in a GAS may bind a 
/// single opacity micromap array and can use opacity micromaps from that array only.
///
/// Each micro-triangle within a opacity micromap can be in one of four states: Transparent, 
/// Opaque, Unknown-Transparent or Unknown-Opaque. During traversal, if a triangle with a 
/// opacity micromap attached is intersected, the opacity micromap is queried to categorize 
/// the hit as either opaque, unknown (alpha) or a miss. Geometry, ray or instance flags that 
/// modify the alpha/opaque behavior are applied _after_ this opacity micromap query.
///
/// The opacity micromap query may operate in 2-state mode (alpha testing) or 4-state mode (AHS culling), 
/// depending on the opacity micromap type and ray/instance flags. When operating in 2-state 
/// mode, alpha hits will not be reported, and transparent and opaque hits must be accurate.
///
/// \param[in] context
/// \param[in] stream
/// \param[in] buildInput             a single build input object referencing many opacity micromaps
/// \param[in] buffers                the buffers used for build
/// \param[in/out] emittedProperties  types of requested properties and output buffers
/// \param[in] numEmittedProperties   number of post-build properties to populate (may be zero)
OptixResult optixOpacityMicromapArrayBuild( OptixDeviceContext                         context,
                                            CUstream                                   stream,
                                            const OptixOpacityMicromapArrayBuildInput* buildInput,
                                            const OptixMicromapBuffers*                buffers );

/// Obtain relocation information, stored in OptixRelocationInfo, for a given context
/// and opacity micromap array.
///
/// The relocation information can be passed to optixCheckRelocationCompatibility to
/// determine if a opacity micromap array, referenced by buffers, can be relocated to a
/// different device's memory space (see #optixCheckRelocationCompatibility).
///
/// When used with optixOpacityMicromapArrayRelocate, it provides data necessary for doing the relocation.
///
/// If the opacity micromap array data associated with 'opacityMicromapArray' is copied multiple times,
/// the same OptixRelocationInfo can also be used on all copies.
///
/// \param[in]  context
/// \param[in]  opacityMicromapArray
/// \param[out] info
OptixResult optixOpacityMicromapArrayGetRelocationInfo( OptixDeviceContext context, CUdeviceptr opacityMicromapArray, OptixRelocationInfo* info );

/// optixOpacityMicromapArrayRelocate is called to update the opacity micromap array after it has been
/// relocated.  Relocation is necessary when the opacity micromap array's location in device
/// memory has changed.  optixOpacityMicromapArrayRelocate does not copy the memory.  This function only
/// operates on the relocated memory whose new location is specified by 'targetOpacityMicromapArray'.
/// The original memory (source) is not required to be valid, only the
/// OptixRelocationInfo.
///
/// Before calling optixOpacityMicromapArrayRelocate, optixCheckRelocationCompatibility should be called
/// to ensure the copy will be compatible with the destination device context.
///
/// The memory pointed to by 'targetOpacityMicromapArray' should be allocated with the same size as the
/// source opacity micromap array.  Similar to the 'OptixMicromapBuffers::output' used in optixOpacityMicromapArrayBuild, 
/// this pointer must be a multiple of OPTIX_OPACITY_MICROMAP_ARRAY_BUFFER_BYTE_ALIGNMENT.
///
/// The memory in 'targetOpacityMicromapArray' must be allocated as long as the opacity micromap array is in use.
///
/// Note that any Acceleration Structures build using the original memory (source) as input will
/// still be associated with this original memory. To associate an existing (possibly relocated) 
/// Acceleration Structures with the relocated opacity micromap array, use optixAccelBuild 
/// to update the existing Acceleration Structures (See OPTIX_BUILD_OPERATION_UPDATE)
///
/// \param[in] context
/// \param[in] stream
/// \param[in] info
/// \param[in] targetOpacityMicromapArray
/// \param[in] targetOpacityMicromapArraySizeInBytes
OptixResult optixOpacityMicromapArrayRelocate( OptixDeviceContext         context,
                                               CUstream                   stream,
                                               const OptixRelocationInfo* info,
                                               CUdeviceptr                targetOpacityMicromapArray,
                                               size_t                     targetOpacityMicromapArraySizeInBytes );



//@}
/// \defgroup optix_host_api_denoiser Denoiser
/// \ingroup optix_host_api
//@{

/// Creates a denoiser object with the given options, using built-in inference models
///
/// 'modelKind' selects the model used for inference.
/// Inference for the built-in models can be guided (giving hints to improve image quality) with
/// albedo and normal vector images in the guide layer (see 'optixDenoiserInvoke').
/// Use of these images must be enabled in 'OptixDenoiserOptions'.
///
/// \param[in] context
/// \param[in] modelKind
/// \param[in] options
/// \param[out] denoiser
OptixResult optixDenoiserCreate( OptixDeviceContext context,
                                 OptixDenoiserModelKind modelKind,
                                 const OptixDenoiserOptions* options,
                                 OptixDenoiser* denoiser );

/// Creates a denoiser object with the given options, using a provided inference model
///
/// 'userData' and 'userDataSizeInBytes' provide a user model for inference.
/// The memory passed in userData will be accessed only during the invocation of this function and
/// can be freed after it returns.
/// The user model must export only one weight set which determines both the model kind and the
/// required set of guide images.
///
/// \param[in] context
/// \param[in] userData
/// \param[in] userDataSizeInBytes
/// \param[out] denoiser
OptixResult optixDenoiserCreateWithUserModel( OptixDeviceContext context,
                                              const void* userData, size_t userDataSizeInBytes, OptixDenoiser* denoiser );

/// Destroys the denoiser object and any associated host resources.
OptixResult optixDenoiserDestroy( OptixDenoiser denoiser );

/// Computes the GPU memory resources required to execute the denoiser.
///
/// Memory for state and scratch buffers must be allocated with the sizes in 'returnSizes' and scratch memory
/// passed to optixDenoiserSetup, optixDenoiserInvoke,
/// optixDenoiserComputeIntensity and optixDenoiserComputeAverageColor.
/// For tiled denoising an overlap area ('overlapWindowSizeInPixels') must be added to each tile on all sides
/// which increases the amount of
/// memory needed to denoise a tile. In case of tiling use withOverlapScratchSizeInBytes for scratch memory size.
/// If only full resolution images are denoised, withoutOverlapScratchSizeInBytes can be used which is always
/// smaller than withOverlapScratchSizeInBytes.
///
/// 'outputWidth' and 'outputHeight' is the dimension of the image to be denoised (without overlap in case tiling
/// is being used).
/// 'outputWidth' and 'outputHeight' must be greater than or equal to the dimensions passed to optixDenoiserSetup.
///
/// \param[in] denoiser
/// \param[in] outputWidth
/// \param[in] outputHeight
/// \param[out] returnSizes
OptixResult optixDenoiserComputeMemoryResources( const OptixDenoiser denoiser,
                                                 unsigned int        outputWidth,
                                                 unsigned int        outputHeight,
                                                 OptixDenoiserSizes* returnSizes );

/// Initializes the state required by the denoiser.
///
/// 'inputWidth' and 'inputHeight' must include overlap on both sides of the image if tiling is being used. The overlap is
/// returned by #optixDenoiserComputeMemoryResources.
/// For subsequent calls to #optixDenoiserInvoke 'inputWidth' and 'inputHeight' are the maximum dimensions
/// of the input layers. Dimensions of the input layers passed to #optixDenoiserInvoke may be different in each
/// invocation however they always must be smaller than 'inputWidth' and 'inputHeight' passed to #optixDenoiserSetup.
///
/// \param[in] denoiser
/// \param[in] stream
/// \param[in] inputWidth
/// \param[in] inputHeight
/// \param[in] denoiserState
/// \param[in] denoiserStateSizeInBytes
/// \param[in] scratch
/// \param[in] scratchSizeInBytes
OptixResult optixDenoiserSetup( OptixDenoiser denoiser,
                                CUstream      stream,
                                unsigned int  inputWidth,
                                unsigned int  inputHeight,
                                CUdeviceptr   denoiserState,
                                size_t        denoiserStateSizeInBytes,
                                CUdeviceptr   scratch,
                                size_t        scratchSizeInBytes );

/// Invokes denoiser on a set of input data and produces at least one output image.
/// State memory must be available during the execution of the
/// denoiser (or until optixDenoiserSetup is called with a new state memory pointer).
/// Scratch memory passed is used only for the duration of this function.
/// Scratch and state memory sizes must have a size greater than or equal to the sizes as returned by
/// optixDenoiserComputeMemoryResources.
///
/// 'inputOffsetX' and 'inputOffsetY' are pixel offsets in the 'inputLayers' image
/// specifying the beginning of the image without overlap. When denoising an entire image without tiling
/// there is no overlap and 'inputOffsetX' and 'inputOffsetY' must be zero. When denoising a tile which is
/// adjacent to one of the four sides of the entire image the corresponding offsets must also be zero since
/// there is no overlap at the side adjacent to the image border.
///
/// 'guideLayer' provides additional information to the denoiser. When providing albedo and normal vector
/// guide images, the corresponding fields in the 'OptixDenoiserOptions' must be
/// enabled, see #optixDenoiserCreate.
/// 'guideLayer' must not be null. If a guide image in 'OptixDenoiserOptions' is not enabled, the
/// corresponding image in 'OptixDenoiserGuideLayer' is ignored.
///
/// If OPTIX_DENOISER_MODEL_KIND_TEMPORAL or OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV  is selected, a 2d flow
/// image must be given in 'OptixDenoiserGuideLayer'.
/// It describes for each pixel the flow from the previous to the current frame (a 2d vector in pixel space).
/// The denoised beauty/AOV of the previous frame must be given in 'previousOutput'.
/// If this image is not available in the first frame of a sequence, the noisy beauty/AOV from the first frame
/// and zero flow vectors could be given as a substitute.
/// For non-temporal model kinds the flow image in 'OptixDenoiserGuideLayer' is ignored.
/// 'previousOutput' and
/// 'output' may refer to the same buffer, i.e. 'previousOutput' is first read by this function and later
/// overwritten with the denoised result. 'output' can be passed as 'previousOutput' to the next frame.
/// In other model kinds (not temporal) 'previousOutput' is ignored.
///
/// The beauty layer must be given as the first entry in 'layers'.
/// In AOV type model kinds (OPTIX_DENOISER_MODEL_KIND_AOV or in user defined models implementing
/// kernel-prediction) additional layers for the AOV images can be given.
/// In each layer the noisy input image is given in 'input', the denoised output is written into the
/// 'output' image. input and output images may refer to the same buffer, with the restriction that
/// the pixel formats must be identical for input and output when the blend mode is selected (see
/// #OptixDenoiserParams).
///
/// If OPTIX_DENOISER_MODEL_KIND_TEMPORAL or OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV  is selected, the denoised
/// image from the previous frame must be given in 'previousOutput' in the layer. 'previousOutput' and
/// 'output' may refer to the same buffer, i.e. 'previousOutput' is first read by this function and later
/// overwritten with the denoised result. 'output' can be passed as 'previousOutput' to the next frame.
/// In other model kinds (not temporal) 'previousOutput' is ignored.
///
/// If OPTIX_DENOISER_MODEL_KIND_TEMPORAL or OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV  is selected, the
/// normal vector guide image must be given as 3d vectors in camera space. In the other models only
/// the x and y channels are used and other channels are ignored.
///
/// \param[in] denoiser
/// \param[in] stream
/// \param[in] params
/// \param[in] denoiserState
/// \param[in] denoiserStateSizeInBytes
/// \param[in] guideLayer
/// \param[in] layers
/// \param[in] numLayers
/// \param[in] inputOffsetX
/// \param[in] inputOffsetY
/// \param[in] scratch
/// \param[in] scratchSizeInBytes
OptixResult optixDenoiserInvoke( OptixDenoiser                   denoiser,
                                 CUstream                        stream,
                                 const OptixDenoiserParams*      params,
                                 CUdeviceptr                     denoiserState,
                                 size_t                          denoiserStateSizeInBytes,
                                 const OptixDenoiserGuideLayer*  guideLayer,
                                 const OptixDenoiserLayer*       layers,
                                 unsigned int                    numLayers,
                                 unsigned int                    inputOffsetX,
                                 unsigned int                    inputOffsetY,
                                 CUdeviceptr                     scratch,
                                 size_t                          scratchSizeInBytes );

/// Computes the logarithmic average intensity of the given image. The returned value 'outputIntensity'
/// is multiplied with the RGB values of the input image/tile in optixDenoiserInvoke if given in the parameter
/// OptixDenoiserParams::hdrIntensity (otherwise 'hdrIntensity' must be a null pointer). This is useful for
/// denoising HDR images which are very dark or bright.
/// When denoising tiles the intensity of the entire image should be computed, i.e. not per tile to get
/// consistent results.
///
/// For each RGB pixel in the inputImage the intensity is calculated and summed if it is greater than 1e-8f:
/// intensity = log(r * 0.212586f + g * 0.715170f + b * 0.072200f).
/// The function returns 0.18 / exp(sum of intensities / number of summed pixels).
/// More details could be found in the Reinhard tonemapping paper:
/// http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf
///
/// The size of scratch memory required can be queried with #optixDenoiserComputeMemoryResources.
///
/// data type unsigned char is not supported for 'inputImage', it must be 3 or 4 component half/float.
///
/// \param[in] denoiser
/// \param[in] stream
/// \param[in] inputImage
/// \param[out] outputIntensity    single float
/// \param[in] scratch
/// \param[in] scratchSizeInBytes
OptixResult optixDenoiserComputeIntensity( OptixDenoiser       denoiser,
                                           CUstream            stream,
                                           const OptixImage2D* inputImage,
                                           CUdeviceptr         outputIntensity,
                                           CUdeviceptr         scratch,
                                           size_t              scratchSizeInBytes );

/// Compute average logarithmic for each of the first three channels for the given image.
/// When denoising tiles the intensity of the entire image should be computed, i.e. not per tile to get
/// consistent results.
///
/// The size of scratch memory required can be queried with #optixDenoiserComputeMemoryResources.
///
/// data type unsigned char is not supported for 'inputImage', it must be 3 or 4 component half/float.
///
/// \param[in] denoiser
/// \param[in] stream
/// \param[in] inputImage
/// \param[out] outputAverageColor three floats
/// \param[in] scratch
/// \param[in] scratchSizeInBytes
OptixResult optixDenoiserComputeAverageColor( OptixDenoiser       denoiser,
                                              CUstream            stream,
                                              const OptixImage2D* inputImage,
                                              CUdeviceptr         outputAverageColor,
                                              CUdeviceptr         scratch,
                                              size_t              scratchSizeInBytes );

//@}

#ifdef __cplusplus
}
#endif

#include "optix_function_table.h"

#endif  // __optix_optix_7_host_h__
