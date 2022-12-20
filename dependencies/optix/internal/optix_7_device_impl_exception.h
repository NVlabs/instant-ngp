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

/**
* @file   optix_7_device_impl_exception.h
* @author NVIDIA Corporation
* @brief  OptiX public API
*
* OptiX public API Reference - Device side implementation for exception helper function.
*/

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_7_device_impl_exception.h is an internal header file and must not be used directly.  Please use optix_device.h or optix.h instead.")
#endif

#ifndef __optix_optix_7_device_impl_exception_h__
#define __optix_optix_7_device_impl_exception_h__

#if !defined(__CUDACC_RTC__)
#include <cstdio> /* for printf */
#endif

namespace optix_impl {

    static __forceinline__ __device__ void optixDumpStaticTransformFromHandle( OptixTraversableHandle handle )
    {
        const OptixStaticTransform* traversable = optixGetStaticTransformFromHandle( handle );
        if( traversable )
        {
            const uint3 index = optixGetLaunchIndex();
            printf( "(%4i,%4i,%4i)     OptixStaticTransform@%p = {\n"
                    "                       child        = %p,\n"
                    "                       transform    = { %f,%f,%f,%f,\n"
                    "                                        %f,%f,%f,%f,\n"
                    "                                        %f,%f,%f,%f } }\n",
                index.x,index.y,index.z,
                traversable,
                (void*)traversable->child,
                traversable->transform[0], traversable->transform[1], traversable->transform[2], traversable->transform[3],
                traversable->transform[4], traversable->transform[5], traversable->transform[6], traversable->transform[7],
                traversable->transform[8], traversable->transform[9], traversable->transform[10], traversable->transform[11] );
        }
    }

    static __forceinline__ __device__ void optixDumpMotionMatrixTransformFromHandle( OptixTraversableHandle handle )
    {
        const OptixMatrixMotionTransform* traversable =  optixGetMatrixMotionTransformFromHandle( handle );
        if( traversable )
        {
            const uint3 index = optixGetLaunchIndex();
            printf( "(%4i,%4i,%4i)     OptixMatrixMotionTransform@%p = {\n"
                    "                       child         = %p,\n"
                    "                       motionOptions = { numKeys = %i, flags = %i, timeBegin = %f, timeEnd = %f },\n"
                    "                       transform     = { { %f,%f,%f,%f,\n"
                    "                                           %f,%f,%f,%f,\n"
                    "                                           %f,%f,%f,%f }, ... }\n",
                index.x,index.y,index.z,
                traversable,
                (void*)traversable->child,
                (int)traversable->motionOptions.numKeys, (int)traversable->motionOptions.flags, traversable->motionOptions.timeBegin, traversable->motionOptions.timeEnd,
                traversable->transform[0][0], traversable->transform[0][1], traversable->transform[0][2],  traversable->transform[0][3],
                traversable->transform[0][4], traversable->transform[0][5], traversable->transform[0][6],  traversable->transform[0][7],
                traversable->transform[0][8], traversable->transform[0][9], traversable->transform[0][10], traversable->transform[0][11] );
        }
    }

    static __forceinline__ __device__ void optixDumpSrtMatrixTransformFromHandle( OptixTraversableHandle handle )
    {
        const OptixSRTMotionTransform* traversable =  optixGetSRTMotionTransformFromHandle( handle );
        if( traversable )
        {
            const uint3 index = optixGetLaunchIndex();
            printf( "(%4i,%4i,%4i)     OptixSRTMotionTransform@%p = {\n"
                    "                       child         = %p,\n"
                    "                       motionOptions = { numKeys = %i, flags = %i, timeBegin = %f, timeEnd = %f },\n"
                    "                       srtData       = { { sx  = %f,  a = %f,   b = %f, pvx = %f,\n"
                    "                                           sy  = %f,  c = %f, pvy = %f,  sz = %f,\n"
                    "                                           pvz = %f, qx = %f,  qy = %f,  qz = %f,\n"
                    "                                           qw  = %f, tx = %f,  ty = %f,  tz = %f }, ... }\n",
                index.x,index.y,index.z,
                traversable,
                (void*)traversable->child,
                (int)traversable->motionOptions.numKeys, (int)traversable->motionOptions.flags, traversable->motionOptions.timeBegin, traversable->motionOptions.timeEnd,
                traversable->srtData[0].sx, traversable->srtData[0].a, traversable->srtData[0].b,  traversable->srtData[0].pvx,
                traversable->srtData[0].sy, traversable->srtData[0].c, traversable->srtData[0].pvy,traversable->srtData[0].sz,
                traversable->srtData[0].pvz,traversable->srtData[0].qx,traversable->srtData[0].qy, traversable->srtData[0].qz,
                traversable->srtData[0].qw, traversable->srtData[0].tx,traversable->srtData[0].ty, traversable->srtData[0].tz );
        }
    }

    static __forceinline__ __device__ void optixDumpInstanceFromHandle( OptixTraversableHandle handle )
    {
        if( optixGetTransformTypeFromHandle( handle ) == OPTIX_TRANSFORM_TYPE_INSTANCE )
        {
            unsigned int instanceId = optixGetInstanceIdFromHandle( handle );
            const float4* transform = optixGetInstanceTransformFromHandle( handle );

            const uint3 index = optixGetLaunchIndex();
            printf( "(%4i,%4i,%4i)     OptixInstance = {\n"
                    "                       instanceId = %i,\n"
                    "                       transform  = { %f,%f,%f,%f,\n"
                    "                                      %f,%f,%f,%f,\n"
                    "                                      %f,%f,%f,%f } }\n",
                index.x,index.y,index.z,
                instanceId,
                transform[0].x, transform[0].y, transform[0].z,  transform[0].w,
                transform[1].x, transform[1].y, transform[1].z,  transform[1].w,
                transform[2].x, transform[2].y, transform[2].z,  transform[2].w );
        }
    }

    static __forceinline__ __device__ void optixDumpTransform( OptixTraversableHandle handle )
    {
        const OptixTransformType type = optixGetTransformTypeFromHandle( handle );
        const uint3 index = optixGetLaunchIndex();

        switch( type )
        {
            case OPTIX_TRANSFORM_TYPE_NONE:
                break;
            case OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM:
                optixDumpStaticTransformFromHandle( handle );
                break;
            case OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM:
                optixDumpMotionMatrixTransformFromHandle( handle );
                break;
            case OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM:
                optixDumpSrtMatrixTransformFromHandle( handle );
                break;
            case OPTIX_TRANSFORM_TYPE_INSTANCE:
                optixDumpInstanceFromHandle( handle );
                break;
            default:
                break;
        }
    }

    static __forceinline__ __device__ void optixDumpTransformList()
    {
        const int tlistSize = optixGetTransformListSize();
        const uint3 index = optixGetLaunchIndex();

        printf("(%4i,%4i,%4i) transform list of size %i:\n", index.x,index.y,index.z, tlistSize);

        for( unsigned int i = 0 ; i < tlistSize ; ++i )
        {
            OptixTraversableHandle handle = optixGetTransformListHandle( i );
            printf("(%4i,%4i,%4i)   transform[%i] = %p\n", index.x, index.y, index.z, i, (void*)handle);
            optixDumpTransform(handle);
        }
    }

    static __forceinline__ __device__ void optixDumpExceptionDetails()
    {
        bool dumpTlist = false;
        const int exceptionCode = optixGetExceptionCode();
        const uint3 index = optixGetLaunchIndex();

        if( exceptionCode == OPTIX_EXCEPTION_CODE_STACK_OVERFLOW )
        {
            printf("(%4i,%4i,%4i) error: stack overflow\n", index.x,index.y,index.z);
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED )
        {
            printf("(%4i,%4i,%4i) error: trace depth exceeded\n", index.x,index.y,index.z);
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED )
        {
            printf("(%4i,%4i,%4i) error: traversal depth exceeded\n", index.x,index.y,index.z);
            dumpTlist = true;
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE )
        {
            OptixTraversableHandle handle = optixGetExceptionInvalidTraversable();
            printf("(%4i,%4i,%4i) error: invalid traversable %p\n", index.x,index.y,index.z, (void*)handle);
            dumpTlist = true;
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT )
        {
            int sbtOffset = optixGetExceptionInvalidSbtOffset();
            printf("(%4i,%4i,%4i) error: invalid miss sbt of %i\n", index.x,index.y,index.z, sbtOffset);
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT )
        {
            int sbtOffset = optixGetExceptionInvalidSbtOffset();
            printf("(%4i,%4i,%4i) error: invalid hit sbt of %i at primitive with gas sbt index %i\n", index.x,index.y,index.z, sbtOffset, optixGetSbtGASIndex() );
            dumpTlist = true;
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE )
        {
            dumpTlist = true;
            printf( "(%4i,%4i,%4i) error: shader encountered unsupported builtin type\n"
                    "       call location:   %s\n", index.x, index.y, index.z, optixGetExceptionLineInfo() );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_INVALID_RAY )
        {
            OptixInvalidRayExceptionDetails ray = optixGetExceptionInvalidRay();
            printf( "(%4i,%4i,%4i) error: encountered an invalid ray:\n", index.x, index.y, index.z );
            printf(
                "       origin:          [%f, %f, %f]\n"
                "       direction:       [%f, %f, %f]\n"
                "       tmin:            %f\n"
                "       tmax:            %f\n"
                "       rayTime:         %f\n"
                "       call location:   %s\n",
                ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y,
                ray.direction.z, ray.tmin, ray.tmax, ray.time, optixGetExceptionLineInfo() );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH )
        {
             OptixParameterMismatchExceptionDetails details = optixGetExceptionParameterMismatch();
             printf( "(%4i,%4i,%4i) error: parameter mismatch in callable call.\n", index.x, index.y, index.z );
             printf(
                "       passed packed arguments:       %u 32 Bit values\n"
                "       expected packed parameters:    %u 32 Bit values\n"
                "       SBT index:                     %u\n"
                "       called function:               %s\n"
                "       call location:                 %s\n",
                details.passedArgumentCount, details.expectedParameterCount, details.sbtIndex,
                details.callableName, optixGetExceptionLineInfo() );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH )
        {
            dumpTlist = true;
            printf("(%4i,%4i,%4i) error: mismatch between builtin IS shader and build input\n"
                   "       call location:   %s\n", index.x,index.y,index.z, optixGetExceptionLineInfo() );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_CALLABLE_INVALID_SBT )
        {
            int sbtOffset = optixGetExceptionInvalidSbtOffset();
            printf( "(%4i,%4i,%4i) error: invalid sbt offset of %i for callable program\n", index.x, index.y, index.z, sbtOffset );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD )
        {
            int sbtOffset = optixGetExceptionInvalidSbtOffset();
            printf( "(%4i,%4i,%4i) error: invalid sbt offset of %i for direct callable program\n", index.x, index.y, index.z, sbtOffset );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD )
        {
            int sbtOffset = optixGetExceptionInvalidSbtOffset();
            printf( "(%4i,%4i,%4i) error: invalid sbt offset of %i for continuation callable program\n", index.x, index.y, index.z, sbtOffset );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS )
        {
            OptixTraversableHandle handle = optixGetExceptionInvalidTraversable();
            printf("(%4i,%4i,%4i) error: unsupported single GAS traversable graph %p\n", index.x,index.y,index.z, (void*)handle);
            dumpTlist = true;
        }
        else if( ( exceptionCode <= OPTIX_EXCEPTION_CODE_INVALID_VALUE_ARGUMENT_0 ) && ( exceptionCode >= OPTIX_EXCEPTION_CODE_INVALID_VALUE_ARGUMENT_2 ) )
        {
            printf("(%4i,%4i,%4i) error: invalid value for argument %i\n", index.x,index.y,index.z, -(exceptionCode - OPTIX_EXCEPTION_CODE_INVALID_VALUE_ARGUMENT_0) );
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_UNSUPPORTED_DATA_ACCESS )
        {
            printf("(%4i,%4i,%4i) error: unsupported random data access\n", index.x,index.y,index.z);
        }
        else if( exceptionCode == OPTIX_EXCEPTION_CODE_PAYLOAD_TYPE_MISMATCH )
        {
            printf("(%4i,%4i,%4i) error: payload type mismatch between program and optixTrace call\n", index.x,index.y,index.z);
        }
        else if( exceptionCode >= 0 )
        {
            dumpTlist = true;
            printf( "(%4i,%4i,%4i) error: user exception with error code %i\n"
                    "       call location:   %s\n", index.x, index.y, index.z, exceptionCode, optixGetExceptionLineInfo() );
        }
        else
        {
            printf("(%4i,%4i,%4i) error: unknown exception with error code %i\n", index.x,index.y,index.z, exceptionCode);
        }

        if( dumpTlist )
            optixDumpTransformList();
    }

}  // namespace optix_impl

#endif
