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
* @file   optix_7_device_impl.h
* @author NVIDIA Corporation
* @brief  OptiX public API
*
* OptiX public API Reference - Device side implementation
*/

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_7_device_impl.h is an internal header file and must not be used directly.  Please use optix_device.h or optix.h instead.")
#endif

#ifndef __optix_optix_7_device_impl_h__
#define __optix_optix_7_device_impl_h__

#include "internal/optix_7_device_impl_exception.h"
#include "internal/optix_7_device_impl_transformations.h"

#ifndef __CUDACC_RTC__
#include <initializer_list>
#include <type_traits>
#endif

namespace optix_internal {
template <typename...>
struct TypePack{};
}  // namespace optix_internal

template <typename... Payload>
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   Payload&...            payload )
{
    static_assert( sizeof...( Payload ) <= 32, "Only up to 32 payload values are allowed." );
    // std::is_same compares each type in the two TypePacks to make sure that all types are unsigned int.
    // TypePack 1    unsigned int    T0      T1      T2   ...   Tn-1        Tn
    // TypePack 2      T0            T1      T2      T3   ...   Tn        unsigned int
#ifndef __CUDACC_RTC__
    static_assert( std::is_same<optix_internal::TypePack<unsigned int, Payload...>, optix_internal::TypePack<Payload..., unsigned int>>::value,
                   "All payload parameters need to be unsigned int." );
#endif

    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p[33]       = { 0, payload... };
    int          payloadSize = (int)sizeof...( Payload );
    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%"
        "29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%"
        "59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"( p[1] ), "=r"( p[2] ), "=r"( p[3] ), "=r"( p[4] ), "=r"( p[5] ), "=r"( p[6] ), "=r"( p[7] ),
          "=r"( p[8] ), "=r"( p[9] ), "=r"( p[10] ), "=r"( p[11] ), "=r"( p[12] ), "=r"( p[13] ), "=r"( p[14] ),
          "=r"( p[15] ), "=r"( p[16] ), "=r"( p[17] ), "=r"( p[18] ), "=r"( p[19] ), "=r"( p[20] ), "=r"( p[21] ),
          "=r"( p[22] ), "=r"( p[23] ), "=r"( p[24] ), "=r"( p[25] ), "=r"( p[26] ), "=r"( p[27] ), "=r"( p[28] ),
          "=r"( p[29] ), "=r"( p[30] ), "=r"( p[31] ), "=r"( p[32] )
        : "r"( 0 ), "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ),
          "f"( tmax ), "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( payloadSize ), "r"( p[1] ), "r"( p[2] ), "r"( p[3] ), "r"( p[4] ), "r"( p[5] ),
          "r"( p[6] ), "r"( p[7] ), "r"( p[8] ), "r"( p[9] ), "r"( p[10] ), "r"( p[11] ), "r"( p[12] ), "r"( p[13] ),
          "r"( p[14] ), "r"( p[15] ), "r"( p[16] ), "r"( p[17] ), "r"( p[18] ), "r"( p[19] ), "r"( p[20] ),
          "r"( p[21] ), "r"( p[22] ), "r"( p[23] ), "r"( p[24] ), "r"( p[25] ), "r"( p[26] ), "r"( p[27] ),
          "r"( p[28] ), "r"( p[29] ), "r"( p[30] ), "r"( p[31] ), "r"( p[32] )
        : );
    unsigned int index = 1;
    (void)std::initializer_list<unsigned int>{ index, ( payload = p[index++] )... };
}

template <typename... Payload>
static __forceinline__ __device__ void optixTrace( OptixPayloadTypeID     type,
                                                   OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   Payload&...            payload )
{
    // std::is_same compares each type in the two TypePacks to make sure that all types are unsigned int.
    // TypePack 1    unsigned int    T0      T1      T2   ...   Tn-1        Tn
    // TypePack 2      T0            T1      T2      T3   ...   Tn        unsigned int
    static_assert( sizeof...( Payload ) <= 32, "Only up to 32 payload values are allowed." );
    static_assert( std::is_same<optix_internal::TypePack<unsigned int, Payload...>, optix_internal::TypePack<Payload..., unsigned int>>::value,
                   "All payload parameters need to be unsigned int." );

    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p[33]       = { 0, payload... };
    int          payloadSize = (int)sizeof...( Payload );

    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%"
        "29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%"
        "59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"( p[1] ), "=r"( p[2] ), "=r"( p[3] ), "=r"( p[4] ), "=r"( p[5] ), "=r"( p[6] ), "=r"( p[7] ),
          "=r"( p[8] ), "=r"( p[9] ), "=r"( p[10] ), "=r"( p[11] ), "=r"( p[12] ), "=r"( p[13] ), "=r"( p[14] ),
          "=r"( p[15] ), "=r"( p[16] ), "=r"( p[17] ), "=r"( p[18] ), "=r"( p[19] ), "=r"( p[20] ), "=r"( p[21] ),
          "=r"( p[22] ), "=r"( p[23] ), "=r"( p[24] ), "=r"( p[25] ), "=r"( p[26] ), "=r"( p[27] ), "=r"( p[28] ),
          "=r"( p[29] ), "=r"( p[30] ), "=r"( p[31] ), "=r"( p[32] )
        : "r"( type ), "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ),
          "f"( tmax ), "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( payloadSize ), "r"( p[1] ), "r"( p[2] ), "r"( p[3] ), "r"( p[4] ), "r"( p[5] ),
          "r"( p[6] ), "r"( p[7] ), "r"( p[8] ), "r"( p[9] ), "r"( p[10] ), "r"( p[11] ), "r"( p[12] ), "r"( p[13] ),
          "r"( p[14] ), "r"( p[15] ), "r"( p[16] ), "r"( p[17] ), "r"( p[18] ), "r"( p[19] ), "r"( p[20] ),
          "r"( p[21] ), "r"( p[22] ), "r"( p[23] ), "r"( p[24] ), "r"( p[25] ), "r"( p[26] ), "r"( p[27] ),
          "r"( p[28] ), "r"( p[29] ), "r"( p[30] ), "r"( p[31] ), "r"( p[32] )
        : );
    unsigned int index = 1;
    (void)std::initializer_list<unsigned int>{ index, ( payload = p[index++] )... };
}

static __forceinline__ __device__ void optixSetPayload_0( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 0 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_1( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 1 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_2( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 2 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_3( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 3 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_4( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 4 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_5( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 5 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_6( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 6 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_7( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 7 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_8( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 8 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_9( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 9 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_10( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 10 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_11( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 11 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_12( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 12 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_13( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 13 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_14( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 14 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_15( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 15 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_16( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 16 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_17( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 17 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_18( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 18 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_19( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 19 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_20( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 20 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_21( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 21 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_22( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 22 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_23( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 23 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_24( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 24 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_25( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 25 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_26( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 26 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_27( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 27 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_28( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 28 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_29( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 29 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_30( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 30 ), "r"( p ) : );
}

static __forceinline__ __device__ void optixSetPayload_31( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 31 ), "r"( p ) : );
}

static __forceinline__ __device__ unsigned int optixGetPayload_0()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 0 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_1()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 1 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_2()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 2 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_3()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 3 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_4()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 4 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_5()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 5 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_6()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 6 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_7()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 7 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_8()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 8 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_9()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 9 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_10()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 10 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_11()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 11 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_12()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 12 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_13()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 13 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_14()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 14 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_15()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 15 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_16()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 16 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_17()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 17 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_18()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 18 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_19()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 19 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_20()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 20 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_21()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 21 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_22()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 22 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_23()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 23 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_24()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 24 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_25()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 25 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_26()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 26 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_27()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 27 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_28()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 28 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_29()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 29 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_30()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 30 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_31()
{
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 31 ) : );
    return result;
}

static __forceinline__ __device__ void optixSetPayloadTypes( unsigned int types )
{
    asm volatile( "call _optix_set_payload_types, (%0);" : : "r"( types ) : );
}

static __forceinline__ __device__ unsigned int optixUndefinedValue()
{
    unsigned int u0;
    asm( "call (%0), _optix_undef_value, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ float3 optixGetWorldRayOrigin()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_world_ray_origin_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_world_ray_origin_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_world_ray_origin_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float3 optixGetWorldRayDirection()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_world_ray_direction_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_world_ray_direction_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_world_ray_direction_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float3 optixGetObjectRayOrigin()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_object_ray_origin_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_object_ray_origin_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_object_ray_origin_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float3 optixGetObjectRayDirection()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_object_ray_direction_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_object_ray_direction_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_object_ray_direction_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float optixGetRayTmin()
{
    float f0;
    asm( "call (%0), _optix_get_ray_tmin, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ float optixGetRayTmax()
{
    float f0;
    asm( "call (%0), _optix_get_ray_tmax, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ float optixGetRayTime()
{
    float f0;
    asm( "call (%0), _optix_get_ray_time, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ unsigned int optixGetRayFlags()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_ray_flags, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetRayVisibilityMask()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_ray_visibility_mask, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ OptixTraversableHandle optixGetInstanceTraversableFromIAS( OptixTraversableHandle ias,
                                                                                             unsigned int           instIdx )
{
    unsigned long long handle;
    asm( "call (%0), _optix_get_instance_traversable_from_ias, (%1, %2);"
         : "=l"( handle ) : "l"( ias ), "r"( instIdx ) );
    return (OptixTraversableHandle)handle;
}


static __forceinline__ __device__ void optixGetTriangleVertexData( OptixTraversableHandle gas,
                                                                   unsigned int           primIdx,
                                                                   unsigned int           sbtGASIndex,
                                                                   float                  time,
                                                                   float3                 data[3] )
{
    asm( "call (%0, %1, %2, %3, %4, %5, %6, %7, %8), _optix_get_triangle_vertex_data, "
         "(%9, %10, %11, %12);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[1].x ), "=f"( data[1].y ),
           "=f"( data[1].z ), "=f"( data[2].x ), "=f"( data[2].y ), "=f"( data[2].z )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}


static __forceinline__ __device__ void optixGetLinearCurveVertexData( OptixTraversableHandle gas,
                                                                      unsigned int           primIdx,
                                                                      unsigned int           sbtGASIndex,
                                                                      float                  time,
                                                                      float4                 data[2] )
{
    asm( "call (%0, %1, %2, %3,  %4, %5, %6, %7), _optix_get_linear_curve_vertex_data, "
         "(%8, %9, %10, %11);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w ),
           "=f"( data[1].x ), "=f"( data[1].y ), "=f"( data[1].z ), "=f"( data[1].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ void optixGetQuadraticBSplineVertexData( OptixTraversableHandle gas,
                                                                           unsigned int         primIdx,
                                                                           unsigned int         sbtGASIndex,
                                                                           float                time,
                                                                           float4               data[3] )
{
    asm( "call (%0, %1, %2, %3,  %4, %5, %6, %7,  %8, %9, %10, %11), _optix_get_quadratic_bspline_vertex_data, "
         "(%12, %13, %14, %15);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w ), 
           "=f"( data[1].x ), "=f"( data[1].y ), "=f"( data[1].z ), "=f"( data[1].w ),
           "=f"( data[2].x ), "=f"( data[2].y ), "=f"( data[2].z ), "=f"( data[2].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ void optixGetCubicBSplineVertexData( OptixTraversableHandle gas,
                                                                       unsigned int         primIdx,
                                                                       unsigned int         sbtGASIndex,
                                                                       float                time,
                                                                       float4               data[4] )
{
    asm( "call (%0, %1, %2, %3,  %4, %5, %6, %7,  %8, %9, %10, %11,  %12, %13, %14, %15), "
         "_optix_get_cubic_bspline_vertex_data, "
         "(%16, %17, %18, %19);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w ), 
           "=f"( data[1].x ), "=f"( data[1].y ), "=f"( data[1].z ), "=f"( data[1].w ),
           "=f"( data[2].x ), "=f"( data[2].y ), "=f"( data[2].z ), "=f"( data[2].w ),
           "=f"( data[3].x ), "=f"( data[3].y ), "=f"( data[3].z ), "=f"( data[3].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ void optixGetCatmullRomVertexData( OptixTraversableHandle gas,
                                                                     unsigned int           primIdx,
                                                                     unsigned int           sbtGASIndex,
                                                                     float                  time,
                                                                     float4                 data[4] )
{
    asm( "call (%0, %1, %2, %3,  %4, %5, %6, %7,  %8, %9, %10, %11,  %12, %13, %14, %15), "
         "_optix_get_catmullrom_vertex_data, "
         "(%16, %17, %18, %19);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w ), "=f"( data[1].x ),
           "=f"( data[1].y ), "=f"( data[1].z ), "=f"( data[1].w ), "=f"( data[2].x ), "=f"( data[2].y ),
           "=f"( data[2].z ), "=f"( data[2].w ), "=f"( data[3].x ), "=f"( data[3].y ), "=f"( data[3].z ), "=f"( data[3].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ void optixGetSphereData( OptixTraversableHandle gas,
                                                           unsigned int           primIdx,
                                                           unsigned int           sbtGASIndex,
                                                           float                  time,
                                                           float4                 data[1] )
{
    asm( "call (%0, %1, %2, %3), "
         "_optix_get_sphere_data, "
         "(%4, %5, %6, %7);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ OptixTraversableHandle optixGetGASTraversableHandle()
{
    unsigned long long handle;
    asm( "call (%0), _optix_get_gas_traversable_handle, ();" : "=l"( handle ) : );
    return (OptixTraversableHandle)handle;
}

static __forceinline__ __device__ float optixGetGASMotionTimeBegin( OptixTraversableHandle handle )
{
    float f0;
    asm( "call (%0), _optix_get_gas_motion_time_begin, (%1);" : "=f"( f0 ) : "l"( handle ) : );
    return f0;
}

static __forceinline__ __device__ float optixGetGASMotionTimeEnd( OptixTraversableHandle handle )
{
    float f0;
    asm( "call (%0), _optix_get_gas_motion_time_end, (%1);" : "=f"( f0 ) : "l"( handle ) : );
    return f0;
}

static __forceinline__ __device__ unsigned int optixGetGASMotionStepCount( OptixTraversableHandle handle )
{
    unsigned int u0;
    asm( "call (%0), _optix_get_gas_motion_step_count, (%1);" : "=r"( u0 ) : "l"( handle ) : );
    return u0;
}

static __forceinline__ __device__ void optixGetWorldToObjectTransformMatrix( float m[12] )
{
    if( optixGetTransformListSize() == 0 )
    {
        m[0]  = 1.0f;
        m[1]  = 0.0f;
        m[2]  = 0.0f;
        m[3]  = 0.0f;
        m[4]  = 0.0f;
        m[5]  = 1.0f;
        m[6]  = 0.0f;
        m[7]  = 0.0f;
        m[8]  = 0.0f;
        m[9]  = 0.0f;
        m[10] = 1.0f;
        m[11] = 0.0f;
        return;
    }

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );
    m[0]  = m0.x;
    m[1]  = m0.y;
    m[2]  = m0.z;
    m[3]  = m0.w;
    m[4]  = m1.x;
    m[5]  = m1.y;
    m[6]  = m1.z;
    m[7]  = m1.w;
    m[8]  = m2.x;
    m[9]  = m2.y;
    m[10] = m2.z;
    m[11] = m2.w;
}

static __forceinline__ __device__ void optixGetObjectToWorldTransformMatrix( float m[12] )
{
    if( optixGetTransformListSize() == 0 )
    {
        m[0]  = 1.0f;
        m[1]  = 0.0f;
        m[2]  = 0.0f;
        m[3]  = 0.0f;
        m[4]  = 0.0f;
        m[5]  = 1.0f;
        m[6]  = 0.0f;
        m[7]  = 0.0f;
        m[8]  = 0.0f;
        m[9]  = 0.0f;
        m[10] = 1.0f;
        m[11] = 0.0f;
        return;
    }

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );
    m[0]  = m0.x;
    m[1]  = m0.y;
    m[2]  = m0.z;
    m[3]  = m0.w;
    m[4]  = m1.x;
    m[5]  = m1.y;
    m[6]  = m1.z;
    m[7]  = m1.w;
    m[8]  = m2.x;
    m[9]  = m2.y;
    m[10] = m2.z;
    m[11] = m2.w;
}

static __forceinline__ __device__ float3 optixTransformPointFromWorldToObjectSpace( float3 point )
{
    if( optixGetTransformListSize() == 0 )
        return point;

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformPoint( m0, m1, m2, point );
}

static __forceinline__ __device__ float3 optixTransformVectorFromWorldToObjectSpace( float3 vec )
{
    if( optixGetTransformListSize() == 0 )
        return vec;

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformVector( m0, m1, m2, vec );
}

static __forceinline__ __device__ float3 optixTransformNormalFromWorldToObjectSpace( float3 normal )
{
    if( optixGetTransformListSize() == 0 )
        return normal;

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );  // inverse of optixGetWorldToObjectTransformMatrix()
    return optix_impl::optixTransformNormal( m0, m1, m2, normal );
}

static __forceinline__ __device__ float3 optixTransformPointFromObjectToWorldSpace( float3 point )
{
    if( optixGetTransformListSize() == 0 )
        return point;

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformPoint( m0, m1, m2, point );
}

static __forceinline__ __device__ float3 optixTransformVectorFromObjectToWorldSpace( float3 vec )
{
    if( optixGetTransformListSize() == 0 )
        return vec;

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformVector( m0, m1, m2, vec );
}

static __forceinline__ __device__ float3 optixTransformNormalFromObjectToWorldSpace( float3 normal )
{
    if( optixGetTransformListSize() == 0 )
        return normal;

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );  // inverse of optixGetObjectToWorldTransformMatrix()
    return optix_impl::optixTransformNormal( m0, m1, m2, normal );
}

static __forceinline__ __device__ unsigned int optixGetTransformListSize()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_transform_list_size, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ OptixTraversableHandle optixGetTransformListHandle( unsigned int index )
{
    unsigned long long u0;
    asm( "call (%0), _optix_get_transform_list_handle, (%1);" : "=l"( u0 ) : "r"( index ) : );
    return u0;
}

static __forceinline__ __device__ OptixTransformType optixGetTransformTypeFromHandle( OptixTraversableHandle handle )
{
    int i0;
    asm( "call (%0), _optix_get_transform_type_from_handle, (%1);" : "=r"( i0 ) : "l"( handle ) : );
    return (OptixTransformType)i0;
}

static __forceinline__ __device__ const OptixStaticTransform* optixGetStaticTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_static_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const OptixStaticTransform*)ptr;
}

static __forceinline__ __device__ const OptixSRTMotionTransform* optixGetSRTMotionTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_srt_motion_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const OptixSRTMotionTransform*)ptr;
}

static __forceinline__ __device__ const OptixMatrixMotionTransform* optixGetMatrixMotionTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_matrix_motion_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const OptixMatrixMotionTransform*)ptr;
}

static __forceinline__ __device__ unsigned int optixGetInstanceIdFromHandle( OptixTraversableHandle handle )
{
    int i0;
    asm( "call (%0), _optix_get_instance_id_from_handle, (%1);" : "=r"( i0 ) : "l"( handle ) : );
    return i0;
}

static __forceinline__ __device__ OptixTraversableHandle optixGetInstanceChildFromHandle( OptixTraversableHandle handle )
{
    unsigned long long i0;
    asm( "call (%0), _optix_get_instance_child_from_handle, (%1);" : "=l"( i0 ) : "l"( handle ) : );
    return (OptixTraversableHandle)i0;
}

static __forceinline__ __device__ const float4* optixGetInstanceTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_instance_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const float4*)ptr;
}

static __forceinline__ __device__ const float4* optixGetInstanceInverseTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_instance_inverse_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const float4*)ptr;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_0"
        ", (%1, %2);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_1"
        ", (%1, %2, %3);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0, unsigned int a1 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_2"
        ", (%1, %2, %3, %4);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0, unsigned int a1, unsigned int a2 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_3"
        ", (%1, %2, %3, %4, %5);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_4"
        ", (%1, %2, %3, %4, %5, %6);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_5"
        ", (%1, %2, %3, %4, %5, %6, %7);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_6"
        ", (%1, %2, %3, %4, %5, %6, %7, %8);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 ), "r"( a5 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5,
                                                                unsigned int a6 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_7"
        ", (%1, %2, %3, %4, %5, %6, %7, %8, %9);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 ), "r"( a5 ), "r"( a6 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5,
                                                                unsigned int a6,
                                                                unsigned int a7 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_8"
        ", (%1, %2, %3, %4, %5, %6, %7, %8, %9, %10);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 ), "r"( a5 ), "r"( a6 ), "r"( a7 )
        : );
    return ret;
}

#define OPTIX_DEFINE_optixGetAttribute_BODY( which )                                                                   \
    unsigned int ret;                                                                                                  \
    asm( "call (%0), _optix_get_attribute_" #which ", ();" : "=r"( ret ) : );                                          \
    return ret;

static __forceinline__ __device__ unsigned int optixGetAttribute_0()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 0 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_1()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 1 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_2()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 2 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_3()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 3 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_4()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 4 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_5()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 5 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_6()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 6 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_7()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 7 );
}

#undef OPTIX_DEFINE_optixGetAttribute_BODY

static __forceinline__ __device__ void optixTerminateRay()
{
    asm volatile( "call _optix_terminate_ray, ();" );
}

static __forceinline__ __device__ void optixIgnoreIntersection()
{
    asm volatile( "call _optix_ignore_intersection, ();" );
}

static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_primitive_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetSbtGASIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_sbt_gas_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetInstanceId()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_instance_id, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetInstanceIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_instance_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetHitKind()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_hit_kind, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType(unsigned int hitKind)
{
    unsigned int u0;
    asm( "call (%0), _optix_get_primitive_type_from_hit_kind, (%1);" : "=r"( u0 ) : "r"( hitKind ) );
    return (OptixPrimitiveType)u0;
}

static __forceinline__ __device__ bool optixIsBackFaceHit( unsigned int hitKind )
{
    unsigned int u0;
    asm( "call (%0), _optix_get_backface_from_hit_kind, (%1);" : "=r"( u0 ) : "r"( hitKind ) );
    return (u0 == 0x1);
}

static __forceinline__ __device__ bool optixIsFrontFaceHit( unsigned int hitKind )
{
    return !optixIsBackFaceHit( hitKind );
}


static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType()
{
    return optixGetPrimitiveType( optixGetHitKind() );
}

static __forceinline__ __device__ bool optixIsBackFaceHit()
{
    return optixIsBackFaceHit( optixGetHitKind() );
}

static __forceinline__ __device__ bool optixIsFrontFaceHit()
{
    return optixIsFrontFaceHit( optixGetHitKind() );
}

static __forceinline__ __device__ bool optixIsTriangleHit()
{
    return optixIsTriangleFrontFaceHit() || optixIsTriangleBackFaceHit();
}

static __forceinline__ __device__ bool optixIsTriangleFrontFaceHit()
{
    return optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE;
}

static __forceinline__ __device__ bool optixIsTriangleBackFaceHit()
{
    return optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE;
}

static __forceinline__ __device__ float optixGetCurveParameter()
{
    return __int_as_float( optixGetAttribute_0() );
}

static __forceinline__ __device__ float2 optixGetTriangleBarycentrics()
{
    float f0, f1;
    asm( "call (%0, %1), _optix_get_triangle_barycentrics, ();" : "=f"( f0 ), "=f"( f1 ) : );
    return make_float2( f0, f1 );
}

static __forceinline__ __device__ uint3 optixGetLaunchIndex()
{
    unsigned int u0, u1, u2;
    asm( "call (%0), _optix_get_launch_index_x, ();" : "=r"( u0 ) : );
    asm( "call (%0), _optix_get_launch_index_y, ();" : "=r"( u1 ) : );
    asm( "call (%0), _optix_get_launch_index_z, ();" : "=r"( u2 ) : );
    return make_uint3( u0, u1, u2 );
}

static __forceinline__ __device__ uint3 optixGetLaunchDimensions()
{
    unsigned int u0, u1, u2;
    asm( "call (%0), _optix_get_launch_dimension_x, ();" : "=r"( u0 ) : );
    asm( "call (%0), _optix_get_launch_dimension_y, ();" : "=r"( u1 ) : );
    asm( "call (%0), _optix_get_launch_dimension_z, ();" : "=r"( u2 ) : );
    return make_uint3( u0, u1, u2 );
}

static __forceinline__ __device__ CUdeviceptr optixGetSbtDataPointer()
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_sbt_data_ptr_64, ();" : "=l"( ptr ) : );
    return (CUdeviceptr)ptr;
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode )
{
    asm volatile(
        "call _optix_throw_exception_0, (%0);"
        : /* no return value */
        : "r"( exceptionCode )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0 )
{
    asm volatile(
        "call _optix_throw_exception_1, (%0, %1);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1 )
{
    asm volatile(
        "call _optix_throw_exception_2, (%0, %1, %2);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2 )
{
    asm volatile(
        "call _optix_throw_exception_3, (%0, %1, %2, %3);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3 )
{
    asm volatile(
        "call _optix_throw_exception_4, (%0, %1, %2, %3, %4);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4 )
{
    asm volatile(
        "call _optix_throw_exception_5, (%0, %1, %2, %3, %4, %5);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4, unsigned int exceptionDetail5 )
{
    asm volatile(
        "call _optix_throw_exception_6, (%0, %1, %2, %3, %4, %5, %6);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 ), "r"( exceptionDetail5 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4, unsigned int exceptionDetail5, unsigned int exceptionDetail6 )
{
    asm volatile(
        "call _optix_throw_exception_7, (%0, %1, %2, %3, %4, %5, %6, %7);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 ), "r"( exceptionDetail5 ), "r"( exceptionDetail6 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4, unsigned int exceptionDetail5, unsigned int exceptionDetail6, unsigned int exceptionDetail7 )
{
    asm volatile(
        "call _optix_throw_exception_8, (%0, %1, %2, %3, %4, %5, %6, %7, %8);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 ), "r"( exceptionDetail5 ), "r"( exceptionDetail6 ), "r"( exceptionDetail7 )
        : );
}

static __forceinline__ __device__ int optixGetExceptionCode()
{
    int s0;
    asm( "call (%0), _optix_get_exception_code, ();" : "=r"( s0 ) : );
    return s0;
}

#define OPTIX_DEFINE_optixGetExceptionDetail_BODY( which )                                                             \
    unsigned int ret;                                                                                                  \
    asm( "call (%0), _optix_get_exception_detail_" #which ", ();" : "=r"( ret ) : );                                   \
    return ret;

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_0()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 0 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_1()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 1 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_2()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 2 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_3()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 3 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_4()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 4 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_5()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 5 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_6()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 6 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_7()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 7 );
}

#undef OPTIX_DEFINE_optixGetExceptionDetail_BODY

static __forceinline__ __device__ OptixTraversableHandle optixGetExceptionInvalidTraversable()
{
    unsigned long long handle;
    asm( "call (%0), _optix_get_exception_invalid_traversable, ();" : "=l"( handle ) : );
    return (OptixTraversableHandle)handle;
}

static __forceinline__ __device__ int optixGetExceptionInvalidSbtOffset()
{
    int s0;
    asm( "call (%0), _optix_get_exception_invalid_sbt_offset, ();" : "=r"( s0 ) : );
    return s0;
}

static __forceinline__ __device__ OptixInvalidRayExceptionDetails optixGetExceptionInvalidRay()
{
    float rayOriginX, rayOriginY, rayOriginZ, rayDirectionX, rayDirectionY, rayDirectionZ, tmin, tmax, rayTime;
    asm( "call (%0, %1, %2, %3, %4, %5, %6, %7, %8), _optix_get_exception_invalid_ray, ();"
         : "=f"( rayOriginX ), "=f"( rayOriginY ), "=f"( rayOriginZ ), "=f"( rayDirectionX ), "=f"( rayDirectionY ),
           "=f"( rayDirectionZ ), "=f"( tmin ), "=f"( tmax ), "=f"( rayTime )
         : );
    OptixInvalidRayExceptionDetails ray;
    ray.origin    = make_float3( rayOriginX, rayOriginY, rayOriginZ );
    ray.direction = make_float3( rayDirectionX, rayDirectionY, rayDirectionZ );
    ray.tmin      = tmin;
    ray.tmax      = tmax;
    ray.time      = rayTime;
    return ray;
}

static __forceinline__ __device__ OptixParameterMismatchExceptionDetails optixGetExceptionParameterMismatch()
{
    unsigned int expected, actual, sbtIdx;
    unsigned long long calleeName;
    asm(
        "call (%0, %1, %2, %3), _optix_get_exception_parameter_mismatch, ();"
        : "=r"(expected), "=r"(actual), "=r"(sbtIdx), "=l"(calleeName) : );
    OptixParameterMismatchExceptionDetails details;
    details.expectedParameterCount = expected;
    details.passedArgumentCount = actual;
    details.sbtIndex = sbtIdx;
    details.callableName = (char*)calleeName;
    return details;
}

static __forceinline__ __device__ char* optixGetExceptionLineInfo()
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_exception_line_info, ();" : "=l"(ptr) : );
    return (char*)ptr;
}

template <typename ReturnT, typename... ArgTypes>
static __forceinline__ __device__ ReturnT optixDirectCall( unsigned int sbtIndex, ArgTypes... args )
{
    unsigned long long func;
    asm( "call (%0), _optix_call_direct_callable,(%1);" : "=l"( func ) : "r"( sbtIndex ) : );
    using funcT = ReturnT ( * )( ArgTypes... );
    funcT call  = ( funcT )( func );
    return call( args... );
}

template <typename ReturnT, typename... ArgTypes>
static __forceinline__ __device__ ReturnT optixContinuationCall( unsigned int sbtIndex, ArgTypes... args )
{
    unsigned long long func;
    asm( "call (%0), _optix_call_continuation_callable,(%1);" : "=l"( func ) : "r"( sbtIndex ) : );
    using funcT = ReturnT ( * )( ArgTypes... );
    funcT call  = ( funcT )( func );
    return call( args... );
}
#endif

static __forceinline__ __device__ uint4 optixTexFootprint2D( unsigned long long tex, unsigned int texInfo, float x, float y, unsigned int* singleMipLevel )
{
    uint4              result;
    unsigned long long resultPtr         = reinterpret_cast<unsigned long long>( &result );
    unsigned long long singleMipLevelPtr = reinterpret_cast<unsigned long long>( singleMipLevel );
    // Cast float args to integers, because the intrinics take .b32 arguments when compiled to PTX.
    asm volatile(
        "call _optix_tex_footprint_2d_v2"
        ", (%0, %1, %2, %3, %4, %5);"
        :
        : "l"( tex ), "r"( texInfo ), "r"( __float_as_uint( x ) ), "r"( __float_as_uint( y ) ),
          "l"( singleMipLevelPtr ), "l"( resultPtr )
        : );
    return result;
}

static __forceinline__ __device__ uint4 optixTexFootprint2DGrad( unsigned long long tex,
                                                                 unsigned int       texInfo,
                                                                 float              x,
                                                                 float              y,
                                                                 float              dPdx_x,
                                                                 float              dPdx_y,
                                                                 float              dPdy_x,
                                                                 float              dPdy_y,
                                                                 bool               coarse,
                                                                 unsigned int*      singleMipLevel )
{
    uint4              result;
    unsigned long long resultPtr         = reinterpret_cast<unsigned long long>( &result );
    unsigned long long singleMipLevelPtr = reinterpret_cast<unsigned long long>( singleMipLevel );
    // Cast float args to integers, because the intrinics take .b32 arguments when compiled to PTX.
    asm volatile(
        "call _optix_tex_footprint_2d_grad_v2"
        ", (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10);"
        :
        : "l"( tex ), "r"( texInfo ), "r"( __float_as_uint( x ) ), "r"( __float_as_uint( y ) ),
          "r"( __float_as_uint( dPdx_x ) ), "r"( __float_as_uint( dPdx_y ) ), "r"( __float_as_uint( dPdy_x ) ),
          "r"( __float_as_uint( dPdy_y ) ), "r"( static_cast<unsigned int>( coarse ) ), "l"( singleMipLevelPtr ), "l"( resultPtr )
        : );

    return result;
}

static __forceinline__ __device__ uint4
optixTexFootprint2DLod( unsigned long long tex, unsigned int texInfo, float x, float y, float level, bool coarse, unsigned int* singleMipLevel )
{
    uint4              result;
    unsigned long long resultPtr         = reinterpret_cast<unsigned long long>( &result );
    unsigned long long singleMipLevelPtr = reinterpret_cast<unsigned long long>( singleMipLevel );
    // Cast float args to integers, because the intrinics take .b32 arguments when compiled to PTX.
    asm volatile(
        "call _optix_tex_footprint_2d_lod_v2"
        ", (%0, %1, %2, %3, %4, %5, %6, %7);"
        :
        : "l"( tex ), "r"( texInfo ), "r"( __float_as_uint( x ) ), "r"( __float_as_uint( y ) ),
          "r"( __float_as_uint( level ) ), "r"( static_cast<unsigned int>( coarse ) ), "l"( singleMipLevelPtr ), "l"( resultPtr )
        : );
    return result;
}
