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
* @file   optix_7_device_impl_transformations.h
* @author NVIDIA Corporation
* @brief  OptiX public API
*
* OptiX public API Reference - Device side implementation for transformation helper functions.
*/

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_7_device_impl_transformations.h is an internal header file and must not be used directly.  Please use optix_device.h or optix.h instead.")
#endif

#ifndef __optix_optix_7_device_impl_transformations_h__
#define __optix_optix_7_device_impl_transformations_h__

namespace optix_impl {

static __forceinline__ __device__ float4 optixAddFloat4( const float4& a, const float4& b )
{
    return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

static __forceinline__ __device__ float4 optixMulFloat4( const float4& a, float b )
{
    return make_float4( a.x * b, a.y * b, a.z * b, a.w * b );
}

static __forceinline__ __device__ uint4 optixLdg( unsigned long long addr )
{
    const uint4* ptr;
    asm volatile( "cvta.to.global.u64 %0, %1;" : "=l"( ptr ) : "l"( addr ) );
    uint4 ret;
    asm volatile( "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                  : "=r"( ret.x ), "=r"( ret.y ), "=r"( ret.z ), "=r"( ret.w )
                  : "l"( ptr ) );
    return ret;
}

template <class T>
static __forceinline__ __device__ T optixLoadReadOnlyAlign16( const T* ptr )
{
    T v;
    for( int ofs                     = 0; ofs < sizeof( T ); ofs += 16 )
        *(uint4*)( (char*)&v + ofs ) = optixLdg( (unsigned long long)( (char*)ptr + ofs ) );
    return v;
}

// Multiplies the row vector vec with the 3x4 matrix with rows m0, m1, and m2
static __forceinline__ __device__ float4 optixMultiplyRowMatrix( const float4 vec, const float4 m0, const float4 m1, const float4 m2 )
{
    float4 result;

    result.x = vec.x * m0.x + vec.y * m1.x + vec.z * m2.x;
    result.y = vec.x * m0.y + vec.y * m1.y + vec.z * m2.y;
    result.z = vec.x * m0.z + vec.y * m1.z + vec.z * m2.z;
    result.w = vec.x * m0.w + vec.y * m1.w + vec.z * m2.w + vec.w;

    return result;
}

// Converts the SRT transformation srt into a 3x4 matrix with rows m0, m1, and m2
static __forceinline__ __device__ void optixGetMatrixFromSrt( float4& m0, float4& m1, float4& m2, const OptixSRTData& srt )
{
    const float4 q = {srt.qx, srt.qy, srt.qz, srt.qw};

    // normalize
    const float  inv_sql = 1.f / ( srt.qx * srt.qx + srt.qy * srt.qy + srt.qz * srt.qz + srt.qw * srt.qw );
    const float4 nq      = optixMulFloat4( q, inv_sql );

    const float sqw = q.w * nq.w;
    const float sqx = q.x * nq.x;
    const float sqy = q.y * nq.y;
    const float sqz = q.z * nq.z;

    const float xy = q.x * nq.y;
    const float zw = q.z * nq.w;
    const float xz = q.x * nq.z;
    const float yw = q.y * nq.w;
    const float yz = q.y * nq.z;
    const float xw = q.x * nq.w;

    m0.x = ( sqx - sqy - sqz + sqw );
    m0.y = 2.0f * ( xy - zw );
    m0.z = 2.0f * ( xz + yw );

    m1.x = 2.0f * ( xy + zw );
    m1.y = ( -sqx + sqy - sqz + sqw );
    m1.z = 2.0f * ( yz - xw );

    m2.x = 2.0f * ( xz - yw );
    m2.y = 2.0f * ( yz + xw );
    m2.z = ( -sqx - sqy + sqz + sqw );

    m0.w = m0.x * srt.pvx + m0.y * srt.pvy + m0.z * srt.pvz + srt.tx;
    m1.w = m1.x * srt.pvx + m1.y * srt.pvy + m1.z * srt.pvz + srt.ty;
    m2.w = m2.x * srt.pvx + m2.y * srt.pvy + m2.z * srt.pvz + srt.tz;

    m0.z = m0.x * srt.b + m0.y * srt.c + m0.z * srt.sz;
    m1.z = m1.x * srt.b + m1.y * srt.c + m1.z * srt.sz;
    m2.z = m2.x * srt.b + m2.y * srt.c + m2.z * srt.sz;

    m0.y = m0.x * srt.a + m0.y * srt.sy;
    m1.y = m1.x * srt.a + m1.y * srt.sy;
    m2.y = m2.x * srt.a + m2.y * srt.sy;

    m0.x = m0.x * srt.sx;
    m1.x = m1.x * srt.sx;
    m2.x = m2.x * srt.sx;
}

// Inverts a 3x4 matrix in place
static __forceinline__ __device__ void optixInvertMatrix( float4& m0, float4& m1, float4& m2 )
{
    const float det3 =
        m0.x * ( m1.y * m2.z - m1.z * m2.y ) - m0.y * ( m1.x * m2.z - m1.z * m2.x ) + m0.z * ( m1.x * m2.y - m1.y * m2.x );

    const float inv_det3 = 1.0f / det3;

    float inv3[3][3];
    inv3[0][0] = inv_det3 * ( m1.y * m2.z - m2.y * m1.z );
    inv3[0][1] = inv_det3 * ( m0.z * m2.y - m2.z * m0.y );
    inv3[0][2] = inv_det3 * ( m0.y * m1.z - m1.y * m0.z );

    inv3[1][0] = inv_det3 * ( m1.z * m2.x - m2.z * m1.x );
    inv3[1][1] = inv_det3 * ( m0.x * m2.z - m2.x * m0.z );
    inv3[1][2] = inv_det3 * ( m0.z * m1.x - m1.z * m0.x );

    inv3[2][0] = inv_det3 * ( m1.x * m2.y - m2.x * m1.y );
    inv3[2][1] = inv_det3 * ( m0.y * m2.x - m2.y * m0.x );
    inv3[2][2] = inv_det3 * ( m0.x * m1.y - m1.x * m0.y );

    const float b[3] = {m0.w, m1.w, m2.w};

    m0.x = inv3[0][0];
    m0.y = inv3[0][1];
    m0.z = inv3[0][2];
    m0.w = -inv3[0][0] * b[0] - inv3[0][1] * b[1] - inv3[0][2] * b[2];

    m1.x = inv3[1][0];
    m1.y = inv3[1][1];
    m1.z = inv3[1][2];
    m1.w = -inv3[1][0] * b[0] - inv3[1][1] * b[1] - inv3[1][2] * b[2];

    m2.x = inv3[2][0];
    m2.y = inv3[2][1];
    m2.z = inv3[2][2];
    m2.w = -inv3[2][0] * b[0] - inv3[2][1] * b[1] - inv3[2][2] * b[2];
}

static __forceinline__ __device__ void optixLoadInterpolatedMatrixKey( float4& m0, float4& m1, float4& m2, const float4* matrix, const float t1 )
{
    m0 = optixLoadReadOnlyAlign16( &matrix[0] );
    m1 = optixLoadReadOnlyAlign16( &matrix[1] );
    m2 = optixLoadReadOnlyAlign16( &matrix[2] );

    // The conditional prevents concurrent loads leading to spills
    if( t1 > 0.0f )
    {
        const float t0 = 1.0f - t1;
        m0 = optixAddFloat4( optixMulFloat4( m0, t0 ), optixMulFloat4( optixLoadReadOnlyAlign16( &matrix[3] ), t1 ) );
        m1 = optixAddFloat4( optixMulFloat4( m1, t0 ), optixMulFloat4( optixLoadReadOnlyAlign16( &matrix[4] ), t1 ) );
        m2 = optixAddFloat4( optixMulFloat4( m2, t0 ), optixMulFloat4( optixLoadReadOnlyAlign16( &matrix[5] ), t1 ) );
    }
}

static __forceinline__ __device__ void optixLoadInterpolatedSrtKey( float4&       srt0,
                                                                    float4&       srt1,
                                                                    float4&       srt2,
                                                                    float4&       srt3,
                                                                    const float4* srt,
                                                                    const float   t1 )
{
    srt0 = optixLoadReadOnlyAlign16( &srt[0] );
    srt1 = optixLoadReadOnlyAlign16( &srt[1] );
    srt2 = optixLoadReadOnlyAlign16( &srt[2] );
    srt3 = optixLoadReadOnlyAlign16( &srt[3] );

    // The conditional prevents concurrent loads leading to spills
    if( t1 > 0.0f )
    {
        const float t0 = 1.0f - t1;
        srt0 = optixAddFloat4( optixMulFloat4( srt0, t0 ), optixMulFloat4( optixLoadReadOnlyAlign16( &srt[4] ), t1 ) );
        srt1 = optixAddFloat4( optixMulFloat4( srt1, t0 ), optixMulFloat4( optixLoadReadOnlyAlign16( &srt[5] ), t1 ) );
        srt2 = optixAddFloat4( optixMulFloat4( srt2, t0 ), optixMulFloat4( optixLoadReadOnlyAlign16( &srt[6] ), t1 ) );
        srt3 = optixAddFloat4( optixMulFloat4( srt3, t0 ), optixMulFloat4( optixLoadReadOnlyAlign16( &srt[7] ), t1 ) );

        float inv_length = 1.f / sqrt( srt2.y * srt2.y + srt2.z * srt2.z + srt2.w * srt2.w + srt3.x * srt3.x );
        srt2.y *= inv_length;
        srt2.z *= inv_length;
        srt2.w *= inv_length;
        srt3.x *= inv_length;
    }
}

static __forceinline__ __device__ void optixResolveMotionKey( float& localt, int& key, const OptixMotionOptions& options, const float globalt )
{
    const float timeBegin    = options.timeBegin;
    const float timeEnd      = options.timeEnd;
    const float numIntervals = (float)( options.numKeys - 1 );

    // No need to check the motion flags. If data originates from a valid transform list handle, then globalt is in
    // range, or vanish flags are not set.

    const float time = max( 0.f, min( numIntervals, ( globalt - timeBegin ) * numIntervals / ( timeEnd - timeBegin ) ) );
    const float fltKey = floorf( time );

    localt = time - fltKey;
    key    = (int)fltKey;
}

// Returns the interpolated transformation matrix for a particular matrix motion transformation and point in time.
static __forceinline__ __device__ void optixGetInterpolatedTransformation( float4&                           trf0,
                                                                           float4&                           trf1,
                                                                           float4&                           trf2,
                                                                           const OptixMatrixMotionTransform* transformData,
                                                                           const float                       time )
{
    // Compute key and intra key time
    float keyTime;
    int   key;
    optixResolveMotionKey( keyTime, key, optixLoadReadOnlyAlign16( transformData ).motionOptions, time );

    // Get pointer to left key
    const float4* transform = (const float4*)( &transformData->transform[key][0] );

    // Load and interpolate matrix keys
    optixLoadInterpolatedMatrixKey( trf0, trf1, trf2, transform, keyTime );
}

// Returns the interpolated transformation matrix for a particular SRT motion transformation and point in time.
static __forceinline__ __device__ void optixGetInterpolatedTransformation( float4&                        trf0,
                                                                           float4&                        trf1,
                                                                           float4&                        trf2,
                                                                           const OptixSRTMotionTransform* transformData,
                                                                           const float                    time )
{
    // Compute key and intra key time
    float keyTime;
    int   key;
    optixResolveMotionKey( keyTime, key, optixLoadReadOnlyAlign16( transformData ).motionOptions, time );

    // Get pointer to left key
    const float4* dataPtr = reinterpret_cast<const float4*>( &transformData->srtData[key] );

    // Load and interpolated SRT keys
    float4 data[4];
    optixLoadInterpolatedSrtKey( data[0], data[1], data[2], data[3], dataPtr, keyTime );

    OptixSRTData srt = {data[0].x, data[0].y, data[0].z, data[0].w, data[1].x, data[1].y, data[1].z, data[1].w,
                        data[2].x, data[2].y, data[2].z, data[2].w, data[3].x, data[3].y, data[3].z, data[3].w};

    // Convert SRT into a matrix
    optixGetMatrixFromSrt( trf0, trf1, trf2, srt );
}

// Returns the interpolated transformation matrix for a particular traversable handle and point in time.
static __forceinline__ __device__ void optixGetInterpolatedTransformationFromHandle( float4&                      trf0,
                                                                                     float4&                      trf1,
                                                                                     float4&                      trf2,
                                                                                     const OptixTraversableHandle handle,
                                                                                     const float                  time,
                                                                                     const bool objectToWorld )
{
    const OptixTransformType type = optixGetTransformTypeFromHandle( handle );

    if( type == OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM || type == OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM )
    {
        if( type == OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM )
        {
            const OptixMatrixMotionTransform* transformData = optixGetMatrixMotionTransformFromHandle( handle );
            optixGetInterpolatedTransformation( trf0, trf1, trf2, transformData, time );
        }
        else
        {
            const OptixSRTMotionTransform* transformData = optixGetSRTMotionTransformFromHandle( handle );
            optixGetInterpolatedTransformation( trf0, trf1, trf2, transformData, time );
        }

        if( !objectToWorld )
            optixInvertMatrix( trf0, trf1, trf2 );
    }
    else if( type == OPTIX_TRANSFORM_TYPE_INSTANCE || type == OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM )
    {
        const float4* transform;

        if( type == OPTIX_TRANSFORM_TYPE_INSTANCE )
        {
            transform = ( objectToWorld ) ? optixGetInstanceTransformFromHandle( handle ) :
                                            optixGetInstanceInverseTransformFromHandle( handle );
        }
        else
        {
            const OptixStaticTransform* traversable = optixGetStaticTransformFromHandle( handle );
            transform = (const float4*)( ( objectToWorld ) ? traversable->transform : traversable->invTransform );
        }

        trf0 = optixLoadReadOnlyAlign16( &transform[0] );
        trf1 = optixLoadReadOnlyAlign16( &transform[1] );
        trf2 = optixLoadReadOnlyAlign16( &transform[2] );
    }
    else
    {
        trf0 = {1.0f, 0.0f, 0.0f, 0.0f};
        trf1 = {0.0f, 1.0f, 0.0f, 0.0f};
        trf2 = {0.0f, 0.0f, 1.0f, 0.0f};
    }
}

// Returns the world-to-object transformation matrix resulting from the current transform stack and current ray time.
static __forceinline__ __device__ void optixGetWorldToObjectTransformMatrix( float4& m0, float4& m1, float4& m2 )
{
    const unsigned int size = optixGetTransformListSize();
    const float        time = optixGetRayTime();

#pragma unroll 1
    for( unsigned int i = 0; i < size; ++i )
    {
        OptixTraversableHandle handle = optixGetTransformListHandle( i );

        float4 trf0, trf1, trf2;
        optixGetInterpolatedTransformationFromHandle( trf0, trf1, trf2, handle, time, /*objectToWorld*/ false );

        if( i == 0 )
        {
            m0 = trf0;
            m1 = trf1;
            m2 = trf2;
        }
        else
        {
            // m := trf * m
            float4 tmp0 = m0, tmp1 = m1, tmp2 = m2;
            m0 = optixMultiplyRowMatrix( trf0, tmp0, tmp1, tmp2 );
            m1 = optixMultiplyRowMatrix( trf1, tmp0, tmp1, tmp2 );
            m2 = optixMultiplyRowMatrix( trf2, tmp0, tmp1, tmp2 );
        }
    }
}

// Returns the object-to-world transformation matrix resulting from the current transform stack and current ray time.
static __forceinline__ __device__ void optixGetObjectToWorldTransformMatrix( float4& m0, float4& m1, float4& m2 )
{
    const int   size = optixGetTransformListSize();
    const float time = optixGetRayTime();

#pragma unroll 1
    for( int i = size - 1; i >= 0; --i )
    {
        OptixTraversableHandle handle = optixGetTransformListHandle( i );

        float4 trf0, trf1, trf2;
        optixGetInterpolatedTransformationFromHandle( trf0, trf1, trf2, handle, time, /*objectToWorld*/ true );

        if( i == size - 1 )
        {
            m0 = trf0;
            m1 = trf1;
            m2 = trf2;
        }
        else
        {
            // m := trf * m
            float4 tmp0 = m0, tmp1 = m1, tmp2 = m2;
            m0 = optixMultiplyRowMatrix( trf0, tmp0, tmp1, tmp2 );
            m1 = optixMultiplyRowMatrix( trf1, tmp0, tmp1, tmp2 );
            m2 = optixMultiplyRowMatrix( trf2, tmp0, tmp1, tmp2 );
        }
    }
}

// Multiplies the 3x4 matrix with rows m0, m1, m2 with the point p.
static __forceinline__ __device__ float3 optixTransformPoint( const float4& m0, const float4& m1, const float4& m2, const float3& p )
{
    float3 result;
    result.x = m0.x * p.x + m0.y * p.y + m0.z * p.z + m0.w;
    result.y = m1.x * p.x + m1.y * p.y + m1.z * p.z + m1.w;
    result.z = m2.x * p.x + m2.y * p.y + m2.z * p.z + m2.w;
    return result;
}

// Multiplies the 3x3 linear submatrix of the 3x4 matrix with rows m0, m1, m2 with the vector v.
static __forceinline__ __device__ float3 optixTransformVector( const float4& m0, const float4& m1, const float4& m2, const float3& v )
{
    float3 result;
    result.x = m0.x * v.x + m0.y * v.y + m0.z * v.z;
    result.y = m1.x * v.x + m1.y * v.y + m1.z * v.z;
    result.z = m2.x * v.x + m2.y * v.y + m2.z * v.z;
    return result;
}

// Multiplies the transpose of the 3x3 linear submatrix of the 3x4 matrix with rows m0, m1, m2 with the normal n.
// Note that the given matrix is supposed to be the inverse of the actual transformation matrix.
static __forceinline__ __device__ float3 optixTransformNormal( const float4& m0, const float4& m1, const float4& m2, const float3& n )
{
    float3 result;
    result.x = m0.x * n.x + m1.x * n.y + m2.x * n.z;
    result.y = m0.y * n.x + m1.y * n.y + m2.y * n.z;
    result.z = m0.z * n.x + m1.z * n.y + m2.z * n.z;
    return result;
}

}  // namespace optix_impl

#endif
