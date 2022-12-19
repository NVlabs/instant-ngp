/*
 * Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
* @file   optix_micromap_impl.h
* @author NVIDIA Corporation
* @brief  OptiX micromap helper functions
*/

#ifndef __optix_optix_micromap_impl_h__
#define __optix_optix_micromap_impl_h__

#include <cstdint>

#if __CUDACC__
#include <cuda_runtime.h>
#endif

#ifndef OPTIX_MICROMAP_FUNC
#if __CUDACC__
#define OPTIX_MICROMAP_FUNC __host__ __device__
#else
#define OPTIX_MICROMAP_FUNC
#endif
#endif

namespace optix_impl {

/** \addtogroup optix_utilities
@{
*/

#define OPTIX_MICROMAP_INLINE_FUNC OPTIX_MICROMAP_FUNC inline

#if __CUDACC__
// the device implementation of __uint_as_float is declared in cuda_runtime.h
#else
OPTIX_MICROMAP_INLINE_FUNC float __uint_as_float( uint32_t x )
{
    union { float f; uint32_t i; } var;
    var.i = x;
    return var.f;
}
#endif


// Deinterleave bits from x into even and odd halves
OPTIX_MICROMAP_INLINE_FUNC uint32_t deinterleaveBits( uint32_t x )
{
    x = ( ( ( ( x >> 1 ) & 0x22222222u ) | ( ( x << 1 ) & ~0x22222222u ) ) & 0x66666666u ) | ( x & ~0x66666666u );
    x = ( ( ( ( x >> 2 ) & 0x0c0c0c0cu ) | ( ( x << 2 ) & ~0x0c0c0c0cu ) ) & 0x3c3c3c3cu ) | ( x & ~0x3c3c3c3cu );
    x = ( ( ( ( x >> 4 ) & 0x00f000f0u ) | ( ( x << 4 ) & ~0x00f000f0u ) ) & 0x0ff00ff0u ) | ( x & ~0x0ff00ff0u );
    x = ( ( ( ( x >> 8 ) & 0x0000ff00u ) | ( ( x << 8 ) & ~0x0000ff00u ) ) & 0x00ffff00u ) | ( x & ~0x00ffff00u );
    return x;
}

// Extract even bits
OPTIX_MICROMAP_INLINE_FUNC uint32_t extractEvenBits( uint32_t x )
{
    x &= 0x55555555;
    x = ( x | ( x >> 1 ) ) & 0x33333333;
    x = ( x | ( x >> 2 ) ) & 0x0f0f0f0f;
    x = ( x | ( x >> 4 ) ) & 0x00ff00ff;
    x = ( x | ( x >> 8 ) ) & 0x0000ffff;
    return x;
}


// Calculate exclusive prefix or (log(n) XOR's and SHF's)
OPTIX_MICROMAP_INLINE_FUNC uint32_t prefixEor( uint32_t x )
{
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    return x;
}


// Convert distance along the curve to discrete barycentrics
OPTIX_MICROMAP_INLINE_FUNC void index2dbary( uint32_t index, uint32_t& u, uint32_t& v, uint32_t& w )
{
    uint32_t b0 = extractEvenBits( index );
    uint32_t b1 = extractEvenBits( index >> 1 );

    uint32_t fx = prefixEor( b0 );
    uint32_t fy = prefixEor( b0 & ~b1 );

    uint32_t t = fy ^ b1;

    u = ( fx & ~t ) | ( b0 & ~t ) | ( ~b0 & ~fx & t );
    v = fy ^ b0;
    w = ( ~fx & ~t ) | ( b0 & ~t ) | ( ~b0 & fx & t );
}


// Compute barycentrics for micro triangle
OPTIX_MICROMAP_INLINE_FUNC void micro2bary( uint32_t index, uint32_t subdivisionLevel, float2& uv0, float2& uv1, float2& uv2 )
{
    if( subdivisionLevel == 0 )
    {
        uv0 = { 0, 0 };
        uv1 = { 1, 0 };
        uv2 = { 0, 1 };
        return;
    }

    uint32_t iu, iv, iw;
    index2dbary( index, iu, iv, iw );

    // we need to only look at "level" bits
    iu = iu & ( ( 1 << subdivisionLevel ) - 1 );
    iv = iv & ( ( 1 << subdivisionLevel ) - 1 );
    iw = iw & ( ( 1 << subdivisionLevel ) - 1 );

    bool upright = ( iu & 1 ) ^ ( iv & 1 ) ^ ( iw & 1 );
    if( !upright )
    {
        iu = iu + 1;
        iv = iv + 1;
    }

    const float levelScale = __uint_as_float( ( 127u - subdivisionLevel ) << 23 );

    // scale the barycentic coordinate to the global space/scale
    float du = 1.f * levelScale;
    float dv = 1.f * levelScale;

    // scale the barycentic coordinate to the global space/scale
    float u = (float)iu * levelScale;
    float v = (float)iv * levelScale;

    if( !upright )
    {
        du = -du;
        dv = -dv;
    }

    uv0 = { u, v };
    uv1 = { u + du, v };
    uv2 = { u, v + dv };
}


/*@}*/  // end group optix_utilities

}  // namespace optix_impl

#endif  // __optix_optix_micromap_impl_h__
