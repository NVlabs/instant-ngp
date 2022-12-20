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
* @file   optix_micromap.h
* @author NVIDIA Corporation
* @brief  OptiX micromap helper functions
*
* OptiX micromap helper functions. Useable on either host or device.
*/

#ifndef __optix_optix_micromap_h__
#define __optix_optix_micromap_h__

#if !defined( OPTIX_DONT_INCLUDE_CUDA )
// If OPTIX_DONT_INCLUDE_CUDA is defined, cuda driver type float2 must be defined through other
// means before including optix headers.
#include <vector_types.h>
#endif
#include "internal/optix_micromap_impl.h"

/// Convert a micromap triangle index to three base-triangle barycentric coordinates of the micro triangle vertices.
/// The base triangle is the triangle that the micromap is applied to.
///
/// \param[in]  microTriangleIndex  Index of a micro triangle withing a micromap.
/// \param[in]  subdivisionLevel    Subdivision level of the micromap.
/// \param[out] baseBarycentrics0   Barycentric coordinates in the space of the base triangle of vertex 0 of the micro triangle.
/// \param[out] baseBarycentrics1   Barycentric coordinates in the space of the base triangle of vertex 1 of the micro triangle.
/// \param[out] baseBarycentrics2   Barycentric coordinates in the space of the base triangle of vertex 2 of the micro triangle.
OPTIX_MICROMAP_INLINE_FUNC void optixMicromapIndexToBaseBarycentrics( uint32_t microTriangleIndex,
                                                                      uint32_t subdivisionLevel,
                                                                      float2&  baseBarycentrics0,
                                                                      float2&  baseBarycentrics1,
                                                                      float2&  baseBarycentrics2 )
{
    optix_impl::
        micro2bary( microTriangleIndex, subdivisionLevel, baseBarycentrics0, baseBarycentrics1, baseBarycentrics2 );
}


#endif  // __optix_optix_micromap_h__
