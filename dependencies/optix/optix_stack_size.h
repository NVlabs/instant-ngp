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

/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header

#ifndef __optix_optix_stack_size_h__
#define __optix_optix_stack_size_h__

#include "optix.h"

#include <algorithm>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup optix_utilities
@{
*/

/// Retrieves direct and continuation stack sizes for each program in the program group and accumulates the upper bounds
/// in the correponding output variables based on the semantic type of the program. Before the first invocation of this
/// function with a given instance of #OptixStackSizes, the members of that instance should be set to 0.
inline OptixResult optixUtilAccumulateStackSizes( OptixProgramGroup programGroup, OptixStackSizes* stackSizes )
{
    if( !stackSizes )
        return OPTIX_ERROR_INVALID_VALUE;

    OptixStackSizes localStackSizes;
    OptixResult     result = optixProgramGroupGetStackSize( programGroup, &localStackSizes );
    if( result != OPTIX_SUCCESS )
        return result;

    stackSizes->cssRG = std::max( stackSizes->cssRG, localStackSizes.cssRG );
    stackSizes->cssMS = std::max( stackSizes->cssMS, localStackSizes.cssMS );
    stackSizes->cssCH = std::max( stackSizes->cssCH, localStackSizes.cssCH );
    stackSizes->cssAH = std::max( stackSizes->cssAH, localStackSizes.cssAH );
    stackSizes->cssIS = std::max( stackSizes->cssIS, localStackSizes.cssIS );
    stackSizes->cssCC = std::max( stackSizes->cssCC, localStackSizes.cssCC );
    stackSizes->dssDC = std::max( stackSizes->dssDC, localStackSizes.dssDC );

    return OPTIX_SUCCESS;
}

/// Computes the stack size values needed to configure a pipeline.
///
/// See the programming guide for an explanation of the formula.
///
/// \param[in] stackSizes                              Accumulated stack sizes of all programs in the call graph.
/// \param[in] maxTraceDepth                           Maximum depth of #optixTrace() calls.
/// \param[in] maxCCDepth                              Maximum depth of calls trees of continuation callables.
/// \param[in] maxDCDepth                              Maximum depth of calls trees of direct callables.
/// \param[out] directCallableStackSizeFromTraversal   Direct stack size requirement for direct callables invoked from
///                                                    IS or AH.
/// \param[out] directCallableStackSizeFromState       Direct stack size requirement for direct callables invoked from
///                                                    RG, MS, or CH.
/// \param[out] continuationStackSize                  Continuation stack requirement.
inline OptixResult optixUtilComputeStackSizes( const OptixStackSizes* stackSizes,
                                               unsigned int           maxTraceDepth,
                                               unsigned int           maxCCDepth,
                                               unsigned int           maxDCDepth,
                                               unsigned int*          directCallableStackSizeFromTraversal,
                                               unsigned int*          directCallableStackSizeFromState,
                                               unsigned int*          continuationStackSize )
{
    if( !stackSizes )
        return OPTIX_ERROR_INVALID_VALUE;

    const unsigned int cssRG = stackSizes->cssRG;
    const unsigned int cssMS = stackSizes->cssMS;
    const unsigned int cssCH = stackSizes->cssCH;
    const unsigned int cssAH = stackSizes->cssAH;
    const unsigned int cssIS = stackSizes->cssIS;
    const unsigned int cssCC = stackSizes->cssCC;
    const unsigned int dssDC = stackSizes->dssDC;

    if( directCallableStackSizeFromTraversal )
        *directCallableStackSizeFromTraversal = maxDCDepth * dssDC;
    if( directCallableStackSizeFromState )
        *directCallableStackSizeFromState = maxDCDepth * dssDC;

    // upper bound on continuation stack used by call trees of continuation callables
    unsigned int cssCCTree = maxCCDepth * cssCC;

    // upper bound on continuation stack used by CH or MS programs including the call tree of
    // continuation callables
    unsigned int cssCHOrMSPlusCCTree = std::max( cssCH, cssMS ) + cssCCTree;

    // clang-format off
    if( continuationStackSize )
        *continuationStackSize
            = cssRG + cssCCTree
            + ( std::max( maxTraceDepth, 1u ) - 1 ) * cssCHOrMSPlusCCTree
            + std::min( maxTraceDepth, 1u ) * std::max( cssCHOrMSPlusCCTree, cssIS + cssAH );
    // clang-format on

    return OPTIX_SUCCESS;
}

/// Computes the stack size values needed to configure a pipeline.
///
/// This variant is similar to #optixUtilComputeStackSizes(), except that it expects the values dssDC and
/// maxDCDepth split by call site semantic.
///
/// See programming guide for an explanation of the formula.
///
/// \param[in] stackSizes                              Accumulated stack sizes of all programs in the call graph.
/// \param[in] dssDCFromTraversal                      Accumulated direct stack size of all DC programs invoked from IS
///                                                    or AH.
/// \param[in] dssDCFromState                          Accumulated direct stack size of all DC programs invoked from RG,
///                                                    MS, or CH.
/// \param[in] maxTraceDepth                           Maximum depth of #optixTrace() calls.
/// \param[in] maxCCDepth                              Maximum depth of calls trees of continuation callables.
/// \param[in] maxDCDepthFromTraversal                 Maximum depth of calls trees of direct callables invoked from IS
///                                                    or AH.
/// \param[in] maxDCDepthFromState                     Maximum depth of calls trees of direct callables invoked from RG,
///                                                    MS, or CH.
/// \param[out] directCallableStackSizeFromTraversal   Direct stack size requirement for direct callables invoked from
///                                                    IS or AH.
/// \param[out] directCallableStackSizeFromState       Direct stack size requirement for direct callables invoked from
///                                                    RG, MS, or CH.
/// \param[out] continuationStackSize                  Continuation stack requirement.
inline OptixResult optixUtilComputeStackSizesDCSplit( const OptixStackSizes* stackSizes,
                                                      unsigned int           dssDCFromTraversal,
                                                      unsigned int           dssDCFromState,
                                                      unsigned int           maxTraceDepth,
                                                      unsigned int           maxCCDepth,
                                                      unsigned int           maxDCDepthFromTraversal,
                                                      unsigned int           maxDCDepthFromState,
                                                      unsigned int*          directCallableStackSizeFromTraversal,
                                                      unsigned int*          directCallableStackSizeFromState,
                                                      unsigned int*          continuationStackSize )
{
    if( !stackSizes )
        return OPTIX_ERROR_INVALID_VALUE;

    const unsigned int cssRG = stackSizes->cssRG;
    const unsigned int cssMS = stackSizes->cssMS;
    const unsigned int cssCH = stackSizes->cssCH;
    const unsigned int cssAH = stackSizes->cssAH;
    const unsigned int cssIS = stackSizes->cssIS;
    const unsigned int cssCC = stackSizes->cssCC;
    // use dssDCFromTraversal and dssDCFromState instead of stackSizes->dssDC

    if( directCallableStackSizeFromTraversal )
        *directCallableStackSizeFromTraversal = maxDCDepthFromTraversal * dssDCFromTraversal;
    if( directCallableStackSizeFromState )
        *directCallableStackSizeFromState = maxDCDepthFromState * dssDCFromState;

    // upper bound on continuation stack used by call trees of continuation callables
    unsigned int cssCCTree = maxCCDepth * cssCC;

    // upper bound on continuation stack used by CH or MS programs including the call tree of
    // continuation callables
    unsigned int cssCHOrMSPlusCCTree = std::max( cssCH, cssMS ) + cssCCTree;

    // clang-format off
    if( continuationStackSize )
        *continuationStackSize
            = cssRG + cssCCTree
            + ( std::max( maxTraceDepth, 1u ) - 1 ) * cssCHOrMSPlusCCTree
            + std::min( maxTraceDepth, 1u ) * std::max( cssCHOrMSPlusCCTree, cssIS + cssAH );
    // clang-format on

    return OPTIX_SUCCESS;
}

/// Computes the stack size values needed to configure a pipeline.
///
/// This variant is similar to #optixUtilComputeStackSizes(), except that it expects the value cssCCTree
/// instead of cssCC and maxCCDepth.
///
/// See programming guide for an explanation of the formula.
///
/// \param[in] stackSizes                              Accumulated stack sizes of all programs in the call graph.
/// \param[in] cssCCTree                               Maximum stack size used by calls trees of continuation callables.
/// \param[in] maxTraceDepth                           Maximum depth of #optixTrace() calls.
/// \param[in] maxDCDepth                              Maximum depth of calls trees of direct callables.
/// \param[out] directCallableStackSizeFromTraversal   Direct stack size requirement for direct callables invoked from
///                                                    IS or AH.
/// \param[out] directCallableStackSizeFromState       Direct stack size requirement for direct callables invoked from
///                                                    RG, MS, or CH.
/// \param[out] continuationStackSize                  Continuation stack requirement.
inline OptixResult optixUtilComputeStackSizesCssCCTree( const OptixStackSizes* stackSizes,
                                                        unsigned int           cssCCTree,
                                                        unsigned int           maxTraceDepth,
                                                        unsigned int           maxDCDepth,
                                                        unsigned int*          directCallableStackSizeFromTraversal,
                                                        unsigned int*          directCallableStackSizeFromState,
                                                        unsigned int*          continuationStackSize )
{
    if( !stackSizes )
        return OPTIX_ERROR_INVALID_VALUE;

    const unsigned int cssRG = stackSizes->cssRG;
    const unsigned int cssMS = stackSizes->cssMS;
    const unsigned int cssCH = stackSizes->cssCH;
    const unsigned int cssAH = stackSizes->cssAH;
    const unsigned int cssIS = stackSizes->cssIS;
    // use cssCCTree instead of stackSizes->cssCC and maxCCDepth
    const unsigned int dssDC = stackSizes->dssDC;

    if( directCallableStackSizeFromTraversal )
        *directCallableStackSizeFromTraversal = maxDCDepth * dssDC;
    if( directCallableStackSizeFromState )
        *directCallableStackSizeFromState = maxDCDepth * dssDC;

    // upper bound on continuation stack used by CH or MS programs including the call tree of
    // continuation callables
    unsigned int cssCHOrMSPlusCCTree = std::max( cssCH, cssMS ) + cssCCTree;

    // clang-format off
    if( continuationStackSize )
        *continuationStackSize
            = cssRG + cssCCTree
            + ( std::max( maxTraceDepth, 1u ) - 1 ) * cssCHOrMSPlusCCTree
            + std::min( maxTraceDepth, 1u ) * std::max( cssCHOrMSPlusCCTree, cssIS + cssAH );
    // clang-format on

    return OPTIX_SUCCESS;
}

/// Computes the stack size values needed to configure a pipeline.
///
/// This variant is a specialization of #optixUtilComputeStackSizes() for a simple path tracer with the following
/// assumptions: There are only two ray types, camera rays and shadow rays. There are only RG, MS, and CH programs, and
/// no AH, IS, CC, or DC programs. The camera rays invoke only the miss and closest hit programs MS1 and CH1,
/// respectively. The CH1 program might trace shadow rays, which invoke only the miss and closest hit programs MS2 and
/// CH2, respectively.
///
/// For flexibility, we allow for each of CH1 and CH2 not just one single program group, but an array of programs
/// groups, and compute the maximas of the stack size requirements per array.
///
/// See programming guide for an explanation of the formula.
inline OptixResult optixUtilComputeStackSizesSimplePathTracer( OptixProgramGroup        programGroupRG,
                                                               OptixProgramGroup        programGroupMS1,
                                                               const OptixProgramGroup* programGroupCH1,
                                                               unsigned int             programGroupCH1Count,
                                                               OptixProgramGroup        programGroupMS2,
                                                               const OptixProgramGroup* programGroupCH2,
                                                               unsigned int             programGroupCH2Count,
                                                               unsigned int* directCallableStackSizeFromTraversal,
                                                               unsigned int* directCallableStackSizeFromState,
                                                               unsigned int* continuationStackSize )
{
    if( !programGroupCH1 && ( programGroupCH1Count > 0 ) )
        return OPTIX_ERROR_INVALID_VALUE;
    if( !programGroupCH2 && ( programGroupCH2Count > 0 ) )
        return OPTIX_ERROR_INVALID_VALUE;

    OptixResult result;

    OptixStackSizes stackSizesRG = {};
    result                       = optixProgramGroupGetStackSize( programGroupRG, &stackSizesRG );
    if( result != OPTIX_SUCCESS )
        return result;

    OptixStackSizes stackSizesMS1 = {};
    result                        = optixProgramGroupGetStackSize( programGroupMS1, &stackSizesMS1 );
    if( result != OPTIX_SUCCESS )
        return result;

    OptixStackSizes stackSizesCH1 = {};
    for( unsigned int i = 0; i < programGroupCH1Count; ++i )
    {
        result = optixUtilAccumulateStackSizes( programGroupCH1[i], &stackSizesCH1 );
        if( result != OPTIX_SUCCESS )
            return result;
    }

    OptixStackSizes stackSizesMS2 = {};
    result                        = optixProgramGroupGetStackSize( programGroupMS2, &stackSizesMS2 );
    if( result != OPTIX_SUCCESS )
        return result;

    OptixStackSizes stackSizesCH2 = {};
    memset( &stackSizesCH2, 0, sizeof( OptixStackSizes ) );
    for( unsigned int i = 0; i < programGroupCH2Count; ++i )
    {
        result = optixUtilAccumulateStackSizes( programGroupCH2[i], &stackSizesCH2 );
        if( result != OPTIX_SUCCESS )
            return result;
    }

    const unsigned int cssRG  = stackSizesRG.cssRG;
    const unsigned int cssMS1 = stackSizesMS1.cssMS;
    const unsigned int cssCH1 = stackSizesCH1.cssCH;
    const unsigned int cssMS2 = stackSizesMS2.cssMS;
    const unsigned int cssCH2 = stackSizesCH2.cssCH;
    // no AH, IS, CC, or DC programs

    if( directCallableStackSizeFromTraversal )
        *directCallableStackSizeFromTraversal = 0;
    if( directCallableStackSizeFromState )
        *directCallableStackSizeFromState = 0;

    if( continuationStackSize )
        *continuationStackSize = cssRG + std::max( cssMS1, cssCH1 + std::max( cssMS2, cssCH2 ) );

    return OPTIX_SUCCESS;
}

/*@}*/  // end group optix_utilities

#ifdef __cplusplus
}
#endif

#endif  // __optix_optix_stack_size_h__
