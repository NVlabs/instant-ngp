/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   tinyexr_wrapper.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Wrapper around the tinyexr library, providing a simple interface
 *          to load and store EXR images.
 */

#pragma once

#include <neural-graphics-primitives/common.h>

NGP_NAMESPACE_BEGIN

void save_exr(const float* data, int width, int height, int nChannels, int channelStride, const fs::path& path);
void load_exr(float** data, int* width, int* height, const fs::path& path);
__half* load_exr_to_gpu(int* width, int* height, const fs::path& path, bool fix_premult);

NGP_NAMESPACE_END
