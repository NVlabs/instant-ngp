#########################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#########################################################################

# FLIP: A Difference Evaluator for Alternating Images
# High Performance Graphics, 2020.
# by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller, Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild
#
# Pointer to our paper: https://research.nvidia.com/publication/2020-07_FLIP
# code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller

import numpy as np
from scipy import signal

def color_space_transform(input_color, fromSpace2toSpace):
    dim = input_color.shape

    if fromSpace2toSpace == "srgb2linrgb":
        limit = 0.04045
        transformed_color = np.where(input_color > limit, np.power((input_color + 0.055) / 1.055, 2.4), input_color / 12.92)

    elif fromSpace2toSpace == "linrgb2srgb":
        limit = 0.0031308
        transformed_color = np.where(input_color > limit, 1.055 * (input_color ** (1.0 / 2.4)) - 0.055, 12.92 * input_color)

    elif fromSpace2toSpace == "linrgb2xyz" or fromSpace2toSpace == "xyz2linrgb":
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        a11 = 10135552 / 24577794
        a12 = 8788810  / 24577794
        a13 = 4435075  / 24577794
        a21 = 2613072  / 12288897
        a22 = 8788810  / 12288897
        a23 = 887015   / 12288897
        a31 = 1425312  / 73733382
        a32 = 8788810  / 73733382
        a33 = 70074185 / 73733382
        A = np.array([[a11, a12, a13],
                        [a21, a22, a23],
                        [a31, a32, a33]])

        input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
        if fromSpace2toSpace == "xyz2linrgb":
            A = np.linalg.inv(A)
        transformed_color = np.matmul(A, input_color)
        transformed_color = np.transpose(transformed_color, (1, 2, 0))

    elif fromSpace2toSpace == "xyz2ycxcz":
        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        input_color = np.divide(input_color, reference_illuminant)
        y = 116 * input_color[1:2, :, :] - 16
        cx = 500 * (input_color[0:1, :, :] - input_color[1:2, :, :])
        cz = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])
        transformed_color = np.concatenate((y, cx, cz), 0)

    elif fromSpace2toSpace == "ycxcz2xyz":
        y = (input_color[0:1, :, :] + 16) / 116
        cx = input_color[1:2, :, :] / 500
        cz = input_color[2:3, :, :] / 200

        x = y + cx
        z = y - cz
        transformed_color = np.concatenate((x, y, z), 0)

        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        transformed_color = np.multiply(transformed_color, reference_illuminant)

    elif fromSpace2toSpace == "xyz2lab":
        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        input_color = np.divide(input_color, reference_illuminant)
        delta = 6 / 29
        limit = 0.00885

        input_color = np.where(input_color > limit, np.power(input_color, 1 / 3), (input_color / (3 * delta * delta)) + (4 / 29))

        l = 116 * input_color[1:2, :, :] - 16
        a = 500 * (input_color[0:1,:, :] - input_color[1:2, :, :])
        b = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])

        transformed_color = np.concatenate((l, a, b), 0)

    elif fromSpace2toSpace == "lab2xyz":
        y = (input_color[0:1, :, :] + 16) / 116
        a =  input_color[1:2, :, :] / 500
        b =  input_color[2:3, :, :] / 200

        x = y + a
        z = y - b

        xyz = np.concatenate((x, y, z), 0)
        delta = 6 / 29
        xyz = np.where(xyz > delta,  xyz ** 3, 3 * delta ** 2 * (xyz - 4 / 29))

        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        transformed_color = np.multiply(xyz, reference_illuminant)

    elif fromSpace2toSpace == "srgb2xyz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color,'linrgb2xyz')
    elif fromSpace2toSpace == "srgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "linrgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "srgb2lab":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "linrgb2lab":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "ycxcz2linrgb":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
    elif fromSpace2toSpace == "lab2srgb":
        transformed_color = color_space_transform(input_color, 'lab2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
    elif fromSpace2toSpace == "ycxcz2lab":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    else:
        print('The color transform is not defined!')
        transformed_color = input_color

    return transformed_color

def generate_spatial_filter(pixels_per_degree, channel):
    a1_A = 1 
    b1_A = 0.0047
    a2_A = 0
    b2_A = 1e-5 # avoid division by 0
    a1_rg = 1
    b1_rg = 0.0053
    a2_rg = 0
    b2_rg = 1e-5 # avoid division by 0
    a1_by = 34.1
    b1_by = 0.04
    a2_by = 13.5
    b2_by = 0.025
    if channel == "A": #Achromatic CSF
        a1 = a1_A
        b1 = b1_A
        a2 = a2_A
        b2 = b2_A
    elif channel == "RG": #Red-Green CSF
        a1 = a1_rg
        b1 = b1_rg
        a2 = a2_rg
        b2 = b2_rg
    elif channel == "BY": # Blue-Yellow CSF
        a1 = a1_by
        b1 = b1_by
        a2 = a2_by
        b2 = b2_by

    # Determine evaluation domain
    max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
    r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2)) * pixels_per_degree)
    r = int(r)
    deltaX = 1.0 / pixels_per_degree
    x, y = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
    z = (x * deltaX)**2 + (y * deltaX)**2
    
    # Generate weights
    g = a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi**2 * z / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(-np.pi**2 * z / b2)
    g = g / np.sum(g)

    return g, r

def spatial_filter(img, s_a, s_rg, s_by, radius):
    # Filters image img using Contrast Sensitivity Functions.
    # Returns linear RGB

    dim = img.shape
    # Prepare convolution input
    img_pad_a = np.pad(img[0:1, :, :], ((0, 0), (radius, radius), (radius, radius)), mode='edge')
    img_pad_rg = np.pad(img[1:2, :, :], ((0, 0), (radius, radius), (radius, radius)), mode='edge')
    img_pad_by = np.pad(img[2:3, :, :], ((0, 0), (radius, radius), (radius, radius)), mode='edge')

    # Apply Gaussian filters
    img_tilde_opponent = np.zeros((dim[0], dim[1], dim[2]))
    img_tilde_opponent[0:1, :, :] = signal.convolve2d(img_pad_a.squeeze(0), s_a, mode='valid')
    img_tilde_opponent[1:2, :, :] = signal.convolve2d(img_pad_rg.squeeze(0), s_rg, mode='valid')
    img_tilde_opponent[2:3, :, :] = signal.convolve2d(img_pad_by.squeeze(0), s_by, mode='valid')

    # Transform to linear RGB for clamp
    img_tilde_linear_rgb = color_space_transform(img_tilde_opponent, 'ycxcz2linrgb')
    
    # Clamp to RGB box
    return np.clip(img_tilde_linear_rgb, 0.0, 1.0)

def hunt_adjustment(img):
    # Applies Hunt adjustment to L*a*b* image img
    
    # Extract luminance component
    L = img[0:1, :, :]
    
    # Apply Hunt adjustment
    img_h = np.zeros(img.shape)
    img_h[0:1, :, :] = L
    img_h[1:2, :, :] = np.multiply((0.01 * L), img[1:2, :, :])
    img_h[2:3, :, :] = np.multiply((0.01 * L), img[2:3, :, :])

    return img_h

def hyab(reference, test):
    # Computes HyAB distance between L*a*b* images reference and test
    delta = reference - test
    return abs(delta[0:1, :, :]) + np.linalg.norm(delta[1:3, :, :], axis=0)

def redistribute_errors(power_deltaE_hyab, cmax):
    # Set redistribution parameters
    pc = 0.4
    pt = 0.95
    
    # Re-map error to 0-1 range. Values between 0 and
    # pccmax are mapped to the range [0, pt],
    # while the rest are mapped to the range (pt, 1]
    deltaE_c = np.zeros(power_deltaE_hyab.shape)
    pccmax = pc * cmax
    deltaE_c = np.where(power_deltaE_hyab < pccmax, (pt / pccmax) * power_deltaE_hyab, pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt))

    return deltaE_c

def feature_detection(imgy, pixels_per_degree, feature_type):
    # Finds features of type feature_type in image img based on current PPD
    
    # Set peak to trough value (2x standard deviations) of human edge
    # detection filter
    w = 0.082
    
    # Compute filter radius
    sd = 0.5 * w * pixels_per_degree
    radius = int(np.ceil(3 * sd))

    # Compute 2D Gaussian
    [x, y] = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))
    
    if feature_type == 'edge': # Edge detector
        # Compute partial derivative in x-direction
        Gx = np.multiply(-x, g)
    else: # Point detector
        # Compute second partial derivative in x-direction
        Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)
 
    # Normalize positive weights to sum to 1 and negative weights to sum to -1
    negative_weights_sum = -np.sum(Gx[Gx < 0])
    positive_weights_sum = np.sum(Gx[Gx > 0])
    Gx = np.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)
    
    # Detect features
    imgy_pad = np.pad(imgy, ((0, 0), (radius, radius), (radius, radius)), mode='edge').squeeze(0)
    featuresX = signal.convolve2d(imgy_pad, Gx, mode='valid')
    featuresY = signal.convolve2d(imgy_pad, np.transpose(Gx), mode='valid')

    return np.stack((featuresX, featuresY))

def compute_flip(reference, test, pixels_per_degree):
    assert reference.shape == test.shape

    # Set color and feature exponents
    qc = 0.7
    qf = 0.5

    # Transform reference and test to opponent color space
    reference = color_space_transform(reference, 'srgb2ycxcz')
    test = color_space_transform(test, 'srgb2ycxcz')

    # --- Color pipeline ---
    # Spatial filtering
    s_a, radius_a = generate_spatial_filter(pixels_per_degree, 'A')
    s_rg, radius_rg = generate_spatial_filter(pixels_per_degree, 'RG')
    s_by, radius_by = generate_spatial_filter(pixels_per_degree, 'BY')
    radius = max(radius_a, radius_rg, radius_by)
    filtered_reference = spatial_filter(reference, s_a, s_rg, s_by, radius)
    filtered_test = spatial_filter(test, s_a, s_rg, s_by, radius)

    # Perceptually Uniform Color Space
    preprocessed_reference = hunt_adjustment(color_space_transform(filtered_reference, 'linrgb2lab'))
    preprocessed_test = hunt_adjustment(color_space_transform(filtered_test, 'linrgb2lab'))

    # Color metric
    deltaE_hyab = hyab(preprocessed_reference, preprocessed_test)
    hunt_adjusted_green = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[1.0]], [[0.0]]]), 'linrgb2lab'))
    hunt_adjusted_blue = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[0.0]], [[1.0]]]), 'linrgb2lab'))
    cmax = np.power(hyab(hunt_adjusted_green, hunt_adjusted_blue), qc)
    deltaE_c = redistribute_errors(np.power(deltaE_hyab, qc), cmax)

    # --- Feature pipeline ---
    # Extract and normalize achromatic component
    reference_y = (reference[0:1, :, :] + 16) / 116
    test_y = (test[0:1, :, :] + 16) / 116

    # Edge and point detection
    edges_reference = feature_detection(reference_y, pixels_per_degree, 'edge')
    points_reference = feature_detection(reference_y, pixels_per_degree, 'point')
    edges_test = feature_detection(test_y, pixels_per_degree, 'edge')
    points_test = feature_detection(test_y, pixels_per_degree, 'point')

    # Feature metric
    deltaE_f = np.maximum(abs(np.linalg.norm(edges_reference, axis=0) - np.linalg.norm(edges_test, axis=0)), abs(np.linalg.norm(points_test, axis=0) - np.linalg.norm(points_reference, axis=0)))
    deltaE_f = np.power(((1 / np.sqrt(2)) * deltaE_f), qf)

    # --- Final error ---
    return np.power(deltaE_c, 1 - deltaE_f)
