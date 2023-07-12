#!/usr/bin/python3
"""Tonemapper"""
import glob
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import imageio
import numpy as np
from matplotlib.cm import get_cmap
from tqdm import tqdm

CM_MAGMA: np.ndarray = (np.array([get_cmap('magma').colors]).transpose(
    [1, 0, 2]) * 255)[..., ::-1].astype(np.uint8)


@dataclass
class TonemapParameters:
    """ToneMap Parameters."""
    min_value: float = 0
    max_value: float = 8  # Compute Scale from NeRF Resizing
    colormap: Optional[np.ndarray] = CM_MAGMA
    gamma: float = 0.5
    color_gamma: float = 2.2


def tonemap(image: np.ndarray, parameters: TonemapParameters = TonemapParameters()) -> np.ndarray:
    """Applied colormap"""

    image_clip = np.clip(image, parameters.min_value, parameters.max_value)
    image_scaled = (image_clip - parameters.min_value) / \
        (parameters.max_value - parameters.min_value)
    image_scaled = image_scaled ** parameters.gamma
    image_scaled_uint8 = np.array(image_scaled * 255, dtype=np.uint8)

    if parameters.colormap is None:
        return image_scaled_uint8

    image_tonemapped_uint8 = cv2.applyColorMap(
        image_scaled_uint8, parameters.colormap)
    return (((image_tonemapped_uint8 / 255.0) ** parameters.color_gamma) * 255).astype(np.uint8)[..., ::-1]


def tonemap_folder(raw_depth_folder: str, color_depth_folder: str):
    """Color Depth"""

    os.makedirs(color_depth_folder, exist_ok=True)

    files = list(glob.glob(os.path.join(raw_depth_folder, "*.npy")))
    with tqdm(desc="Tonemap", total=len(files), unit="frame") as t:
        for file in files:
            outname = os.path.join(color_depth_folder,
                                   os.path.splitext(os.path.basename(file))[0]+".png")
            image = np.load(file)
            color = tonemap(image)
            imageio.imwrite(outname, color)
            t.update(1)


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert SRGB Image to Linear."""
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert Linear Image to SRGB."""
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
