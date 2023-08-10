#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import code
import glob
import imageio
import numpy as np
import os
from pathlib import Path, PurePosixPath
from scipy.ndimage.filters import convolve1d
import struct
import sys

import flip
import flip.utils

PAPER_FOLDER = Path(__file__).resolve().parent.parent
SUPPL_FOLDER = PAPER_FOLDER/"supplemental"
SCRIPTS_FOLDER = PAPER_FOLDER/"scripts"
TEMPLATE_FOLDER = SCRIPTS_FOLDER/"template"
DATA_FOLDER = SCRIPTS_FOLDER/"data"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
NGP_DATA_FOLDER = os.environ.get("NGP_DATA_FOLDER") or os.path.join(ROOT_DIR, "data")


NERF_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "nerf")
SDF_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "sdf")
IMAGE_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "image")
VOLUME_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "volume")

# Search for pyngp in the build folder.
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

def repl(testbed):
	print("-------------------\npress Ctrl-Z to return to gui\n---------------------------")
	code.InteractiveConsole(locals=locals()).interact()
	print("------- returning to gui...")

def mse2psnr(x): return -10.*np.log(x)/np.log(10.)

def sanitize_path(path):
	return str(PurePosixPath(path.relative_to(PAPER_FOLDER)))

# from https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays
def trapez(y,y0,w):
	return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
	# The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
	# If either of these cases are violated, do some switches.
	if abs(c1-c0) < abs(r1-r0):
		# Switch x and y, and switch again when returning.
		xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
		return (yy, xx, val)

	# At this point we know that the distance in columns (x) is greater
	# than that in rows (y). Possibly one more switch if c0 > c1.
	if c0 > c1:
		return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

	# The following is now always < 1 in abs
	slope = (r1-r0) / (c1-c0)

	# Adjust weight by the slope
	w *= np.sqrt(1+np.abs(slope)) / 2

	# We write y as a function of x, because the slope is always <= 1
	# (in absolute value)
	x = np.arange(c0, c1+1, dtype=float)
	y = x * slope + (c1*r0-c0*r1) / (c1-c0)

	# Now instead of 2 values for y, we have 2*np.ceil(w/2).
	# All values are 1 except the upmost and bottommost.
	thickness = np.ceil(w/2)
	yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
	xx = np.repeat(x, yy.shape[1])
	vals = trapez(yy, y.reshape(-1,1), w).flatten()

	yy = yy.flatten()

	# Exclude useless parts and those outside of the interval
	# to avoid parts outside of the picture
	mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

	return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

def diagonally_truncated_mask(shape, x_threshold, angle):
	result = np.zeros(shape, dtype=bool)
	for x in range(shape[1]):
		for y in range(shape[0]):
			thres = x_threshold * shape[1] - (angle * shape[0] / 2) + y * angle
			result[y, x, ...] = x < thres
	return result

def diagonally_combine_two_images(img1, img2, x_threshold, angle, gap=0, color=1):
	if img2.shape != img1.shape:
		raise ValueError(f"img1 and img2 must have the same shape; {img1.shape} vs {img2.shape}")
	mask = diagonally_truncated_mask(img1.shape, x_threshold, angle)
	result = img2.copy()
	result[mask] = img1[mask]
	if gap > 0:
		rr, cc, val = weighted_line(0, int(x_threshold * img1.shape[1] - (angle * img1.shape[0] / 2)), img1.shape[0]-1, int(x_threshold * img1.shape[1] + (angle * img1.shape[0] / 2)), gap)
		result[rr, cc, :] = result[rr, cc, :] * (1 - val[...,np.newaxis]) + val[...,np.newaxis] * color
	return result

def diagonally_combine_images(images, x_thresholds, angle, gap=0, color=1):
	result = images[0]
	for img, thres in zip(images[1:], x_thresholds):
		result = diagonally_combine_two_images(result, img, thres, angle, gap, color)
	return result

def write_image_imageio(img_file, img, quality):
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:,:,:3]
		kwargs["quality"] = quality
		kwargs["subsampling"] = 0
	imageio.imwrite(img_file, img, **kwargs)

def read_image_imageio(img_file):
	img = imageio.imread(img_file)
	img = np.asarray(img).astype(np.float32)
	if len(img.shape) == 2:
		img = img[:,:,np.newaxis]
	return img / 255.0

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def read_image(file):
	if os.path.splitext(file)[1] == ".bin":
		with open(file, "rb") as f:
			bytes = f.read()
			h, w = struct.unpack("ii", bytes[:8])
			img = np.frombuffer(bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
	else:
		img = read_image_imageio(file)
		if img.shape[2] == 4:
			img[...,0:3] = srgb_to_linear(img[...,0:3])
			# Premultiply alpha
			img[...,0:3] *= img[...,3:4]
		else:
			img = srgb_to_linear(img)
	return img

def write_image(file, img, quality=95):
	ext = os.path.splitext(file)[1]
	if ext == ".bin":
		if img.shape[2] < 4:
			img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
		with open(file, "wb") as f:
			f.write(struct.pack("ii", img.shape[0], img.shape[1]))
			f.write(img.astype(np.float16).tobytes())
	elif ext == '.exr':
		import pyexr
		pyexr.write(file, img)
	else:
		if img.shape[2] == 4:
			img = np.copy(img)
			# Unmultiply alpha
			img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
			img[...,0:3] = linear_to_srgb(img[...,0:3])
		else:
			img = linear_to_srgb(img)
		write_image_imageio(file, img, quality)

def trim(error, skip=0.000001):
	error = np.sort(error.flatten())
	size = error.size
	skip = int(skip * size)
	return error[skip:size-skip].mean()

def luminance(a):
	return 0.2126 * a[:,:,0] + 0.7152 * a[:,:,1] + 0.0722 * a[:,:,2]

def SSIM(a, b):
	def blur(a):
		k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
		x = convolve1d(a, k, axis=0)
		return convolve1d(x, k, axis=1)
	a = luminance(a)
	b = luminance(b)
	mA = blur(a)
	mB = blur(b)
	sA = blur(a*a) - mA**2
	sB = blur(b*b) - mB**2
	sAB = blur(a*b) - mA*mB
	c1 = 0.01**2
	c2 = 0.03**2
	p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
	p2 = (2.0*sAB + c2)/(sA + sB + c2)
	error = p1 * p2
	return error

def L1(img, ref):
	return np.abs(img - ref)

def APE(img, ref):
	return L1(img, ref) / (1e-2 + ref)

def SAPE(img, ref):
	return L1(img, ref) / (1e-2 + (ref + img) / 2.)

def L2(img, ref):
	return (img - ref)**2

def RSE(img, ref):
	return L2(img, ref) / (1e-2 + ref**2)

def rgb_mean(img):
	return np.mean(img, axis=2)

def compute_error_img(metric, img, ref):
	img[np.logical_not(np.isfinite(img))] = 0
	img = np.maximum(img, 0.)
	if metric == "MAE":
		return L1(img, ref)
	elif metric == "MAPE":
		return APE(img, ref)
	elif metric == "SMAPE":
		return SAPE(img, ref)
	elif metric == "MSE":
		return L2(img, ref)
	elif metric == "MScE":
		return L2(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric == "MRSE":
		return RSE(img, ref)
	elif metric == "MtRSE":
		return trim(RSE(img, ref))
	elif metric == "MRScE":
		return RSE(np.clip(img, 0, 100), np.clip(ref, 0, 100))
	elif metric == "SSIM":
		return SSIM(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric in ["FLIP", "\FLIP"]:
		# Set viewing conditions
		monitor_distance = 0.7
		monitor_width = 0.7
		monitor_resolution_x = 3840
		# Compute number of pixels per degree of visual angle
		pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

		ref_srgb = np.clip(flip.color_space_transform(ref, "linrgb2srgb"), 0, 1)
		img_srgb = np.clip(flip.color_space_transform(img, "linrgb2srgb"), 0, 1)
		result = flip.compute_flip(flip.utils.HWCtoCHW(ref_srgb), flip.utils.HWCtoCHW(img_srgb), pixels_per_degree)
		assert np.isfinite(result).all()
		return flip.utils.CHWtoHWC(result)

	raise ValueError(f"Unknown metric: {metric}.")

def compute_error(metric, img, ref):
	metric_map = compute_error_img(metric, img, ref)
	metric_map[np.logical_not(np.isfinite(metric_map))] = 0
	if len(metric_map.shape) == 3:
		metric_map = np.mean(metric_map, axis=2)
	mean = np.mean(metric_map)
	return mean
