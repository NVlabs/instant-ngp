#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import common
from PIL import Image
import pyexr as exr
import numpy as np

import struct

from common import read_image, write_image

def parse_args():
	parser = argparse.ArgumentParser(description="Convert image into testbed binary fp16 format (helps quickly load super large images).")
	parser.add_argument("--image", default="", help="Path to the image to train from.")
	parser.add_argument("--exr_out", action="store_true", help="output half precision exr")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	Image.MAX_IMAGE_PIXELS = 10000000000
	print(f"loading {args.image}...")
	img = common.read_image(args.image)
	print(f"{img.shape[1]} x {img.shape[0]} pixels, {img.shape[2]} channels")
	if args.exr_out:
		outpath = os.path.splitext(args.image)[0] + ".exr"
		print(f"writing {outpath}...")
		common.write_image(outpath, img.astype(np.float16))
		exit()

	img = read_image(args.image)
	write_image(os.path.splitext(args.image)[0] + ".bin", img)
