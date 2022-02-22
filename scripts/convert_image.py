#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import common
import numpy as np
import os
import PIL

def parse_args():
	parser = argparse.ArgumentParser(description="Convert image into a different format. By default, converts to our binary fp16 '.bin' format, which helps quickly load large images.")
	parser.add_argument("--input", default="", help="Path to the image to convert.")
	parser.add_argument("--output", default="", help="Path to the output. Defaults to <input>.bin")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	PIL.Image.MAX_IMAGE_PIXELS = 10000000000
	print(f"Loading {args.input}")
	img = common.read_image(args.input)
	print(f"{img.shape[1]}x{img.shape[0]} pixels, {img.shape[2]} channels")

	if not args.output:
		output = os.path.splitext(args.input)[0] + ".bin"
	else:
		output = args.output

	print(f"Writing {output}")
	common.write_image(output, img.astype(np.float16))
