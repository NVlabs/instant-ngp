#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import cv2
import json
import numpy as np
import os
import sys
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")

def parse_args():
	parser = argparse.ArgumentParser(description="Generate masks for set of images to exclude objects like cars, persons, animals.")

	parser.add_argument("--images", default="images", help="Input path to the images.")
	parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories that should be masked out from the training images. See `scripts/category2id.json` for supported categories.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	IMAGE_FOLDER = args.images

	if len(args.mask_categories) > 0:
		# Check if detectron2 is installed. If not, install it.
		try:
			import detectron2
		except ModuleNotFoundError:
			try:
				import torch
			except ModuleNotFoundError:
				print("PyTorch is not installed. For automatic masking, install PyTorch from https://pytorch.org/")
				sys.exit(1)

			input("Detectron2 is not installed. Press enter to install it.")
			import subprocess
			package = 'git+https://github.com/facebookresearch/detectron2.git'
			subprocess.check_call([sys.executable, "-m", "pip", "install", package])
			import detectron2

		import torch
		from pathlib import Path
		from detectron2.config import get_cfg
		from detectron2 import model_zoo
		from detectron2.engine import DefaultPredictor

		category2id = json.load(open(os.path.join(SCRIPTS_FOLDER, "category2id.json"), "r"))
		mask_ids = [category2id[c] for c in args.mask_categories]

		cfg = get_cfg()
		# Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
		cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
		# Find a model from detectron2's model zoo.
		cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		predictor = DefaultPredictor(cfg)

		for filename in tqdm(os.listdir(IMAGE_FOLDER), desc="Masking images", unit="images"):
			basename, ext = os.path.splitext(filename)
			ext = ext.lower()

			# Only consider image files
			if ext != ".jpg" and ext != ".jpeg" and ext != ".png" and ext != ".exr" and ext != ".bmp":
				continue

			img = cv2.imread(os.path.join(IMAGE_FOLDER, filename))
			outputs = predictor(img)

			output_mask = np.zeros((img.shape[0], img.shape[1]))
			for i in range(len(outputs['instances'])):
				if outputs['instances'][i].pred_classes.cpu().numpy()[0] in mask_ids:
					pred_mask = outputs['instances'][i].pred_masks.cpu().numpy()[0]
					output_mask = np.logical_or(output_mask, pred_mask)

			cv2.imwrite(os.path.join(IMAGE_FOLDER, f"dynamic_mask_{basename}.png"), (output_mask*255).astype(np.uint8))
