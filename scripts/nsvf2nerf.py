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

import numpy as np
import json
import sys
import math
import cv2
import glob

def parse_args():
	parser = argparse.ArgumentParser(description="convert a dataset from the nsvf paper format to nerf format transforms.json")

	parser.add_argument("--aabb_scale", default=1, help="large scene scale factor")
	parser.add_argument("--white_transparent", action="store_true", help="White is transparent")
	parser.add_argument("--black_transparent", action="store_true", help="White is transparent")
	args = parser.parse_args()
	return args

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

if __name__ == "__main__":
	args = parse_args()
	AABB_SCALE = int(args.aabb_scale)
	SKIP_EARLY = 0
	IMAGE_FOLDER = "."
	img_files = [[],[],[]]
	img_files[0] = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "rgb", f"0_*.png")))
	img_files[1] = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "rgb", f"1_*.png")))
	img_files[2] = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "rgb", f"2_*.png")))
	xx = open("bbox.txt").readline().strip().split(" ")
	xx = [x for x in xx if x] # remove empty elements
	bbox = tuple(map(float,xx))

	image = cv2.imread(img_files[0][0],cv2.IMREAD_UNCHANGED)
	w = image.shape[1]
	h = image.shape[0]
	if (image.shape[2] == 3 or (image.shape[2] == 4 and image[0][0][3] != 0)):
		x = w-1
		if (image[0][0][0] == 0 and image[0][0][1] == 0 and image[0][0][2] == 0):
			print("black opaque background detected")
			args.black_transparent=True
		elif (image[0][0][0] == 255 and image[0][0][1] == 255 and image[0][0][2] == 255):
			print("white opaque background detected")
			args.white_transparent=True
		elif (image[0][x][0] == 0 and image[0][x][1] == 0 and image[0][x][2] == 0):
			print("black opaque background detected")
			args.black_transparent=True
		elif (image[0][x][0] == 255 and image[0][x][1] == 255 and image[0][x][2] == 255):
			print("white opaque background detected")
			args.white_transparent=True
		else:
			print("cant detect background")
			exit()
	elif (image.shape[2] == 4):
		print("transparent alpha channel detected, first pixel alpha = ", image[0][0][3])

	lines = map(str.strip,open("intrinsics.txt","r").readlines())
	els = tuple(map(float, " ".join(lines).split(" ")))
	print(els)
	if len(els) == 11:
		fl_x = els[0]
		fl_y = els[0]
		cx = els[1]
		cy = els[2]
	elif len(els) == 16:
		angle_x=math.pi/2
		fl_x = els[0]
		fl_y = els[5]
		cx = els[2]
		cy = els[6]
	else:
		print("dont understand intrinsics file", els)
		exit()
	# fl = 0.5 * w / tan(0.5 * angle_x);
	angle_x = math.atan(w/(fl_x*2))*2
	angle_y = math.atan(h/(fl_y*2))*2
	fovx = angle_x*180/math.pi
	fovy = angle_y*180/math.pi
	k1 = 0
	k2 = 0
	p1 = 0
	p2 = 0

	print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2}")
	centroid = [(bbox[0]+bbox[3])*0.5,(bbox[1]+bbox[4])*0.5,(bbox[2]+bbox[5])*0.5]
	print("bbox is ", bbox)
	print("centroid is ", centroid)
	radius = [(bbox[3]-bbox[0])*0.5,(bbox[4]-bbox[1])*0.5,(bbox[5]-bbox[2])*0.5]
	scale = 0.5/np.max(radius)
	print("radius is ", np.max(radius))

	for itype in [0,1,2]:
		if (img_files[2]):
			OUT_PATH = ["transforms_train.json", "transforms_val.json", "transforms_test.json"][itype]
		else:
			OUT_PATH = ["transforms_train.json", "transforms_test.json", ""][itype]
		if OUT_PATH == "":
			break
		out = {
			"camera_angle_x": angle_x,
			"camera_angle_y": angle_y,
			"fl_x": fl_x,
			"fl_y": fl_y,
			"k1": k1,
			"k2": k2,
			"p1": p1,
			"p2": p2,
			"cx": cx,
			"cy": cy,
			"w": w,
			"h": h,
			"scale": 1,
			"white_transparent": args.white_transparent,
			"black_transparent": args.black_transparent,
			"aabb_scale": AABB_SCALE,"frames":[]
		}
		for img_f in img_files[itype]:
			pose_f = os.path.join(IMAGE_FOLDER,"pose",os.path.splitext(os.path.basename(img_f))[0]+".txt")
			elems = tuple(map(float," ".join(open(pose_f).readlines()).split(" ")))
			name = img_f
			m = np.array(elems).reshape(4,4)
			b = sharpness(name)
			#print(name, "sharpness=",b)
			c2w = m # np.linalg.inv(m)
			c2w[0:3,3] -= centroid
			c2w[0:3,3] *= scale
			#print(name,c2w)
			c2w[0:3,2] *= -1 # flip the y and z axis
			c2w[0:3,1] *= -1
			c2w = c2w[[0,2,1,3],:] # swap y and z 012 201 102
			c2w[2,:] *= -1 # flip whole world upside down

			frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
			out["frames"].append(frame)

		nframes = len(out["frames"])


		for f in out["frames"]:
			f["transform_matrix"] = f["transform_matrix"].tolist()
		print(nframes,"frames")
		print(f"writing {OUT_PATH}...")
		with open(OUT_PATH, "w") as outfile:
			json.dump(out, outfile, indent=2)
