import argparse

import json
import os.path as osp
import os 

import shutil 
from tqdm import tqdm 

import cv2 
import imutils
import imutils.video

parser = argparse.ArgumentParser()
parser.add_argument(
    "--images",
    "-i",
    default="",
    help="Path to images from kinect camera",
)
parser.add_argument(
    "--sharpness_threshold",
    "--st",
    "--thresh",
    default=0,
    type=int,
    help="threshold to filter out blurry images",
)
parser.add_argument(
    "--sharpness_filter_window",
    "--sw",
    "--filter",
    default=10,
    type=int,
    help="the window size for the sharpness filter",
)
args = parser.parse_args()



def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def image_sharpness(image):
    smaller = imutils.resize(image, width=640)
    gray = cv2.cvtColor(smaller, cv2.COLOR_BGR2GRAY)
    sharpness = variance_of_laplacian(gray)
    return sharpness

output_dir = args.images + "_filtered"
os.makedirs(output_dir, exist_ok=True)

all_images = sorted(os.listdir(args.images))
count = 0 
buffer = []
for file in tqdm(all_images, desc="Filtering images..."):
    img = cv2.imread(osp.join(args.images, file))

    buffer.append((file, image_sharpness(img)))
    if len(buffer) == args.sharpness_filter_window:
        sharpest = max(buffer, key=lambda x: x[1])

        if sharpest[1] >= args.sharpness_threshold:
            # save image 
            count += 1 
            shutil.copy(osp.join(args.images, sharpest[0]), osp.join(output_dir, sharpest[0]))

            buffer.clear()
        else:
            buffer.pop(0)

print(f"Filtered {len(all_images)} images down to {count} images")
