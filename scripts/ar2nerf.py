#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
from pathlib import Path

import cv2
import imutils
import imutils.video
import numpy as np
import tqdm
from pyquaternion import Quaternion

# TODO
# From Slack comments by Zan Gojcic:
# https://nvidia.slack.com/archives/C02LK48QU31/p1656889021750889?thread_ts=1656685574.397619&cid=C02LK48QU31
# Yes internally the axes will be swapped s.t. x,y,z)_NGP = (y,z,x)_input .
# What confused me was that we are also loading lidar rays, which are saved in the input convention,
# but there is a function nerf_ray_to_ngp that does the axes swap for the rays.
#
# If you would like to disable the conversions (swap and negation) you could just change the functions
# nerf_matrix_to_ngp , ngp_matrix_to_nerf , nerf_ray_to_ngp , nerf_direction_to_ngp  to only apply scale and
# offset only. They are all in nerf_loader.h. Your cameras should be in the opencv (and NGP) convention,
# i.e. Right, Down, Forward (edited)

# args
parser = argparse.ArgumentParser(
    description="Convert CamTrackAR data to a NERF dataset."
)
parser.add_argument(
    "ar_json_fn",
    # default="ar_track.json",
    help="the CamTrackAR JSON file to load",
)
parser.add_argument(
    "video_fn",
    # default="video.mp4",
    help="the video to load",
)
parser.add_argument(
    "--output_dir", "-o", default=".", help="the directory to save the output files to"
)
parser.add_argument(
    "--images_dir",
    default="images_ar",
    help="the directory to save extracted video frames",
)
parser.add_argument(
    "--sharpness_filter_window",
    "--sw",
    default=10,
    type=int,
    help="the window size for the sharpness filter",
)
parser.add_argument(
    "--aabb_scale",
    default=8,
    type=int,
    help="the NGP aabb_scale",
)
parser.add_argument(
    "--out_json_fn",
    default="transforms.json",
    help="the filename to save the output transforms",
)
parser.add_argument(
    "--max_num_frames",
    default=None,
    type=int,
    help="the maximum number of frames to load",
)
# get scale_factor
parser.add_argument(
    "--scale_factor",
    default=None,
    type=float,
    help="the scale factor to apply to camera positions",
)

parser.add_argument(
    "--sharpness_threshold",
    "--st",
    default=0,
    type=int,
    help="threshold to filter out blurry images",
)
args = parser.parse_args()

# ensure paths exist
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
images_dir = Path(args.images_dir)
(output_dir / images_dir).mkdir(exist_ok=True)

# helper functions
# (from https://answers.opencv.org/question/214981/calculating-intrinsic-parameters-from-current-optical-zoom/)
def fov2fl(fov, size):
    return size / 2 / np.tan(fov / 2)


def fl2fov(fl, size):
    return 2 * np.arctan(size / 2 / fl)


# load CamTrackAR output
with open(args.ar_json_fn) as fp:
    ar_track = json.load(fp)
ar_frames = ar_track["cameras"][0]["keyframes"]

ar_video = ar_track["videoData"]
n_frames = ar_video["numFrames"]
assert n_frames == len(ar_frames)
if args.max_num_frames is not None and args.max_num_frames < n_frames:
    n_frames = args.max_num_frames
    print(f"Using first {n_frames} frames")

ar_anchors = ar_track["anchors"]
ar_camera_positions = np.array([frame["position"] for frame in ar_frames])

# translate the camera positions such that the x-z origin is at the median camera position
median_camera_position = np.median(ar_camera_positions, axis=0)
median_camera_position[1] = 0.0
ar_camera_positions_centered = ar_camera_positions - median_camera_position
print(
    f"Centered the camera positions around the median camera position (excluding the vertical axis): \n\t{median_camera_position}"
)

# scale the camera positions such that the range is [-std/2, std/2]
if args.scale_factor is None:
    scale_factor = ar_camera_positions_centered.std() / 2.0
    scale_source = "[-std/2, std/2] of camera positions"
else:
    scale_factor = args.scale_factor
    scale_source = f"user-specified scale factor"
ar_camera_positions_scaled = ar_camera_positions_centered / scale_factor
print(f"Scaled the camera positions to {scale_source}: {scale_factor}")

# define a rotation matrix to swap axes
axis_rot_mtx = np.array([[0.0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

# record normalized camera positions
for frame, normalized_position in zip(ar_frames, ar_camera_positions_scaled):
    frame["position_normalized"] = normalized_position

# get camera parameters
width = ar_video["compWidth"]
height = ar_video["compHeight"]
focal_length = ar_frames[0]["lensZoom"]
fov_x = ar_frames[0]["fov"]
fov_y = fl2fov(focal_length, height)


# sharpness
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def image_sharpness(image):
    smaller = imutils.resize(image, width=640)
    gray = cv2.cvtColor(smaller, cv2.COLOR_BGR2GRAY)
    sharpness = variance_of_laplacian(gray)
    return sharpness


# initialize transforms.json
transforms_json = {
    "source": "CamTrackAR",
    "camera_angle_x": fov_x,
    "camera_angle_y": fov_y,
    "fl_x": focal_length,
    "fl_y": focal_length,
    "k1": 0.0,
    "k2": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "cx": width / 2,
    "cy": height / 2,
    "w": width,
    "h": height,
    "aabb_scale": args.aabb_scale,
    "scale_factor": scale_factor,
    "scale_source": scale_source,
    "frames": [],
}


# iterate over frames in video using imutils
video = imutils.video.FileVideoStream(
    args.video_fn,
    transform=lambda im: (im, image_sharpness(im) if im is not None else (None, 0)),
).start()
frame_queue = []
print(f"Processing video with {n_frames} frames...")
for idx in tqdm.tqdm(range(n_frames)):
    # read frame
    image, sharpness = video.read()

    # package frame
    frame = {
        "idx": idx,
        "image": image,
        "sharpness": sharpness,
        "ar_info": ar_frames[idx],
    }
    frame_queue.append(frame)

    # keep sharpest image in window
    if len(frame_queue) >= args.sharpness_filter_window:
        # get sharpest frame
        frames_by_sharpness = [(frame["sharpness"], frame) for frame in frame_queue]
        sharpeness_val, sharpest = max(frames_by_sharpness)

        if sharpeness_val > args.sharpness_threshold:
            idx, image, sharpness, ar_info = sharpest.values()

            # save frame
            image_fn = images_dir / f"{idx:04d}.png"
            cv2.imwrite(str(output_dir / image_fn), image)

            # compute transform matrix
            transform_matrix = np.eye(4)
            x, y, z, w = ar_info["rotationQuat"]  # CamTrackAR uses xyzw quaternions
            transform_matrix[:3, :3] = Quaternion(w, x, y, z).rotation_matrix
            transform_matrix[:3, -1] = np.array(ar_info["position_normalized"])
            transform_matrix = axis_rot_mtx @ transform_matrix

            # frame descriptor for transforms.json
            frame_transform = {
                "file_path": str(image_fn),
                "sharpness": sharpness,
                "transform_matrix": transform_matrix.tolist(),
            }
            transforms_json["frames"].append(frame_transform)

        # clear frame queue
        frame_queue = []
print(f'Frames written to "{output_dir / images_dir}".')
video.stop()

# output transforms.json
with open(output_dir / args.out_json_fn, "w") as fp:
    json.dump(transforms_json, fp, indent=2)
print(f"Results written to {output_dir / args.out_json_fn}.")
