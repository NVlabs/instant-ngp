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
import numpy as np
import rich.traceback
import tqdm
from renderer import NerfRenderer, RenderMode
from rich.console import Console


# function to convert float array to uint8 array
def float_to_uint8(x):
    return (x * 255).astype(np.uint8)


# install rich for print method and traceback handler
console = Console()
print = console.print
rich.traceback.install(show_locals=False)

# parse command line args
parser = argparse.ArgumentParser(
    description="Use NGP depth map renderings to compute object masks from oriented bounding boxes"
)
parser.add_argument(
    "--scene_path",
    default=".",
    help="base path to the scene",
)
parser.add_argument(
    "--scene_transforms",
    default="transforms_optimized_extrinsics.json",
    help="scene transforms file",
)
parser.add_argument(
    "--snapshot",
    default="base.msgpack",
    help="saved weights snapshot",
)
parser.add_argument(
    "--crop_settings",
    default="crop_settings.json",
    help="file with crop settings for objects in scene",
)
parser.add_argument(
    "--interactive_crop",
    action="store_true",
    help="prompt user to manually set the crop",
)
parser.add_argument(
    "--binarize",
    default=None,
    type=float,
    help="binarize masks at the given threshold",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="overwrite existing masks, if any",
)
parser.add_argument(
    "--save_masked_orig",
    action="store_true",
    help="save copy of original images with masked alpha channel",
)
parser.add_argument(
    "--eps",
    default=0.005,
    type=float,
    help="epsilon for mask computation",
)
args = parser.parse_args()

# convert arg paths to Path objects and append scene path
args.scene_path = Path(args.scene_path)
args.scene_transforms = args.scene_path / Path(args.scene_transforms)
args.snapshot = args.scene_path / Path(args.snapshot)
args.crop_settings = args.scene_path / Path(args.crop_settings)

# load scene NeRF
nerf = NerfRenderer(args.scene_transforms, args.snapshot)

# load scene transforms for list of camera poses
with open(args.scene_transforms) as f:
    scene_transforms = json.load(f)
camera_views = scene_transforms["frames"]

# if interactive crop, prompt user to manually set the crop
# and save the settings to the crop settings file
if args.interactive_crop:
    assert not args.crop_settings.exists() or args.overwrite, (
        f"Crop settings file [red]{args.crop_settings}[/red] already exists. "
        + f"Use --overwrite to overwrite when using --interactive_crop."
    )
    crop_settings = {}
    while True:
        print("Please set the crop manually and then close the NGP window to save.")
        nerf.interactive()
        crop = nerf.get_crop()
        crop_settings[f"object_{len(crop_settings)}"] = crop
        if input("Create another crop? [y/N] ").lower() != "y":
            break
    with open(args.crop_settings, "w") as fp:
        json.dump(crop_settings, fp, indent=2)

# load object crop settings
with open(args.crop_settings) as f:
    crop_settings = json.load(f)
object_crop_settings = {k: v for k, v in crop_settings.items() if "object" in k}
default_crop_setting = nerf.get_crop()

# iterate over objects in scene
# TODO add progress bar
for object_name, object_crop_setting in object_crop_settings.items():
    # make directory for object masks
    object_mask_dir = args.scene_path / "masks" / object_name
    print(f"Making directory [green]{object_mask_dir}[/green]")
    if object_mask_dir.exists() and not args.overwrite:
        print(f"[red]{object_mask_dir}[/red] already exists. Skipping...")
        continue
    object_mask_dir.mkdir(parents=True, exist_ok=args.overwrite)

    # iterate over camera views
    for camera_view in tqdm.tqdm(
        sorted(camera_views, key=lambda x: x["file_path"]),
        desc=f"Computing {object_name} masks",
    ):
        # render nerf depth map with and without object crop
        nerf.apply_crop_from_dict(default_crop_setting)
        scene_depth, scene_alpha = nerf.render(
            camera_view["transform_matrix"],
            RenderMode.Depth,
            # output_file="scene_depth.png",
            # overwrite=args.overwrite,
        )
        nerf.apply_crop_from_dict(object_crop_setting)
        object_depth, object_alpha = nerf.render(
            camera_view["transform_matrix"],
            RenderMode.Depth,
            # output_file="object_depth.png",
            # overwrite=args.overwrite,
        )

        # compute object mask from depth maps
        object_mask = object_alpha
        object_mask_visib = (object_depth + args.eps >= scene_depth) * object_alpha

        # binarize masks, if requested
        if args.binarize is not None:
            assert (
                args.binarize > 0.0 and args.binarize <= 1.0
            ), "binarization threshold (--binarize) must be between 0 and 1"
            object_mask = object_mask > args.binarize
            object_mask_visib = object_mask_visib > args.binarize

        # TODO erode and dilate the masks?

        # save object masks
        camera_view_fn = Path(camera_view["file_path"])

        object_mask_path = object_mask_dir / f"{camera_view_fn.stem}.png"
        if args.overwrite and object_mask_path.exists():
            object_mask_path.unlink()
        cv2.imwrite(str(object_mask_path), float_to_uint8(object_mask))

        object_mask_visib_path = object_mask_dir / f"{camera_view_fn.stem}_visib.png"
        if args.overwrite and object_mask_visib_path.exists():
            object_mask_visib_path.unlink()
        cv2.imwrite(str(object_mask_visib_path), float_to_uint8(object_mask_visib))

        # write masked original image
        if args.save_masked_orig:
            orig_image_path = args.scene_path / camera_view_fn
            object_mask_orig_image_path = (
                object_mask_dir / f"{camera_view_fn.stem}_orig_visib.png"
            )

            orig_image = cv2.imread(str(orig_image_path))
            b, g, r = cv2.split(orig_image)
            masked_img = cv2.merge([b, g, r, float_to_uint8(object_mask_visib)], 4)

            if args.overwrite and object_mask_orig_image_path.exists():
                object_mask_orig_image_path.unlink()
            cv2.imwrite(str(object_mask_orig_image_path), masked_img)
