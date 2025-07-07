#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from pathlib import Path


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
