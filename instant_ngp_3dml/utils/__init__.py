#!/usr/bin/python3
"""Constant Information."""
import os
from pathlib import Path
from typing import Final

from utils_3dml.file.aws import is_aws_job

DIR_PATH: Final[str] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
HOME_DIR: Final[str] = str(Path.home())+"/"

DATA_DIR: Final[str] = os.path.join(HOME_DIR if is_aws_job() else DIR_PATH, "data/")
NERF_CONFIG: Final[str] = os.path.join(DIR_PATH, "configs", "nerf", "{config}.json")
