#!/usr/bin/python3
"""Constant Information."""
import os
from pathlib import Path
from typing import Final

from utils_3dml.file.aws import is_aws_job

DIR_PATH_3DML: Final[str] = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PATH: Final[str] = os.path.dirname(DIR_PATH_3DML)
HOME_DIR: Final[str] = str(Path.home())+"/"

DATA_DIR: Final[str] = os.path.join(HOME_DIR if is_aws_job() else DIR_PATH_3DML, "data/")
TEST_DIR: Final[str] = os.path.join(DATA_DIR, "test")
