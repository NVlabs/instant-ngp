#!/usr/bin/python3
"""Instant NGP."""
import glob
import os
import sys

from utils_3dml.monitoring.log import Logger

from instant_ngp_3dml.utils import DIR_PATH

# Add pyngp to PYTHONPATH
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(DIR_PATH, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(DIR_PATH, "build*", "**/*.so"), recursive=True)]

logger = Logger("3dml-instant-ngp")
