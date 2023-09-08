#!/usr/bin/python3
"""Nerf network Config."""

import os
from typing import Final
from typing import Set

import pyngp as ngp  # noqa
from utils_3dml.file.extensions import FileExt
from utils_3dml.file.file import list_files
from utils_3dml.monitoring.decorators import cache
from utils_3dml.utils.asserts import assert_in

CONFIG_FOLDER :Final[str]= "configs/nerf"

@cache
def get_available_nerf_configs()-> Set[str]:
    """List available nerf network configuration names."""
    return set(FileExt.remove_ext(name) for name in list_files(CONFIG_FOLDER))

def get_nerf_config_json(config_name:str) ->str:
    """Get path to the corresponding NeRF network JSON config."""
    config_name = config_name.lower()
    assert_in(config_name, get_available_nerf_configs() )
    return os.path.join(CONFIG_FOLDER, f"{config_name}.json")