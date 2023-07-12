#!/usr/bin/python3
"""NeRF Utils Software."""
import sys
from typing import Callable
from typing import Dict

import fire

from instant_ngp_3dml.rendering import main as render
from instant_ngp_3dml.training import main as train

modules: Dict[str, Callable] = {
    "rendering": render,
    "training": train
}

HELP = """Geometry
positional arguments:
  """+str(list(modules.keys()))+"""     Software name
optional arguments:
  -h, --help            show this help message and exit
other:
    software_arguments  the arguments of software
"""

if __name__ == "__main__":
    argv = sys.argv

    if len(argv) > 1 and argv[1] in modules:
        fire.Fire(modules[argv[1]], command=argv[2:])
    else:
        print(HELP)
        sys.exit(1)
