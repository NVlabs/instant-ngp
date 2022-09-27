# Add instant-ngp to the path
import sys
import os
import glob
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(str(ROOT_DIR / "build*" / "**" / "*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(str(ROOT_DIR / "build*" / "**" / "*.so"), recursive=True)]
