import pymeshfix 
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Fix mesh using PyMeshFix "
)

parser.add_argument(
    "--mesh",
    required=True,
    help="path to mesh file",
)

args = parser.parse_args()

# meshfix = pymeshfix.MeshFix()

# tin = pymeshfix._meshfix.PyTMesh()

# tin.load_file(Path(args.mesh))

# tin.clean(max_iters=100, inner_loops=3)

mesh_path = args.mesh.split('.') 
output_file = mesh_path[0] + "_fixed." + mesh_path[1]

pymeshfix.clean_from_file(args.mesh, output_file)

# with open(Path(output_file)) as f:

#     tin.save_file(f)

