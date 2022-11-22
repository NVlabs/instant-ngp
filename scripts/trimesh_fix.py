import trimesh 
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

mesh = trimesh.load_mesh(Path(args.mesh))

mesh_1 = mesh.copy()
mesh_2 = mesh.copy()
mesh_3 = mesh.copy()
mesh_4 = mesh.copy()

trimesh.smoothing.filter_humphrey(mesh_1)
trimesh.smoothing.filter_laplacian(mesh_2)
trimesh.smoothing.filter_mut_dif_laplacian(mesh_3)
trimesh.smoothing.filter_taubin(mesh_4)

mesh_path = args.mesh.split('.') 
output_path_1 = mesh_path[0] + "_trimesh_humphrey." + mesh_path[1]
output_path_2 = mesh_path[0] + "_trimesh_laplacian." + mesh_path[1]
output_path_3 = mesh_path[0] + "_trimesh_mut_dif_laplacian." + mesh_path[1]
output_path_4 = mesh_path[0] + "_trimesh_taubin." + mesh_path[1]

mesh_1.export(Path(output_path_1))
mesh_2.export(Path(output_path_2))
mesh_3.export(Path(output_path_3))
mesh_4.export(Path(output_path_4))
