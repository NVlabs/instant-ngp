#!/usr/bin/env python3
# Streaming/Dataset capture script for the NeRFCapture iOS App

import argparse
import cv2
from pathlib import Path
import json
import shutil

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from dataclasses import dataclass
from cyclonedds.domain import DomainParticipant, Domain
from cyclonedds.core import Qos, Policy
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from common import *
import pyngp as ngp  # noqa

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--stream", action="store_true", help="Stream images directly to InstantNGP.")
	parser.add_argument("--n_frames", default=10, type=int, help="Number of frames before saving the dataset. Also used as the number of cameras to remember when streaming.")
	parser.add_argument("--save_path", required='--stream' not in sys.argv, type=str, help="Path to save the dataset.")
	parser.add_argument("--depth_scale", default=10.0, type=float, help="Depth scale used when saving depth. Only used when saving dataset.")
	parser.add_argument("--overwrite", action="store_true", help="Rewrite over dataset if it exists.")
	return parser.parse_args()


# DDS
# ==================================================================================================
@dataclass
@annotate.final
@annotate.autoid("sequential")
class NeRFCaptureFrame(idl.IdlStruct, typename="NeRFCaptureData.NeRFCaptureFrame"):
	id: types.uint32
	annotate.key("id")
	timestamp: types.float64
	fl_x: types.float32
	fl_y: types.float32
	cx: types.float32
	cy: types.float32
	transform_matrix: types.array[types.float32, 16]
	width: types.uint32
	height: types.uint32
	image: types.sequence[types.uint8]
	has_depth: bool
	depth_width: types.uint32
	depth_height: types.uint32
	depth_scale: types.float32
	depth_image: types.sequence[types.uint8]


dds_config = """<?xml version="1.0" encoding="UTF-8" ?> \
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd"> \
    <Domain id="any"> \
        <Internal> \
            <MinimumSocketReceiveBufferSize>10MB</MinimumSocketReceiveBufferSize> \
        </Internal> \
        <Tracing> \
            <Verbosity>config</Verbosity> \
            <OutputFile>stdout</OutputFile> \
        </Tracing> \
    </Domain> \
</CycloneDDS> \
"""
# ==================================================================================================

def set_frame(testbed, frame_idx: int, rgb: np.ndarray, depth: np.ndarray, depth_scale: float, X_WV: np.ndarray, fx: float, fy: float, cx: float, cy: float):
	testbed.nerf.training.set_image(frame_idx = frame_idx, img=rgb, depth_img=depth, depth_scale=depth_scale*testbed.nerf.training.dataset.scale)
	testbed.nerf.training.set_camera_extrinsics(frame_idx=frame_idx, camera_to_world=X_WV)
	testbed.nerf.training.set_camera_intrinsics(frame_idx=frame_idx, fx=fx, fy=fy, cx=cx, cy=cy)


def live_streaming_loop(reader: DataReader, max_cameras: int):
	# Start InstantNGP
	testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
	testbed.init_window(1920, 1080)
	testbed.reload_network_from_file()
	testbed.visualize_unit_cube = True
	testbed.nerf.visualize_cameras = True

	camera_index = 0  # Current camera index we are replacing in InstantNGP
	total_frames = 0 # Total frames received

	# Create Empty Dataset
	testbed.create_empty_nerf_dataset(max_cameras, aabb_scale=1)
	testbed.nerf.training.n_images_for_training = 0
	testbed.up_dir = np.array([1.0, 0.0, 0.0])

	# Start InstantNGP and DDS Loop
	while testbed.frame():
		sample = reader.read_next() # Get frame from NeRFCapture
		if sample:
			print(f"Frame {total_frames + 1} received")

			# RGB
			image = np.asarray(sample.image, dtype=np.uint8).reshape(
				(sample.height, sample.width, 3)).astype(np.float32)/255.0
			image = np.concatenate(
				[image, np.zeros((sample.height, sample.width, 1), dtype=np.float32)], axis=-1)

			# Depth if available
			depth = None
			if sample.has_depth:
				depth = np.asarray(sample.depth_image, dtype=np.uint8).view(
					dtype=np.float32).reshape((sample.depth_height, sample.depth_width))
				depth = cv2.resize(depth, dsize=(
					sample.width, sample.height), interpolation=cv2.INTER_NEAREST)

			# Transform
			X_WV = np.asarray(sample.transform_matrix,
							dtype=np.float32).reshape((4, 4)).T[:3, :].copy()

			# Add frame to InstantNGP
			set_frame(testbed,
					frame_idx=camera_index,
					rgb=srgb_to_linear(image),
					depth=depth,
					depth_scale=1,
					X_WV=X_WV,
					fx=sample.fl_x,
					fy=sample.fl_y,
					cx=sample.cx,
					cy=sample.cy)

			# Update index
			total_frames += 1
			testbed.nerf.training.n_images_for_training = min(total_frames, max_cameras)
			camera_index = (camera_index + 1) % max_cameras

			if total_frames == 1:
				testbed.first_training_view()
				testbed.render_groundtruth = True

def dataset_capture_loop(reader: DataReader, save_path: Path, overwrite: bool, n_frames: int):
	if save_path.exists():
		if overwrite:
			# Prompt user to confirm deletion
			if (input(f"warning! folder '{save_path}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
				sys.exit(1)
			shutil.rmtree(save_path)
		else:
			print(f"save_path {save_path} already exists")
			sys.exit(1)

	print("Waiting for frames...")
	# Make directory
	images_dir = save_path.joinpath("images")

	manifest = {
		"fl_x":  0.0,
		"fl_y":  0.0,
		"cx": 0.0,
		"cy": 0.0,
		"w": 0.0,
		"h": 0.0,
		"frames": []
	}

	total_frames = 0 # Total frames received

	# Start DDS Loop
	while True:
		sample = reader.read_next() # Get frame from NeRFCapture
		if sample:
			print(f"{total_frames + 1}/{n_frames} frames received")

			if total_frames == 0:
				save_path.mkdir(parents=True)
				images_dir.mkdir()
				manifest["w"] = sample.width
				manifest["h"] = sample.height
				manifest["cx"] = sample.cx
				manifest["cy"] = sample.cy
				manifest["fl_x"] = sample.fl_x
				manifest["fl_y"] = sample.fl_y
				manifest["integer_depth_scale"] = float(args.depth_scale)/65535.0

			# RGB
			image = np.asarray(sample.image, dtype=np.uint8).reshape((sample.height, sample.width, 3))
			cv2.imwrite(str(images_dir.joinpath(f"{total_frames}.png")), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

			# Depth if avaiable
			depth = None
			if sample.has_depth:
				depth = np.asarray(sample.depth_image, dtype=np.uint8).view(
					dtype=np.float32).reshape((sample.depth_height, sample.depth_width))
				depth = (depth*65535/float(args.depth_scale)).astype(np.uint16)
				depth = cv2.resize(depth, dsize=(
					sample.width, sample.height), interpolation=cv2.INTER_NEAREST)
				cv2.imwrite(str(images_dir.joinpath(f"{total_frames}.depth.png")), depth)

			# Transform
			X_WV = np.asarray(sample.transform_matrix,
							  dtype=np.float32).reshape((4, 4)).T

			frame = {
				"transform_matrix": X_WV.tolist(),
				"file_path": f"images/{total_frames}",
				"fl_x": sample.fl_x,
				"fl_y": sample.fl_y,
				"cx": sample.cx,
				"cy": sample.cy,
				"w": sample.width,
				"h": sample.height
			}

			if depth is not None:
				frame["depth_path"] = f"images/{total_frames}.depth.png"

			manifest["frames"].append(frame)

			# Update index
			if total_frames == n_frames - 1:
				print("Saving manifest...")
				# Write manifest as json
				manifest_json = json.dumps(manifest, indent=4)
				with open(save_path.joinpath("transforms.json"), "w") as f:
					f.write(manifest_json)
				print("Done")
				sys.exit(0)
			total_frames += 1


if __name__ == "__main__":
	args = parse_args()

	# Setup DDS
	domain = Domain(domain_id=0, config=dds_config)
	participant = DomainParticipant()
	qos = Qos(Policy.Reliability.Reliable(
		max_blocking_time=duration(seconds=1)))
	topic = Topic(participant, "Frames", NeRFCaptureFrame, qos=qos)
	reader = DataReader(participant, topic)

	if args.stream:
		live_streaming_loop(reader, args.n_frames)
	else:
		dataset_capture_loop(reader, Path(args.save_path), args.overwrite, args.n_frames)
