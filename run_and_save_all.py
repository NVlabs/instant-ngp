from pathlib import Path
import argparse
import tempfile
import shutil
import json
import os

import time
import tqdm
import numpy as np

import struct
import imageio

import swn.read_write_model as rwm
from swn.data.kitti import KITTI_TEST_SEQS

import pyngp as ngp # noqa


def write_image_imageio(img_file, img, quality):
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:,:,:3]
		kwargs["quality"] = quality
		kwargs["subsampling"] = 0
	imageio.imwrite(img_file, img, **kwargs)

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def write_image(file, img, quality=95):
    if os.path.splitext(file)[1] == ".bin":
        if img.shape[2] < 4:
            img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
        with open(file, "wb") as f:
            f.write(struct.pack("ii", img.shape[0], img.shape[1]))
            f.write(img.astype(np.float16).tobytes())
    else:
        if img.shape[2] == 4:
            img = np.copy(img)
            # Unmultiply alpha
            img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
            img[...,0:3] = linear_to_srgb(img[...,0:3])
        else:
            img = linear_to_srgb(img)
        write_image_imageio(file, img, quality)


def main(args):

    PATH_RAW_DATA = args.path_raw_data
    PATH_DEPTH_GT = args.path_depth_gt
    PATH_COLMAP = args.path_colmap
    PATH_OUTPUT = args.path_output

    for seq in KITTI_TEST_SEQS:
        print(f'Processing sequence: {seq}')

        #####################################################
        # 1. Preprocess the sequence and create scene files #
        #####################################################

        temp_dir = Path(tempfile.mkdtemp())

        scene_file = PATH_OUTPUT / seq
        scene_file.mkdir(parents=True, exist_ok=True)
        scene_file = scene_file / 'transforms.json'

        # Convert COLMAP model into text
        bincam = rwm.read_cameras_binary(PATH_COLMAP / seq / 'colmap' / 'sparse' / 'cameras.bin')
        rwm.write_cameras_text(bincam, temp_dir / 'cameras.txt')
        binimg = rwm.read_images_binary(PATH_COLMAP / seq / 'colmap' / 'sparse' / 'images.bin')
        rwm.write_images_text(binimg, temp_dir / 'images.txt')

        # Run COLMAP2NERF
        os.system(' '.join([
            'python3 scripts/colmap2nerf.py',
            '--images', str(PATH_RAW_DATA / seq[:10] / seq / 'image_02' / 'data'),
            '--text', str(temp_dir),
            '--out', str(scene_file),
        ]))

        with open(scene_file, 'r') as f:
            data = json.load(f)

        for frame in data['frames']:

            img_path = PATH_RAW_DATA / seq[:10] / seq / 'image_02' / 'data' / frame['file_path'][-14:]
            frame['file_path'] = os.path.relpath(str(img_path), scene_file.parent.resolve())

            if args.colmap_depth or args.gt_depth:
                depth_path_1 = PATH_DEPTH_GT / 'val' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / frame['file_path'].split('/')[-1]
                depth_path_2 = PATH_DEPTH_GT / 'train' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / frame['file_path'].split('/')[-1]
                if depth_path_1.is_file():
                    frame['depth_path'] = os.path.relpath(depth_path_1.resolve(), scene_file.parent.resolve())
                elif depth_path_2.is_file():
                    frame['depth_path'] = os.path.relpath(depth_path_2.resolve(), scene_file.parent.resolve())
        
        with open(scene_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        #################
        # 2. Train Nerf #
        #################

        # Args
        sharpen = 0.0
        exposure = 0.0
        n_steps = args.n_steps

        network = Path(__file__).parent / 'configs' / 'nerf' / 'base.json'
        network = network.resolve()
  
        mode = ngp.TestbedMode.Nerf
        testbed = ngp.Testbed(mode)
        testbed.nerf.sharpen = sharpen
        testbed.exposure = exposure
        testbed.load_training_data(str(scene_file.parent))
        testbed.reload_network_from_file(str(network))
        testbed.shall_train = True
        testbed.nerf.render_with_camera_distortion = True

        old_training_step = 0
        n_steps = n_steps

        tqdm_last_update = 0
        if n_steps > 0:
            with tqdm.tqdm(desc="Training", total=n_steps, unit="step") as t:
                while testbed.frame():
                    # What will happen when training is done?
                    if testbed.training_step >= n_steps:
                        break

                    # Update progress bar
                    if testbed.training_step < old_training_step or old_training_step == 0:
                        old_training_step = 0
                        t.reset()

                    now = time.monotonic()
                    if now - tqdm_last_update > 0.1:
                        t.update(testbed.training_step - old_training_step)
                        t.set_postfix(loss=testbed.loss)
                        old_training_step = testbed.training_step
                        tqdm_last_update = now


        ###########################
        # 2. Save Nerf renderings #
        ###########################

        print('Evaluating sequence:', seq)
        with open(scene_file) as f:
            test_transforms = json.load(f)
    
        # Evaluate metrics on black background
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
        # Prior nerf papers don't typically do multi-sample anti aliasing.
        # So snap all pixels to the pixel centers.
        testbed.snap_to_pixel_centers = True
        spp = 8
        testbed.nerf.rendering_min_transmittance = 1e-4
        testbed.fov_axis = 0
        testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
        testbed.shall_train = False

        (scene_file.parent / 'rgb').mkdir(parents=True, exist_ok=True)
        (scene_file.parent / 'depth').mkdir(parents=True, exist_ok=True)

        H, W = int(test_transforms["h"]), int(test_transforms["w"])
        with tqdm.tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
            for i, frame in t:
              
                testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])

                # TODO
                # DO fancy stuff with linear and srgb

                # Render and save RGB
                testbed.render_mode = ngp.Shade
                image = testbed.render(W, H, spp, True)
                write_image(str(scene_file.parent / 'rgb' / frame['file_path'].split('/')[-1]), image)

                # Render and save depth
                testbed.render_mode = ngp.Depth
                image = testbed.render(W, H, spp, True)
                np.save(str(scene_file.parent / 'depth' / frame['file_path'].split('/')[-1][:10]), image[:,:,0])






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run fancy stuff")
    parser.add_argument("--colmap_depth", action="store_true", help="Use COLMAP sparse depth as supervision")
    parser.add_argument("--gt_depth",action="store_true", help="Use GT depth as supervision")
    parser.add_argument("--n_steps", type=int, help="Number of steps in Nerf training", default=20000)
    parser.add_argument("--path_raw_data", type=Path, help="Path to raw KITTI data")
    parser.add_argument("--path_depth_gt", type=Path, help="Path to GT depth data")
    parser.add_argument("--path_colmap", type=Path, help="Path to COLMAP data")
    parser.add_argument("--path_output", type=Path, help="Path to output data")
    main(parser.parse_args())
