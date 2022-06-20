import argparse
import json
import math
import os
import re

import numpy as np

from pathlib import Path

# local imports
from common import write_image

# pyngp
import pyngp as ngp
ngp.BoundingBox()
# convenience method to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Render script")

    parser.add_argument("--snapshot", required=True, type=str, help="Snapshot file (.msgpack) to use for rendering.")

    parser.add_argument("--frames_json", required=True, type=str, help="Path to a nerf-style transforms.json containing frames to render.")
    parser.add_argument("--frames_path", required=True, type=str, help="Path to a folder to save the rendered frames.")
    parser.add_argument("--overwrite_frames", action="store_true", help="If enabled, images in the `--images_path` will be overwritten.  If not enabled, frames that already exist will not be re-rendered.")
    
    parser.add_argument("--samples_per_pixel", type=int, default=16, help="Number of samples per pixel.")

    parser.add_argument("--video_out", type=str, help="Path to a video to be exported. Uses ffmpeg to combine frames in order.")
    parser.add_argument("--video_fps", type=str, default="30", help="Use in combination with `--video_out`. Sets the fps of the output video.")

    return parser.parse_args()

def safe_str(string_like) -> str:
    return re.sub(r'([\" \'])', r'\\\1', str(string_like))

if __name__ == "__main__":
    args = parse_args()

    # load render json
    render_data = {}
    with open(args.frames_json, 'r') as json_file:
        render_data = json.load(json_file)
    
    # initialize testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(args.snapshot)
    testbed.shall_train = False
    
    testbed.fov_axis = 0
    testbed.fov = math.degrees(render_data["camera_angle_x"])

    # global render props
    frame_width = int(render_data["w"])
    frame_height = int(render_data["h"])
    render_spp = args.samples_per_pixel

    # prepare frames directory
    frames_path = Path(args.frames_path)
    frames_path.mkdir(exist_ok=True)
    rendered_frame_paths = []

    # render each frame via testbed
    for frame in render_data["frames"]:

        # prepare output_path
        frame_path = Path(frame["file_path"])
        if frame_path.suffix == '':
            frame_path = f"{frame_path}.png"

        output_path = frames_path / frame_path
        rendered_frame_paths.append(output_path)
        
        print(f"Rendering frame: {output_path}")

        # check if we can skip rendering this frame
        if not args.overwrite_frames and output_path.exists():
            print(f"Frame already exists! Skipping...")
            continue
        
        # save some convenience properties from the frame json
        cam_matrix = frame["transform_matrix"]

        # prepare testbed to render this frame
        testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
        testbed.render_aabb = ngp.BoundingBox([-1, -1, -1], [1, 1, 1])
        
        # render the frame
        image = testbed.render(frame_width, frame_height, render_spp, True)

        # save frame as image
        write_image(output_path, image)
    
    # combine frames into a video via ffmpeg
    if args.video_out != None:
        video_path = Path(args.video_out)
        video_path.unlink(missing_ok=True)
        fps = args.video_fps

        # fetch all images and save to a playlist
        playlist_path = video_path.parent / f"{video_path.stem}-playlist.txt"
        playlist_path.unlink(missing_ok=True)
        print(playlist_path)

        # prepare ffmpeg playlist.txt, each line is `file 'path/to/image'`
        ffmpeg_files = [f"file '{safe_str(p.absolute())}'" for p in rendered_frame_paths]
        playlist_str = "\n".join(ffmpeg_files)

        with open(playlist_path, "w+") as f:
            f.write(playlist_str)
        
        os.system(f"\
            ffmpeg \
                -f concat \
                -safe 0 \
                -r {fps} \
                -i \"{playlist_path}\" \
                -c:v libx264 \
                -pix_fmt yuv420p \
                -vf fps={fps} \
                \"{video_path}\" \
            ")
        
        playlist_path.unlink(missing_ok=True)

