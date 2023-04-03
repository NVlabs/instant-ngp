
import argparse

import json
import os.path as osp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--transforms_json",
    "-j",
    default="",
    help="Path to transforms.json file",
)
parser.add_argument(
    "--sharpness_threshold",
    "--st",
    "--thresh",
    default=0,
    type=int,
    help="threshold to filter out blurry images",
)
parser.add_argument(
    "--sharpness_filter_window",
    "--sw",
    "--filter",
    default=10,
    type=int,
    help="the window size for the sharpness filter",
)
args = parser.parse_args()

with open(args.transforms_json, "r") as readfile:
    transforms_json_data = json.load(readfile)


dirname = osp.dirname(args.transforms_json)
filename = osp.basename(args.transforms_json).split(".")[0]

output = transforms_json_data.copy() 

frames = output["frames"].copy()
output["frames"] = []

frames = sorted(frames, key=lambda x: x["file_path"])

buffer = []

for i, frame in enumerate(frames):
    buffer.append(frame)

    if len(buffer) == args.sharpness_filter_window:
        sharpest = max(buffer, key=lambda x: x["sharpness"])

        if sharpest["sharpness"] >= args.sharpness_threshold:
            output["frames"].append(sharpest)
            buffer.clear()
        else:
            buffer.pop(0)
    
num_frames = len(output["frames"])
print(f"Filtered down to {num_frames} frames")
with open(osp.join(dirname, f"filtered_{filename}.json"), "w") as writefile:
    json.dump(output, writefile, indent=4)
        
