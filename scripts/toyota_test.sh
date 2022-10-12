
#Train the model with selected configurations
python ./scripts/run.py --mode nerf --train \
                        --scene ./data/nerf/toyota \
                        --network hashgrid.json \
                        --save_snapshot ./data/nerf/toyota/hashgrid30000.msgpack \
                        --n_steps 30000

#Render 3 RGB frames from toyota imageset
python ./scripts/run.py --mode nerf \
                        --scene ./data/nerf/toyota \
                        --load_snapshot ./data/nerf/toyota/hashgrid30000.msgpack \
                        --screenshot_transforms ./data/nerf/toyota/transforms.json \
                        --screenshot_frames 4 12 24 \
                        --screenshot_dir ./data/nerf/toyota/screenshots/ \
                        --width 2048 --height 1080

#Render 3 depth frames from toyota imageset
python ./scripts/run.py --mode nerf --depth \
                        --scene ./data/nerf/toyota \
                        --load_snapshot ./data/nerf/toyota/hashgrid30000.msgpack \
                        --screenshot_transforms ./data/nerf/toyota/transforms.json \
                        --screenshot_frames 4 12 24 \
                        --screenshot_dir ./data/nerf/toyota/screenshots/ \
                        --width 2048 --height 1080 \
                        --exposure -5


