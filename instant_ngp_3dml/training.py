#!/usr/bin/python3
"""Training Script."""
import time

import pyngp as ngp  # noqa
from tqdm import tqdm
from utils_3dml.file.json_utils import write_json
from utils_3dml.monitoring.profiler import profile
from utils_3dml.utils.asserts import assert_gt
from utils_3dml.utils.dataclass import _asdict_inner

from instant_ngp_3dml import logger
from instant_ngp_3dml.utils.training_info import StepInfo
from instant_ngp_3dml.utils.training_info import TrainingInfo


def __train(testbed: ngp.Testbed, n_steps: int, enable_depth_supervision: bool) -> TrainingInfo:

    old_training_step = 0
    steps_info = []
    begin_time = time.monotonic()
    tqdm_last_update = 0.0
    with tqdm(desc="Training", total=n_steps, unit="step") as t:
        while testbed.frame():

            # What will happen when training is done?
            if testbed.training_step >= n_steps:
                break

            # Update progress bar
            if testbed.training_step < old_training_step or old_training_step == 0:
                old_training_step = 0
                t.reset()

            if enable_depth_supervision:
                depth_supervision_lambda = max(1.0 - testbed.training_step / 2000, 0.2)
                testbed.nerf.training.depth_supervision_lambda = depth_supervision_lambda
            else:
                depth_supervision_lambda = 0.0

            now = time.monotonic()

            steps_info.append(StepInfo(step=testbed.training_step,
                                       loss=testbed.loss,
                                       time=now,
                                       depth_supervision_lambda=depth_supervision_lambda))

            if now - tqdm_last_update > 0.1:
                t.update(testbed.training_step - old_training_step)
                t.set_postfix(loss=testbed.loss, depth=depth_supervision_lambda)
                old_training_step = testbed.training_step
                tqdm_last_update = now

    end_time = time.monotonic()

    return TrainingInfo(begin_time=begin_time,
                        end_time=end_time,
                        steps_info=steps_info,
                        n_steps=n_steps,
                        enable_depth_supervision=enable_depth_supervision)


@profile
def main(nerf_transform_json: str,
         nerf_network_configuration_json: str,
         out_snapshot_msgpack: str,
         out_training_info_json: str = "",
         snapshot_msgpack: str = "",
         n_steps: int = 100000,
         enable_depth_supervision: bool = False):
    """Train NeRF Scene.

    Args:
        nerf_transform_json: Input NeRF Transform Json
        nerf_network_configuration_json: Input configuration for NeRF Network
        out_snapshot_msgpack: Output NeRF Weight
        out_training_info_json: Output Json with Training Information
        snapshot_msgpack: Optional Input NeRF Weight
        n_steps: Nb training iterations
        enable_depth_supervision: If specified, NeRF is train with Depth Supervision

    """
    # pylint: disable=too-many-arguments,no-member
    assert_gt(n_steps, 0)

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    testbed.load_training_data(nerf_transform_json)
    testbed.reload_network_from_file(nerf_network_configuration_json)

    if snapshot_msgpack != "":
        logger.info(f"Loading snapshot {snapshot_msgpack}")
        testbed.load_snapshot(snapshot_msgpack)

    testbed.shall_train = True
    testbed.nerf.render_with_camera_distortion = True

    if not enable_depth_supervision:
        testbed.nerf.training.depth_supervision_lambda = 0.0

    info = __train(testbed, n_steps, enable_depth_supervision)

    if out_snapshot_msgpack != "":
        logger.info(f"Saving snapshot {out_snapshot_msgpack}")
        testbed.save_snapshot(out_snapshot_msgpack, False)

    if out_training_info_json != "":
        logger.info(f"Save training info {out_training_info_json}")
        write_json(out_training_info_json, _asdict_inner(info), pretty=True)
