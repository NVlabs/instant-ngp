"""Test Multi-Cam rendering."""
import os
from typing import Dict
from typing import Final
from typing import List
from typing import Type

import cv2
import numpy as np
import pyngp as ngp  # noqa
from utils_3dml.camera_extrinsics.coordinate_system import CoordinateSystem
from utils_3dml.camera_extrinsics.matrix.pose_matrix import PoseMatrix
from utils_3dml.camera_extrinsics.matrix.translation_vector import TranslationVector
from utils_3dml.camera_extrinsics.pose import Pose
from utils_3dml.camera_intrinsics.colmap_intrinsics import focal_to_fov
from utils_3dml.monitoring.profiler import profile
from utils_3dml.structure.nerf.nerf_frame import NerfFrame
from utils_3dml.structure.nerf.nerf_frame import NerfHalfLatLongFrame
from utils_3dml.structure.nerf.nerf_frame import NerfLatLongFrame
from utils_3dml.structure.nerf.nerf_frame import NerfOpencvFrame
from utils_3dml.structure.nerf.nerf_frame import NerfPerspectiveFrame
from utils_3dml.structure.nerf.nerf_transforms import NerfTransforms
from utils_3dml.utils.asserts import assert_eq
from utils_3dml.utils.asserts import assert_len
from utils_3dml.utils.asserts import assert_np_close
from utils_3dml.utils.asserts import assert_same_keys

from instant_ngp_3dml.utils import TEST_DIR

DISTORTION_MODES: Final[Dict[Type[NerfFrame], ngp.LensMode]] = {
    NerfLatLongFrame: ngp.LensMode.LatLong,
    NerfHalfLatLongFrame: ngp.LensMode.HalfLatLong,
    NerfPerspectiveFrame: ngp.LensMode.Perspective,
    NerfOpencvFrame: ngp.LensMode.OpenCV
}
# N.B. LatLong refers to the actual theta/phi parametrization of a spherical image,
# while the Equirectangular mode dilates differently the Y axis

MULTI_CAM_TEST_DIR = os.path.join(TEST_DIR, "multi_camera")


def _get_lens_params(frame: NerfFrame) -> np.ndarray:
    if isinstance(frame, NerfOpencvFrame):
        return np.array((frame.k1, frame.k2, frame.p1, frame.p2, 0.0, 0.0, 0.0))
    return np.zeros((7,), dtype=float)


@profile
def test_multi_cam_rendering_intrinsics():  # noqa: PLR0915
    """Test Multi Cameras Rendering Intrinsics."""
    # Check if set_camera_to_training_view is setting the correct intrinsics

    # GIVEN
    rng = np.random.default_rng(42)  # Fix random seed

    nerf_transform_json = os.path.join(MULTI_CAM_TEST_DIR, "nerf_transform.json")
    IMAGES_FOLDER = os.path.join(MULTI_CAM_TEST_DIR, "image")
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    n_cameras = 10
    nerf_types = [nerf_type
                  for _ in range(n_cameras)
                  for nerf_type in [NerfLatLongFrame, NerfHalfLatLongFrame, NerfPerspectiveFrame, NerfOpencvFrame]]

    nerf_frames: List[NerfFrame] = []
    for nerf_type in nerf_types:
        h = rng.integers(low=50, high=100)

        if nerf_type == NerfLatLongFrame:
            w = 2*h
        elif nerf_type == NerfHalfLatLongFrame:
            w = 4*h
        else:
            w = rng.integers(low=80, high=200)

        file_path = os.path.join(IMAGES_FOLDER, f"image_{len(nerf_frames):04d}.png")
        cv2.imwrite(file_path, np.zeros((h, w, 3), dtype=np.uint8))

        transform_matrix = Pose.random(rng).transform_matrix().tolist()
        sharpness = 421.5

        if nerf_type == NerfLatLongFrame:
            nerf_frames.append(NerfLatLongFrame(w=w, h=h, fl_x=1.0, fl_y=1.0, file_path=file_path,
                                                transform_matrix=transform_matrix, sharpness=sharpness))
        elif nerf_type == NerfHalfLatLongFrame:
            nerf_frames.append(NerfHalfLatLongFrame(w=w, h=h, cy=h, fl_x=1.0, fl_y=1.0, file_path=file_path,
                                                    transform_matrix=transform_matrix, sharpness=sharpness))
        else:
            cx = rng.uniform(low=0.3*w, high=0.6*w)
            cy = rng.uniform(low=0.4*h, high=0.7*h)
            fl_x = rng.uniform(low=0.5*w, high=0.6*w)
            fl_y = rng.uniform(low=0.2*h, high=0.4*h)

            if nerf_type == NerfPerspectiveFrame:
                nerf_frames.append(NerfPerspectiveFrame(w=w, h=h, cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y,
                                                        file_path=file_path, transform_matrix=transform_matrix,
                                                        sharpness=sharpness))
            else:
                assert nerf_type == NerfOpencvFrame
                nerf_frames.append(NerfOpencvFrame(w=w, h=h, cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y, file_path=file_path,
                                                   transform_matrix=transform_matrix, sharpness=sharpness,
                                                   k1=rng.uniform(low=-0.1, high=0.1),
                                                   k2=rng.uniform(low=-0.1, high=0.1),
                                                   p1=rng.uniform(low=-0.01, high=0.01),
                                                   p2=rng.uniform(low=-0.001, high=0.001)))

    nerf_transforms = NerfTransforms(offset=[1.0, 2.0, 3.0], scale=0.2, aabb_scale=32, frames=nerf_frames)
    nerf_transforms.write(nerf_transform_json)

    # WHEN
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_training_data(nerf_transform_json)

    frames_by_filepath: Dict[str, NerfFrame] = {f.file_path: f for f in nerf_transforms.frames}
    ordered_paths: List[str] = testbed.nerf.training.dataset.paths
    assert_same_keys(frames_by_filepath, ordered_paths)
    ordered_frames = [frames_by_filepath[filepath] for filepath in ordered_paths]

    # THEN the loaded NeRF training dataset corresponds to the input NerfTransforms
    assert_len(nerf_transforms.frames, testbed.nerf.training.dataset.n_images)
    for trainview, (filepath, frame) in enumerate(zip(testbed.nerf.training.dataset.paths, ordered_frames)):
        assert_eq(filepath, frame.file_path)
        assert_eq(tuple(testbed.nerf.training.dataset.metadata[trainview].resolution), (frame.w, frame.h))
        assert_np_close(np.array(testbed.nerf.training.dataset.metadata[trainview].focal_length),
                        np.array((frame.fl_x, frame.fl_y)), eps=1e-4)
        assert_np_close(np.array(testbed.nerf.training.dataset.metadata[trainview].principal_point),
                        np.array((frame.cx/frame.w, frame.cy/frame.h)), eps=1e-5)

        assert_eq(testbed.nerf.training.dataset.metadata[trainview].lens.mode, DISTORTION_MODES[type(frame)])
        assert_np_close(np.array(testbed.nerf.training.dataset.metadata[trainview].lens.params),
                        _get_lens_params(frame), eps=1e-8)

    # THEN the intrinsics/extrinsics set by 'set_camera_to_training_view' correspond to the input NerfTransforms
    for trainview, frame in enumerate(ordered_frames):
        testbed.set_camera_to_training_view(trainview)

        # Extrinsics
        nerf_rot, nerf_t = PoseMatrix(np.array(frame.transform_matrix)[:3, :]).to_rt()
        nerf_t = nerf_t*nerf_transforms.scale + TranslationVector(nerf_transforms.offset)
        ngp_mat = Pose.from_rt(nerf_rot, nerf_t, CoordinateSystem.NERF).transform_matrix(CoordinateSystem.NGP)[:3, :]
        assert_np_close(np.array(testbed.camera_matrix), ngp_mat, eps=1e-6)

        # Lens
        assert testbed.nerf.render_with_camera_distortion
        assert_eq(testbed.nerf.render_lens.mode, DISTORTION_MODES[type(frame)])
        assert_np_close(np.array(testbed.nerf.render_lens.params), _get_lens_params(frame), eps=1e-8)

        # Intrinsics
        assert_np_close(np.array(testbed.screen_center),
                        np.array((1.0-frame.cx/frame.w, 1.0-frame.cy/frame.h)), eps=1e-5)
        resolution_ref = frame.w if testbed.fov_axis == 0 else frame.h
        gt_fov_x = np.rad2deg(focal_to_fov(frame.fl_x, resolution_ref))
        gt_fov_y = np.rad2deg(focal_to_fov(frame.fl_y, resolution_ref))
        assert_np_close(np.array(testbed.fov_xy), np.array((gt_fov_x, gt_fov_y)), eps=1e-4)
