#!/usr/bin/python3
"""Rendering Script"""
import os
from typing import Dict
from typing import Tuple

import imageio
import numpy as np
import pyngp as ngp  # noqa
from tqdm import tqdm
from utils_3dml.file.json_utils import JSONType
from utils_3dml.file.json_utils import read_json
from utils_3dml.monitoring.profiler import profile

from instant_ngp_3dml import logger
from instant_ngp_3dml.utils.tonemapper import linear_to_srgb
from instant_ngp_3dml.utils.tonemapper import tonemap

RENDER_MODES: Dict[str, ngp.RenderMode] = {"depth": ngp.RenderMode.Depth,
                                           "color": ngp.RenderMode.Shade,
                                           "confidence": ngp.RenderMode.Confidence}


@profile
def __save_color(outname, image):
    image = np.copy(image)
    # Unmultiply alpha
    image[..., 0:3] = np.divide(
        image[..., 0:3],
        image[..., 3:4],
        out=np.zeros_like(image[..., 0:3]),
        where=image[..., 3:4] != 0)
    image[..., 0:3] = linear_to_srgb(image[..., 0:3])
    image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # Some NeRF datasets lack the .png suffix in the dataset metadata
    if os.path.splitext(outname)[1] != ".png":
        outname = os.path.splitext(outname)[0] + ".png"

    os.makedirs(os.path.dirname(outname), exist_ok=True)
    imageio.imwrite(outname, image)


def get_testbed_and_spp(snapshot_msgpack: str,
                        ref_transforms: JSONType,
                        equirectangular: bool,
                        render_mode: str,
                        spp: int) -> Tuple[ngp.Testbed, int]:
    """Init TestBed and Spp for Rendering"""
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    logger.info(f"Loading snapshot {snapshot_msgpack}")
    assert snapshot_msgpack.endswith(".msgpack")
    testbed.load_snapshot(snapshot_msgpack)

    testbed.shall_train = False
    testbed.nerf.render_with_camera_distortion = True
    testbed.display_gui = False

    testbed.fov_axis = 0
    testbed.fov = np.rad2deg(ref_transforms["camera_angle_x"])
    testbed.dynamic_res = False
    testbed.fixed_res_factor = 1

    if equirectangular:
        testbed.nerf.render_with_lens_distortion = True
        testbed.nerf.render_distortion.mode = ngp.LensMode.Equirectangular

    assert render_mode.lower() in RENDER_MODES, \
        f"Invalid render mode '{render_mode}'. Should be in {RENDER_MODES.keys()}"
    testbed.render_mode = RENDER_MODES[render_mode.lower()]

    # If "color", keep spp
    if render_mode == "depth":
        logger.info("Set depth rendering params")
        testbed.tonemap_curve = ngp.TonemapCurve.Identity
        testbed.color_space = ngp.ColorSpace.Linear
        testbed.render_mode = ngp.RenderMode.Depth
        spp = 1
    elif render_mode == "confidence":
        logger.info("Set confidence rendering params")
        testbed.tonemap_curve = ngp.TonemapCurve.Identity
        testbed.color_space = ngp.ColorSpace.Linear
        testbed.render_mode = ngp.RenderMode.Confidence
        spp = 1
    else:
        raise ValueError(f"Unhandled rendering mode: {render_mode}")

    return testbed, spp


@profile
def main(snapshot_msgpack: str,
         nerf_transform_json: str,
         out_rendering_folder: str,
         spp: int = 4,
         downscale_factor: float = 1.0,
         render_mode: str = "color",
         equirectangular: bool = False,
         color_depth: bool = True):
    """Render NeRF Scene.

        Args:
            snapshot_msgpack: Input NeRF Weight
            nerf_transform_json: Input NeRF Transform Json
            out_rendering_folder: Output Folder with rendered images
            spp: Sample per pixel
            downscale_factor: Downscale rendered frames
            render_mode: Renderer method
            lens_mode: Camera model
            color_depth: Tonemap the generated Depthmaps, if render_mode=="depth"

        Raises:
            ValueError: if CameraMode or RenderMode doesn't exist
    """
    # pylint: disable=too-many-arguments,too-many-statements,too-many-branches
    logger.debug(f"Load rendering transforms from {nerf_transform_json}")
    ref_transforms = read_json(nerf_transform_json)
    # TODO: DT-609: R&D: Move NeRF transform json schema from core to utils to reuse it in instant ngp
    # https://checkandvisit.atlassian.net/browse/DT-609

    # Pick a sensible GUI resolution depending on arguments.
    sw = int(ref_transforms["w"] / downscale_factor)
    sh = int(ref_transforms["h"] / downscale_factor)
    while sw*sh > 1920*1080*4:
        sw = int(sw / 2)
        sh = int(sh / 2)

    testbed, spp = get_testbed_and_spp(snapshot_msgpack, ref_transforms, equirectangular,
                                       render_mode, spp)

    nb_frames = len(ref_transforms["frames"])

    for idx in tqdm(range(nb_frames), desc="Rendering", unit="frame"):
        f = ref_transforms["frames"][int(idx)]
        cam_matrix = f["transform_matrix"]
        testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
        outname = os.path.join(out_rendering_folder,
                               os.path.basename(f["file_path"]))

        image = testbed.render(sw, sh, spp, True)
        image[..., :3] *= 2**(-1*testbed.exposure)

        if render_mode == "color":
            __save_color(outname, image)
        elif render_mode == "depth":
            # Force depth in numpy format
            outname = os.path.splitext(outname)[0] + ".npy"
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            raw_depth = image[..., 0]
            np.save(outname, raw_depth)

            if color_depth:
                outname = os.path.splitext(outname)[0] + ".png"
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                imageio.imwrite(outname, tonemap(raw_depth))

        elif render_mode == "confidence":
            __save_color(outname, image)
        else:
            raise ValueError(
                f"Invalid render mode '{render_mode}'. Should be in {RENDER_MODES.keys()}")
