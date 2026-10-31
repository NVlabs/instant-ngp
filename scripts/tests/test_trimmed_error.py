import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import compute_error, compute_error_img


def test_trimmed_relative_squared_error():
    image = np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 24
    reference = np.full_like(image, 0.5)
    expected = np.mean((image - reference) ** 2 / (0.01 + reference ** 2))
    assert compute_error("MtRSE", image, reference) == pytest.approx(expected)


def test_trimmed_error_removes_both_tails():
    image = np.ones((1, 1_000_000, 1), dtype=np.float32)
    reference = np.zeros_like(image)
    image[0, 0, 0] = 0
    image[0, -1, 0] = 10_000
    assert compute_error("MtRSE", image, reference) == pytest.approx(100)


@pytest.mark.parametrize("metric", ["MAE", "MSE", "MRSE", "MScE", "SSIM"])
def test_image_error_reduction_is_unchanged(metric):
    image = np.full((3, 4, 3), 0.25, dtype=np.float32)
    reference = np.full_like(image, 0.5)
    expected = np.mean(compute_error_img(metric, image.copy(), reference))
    assert compute_error(metric, image, reference) == pytest.approx(expected)
