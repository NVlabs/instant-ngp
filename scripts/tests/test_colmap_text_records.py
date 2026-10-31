import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "colmap2nerf.py"


@pytest.mark.parametrize("empty_observations", [False, True])
@pytest.mark.parametrize("filename", ["frame.png", "frame with  two spaces.png"])
def test_colmap_text_records_preserve_images(tmp_path, empty_observations, filename):
    images = tmp_path / "images"
    text = tmp_path / "colmap_text"
    images.mkdir()
    text.mkdir()
    Image.new("RGB", (8, 6), (32, 96, 160)).save(images / filename)
    Image.new("RGB", (8, 6), (64, 128, 192)).save(images / "second.png")
    (text / "cameras.txt").write_text("# cameras\n1 PINHOLE 8 6 5 5 4 3\n")
    observation_line = "" if empty_observations else "1 2 -1"
    (text / "images.txt").write_text(
        "# images\n"
        f"1 1 0 0 0 0 0 0 1 {filename}\n{observation_line}\n"
        "2 1 0 0 0 1 2 3 1 second.png\n1 2 -1\n"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--keep_colmap_coords"],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads((tmp_path / "transforms.json").read_text())
    assert len(output["frames"]) == 2
    assert Path(output["frames"][0]["file_path"]).name == filename
    assert Path(output["frames"][1]["file_path"]).name == "second.png"
    for frame in output["frames"]:
        assert (tmp_path / frame["file_path"]).is_file()
        assert np.isfinite(frame["transform_matrix"]).all()
    np.testing.assert_allclose(np.array(output["frames"][1]["transform_matrix"])[:3, 3], [-1, -2, -3])
