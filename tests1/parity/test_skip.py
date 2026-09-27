"""should_skip decides like geodataset's tile check, for RGB and RGBA tiles."""

from types import SimpleNamespace

import numpy as np
import pytest

from geodataset.tilerize import RasterTilerizer

from canopyrs1.core.raster.read import should_skip


def _random_tiles(bands, n=300, seed=0):
    """Yield random uint8 tiles in which a random share of the pixels are black, white or (with
    4 bands) transparent."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        tile = rng.integers(1, 255, (bands, 16, 16), dtype=np.uint8)
        for value, share in ((0, rng.uniform(0, 0.6)), (255, rng.uniform(0, 0.4))):
            pixels = rng.random((16, 16)) < share
            tile[: min(bands, 3), pixels] = value
        if bands == 4:
            tile[3] = np.where(rng.random((16, 16)) < rng.uniform(0, 0.5), 0, 255)
        yield tile


THRESHOLDS = [0.1, 0.5, 0.75, 0.9]


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("bands", [3, 4])
def test_same_decisions_as_geodataset(bands, threshold):
    old_tilerizer = SimpleNamespace(ignore_black_white_alpha_tiles_threshold=threshold)
    for tile in _random_tiles(bands):
        old = RasterTilerizer._check_skip_tile_data(old_tilerizer, tile)
        colours, alpha = (tile[:3], tile[3]) if bands == 4 else (tile, None)
        conditions = {"ignore_black_white_alpha_tiles_threshold": threshold}
        new = should_skip(colours, conditions, alpha=alpha)
        assert new == old
