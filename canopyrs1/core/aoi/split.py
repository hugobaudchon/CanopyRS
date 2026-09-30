"""Splitting tiles assigned to AOIs, and the objects on them, into one set per AOI."""

import numpy as np
import pandas as pd

from canopyrs1.core.constants import Col
from canopyrs1.core.tables.imagery import Imagery
from canopyrs1.core.tables.objects import Objects


def split_by_aoi(
    tiles: Imagery,
    objects: Objects | None = None,
) -> dict[str, tuple[Imagery, Objects | None]]:
    """Return the tiles and objects of each AOI, as {name: (tiles, objects)}, in the order the
    AOIs first appear. ``tiles`` are copies assigned to AOIs (see ``assign_to_aois``), and
    ``objects``, if given, are on them: each goes with its tile. Each AOI's tables are new ones
    whose rows point to the rows they came from (see ``select``). Raises a ValueError if
    ``objects`` aren't on ``tiles``."""
    if objects is not None and objects.parent_imagery is not tiles:
        raise ValueError("The objects to split must be on the tiles to split")

    names = tiles.df[Col.AOI].to_numpy()
    folds = {}
    for name in pd.unique(names):
        # the AOI's tiles
        tile_ids = np.flatnonzero(names == name)
        fold_tiles = tiles.select(tile_ids)
        if objects is None:
            folds[name] = (fold_tiles, None)
            continue

        # the AOI's objects, moved onto the AOI's tiles
        image_ids = objects.df[Col.PARENT_IMAGE_ID].to_numpy()
        object_ids = np.flatnonzero(np.isin(image_ids, tile_ids))
        fold_objects = objects.select(
            object_ids,
            parent_image_id=np.searchsorted(tile_ids, image_ids[object_ids]),
            parent_imagery=fold_tiles,
        )
        folds[name] = (fold_tiles, fold_objects)
    return folds
