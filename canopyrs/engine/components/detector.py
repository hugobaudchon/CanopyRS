"""Detector: Tiles -> Objects (boxes). Runs a v1 detector model over the tiles, one Object per box.

Reuses the v1 detector model unchanged via ``infer_v2``. Boxes come out in tile-pixel coords
(crs=None), exactly as v1 emits them — georeferencing happens later, in aggregation. Each output
object points to the tile it was found in (``tile_id``).
"""

from canopyrs.engine.models.registry import DETECTOR_REGISTRY
from canopyrs.engine.constants import Col, BOX
from canopyrs.engine.data import Tiles, Objects
from canopyrs.engine.contracts import Need, one_of
from canopyrs.engine.components.base import Component, flatten_by_tile, register_component


@register_component("detector")
class Detector(Component):
    def __init__(self, config):
        super().__init__(config)
        self._model_class = self._model(DETECTOR_REGISTRY)
        # Each tile is read from a pre-cut tile_path, or else from its source's window — one_of suffices.
        self.requires = (one_of(Need(Tiles, columns=(Col.TILE_PATH,)), Need(Tiles, links=("sources",))),)
        # Boxes in tile-pixel coords (crs=False), one per detection, each pointing at its tile.
        self.produces = Need(Objects, columns=(Col.DETECTOR_SCORE, Col.DETECTOR_CLASS), links=("tiles",), crs=False)

    def run(self, tiles: Tiles) -> Objects:
        detector = self._model_class(self.config)
        loader = self._loader(tiles, batch_size=self.config.batch_size)
        tile_ids, boxes, scores, classes = detector.infer_v2(loader)

        # Flatten per-tile predictions into one row per box, carrying each box's tile (the FK).
        flat_tile_ids, columns = flatten_by_tile(
            tile_ids,
            **{Col.GEOMETRY: boxes, Col.DETECTOR_SCORE: scores, Col.DETECTOR_CLASS: classes},
        )
        objects = Objects.build(
            geometry=columns.pop(Col.GEOMETRY), geom_kind=BOX, tile_id=flat_tile_ids, tiles=tiles, **columns,
        )
        print(f"Detector: {len(objects)} detections from {len(tiles)} tiles.")
        return objects
