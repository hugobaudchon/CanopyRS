"""Detector: Tiles -> Objects (boxes). Runs a detector model over the tiles, one Object per box.

Boxes come out in tile-pixel coords (crs=None) — georeferencing happens later, in aggregation. Each
output object points to the image it was found in (``image_id``). Requiring ``Tiles`` statically
refuses an untiled source scene; readability (own file, or a window into a materialized ancestor) is
guaranteed by the imagery tree, so no further alternatives are needed.
"""

from canopyrs.engine.models.registry import DETECTOR_REGISTRY
from canopyrs.engine.constants import Col, GeomKind
from canopyrs.engine.data import Objects, Tiles
from canopyrs.engine.contracts import Need
from canopyrs.engine.components.base import Component, flatten_by_tile, register_component


@register_component("detector")
class Detector(Component):
    def __init__(self, config):
        super().__init__(config)
        self._model_class = self._model(DETECTOR_REGISTRY)
        self.requires = (Need(Tiles),)
        # Boxes in tile-pixel coords (crs=False), one per detection, each pointing at its tile.
        self.produces = Need(Objects, columns=(Col.DETECTOR_SCORE, Col.DETECTOR_CLASS),
                             links=("imagery",), crs=False, on=Tiles)

    def run(self, tiles: Tiles) -> Objects:
        detector = self._model_class(self.config)
        loader = self._loader(tiles, batch_size=self.config.batch_size)
        image_ids, boxes, scores, classes = detector.infer_v2(loader)

        # Flatten per-tile predictions into one row per box, carrying each box's image (the FK).
        flat_image_ids, columns = flatten_by_tile(
            image_ids,
            **{Col.GEOMETRY: boxes, Col.DETECTOR_SCORE: scores, Col.DETECTOR_CLASS: classes},
        )
        objects = Objects.build(
            geometry=columns.pop(Col.GEOMETRY), geom_kind=GeomKind.BOX,
            image_id=flat_image_ids, imagery=tiles, **columns,
        )
        print(f"Detector: {len(objects)} detections from {len(tiles)} tiles.")
        return objects
