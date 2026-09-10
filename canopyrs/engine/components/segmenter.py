"""Segmenter: masks via a segmenter model. Two modes (by the model's ``REQUIRES_BOX_PROMPT``):

  - **prompted** (e.g. sam2): segments input prompt boxes. Each prompt is an Object that points at its
    image (``object.imagery``); SAM needs tile-pixel boxes, so a CRS prompt (e.g. an aggregated
    detection) is inverse-transformed into its image's pixels, while a tile-pixel prompt (a raw detector
    box) is used as-is. The model is prompted once per image. One mask per prompt, each pointing at its
    image (``image_id``) and the prompt it came from (``prev_object_id``).
  - **automatic** (e.g. detectree2): generates masks over each tile wholesale. One mask per region,
    pointing at its image; masks are new objects (no prompt).

Masks come out in tile-pixel coords (crs=False); a downstream aggregator georeferences + NMSes them.
The model post-processes masks with a multiprocessing pool, so the run must use the spawn start method
under a ``__main__`` guard — otherwise the workers deadlock on the live CUDA context.
"""

import geopandas as gpd

from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.constants import Col, GeomKind
from canopyrs.engine.data import Objects, Tiles
from canopyrs.engine.contracts import Need
from canopyrs.engine.components.base import Component, flatten_by_tile, register_component
from canopyrs.engine.tilemeta import crs_to_pixel


@register_component("segmenter")
class Segmenter(Component):
    def __init__(self, config):
        super().__init__(config)
        self._model_class = self._model(SEGMENTER_REGISTRY)
        self.prompted = self._model_class.REQUIRES_BOX_PROMPT
        if self.prompted:
            # prompt boxes living on tiles (object.imagery) — pixels are read from there. No CRS
            # constraint: a CRS prompt is inverse-transformed, a tile-pixel one used as-is.
            self.requires = (Need(Objects, links=("imagery",), on=Tiles),)
            self.produces = Need(Objects, columns=(Col.SEGMENTER_SCORE,),
                                 links=("imagery", "prev_objects"), crs=False, on=Tiles)
        else:
            self.requires = (Need(Tiles),)
            self.produces = Need(Objects, columns=(Col.SEGMENTER_SCORE,), links=("imagery",),
                                 crs=False, on=Tiles)

    def run(self, data) -> Objects:
        """``data`` is the prompts (Objects) in prompted mode, or the tiles (Tiles) in automatic mode
        — matching ``requires``."""
        segmenter = self._model_class(self.config)
        out = self._prompted(data, segmenter) if self.prompted else self._automatic(data, segmenter)
        print(f"Segmenter[{'prompted' if self.prompted else 'auto'}]: {len(out)} masks.")
        return out

    def _automatic(self, tiles: Tiles, segmenter) -> Objects:
        loader = self._loader(tiles, batch_size=self.config.image_batch_size)
        image_ids, _, polygons_per_tile, scores_per_tile = segmenter.infer(loader)

        flat_image_ids, columns = flatten_by_tile(
            image_ids, **{Col.GEOMETRY: polygons_per_tile, Col.SEGMENTER_SCORE: scores_per_tile},
        )
        return Objects.build(geometry=columns.pop(Col.GEOMETRY), geom_kind=GeomKind.MASK,
                             image_id=flat_image_ids, imagery=tiles, **columns)

    def _prompted(self, prompts: Objects, segmenter) -> Objects:
        # The prompts live in an image, resolved through the ancestry (an aggregated prompt finds the
        # grid tile of the detection it kept). SAM is prompted with tile-pixel boxes: a CRS prompt is
        # inverse-transformed into its image's pixels, a tile-pixel one used as-is. Then grouped by image.
        tiles = prompts.linked("imagery")
        image_id = prompts.column(Col.IMAGE_ID)
        if prompts.df.crs is not None:
            meta = image_id.map(tiles.df.set_index(Col.IMAGE_ID)[Col.METADATA])
            boxes = gpd.GeoSeries([crs_to_pixel(geom, m) for geom, m in zip(prompts.df.geometry.values, meta.values)])
        else:
            boxes = prompts.df.geometry
        bounds = boxes.bounds.assign(tile=image_id.values, prompt=prompts.df[Col.OBJECT_ID].values)
        boxes_by_tile = {
            tid: (group[["minx", "miny", "maxx", "maxy"]].to_numpy(float), group["prompt"].tolist())
            for tid, group in bounds.groupby("tile", sort=False)
        }

        frame = tiles.reading_frame()
        frame = frame[frame[Col.IMAGE_ID].isin(boxes_by_tile)].reset_index(drop=True)   # only images with prompts
        loader = self._loader(frame, batch_size=self.config.image_batch_size)
        image_ids, prompt_ids_per_tile, polygons_per_tile, scores_per_tile = segmenter.infer(loader, boxes_by_tile)

        flat_image_ids, columns = flatten_by_tile(
            image_ids,
            **{Col.GEOMETRY: polygons_per_tile, Col.PREV_OBJECT_ID: prompt_ids_per_tile,
               Col.SEGMENTER_SCORE: scores_per_tile},
        )
        return Objects.build(geometry=columns.pop(Col.GEOMETRY), geom_kind=GeomKind.MASK,
                             image_id=flat_image_ids,
                             prev_object_id=columns.pop(Col.PREV_OBJECT_ID),
                             imagery=tiles, prev_objects=prompts, **columns)
