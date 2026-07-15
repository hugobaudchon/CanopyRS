"""Segmenter: masks via a reused v1 segmenter model. Two modes (by the model's ``REQUIRES_BOX_PROMPT``):

  - **prompted** (e.g. sam2): segments input prompt boxes. Each prompt is an Object that points at its
    tile (``object.tiles``); SAM needs tile-pixel boxes, so a CRS prompt (e.g. an aggregated detection)
    is inverse-transformed into its tile's pixels, while a tile-pixel prompt (a raw detector box, as in
    v1) is used as-is. The model is prompted once per tile. One mask per prompt, each pointing at its
    tile (``tile_id``) and the prompt it came from (``prev_object_id``).
  - **automatic** (e.g. detectree2): generates masks over each tile wholesale. One mask per region,
    pointing at its tile; masks are new objects (no prompt).

Masks come out in tile-pixel coords (crs=False); a downstream aggregator georeferences + NMSes them.
The model post-processes masks with a multiprocessing pool, so the run must use the spawn start method
under a ``__main__`` guard (as v1 does) — otherwise the workers deadlock on the live CUDA context.
"""

import geopandas as gpd

from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.constants import Col, MASK
from canopyrs.engine.data import Tiles, Objects
from canopyrs.engine.contracts import Need, one_of
from canopyrs.engine.components.base import Component, flatten_by_tile, register_component
from canopyrs.engine.tilemeta import crs_to_pixel


@register_component("segmenter")
class Segmenter(Component):
    def __init__(self, config):
        super().__init__(config)
        self._model_class = self._model(SEGMENTER_REGISTRY)
        self.prompted = self._model_class.REQUIRES_BOX_PROMPT
        if self.prompted:
            # prompt boxes that point at their tile (object.tiles) — images are read from there. No CRS
            # constraint: a CRS prompt is inverse-transformed, a tile-pixel one used as-is.
            self.requires = (Need(Objects, links=("tiles",)),)
            self.produces = Need(Objects, columns=(Col.SEGMENTER_SCORE,), links=("tiles", "prev_objects"), crs=False)
        else:
            self.requires = (one_of(Need(Tiles, columns=(Col.TILE_PATH,)), Need(Tiles, links=("sources",))),)
            self.produces = Need(Objects, columns=(Col.SEGMENTER_SCORE,), links=("tiles",), crs=False)

    def run(self, data) -> Objects:
        """``data`` is the prompts (Objects) in prompted mode, or the tiles (Tiles) in automatic mode —
        matching ``requires``."""
        segmenter = self._model_class(self.config)
        out = self._prompted(data, segmenter) if self.prompted else self._automatic(data, segmenter)
        print(f"Segmenter[{'prompted' if self.prompted else 'auto'}]: {len(out)} masks.")
        return out

    def _automatic(self, tiles: Tiles, segmenter) -> Objects:
        loader = self._loader(tiles, batch_size=self.config.image_batch_size)
        tile_ids, _, polygons_per_tile, scores_per_tile = segmenter.infer_v2(loader)

        flat_tile_ids, columns = flatten_by_tile(
            tile_ids, **{Col.GEOMETRY: polygons_per_tile, Col.SEGMENTER_SCORE: scores_per_tile},
        )
        return Objects.build(geometry=columns.pop(Col.GEOMETRY), geom_kind=MASK,
                             tile_id=flat_tile_ids, tiles=tiles, **columns)

    def _prompted(self, prompts: Objects, segmenter) -> Objects:
        # The prompts live in a tile, resolved through the ancestry (an aggregated prompt finds the grid
        # tile of the detection it kept). SAM is prompted with tile-pixel boxes: a CRS prompt is inverse-
        # transformed into its tile's pixels, a tile-pixel one used as-is. Then grouped by tile.
        tiles = prompts.linked("tiles")
        tile_id = prompts.column(Col.TILE_ID)
        if prompts.df.crs is not None:
            meta = tile_id.map(tiles.df.set_index(Col.TILE_ID)[Col.TILE_METADATA])
            boxes = gpd.GeoSeries([crs_to_pixel(geom, m) for geom, m in zip(prompts.df.geometry.values, meta.values)])
        else:
            boxes = prompts.df.geometry
        bounds = boxes.bounds.assign(tile=tile_id.values, prompt=prompts.df[Col.OBJECT_ID].values)
        boxes_by_tile = {
            tid: (group[["minx", "miny", "maxx", "maxy"]].to_numpy(float), group["prompt"].tolist())
            for tid, group in bounds.groupby("tile", sort=False)
        }

        frame = tiles.reading_frame()
        frame = frame[frame[Col.TILE_ID].isin(boxes_by_tile)].reset_index(drop=True)   # only tiles with prompts
        loader = self._loader(frame, batch_size=self.config.image_batch_size)
        tile_ids, prompt_ids_per_tile, polygons_per_tile, scores_per_tile = segmenter.infer_v2(loader, boxes_by_tile)

        flat_tile_ids, columns = flatten_by_tile(
            tile_ids,
            **{Col.GEOMETRY: polygons_per_tile, Col.PREV_OBJECT_ID: prompt_ids_per_tile,
               Col.SEGMENTER_SCORE: scores_per_tile},
        )
        return Objects.build(geometry=columns.pop(Col.GEOMETRY), geom_kind=MASK, tile_id=flat_tile_ids,
                             prev_object_id=columns.pop(Col.PREV_OBJECT_ID),
                             tiles=tiles, prev_objects=prompts, **columns)
