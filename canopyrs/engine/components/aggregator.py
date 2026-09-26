"""Aggregator: cross-tile NMS over the latest Objects, reusing geodataset's ``Aggregator``.

Two inputs: the objects (detector boxes, segmenter masks, or a classifier's output) and the grid
tiles they were detected in — the NMS tile frames, requested explicitly as ``Tiles`` so per-object
crops never stand in for them. Each object is mapped to its tile (identity for
detections; a parent hop for objects sitting on crops), pixel geometry is mapped to CRS via its
tile's transform — applied per tile group in one vectorized call, never per object — and already-CRS
geometry passes through. The CRS polygons go to geodataset's ``Aggregator`` for NMS. Aggregation only
suppresses, so each survivor points back to the input object it kept (``prev_object_id``).

The NMS weights (detector / segmenter / classifier) decide which score columns matter — and a weighted
score may have been produced several components back (e.g. ``detector_score`` at a late, post-classifier
aggregator). We don't require the input to carry them forward: ``Objects.column`` walks the
``prev_objects`` ancestry to find each weighted score wherever it lives.
"""

import geopandas as gpd
import pandas as pd

from geodataset.aggregator import Aggregator as GdAggregator

from canopyrs.engine.constants import Col
from canopyrs.engine.data import Crops, Objects, Tiles
from canopyrs.engine.contracts import Need
from canopyrs.engine.components.base import Component, register_component
from canopyrs.engine.tilemeta import affine_params, box_of

# geodataset's Aggregator expects these column names in the frames it receives (external vocabulary).
GD_TILE_ID = "tile_id"


@register_component("aggregator")
class Aggregator(Component):
    def __init__(self, config):
        super().__init__(config)
        self._score_cols = self._weighted_score_columns()
        # Aggregator needs the input Objects to carry every weighted score column — raw detections on
        # tiles, or classified objects on crops of tiles (pixel or CRS coords both work) — and the
        # grid tiles the detections were found in: the NMS tile frames, asked for explicitly so
        # per-object crops can never stand in for them.
        self.requires = (Need(Objects, links=("imagery",), columns=tuple(self._score_cols),
                              on=(Tiles, Crops)),
                         Need(Tiles))
        # Survivors in CRS coords, pointing back to the input each kept (prev_objects). No imagery link:
        # they're image-agnostic, and a consumer that needs the image resolves it through the ancestry.
        self.produces = Need(Objects, columns=(Col.AGGREGATOR_SCORE,), links=("prev_objects",), crs=True)

    def run(self, objects: Objects, tiles: Tiles) -> Objects:
        assert self.out_dir is not None, "Aggregator needs an output dir (set Pipeline(output_dir=...))"
        crs = tiles.df[Col.METADATA].iloc[0]["crs"] if len(tiles) else None
        if len(objects) == 0:
            print("Aggregator: no input objects; nothing to aggregate.")
            return self._empty(objects, crs)

        # Each object's tile: its image expressed at the tiles level — identity for detections (their
        # images are the tiles), one parent hop for objects sitting on per-object crops.
        tile_id = objects.linked("imagery").ancestor_ids(objects.column(Col.IMAGE_ID), tiles)
        polygons_gdf, tiles_extent_gdf = self._georeference(objects, tiles, tile_id, crs)
        scores_names, scores_weights = self._scores()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        agg = GdAggregator(
            output_path=self.out_dir / "aggregator.gpkg",
            polygons_gdf=polygons_gdf,
            scores_names=scores_names,
            other_attributes_names=[Col.PREV_OBJECT_ID],
            scores_weights=scores_weights,
            tiles_extent_gdf=tiles_extent_gdf,
            tile_ids_to_path=self._tile_paths(tiles, tiles_extent_gdf[GD_TILE_ID]),
            scores_weighting_method=self.config.scores_weighting_method,
            min_centroid_distance_weight=self.config.min_centroid_distance_weight,
            score_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
            nms_algorithm=self.config.nms_algorithm,
            edge_band_buffer_percentage=self.config.edge_band_buffer_percentage,
            best_geom_keep_area_ratio=self.config.best_geom_keep_area_ratio,
        )
        survivors = agg.polygons_gdf

        out = Objects.build(
            geometry=survivors.geometry.values,
            geom_kind=objects.df[Col.GEOM_KIND].iloc[0],
            prev_object_id=survivors[Col.PREV_OBJECT_ID].values,
            crs=survivors.crs,
            prev_objects=objects,
            **{Col.AGGREGATOR_SCORE: survivors[Col.AGGREGATOR_SCORE].values,
               **{col: survivors[col].values for col in self._score_cols}},
        )
        print(f"Aggregator: kept {len(out)} of {len(objects)} objects.")
        return out

    def _georeference(self, objects: Objects, tiles: Tiles, tile_id, crs):
        """The CRS ``polygons_gdf`` (with its scores + the ``prev_object_id`` carry) and the per-tile
        ``tiles_extent_gdf`` geodataset needs. Pixel geometry is mapped to CRS via its tile's affine
        transform (origin + GSD), grouped by tile and transformed with one vectorized
        ``affine_transform`` per group — the affine is constant within a tile, so the loop runs over
        distinct tiles, never over objects. CRS geometry passes through unchanged."""
        data = {
            Col.GEOMETRY: objects.df.geometry.values,
            GD_TILE_ID: tile_id.values,
            Col.PREV_OBJECT_ID: objects.df[Col.OBJECT_ID].values,   # the input object each survivor came from
        }
        for col in self._score_cols:
            data[col] = objects.column(col).values                 # walk the ancestry for each weighted score
        polygons_gdf = gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=crs)

        if not objects.crs_set:
            meta_by_id = tiles.df.set_index(Col.IMAGE_ID)[Col.METADATA]
            transformed = [group.geometry.affine_transform(affine_params(meta_by_id[iid]))
                           for iid, group in polygons_gdf.groupby(GD_TILE_ID, sort=False)]
            polygons_gdf[Col.GEOMETRY] = pd.concat(transformed).reindex(polygons_gdf.index)

        tiles_extent_gdf = gpd.GeoDataFrame({
            GD_TILE_ID: tiles.df[Col.IMAGE_ID].values,
            Col.GEOMETRY: tiles.df[Col.METADATA].map(box_of).values,
        }, geometry=Col.GEOMETRY, crs=crs)
        return polygons_gdf, tiles_extent_gdf

    def _tile_paths(self, imagery: Tiles, image_ids):
        """image_id -> file path for geodataset. Real ``path`` when tiles were cut to disk; else a
        clearly-fake ``unsaved_tile_{id}`` (windows read on demand have no file)."""
        paths = imagery.df.set_index(Col.IMAGE_ID)[Col.PATH] if Col.PATH in imagery.df.columns else None

        def path_for(image_id):
            path = paths.get(image_id) if paths is not None else None
            return str(path) if (path is not None and path == path and path != "") else f"unsaved_tile_{image_id}"

        return {image_id: path_for(image_id) for image_id in image_ids}

    def _empty(self, objects, crs) -> Objects:
        """An empty output (no input objects), still satisfying ``produces``: the score columns, the
        ``prev_objects`` link (to the empty input), and the CRS."""
        columns = {Col.AGGREGATOR_SCORE: [], **{col: [] for col in self._score_cols}}
        return Objects.build(geometry=[], geom_kind=[], prev_object_id=[], crs=crs,
                             prev_objects=objects, **columns)

    def _weights(self):
        return {
            Col.DETECTOR_SCORE: self.config.detector_score_weight,
            Col.SEGMENTER_SCORE: self.config.segmenter_score_weight,
            Col.CLASSIFIER_SCORE: self.config.classifier_score_weight,
        }

    def _weighted_score_columns(self):
        return {col for col, weight in self._weights().items() if weight > 0}

    def _scores(self):
        """Weighted score columns, normalized to sum to 1 (geodataset requires that)."""
        active = {col: weight for col, weight in self._weights().items() if weight > 0} or {Col.DETECTOR_SCORE: 1.0}
        total = sum(active.values())
        return list(active), [weight / total for weight in active.values()]
