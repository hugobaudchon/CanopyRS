"""Aggregator: cross-tile NMS over the latest Objects, reusing geodataset's ``Aggregator``.

Inputs arrive in tile-pixel coords (crs=None) — detector boxes or segmenter masks. Each object's tile
is looked up in the latest Tiles (by ``tile_id``), and its geometry mapped from that tile's pixels to
CRS via the tile's own transform (boxes vectorized via bounds; masks vertex-wise). The CRS polygons go
to geodataset's ``Aggregator`` for NMS. Aggregation only suppresses, so each survivor points back to
the input object it kept (``prev_object_id``).

The NMS weights (detector / segmenter / classifier) decide which score columns matter — and a weighted
score may have been produced several components back (e.g. ``detector_score`` at a late, post-classifier
aggregator). We don't require the input to carry them forward: ``Objects.column`` walks the
``prev_objects`` ancestry to find each weighted score wherever it lives.
"""

import geopandas as gpd

from geodataset.aggregator import Aggregator as GdAggregator

from canopyrs.engine.constants import Col
from canopyrs.engine.data import Tiles, Objects
from canopyrs.engine.contracts import Need
from canopyrs.engine.components.base import Component, register_component
from canopyrs.engine.tilemeta import box_of, pixel_to_crs


@register_component("aggregator")
class Aggregator(Component):
    def __init__(self, config):
        super().__init__(config)
        self._score_cols = self._weighted_score_columns()
        # Aggregator needs the input Objects to carry every weighted score column, and to have their
        # tiles linked (for georeferencing).
        self.requires = (Need(Objects, links=("tiles",), columns=tuple(self._score_cols), crs=False),)
        # Survivors in CRS coords, pointing back to the input each kept (prev_objects). No tiles link:
        # they're tile-agnostic, and a consumer that needs the tile resolves it through the ancestry.
        self.produces = Need(Objects, columns=(Col.AGGREGATOR_SCORE,), links=("prev_objects",), crs=True)

    def run(self, objects: Objects) -> Objects:
        assert self.out_dir is not None, "Aggregator needs an output dir (set Pipeline(output_dir=...))"
        tiles = objects.linked("tiles")   # the input's tiles — directly, or resolved through its ancestry
        crs = tiles.df[Col.TILE_METADATA].iloc[0]["crs"] if len(tiles) else None
        if len(objects) == 0:
            print("Aggregator: no input objects; nothing to aggregate.")
            return self._empty(objects, crs)

        polygons_gdf, tiles_extent_gdf = self._georeference(objects, tiles, crs)
        scores_names, scores_weights = self._scores()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        agg = GdAggregator(
            output_path=self.out_dir / "aggregator.gpkg",
            polygons_gdf=polygons_gdf,
            scores_names=scores_names,
            other_attributes_names=[Col.PREV_OBJECT_ID],
            scores_weights=scores_weights,
            tiles_extent_gdf=tiles_extent_gdf,
            tile_ids_to_path=self._tile_paths(tiles, tiles_extent_gdf[Col.TILE_ID]),
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

    def _georeference(self, objects: Objects, tiles: Tiles, crs):
        """Map each object from its tile's pixel coords to CRS via the tile's own affine transform
        (origin + GSD). Each object's ``tile_id`` is resolved through the ancestry (so an aggregated
        input finds the tile of the detection it kept), then joined to ``tiles``. Boxes and masks alike
        are transformed vertex-wise (``pixel_to_crs``). Returns the CRS ``polygons_gdf`` (with its
        scores + the ``prev_object_id`` carry) and the per-tile ``tiles_extent_gdf`` geodataset needs."""
        tile_id = objects.column(Col.TILE_ID)
        metadata = tile_id.map(tiles.df.set_index(Col.TILE_ID)[Col.TILE_METADATA]).values
        geometry = [pixel_to_crs(geom, meta) for geom, meta in zip(objects.df.geometry.values, metadata)]

        data = {
            Col.GEOMETRY: geometry,
            Col.TILE_ID: tile_id.values,
            Col.PREV_OBJECT_ID: objects.df[Col.OBJECT_ID].values,   # the input object each survivor came from
        }
        for col in self._score_cols:
            data[col] = objects.column(col).values                 # walk the ancestry for each weighted score
        polygons_gdf = gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=crs)
        tiles_extent_gdf = gpd.GeoDataFrame({
            Col.TILE_ID: tiles.df[Col.TILE_ID].values,
            Col.GEOMETRY: tiles.df[Col.TILE_METADATA].map(box_of).values,
        }, geometry=Col.GEOMETRY, crs=crs)
        return polygons_gdf, tiles_extent_gdf

    def _tile_paths(self, tiles: Tiles, tile_ids):
        """tile_id -> source path for geodataset. Real ``tile_path`` when tiles were cut to disk; else a
        clearly-fake ``unsaved_tile_{id}`` (windows read on demand have no file)."""
        paths = tiles.df.set_index(Col.TILE_ID)[Col.TILE_PATH] if Col.TILE_PATH in tiles.df.columns else None

        def path_for(tile_id):
            path = paths.get(tile_id) if paths is not None else None
            return str(path) if (path is not None and path == path and path != "") else f"unsaved_tile_{tile_id}"

        return {tile_id: path_for(tile_id) for tile_id in tile_ids}

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
