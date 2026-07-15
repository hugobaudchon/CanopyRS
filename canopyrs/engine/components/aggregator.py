"""Aggregator: cross-tile NMS over the latest Objects, reusing geodataset's ``Aggregator``.

Inputs arrive in tile-pixel coords (crs=None) — detector boxes or segmenter masks. Each object's image
is looked up in its linked Imagery (by ``image_id``), and its geometry mapped from that image's pixels
to CRS via the image's own transform — applied per image group in one vectorized call, never per
object. The CRS polygons go to geodataset's ``Aggregator`` for NMS. Aggregation only suppresses, so
each survivor points back to the input object it kept (``prev_object_id``).

The NMS weights (detector / segmenter / classifier) decide which score columns matter — and a weighted
score may have been produced several components back (e.g. ``detector_score`` at a late, post-classifier
aggregator). We don't require the input to carry them forward: ``Objects.column`` walks the
``prev_objects`` ancestry to find each weighted score wherever it lives.
"""

import geopandas as gpd
import pandas as pd

from geodataset.aggregator import Aggregator as GdAggregator

from canopyrs.engine.constants import Col
from canopyrs.engine.data import Imagery, Objects
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
        # Aggregator needs the input Objects to carry every weighted score column, and to have their
        # imagery linked (for georeferencing).
        self.requires = (Need(Objects, links=("imagery",), columns=tuple(self._score_cols), crs=False),)
        # Survivors in CRS coords, pointing back to the input each kept (prev_objects). No imagery link:
        # they're image-agnostic, and a consumer that needs the image resolves it through the ancestry.
        self.produces = Need(Objects, columns=(Col.AGGREGATOR_SCORE,), links=("prev_objects",), crs=True)

    def run(self, objects: Objects) -> Objects:
        assert self.out_dir is not None, "Aggregator needs an output dir (set Pipeline(output_dir=...))"
        tiles = objects.linked("imagery")   # the input's imagery — directly, or resolved through its ancestry
        crs = tiles.df[Col.METADATA].iloc[0]["crs"] if len(tiles) else None
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

    def _georeference(self, objects: Objects, imagery: Imagery, crs):
        """Map each object from its image's pixel coords to CRS via the image's own affine transform
        (origin + GSD). Each object's ``image_id`` is resolved through the ancestry (so an aggregated
        input finds the image of the detection it kept), then grouped by image and transformed with one
        vectorized ``affine_transform`` per group — the affine is constant within an image, so the loop
        runs over distinct images, never over objects. Returns the CRS ``polygons_gdf`` (with its
        scores + the ``prev_object_id`` carry) and the per-image ``tiles_extent_gdf`` geodataset needs."""
        image_id = objects.column(Col.IMAGE_ID)
        meta_by_id = imagery.df.set_index(Col.IMAGE_ID)[Col.METADATA]

        data = {
            Col.GEOMETRY: objects.df.geometry.values,
            GD_TILE_ID: image_id.values,
            Col.PREV_OBJECT_ID: objects.df[Col.OBJECT_ID].values,   # the input object each survivor came from
        }
        for col in self._score_cols:
            data[col] = objects.column(col).values                 # walk the ancestry for each weighted score
        polygons_gdf = gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=crs)

        transformed = [group.geometry.affine_transform(affine_params(meta_by_id[iid]))
                       for iid, group in polygons_gdf.groupby(GD_TILE_ID, sort=False)]
        polygons_gdf[Col.GEOMETRY] = pd.concat(transformed).reindex(polygons_gdf.index)

        tiles_extent_gdf = gpd.GeoDataFrame({
            GD_TILE_ID: imagery.df[Col.IMAGE_ID].values,
            Col.GEOMETRY: imagery.df[Col.METADATA].map(box_of).values,
        }, geometry=Col.GEOMETRY, crs=crs)
        return polygons_gdf, tiles_extent_gdf

    def _tile_paths(self, imagery: Imagery, image_ids):
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
