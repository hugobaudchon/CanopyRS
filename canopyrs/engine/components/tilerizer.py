"""Tilerizer: produces Tiles or Crops. Modes: a window grid over each source, one crop per Object
(cut from each object's own image file), or grid + re-tiled labels.

Reuses geodataset for the actual tiling. Tiles are flat: one row per (footprint, modality, timestamp),
each a child of the source it was cut from (``parent_id``) — read from its own ``path`` if materialized,
else as a window into the source. Single modality = one row per footprint; multimodal/temporal = several
rows sharing an ``instance_id`` (TODO).
"""

from geodataset.tilerize import RasterTilerizer, RasterPolygonTilerizer, LabeledRasterTilerizer

from canopyrs.engine.tilemeta import serialize_meta
from canopyrs.engine.constants import Col
from canopyrs.engine.data import Crops, Objects, Sources, Tiles
from canopyrs.engine.contracts import Need
from canopyrs.engine.components.base import Component, register_component

# geodataset's own column names in the frames it returns (external vocabulary, not ours).
GD_TILE_ID = "tile_id"
GD_TILE_METADATA = "tile_metadata"
GD_TILE_PATH = "tile_path"


@register_component("tilerizer")
class Tilerizer(Component):
    """``config.tile_type``:
      - ``'tile'``    : grid over each source            -> Tiles
      - ``'polygon'`` : one crop per Object              -> (Crops, Objects)
      - ``'labeled'`` : grid + input Objects re-tiled    -> (Tiles, Objects)
    ``requires`` / ``produces`` reflect the mode."""

    def __init__(self, config, aois_config=None):
        super().__init__(config)
        self.aois_config = aois_config   # geodataset AOIConfig (run-level, from Pipeline.from_config); None = whole raster
        if config.tile_type == "tile":
            self.requires = (Need(Sources),)
            self.produces = Need(Tiles, links=("parent",))
        elif config.tile_type == "polygon":
            # the objects' imagery link supplies the files to crop from (CRS objects over a raster,
            # or tile-pixel detections over tiles) — no separate imagery input.
            self.requires = (Need(Objects, links=("imagery",)),)
            # crops (children of each object's image) + the input objects carried forward (each -> its crop).
            self.produces = (Need(Crops, links=("parent",)),
                             Need(Objects, links=("imagery", "prev_objects"), crs=True, on=Crops))
        elif config.tile_type == "labeled":
            self.requires = (Need(Sources), Need(Objects, crs=True))
            # grid tiles + re-tiled label objects in tile-pixel coords (crs=False).
            self.produces = (Need(Tiles, links=("parent",)),
                             Need(Objects, links=("imagery", "prev_objects"), crs=False, on=Tiles))
        else:
            raise ValueError(f"unknown tile_type '{config.tile_type}'")

    def run(self, *inputs):
        if self.config.tile_type == "tile":
            return self._grid(*inputs)
        if self.config.tile_type == "polygon":
            return self._per_object(*inputs)
        if self.config.tile_type == "labeled":
            return self._labeled(*inputs)
        raise ValueError(f"unknown tile_type '{self.config.tile_type}'")

    def _meta_and_paths(self, gdf):
        """Serialized tile metadata + per-tile paths (the on-disk tile when saved, else None)."""
        metadata = [serialize_meta(meta) for meta in gdf[GD_TILE_METADATA]]
        paths = (list(gdf[GD_TILE_PATH].values) if self.config.save_tiles_to_disk
                 else [None] * len(metadata))
        return metadata, paths

    def _grid(self, sources: Sources) -> Tiles:
        metadata, tile_paths, parent_ids = [], [], []
        for _, source in sources.df.iterrows():
            gdf = RasterTilerizer(
                raster_path=source[Col.PATH],
                output_path=self.out_dir,
                tile_size=self.config.tile_size,
                tile_overlap=self.config.tile_overlap,
                aois_config=self.aois_config,
                scale_factor=self.config.scale_factor,
                ground_resolution=self.config.ground_resolution,
            ).generate_tiles(save_tiles_to_disk=self.config.save_tiles_to_disk)
            source_metadata, source_paths = self._meta_and_paths(gdf)
            metadata += source_metadata
            tile_paths += source_paths
            parent_ids += [source[Col.IMAGE_ID]] * len(source_metadata)
        return Tiles.build(parent_id=parent_ids, metadata=metadata,
                           path=tile_paths, parent=sources)

    def _per_object(self, objects: Objects):
        """One crop per input Object, cut from each object's own image file — its nearest materialized
        ancestor — with one cropper run per distinct file: a single raster degenerates to one run, a
        folder of on-disk tiles runs once per tile file. Returns (Crops, Objects): the crops
        (children of each object's image) and the input objects carried forward — each re-parented to
        its crop (``image_id`` -> the crop) with a ``prev_object_id`` -> the original (so the ancestry
        walk still reaches its scores), geometry in CRS coords. The classifier consumes these objects,
        each finding its crop via ``object.imagery``."""
        metadata, tile_paths, parent_ids, source_object_ids, geometry, crs = [], [], [], [], [], None
        for path, group in objects.group_by_materialized_source():
            gdf = RasterPolygonTilerizer(
                raster_path=path,
                labels_path=None,
                labels_gdf=group[[Col.OBJECT_ID, Col.GEOMETRY]],
                output_path=self.out_dir,
                tile_size=self.config.tile_size,
                use_variable_tile_size=self.config.use_variable_tile_size,
                variable_tile_size_pixel_buffer=self.config.variable_tile_size_pixel_buffer,
                aois_config=self.aois_config,
                other_labels_attributes_column_names=[Col.OBJECT_ID],   # carry the source object id through geodataset
                scale_factor=self.config.scale_factor,
                ground_resolution=self.config.ground_resolution,
            ).generate_tiles_gdf(save_tiles=self.config.save_tiles_to_disk)
            group_meta, group_paths = self._meta_and_paths(gdf)
            kept = gdf[Col.OBJECT_ID].values                      # crop order; geodataset may drop empty crops
            by_object = group.set_index(Col.OBJECT_ID)
            metadata += group_meta
            tile_paths += group_paths
            parent_ids += list(by_object.loc[kept, Col.IMAGE_ID].values)   # crop's parent = the object's image
            geometry += list(by_object.loc[kept, Col.GEOMETRY].values)     # the CRS geometry from the grouping
            source_object_ids += list(kept)
            assert crs is None or group.crs == crs, "crop sources span multiple CRS; not supported"
            crs = group.crs
        crops = Crops.build(parent_id=parent_ids, metadata=metadata,
                            path=tile_paths, parent=objects.linked("imagery"))

        # Carry each crop's source object forward (same kind, CRS geometry), now pointing at its crop.
        by_id = objects.df.set_index(Col.OBJECT_ID)
        carried = Objects.build(
            geometry=geometry,
            geom_kind=by_id.loc[source_object_ids, Col.GEOM_KIND].values,
            image_id=crops.df[Col.IMAGE_ID].values,               # crop per object (1:1, same order)
            prev_object_id=source_object_ids,
            crs=crs,
            imagery=crops,
            prev_objects=objects,
        )
        return crops, carried

    def _labeled(self, sources: Sources, labels: Objects):
        """Grid tiles + the input label Objects re-tiled into each tile (tile-pixel coords), via
        geodataset's LabeledRasterTilerizer. Returns (Tiles, Objects): each output object -> its tile
        (image_id) and its source label (prev_object_id)."""
        assert len(sources.df) == 1, "labeled tilerizer currently supports a single source"
        source = sources.df.iloc[0]
        carry = [Col.OBJECT_ID] + ([Col.GEOM_KIND] if Col.GEOM_KIND in labels.df.columns else [])
        tiles_gdf, labels_gdf = LabeledRasterTilerizer(
            raster_path=source[Col.PATH],
            labels_path=None,
            labels_gdf=labels.df,
            output_path=self.out_dir,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            aois_config=self.aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            other_labels_attributes_column_names=carry,   # carries object_id (+ geom_kind) onto the labels
        ).generate_tiles_gdf(save_tiles=self.config.save_tiles_to_disk)

        metadata, tile_paths = self._meta_and_paths(tiles_gdf)
        tiles = Tiles.build(parent_id=source[Col.IMAGE_ID],
                            image_id=tiles_gdf[GD_TILE_ID].values,
                            metadata=metadata, path=tile_paths, parent=sources)

        objects = Objects.build(
            geometry=labels_gdf.geometry.values,
            geom_kind=labels_gdf[Col.GEOM_KIND].values if Col.GEOM_KIND in labels_gdf else None,
            image_id=labels_gdf[GD_TILE_ID].values,
            prev_object_id=labels_gdf[Col.OBJECT_ID].values,
            imagery=tiles,
            prev_objects=labels,
        )
        return tiles, objects
