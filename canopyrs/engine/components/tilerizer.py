"""Tilerizer: source Imagery -> tile Imagery. Modes: a window grid over each source, one crop per
Object, or grid + re-tiled labels.

Reuses geodataset for the actual tiling. Tiles are flat: one row per (footprint, modality, timestamp),
each a child of the source it was cut from (``parent_id``) — read from its own ``path`` if materialized,
else as a window into the source. Single modality = one row per footprint; multimodal/temporal = several
rows sharing an ``instance_id`` (TODO).
"""

from geodataset.tilerize import RasterTilerizer, RasterPolygonTilerizer, LabeledRasterTilerizer

from canopyrs.engine.tilemeta import serialize_meta
from canopyrs.engine.constants import Col, ImageKind
from canopyrs.engine.data import Imagery, Objects
from canopyrs.engine.contracts import Need
from canopyrs.engine.components.base import Component, register_component

# geodataset's own column names in the frames it returns (external vocabulary, not ours).
GD_TILE_ID = "tile_id"
GD_TILE_METADATA = "tile_metadata"
GD_TILE_PATH = "tile_path"


@register_component("tilerizer")
class Tilerizer(Component):
    """``config.tile_type``:
      - ``'tile'``    : grid over each source            -> Imagery (tiles)
      - ``'polygon'`` : one crop per Object              -> (Imagery, Objects)
      - ``'labeled'`` : grid + input Objects re-tiled    -> (Imagery, Objects)
    ``requires`` / ``produces`` reflect the mode."""

    def __init__(self, config, aois_config=None):
        super().__init__(config)
        self.aois_config = aois_config   # geodataset AOIConfig (run-level, from Pipeline.from_config); None = whole raster
        if config.tile_type == "tile":
            self.requires = (Need(Imagery, kind=ImageKind.SOURCE),)
            self.produces = Need(Imagery, kind=ImageKind.TILE, links=("parent",))
        elif config.tile_type == "polygon":
            self.requires = (Need(Imagery, kind=ImageKind.SOURCE), Need(Objects, crs=True))
            # crops (children of the source) + the input objects carried forward (each -> its crop).
            self.produces = (Need(Imagery, kind=ImageKind.TILE, links=("parent",)),
                             Need(Objects, links=("imagery", "prev_objects"), crs=True))
        elif config.tile_type == "labeled":
            self.requires = (Need(Imagery, kind=ImageKind.SOURCE), Need(Objects, crs=True))
            # grid tiles + re-tiled label objects in tile-pixel coords (crs=False).
            self.produces = (Need(Imagery, kind=ImageKind.TILE, links=("parent",)),
                             Need(Objects, links=("imagery", "prev_objects"), crs=False))
        else:
            raise ValueError(f"unknown tile_type '{config.tile_type}'")

    def run(self, sources: Imagery, objects: Objects = None):
        if self.config.tile_type == "tile":
            return self._grid(sources)
        if self.config.tile_type == "polygon":
            return self._per_object(sources, objects)
        if self.config.tile_type == "labeled":
            return self._labeled(sources, objects)
        raise ValueError(f"unknown tile_type '{self.config.tile_type}'")

    def _meta_and_paths(self, gdf):
        """Serialized tile metadata + per-tile paths (the on-disk tile when saved, else None)."""
        metadata = [serialize_meta(meta) for meta in gdf[GD_TILE_METADATA]]
        paths = (list(gdf[GD_TILE_PATH].values) if self.config.save_tiles_to_disk
                 else [None] * len(metadata))
        return metadata, paths

    def _grid(self, sources: Imagery) -> Imagery:
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
        return Imagery.build(kind=ImageKind.TILE, parent_id=parent_ids, metadata=metadata,
                             path=tile_paths, parent=sources)

    def _per_object(self, sources: Imagery, objects: Objects):
        """One crop per input Object. Returns (Imagery, Objects): the crops (children of the source),
        and the input objects carried forward — each re-parented to its crop (``image_id`` -> the crop)
        with a ``prev_object_id`` -> the original (so the ancestry walk still reaches its scores). The
        classifier consumes these objects, each finding its crop via ``object.imagery``."""
        assert len(sources.df) == 1, "object tilerizer currently supports a single source"
        source = sources.df.iloc[0]
        gdf = RasterPolygonTilerizer(
            raster_path=source[Col.PATH],
            labels_path=None,
            labels_gdf=objects.df[[Col.OBJECT_ID, Col.GEOMETRY]],
            output_path=self.out_dir,
            tile_size=self.config.tile_size,
            use_variable_tile_size=self.config.use_variable_tile_size,
            variable_tile_size_pixel_buffer=self.config.variable_tile_size_pixel_buffer,
            aois_config=self.aois_config,
            other_labels_attributes_column_names=[Col.OBJECT_ID],   # carry the source object id through geodataset
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
        ).generate_tiles_gdf(save_tiles=self.config.save_tiles_to_disk)
        metadata, tile_paths = self._meta_and_paths(gdf)
        crops = Imagery.build(kind=ImageKind.TILE, parent_id=source[Col.IMAGE_ID], metadata=metadata,
                              path=tile_paths, parent=sources)

        # Carry each crop's source object forward (same geometry/kind), now pointing at its crop.
        source_object_ids = gdf[Col.OBJECT_ID].values             # the input object each crop came from, in crop order
        by_id = objects.df.set_index(Col.OBJECT_ID)
        carried = Objects.build(
            geometry=by_id.loc[source_object_ids, Col.GEOMETRY].values,
            geom_kind=by_id.loc[source_object_ids, Col.GEOM_KIND].values,
            image_id=crops.df[Col.IMAGE_ID].values,               # crop per object (1:1, same order)
            prev_object_id=source_object_ids,
            crs=objects.df.crs,                                   # carried geometry stays in the input's CRS
            imagery=crops,
            prev_objects=objects,
        )
        return crops, carried

    def _labeled(self, sources: Imagery, labels: Objects):
        """Grid tiles + the input label Objects re-tiled into each tile (tile-pixel coords), via
        geodataset's LabeledRasterTilerizer. Returns (Imagery, Objects): each output object -> its tile
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
        tiles = Imagery.build(kind=ImageKind.TILE, parent_id=source[Col.IMAGE_ID],
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
