"""v3 column vocabulary. Keys (PK/FK) drive the data classes; the rest are data columns that just ride
in the gdfs. Kept here so names stay consistent and ideas don't get lost."""


class Col:
    # primary keys (one per type)
    SOURCE_ID = "source_id"
    TILE_ID = "tile_id"
    OBJECT_ID = "object_id"

    # foreign keys
    # tiles -> sources : SOURCE_ID ; tiles -> objects : OBJECT_ID ; objects -> tiles : TILE_ID
    PREV_OBJECT_ID = "prev_object_id"   # objects -> objects, lineage / history (kept distinct from grouping)

    # grouping
    INSTANCE_ID = "instance_id"         # internal: the (modality, timestamp)-invariant identity. Rows sharing
                                        # it are the same instance observed differently — an object across dates,
                                        # or a tile/footprint across modalities. Type-relative; grouped within a table.

    # observation axes (a source / tile / object is one (modality, timestamp))
    MODALITY = "modality"               # rgb | msi | pointcloud | ...
    TIMESTAMP = "timestamp"             # acquisition date

    # objects
    GEOMETRY = "geometry"
    GEOM_KIND = "geom_kind"             # box | mask | point

    # sources
    SOURCE_PATH = "source_path"

    # tiles (flat: one row per (footprint, modality, timestamp))
    TILE_METADATA = "tile_metadata"     # window: transform/crs/size/gsd
    BANDS = "bands"                     # band indices to read (rgb = [1, 2, 3])
    TILE_PATH = "tile_path"             # pre-cut tile image on disk, else read the window from the source

    # component-specific object attributes
    DETECTOR_SCORE = "detector_score"
    DETECTOR_CLASS = "detector_class"
    SEGMENTER_SCORE = "segmenter_score"
    CLASSIFIER_SCORE = "classifier_score"            # score of the predicted class
    CLASSIFIER_CLASS = "classifier_class"            # predicted class index
    CLASSIFIER_CLASS_NAME = "classifier_class_name"  # human-readable name (from config.class_names)
    CLASSIFIER_SCORES = "classifier_scores"          # full per-class score list
    AGGREGATOR_SCORE = "aggregator_score"


# geom_kind values (what Col.GEOM_KIND holds)
BOX = "box"
MASK = "mask"
POINT = "point"
GEOM_KINDS = {BOX, MASK, POINT}
