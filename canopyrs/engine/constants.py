"""Column vocabulary and value classes. Keys (PK/FK) drive the data classes; the rest are data columns
that just ride in the dataframes. ``Col`` names the columns; ``ImageKind`` / ``GeomKind`` / ``Modality``
name their values. Kept here so names stay consistent and ideas don't get lost."""


class Col:
    # primary keys (one per type)
    IMAGE_ID = "image_id"
    OBJECT_ID = "object_id"

    # foreign keys
    PARENT_ID = "parent_id"             # imagery -> imagery, containment (a tile in its raster, a crop in its tile)
    PREV_OBJECT_ID = "prev_object_id"   # objects -> objects, lineage / history (kept distinct from grouping)
    # objects -> imagery reuses IMAGE_ID (the image an object was found in)

    # grouping
    INSTANCE_ID = "instance_id"         # internal: the (modality, timestamp)-invariant identity. Rows sharing
                                        # it are the same instance observed differently — an object across dates,
                                        # or a footprint across modalities. Type-relative; grouped within a table.

    # observation axes (an image / object is one (modality, timestamp))
    MODALITY = "modality"               # see Modality
    TIMESTAMP = "timestamp"             # acquisition date

    # imagery
    KIND = "kind"                       # see ImageKind; uniform per table, checked by Need like crs
    PATH = "path"                       # file on disk; null = a window into the parent image
    METADATA = "metadata"               # per-row dict; grid rows: transform/crs/size (see tilemeta)
    BANDS = "bands"                     # band indices to read (rgb = [1, 2, 3])

    # objects
    GEOMETRY = "geometry"
    GEOM_KIND = "geom_kind"             # see GeomKind

    # component-specific object attributes
    DETECTOR_SCORE = "detector_score"
    DETECTOR_CLASS = "detector_class"
    SEGMENTER_SCORE = "segmenter_score"
    CLASSIFIER_SCORE = "classifier_score"            # score of the predicted class
    CLASSIFIER_CLASS = "classifier_class"            # predicted class index
    CLASSIFIER_CLASS_NAME = "classifier_class_name"  # human-readable name (from config.class_names)
    CLASSIFIER_SCORES = "classifier_scores"          # full per-class score list
    AGGREGATOR_SCORE = "aggregator_score"


class ImageKind:
    """What an Imagery row is: a whole input scene, or a model-consumable tile (crops included).
    Closed set — contracts and components branch on it."""
    SOURCE = "source"
    TILE = "tile"
    ALL = {SOURCE, TILE}


class GeomKind:
    """What Col.GEOM_KIND holds. Closed set — the engine branches on it (masks -> RLE, boxes -> bounds)."""
    BOX = "box"
    MASK = "mask"
    POINT = "point"
    ALL = {BOX, MASK, POINT}


class Modality:
    """Canonical modality strings (Col.MODALITY). Open set for now — no engine dispatch exists yet, so
    unknown values are allowed; the class fixes the spelling before values land in persisted run records."""
    RGB = "rgb"
    MSI = "msi"              # multispectral
    HSI = "hsi"              # hyperspectral
    THERMAL = "thermal"
    POINTCLOUD = "pointcloud"
