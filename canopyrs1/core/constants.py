"""Names of the table columns (``Col``) and of the values some columns take (``GeomKind``,
``Modality``).

These names are saved in run files and in exported GeoPackages. Changing one breaks the runs and
files saved before the change.
"""


class Col:
    """The names of the table columns."""

    # Ids: each table has one column that identifies its rows.
    IMAGE_ID = "image_id"               # in Objects: the image the object was found in
    OBJECT_ID = "object_id"

    # Links between rows.
    PARENT_ID = "parent_id"             # the image this image is part of: a tile's raster, a crop's tile
    PREV_OBJECT_ID = "prev_object_id"   # the object this object was made from: a mask's box

    # Dates and modalities.
    MODALITY = "modality"               # see Modality
    TIMESTAMP = "timestamp"             # the acquisition date
    INSTANCE_ID = "instance_id"         # rows with the same instance id show the same thing: one tree at
                                        # several dates, or one area in several modalities. Unused so far

    # Images.
    PATH = "path"                       # the image's own file; empty for a window into its parent
    METADATA = "metadata"               # the image's georeferencing dict (see core/geometry/georef.py)
    BANDS = "bands"                     # the band numbers to read, starting at 1 (see RGB_BANDS)
    LAZY_CONDITIONS = "lazy_conditions" # for a window: when to skip it, tested once its pixels are
                                        # read (see should_skip); empty to never skip

    # Objects.
    GEOMETRY = "geometry"
    GEOM_KIND = "geom_kind"             # see GeomKind

    # What each model adds to the objects.
    DETECTOR_SCORE = "detector_score"
    DETECTOR_CLASS = "detector_class"
    SEGMENTER_SCORE = "segmenter_score"
    CLASSIFIER_SCORE = "classifier_score"            # the score of the predicted class
    CLASSIFIER_CLASS = "classifier_class"            # the index of the predicted class
    CLASSIFIER_CLASS_NAME = "classifier_class_name"  # the name of the predicted class
    CLASSIFIER_SCORES = "classifier_scores"          # the scores of every class
    AGGREGATOR_SCORE = "aggregator_score"


# The band numbers of an RGB raster: the default value of Col.BANDS.
RGB_BANDS = [1, 2, 3]


class GeomKind:
    """The values of Col.GEOM_KIND. The code handles each kind differently (a mask is saved as a
    COCO RLE, a box as its bounds), so no other value is allowed."""
    BOX = "box"
    MASK = "mask"
    POINT = "point"
    ALL = {BOX, MASK, POINT}


class Modality:
    """The spellings of the common values of Col.MODALITY. Other values are allowed, since no code
    depends on the modality yet."""
    RGB = "rgb"
    MSI = "msi"              # multispectral
    HSI = "hsi"              # hyperspectral
    THERMAL = "thermal"
    POINTCLOUD = "pointcloud"
