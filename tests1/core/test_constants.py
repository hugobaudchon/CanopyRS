"""The column names and values are saved in run files and GeoPackages: they must not change."""

from canopyrs1.core.constants import Col, GeomKind, Modality, RGB_BANDS


def _names(cls):
    """Return the public attributes of ``cls`` as a dict, e.g. {"IMAGE_ID": "image_id", ...}."""
    return {name: value for name, value in vars(cls).items() if name.isupper()}


def test_column_names():
    assert _names(Col) == {
        "IMAGE_ID": "image_id",
        "OBJECT_ID": "object_id",
        "PARENT_ID": "parent_id",
        "PREV_OBJECT_ID": "prev_object_id",
        "MODALITY": "modality",
        "TIMESTAMP": "timestamp",
        "INSTANCE_ID": "instance_id",
        "PATH": "path",
        "GEOREF": "georef",
        "BANDS": "bands",
        "LAZY_CONDITIONS": "lazy_conditions",
        "GEOMETRY": "geometry",
        "GEOM_KIND": "geom_kind",
        "DETECTOR_SCORE": "detector_score",
        "DETECTOR_CLASS": "detector_class",
        "SEGMENTER_SCORE": "segmenter_score",
        "CLASSIFIER_SCORE": "classifier_score",
        "CLASSIFIER_CLASS": "classifier_class",
        "CLASSIFIER_CLASS_NAME": "classifier_class_name",
        "CLASSIFIER_SCORES": "classifier_scores",
        "AGGREGATOR_SCORE": "aggregator_score",
    }


def test_geom_kinds():
    assert (GeomKind.BOX, GeomKind.MASK, GeomKind.POINT) == ("box", "mask", "point")
    assert GeomKind.ALL == {"box", "mask", "point"}


def test_modalities():
    assert _names(Modality) == {
        "RGB": "rgb", "MSI": "msi", "HSI": "hsi", "THERMAL": "thermal", "POINTCLOUD": "pointcloud",
    }


def test_rgb_bands():
    assert RGB_BANDS == [1, 2, 3]
