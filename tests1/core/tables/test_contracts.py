import pytest
from shapely.geometry import box

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.geometry.georef import make_georef, window_georef
from canopyrs1.core.tables.contracts import Need, Schema
from canopyrs1.core.tables.imagery import Crops, Imagery, Sources, Tiles
from canopyrs1.core.tables.objects import Objects

RASTER = make_georef(
    transform=[0.1, 0, 600000, 0, -0.1, 5040000],
    crs="EPSG:32618",
    width=2048,
    height=2048,
    count=3,
    dtype="uint8",
)


def _pipeline_tables():
    """Return (sources, tiles, boxes, masks, crops, classified): boxes found on tiles of a source,
    masks made from the boxes (linked to them only), crops cut around the masks, and the masks
    carried onto their crops by a classifier."""
    sources = Sources.build(georef=[RASTER], path="ortho.tif")
    tile = window_georef(RASTER, col_off=0, row_off=0, width=1024, height=1024)
    tiles = Tiles.build(georef=[tile], parent_image_id=0, parent_imagery=sources)
    boxes = Objects.build(
        geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)],
        geom_kind=GeomKind.BOX,
        parent_image_id=0,
        parent_imagery=tiles,
        columns={Col.DETECTOR_SCORE: [0.9, 0.8]},
    )
    masks = Objects.build(
        geometry=[box(1, 1, 9, 9), box(21, 21, 29, 29)],
        geom_kind=GeomKind.MASK,
        parent_object_id=[0, 1],
        parent_objects=boxes,
        columns={Col.SEGMENTER_SCORE: [0.7, 0.6]},
    )
    crop = window_georef(tile, col_off=0, row_off=0, width=64, height=64)
    crops = Crops.build(georef=[crop, crop], parent_image_id=0, parent_imagery=tiles)
    classified = Objects.build(
        geometry=[box(1, 1, 9, 9), box(21, 21, 29, 29)],
        geom_kind=GeomKind.MASK,
        crs="EPSG:32618",
        parent_image_id=[0, 1],
        parent_imagery=crops,
        parent_object_id=[0, 1],
        parent_objects=masks,
        columns={Col.CLASSIFIER_SCORE: [0.5, 0.4]},
    )
    return sources, tiles, boxes, masks, crops, classified


# =============================================================================
# Need.check
# =============================================================================


def test_a_schema_that_meets_the_need():
    need = Need(
        Objects,
        columns=(Col.DETECTOR_SCORE,),
        links=("parent_imagery",),
        has_crs=False,
        on=Tiles,
    )
    schema = Schema(
        columns={Col.DETECTOR_SCORE, Col.GEOMETRY},
        links={"parent_imagery"},
        has_crs=False,
        on=Tiles,
    )
    assert need.check(schema) == []
    assert Need(Objects).check(Schema()) == []  # a bare type needs nothing more


def test_a_missing_link():
    need = Need(Objects, links=("parent_imagery", "parent_objects"))
    assert need.check(Schema(links={"parent_imagery"})) == [
        "Objects must be linked to their parent_objects"
    ]
    assert need.check(Schema()) == [
        "Objects must be linked to their parent_imagery and parent_objects"
    ]


def test_missing_columns():
    need = Need(Objects, columns=(Col.DETECTOR_SCORE, Col.SEGMENTER_SCORE, Col.GEOMETRY))
    assert need.check(Schema(columns={Col.GEOMETRY})) == [
        "Objects must have the columns detector_score, segmenter_score"
    ]
    assert need.check(Schema(columns={Col.GEOMETRY, Col.DETECTOR_SCORE})) == [
        "Objects must have the column segmenter_score"
    ]


def test_geometry_in_a_crs_or_in_pixels():
    in_crs, in_pixels = Schema(has_crs=True), Schema(has_crs=False)
    assert Need(Objects, has_crs=True).check(in_crs) == []
    assert Need(Objects, has_crs=True).check(in_pixels) == [
        "Objects must have their geometry in a CRS"
    ]
    assert Need(Objects, has_crs=False).check(in_crs) == [
        "Objects must have their geometry in pixel coordinates"
    ]
    assert Need(Objects).check(in_crs) == []  # either


def test_the_imagery_objects_were_found_in():
    on_tiles = Schema(on=Tiles)
    assert Need(Objects, on=Tiles).check(on_tiles) == []
    assert Need(Objects, on=(Tiles, Crops)).check(on_tiles) == []
    assert Need(Objects, on=Imagery).check(on_tiles) == []  # any imagery
    assert Need(Objects, on=Crops).check(on_tiles) == [
        "Objects must be found in Crops, not in Tiles"
    ]
    assert Need(Objects, on=(Sources, Crops)).check(on_tiles) == [
        "Objects must be found in Sources or Crops, not in Tiles"
    ]


def test_every_problem_is_listed():
    need = Need(
        Objects,
        columns=(Col.DETECTOR_SCORE,),
        links=("parent_imagery",),
        has_crs=True,
        on=Crops,
    )
    assert need.check(Schema(has_crs=False, on=Tiles)) == [
        "Objects must be linked to their parent_imagery",
        "Objects must have the column detector_score",
        "Objects must have their geometry in a CRS",
        "Objects must be found in Crops, not in Tiles",
    ]


UNKNOWN_FIELDS = [
    Need(Objects, has_crs=True),
    Need(Objects, has_crs=False),
    Need(Objects, on=Crops),
]


@pytest.mark.parametrize("need", UNKNOWN_FIELDS)
def test_unknown_fields_are_not_checked(need):
    assert need.check(Schema()) == []


# =============================================================================
# Each table's schema
# =============================================================================


def test_the_schema_of_images():
    sources, tiles, _, _, crops, _ = _pipeline_tables()
    assert sources.schema().links == set()
    assert tiles.schema().links == {"parent_imagery"}
    assert crops.schema().links == {"parent_imagery"}
    assert tiles.schema().has_crs is True
    assert tiles.schema().on is None
    assert {Col.IMAGE_ID, Col.GEOREF, Col.GEOMETRY} <= tiles.schema().columns


def test_columns_without_values_are_not_offered():
    sources, tiles, _, _, _, _ = _pipeline_tables()
    assert Col.PATH in sources.schema().columns
    assert Col.PATH not in tiles.schema().columns  # windows: every path is missing
    assert Col.PARENT_IMAGE_ID not in sources.schema().columns


def test_the_schema_of_objects_found_on_tiles():
    _, _, boxes, _, _, _ = _pipeline_tables()
    schema = boxes.schema()
    assert schema.links == {"parent_imagery"}
    assert schema.on is Tiles
    assert schema.has_crs is False
    assert Col.DETECTOR_SCORE in schema.columns
    assert Col.PARENT_OBJECT_ID not in schema.columns


def test_objects_offer_the_columns_and_imagery_of_their_history():
    _, _, _, masks, _, classified = _pipeline_tables()
    # The masks are linked to the boxes only: they reach the tiles through them.
    schema = masks.schema()
    assert schema.links == {"parent_imagery", "parent_objects"}
    assert schema.on is Tiles
    assert {Col.SEGMENTER_SCORE, Col.DETECTOR_SCORE} <= schema.columns
    # Two steps further back, and on their own crops.
    schema = classified.schema()
    assert schema.on is Crops and schema.has_crs is True
    assert {Col.CLASSIFIER_SCORE, Col.SEGMENTER_SCORE, Col.DETECTOR_SCORE} <= schema.columns


def test_a_column_without_values_hides_the_history_s():
    # Like get_column, a column the objects have is read from them, even if its values are missing.
    _, _, boxes, _, _, _ = _pipeline_tables()
    rescored = Objects.build(
        geometry=[box(0, 0, 10, 10)],
        geom_kind=GeomKind.BOX,
        parent_object_id=[0],
        parent_objects=boxes,
        columns={Col.DETECTOR_SCORE: [None]},
    )
    assert Col.DETECTOR_SCORE not in rescored.schema().columns


def test_objects_without_imagery():
    lone = Objects.build(geometry=[box(0, 0, 1, 1)], geom_kind=GeomKind.BOX)
    schema = lone.schema()
    assert schema.links == set() and schema.on is None


def test_the_empty_tables_of_a_step_that_found_nothing():
    # An empty table offers the columns it has, so a step's checks pass on it.
    _, tiles, _, _, _, _ = _pipeline_tables()
    empty = Objects.build(
        geometry=[],
        geom_kind=GeomKind.BOX,
        parent_imagery=tiles,
        columns={Col.DETECTOR_SCORE: []},
    )
    need = Need(Objects, columns=(Col.DETECTOR_SCORE,), links=("parent_imagery",), on=Tiles)
    assert need.check(empty.schema()) == []


def test_a_classifier_s_need_on_real_tables():
    _, _, boxes, _, _, classified = _pipeline_tables()
    classifier_needs = Need(Objects, links=("parent_imagery",), on=Crops)
    assert classifier_needs.check(classified.schema()) == []
    assert classifier_needs.check(boxes.schema()) == [
        "Objects must be found in Crops, not in Tiles"
    ]
