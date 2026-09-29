import pytest
from shapely.geometry import box

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.geometry.georef import make_georef, window_georef
from canopyrs1.core.tables.contracts import Need
from canopyrs1.core.tables.imagery import Crops, Sources, Tiles
from canopyrs1.core.tables.objects import Objects
from canopyrs1.pipeline.simulate import simulate

# =============================================================================
# Fake components, declared like the pipeline's
# =============================================================================


class GridTilerizer:
    requires = (Need(Sources),)
    produces = (Need(Tiles, links=("parent_imagery",)),)


class Detector:
    requires = (Need(Tiles),)
    produces = (
        Need(
            Objects,
            columns=(Col.DETECTOR_SCORE,),
            links=("parent_imagery",),
            has_crs=False,
            on=Tiles,
        ),
    )


class Aggregator:
    requires = (
        Need(Objects, columns=(Col.DETECTOR_SCORE,), links=("parent_imagery",), on=Tiles),
        Need(Tiles),
    )
    produces = (
        Need(
            Objects,
            columns=(Col.AGGREGATOR_SCORE,),
            links=("parent_imagery", "parent_objects"),
            has_crs=True,
            on=Tiles,
        ),
    )


class CropTilerizer:
    requires = (Need(Objects, links=("parent_imagery",)),)
    produces = (
        Need(Crops, links=("parent_imagery",)),
        Need(
            Objects,
            links=("parent_imagery", "parent_objects"),
            has_crs=True,
            on=Crops,
        ),
    )


class Classifier:
    requires = (Need(Objects, links=("parent_imagery",), on=Crops),)
    # Its objects are linked to the ones it classified only: they find their crops through them.
    produces = (Need(Objects, columns=(Col.CLASSIFIER_SCORE,), links=("parent_objects",)),)


class Relabeler:
    """Makes new objects from nothing it declares: they don't keep any history."""

    requires = (Need(Objects),)
    produces = (Need(Objects, columns=(Col.CLASSIFIER_CLASS,)),)


FULL_PIPELINE = [GridTilerizer(), Detector(), Aggregator(), CropTilerizer(), Classifier()]


def _sources():
    raster = make_georef(
        transform=[0.1, 0, 600000, 0, -0.1, 5040000],
        crs="EPSG:32618",
        width=2048,
        height=2048,
        count=3,
        dtype="uint8",
    )
    return Sources.build(georef=[raster], path="ortho.tif")


# =============================================================================
# simulate
# =============================================================================


def test_a_full_pipeline():
    simulated = simulate([_sources()], FULL_PIPELINE)
    assert [s.component for s in simulated] == FULL_PIPELINE
    objects = simulated[-1].after[Objects]
    # The classified objects keep every score of their history, and reach their crops through it.
    assert {Col.DETECTOR_SCORE, Col.AGGREGATOR_SCORE, Col.CLASSIFIER_SCORE} <= objects.columns
    assert objects.links == {"parent_imagery", "parent_objects"}
    assert objects.on is Crops


def test_what_each_step_finds_before_it():
    tiling, detection, aggregation = simulate([_sources()], FULL_PIPELINE[:3])
    assert set(tiling.before) == {Sources}
    assert set(tiling.after) == {Sources, Tiles}
    assert set(detection.before) == {Sources, Tiles}
    # The aggregator reads the detector's objects, and replaces them as the newest.
    assert aggregation.before[Objects].has_crs is False
    assert aggregation.after[Objects].has_crs is True
    assert Col.DETECTOR_SCORE in aggregation.after[Objects].columns  # kept from its history


def test_objects_found_in_their_own_imagery():
    # The crop tilerizer's objects are linked to both their crops and the aggregated objects: they
    # are found in their crops, not in the tiles of their history.
    simulated = simulate([_sources()], FULL_PIPELINE[:4])
    assert simulated[-1].after[Objects].on is Crops


def test_objects_without_history():
    simulated = simulate([_sources()], [GridTilerizer(), Detector(), Relabeler()])
    objects = simulated[-1].after[Objects]
    assert objects.columns == {Col.CLASSIFIER_CLASS}
    assert objects.links == set() and objects.on is None


def test_starting_from_other_tables():
    # A pipeline can start from any tables, here the objects found by an earlier run.
    sources = _sources()
    tile = window_georef(sources.df[Col.GEOREF][0], col_off=0, row_off=0, width=1024, height=1024)
    tiles = Tiles.build(georef=[tile], parent_image_id=0, parent_imagery=sources)
    boxes = Objects.build(
        geometry=[box(0, 0, 10, 10)],
        geom_kind=GeomKind.BOX,
        parent_image_id=0,
        parent_imagery=tiles,
        columns={Col.DETECTOR_SCORE: [0.9]},
    )
    simulated = simulate([tiles, boxes], [Aggregator(), CropTilerizer(), Classifier()])
    assert simulated[0].before[Objects].columns >= {Col.DETECTOR_SCORE}


def test_a_tuple_in_produces():
    class Wrong:
        requires = ()
        produces = (Need(Objects, on=(Tiles, Crops)),)

    with pytest.raises(ValueError, match="a tuple in on= is only for requires"):
        simulate([], [Wrong()])


# =============================================================================
# Bad wiring
# =============================================================================


def test_a_step_without_its_tables():
    with pytest.raises(ValueError) as error:
        simulate([_sources()], [Detector()])
    assert (
        str(error.value) == "Step 1, Detector, can't run:\n  - There are no Tiles before this step"
    )


def test_a_classifier_right_after_a_detector():
    components = [GridTilerizer(), Detector(), Classifier()]
    with pytest.raises(ValueError) as error:
        simulate([_sources()], components)
    assert str(error.value) == (
        "Step 3, Classifier, can't run:\n  - Objects must be found in Crops, not in Tiles"
    )


def test_every_problem_of_the_step_is_listed():
    with pytest.raises(ValueError) as error:
        simulate([_sources()], [Aggregator()])
    assert str(error.value) == (
        "Step 1, Aggregator, can't run:\n"
        "  - There are no Objects before this step\n"
        "  - There are no Tiles before this step"
    )


def test_only_the_first_failing_step_is_reported():
    # Without a tilerizer, the detector fails, and so would the aggregator after it.
    with pytest.raises(ValueError) as error:
        simulate([_sources()], FULL_PIPELINE[1:])
    assert str(error.value).startswith("Step 1, Detector, can't run:")
    assert "Aggregator" not in str(error.value)
