"""Unit tests for the data contracts (Need / one_of / Schema) and the pipeline's input matching
(newest table of the requested type, checked strictly) + static validation + seed rules."""

import pytest

from canopyrs.engine.contracts import Need, one_of, Schema
from canopyrs.engine.data import Crops, Objects, Sources, Tiles
from canopyrs.engine.constants import Col, Modality
from canopyrs.engine.pipeline import Pipeline
from tests.conftest import make_tile_metadata


class FakeComponent:
    """A stand-in component carrying only the contract declarations validate() reads."""

    def __init__(self, requires, produces, name="fake"):
        self.requires = requires if isinstance(requires, tuple) else (requires,)
        self.produces = produces
        self.name = name
        self.component_id = None
        self.out_dir = None

    @property
    def label(self):
        return f"{self.component_id}_{self.name}" if self.component_id is not None else self.name


def _tilerizer():
    return FakeComponent(requires=Need(Sources),
                         produces=Need(Tiles, links=("parent",)),
                         name="tilerizer")


def _detector():
    return FakeComponent(
        requires=Need(Tiles),
        produces=Need(Objects, columns=(Col.DETECTOR_SCORE,), links=("imagery",), crs=False, on=Tiles),
        name="detector",
    )


def _polygon_tilerizer():
    return FakeComponent(
        requires=Need(Objects, links=("imagery",)),
        produces=(Need(Crops, links=("parent",)),
                  Need(Objects, links=("imagery", "prev_objects"), crs=True, on=Crops)),
        name="polygon_tilerizer",
    )


def _classifier():
    return FakeComponent(
        requires=Need(Objects, links=("imagery",), on=Crops),
        produces=Need(Objects, columns=(Col.CLASSIFIER_CLASS,), links=("prev_objects",), on=Crops),
        name="classifier",
    )


def _aggregator():
    return FakeComponent(
        requires=(Need(Objects, links=("imagery",), columns=(Col.DETECTOR_SCORE,), on=(Tiles, Crops)),
                  Need(Tiles)),
        produces=Need(Objects, columns=(Col.AGGREGATOR_SCORE,), links=("prev_objects",), crs=True),
        name="aggregator",
    )


# --- Need.check against a live table's schema snapshot ------------------------

def test_need_columns(objects_seed):
    assert Need(Objects, columns=(Col.DETECTOR_SCORE,)).check(objects_seed.schema()) == ""
    assert "must expose column" in Need(Objects, columns=("nope",)).check(objects_seed.schema())


def test_need_links(objects_seed):
    assert Need(Objects, links=("imagery",)).check(objects_seed.schema()) == ""
    assert "must be linked" in Need(Objects, links=("prev_objects",)).check(objects_seed.schema())


def test_need_crs(objects_seed):
    # detector boxes are in tile-pixel coords (crs=None)
    assert Need(Objects, crs=False).check(objects_seed.schema()) == ""
    assert "CRS coords" in Need(Objects, crs=True).check(objects_seed.schema())


def test_need_on(objects_seed):
    # objects_seed lives on the tiles_seed (Tiles)
    assert Need(Objects, on=Tiles).check(objects_seed.schema()) == ""
    assert Need(Objects, on=(Tiles, Crops)).check(objects_seed.schema()) == ""
    assert "must live on Crops" in Need(Objects, on=Crops).check(objects_seed.schema())


def test_need_modalities(sources_seed):
    # from_paths defaults every row to rgb
    assert Need(Sources, modalities=(Modality.RGB,)).check(sources_seed.schema()) == ""
    assert "modality" in Need(Sources, modalities=(Modality.POINTCLOUD,)).check(sources_seed.schema())


# --- Schema descriptor -------------------------------------------------------

def test_schema_descriptor():
    schema = Schema(columns={Col.DETECTOR_SCORE}, links={"imagery"}, crs=False,
                    on=Tiles, modalities={Modality.RGB})
    assert schema.provides(Col.DETECTOR_SCORE) and not schema.provides("x")
    assert schema.has_link("imagery") and not schema.has_link("prev_objects")
    assert schema.crs_set is False
    assert schema.on is Tiles
    assert Need(Objects, on=Tiles, modalities=(Modality.RGB,)).check(schema) == ""


def test_schema_undeclared_attributes_skip_checks():
    # on/modalities None on the schema = producer didn't declare -> checks pass (tri-state)
    schema = Schema(columns=(), links=())
    assert Need(Objects, on=Crops).check(schema) == ""
    assert Need(Tiles, modalities=(Modality.RGB,)).check(schema) == ""


# --- one_of ------------------------------------------------------------------

def test_one_of_picks_first_available(tiles_seed):
    req = one_of(Need(Sources), Need(Tiles))
    desc, err = req.resolve(lambda t: {Tiles: [tiles_seed]}.get(t, ()))
    assert desc is tiles_seed and err == ""


def test_one_of_none_available():
    req = one_of(Need(Sources), Need(Tiles))
    desc, err = req.resolve(lambda t: ())
    assert desc is None and "OR" in err


# --- input matching: newest of the requested type, checked strictly -----------

def test_matching_selects_by_type(sources_seed, tiles_seed):
    """A Sources need binds the source seed no matter how many tiles exist (the tilerizer-after-
    tiles case is a plain type lookup now)."""
    available = {Sources: [sources_seed], Tiles: [tiles_seed]}
    desc, err = Need(Sources).resolve(lambda t: available.get(t, ()))
    assert desc is sources_seed and err == ""


def test_matching_takes_newest_of_type(sources_seed, tiles_seed):
    newer = Tiles.build(metadata=[make_tile_metadata()])
    available = {Tiles: [tiles_seed, newer]}   # oldest -> newest
    desc, _ = Need(Tiles).resolve(lambda t: available.get(t, ()))
    assert desc is newer


def test_matching_missing_type_reports(tiles_seed):
    desc, err = Need(Sources).resolve(lambda t: {Tiles: [tiles_seed]}.get(t, ()))
    assert desc is None and "requires Sources" in err


def test_matching_never_falls_back_past_a_broken_newest(tiles_seed):
    """When the newest table of the requested type fails the check, matching errors instead of
    silently binding an older table (config errors stay loud)."""
    unlinked_tiles = Tiles.build(metadata=[make_tile_metadata()])
    available = {Tiles: [tiles_seed, unlinked_tiles]}   # oldest -> newest; only the old one has 'parent'
    desc, err = Need(Tiles, links=("parent",)).resolve(lambda t: available.get(t, ()))
    assert desc is None and "must be linked" in err


# --- Pipeline.validate (static wiring check) ---------------------------------

def test_validate_rejects_detector_on_untiled_source(sources_seed):
    with pytest.raises(ValueError):
        Pipeline([_detector()], sources=sources_seed)


def test_validate_accepts_wired_pipeline(sources_seed):
    pipe = Pipeline([_tilerizer(), _detector()], sources=sources_seed)
    assert [c.component_id for c in pipe.components] == [0, 1]


def test_validate_class_aware_pipeline(sources_seed):
    """The full class-aware chain validates: grid tilerizer -> detector -> polygon tilerizer ->
    classifier -> aggregator (the aggregator's Tiles input reaches the grid tiles past the crops,
    and the classifier's score/link needs resolve through the lineage)."""
    pipe = Pipeline([_tilerizer(), _detector(), _polygon_tilerizer(), _classifier(), _aggregator()],
                    sources=sources_seed)
    assert len(pipe.components) == 5


def test_validate_source_need_survives_produced_tiles(sources_seed):
    """A second tilerizer's Sources requirement stays satisfiable after the first produced tiles
    (per-type schema lists, not overwrite)."""
    pipe = Pipeline([_tilerizer(), _tilerizer()], sources=sources_seed)
    assert len(pipe.components) == 2


def test_validate_rejects_classifier_on_tile_objects(tiles_seed, objects_seed):
    """Objects on whole tiles must not silently bind the classifier (several objects would share one
    class) — construction fails and names the fix."""
    with pytest.raises(ValueError, match="must live on Crops"):
        Pipeline([_classifier()], tiles=tiles_seed, objects=objects_seed)


def test_validate_rejects_aggregator_missing_score(sources_seed):
    detector_no_score = FakeComponent(requires=Need(Tiles),
                                      produces=Need(Objects, links=("imagery",), crs=False, on=Tiles),
                                      name="detector")
    with pytest.raises(ValueError):
        Pipeline([_tilerizer(), detector_no_score, _aggregator()], sources=sources_seed)


# --- seed rules ----------------------------------------------------------------

def test_seed_image_type_follows_components():
    """A seeded folder takes the type of the first unmet imagery-role reference; Tiles by default."""
    assert Pipeline([_detector()])._seed_image_type() is Tiles
    assert Pipeline([_classifier()])._seed_image_type() is Crops
    assert Pipeline([_polygon_tilerizer(), _classifier()])._seed_image_type() is Tiles   # crops are produced
    assert Pipeline([])._seed_image_type() is Tiles


def test_derived_objects_for_classifier_only_run():
    """A classifier-only run over bare crops derives one Object per crop (1 image = 1 class), so the
    classifier's Objects-on-Crops requirement is met without user input."""
    crops = Crops.build(metadata=[make_tile_metadata(), make_tile_metadata(x0=64.0)])
    pipe = Pipeline([_classifier()], tiles=crops)
    derived = [s for s in pipe.seeds if isinstance(s, Objects)]
    assert len(derived) == 1 and len(derived[0]) == 2
    assert derived[0].on is Crops
