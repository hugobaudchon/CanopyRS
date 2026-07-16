"""Unit tests for the data contracts (Need / one_of / Schema) and the pipeline's input matching
(kind selects, everything else validates) + static validation."""

import pytest

from canopyrs.engine.contracts import Need, one_of, Schema
from canopyrs.engine.data import Imagery, Objects
from canopyrs.engine.constants import Col, ImageKind, Modality
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
    return FakeComponent(requires=Need(Imagery, kind=ImageKind.SOURCE),
                         produces=Need(Imagery, kind=ImageKind.TILE, links=("parent",)),
                         name="tilerizer")


def _detector():
    return FakeComponent(
        requires=Need(Imagery, kind=ImageKind.TILE),
        produces=Need(Objects, columns=(Col.DETECTOR_SCORE,), links=("imagery",), crs=False),
        name="detector",
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


def test_need_kind(sources_seed, tiles_seed):
    assert Need(Imagery, kind=ImageKind.SOURCE).check(sources_seed.schema()) == ""
    assert Need(Imagery, kind=ImageKind.TILE).check(tiles_seed.schema()) == ""
    assert "kind='tile'" in Need(Imagery, kind=ImageKind.TILE).check(sources_seed.schema())


def test_need_modalities(sources_seed):
    # from_paths defaults every row to rgb
    assert Need(Imagery, modalities=(Modality.RGB,)).check(sources_seed.schema()) == ""
    assert "modality" in Need(Imagery, modalities=(Modality.POINTCLOUD,)).check(sources_seed.schema())


# --- Schema descriptor -------------------------------------------------------

def test_schema_descriptor():
    schema = Schema(columns={Col.DETECTOR_SCORE}, links={"imagery"}, crs=False,
                    kind=ImageKind.TILE, modalities={Modality.RGB})
    assert schema.provides(Col.DETECTOR_SCORE) and not schema.provides("x")
    assert schema.has_link("imagery") and not schema.has_link("prev_objects")
    assert schema.crs_set is False
    assert schema.kind == ImageKind.TILE
    assert Need(Imagery, kind=ImageKind.TILE, modalities=(Modality.RGB,)).check(schema) == ""


def test_schema_undeclared_attributes_skip_checks():
    # kind/modalities None on the schema = producer didn't declare -> checks pass (tri-state)
    schema = Schema(columns=(), links=())
    assert Need(Imagery, kind=ImageKind.TILE).check(schema) == ""
    assert Need(Imagery, modalities=(Modality.RGB,)).check(schema) == ""


# --- one_of ------------------------------------------------------------------

def test_one_of_picks_first_available(tiles_seed):
    req = one_of(Need(Imagery, kind=ImageKind.SOURCE), Need(Imagery, kind=ImageKind.TILE))
    desc, err = req.resolve(lambda t: {Imagery: [tiles_seed]}.get(t, ()))
    assert desc is tiles_seed and err == ""


def test_one_of_none_available():
    req = one_of(Need(Imagery, kind=ImageKind.SOURCE), Need(Imagery, kind=ImageKind.TILE))
    desc, err = req.resolve(lambda t: ())
    assert desc is None and "OR" in err


# --- input matching: kind selects, everything else validates ------------------

def test_matching_finds_source_past_newer_tiles(sources_seed, tiles_seed):
    """A kind='source' need binds the source seed even though tiles are newer (the tilerizer-after-
    tiles case)."""
    available = {Imagery: [sources_seed, tiles_seed]}   # oldest -> newest
    desc, err = Need(Imagery, kind=ImageKind.SOURCE).resolve(lambda t: available.get(t, ()))
    assert desc is sources_seed and err == ""


def test_matching_prefers_newest_of_kind(sources_seed, tiles_seed):
    available = {Imagery: [sources_seed, tiles_seed]}
    desc, _ = Need(Imagery, kind=ImageKind.TILE).resolve(lambda t: available.get(t, ()))
    assert desc is tiles_seed


def test_matching_error_reports_candidates(tiles_seed):
    desc, err = Need(Imagery, kind=ImageKind.SOURCE).resolve(lambda t: {Imagery: [tiles_seed]}.get(t, ()))
    assert desc is None and "none of the 1 available" in err


def test_matching_never_falls_back_past_a_broken_newest(tiles_seed):
    """Only kind selects among tables: when the newest table of the right kind fails the rest of the
    check, matching errors instead of silently binding an older table (config errors stay loud)."""
    unlinked_tiles = Imagery.build(kind=ImageKind.TILE, metadata=[make_tile_metadata()])
    available = {Imagery: [tiles_seed, unlinked_tiles]}   # oldest -> newest; only the old one has 'parent'
    desc, err = Need(Imagery, kind=ImageKind.TILE, links=("parent",)).resolve(
        lambda t: available.get(t, ()))
    assert desc is None and "must be linked" in err


# --- Pipeline.validate (static wiring check) ---------------------------------

def test_validate_rejects_detector_on_untiled_source(sources_seed):
    with pytest.raises(ValueError):
        Pipeline([_detector()], sources=sources_seed)


def test_validate_accepts_wired_pipeline(sources_seed):
    pipe = Pipeline([_tilerizer(), _detector()], sources=sources_seed)
    assert [c.component_id for c in pipe.components] == [0, 1]


def test_validate_crop_from_tiles_pipeline(sources_seed):
    """Benchmark type-1 wiring: grid tilerizer -> detector -> polygon tilerizer validates — the
    polygon tilerizer needs only the detections' imagery link, no separate source input."""
    polygon_tilerizer = FakeComponent(
        requires=Need(Objects, links=("imagery",)),
        produces=(Need(Imagery, kind=ImageKind.TILE, links=("parent",)),
                  Need(Objects, links=("imagery", "prev_objects"), crs=True)),
        name="polygon_tilerizer",
    )
    pipe = Pipeline([_tilerizer(), _detector(), polygon_tilerizer], sources=sources_seed)
    assert len(pipe.components) == 3


def test_validate_source_need_survives_produced_tiles(sources_seed):
    """Static mirror of input matching: a second tilerizer's kind='source' requirement stays
    satisfiable after the first tilerizer produced tiles (per-type schema lists, not overwrite)."""
    pipe = Pipeline([_tilerizer(), _tilerizer()], sources=sources_seed)
    assert len(pipe.components) == 2


def test_validate_rejects_aggregator_missing_score(sources_seed):
    detector_no_score = FakeComponent(requires=Need(Imagery, kind=ImageKind.TILE),
                                      produces=Need(Objects, links=("imagery",), crs=False),
                                      name="detector")
    aggregator = FakeComponent(
        requires=Need(Objects, links=("imagery",), columns=(Col.DETECTOR_SCORE,), crs=False),
        produces=Need(Objects, columns=(Col.AGGREGATOR_SCORE,), links=("prev_objects",), crs=True),
        name="aggregator",
    )
    with pytest.raises(ValueError):
        Pipeline([_tilerizer(), detector_no_score, aggregator], sources=sources_seed)
