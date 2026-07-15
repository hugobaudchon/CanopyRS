"""Unit tests for the data contracts (Need / one_of / Schema) and Pipeline.validate wiring."""

import pytest

from canopyrs.engine.contracts import Need, one_of, Schema
from canopyrs.engine.data import Sources, Tiles, Objects
from canopyrs.engine.constants import Col
from canopyrs.engine.pipeline import Pipeline


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


# --- Need.check against a live table -----------------------------------------

def test_need_columns(objects_seed):
    assert Need(Objects, columns=(Col.DETECTOR_SCORE,)).check(objects_seed) == ""
    assert "must expose column" in Need(Objects, columns=("nope",)).check(objects_seed)


def test_need_links(objects_seed):
    assert Need(Objects, links=("tiles",)).check(objects_seed) == ""
    assert "must be linked" in Need(Objects, links=("prev_objects",)).check(objects_seed)


def test_need_crs(objects_seed):
    # detector boxes are in tile-pixel coords (crs=None)
    assert Need(Objects, crs=False).check(objects_seed) == ""
    assert "CRS coords" in Need(Objects, crs=True).check(objects_seed)


# --- Schema descriptor -------------------------------------------------------

def test_schema_descriptor():
    schema = Schema(columns={Col.DETECTOR_SCORE}, links={"tiles"}, crs=False)
    assert schema.provides(Col.DETECTOR_SCORE) and not schema.provides("x")
    assert schema.has_link("tiles") and not schema.has_link("prev_objects")
    assert schema.crs_set is False


# --- one_of ------------------------------------------------------------------

def test_one_of_picks_first_available(tiles_seed):
    # tile_path column is present but all-null; the sources-link alternative wins
    req = one_of(Need(Tiles, columns=(Col.TILE_PATH,)), Need(Tiles, links=("sources",)))
    desc, err = req.resolve(lambda t: {Tiles: tiles_seed}.get(t))
    assert desc is tiles_seed and err == ""


def test_one_of_none_available():
    req = one_of(Need(Tiles, columns=(Col.TILE_PATH,)), Need(Tiles, links=("sources",)))
    desc, err = req.resolve(lambda t: None)
    assert desc is None and "OR" in err


# --- Pipeline.validate (static wiring check) ---------------------------------

def test_validate_rejects_detector_before_tilerizer(sources_seed):
    detector = FakeComponent(requires=Need(Tiles, links=("sources",)),
                             produces=Need(Objects), name="detector")
    with pytest.raises(ValueError):
        Pipeline([detector], sources=sources_seed)


def test_validate_accepts_wired_pipeline(sources_seed):
    tilerizer = FakeComponent(requires=Sources,
                              produces=Need(Tiles, links=("sources",)), name="tilerizer")
    detector = FakeComponent(
        requires=Need(Tiles, links=("sources",)),
        produces=Need(Objects, columns=(Col.DETECTOR_SCORE,), links=("tiles",), crs=False),
        name="detector",
    )
    pipe = Pipeline([tilerizer, detector], sources=sources_seed)
    assert [c.component_id for c in pipe.components] == [0, 1]


def test_validate_rejects_aggregator_missing_score(sources_seed):
    tilerizer = FakeComponent(requires=Sources,
                              produces=Need(Tiles, links=("sources",)), name="tilerizer")
    detector = FakeComponent(requires=Need(Tiles, links=("sources",)),
                             produces=Need(Objects, links=("tiles",), crs=False), name="detector")
    # aggregator needs a weighted score column that no upstream component produces
    aggregator = FakeComponent(
        requires=Need(Objects, links=("tiles",), columns=(Col.DETECTOR_SCORE,), crs=False),
        produces=Need(Objects, columns=(Col.AGGREGATOR_SCORE,), links=("prev_objects",), crs=True),
        name="aggregator",
    )
    with pytest.raises(ValueError):
        Pipeline([tilerizer, detector, aggregator], sources=sources_seed)
