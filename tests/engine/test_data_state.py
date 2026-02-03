"""Tests for DataState class."""
from dataclasses import fields

from canopyrs.engine.constants import StateKey
from canopyrs.engine.data_state import DataState


def test_state_keys_match_datastate_fields():
    """Validate that StateKey constants match DataState field names."""
    field_names = {f.name for f in fields(DataState)}
    state_keys = {v for k, v in vars(StateKey).items() if not k.startswith('_')}
    assert state_keys.issubset(field_names), f"StateKey mismatch: {state_keys - field_names}"
