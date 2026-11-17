"""retrain_if_better.py must be able to record its decision (it crashed on NumPy booleans before)."""

import json

import numpy as np
import pytest

from scripts.retrain_if_better import _json_value


def test_a_decision_with_numpy_values_can_be_written():
    decision = {
        "checks": {"no_worse_than_shorter_history": np.float64(17.1) <= np.float64(17.4), "beats_holding_price_overall": True},
        "overall": {"gain_pct": np.float64(4.3), "rows": np.int64(12)},
    }
    written = json.loads(json.dumps(decision, default=_json_value))
    assert written == {"checks": {"no_worse_than_shorter_history": True, "beats_holding_price_overall": True}, "overall": {"gain_pct": 4.3, "rows": 12}}


def test_other_objects_still_raise():
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=_json_value)
