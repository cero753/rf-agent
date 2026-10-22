import json

import numpy as np
import pytest

import rf_tools as rt
import surrogate
import tools_api as t
from diagnose import CASES, DATA, diagnose, make_demo


@pytest.fixture(scope="module")
def sur():
    return surrogate.load()


def test_surrogate_accurate_in_distribution(sur):
    rng = np.random.default_rng(123)
    X, Y, _ = surrogate.sample(200, rng)
    err = np.abs(sur._raw(X).mean(0)[:, 0] - Y[:, 0])
    assert err.mean() < 0.1          # % resonance error vs reference
    assert np.abs(Y[:, 0]).mean() > 0.5  # and the fast model alone is much worse


def test_surrogate_flags_out_of_distribution(sur):
    for f, name in ((10e9, "Alumina"), (5.8e9, "FR4_thick")):
        w, l = rt.patch_dimensions(f, rt.SUBSTRATES[name])
        assert sur.predict(w, l, 0.0, rt.SUBSTRATES[name])["escalate"]


def test_surrogate_loop_passes_with_fewer_reference_calls():
    sub = rt.SUBSTRATES["RO4003C"]
    d = surrogate.surrogate_loop_design(5.8e9, sub)
    ref = rt.reference_patch(d["w"], d["l"], d["y0"], sub)
    assert ref.s11_db_at(5.8e9) < -20 and not d["escalated"]
    assert d["ref_calls"] < rt.closed_loop_design(5.8e9, sub)["ref_calls"]


def test_reference_budget_enforced(monkeypatch):
    monkeypatch.setenv("RF_REF_BUDGET", "1")
    monkeypatch.setitem(t.STATS, "reference_calls", 0)
    args = dict(w_mm=38.0, l_mm=29.0, y0_mm=9.0, f_target_ghz=2.4, substrate="FR4")
    assert "error" not in t.verify_reference(**args)
    assert "budget" in t.verify_reference(**args)["error"]


@pytest.fixture(scope="module")
def demo_data():
    make_demo()
    return json.loads((DATA / "design_fr4.json").read_text())


def test_diagnose_recovers_lot_permittivity(demo_data):
    out = diagnose(str(DATA / "measured_fr4_lot.s1p"), 2.4, "FR4", demo_data)
    assert out["fit"]["er"] == pytest.approx(CASES["fr4_lot"]["er"], abs=0.05)
    assert not out["ambiguous"]
    assert out["fix"]["predicted_s11_at_target_db"] < -20


def test_diagnose_admits_ambiguity_for_pure_etch(demo_data):
    out = diagnose(str(DATA / "measured_fr4_overetch.s1p"), 2.4, "FR4", demo_data)
    assert out["ambiguous"]
    assert "cannot be separated" in out["diagnosis"]
