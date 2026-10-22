import math

import numpy as np
import pytest

from rf_tools import (SUBSTRATES, Substrate, closed_loop_design, fast_patch, l_match,
                      microstrip_width, microstrip_z0, patch_dimensions, reference_patch)


def test_balanis_patch_example():
    # Balanis, Antenna Theory, Ex. 14.1: 10 GHz, er=2.2, h=1.588 mm -> W=11.86 mm, L=9.06 mm
    w, l = patch_dimensions(10e9, Substrate(2.2, 1.588))
    assert w * 1e3 == pytest.approx(11.86, abs=0.05)
    assert l * 1e3 == pytest.approx(9.06, abs=0.05)


def test_fr4_2g4_patch():
    w, l = patch_dimensions(2.4e9, SUBSTRATES["FR4"])
    assert w * 1e3 == pytest.approx(38.0, abs=0.3)
    assert l * 1e3 == pytest.approx(29.4, abs=0.4)


def test_fast_model_resonates_at_design_frequency():
    sub = SUBSTRATES["FR4"]
    w, l = patch_dimensions(2.4e9, sub)
    assert fast_patch(w, l, 0, sub).fr == pytest.approx(2.4e9, rel=1e-9)


def test_pozar_microstrip_50ohm():
    # Pozar Ex. 3.7: er=2.2, d=0.159 cm, 50 ohm -> W=0.49 cm
    w = microstrip_width(50, 1.59, 2.2)
    assert w == pytest.approx(4.9, abs=0.05)
    assert microstrip_z0(w, 1.59, 2.2) == pytest.approx(50, abs=0.5)


def test_fr4_50ohm_line_is_about_3mm():
    assert microstrip_width(50, 1.6, 4.4) == pytest.approx(3.06, abs=0.1)


@pytest.mark.parametrize("zl", [200 - 100j, 20 + 30j, 75 + 0j, 10 - 40j])
def test_l_match_solutions_match(zl):
    for s in l_match(zl, 50 if zl.real != 200 else 100, 500e6):
        assert s["gamma_after"] < 1e-9


def test_pozar_l_match_values():
    # Pozar Ex. 5.1: ZL=200-j100, Z0=100, 500 MHz -> C=0.92 pF, L=38.8 nH (sol 1)
    s1 = l_match(200 - 100j, 100, 500e6)[0]
    assert s1["shunt"]["value_pF"] == pytest.approx(0.92, abs=0.02)
    assert s1["series"]["value_nH"] == pytest.approx(38.8, abs=0.3)


def test_reference_differs_but_close_for_thin_substrate():
    sub = SUBSTRATES["RT5880"]
    w, l = patch_dimensions(5.8e9, sub)
    fast, ref = fast_patch(w, l, 0, sub), reference_patch(w, l, 0, sub)
    assert abs(ref.fr - fast.fr) / fast.fr < 0.02


@pytest.mark.parametrize("name", ["FR4", "RO4003C", "RT5880", "Alumina"])
def test_closed_loop_converges(name):
    sub = SUBSTRATES[name]
    d = closed_loop_design(2.4e9, sub)
    ref = reference_patch(d["w"], d["l"], d["y0"], sub)
    assert abs(ref.fr - 2.4e9) / 2.4e9 < 1e-3
    assert ref.s11_db_at(2.4e9) < -20
