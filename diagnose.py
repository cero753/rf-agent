"""Diagnose an as-built patch from a measured S11 (.s1p): which physical deviation explains the miss,
how confident we are, and what to change.

Fits substrate permittivity, over-etch and loss tangent to the complex S11 with the reference model,
profiles the er/etch trade-off to report when the data cannot tell them apart, and proposes a fix.

  python diagnose.py --make-demo                    # writes data/measured_*.s1p with hidden truth
  python diagnose.py data/measured_fr4_lot.s1p --f 2.4 --substrate FR4 --design data/design_fr4.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import rf_tools as rt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def as_built(w: float, l: float, y0: float, sub: rt.Substrate, er: float, etch_um: float, tand: float,
             freqs: np.ndarray) -> np.ndarray:
    """Over-etch removes etch_um from every copper edge (W and L shrink by 2x)."""
    e = etch_um * 1e-6
    b = replace(sub, er=er, tand=tand)
    return rt.reference_patch(w - 2 * e, l - 2 * e, y0, b, freqs=freqs).s11


def _fit(freqs, s11, w, l, y0, sub, etch_fixed: float | None = None):
    def resid(p):
        er, etch, tand = (p[0], etch_fixed, p[1]) if etch_fixed is not None else p
        d = as_built(w, l, y0, sub, er, etch, tand, freqs) - s11
        return np.concatenate([d.real, d.imag])

    if etch_fixed is None:
        x0, lo, hi = [sub.er, 0.0, sub.tand or 0.01], [sub.er * 0.85, -150, 1e-4], [sub.er * 1.15, 150, 0.05]
    else:
        x0, lo, hi = [sub.er, sub.tand or 0.01], [sub.er * 0.85, 1e-4], [sub.er * 1.15, 0.05]
    r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")
    return r, float(np.sqrt(np.mean(r.fun ** 2)))


def diagnose(s1p: str, f_ghz: float, substrate: str, design: dict) -> dict:
    import skrf as rf

    ntw = rf.Network(s1p)
    freqs, s11 = ntw.f, ntw.s[:, 0, 0]
    sub = rt.SUBSTRATES[substrate]
    w, l, y0 = design["w_mm"] * 1e-3, design["l_mm"] * 1e-3, design["y0_mm"] * 1e-3
    f0 = f_ghz * 1e9
    s_db = 20 * np.log10(np.abs(s11))
    f_meas = float(freqs[s_db.argmin()])
    nominal = rt.reference_patch(w, l, y0, sub, freqs=freqs)

    full, rms = _fit(freqs, s11, w, l, y0, sub)
    er_fit, etch_fit, tand_fit = full.x
    noise = float(np.sqrt(np.mean(np.abs(np.diff(s11, 2)) ** 2) / 6))

    # Profile: force each explanation alone and compare fit quality.
    only_er, rms_er = _fit(freqs, s11, w, l, y0, sub, etch_fixed=0.0)
    etch_grid = np.linspace(-150, 150, 61)
    rms_etch_only = min(
        (np.sqrt(np.mean(np.abs(as_built(w, l, y0, sub, sub.er, e, sub.tand, freqs) - s11) ** 2)), e)
        for e in etch_grid)
    ambiguous = rms_etch_only[0] < 2 * rms_er

    sub_fit = replace(sub, er=er_fit, tand=tand_fit)
    l_fix = rt.correct_length(w, l - 2 * etch_fit * 1e-6, f_meas, f0, sub_fit) + 2 * etch_fit * 1e-6
    ref_fix = rt.reference_patch(w - 2 * etch_fit * 1e-6, l_fix - 2 * etch_fit * 1e-6, y0, sub_fit)
    y0_fix = (rt.inset_for(ref_fix.r_edge, l_fix) or 0.0)
    check = rt.reference_patch(w - 2 * etch_fit * 1e-6, l_fix - 2 * etch_fit * 1e-6, y0_fix, sub_fit)

    shift = 100 * (f_meas - f0) / f0
    if abs(shift) < 0.3:
        story = "Resonance is on target; no correction needed."
    elif ambiguous:
        story = (f"Resonance is {shift:+.2f}% off. The data fits a permittivity change "
                 f"(er~{only_er.x[0]:.2f}) and an etch error ({rms_etch_only[1]:+.0f} um) about equally well, "
                 f"so they cannot be separated from this antenna alone (joint best guess: er={er_fit:.3f}, "
                 f"etch={etch_fit:+.0f} um). Measure a reference line or ring resonator on the same panel "
                 "to split them. The length fix below holds either way for this panel.")
    else:
        causes = [c for c, hit in (("substrate permittivity (laminate lot variation)", abs(er_fit - sub.er) / sub.er > 0.01),
                                   ("over-etch" if etch_fit > 0 else "under-etch", abs(etch_fit) > 25)) if hit]
        cause = " plus ".join(causes) or "a small mix of tolerances"
        story = (f"Resonance is {shift:+.2f}% off. Best explained by {cause}: fitted er={er_fit:.3f} "
                 f"(nominal {sub.er}), etch={etch_fit:+.0f} um, tand={tand_fit:.4f}. "
                 f"An etch-only explanation fits {rms_etch_only[0] / rms_er:.1f}x worse.")
    return {
        "measured_resonance_ghz": round(f_meas / 1e9, 5),
        "design_resonance_ghz": round(nominal.fr / 1e9, 5),
        "shift_pct": round(shift, 3),
        "measured_min_s11_db": round(float(s_db.min()), 2),
        "s11_at_target_db": round(float(np.interp(f0, freqs, s_db)), 2),
        "fit": {"er": round(er_fit, 4), "etch_um": round(etch_fit, 1), "tand": round(tand_fit, 5),
                "rms_residual": round(rms, 5), "noise_floor_est": round(noise, 5)},
        "etch_only_rms": round(float(rms_etch_only[0]), 5), "er_only_rms": round(rms_er, 5),
        "ambiguous": bool(ambiguous),
        "diagnosis": story,
        "fix": {"l_mm": round(l_fix * 1e3, 4), "y0_mm": round(y0_fix * 1e3, 4),
                "delta_l_mm": round((l_fix - l) * 1e3, 4),
                "predicted_s11_at_target_db": round(check.s11_db_at(f0), 2),
                "predicted_resonance_ghz": round(check.fr / 1e9, 5)},
    }


CASES = {
    "fr4_lot": {"er": 4.15, "etch_um": 0.0, "tand": 0.02},
    "fr4_overetch": {"er": 4.4, "etch_um": 120.0, "tand": 0.02},
    "fr4_both": {"er": 4.25, "etch_um": 60.0, "tand": 0.025},
}


def make_demo(seed: int = 7) -> None:
    import skrf as rf

    rng = np.random.default_rng(seed)
    DATA.mkdir(exist_ok=True)
    sub = rt.SUBSTRATES["FR4"]
    d = rt.closed_loop_design(2.4e9, sub)
    design = {"w_mm": d["w"] * 1e3, "l_mm": d["l"] * 1e3, "y0_mm": d["y0"] * 1e3, "f_ghz": 2.4, "substrate": "FR4"}
    (DATA / "design_fr4.json").write_text(json.dumps(design, indent=1))
    freqs = np.linspace(2.1e9, 2.7e9, 601)
    truth = {}
    for name, c in CASES.items():
        s = as_built(d["w"], d["l"], d["y0"], sub, c["er"], c["etch_um"], c["tand"], freqs)
        s = s + (rng.normal(0, 0.008, s.shape) + 1j * rng.normal(0, 0.008, s.shape))
        ntw = rf.Network(frequency=rf.Frequency.from_f(freqs, unit="hz"), s=s.reshape(-1, 1, 1),
                         name=f"measured_{name}")
        ntw.write_touchstone(str(DATA / f"measured_{name}"))
        truth[name] = c
    (DATA / "hidden_truth.json").write_text(json.dumps(truth, indent=1))
    print("wrote", sorted(p.name for p in DATA.iterdir()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("s1p", nargs="?")
    ap.add_argument("--f", type=float, default=2.4)
    ap.add_argument("--substrate", default="FR4")
    ap.add_argument("--design", default=str(DATA / "design_fr4.json"))
    ap.add_argument("--make-demo", action="store_true")
    a = ap.parse_args()
    if a.make_demo:
        make_demo()
        return
    out = diagnose(a.s1p, a.f, a.substrate, json.loads(Path(a.design).read_text()))
    print(out["diagnosis"])
    print(json.dumps({k: v for k, v in out.items() if k != "diagnosis"}, indent=1))


if __name__ == "__main__":
    main()
