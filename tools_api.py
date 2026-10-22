"""Agent-facing tool layer (mm / GHz in, JSON-able dicts out). Shared by the MCP server and agent.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import rf_tools as rt

STATS = {"fast_calls": 0, "reference_calls": 0}

_SUB_SCHEMA = {
    "substrate": {"type": "string", "description": f"Preset name: {', '.join(rt.SUBSTRATES)}. "
                  "Omit to use custom er/h_mm/tand."},
    "er": {"type": "number"}, "h_mm": {"type": "number"}, "tand": {"type": "number"},
}
_GEOM = {"w_mm": {"type": "number"}, "l_mm": {"type": "number"},
         "y0_mm": {"type": "number", "description": "Inset feed depth"},
         "f_target_ghz": {"type": "number"}}


def _sub(substrate: str | None = None, er: float | None = None, h_mm: float | None = None,
         tand: float | None = None) -> rt.Substrate:
    if substrate:
        if substrate not in rt.SUBSTRATES:
            raise ValueError(f"unknown substrate {substrate!r}; presets: {list(rt.SUBSTRATES)}")
        return rt.SUBSTRATES[substrate]
    if er is None or h_mm is None:
        raise ValueError("give a substrate preset or both er and h_mm")
    return rt.Substrate(er, h_mm, tand or 0.0)


def list_substrates() -> dict:
    return {k: {"er": v.er, "h_mm": v.h_mm, "tand": v.tand} for k, v in rt.SUBSTRATES.items()}


def design_patch(f_ghz: float, substrate: str | None = None, er: float | None = None,
                 h_mm: float | None = None, tand: float | None = None, z0: float = 50.0) -> dict:
    """Closed-form starting point from the fast model: W, L, inset depth, 50-ohm feed width."""
    sub = _sub(substrate, er, h_mm, tand)
    d = rt.open_loop_design(f_ghz * 1e9, sub, z0)
    STATS["fast_calls"] += 1
    return {"w_mm": round(d["w"] * 1e3, 4), "l_mm": round(d["l"] * 1e3, 4),
            "y0_mm": round(d["y0"] * 1e3, 4),
            "feed_width_mm": round(rt.microstrip_width(z0, sub.h_mm, sub.er), 4),
            "note": "Starting point from the fast transmission-line model. Verify before trusting."}


def simulate_fast(w_mm: float, l_mm: float, y0_mm: float, f_target_ghz: float | None = None,
                  substrate: str | None = None, er: float | None = None, h_mm: float | None = None,
                  tand: float | None = None) -> dict:
    """Cheap surrogate (~30 us). Known biases: static eeff, no mutual coupling, no conductor loss."""
    STATS["fast_calls"] += 1
    res = rt.fast_patch(w_mm * 1e-3, l_mm * 1e-3, y0_mm * 1e-3, _sub(substrate, er, h_mm, tand))
    return res.summary(f_target_ghz * 1e9 if f_target_ghz else None)


def verify_reference(w_mm: float, l_mm: float, y0_mm: float, f_target_ghz: float | None = None,
                     substrate: str | None = None, er: float | None = None,
                     h_mm: float | None = None, tand: float | None = None) -> dict:
    """Higher-fidelity reference (~30x slower): dispersion, mutual conductance, dielectric and
    conductor loss. Treat as ground truth and as the expensive call to ration."""
    STATS["reference_calls"] += 1
    res = rt.reference_patch(w_mm * 1e-3, l_mm * 1e-3, y0_mm * 1e-3, _sub(substrate, er, h_mm, tand))
    out = res.summary(f_target_ghz * 1e9 if f_target_ghz else None)
    if f_target_ghz:
        l = l_mm * 1e-3
        out["suggested_inset_mm_for_50ohm"] = (
            round(rt.inset_for(res.r_edge, l) * 1e3, 4) if rt.inset_for(res.r_edge, l) else None)
        out["resonance_error_pct"] = round(100 * (res.fr / 1e9 - f_target_ghz) / f_target_ghz, 3)
    return out


def microstrip(h_mm: float, er: float, z0: float | None = None, w_mm: float | None = None) -> dict:
    """Give z0 to synthesise a width, or w_mm to analyse a line."""
    if z0 is not None:
        w = rt.microstrip_width(z0, h_mm, er)
        return {"w_mm": round(w, 4), "z0_check": round(rt.microstrip_z0(w, h_mm, er), 3),
                "eeff": round(rt.microstrip_eeff(w, h_mm, er), 4)}
    if w_mm is None:
        raise ValueError("give z0 or w_mm")
    return {"z0": round(rt.microstrip_z0(w_mm, h_mm, er), 3),
            "eeff": round(rt.microstrip_eeff(w_mm, h_mm, er), 4)}


def match_l_network(r_load: float, x_load: float, f_ghz: float, z0: float = 50.0) -> dict:
    sols = rt.l_match(complex(r_load, x_load), z0, f_ghz * 1e9)
    for s in sols:
        s.pop("zin")
        s["gamma_after"] = float(f"{s['gamma_after']:.2e}")
        s["X_ohm"], s["B_S"] = round(s["X_ohm"], 4), float(f"{s['B_S']:.5g}")
        for part in ("series", "shunt"):
            s[part] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s[part].items()}
    return {"solutions": sols}


def analyze_touchstone(path: str, threshold_db: float = -10.0) -> dict:
    """Parse a .sNp file (real measured or simulated data) and report return/insertion loss."""
    import skrf as rf

    ntw = rf.Network(path)
    f = ntw.f
    s11 = ntw.s_db[:, 0, 0]
    band = f[s11 < threshold_db]
    out = {"name": ntw.name, "ports": ntw.nports, "points": len(f),
           "f_start_ghz": round(f[0] / 1e9, 5), "f_stop_ghz": round(f[-1] / 1e9, 5),
           "min_s11_db": round(float(s11.min()), 2),
           "f_min_s11_ghz": round(float(f[s11.argmin()]) / 1e9, 5),
           "band_below_threshold_ghz": ([round(band.min() / 1e9, 5), round(band.max() / 1e9, 5)]
                                        if band.size else None)}
    if ntw.nports >= 2:
        s21 = ntw.s_db[:, 1, 0]
        out["max_s21_db"] = round(float(s21.max()), 2)
        out["f_max_s21_ghz"] = round(float(f[s21.argmax()]) / 1e9, 5)
    return out


def plot_design(w_mm: float, l_mm: float, y0_mm: float, f_target_ghz: float, out_png: str,
                substrate: str | None = None, er: float | None = None, h_mm: float | None = None,
                tand: float | None = None) -> dict:
    """Overlay fast vs reference S11 and Smith chart on one sweep, save PNG + Touchstone."""
    from plots import plot_patch, save_touchstone

    sub = _sub(substrate, er, h_mm, tand)
    g = (w_mm * 1e-3, l_mm * 1e-3, y0_mm * 1e-3, sub)
    f0 = f_target_ghz * 1e9
    freqs = np.linspace(f0 * 0.85, f0 * 1.15, 1201)
    fast = rt.fast_patch(*g, freqs=freqs)
    ref = rt.reference_patch(*g, freqs=freqs)
    png = plot_patch({"fast model": fast, "reference": ref}, f0, out_png)
    s1p = save_touchstone(ref, Path(out_png).with_suffix(".s1p"))
    return {"png": png, "touchstone": s1p}


TOOLS = {
    "list_substrates": (list_substrates, "List substrate presets.", {}),
    "design_patch": (design_patch, design_patch.__doc__,
                     {"f_ghz": {"type": "number"}, "z0": {"type": "number"}, **_SUB_SCHEMA}),
    "simulate_fast": (simulate_fast, simulate_fast.__doc__, {**_GEOM, **_SUB_SCHEMA}),
    "verify_reference": (verify_reference, verify_reference.__doc__, {**_GEOM, **_SUB_SCHEMA}),
    "microstrip": (microstrip, microstrip.__doc__,
                   {"h_mm": {"type": "number"}, "er": {"type": "number"},
                    "z0": {"type": "number"}, "w_mm": {"type": "number"}}),
    "match_l_network": (match_l_network, "Lossless L-section match for load R+jX (both solutions).",
                        {"r_load": {"type": "number"}, "x_load": {"type": "number"},
                         "f_ghz": {"type": "number"}, "z0": {"type": "number"}}),
}

REQUIRED = {
    "design_patch": ["f_ghz"],
    "simulate_fast": ["w_mm", "l_mm", "y0_mm"],
    "verify_reference": ["w_mm", "l_mm", "y0_mm"],
    "microstrip": ["h_mm", "er"],
    "match_l_network": ["r_load", "x_load", "f_ghz"],
}


def anthropic_tools() -> list[dict]:
    return [{"name": n, "description": (d or "").strip(),
             "input_schema": {"type": "object", "properties": p, "required": REQUIRED.get(n, [])}}
            for n, (_, d, p) in TOOLS.items()]


def call(name: str, args: dict) -> dict:
    try:
        return TOOLS[name][0](**args)
    except Exception as e:  # surfaced to the model as a tool error
        return {"error": f"{type(e).__name__}: {e}"}
