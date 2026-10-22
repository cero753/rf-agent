"""Monte Carlo manufacturing yield for closed-loop designs: a design that works on paper vs on a panel.

Tolerances (2-sigma): etch +/-50 um per edge, er +/-0.1, thickness +/-5%. Pass = S11 < -10 dB at target.
  python evals/yield_sweep.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rf_tools as rt

CASES = [(2.4, "FR4"), (5.8, "FR4"), (2.4, "RO4003C"), (5.8, "RT5880"), (2.4, "RT5880_thick"),
         (2.4, "Alumina"), (10.0, "Alumina")]
N = 500


def yield_for(f_ghz: float, name: str, rng: np.random.Generator) -> dict:
    sub = rt.SUBSTRATES[name]
    f0 = f_ghz * 1e9
    d = rt.closed_loop_design(f0, sub)
    ok, s11s = 0, []
    for _ in range(N):
        etch = rng.normal(0, 25e-6)
        b = replace(sub, er=rng.normal(sub.er, 0.05), h_mm=sub.h_mm * rng.normal(1, 0.025))
        r = rt.reference_patch(d["w"] - 2 * etch, d["l"] - 2 * etch, d["y0"], b)
        s = r.s11_db_at(f0)
        s11s.append(s)
        ok += s < -10
    nominal = rt.reference_patch(d["w"], d["l"], d["y0"], sub)
    s_db = 20 * np.log10(np.abs(nominal.s11) + 1e-15)
    band = nominal.freqs[s_db < -10]
    return {"f_ghz": f_ghz, "substrate": name, "yield": ok / N, "q": nominal.q,
            "bw_pct": float(100 * (band.max() - band.min()) / f0), "s11_p50": float(np.median(s11s))}


def main() -> None:
    rng = np.random.default_rng(3)
    rows = [yield_for(f, s, rng) for f, s in CASES]
    for r in rows:
        print(f"{r['substrate']:13s} {r['f_ghz']:5.1f} GHz  yield {100 * r['yield']:5.1f}%  "
              f"Q {r['q']:6.1f}  -10dB BW {r['bw_pct']:4.1f}%  median S11 {r['s11_p50']:6.1f} dB")
    (ROOT / "evals" / "yield.json").write_text(json.dumps(rows, indent=1))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for r in rows:
        ax.scatter(r["bw_pct"], 100 * r["yield"], s=40)
        ax.annotate(f"{r['substrate']} {r['f_ghz']:g}G", (r["bw_pct"], 100 * r["yield"]),
                    fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(0, max(r["bw_pct"] for r in rows) * 1.3)
    ax.set(xlabel="Nominal -10 dB bandwidth (%)", ylabel="Yield (%)",
           title="Perfect on paper, not on the panel: yield vs bandwidth (etch, er, h tolerances)")
    ax.set_ylim(-3, 103)
    fig.tight_layout()
    fig.savefig(ROOT / "evals" / "yield.png", dpi=130)
    print("wrote evals/yield.json, evals/yield.png")


if __name__ == "__main__":
    main()
