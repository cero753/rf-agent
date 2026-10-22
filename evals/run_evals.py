"""Eval: how far does trusting the fast model get you, and what does verification buy?

  python evals/run_evals.py                 # open loop vs scripted closed loop
  python evals/run_evals.py --claude        # also the Claude agent (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rf_tools as rt
from agent import run_claude, run_scripted, score
from surrogate import surrogate_loop_design


def drift(f_ghz: float, sub: str) -> dict:
    s = rt.SUBSTRATES[sub]
    d = rt.open_loop_design(f_ghz * 1e9, s)
    g = (d["w"], d["l"], d["y0"], s)
    fast, ref = rt.fast_patch(*g), rt.reference_patch(*g)
    return {"fr_drift_pct": 100 * (ref.fr - fast.fr) / fast.fr,
            "rin_drift_ohm": ref.r_in - fast.r_in,
            "speedup": rt.timed(rt.reference_patch, *g, repeat=10) / rt.timed(rt.fast_patch, *g, repeat=200)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", action="store_true")
    a = ap.parse_args()

    targets = json.loads((ROOT / "evals" / "targets.json").read_text())
    rows = []
    for tg in targets:
        f, sub = tg["f_ghz"], tg["substrate"]
        od = rt.open_loop_design(f * 1e9, rt.SUBSTRATES[sub])
        open_score = score({"w_mm": od["w"] * 1e3, "l_mm": od["l"] * 1e3, "y0_mm": od["y0"] * 1e3}, f, sub)
        cd = run_scripted(f, sub)
        sd = surrogate_loop_design(f * 1e9, rt.SUBSTRATES[sub])
        sd_mm = {"w_mm": sd["w"] * 1e3, "l_mm": sd["l"] * 1e3, "y0_mm": sd["y0"] * 1e3}
        row = {"id": tg["id"], "f_ghz": f, "substrate": sub, **drift(f, sub),
               "open": open_score, "closed": score(cd, f, sub), "closed_ref_calls": cd["ref_calls"],
               "surrogate": score(sd_mm, f, sub), "surrogate_ref_calls": sd["ref_calls"],
               "escalated": sd["escalated"], "in_distribution": sd["in_distribution"]}
        if a.claude:
            cl = run_claude(f, sub, verbose=False)
            row["claude"] = score(cl, f, sub) if "w_mm" in cl else {"pass": False}
            row["claude_ref_calls"] = cl["ref_calls"]
            row["claude_claimed_verified"] = cl.get("verified")
        rows.append(row)
        print(f"{tg['id']:>3} {sub:13s} {f:5.2f} GHz drift {row['fr_drift_pct']:+6.2f}%  "
              f"open {open_score['s11_at_target_db']:6.1f} dB {'PASS' if open_score['pass'] else 'fail'}  "
              f"closed {row['closed']['s11_at_target_db']:6.1f} dB {'PASS' if row['closed']['pass'] else 'fail'} "
              f"({cd['ref_calls']} ref)  surrogate {'PASS' if row['surrogate']['pass'] else 'fail'} "
              f"({sd['ref_calls']} ref, {'ESCALATED' if sd['escalated'] else 'trusted'}, "
              f"{'ID' if sd['in_distribution'] else 'OOD'})"
              + (f"  claude {'PASS' if row['claude']['pass'] else 'fail'}" if a.claude else ""))

    n = len(rows)
    arms = ["open", "closed", "surrogate"] + (["claude"] if a.claude else [])
    summary = {f"pass_{arm}": sum(r[arm]["pass"] for r in rows) / n for arm in arms}
    summary["mean_abs_fr_drift_pct"] = sum(abs(r["fr_drift_pct"]) for r in rows) / n
    summary["mean_closed_ref_calls"] = sum(r["closed_ref_calls"] for r in rows) / n
    summary["mean_surrogate_ref_calls"] = sum(r["surrogate_ref_calls"] for r in rows) / n
    ood = [r for r in rows if not r["in_distribution"]]
    summary["ood_targets"] = len(ood)
    summary["ood_escalated"] = sum(r["escalated"] for r in ood)
    summary["id_escalated"] = sum(r["escalated"] for r in rows if r["in_distribution"])
    summary["median_speedup_x"] = sorted(r["speedup"] for r in rows)[n // 2]
    if a.claude:
        summary["claude_overclaims"] = sum(1 for r in rows
                                           if r.get("claude_claimed_verified") and not r["claude"]["pass"])
    print("\nsummary:", json.dumps({k: round(v, 3) for k, v in summary.items()}, indent=1))

    out = ROOT / "evals" / "results.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for sub in dict.fromkeys(r["substrate"] for r in rows):
        rs = sorted((r for r in rows if r["substrate"] == sub), key=lambda r: r["f_ghz"])
        ax.plot([r["f_ghz"] for r in rs], [r["fr_drift_pct"] for r in rs], "o-", label=sub)
    ax.axhline(0, c="k", lw=0.6)
    ax.set(xlabel="Target frequency (GHz)", ylabel="Reference vs fast resonance (%)",
           title="Where the fast model drifts (out-of-distribution: thick, high-er, high-f)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "evals" / "drift.png", dpi=130)
    print("wrote evals/results.json, evals/drift.png")


if __name__ == "__main__":
    main()
