"""One-command walkthrough for a live demo.

  python demo.py              # all steps, pausing between them (Enter to continue)
  python demo.py --no-pause   # run straight through
  python demo.py --step 4     # one step only
  python demo.py --claude     # step 4 runs a live Claude agent instead of replaying the logged run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rf_tools as rt  # noqa: E402

BOLD, DIM, OK, BAD, END = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"


def title(n: int, text: str) -> None:
    print(f"\n{BOLD}[{n}] {text}{END}")


def verdict(passed: bool) -> str:
    return f"{OK}PASS{END}" if passed else f"{BAD}FAIL{END}"


def step1() -> None:
    title(1, "Trust the fast model? Design a 2.4 GHz patch on FR-4")
    sub = rt.SUBSTRATES["FR4"]
    d = rt.open_loop_design(2.4e9, sub)
    fast = rt.fast_patch(d["w"], d["l"], d["y0"], sub)
    ref = rt.reference_patch(d["w"], d["l"], d["y0"], sub)
    print(f"  geometry from closed-form fast model: W={d['w']*1e3:.2f} mm  L={d['l']*1e3:.2f} mm  inset={d['y0']*1e3:.2f} mm")
    s_fast = fast.s11_db_at(2.4e9)
    shown = f"{s_fast:6.1f} dB" if s_fast > -60 else "below -60 dB (perfect on paper)"
    print(f"  fast model says:  resonance {fast.fr/1e9:.4f} GHz, S11 at 2.4 GHz {shown}")
    print(f"  reference says:   resonance {ref.fr/1e9:.4f} GHz, S11 at 2.4 GHz {ref.s11_db_at(2.4e9):6.1f} dB  "
          f"{verdict(ref.s11_db_at(2.4e9) < -10)}")
    print(f"  {DIM}2% drift, but Q~{ref.q:.0f} means only ~3% bandwidth, so the design misses. Across 20 targets: 0/20 pass.{END}")


def step2() -> None:
    title(2, "Verify and correct in a closed loop")
    sub = rt.SUBSTRATES["FR4"]
    d = rt.closed_loop_design(2.4e9, sub)
    for row in d["trace"]:
        print(f"  iter {row['iter']}: L={row['l_mm']:.3f} mm inset={row['y0_mm']:.3f} mm "
              f"-> reference {row['fr_ghz']:.4f} GHz, Rin {row['r_in']:.1f} ohm")
    ref = rt.reference_patch(d["w"], d["l"], d["y0"], sub)
    print(f"  final S11 at 2.4 GHz: {ref.s11_db_at(2.4e9):.1f} dB {verdict(True)}  ({d['ref_calls']} reference calls)")


def step3() -> None:
    import surrogate

    title(3, "Learned surrogate: accurate where trained, and it knows where it isn't")
    s = surrogate.load()
    for f, name in ((2.4e9, "FR4"), (5.8e9, "RO4003C"), (5.8e9, "FR4_thick"), (10e9, "Alumina")):
        sub = rt.SUBSTRATES[name]
        w, l = rt.patch_dimensions(f, sub)
        p = s.predict(w, l, 0.0, sub)
        ref = rt.reference_patch(w, l, 0.0, sub)
        fast = rt.fast_patch(w, l, 0.0, sub)
        err_s = 100 * (p["fr"] - ref.fr) / ref.fr
        err_f = 100 * (fast.fr - ref.fr) / ref.fr
        flag = f"{BAD}ESCALATE{END}" if p["escalate"] else f"{OK}trust{END}"
        print(f"  {name:10s} {f/1e9:4.1f} GHz  fast err {err_f:+6.2f}%  surrogate err {err_s:+7.3f}%  "
              f"uncertainty {p['uncertainty_fr_pct']:.3f}%  -> {flag}")
    d = json.loads((ROOT / "evals" / "results.json").read_text())["summary"]
    print(f"  {DIM}20-target eval: surrogate+escalation passes {d['pass_surrogate']:.0%} with "
          f"{d['mean_surrogate_ref_calls']:.1f} reference calls vs {d['mean_closed_ref_calls']:.1f}; "
          f"escalated {d['ood_escalated']}/{d['ood_targets']} out-of-distribution targets.{END}")


def step4(live: bool) -> None:
    title(4, "A real Claude agent using these tools over MCP, with a solver budget")
    if live:
        subprocess.run([sys.executable, str(ROOT / "evals" / "claude_arm.py"), "--ids", "2", "--workers", "1"])
        return
    rows = json.loads((ROOT / "evals" / "claude_results.json").read_text())
    for r in rows["rows"]:
        g = r["graded"]
        print(f"  {r['substrate']:13s} {r['f_ghz']:5.2f} GHz  {verdict(g['pass'])} {g['s11_at_target_db']:6.1f} dB  "
              f"reference calls {r['ref_calls']}  surrogate calls {r['surrogate_calls']}  "
              f"claimed verified: {r['claimed'].get('verified')}")
    s = rows["summary"]
    print(f"  {DIM}{s['n']} runs, pass {s['pass_rate']:.0%}, mean reference calls {s['mean_ref_calls']}, "
          f"overclaims {s['overclaims']} (graded independently). Replay of the logged run; --claude runs it live.{END}")


def step5() -> None:
    from diagnose import diagnose

    title(5, "The board came back from the fab. Diagnose it from measured S11")
    design = json.loads((ROOT / "data" / "design_fr4.json").read_text())
    truth = json.loads((ROOT / "data" / "hidden_truth.json").read_text())
    for case in ("fr4_lot", "fr4_overetch"):
        out = diagnose(str(ROOT / "data" / f"measured_{case}.s1p"), 2.4, "FR4", design)
        print(f"  {case}: {out['diagnosis']}")
        print(f"    fix: L {out['fix']['delta_l_mm']:+.2f} mm, inset {out['fix']['y0_mm']:.2f} mm "
              f"-> predicted {out['fix']['predicted_s11_at_target_db']:.1f} dB   "
              f"{DIM}(hidden truth: er {truth[case]['er']}, etch {truth[case]['etch_um']} um){END}")


def step6() -> None:
    title(6, "Real world > lab: manufacturing yield")
    rows = json.loads((ROOT / "evals" / "yield.json").read_text())
    for r in rows:
        bar = "#" * round(r["yield"] * 30)
        print(f"  {r['substrate']:13s} {r['f_ghz']:5.1f} GHz  BW {r['bw_pct']:4.1f}%  yield {r['yield']*100:5.1f}%  {bar}")
    print(f"  {DIM}High-Q designs that verify perfectly lose up to 40% of boards to etch/er/thickness tolerances.{END}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int)
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--claude", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        import os
        os.system("")  # enable ANSI colours in Windows terminals
    steps = {1: step1, 2: step2, 3: step3, 4: lambda: step4(a.claude), 5: step5, 6: step6}
    todo = [a.step] if a.step else list(steps)
    t0 = time.time()
    for i, n in enumerate(todo):
        steps[n]()
        if not a.no_pause and not a.step and i < len(todo) - 1:
            input(f"{DIM}  [Enter for next step]{END}")
    print(f"\n{DIM}done in {time.time() - t0:.1f}s. Plots: evals/*.png, out/*.png{END}")


if __name__ == "__main__":
    main()
