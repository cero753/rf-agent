"""Design agent: fast model inside the loop, reference model to verify, budgeted.

  python agent.py --f 2.4 --substrate FR4               # Claude if ANTHROPIC_API_KEY is set
  python agent.py --f 5.8 --substrate RT5880 --planner scripted
"""
from __future__ import annotations

import argparse
import json
import os
import re

import rf_tools as rt
import tools_api as t

MODEL = os.environ.get("RF_AGENT_MODEL", "claude-sonnet-5")

SYSTEM = """You are an RF design agent. Goal: a rectangular inset-fed microstrip patch antenna
resonating at the target frequency, matched to 50 ohm.
Tools: simulate_fast is cheap but biased (static eeff, no mutual coupling, no conductor loss).
verify_reference is the trusted, expensive check. You have a budget of {budget} reference calls.
Done means verify_reference shows |resonance_error_pct| < 0.1 and s11_at_target_db < -15.
Correct the patch length using the reference resonance (resonance scales ~1/(L + 2*dL)), and the
inset using suggested_inset_mm_for_50ohm. When done, reply with only a JSON object:
{{"w_mm":..,"l_mm":..,"y0_mm":..,"verified":true|false,"notes":"one sentence"}}"""


def run_claude(f_ghz: float, substrate: str, budget: int = 6, max_turns: int = 16,
               verbose: bool = True) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": f"Target {f_ghz} GHz on substrate preset {substrate}."}]
    start = dict(t.STATS)
    for _ in range(max_turns):
        resp = client.messages.create(model=MODEL, max_tokens=2000, tools=t.anthropic_tools(),
                                      system=SYSTEM.format(budget=budget), messages=msgs)
        msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            m = re.search(r"\{.*\}", text, re.S)
            result = json.loads(m.group(0)) if m else {"verified": False, "notes": text[:200]}
            break
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            if b.name == "verify_reference" and t.STATS["reference_calls"] - start["reference_calls"] >= budget:
                out = {"error": "reference budget exhausted"}
            else:
                out = t.call(b.name, b.input)
            if verbose:
                print(f"  -> {b.name}({json.dumps(b.input)}) = {json.dumps(out)[:160]}")
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
        msgs.append({"role": "user", "content": results})
    else:
        result = {"verified": False, "notes": "turn limit"}
    result["fast_calls"] = t.STATS["fast_calls"] - start["fast_calls"]
    result["ref_calls"] = t.STATS["reference_calls"] - start["reference_calls"]
    return result


def run_scripted(f_ghz: float, substrate: str) -> dict:
    d = rt.closed_loop_design(f_ghz * 1e9, rt.SUBSTRATES[substrate])
    return {"w_mm": d["w"] * 1e3, "l_mm": d["l"] * 1e3, "y0_mm": d["y0"] * 1e3,
            "fast_calls": d["fast_calls"], "ref_calls": d["ref_calls"], "trace": d["trace"]}


def score(design: dict, f_ghz: float, substrate: str) -> dict:
    """Independent grading: re-run the reference regardless of what the agent claims."""
    sub = rt.SUBSTRATES[substrate]
    ref = rt.reference_patch(design["w_mm"] * 1e-3, design["l_mm"] * 1e-3,
                             design["y0_mm"] * 1e-3, sub)
    s11 = ref.s11_db_at(f_ghz * 1e9)
    err = 100 * (ref.fr / 1e9 - f_ghz) / f_ghz
    return {"s11_at_target_db": round(s11, 2), "resonance_error_pct": round(err, 3),
            "pass": s11 < -10 and abs(err) < 0.5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=2.4, help="target GHz")
    ap.add_argument("--substrate", default="FR4", choices=list(rt.SUBSTRATES))
    ap.add_argument("--planner", choices=["auto", "claude", "scripted"], default="auto")
    ap.add_argument("--plot", default="out/design.png")
    a = ap.parse_args()

    use_claude = a.planner == "claude" or (a.planner == "auto" and os.environ.get("ANTHROPIC_API_KEY"))
    print(f"planner: {'claude (' + MODEL + ')' if use_claude else 'scripted closed loop'}")
    design = run_claude(a.f, a.substrate) if use_claude else run_scripted(a.f, a.substrate)
    for row in design.get("trace", []):
        print(f"  iter {row['iter']}: L={row['l_mm']:.3f} mm y0={row['y0_mm']:.3f} mm "
              f"-> ref fr={row['fr_ghz']:.4f} GHz Rin={row['r_in']:.1f} ohm")
    open_d = rt.open_loop_design(a.f * 1e9, rt.SUBSTRATES[a.substrate])
    print("open loop (trust fast model):",
          score({"w_mm": open_d["w"] * 1e3, "l_mm": open_d["l"] * 1e3, "y0_mm": open_d["y0"] * 1e3},
                a.f, a.substrate))
    print("closed loop:", {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in design.items() if k != "trace"})
    print("graded:", score(design, a.f, a.substrate))
    if a.plot:
        print(t.plot_design(design["w_mm"], design["l_mm"], design["y0_mm"], a.f, a.plot,
                            substrate=a.substrate))


if __name__ == "__main__":
    main()
