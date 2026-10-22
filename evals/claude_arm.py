"""Run a real Claude agent (Claude Code headless, `claude -p`) against the rf-agent MCP server.

Built-in tools are disabled; the agent can only call rf-agent MCP tools. The server enforces a
reference-call budget and logs every call. Each final design is graded independently.

  python evals/claude_arm.py                 # 8 targets, sonnet
  python evals/claude_arm.py --ids 2 17 --model haiku
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import SYSTEM, score  # noqa: E402

DEFAULT_IDS = [2, 8, 12, 10, 6, 14, 17, 19]
BUDGET = 6
BUILTINS = ["Bash", "PowerShell", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch",
            "Agent", "Skill", "NotebookEdit", "TodoWrite"]


def run_one(tg: dict, model: str, workdir: Path) -> dict:
    tid = tg["id"]
    stats = workdir / f"stats_{tid}.jsonl"
    stats.unlink(missing_ok=True)
    cfg = workdir / f"mcp_{tid}.json"
    cfg.write_text(json.dumps({"mcpServers": {"rf-agent": {
        "command": sys.executable, "args": [str(ROOT / "server.py")],
        "env": {"RF_REF_BUDGET": str(BUDGET), "RF_STATS_FILE": str(stats)}}}}))
    sysfile = workdir / "system.txt"
    sysfile.write_text(SYSTEM.format(budget=BUDGET), encoding="utf-8")
    prompt = (f"Target {tg['f_ghz']} GHz on substrate preset {tg['substrate']}. "
              f"Pass f_target_ghz={tg['f_ghz']} and substrate={tg['substrate']} to every tool call.")
    cmd = [shutil.which("claude.cmd") or shutil.which("claude") or "claude", "-p",
           "--model", model, "--mcp-config", str(cfg), "--strict-mcp-config",
           "--allowedTools", "mcp__rf-agent__*", "--disallowedTools", *BUILTINS,
           "--setting-sources", "project", "--system-prompt-file", str(sysfile),
           "--output-format", "json", "--no-session-persistence"]
    t0 = time.time()
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                          cwd=workdir, timeout=900)
    elapsed = time.time() - t0
    row = {"id": tid, "f_ghz": tg["f_ghz"], "substrate": tg["substrate"], "seconds": round(elapsed, 1)}
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        row.update(error=(proc.stderr or proc.stdout)[-400:], graded={"pass": False})
        return row
    text = out.get("result", "")
    m = re.findall(r"\{[^{}]*\"w_mm\"[^{}]*\}", text, re.S)
    calls = [json.loads(l) for l in stats.read_text(encoding="utf-8").splitlines()] if stats.exists() else []
    row.update(turns=out.get("num_turns"), cost_usd=out.get("total_cost_usd"),
               ref_calls=sum(c["tool"] == "verify_reference" and "error" not in c["out"] for c in calls),
               surrogate_calls=sum(c["tool"] == "predict_surrogate" for c in calls),
               budget_refusals=sum("error" in c["out"] for c in calls))
    if not m:
        row.update(claimed=None, graded={"pass": False}, final_text=text[-300:])
        return row
    design = json.loads(m[-1])
    row["claimed"] = design
    row["graded"] = score(design, tg["f_ghz"], tg["substrate"])
    row["overclaim"] = bool(design.get("verified")) and not row["graded"]["pass"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    targets = {t["id"]: t for t in json.loads((ROOT / "evals" / "targets.json").read_text())}
    work = Path(tempfile.mkdtemp(prefix="rfagent_claude_"))
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda i: run_one(targets[i], a.model, work), a.ids))
    for r in rows:
        g = r["graded"]
        print(f"{r['id']:>3} {r['substrate']:13s} {r['f_ghz']:5.2f} GHz  "
              f"{'PASS' if g['pass'] else 'fail'} {g.get('s11_at_target_db', '-')} dB "
              f"err {g.get('resonance_error_pct', '-')}%  ref {r.get('ref_calls', '-')}  "
              f"surr {r.get('surrogate_calls', '-')}  claimed_verified "
              f"{(r.get('claimed') or {}).get('verified')}  {r['seconds']}s"
              + (f"  ERROR {r['error'][:120]}" if "error" in r else ""))
    n = len(rows)
    summary = {"model": a.model, "n": n, "pass_rate": sum(r["graded"]["pass"] for r in rows) / n,
               "mean_ref_calls": sum(r.get("ref_calls", 0) for r in rows) / n,
               "overclaims": sum(r.get("overclaim", False) for r in rows),
               "total_cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 3)}
    print(json.dumps(summary, indent=1))
    (ROOT / "evals" / "claude_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
