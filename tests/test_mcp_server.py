import json
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def _roundtrip():
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server.py")], cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool("design_patch", {"f_ghz": 2.4, "substrate": "FR4"})
            return names, json.loads(res.content[0].text)


def test_server_lists_tools_and_designs():
    names, design = anyio.run(_roundtrip)
    assert {"design_patch", "simulate_fast", "verify_reference", "analyze_touchstone",
            "match_l_network", "call_stats", "predict_surrogate", "diagnose_measurement"} <= names
    assert 37 < design["w_mm"] < 39
