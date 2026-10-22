"""MCP server exposing the RF tools so any agent (Claude Code, Claude Desktop, Discovery-style
platforms) can call a fast model in its loop and a reference model to verify."""
from mcp.server.mcpserver import MCPServer

import tools_api as t

mcp = MCPServer("rf-agent", instructions="Use simulate_fast inside your design loop; "
                "confirm with verify_reference before claiming a design works.")

for fn in (t.list_substrates, t.design_patch, t.simulate_fast, t.verify_reference,
           t.microstrip, t.match_l_network, t.analyze_touchstone, t.plot_design):
    mcp.tool()(fn)


@mcp.tool()
def call_stats() -> dict:
    """How many fast vs reference evaluations this session has spent."""
    return dict(t.STATS)


if __name__ == "__main__":
    mcp.run()
