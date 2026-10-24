"""
Render the LangGraph workflow to docs/agent_graph.png.

Needs an LLM API key (the agents are constructed to build the graph) and
network access to the mermaid.ink rendering service.

    python scripts/render_agent_graph.py
"""
from pathlib import Path

from medical_deserts.graph import build_enhanced_graph

OUT = Path(__file__).resolve().parents[1] / "docs" / "agent_graph.png"

if __name__ == "__main__":
    app = build_enhanced_graph()
    OUT.write_bytes(app.get_graph().draw_mermaid_png())
    print(f"Wrote {OUT}")
