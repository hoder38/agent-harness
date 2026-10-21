"""Graph entry for `langgraph dev` / Studio (local debugging only; uses LiteLLM alias `local`)."""
from ..llm.client import make_llm
from ..tools import tools_for
from .main import build_graph, system_prompt

graph = build_graph(make_llm("local", "off"), tools_for("chat"), system=system_prompt("chat", "web", "local"))
