from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graphs.main import repair_tool_calls


def test_repair_answers_dangling_tool_calls_only():
    done = AIMessage("", tool_calls=[{"id": "a", "name": "web_search", "args": {}}])
    dangling = AIMessage("", tool_calls=[{"id": "b", "name": "fetch_page", "args": {}}])
    msgs = [HumanMessage("q1"), done, ToolMessage("r", tool_call_id="a"), AIMessage("ans"),
            HumanMessage("q2"), dangling, HumanMessage("q3")]

    out = repair_tool_calls(msgs)

    assert out[:6] == msgs[:6]
    assert isinstance(out[6], ToolMessage) and out[6].tool_call_id == "b"
    assert out[7] is msgs[6]
    assert repair_tool_calls(msgs[:4]) == msgs[:4]


def test_stub_shrinks_only_large_tool_results_from_earlier_turns():
    from app.graphs.main import stub_old_tool_results

    big = "x" * 5000
    call = lambda i: AIMessage("", tool_calls=[{"id": i, "name": "fetch_page", "args": {}}])
    msgs = [HumanMessage("q1"), call("a"), ToolMessage(big, tool_call_id="a", name="fetch_page"),
            ToolMessage("12:00", tool_call_id="t", name="get_current_time"), AIMessage("ans1"),
            HumanMessage("q2"), call("b"), ToolMessage(big, tool_call_id="b", name="fetch_page")]

    out = stub_old_tool_results(msgs)

    assert "omitted" in out[2].content and out[2].tool_call_id == "a"
    assert out[3].content == "12:00"
    assert out[7].content == big
    assert msgs[2].content == big  # checkpointed state untouched


class ScriptedLLM:
    """Non-streaming fake that keeps tool_calls (GenericFakeChatModel drops them when streamed)."""

    @staticmethod
    def make(responses):
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult

        class _Fake(BaseChatModel):
            responses: list

            @property
            def _llm_type(self):
                return "scripted"

            def _generate(self, messages, stop=None, run_manager=None, **kw):
                return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

            def bind_tools(self, tools, **kw):
                return self

        return _Fake(responses=list(responses))


async def test_tool_calls_over_the_per_turn_cap_are_not_run():
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    from app.graphs.main import build_graph

    ran = []

    @tool
    async def web_search(query: str) -> str:
        """search"""
        ran.append(query)
        return "r"

    def calls(prefix, n):
        return AIMessage("", tool_calls=[{"id": f"{prefix}{i}", "name": "web_search", "args": {"query": f"{prefix}{i}"}}
                                         for i in range(n)])

    llm = ScriptedLLM.make([calls("a", 2), calls("b", 2), AIMessage("done1"), calls("c", 3), AIMessage("done2")])
    g = build_graph(llm, [web_search], tool_limits={"web_search": 3}, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}

    out = await g.ainvoke({"messages": [HumanMessage("q1")]}, cfg)
    blocked = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert ran == ["a0", "a1", "b0"] and [m.tool_call_id for m in blocked] == ["b1"]
    assert "Limit reached" in blocked[0].content

    await g.ainvoke({"messages": [HumanMessage("q2")]}, cfg)  # the cap is per user turn
    assert ran[3:] == ["c0", "c1", "c2"]


def test_system_prompt_grounding_rules():
    from app.graphs.main import GROUNDING_OFFLINE, GROUNDING_WEB, system_prompt

    clarify = system_prompt("clarify", "web", "local", web=False)
    assert "ONE short clarifying question" in clarify and GROUNDING_OFFLINE not in clarify
    assert GROUNDING_WEB in system_prompt("chat", "web", "cloud_fast", web=True)
    assert GROUNDING_OFFLINE in system_prompt("chat", "web", "local", web=False)
