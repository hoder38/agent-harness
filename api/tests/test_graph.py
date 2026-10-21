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
