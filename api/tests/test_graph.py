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
