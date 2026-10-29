"""Shopping branch (§12c): slot filling + clarify, real listings only, code-rendered picks."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graphs.grounded import build_grounded_graph
from app.tools import shop
from tests.test_grounded import TaskLLM

LD = """<script type="application/ld+json">{"@type":"ItemList","itemListElement":[{"@type":"Product",
"name":"【iwaki】玻璃調味罐 120ml","url":"https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=111&Area=search&kw=x",
"offers":{"@type":"Offer","price":399,"availability":"https://schema.org/InStock"},
"aggregateRating":{"ratingValue":"4.8","reviewCount":"12"}},{"@type":"Product","name":"缺貨罐",
"url":"https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=222","offers":{"price":"199",
"availability":"https://schema.org/OutOfStock"}}]}</script>"""

P = [{"id": f"pchome:P{i}", "store": "pchome", "name": f"調味料罐{i}號", "brand": "B" if i % 2 else None,
      "price": 100 * i, "url": f"https://24h.pchome.com.tw/prod/P{i}", "rating": None, "reviews": None,
      "in_stock": True, "desc": ""} for i in range(1, 7)]


def test_ld_itemlist_parser_cleans_urls_and_brand():
    items = shop.parse_ld_itemlist(LD, "momo")
    assert items[0]["url"] == "https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=111"
    assert items[0]["brand"] == "iwaki" and items[0]["price"] == 399 and items[0]["rating"] == 4.8
    assert items[1]["in_stock"] is False


async def test_search_products_filters_budget_stock_and_dupes(redis, monkeypatch):
    async def fake_get(url):
        return LD

    monkeypatch.setattr(shop, "_get", fake_get)
    res = await shop.search_products(["罐"], ["momo"], 300, 500)
    assert [p["price"] for p in res["products"]] == [399]  # out-of-stock and outside-budget listings dropped


@pytest.fixture
def pol(policy):
    return {**policy, "grounding": {**policy["grounding"], "mode": "cite_only"},
            "shopping": {**policy["shopping"], "enabled": True}}


def graph(pol, script, found=P):
    llm = TaskLLM(script=script, calls=[], seen=[])
    calls = []

    async def products(terms, stores, lo, hi):
        calls.append((terms, lo, hi))
        return {"products": [p for p in found if (lo or 0) <= p["price"] <= (hi or 10**9)], "errors": [], "calls": 1}

    g = build_grounded_graph(llm, policy=pol, checkpointer=InMemorySaver(), products=products)
    return g, llm, calls


CFG = {"configurable": {"thread_id": "s1"}}
KNOWN = {"use_case": "自用", "must_have": ["好清洗"], "brand_pref": "不限", "store_pref": "不限"}  # nothing left to ask


def inp(text, task_type="shopping", **kw):
    from langchain_core.messages import HumanMessage
    rid = kw.pop("run_id", "r1")
    return {"messages": [HumanMessage(text, id=rid)], "run_id": rid, "thread_id": "s1", "channel": "web",
            "private": False, "model": "local", "task_type": task_type, **kw}


async def test_asks_budget_then_recommends_only_real_listings(pol):
    slots = [{"shopping": True, "item": "調味料罐", "item_specific": True, "search_terms": ["調味料罐 500以下"],
              "question_to_user": "預算大概多少？(選項：300以下 / 300-600)",
              "options": [{"price_range": "NT$300 以下", "description": "x"}, "NT$300–600"]},
             {"shopping": True, "action": "refine", "budget_max": 300, **KNOWN}]
    pick = "幫你挑了幾款：\n[2] 小巧好收納\n[9] 不存在的商品\n[1] 最便宜\n買前看容量。https://evil.example/x"
    g, llm, calls = graph(pol, {"shop_slots": slots, "shop_compose": pick})
    out = await g.ainvoke(inp("推薦調味料罐"), CFG)
    ask = out["__interrupt__"][0].value
    assert ask["options"] == ["NT$300 以下", "NT$300–600"] and ask["question"] == "預算大概多少？"
    out = await g.ainvoke(Command(resume="1"), CFG)
    assert calls == [(["調味料罐"], None, 300)]
    text = out["answer"]["text"]
    assert "1. 調味料罐2號\n   NT$200｜PChome 24h\n   小巧好收納\n   https://24h.pchome.com.tw/prod/P2" in text
    assert "2. 調味料罐1號" in text and "不存在" not in text and "evil.example" not in text
    assert [c["url"] for c in out["answer"]["citations"]] == [P[1]["url"], P[0]["url"]]
    assert "預算 ≤ NT$300" in text
    assert "assess_ambiguity" not in llm.calls and "plan_search" not in llm.calls


async def test_follow_up_refines_last_search(pol):
    slots = [{"shopping": True, "item": "罐", "search_terms": ["調味料罐"], "budget_min": 300, "budget_max": 600, **KNOWN},
             {"shopping": True, "action": "refine", "budget_min": None, "budget_max": 300}]
    g, _, calls = graph(pol, {"shop_slots": slots, "shop_compose": "[1] ok"})
    await g.ainvoke(inp("推薦調味料罐 300到600"), CFG)
    from langchain_core.messages import HumanMessage
    await g.ainvoke({"messages": [HumanMessage("便宜一點", id="r2")], "run_id": "r2"}, CFG)
    assert calls == [(["調味料罐"], 300, 600), (["調味料罐"], None, 300)]


async def test_cheaper_lowers_the_ceiling_even_if_model_raises_the_floor(pol):
    slots = [{"shopping": True, "item": "罐", "search_terms": ["調味料罐"], "budget_min": 800, "budget_max": 1200, **KNOWN},
             {"shopping": True, "action": "refine", "item_specific": False, "budget_min": 500, "budget_max": 1200}]
    g, _, calls = graph(pol, {"shop_slots": slots, "shop_compose": "[1] ok"})
    await g.ainvoke(inp("推薦調味料罐 1000左右"), CFG)
    from langchain_core.messages import HumanMessage
    await g.ainvoke({"messages": [HumanMessage("便宜一點", id="r2")], "run_id": "r2"}, CFG)
    assert calls[1] == (["調味料罐"], None, 840)


def test_relevance_drops_padding_and_bare_modifiers():
    assert shop.strong_terms(["電動", "胡椒研磨機"]) == ["胡椒研磨機"]
    items = [{"name": n} for n in ("【小米】電動沖牙器 2", "不鏽鋼電動胡椒研磨器 陶瓷芯", "電動起子機", "胡椒研磨罐")]
    kept = [p["name"] for p in shop.filter_relevant(items, ["電動", "胡椒研磨機"])]
    assert kept == ["不鏽鋼電動胡椒研磨器 陶瓷芯", "胡椒研磨罐"]


async def test_router_task_type_decides_shopping_and_no_pick_retry(pol):
    g, llm, _ = graph(pol, {"compose": "Hi."})
    out = await g.ainvoke(inp("推薦適合冷凍且富含纖維素的食材", task_type="advice"), CFG)
    assert out["answer"]["text"] == "我理解為：知識或建議—「推薦適合冷凍且富含纖維素的食材」\n\nHi."
    assert "shop_slots" not in llm.calls and "shop_compose" not in llm.calls
    g, llm, _ = graph(pol, {"shop_slots": {"shopping": True, "item": "罐", "search_terms": ["罐"], "no_budget": True, **KNOWN},
                            "shop_compose": ["沒有編號", "[3] 好"]})
    out = await g.ainvoke(inp("想買罐子 預算不限"), {"configurable": {"thread_id": "s2"}})
    assert llm.calls.count("shop_compose") == 2 and out["answer"]["products"][0]["id"] == "pchome:P3"


async def test_no_budget_needs_the_user_to_say_so(pol):
    slots = {"shopping": True, "item": "罐", "item_specific": True, "search_terms": ["調味料罐"], "no_budget": True}
    g, _, calls = graph(pol, {"shop_slots": slots, "shop_compose": "[1] ok"})
    out = await g.ainvoke(inp("推薦調味料罐"), CFG)
    assert out["__interrupt__"][0].value["question"] == "預算大概多少？" and not calls


async def test_standing_preference_saved_even_if_model_forgets(pol, redis):
    slots = {"shopping": True, "item": "罐", "search_terms": ["調味料罐"], "no_budget": True, "brand_pref": "有品牌",
             "remember": {}}
    g, _, _ = graph(pol, {"shop_slots": slots, "shop_compose": "[1] ok"})
    await g.ainvoke(inp("推薦調味料罐 價格不限，我偏好有品牌的，以後都這樣"), CFG)
    import json
    assert json.loads(await redis.get("shop:prefs")) == {"brand_pref": "有品牌"}


def test_parse_budget_backstop():
    from app.graphs.grounded import parse_budget
    assert parse_budget("推薦研磨器 1000以下") == {"budget_max": 1000}
    assert parse_budget("預算 300到600") == {"budget_min": 300, "budget_max": 600}
    assert parse_budget("大概1,000") == {"budget_min": 800, "budget_max": 1200}
    assert parse_budget("2千以上") == {"budget_min": 2000}
    assert parse_budget("under $500") == {"budget_max": 500}
    assert parse_budget("推薦調味料罐") == {}


async def test_model_dropping_budget_and_brand_is_backstopped(pol, redis):
    slots = {"shopping": True, "item": "研磨器", "item_specific": False, "search_terms": ["胡椒研磨器", "知名品牌"],
             "budget_min": 800, "budget_max": 1200,
             "remember": {"preference": "有品牌"}, **{**KNOWN, "brand_pref": None}}
    g, _, calls = graph(pol, {"shop_slots": slots, "shop_compose": "[1] ok"})
    await g.ainvoke(inp("推薦電動胡椒研磨器 1000以下，我偏好有品牌的，以後都這樣"), CFG)
    assert calls == [(["胡椒研磨器"], None, 1000)]
    import json
    assert json.loads(await redis.get("shop:prefs")) == {"brand_pref": "有品牌"}


async def test_asks_one_unknown_detail_per_round_in_order(pol, redis):
    base = {"shopping": True, "item": "調味料罐", "item_specific": True, "search_terms": ["調味料罐"], "budget_max": 500}
    script = [base,
              {**base, "use_case": "通勤", "questions": [
                  {"slot": "brand", "question": "品牌？", "options": ["象印"]},
                  {"slot": "spec:容量", "question": "容量大概多少？", "options": ["350ml", "500ml"]},
                  {"slot": "spec:容量", "question": "dup", "options": []}]},
              {**base, "use_case": "通勤", "questions": [{"slot": "spec:杯子容量", "question": "容量大概多少呢？"}]},
              {**base, "use_case": "通勤", "specs": {"容量": "500ml"}},
              {**base, "use_case": "通勤", "specs": {"容量": "500ml"}, "must_have": ["好清洗"]},
              {**base, "use_case": "通勤", "specs": {"容量": "500ml"}, "must_have": ["好清洗"], "brand_pref": "象印"}]
    g, llm, calls = graph(pol, {"shop_slots": script, "shop_compose": "[1] ok"})
    out = await g.ainvoke(inp("想買調味料罐 500以下"), CFG)
    seen = []
    for ans in ["通勤用", "500ml", "好清洗", "象印"]:  # the reworded capacity question is not asked again
        ask = out["__interrupt__"][0].value
        seen.append(ask["question"])
        out = await g.ainvoke(Command(resume=ans), CFG)
    ask = out["__interrupt__"][0].value
    assert seen == ["主要是誰用、用在什麼情境？", "容量大概多少？", "有沒有一定要有、或想避開的功能？", "品牌？"]
    assert ask["question"] == "有偏好的購物網站嗎？" and ask["options"][-1] == "不限" and not calls
    out = await g.ainvoke(Command(resume="momo購物網"), CFG)
    assert calls == [(["調味料罐"], None, 500)] and out["answer"]["products"]
    assert "商店 momo購物網" in out["answer"]["text"]


async def test_answer_mid_clarify_stays_in_shopping(pol):
    base = {"shopping": True, "item": "調味料罐", "item_specific": True, "search_terms": ["調味料罐"], "budget_max": 500}
    g, llm, calls = graph(pol, {"shop_slots": [base, {"shopping": False}, {**base, **KNOWN}], "shop_compose": "[1] ok"})
    out = await g.ainvoke(inp("想買調味料罐 500以下"), CFG)
    out = await g.ainvoke(Command(resume="自用"), CFG)
    assert out["__interrupt__"] and "assess_ambiguity" not in llm.calls


async def test_skip_stops_asking_and_searches(pol):
    base = {"shopping": True, "item": "調味料罐", "item_specific": True, "search_terms": ["調味料罐"], "budget_max": 500}
    g, _, calls = graph(pol, {"shop_slots": base, "shop_compose": "[1] ok"})
    out = await g.ainvoke(inp("想買調味料罐 500以下"), CFG)
    assert out["__interrupt__"]
    out = await g.ainvoke(Command(resume="/skip"), CFG)
    assert calls and out["answer"]["products"]


async def test_private_threads_never_shop(pol):
    g, llm, calls = graph(pol, {"compose": "ok"})
    await g.ainvoke({**inp("推薦調味料罐"), "private": True}, CFG)
    assert not calls and "shop_slots" not in llm.calls


FULL = {"item": "油罐", "item_specific": True, "search_terms": ["調味料罐"], "no_budget": True, **KNOWN}


@pytest.mark.parametrize("reply,update,shops", [("2", {}, True), ("知識或建議", {}, False),
                                                ("我只是想知道怎麼挑", {"answer_task_type": "advice"}, False),
                                                ("/skip", {}, False)])
async def test_unsure_task_type_is_asked_with_label_options(pol, reply, update, shops):
    g, llm, _ = graph(pol, {"shop_slots": FULL, "shop_compose": "[1] ok", "compose": "Hi."})
    out = await g.ainvoke(inp("推薦適合的油罐 預算不限", task_type="advice", confirm_task=["advice", "shopping"]), CFG)
    ask = out["__interrupt__"][0].value
    assert (ask["question"], ask["options"]) == ("你是想：", ["知識或建議", "找商品購買"]) and not llm.calls
    out = await g.ainvoke(Command(resume=reply, update=update), CFG)
    assert ("shop_slots" in llm.calls) is shops
    label = "找商品購買" if shops else "知識或建議"
    assert out["answer"]["text"].startswith(f"我理解為：{label}—「推薦適合的油罐 預算不限」\n\n")


async def test_correction_merges_messages_and_drops_shopping_state(pol):
    g, llm, calls = graph(pol, {"shop_slots": FULL, "shop_compose": "[1] ok", "compose": "Hi."})
    await g.ainvoke(inp("推薦適合冷凍的食材"), CFG)
    assert (await g.aget_state(CFG)).values["shop"]
    out = await g.ainvoke(inp("不是，我是想知道選哪些蔬菜", task_type="advice", run_id="r2",
                              correction="推薦適合冷凍的食材\n不是，我是想知道選哪些蔬菜"), CFG)
    assert out["shop"] is None and out["correction"] is None and llm.calls.count("shop_slots") == 1
    assert out["answer"]["text"].startswith("我理解為：知識或建議—「推薦適合冷凍的食材 / 不是，我是想知道選哪些蔬菜」")
