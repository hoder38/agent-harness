"""gpu-governor (§11): state machine, publishing, VRAM unload, deferred resume, heartbeat alert."""
import json

import pytest

from app import state
from app.governor import Governor, Sample, publish, unload_gpu_models
from app.jobs import alerts

CFG = {"gaming_util": 30, "gaming_exit_s": 300, "busy_util": 70, "busy_enter_s": 30, "busy_free_vram_gb": 8,
       "busy_exit_s": 120}


def s(t, util=0, enc=0, used=2.0, ollama=0.0, steam=False, own=False):
    return Sample(t=t, util=util, enc=enc, total_gb=16.0, used_gb=used, ollama_gb=ollama, steam=steam, own_jobs=own)


def test_gaming_needs_steam_and_load_and_holds_for_exit_window():
    g = Governor(CFG)
    assert g.step(s(0, util=90)) == "FREE"                  # load without steam-headless is not gaming
    assert g.step(s(1, steam=True)) == "FREE"               # steam idle
    assert g.step(s(2, steam=True, enc=5)) == "GAMING"      # streaming (NVENC)
    assert g.step(s(3, steam=False)) == "GAMING"            # condition cleared…
    assert g.step(s(301, steam=False)) == "GAMING"
    assert g.step(s(302, steam=False)) == "FREE"            # …for 5 min


def test_own_jobs_do_not_count_as_gaming_or_busy_load():
    g = Governor(CFG)
    assert g.step(s(0, util=95, steam=True, own=True)) == "FREE"
    assert g.step(s(40, util=95, steam=True, own=True)) == "FREE"


def test_busy_on_sustained_util_and_exits_after_calm_window():
    g = Governor(CFG)
    assert g.step(s(0, util=80)) == "FREE"
    assert g.step(s(29, util=80)) == "FREE"
    assert g.step(s(30, util=80)) == "BUSY"
    assert g.step(s(31, util=10)) == "BUSY"
    assert g.step(s(150, util=10)) == "BUSY"
    assert g.step(s(151, util=10)) == "FREE"


def test_busy_on_low_vram_but_ollama_counts_as_free():
    g = Governor(CFG)
    assert g.step(s(0, used=12.0, ollama=9.0)) == "FREE"    # 16-12+9 = 13 GB free: our own Qwen
    assert g.step(s(1, used=12.0)) == "BUSY"                # someone else holds 12 GB


def test_restart_in_gaming_holds_until_exit_window():
    g = Governor(CFG, current="GAMING")
    assert g.step(s(1000)) == "GAMING"
    assert g.step(s(1301)) == "FREE"


async def test_publish_sets_state_transition_and_heartbeat(redis):
    sub = redis.pubsub()
    await sub.subscribe("gpu_state_changed")
    assert await publish(redis, "GAMING", reason="r", override=False, sample={"util": 1}, ttl_s=30)
    assert not await publish(redis, "GAMING", reason="r", override=False, sample=None, ttl_s=30)
    assert await redis.get("gpu_state") == "GAMING"
    (t,) = [json.loads(x) for x in await redis.lrange("gpu_governor:transitions", 0, -1)]
    assert (t["from"], t["to"]) == (None, "GAMING")
    assert 0 < await redis.ttl("gpu_governor:heartbeat") <= 30
    msgs = [m async for m in _drain(sub)]
    assert msgs == ["GAMING"]


async def _drain(sub):
    for _ in range(5):
        if m := await sub.get_message(ignore_subscribe_messages=True, timeout=0.05):
            yield m["data"]


async def test_unload_skips_cpu_only_models():
    posted = []

    class Http:
        async def post(self, url, json):
            posted.append(json)

    gone = await unload_gpu_models(Http(), [{"name": "qwen", "size_vram": 6_000_000_000}, {"name": "cpu", "size_vram": 0}])
    assert gone == ["qwen"] and posted == [{"model": "qwen", "keep_alive": 0}]


async def test_deferred_jobs_resume_only_when_not_gaming(redis):
    enq = []

    class Arq:
        async def enqueue_job(self, fn, payload, **kw):
            enq.append((payload["run_id"], kw["_queue_name"]))

    await state.defer_job(redis, "r1", {"run_id": "r1"}, "arq:background")
    await redis.set("gpu_state", "GAMING")
    assert await alerts.resume_deferred({"redis": Arq()}) == 0
    await redis.set("gpu_state", "FREE")
    assert await alerts.resume_deferred({"redis": Arq()}) == 1
    assert enq == [("r1", "arq:background")] and not await redis.hgetall("deferred")


@pytest.mark.parametrize("hb,keys", [(None, ["governor:heartbeat"]), ({"at": 1, "state": "FREE"}, []),
                                     ({"at": 1, "error": "NVMLError"}, ["governor:error"])])
async def test_governor_heartbeat_alert(redis, policy, hb, keys):
    if hb:
        await redis.set("gpu_governor:heartbeat", json.dumps(hb))
    assert [k for k, _ in await alerts.check_governor(policy)] == keys
