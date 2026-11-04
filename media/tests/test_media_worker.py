"""media-worker logic without models: frame picking, transcript text, lock + cache, run hand-off."""
import os

import pytest

os.environ.setdefault("POLICY_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "config", "policy.yaml"))

from media import asr, common, extract, video, worker  # noqa: E402


@pytest.fixture
async def redis():
    import fakeredis

    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    common.set_redis(r)
    yield r
    await r.aclose()


@pytest.fixture
def rows(monkeypatch):
    table: dict[str, dict] = {}

    async def get_att(i):
        return table.get(i)

    async def update_att(i, **f):
        table[i].update(f)

    async def find_cached(sha, exclude):
        return next((r for r in table.values() if r["sha256"] == sha and r["id"] != exclude
                     and r["status"] == "done" and not (r.get("meta") or {}).get("degraded")), None)

    monkeypatch.setattr(common, "get_att", get_att)
    monkeypatch.setattr(common, "update_att", update_att)
    monkeypatch.setattr(common, "find_cached", find_cached)
    return table


def test_pick_times_thins_scenes_and_fills_static_videos():
    assert video.pick_times([], 120, 16) == [0.5, 20.0, 40.0, 60.0, 80.0, 100.0]
    many = video.pick_times([i * 3.0 for i in range(1, 200)], 600, 16)
    assert len(many) <= 16 and many[0] == 0.5 and all(b - a >= 2 for a, b in zip(many, many[1:]))
    assert video.pick_times([], 1, 16) == [0.0]


def test_transcript_text():
    zh = {"language": "zh", "segments": [{"start": 0, "text": "你好"}, {"start": 65, "text": "再見"}]}
    assert asr.as_text(zh, timestamps=False) == "你好再見"
    assert asr.as_text(zh, timestamps=True) == "[00:00] 你好\n[01:05] 再見"
    assert asr.as_text({"language": "en", "segments": [{"start": 0, "text": "hi"}, {"start": 1, "text": "there"}]},
                       timestamps=False) == "hi there"
    assert common.to_trad("软件信息") == "軟體資訊"


async def test_extract_once_then_cache_by_hash(redis, rows, monkeypatch):
    calls = []

    async def fake_run(att, progress=None):
        calls.append(att["id"])
        return {"kind": "doc", "text": "T", "summary": "s", "meta": {"path": "cpu", "degraded": False}}

    monkeypatch.setattr(extract, "run_extractor", fake_run)
    rows["a"] = {"id": "a", "sha256": "h", "kind": "doc", "status": "pending", "meta": {}}
    rows["b"] = {"id": "b", "sha256": "h", "kind": "doc", "status": "pending", "meta": {}}
    assert (await extract.extract("a"))["text"] == "T"
    assert (await extract.extract("a"))["status"] == "done"            # already done: no second run
    b = await extract.extract("b")
    assert calls == ["a"] and b["text"] == "T" and rows["b"]["meta"]["cached_from"] == "a"
    assert not await redis.exists("media:lock:a")


async def test_extract_failure_is_recorded(redis, rows, monkeypatch):
    async def boom(att, progress=None):
        raise ValueError("not a readable audio/video file")

    monkeypatch.setattr(extract, "run_extractor", boom)
    rows["a"] = {"id": "a", "sha256": "h", "kind": "voice", "status": "pending", "meta": {}}
    out = await extract.extract("a")
    assert out["status"] == "error" and "not a readable" in rows["a"]["error"]


class FakeArq:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, fn, *a, **kw):
        self.jobs.append((fn, a, kw))


async def test_media_run_hands_off_to_the_right_queue(redis, rows, monkeypatch):
    async def fake_run(att, progress=None):
        return {"kind": "large_doc" if att["id"] == "big" else "image", "text": "x", "summary": "", "meta": {}}

    monkeypatch.setattr(extract, "run_extractor", fake_run)
    for i in ("img", "big"):
        rows[i] = {"id": i, "sha256": i, "kind": "doc", "status": "pending", "meta": {}}
    arq = FakeArq()
    payload = {"run_id": "r1", "job": {"attachments": [{"id": "img", "kind": "image"}]}}
    assert (await worker.media_run({"redis": arq}, payload))["status"] == "submitted"
    assert arq.jobs[-1][0] == "submit_media_run" and arq.jobs[-1][2] == {"_job_id": "msub:r1", "_defer_by": None,
                                                                         "_queue_name": "arq:interactive"}
    payload = {"run_id": "r2", "job": {"attachments": [{"id": "big", "kind": "doc"}]}}
    await worker.media_run({"redis": arq}, payload)
    assert arq.jobs[-1][2]["_queue_name"] == "arq:background"   # a PDF that turned out large


async def test_media_run_stops_when_cancelled(redis, rows):
    await redis.set("cancel:r3", "1")
    arq = FakeArq()
    out = await worker.media_run({"redis": arq}, {"run_id": "r3", "job": {"attachments": [{"id": "a", "kind": "doc"}]}})
    assert out["status"] == "cancelled" and not arq.jobs


def test_line_file_without_text_waits_for_the_question():
    import time
    now = time.time()
    wait = worker._line_text_wait({"channel": "line", "text": ""}, {"reply_ts": now - 5}, ["image"])
    assert 9 < wait <= 10
    assert worker._line_text_wait({"channel": "line", "text": ""}, {"reply_ts": now - 60}, ["image"]) is None
    assert worker._line_text_wait({"channel": "line", "text": "這是什麼"}, {"reply_ts": now}, ["image"]) is None
    assert worker._line_text_wait({"channel": "line", "text": ""}, {"reply_ts": now}, ["voice"]) is None
    assert worker._line_text_wait({"channel": "web", "text": ""}, {"reply_ts": now}, ["image"]) is None
