"""Speech → text with faster-whisper: CUDA fp16 in every GPU state (GAMING only unloads Qwen, 2026-11-10); CPU int8
only when CUDA fails (§5)."""
import asyncio
import logging
import threading
import time
from typing import Any

from . import common

log = logging.getLogger(__name__)

_models: dict[str, Any] = {}       # "cuda" | "cpu" → WhisperModel
_last_used: dict[str, float] = {}
_lock = threading.Lock()           # one transcription per model at a time; loading is serialized too


def _model(device: str):
    from faster_whisper import WhisperModel

    asr = common.sub("asr")
    if device not in _models:
        name = asr.get("model", "large-v3-turbo")
        t = time.monotonic()
        if device == "cuda":
            _models[device] = WhisperModel(name, device="cuda", compute_type=asr.get("gpu_compute", "float16"),
                                           download_root="/models/whisper")
        else:
            _models[device] = WhisperModel(name, device="cpu", compute_type=asr.get("cpu_compute", "int8"),
                                           cpu_threads=int(asr.get("cpu_threads", 8)), download_root="/models/whisper")
        log.info("loaded whisper %s on %s in %.1fs", name, device, time.monotonic() - t)
    _last_used[device] = time.monotonic()
    return _models[device]


def _pcm(path: str):
    """16 kHz mono WAV (from ffmpeg) → float32 samples. faster-whisper's own PyAV decoding breaks on PyAV ≥ 15."""
    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected 16 kHz mono 16-bit WAV")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def _run(path: str, device: str) -> dict[str, Any]:
    asr = common.sub("asr")
    audio = _pcm(path)
    with _lock:
        model = _model(device)
        segs, info = model.transcribe(
            audio, beam_size=int(asr.get("beam_size", 5)), vad_filter=True,
            language=asr.get("language") or None,
            initial_prompt=asr.get("initial_prompt", "以下是繁體中文的句子。"),
            condition_on_previous_text=False)
        out = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs if s.text.strip()]
        _last_used[device] = time.monotonic()
    if (info.language or "").startswith("zh"):
        for s in out:
            s["text"] = common.to_trad(s["text"])
    return {"segments": out, "language": info.language, "duration_s": round(float(info.duration or 0), 1)}


def unload(device: str | None = None) -> list[str]:
    with _lock:
        gone = [d for d in list(_models) if device in (None, d)]
        for d in gone:
            del _models[d]
            _last_used.pop(d, None)
    if gone:
        import gc
        gc.collect()
        log.info("unloaded whisper on %s", ",".join(gone))
    return gone


def unload_idle(gpu_state: str) -> list[str]:
    """Unload after `asr.keep_alive_s` idle; the GPU state no longer matters (`gpu_state` kept for callers)."""
    keep = float(common.sub("asr").get("keep_alive_s", 300))
    now = time.monotonic()
    out = []
    for d, t in list(_last_used.items()):
        if now - t > keep:
            if not _lock.locked():
                out += unload(d)
    return out


async def transcribe(path: str) -> dict[str, Any]:
    """{segments, language, duration_s, path: gpu|cpu, degraded}. GPU first; CPU is the fallback."""
    try:
        async with common.GpuSemaphore():
            res = await asyncio.to_thread(_run, path, "cuda")
        return {**res, "path": "gpu", "degraded": False}
    except Exception as e:   # no CUDA / out of VRAM / unsupported arch: the CPU path still works
        log.warning("GPU transcription failed, using CPU: %s", e)
        unload("cuda")
    res = await asyncio.to_thread(_run, path, "cpu")
    return {**res, "path": "cpu", "degraded": False}


def as_text(res: dict[str, Any], *, timestamps: bool) -> str:
    segs = res.get("segments") or []
    if not timestamps:
        return "".join(s["text"] if res.get("language", "").startswith(("zh", "ja")) else s["text"] + " "
                       for s in segs).strip()
    return "\n".join(f"[{common.mmss(s['start'])}] {s['text']}" for s in segs)
