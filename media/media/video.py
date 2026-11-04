"""Video / audio files: ffprobe, 16 kHz speech for ASR, scene-change keyframes described or OCR'd, merged into one
timeline (§5). Always a background job."""
import asyncio
import json
import logging
import os
import re
import tempfile
from typing import Any

from . import asr, common, image

log = logging.getLogger(__name__)
_PTS = re.compile(r"pts_time:([0-9.]+)")


async def _run(*args: str, timeout: float = 3600) -> tuple[int, str, str]:
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout)
    except asyncio.TimeoutError:
        p.kill()
        raise
    return p.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def probe(src: str) -> dict[str, Any]:
    rc, out, err = await _run("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", src,
                              timeout=60)
    if rc:
        raise ValueError(f"not a readable audio/video file ({err.strip()[:200]})")
    info = json.loads(out or "{}")
    streams = info.get("streams") or []
    video = [s for s in streams if s.get("codec_type") == "video"
             and not (s.get("disposition") or {}).get("attached_pic")]   # album art is not a video
    return {"duration_s": float((info.get("format") or {}).get("duration") or 0),
            "has_video": bool(video), "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "width": video[0].get("width") if video else None, "height": video[0].get("height") if video else None}


async def to_wav(src: str, dst: str) -> None:
    rc, _, err = await _run("ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
                            "-f", "wav", dst)
    if rc:
        raise ValueError(f"could not decode the audio ({err.strip()[:200]})")


def pick_times(scene_times: list[float], duration: float, max_frames: int) -> list[float]:
    """Scene changes (plus the opening frame), thinned evenly to max_frames; uniform sampling when the video has
    few cuts (talking head, screen recording)."""
    times = sorted({0.5 if duration > 1 else 0.0, *[t for t in scene_times if t < duration]})
    want = min(max_frames, max(4, int(duration // 30) + 1))
    if len(times) < min(4, want) and duration > 2:
        step = duration / (want + 1)
        times = sorted({*times, *[round(step * (i + 1), 2) for i in range(want)]})
    if len(times) > max_frames:
        idx = [round(i * (len(times) - 1) / (max_frames - 1)) for i in range(max_frames)] if max_frames > 1 else [0]
        times = [times[i] for i in sorted(set(idx))]
    # drop frames closer than 2 s to the previous one
    out: list[float] = []
    for t in times:
        if not out or t - out[-1] >= 2:
            out.append(t)
    return out


async def scene_times(src: str, threshold: float) -> list[float]:
    vf = f"scale=320:-2,select='gt(scene,{threshold})',showinfo"
    rc, _, err = await _run("ffmpeg", "-nostdin", "-v", "info", "-i", src, "-an", "-vf", vf, "-f", "null", "-")
    return [float(m.group(1)) for m in _PTS.finditer(err)] if rc == 0 else []


async def grab(src: str, t: float, dst: str, max_side: int) -> bool:
    vf = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    rc, _, _ = await _run("ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1",
                          "-vf", vf, "-q:v", "3", dst, timeout=120)
    return rc == 0 and os.path.exists(dst)


async def extract(att: dict[str, Any], *, progress=None) -> dict[str, Any]:
    v = common.sub("video")
    src = str(common.file_path(att["sha256"]))
    info = await probe(src)
    if info["duration_s"] > float(v.get("max_s", 3 * 3600)):
        raise ValueError(f"too long ({common.mmss(info['duration_s'])}; limit {common.mmss(float(v.get('max_s', 10800)))})")
    meta: dict[str, Any] = {"duration_s": round(info["duration_s"], 1), "degraded": False}
    events: list[tuple[float, str]] = []
    with tempfile.TemporaryDirectory() as d:
        speech: dict[str, Any] = {}
        if info["has_audio"]:
            if progress:
                await progress("transcribing")
            wav = os.path.join(d, "a.wav")
            await to_wav(src, wav)
            speech = await asr.transcribe(wav)
            meta["asr_path"] = speech["path"]
            meta["language"] = speech.get("language")
            events += [(s["start"], f"🗣 {s['text']}") for s in speech["segments"]]
        if not info["has_video"]:   # an audio file that looked like a video container
            dur = speech.get("duration_s") or info["duration_s"]
            meta = {**meta, "path": speech.get("path", "cpu"), "duration_s": dur}
            long = dur > float(common.cfg().get("voice_as_message_s", 120))
            text = asr.as_text(speech, timestamps=long) if speech else ""
            return {"kind": "voice", "text": text, "summary": f"語音 {common.mmss(dur)} · {text[:80]}", "meta": meta}
        if progress:
            await progress("keyframes")
        times = pick_times(await scene_times(src, float(v.get("scene", 0.3))), info["duration_s"],
                           int(v.get("max_frames", 16)))
        path = await image.describe_path(bool(att.get("private")))
        meta.update(path=path, frames=len(times), degraded=path == "cpu")
        for i, t in enumerate(times):
            f = os.path.join(d, f"f{i}.jpg")
            if not await grab(src, t, f, int(common.sub("image").get("max_side", 1280))):
                continue
            img = await asyncio.to_thread(image.load, f)
            desc = ""
            if path != "cpu":
                try:
                    desc = await image.describe(img, path, image.FRAME_PROMPT, 150)
                except Exception as e:
                    log.warning("frame description failed: %s", e)
                    path, meta["path"], meta["degraded"] = "cpu", "cpu", True
            found = (await image.ocr(img)).replace("\n", " / ")
            line = " ".join(x for x in (desc.replace("\n", " "), f"〔畫面文字:{found[:300]}〕" if found else "") if x)
            if line:
                events.append((t, f"🖼 {line}"))
    events.sort(key=lambda e: e[0])
    text = "\n".join(f"[{common.mmss(t)}] {line}" for t, line in events)
    if meta["degraded"]:
        text += "\n(GPU 忙碌中:畫面只做了文字辨識,沒有描述。)"
    summary = f"影片 {common.mmss(info['duration_s'])} · {len(times)} 個畫面" + (" · 有語音" if speech.get("segments") else "")
    return {"kind": "video", "text": text, "summary": summary, "meta": meta}
