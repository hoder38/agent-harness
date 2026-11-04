"""Images: OCR (RapidOCR, CPU, always) + a description (local Qwen vision while the GPU is FREE; optionally a
cloud model while it is busy, never for PRIVATE files) (§5)."""
import asyncio
import base64
import io
import re
import logging
import threading
from typing import Any

import httpx
from PIL import Image, ImageOps

from . import common

log = logging.getLogger(__name__)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

_ocr = None
_ocr_lock = threading.Lock()

CAPTION_PROMPT = ("請用繁體中文描述這張圖片:主要內容、人物或物件、場景,以及圖中重要的文字或數字。"
                  "只描述看得到的東西,不要猜測。")
FRAME_PROMPT = "用一句繁體中文描述這個影片畫面(看得到的內容與文字)。"


def load(path: str, max_side: int | None = None) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)    # apply rotation, then the EXIF is dropped with the re-encode
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def jpeg_b64(img: Image.Image, max_side: int) -> str:
    """Re-encoded JPEG without metadata (EXIF/GPS never leaves the box, §5)."""
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _ocr_sync(img: Image.Image) -> str:
    global _ocr
    import numpy as np

    with _ocr_lock:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
        res, _ = _ocr(np.array(img)[:, :, ::-1])   # BGR like OpenCV
    if not res:
        return ""
    lines: list[list[tuple[float, str]]] = []      # group boxes into rows by their vertical centre
    for box, text, score in sorted(res, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2):
        if float(score) < 0.5 or not text.strip():
            continue
        y = (box[0][1] + box[2][1]) / 2
        h = abs(box[2][1] - box[0][1]) or 1
        if lines and abs(lines[-1][0][0] - y) < h * 0.5:
            lines[-1].append((y, text, box[0][0]))
        else:
            lines.append([(y, text, box[0][0])])
    return "\n".join(" ".join(t for _, t, _ in sorted(row, key=lambda x: x[2])) for row in lines)


async def ocr(img: Image.Image) -> str:
    return await asyncio.to_thread(_ocr_sync, img)


async def caption_local(img: Image.Image, prompt: str = CAPTION_PROMPT, max_tokens: int = 500) -> str:
    m = common.cfg()
    body = {"model": m.get("vision_model", "qwen3.5:9b"), "stream": False, "think": False, "keep_alive": "10m",
            "messages": [{"role": "user", "content": prompt,
                          "images": [jpeg_b64(img, int(common.sub("image").get("max_side", 1280)))]}],
            "options": {"num_predict": max_tokens, "temperature": 0.2}}
    async with common.GpuSemaphore():
        async with httpx.AsyncClient(timeout=180, trust_env=False) as c:
            r = await c.post(f"{common.OLLAMA_URL}/api/chat", json=body)
            r.raise_for_status()
    return common.to_trad((r.json().get("message") or {}).get("content", "").strip(), "s2tw")


async def caption_cloud(img: Image.Image, prompt: str = CAPTION_PROMPT, max_tokens: int = 500) -> str:
    """Only for NORMAL files while the GPU is busy and media.image.when_gpu_busy = cloud."""
    model = common.sub("image").get("cloud_model", "cloud_fast")
    provider = (common.policy().get("providers") or {}).get(model, "openai")
    uri = "data:image/jpeg;base64," + jpeg_b64(img, int(common.sub("image").get("max_side", 1280)))
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": uri}}]}]}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as c:
        r = await c.post(f"{common.LITELLM_BASE_URL}/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {common.LITELLM_MASTER_KEY}"})
        r.raise_for_status()
    try:
        await common.add_spend(provider, float(r.headers.get("x-litellm-response-cost") or 0))
    except ValueError:
        pass
    return common.to_trad(r.json()["choices"][0]["message"]["content"].strip(), "s2tw")


async def describe_path(private: bool) -> str:
    """gpu | cloud | cpu: who may describe a picture right now."""
    if await common.gpu_state() == "FREE":
        return "gpu"
    if (common.sub("image").get("when_gpu_busy", "ocr_only") == "cloud" and not private
            and await common.killswitch() == "off"):
        provider = (common.policy().get("providers") or {}).get(common.sub("image").get("cloud_model", "cloud_fast"),
                                                                "openai")
        if await common.budget_left(provider):
            return "cloud"
    return "cpu"


async def describe(img: Image.Image, path: str, prompt: str = CAPTION_PROMPT, max_tokens: int = 500) -> str:
    if path == "gpu":
        return await caption_local(img, prompt, max_tokens)
    if path == "cloud":
        return await caption_cloud(img, prompt, max_tokens)
    return ""


async def extract(att: dict[str, Any]) -> dict[str, Any]:
    src = str(common.file_path(att["sha256"]))
    img = await asyncio.to_thread(load, src)
    path = await describe_path(bool(att.get("private")))
    text_task = asyncio.create_task(ocr(img))
    caption, err = "", None
    try:
        caption = await describe(img, path)
    except Exception as e:
        log.warning("image description failed (%s): %s", path, e)
        err, path = str(e)[:200], "cpu"
    found = await text_task
    parts = []
    if caption:
        parts.append(f"【圖片描述】\n{caption}")
    parts.append(f"【圖中文字】\n{found}" if found else "【圖中文字】(無)")
    if path == "cpu":
        parts.append("(GPU 忙碌中,只做了文字辨識,沒有圖片描述。)" if not err else "(圖片描述失敗,只有文字辨識。)")
    meta = {"path": path, "degraded": path == "cpu", "width": img.width, "height": img.height}
    if err:
        meta["caption_error"] = err
    summary = re.sub(r"[*#`]+|\s+", " ", caption or found or "圖片").strip()[:120]
    return {"kind": "image", "text": "\n\n".join(parts), "summary": summary, "meta": meta}
