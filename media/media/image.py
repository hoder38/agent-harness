"""Images: OCR (RapidOCR, CPU, always) + a description (local Qwen vision while the GPU is FREE; optionally a
cloud model while it is busy, never for PRIVATE files) (§5)."""
import asyncio
import base64
import io
import math
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
                  "只描述看得到的東西,不要猜測;圖中沒寫出的貨幣、單位或品牌不要自行補上。")
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


def _cuda_ok() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def _ocr_boxes(img: Image.Image) -> list:
    global _ocr
    import numpy as np

    with _ocr_lock:
        if _ocr is None:
            from rapidocr import ModelType, OCRVersion, RapidOCR
            v = OCRVersion.PPOCRV6   # one model for Simplified, Traditional, Japanese and English (baked in the image)
            params = {"Global.log_level": "warning", "Det.ocr_version": v, "Rec.ocr_version": v,
                      "Det.model_type": ModelType.SMALL, "Rec.model_type": ModelType.SMALL}
            if _cuda_ok():   # onnxruntime-gpu; also while GAMING (only Qwen leaves the GPU)
                try:
                    # EXHAUSTIVE/HEURISTIC conv search re-runs per input shape: 4 s per image vs 0.5 s (DEFAULT)
                    _ocr = RapidOCR(params={**params, "EngineConfig.onnxruntime.use_cuda": True,
                                            "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "DEFAULT"})
                except Exception as e:  # noqa: BLE001  (CUDA/cuDNN missing, out of VRAM): the CPU still works
                    log.warning("RapidOCR on CUDA failed, using CPU: %s", e)
            if _ocr is None:
                _ocr = RapidOCR(params=params)
        res = _ocr(np.array(img)[:, :, ::-1])   # BGR like OpenCV
    if res.boxes is None:
        return []
    return [(box.tolist(), _trad(text.strip())) for box, text, score in zip(res.boxes, res.txts, res.scores)
            if float(score) >= 0.5 and text.strip()]


def _simplified_only(c: str) -> bool:
    """A Simplified form that Japanese never uses (绳, 经, 带), unlike shared kanji such as 干, 国, 冲."""
    if common.to_trad(c, "s2t") == c:
        return False
    try:
        c.encode("shift_jis")
        return False
    except UnicodeEncodeError:
        return True


def _trad(line: str) -> str:
    """The model prints what it sees, but drifts to Simplified on Chinese labels: convert only lines that are
    plainly Simplified Chinese, so Japanese (干物, 国内製造) stays as printed."""
    if KANA_RE.search(line) or not any(_simplified_only(c) for c in line):
        return line
    return common.to_trad(line, "s2tw")


def _rows(items: list) -> str:
    lines: list[list[tuple[float, str, float]]] = []      # group boxes into rows by their vertical centre
    for box, text in sorted(items, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2):
        y = (box[0][1] + box[2][1]) / 2
        h = abs(box[2][1] - box[0][1]) or 1
        if lines and abs(lines[-1][0][0] - y) < h * 0.5:
            lines[-1].append((y, text, box[0][0]))
        else:
            lines.append([(y, text, box[0][0])])
    return "\n".join(" ".join(t for _, t, _ in sorted(row, key=lambda x: x[2])) for row in lines)


PRICE_RE = re.compile(r"^(?:NT\$|\$|＄)?\s*(\d{1,3}(?:,\d{3})+|\d{2,5})\s*(?:元|円)?$")
KANA_RE = re.compile(r"[\u3040-\u30ff]")   # Japanese: leave its kanji alone
LOOKALIKE = str.maketrans("IlZzOoSB|", "112200581")   # price-tag fonts: IZO = 120, Z88 = 288
LOOKALIKE_RE = re.compile(r"^[\dIlZzOoSB|]{3,5}$")
SIZE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:g|G|kg|KG|克|公克|ml|mL|ML|l|L|入|包|片|枚)")


def _rect(box: list) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def price_pairs(items: list) -> list[str]:
    """Shelf labels: pair each product line that states a size (25G, 500ml…) with the nearest bare price box,
    one-to-one by the gap between the boxes, so prices stop floating free of their products in the flat rows.
    Shelf-tag prices are printed bigger than the product line, which keeps small package numbers out."""
    names, prices = [], []
    for box, text in items:
        r = _rect(box)
        if (m := PRICE_RE.match(text)) and not SIZE_RE.search(text):
            prices.append((r, m.group(1)))
        elif LOOKALIKE_RE.match(text) and re.search(r"[\dZ]", text):
            prices.append((r, f"{text.translate(LOOKALIKE)}(辨識為「{text}」)"))
        elif SIZE_RE.search(text) and len(re.sub(r"[\d\W_]", "", text)) >= 3:
            names.append((r, text))
    cands = []
    for i, (nr, _) in enumerate(names):
        h = nr[3] - nr[1] or 1
        for j, (pr, _) in enumerate(prices):
            if pr[3] - pr[1] < h * 1.2:
                continue
            gap = math.hypot(max(nr[0] - pr[2], pr[0] - nr[2], 0), max(nr[1] - pr[3], pr[1] - nr[3], 0))
            if gap <= h * 6:
                cands.append((gap, i, j))
    used_n, used_p, out = set(), set(), {}
    for _, i, j in sorted(cands):
        if i not in used_n and j not in used_p:
            used_n.add(i)
            used_p.add(j)
            out[i] = prices[j][1]
    return [f"{names[i][1]} — {out[i]}" for i in sorted(out, key=lambda i: (names[i][0][1], names[i][0][0]))]


def _ocr_sync(img: Image.Image) -> str:
    return _rows(_ocr_boxes(img))


def _ocr_layout(img: Image.Image) -> tuple[str, list[str]]:
    items = _ocr_boxes(img)
    return _rows(items), price_pairs(items)


async def ocr(img: Image.Image) -> str:
    return await asyncio.to_thread(_ocr_sync, img)


async def caption_local(img: Image.Image, prompt: str = CAPTION_PROMPT, max_tokens: int = 500) -> str:
    m = common.cfg()
    body = {"model": m.get("vision_model", "qwen3.5:9b"), "stream": False, "think": False, "keep_alive": -1,
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
    text_task = asyncio.create_task(asyncio.to_thread(_ocr_layout, img))
    caption, err = "", None
    try:
        caption = await describe(img, path)
    except Exception as e:
        log.warning("image description failed (%s): %s", path, e)
        err, path = str(e)[:200], "cpu"
    found, pairs = await text_task
    parts = []
    if caption:
        parts.append(f"【圖片描述】\n{caption}")
    parts.append(f"【圖中文字】\n{found}" if found else "【圖中文字】(無)")
    if pairs:
        parts.append("【標價配對】(依位置推測,商品 — 標價)\n" + "\n".join(pairs))
    if path == "cpu":
        parts.append("(GPU 忙碌中,只做了文字辨識,沒有圖片描述。)" if not err else "(圖片描述失敗,只有文字辨識。)")
    meta = {"path": path, "degraded": path == "cpu", "width": img.width, "height": img.height}
    if err:
        meta["caption_error"] = err
    summary = re.sub(r"[*#`]+|\s+", " ", caption or found or "圖片").strip()[:120]
    return {"kind": "image", "text": "\n\n".join(parts), "summary": summary, "meta": meta}
