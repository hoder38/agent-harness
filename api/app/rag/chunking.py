"""Markdown → heading-aware chunks (§10) and the keyword tokens used for the tsvector column.

Pure functions, no I/O. Token counts are estimates (CJK char ≈ 1 token, Latin word ≈ 1.3).
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

VERSION = "2"  # bump when chunk boundaries change: documents are re-indexed

_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_WORD = re.compile("[A-Za-z0-9\u00c0-\u024f]+")
_TOKEN = re.compile(f"[A-Za-z0-9\u00c0-\u024f]+|[{_CJK}]+")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.S)
_SENTENCE = re.compile(r"(?<=[。！？!?.;；])\s*")

# English stop words are left out of the keyword index (they match everything).
_STOP = frozenset("a an and are as at be by for from has have how i in is it its of on or that the this to was "
                  "were what when where which who why will with you your do does can my me".split())


def est_tokens(text: str) -> int:
    cjk = sum(len(m) for m in _CJK_RUN.findall(text))
    words = len(_WORD.findall(text))
    return cjk + int(words * 1.3) + 1


def kw_tokens(text: str) -> list[str]:
    """Lower-cased Latin words (no stop words) plus CJK character bigrams (unigram for a lone character).

    Postgres has no Chinese word segmenter, so CJK text is indexed as overlapping bigrams; the query side uses
    the same function, so 「無糖綠茶」 matches 「綠茶」.
    """
    out: list[str] = []
    for m in _TOKEN.finditer(text):
        t = m.group(0)
        if _CJK_RUN.fullmatch(t):
            out += [t] if len(t) == 1 else [t[i:i + 2] for i in range(len(t) - 1)]
        else:
            t = t.lower()
            if t not in _STOP and (len(t) > 1 or t.isdigit()):
                out.append(t)
    return out


def tsquery_text(query: str) -> str | None:
    """OR-query of the keyword tokens, safe for to_tsquery('simple', …) (tokens are letters/digits/CJK only)."""
    toks = list(dict.fromkeys(kw_tokens(query)))[:64]
    return " | ".join(toks) if toks else None


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[m.end():]


@dataclass
class Section:
    path: list[str]                 # heading path, e.g. ["Setup", "Docker"]
    paras: list[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return " › ".join(self.path)


@dataclass
class Chunk:
    heading: str
    text: str

    @property
    def hash(self) -> str:
        return sha256(self.heading + "\n" + self.text)


def sections(body: str) -> list[Section]:
    """Split markdown into sections at headings; '#' inside code fences is not a heading.
    Paragraphs are blank-line separated; a fenced code block is kept as one paragraph."""
    out = [Section([])]
    stack: list[tuple[int, str]] = []
    para: list[str] = []
    fence = False

    def flush() -> None:
        if para and "".join(para).strip():
            out[-1].paras.append("\n".join(para).strip("\n"))
        para.clear()

    for line in body.splitlines():
        if _FENCE.match(line):
            if not fence:
                flush()
            para.append(line)
            fence = not fence
            if not fence:
                flush()
            continue
        if fence:
            para.append(line)
            continue
        h = _HEADING.match(line)
        if h:
            flush()
            level = len(h.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, h.group(2).strip())]
            out.append(Section([t for _, t in stack]))
            continue
        if not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    return [s for s in out if s.paras]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """A paragraph longer than max_tokens: split at sentence ends, then by characters."""
    parts, cur = [], ""
    for sent in _SENTENCE.split(text):
        if cur and est_tokens(cur + sent) > max_tokens:
            parts.append(cur)
            cur = ""
        while est_tokens(sent) > max_tokens:  # one huge "sentence" (tables, minified text)
            cut = max(1, int(len(sent) * max_tokens / est_tokens(sent)))
            while cut > 1 and est_tokens(sent[:cut]) > max_tokens:
                cut = max(1, int(cut * 0.9))
            space = sent.rfind(" ", cut // 2, cut)
            cut = space + 1 if space > 0 else cut
            parts.append(sent[:cut])
            sent = sent[cut:]
        cur += sent
    if cur.strip():
        parts.append(cur)
    return parts


def chunk_markdown(body: str, *, min_tokens: int = 400, max_tokens: int = 800, overlap: float = 0.15) -> list[Chunk]:
    """Heading-aware chunks of ≤ max_tokens.

    - Each section is chunked on its own, so a citation's heading is exact (small sections stay small).
    - A tiny section (< min_tokens / 4) directly followed by its own subsection (an intro before "### 9a") is merged
      into that subsection; its heading line is kept in the text.
    - A long section is packed paragraph by paragraph; the next chunk repeats the trailing paragraphs that fit in
      `overlap` × max_tokens.
    """
    merged: list[Section] = []
    carry: list[str] = []
    secs = sections(body)
    for i, s in enumerate(secs):
        paras = carry + s.paras
        carry = []
        nxt = secs[i + 1] if i + 1 < len(secs) else None
        if (nxt and len(nxt.path) > len(s.path) and nxt.path[:len(s.path)] == s.path
                and est_tokens("\n\n".join(paras)) < min_tokens // 4):
            carry = ([s.path[-1]] if s.path else []) + paras
            continue
        merged.append(Section(s.path, paras))

    chunks: list[Chunk] = []
    ov_budget = int(max_tokens * overlap)
    for s in merged:
        paras: list[str] = []
        for p in s.paras:
            paras += _hard_split(p, max_tokens) if est_tokens(p) > max_tokens else [p]
        cur: list[str] = []
        fresh = 0  # paragraphs in `cur` that are not overlap
        for p in paras:
            if cur and fresh and est_tokens("\n\n".join(cur + [p])) > max_tokens:
                chunks.append(Chunk(s.heading, "\n\n".join(cur)))
                tail: list[str] = []
                for q in reversed(cur):
                    if est_tokens("\n\n".join([q] + tail)) > ov_budget:
                        break
                    tail.insert(0, q)
                if est_tokens("\n\n".join(tail + [p])) > max_tokens:
                    tail = []
                cur, fresh = tail, 0
            cur.append(p)
            fresh += 1
        if fresh:
            chunks.append(Chunk(s.heading, "\n\n".join(cur)))
    return [c for c in chunks if c.text.strip()]


def doc_title(meta: dict[str, Any], body: str, fallback: str) -> str:
    if isinstance(meta.get("title"), str) and meta["title"].strip():
        return meta["title"].strip()
    for line in body.splitlines():
        h = _HEADING.match(line)
        if h and len(h.group(1)) == 1:
            return h.group(2).strip()
    return fallback
