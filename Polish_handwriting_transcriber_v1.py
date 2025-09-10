#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polish_handwriting_transcriber_v1

Folder layout (script's directory is the base directory):
  ./Polish_handwriting_transcriber_v1.py
  ./input_pages/          # PNGs named Eugenia_pg###.png (e.g., Eugenia_pg023.png)
  ./output_transcripts/   # Generated TXT files (e.g., Eugenia_pg023.txt)

What it does:
- Finds input images matching ^Eugenia_pg(\d{3})\.png$ in ./input_pages
- Processes images starting from START_AT_NUM (e.g., 23 → *_pg023.png),
  and continues for the next MAX_FILES files (set MAX_FILES=None for “all”).
- Calls OpenAI Responses API (gpt-5) with an image+instruction prompt
- Saves a literal Polish transcription to ./output_transcripts/Eugenia_pg###.txt
- Preserves spelling/idioms; does NOT mimic page layout
- Joins hyphenated line-break splits (e.g., "prze-\\nkazać" → "przekazać")

Dependencies: `pip install openai`
Auth: set environment variable OPENAI_API_KEY.
"""

from __future__ import annotations

import base64
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# --- Configuration (hardcoded; no CLI args) -----------------------------------

MODEL_ID = "gpt-5"  # Generic alias, avoids pinning to a specific dated variant.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 4.0  # exponential backoff base
OVERWRITE_EXISTING = False    # True → re-transcribe even if .txt exists

# NEW: starting point and count controls
START_AT_NUM: int = 128        # e.g., 23 → start from *_pg023.png
MAX_FILES: Optional[int] = 6  # e.g., 10 → process next 10; None → process all from start

# If you *really* want to hardcode a key (not recommended), put it here.
# Otherwise leave as None and use the OPENAI_API_KEY environment variable.
HARDCODED_OPENAI_API_KEY: Optional[str] = None

# System/user prompts in Polish to keep the model focused on literal transcription.
SYSTEM_PROMPT = (
    "Jesteś ekspertem w odczytywaniu polskich rękopisów (pismo odręczne). "
    "Transkrybujesz tekst możliwie LITERALNIE i w języku polskim."
)

USER_INSTRUCTIONS = (
    "ZADANIE: Przepisz dokładnie treść z obrazu strony, słowo po słowie.\n"
    "ZASADY (przestrzegaj wszystkich):\n"
    "1) Zachowaj ORYGINALNĄ pisownię (w tym błędy) i lokalne idiomy.\n"
    "2) Zachowaj polskie znaki diakrytyczne.\n"
    "3) Nie odwzorowuj układu strony (nie rób kolumn, łamań, nagłówków, ramek itp.).\n"
    "4) Jeżeli na końcu wersu jest podział wyrazu z łącznikiem (np. 'prze-\\nkazać'), "
    "   USUŃ łącznik i złącz wyraz (→ 'przekazać').\n"
    "5) Fragmenty nieczytelne oznaczaj jako [***].\n"
    "6) Zwróć wyłącznie sam przepisany tekst, bez komentarzy.\n"
)

# --- OpenAI SDK ---------------------------------------------------------------

try:
    from openai import OpenAI
    # APIStatusError is available in recent SDKs; if missing, we catch generic Exception below.
    try:
        from openai import APIStatusError  # type: ignore
    except Exception:  # pragma: no cover
        APIStatusError = Exception  # type: ignore
except Exception:
    print("[ERROR] The 'openai' package is not installed. Run:  pip install openai", file=sys.stderr)
    sys.exit(1)

# --- Utilities ----------------------------------------------------------------

def base_dir() -> Path:
    """Return the directory where this script resides."""
    return Path(__file__).resolve().parent


def ensure_dirs() -> Tuple[Path, Path]:
    """Ensure base subfolders exist; create output if needed."""
    b = base_dir()
    input_dir = b / "input_pages"
    output_dir = b / "output_transcripts"
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Missing required directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(exist_ok=True)
    return input_dir, output_dir


# Accept 'Eugenia' prefix
PNG_NAME_RE = re.compile(r"^Eugenia_pg(\d{3})\.png$", re.IGNORECASE)


def extract_index_from_name(name: str) -> Optional[int]:
    m = PNG_NAME_RE.match(name)
    return int(m.group(1)) if m else None


def list_input_images(input_dir: Path) -> List[Path]:
    """Return sorted list of PNGs matching the Eu[g|r]_pg###.png pattern (case-insensitive)."""
    candidates = [p for p in input_dir.iterdir() if p.is_file() and PNG_NAME_RE.match(p.name)]

    def key_func(p: Path) -> int:
        idx = extract_index_from_name(p.name)
        return idx if idx is not None else 0

    return sorted(candidates, key=key_func)


def file_to_data_uri_png(path: Path) -> str:
    """Encode a PNG as a data URI string compatible with image_url for Responses API."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def dehyphenate_linebreaks(text: str) -> str:
    """
    Join hyphenated line-break splits without altering normal hyphens.
    Example: 'prze-\\n  kazać' → 'przekazać'
    """
    return re.sub(r"-\s*\n\s*", "", text)


def extract_text_from_response(resp) -> str:
    """
    Robustly obtain plain text from a Responses API result.
    """
    # 1) SDK convenience (may not exist depending on SDK version)
    try:
        ot = getattr(resp, "output_text", None)
        if isinstance(ot, str) and ot.strip():
            return ot
    except Exception:
        pass

    # 2/3) Structured walk
    try:
        data = resp.model_dump()  # pydantic model -> dict
    except Exception:
        return str(resp)

    pieces: List[str] = []

    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                ctype = c.get("type")
                if ctype == "output_text" and "text" in c:
                    pieces.append(c["text"])
                elif "text" in c:
                    pieces.append(c["text"])

    if not pieces:
        maybe = data.get("response", {}).get("output_text")
        if isinstance(maybe, str) and maybe.strip():
            pieces.append(maybe)

    return "\n".join(pieces).strip()


@dataclass
class TranscriptionResult:
    page_stem: str
    ok: bool
    message: str


def call_openai_transcribe_image(client: OpenAI, data_uri: str) -> str:
    """
    Minimal Responses API call: system + (instructions + image).
    """
    resp = client.responses.create(
        model=MODEL_ID,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_INSTRUCTIONS},
                    {"type": "input_image", "image_url": data_uri},
                ],
            }, # type: ignore
        ],
        # Keep parameters minimal; we rely on plain text output.
        # If you ever need to enforce it strictly:
        # response_format={"type": "text"},
    )
    text = extract_text_from_response(resp)
    if not text:
        raise RuntimeError("Empty transcription from API.")
    text = dehyphenate_linebreaks(text)
    return text


def transcribe_one(client: OpenAI, img_path: Path, out_dir: Path) -> TranscriptionResult:
    stem = img_path.stem  # e.g., "Eugenia_pg023"
    out_path = out_dir / f"{stem}.txt"

    if out_path.exists() and not OVERWRITE_EXISTING:
        return TranscriptionResult(stem, True, "exists (skipped)")

    data_uri = file_to_data_uri_png(img_path)

    delay = RETRY_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = call_openai_transcribe_image(client, data_uri)
            out_path.write_text(text, encoding="utf-8")
            return TranscriptionResult(stem, True, f"ok ({len(text)} chars)")
        except APIStatusError as e:  # type: ignore
            code = getattr(e, "status_code", "unknown")
            if attempt < MAX_RETRIES and code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            return TranscriptionResult(stem, False, f"API error {code}: {e}")
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
                continue
            return TranscriptionResult(stem, False, f"ERROR: {e}")

    return TranscriptionResult(stem, False, "unexpected fallthrough")


def select_window(files: List[Path], start_at: int, max_files: Optional[int]) -> List[Path]:
    """
    From the already-sorted list of matching files, pick those whose numeric index
    is >= start_at. If max_files is not None, take only the first N of those.
    Missing indices (gaps in numbering) are naturally skipped.
    """
    filtered: List[Path] = []
    for p in files:
        idx = extract_index_from_name(p.name)
        if idx is None:
            continue
        if idx >= start_at:
            filtered.append(p)
    if max_files is not None:
        filtered = filtered[:max_files]
    return filtered


def main() -> None:
    # Basic sanity on configuration
    if START_AT_NUM < 0 or START_AT_NUM > 999:
        print(f"[ERROR] START_AT_NUM must be in [0, 999], got {START_AT_NUM}", file=sys.stderr)
        sys.exit(1)
    if MAX_FILES is not None and MAX_FILES <= 0:
        print(f"[ERROR] MAX_FILES must be positive or None, got {MAX_FILES}", file=sys.stderr)
        sys.exit(1)

    input_dir, output_dir = ensure_dirs()
    all_files = list_input_images(input_dir)
    if not all_files:
        print(f"[ERROR] No input images found in {input_dir} matching Eugenia_pg###.png", file=sys.stderr)
        sys.exit(1)

    # Apply the start/count window
    window = select_window(all_files, START_AT_NUM, MAX_FILES)
    if not window:
        print(f"[ERROR] No files at or beyond index {START_AT_NUM:03d} found.", file=sys.stderr)
        sys.exit(1)

    api_key = HARDCODED_OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY is not set and HARDCODED_OPENAI_API_KEY is None.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    min_idx = extract_index_from_name(window[0].name) or START_AT_NUM
    max_idx = extract_index_from_name(window[-1].name) or START_AT_NUM

    print(f"[info] Base dir: {base_dir()}")
    print(f"[info] Model: {MODEL_ID}")
    print(f"[info] Found {len(all_files)} matching image(s) in {input_dir}")
    if MAX_FILES is None:
        print(f"[info] Processing from index {START_AT_NUM:03d} through available end (selected {len(window)} file(s): {min_idx:03d}..{max_idx:03d})")
    else:
        print(f"[info] Processing from index {START_AT_NUM:03d}, up to {MAX_FILES} file(s) (selected {len(window)}: {min_idx:03d}..{max_idx:03d})")
    print(f"[info] Writing outputs to {output_dir}")
    print(f"[info] Overwrite existing: {OVERWRITE_EXISTING}")

    ok_count = 0
    for i, img in enumerate(window, start=1):
        res = transcribe_one(client, img, output_dir)
        status = "OK" if res.ok else "FAIL"
        print(f"[{i:03d}/{len(window):03d}] {img.name:>14}  -> {res.page_stem}.txt  [{status}]  {res.message}")
        if res.ok:
            ok_count += 1

    print(f"[done] {ok_count}/{len(window)} page(s) transcribed.")


if __name__ == "__main__":
    main()
