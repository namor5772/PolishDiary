# Polish_handwriting_transcriber_v4.py
# Roman: single-file, no CLI args; uses OpenAI Responses API (gpt-5) without response_format.
# Input:  ./input_pages/Eugenia_pg###.png
# Output: ./output_transcripts/Eugenia_pg###.txt
# Logs per-file char count and timings; unreadables -> [**]; safe BICUBIC.

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageFilter, ImageOps

# --- OpenAI (Responses API) ---
from openai import OpenAI


# =========================
# ===== CONFIGURATION =====
# =========================

# Model alias is safer than dated build strings; change here if you want a pinned revision.
MODEL_NAME = "gpt-5"

# Base directory = folder containing this .py file.
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input_pages"
OUTPUT_DIR = BASE_DIR / "output_transcripts"

# Page range: start number and how many consecutive pages to attempt.
START_NUM = 134      # e.g., 23  -> starts at Eugenia_pg023.png
COUNT = 10           # e.g., 3   -> 023, 024, 025

# Optional: upscale factor before binarization (helps readability).
UPSCALE = 1.5       # 1.0 means no scaling

# Logging verbosity
LOG_LEVEL = logging.INFO


# =========================
# ====== PROMPTS ==========
# =========================

SYSTEM_PROMPT = (
    "Jesteś ekspertem w odczytywaniu polskich rękopisów.\n"
    "Twoim zadaniem jest WIERNE przepisanie treści do zwykłego tekstu.\n"
    "Wytyczne (stosuj dokładnie):\n"
    "1) Przepisuj DOKŁADNIE to, co widzisz (słowo po słowie); zachowaj oryginalną pisownię, błędy i idiomy.\n"
    "2) NIE naśladuj układu strony (kolumn, ramek); zapisuj tekst ciągły (naturalne akapity ok).\n"
    "3) Usuwaj WYŁĄCZNIE dzielenie wyrazów na końcu wersu (np. 'prze-\\nkazać' → 'przekazać').\n"
    "4) Jeśli fragment jest nieczytelny, zastąp go dokładnie ciągiem: [**]\n"
    "5) Odpowiadaj WYŁĄCZNIE surowym tekstem – BEZ komentarzy, objaśnień, nagłówków i oznaczeń kodu.\n"
    "6) Zachowaj polskie diakrytyki.\n"
)

USER_PROMPT = (
    "Przepisz wiernie polski rękopis widoczny powyżej.\n"
    "Pamiętaj: tylko czysty tekst; nie poprawiaj błędów; [**] dla nieczytelnych fragmentów; "
    "nie odwzorowuj układu strony; usuń sztuczne podziały wyrazów na końcach wierszy."
)


# =========================
# ====== UTILITIES ========
# =========================

# Pillow changed resample enums; detect safely.
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC  # Pillow ≥ 10
except Exception:  # Pillow < 10
    RESAMPLE_BICUBIC = Image.BICUBIC # type: ignore


@dataclass
class PageResult:
    page_num: int
    status: str            # 'ok' | 'missing' | 'error'
    chars: int
    bytes_utf8: int
    seconds_total: float
    seconds_preproc: float
    message: str = ""


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        logging.error("Input dir does not exist: %s", INPUT_DIR)
    if not OUTPUT_DIR.exists():
        logging.error("Output dir could not be created: %s", OUTPUT_DIR)


def load_image(path: Path) -> Image.Image:
    # Pillow sometimes leaves files open; use context when saving elsewhere.
    return Image.open(path)


def otsu_threshold_from_hist(hist: List[int]) -> int:
    """Compute Otsu threshold from a 256-bin histogram (no numpy)."""
    total = sum(hist)
    if total == 0:
        return 127
    sum_total = sum(i * hist[i] for i in range(256))
    sum_b = 0.0
    w_b = 0
    max_between = -1.0
    threshold = 127
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_between:
            max_between = between
            threshold = i
    return threshold


def preprocess_image(img: Image.Image, upscale: float = UPSCALE) -> Tuple[Image.Image, float]:
    """
    Lightweight, OCR-friendly preprocessing:
    - Grayscale
    - Optional BICUBIC upscale
    - Autocontrast + mild unsharp mask
    - Otsu binarization
    Returns (processed_image, elapsed_seconds)
    """
    t0 = time.perf_counter()

    # 1) Grayscale (ensures point threshold lambda sees ints, not tuples)
    g = img.convert("L")

    # 2) Optional upscale (helps faint strokes; safe BICUBIC)
    if upscale and upscale != 1.0:
        new_size = (max(1, int(g.width * upscale)), max(1, int(g.height * upscale)))
        g = g.resize(new_size, RESAMPLE_BICUBIC)

    # 3) Contrast and gentle sharpen
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.4, percent=150, threshold=2))

    # 4) Otsu binarization (histogram-based; no numpy)
    hist = g.histogram()
    thr = otsu_threshold_from_hist(hist)
    bw = g.point(lambda p, t=thr: 255 if p > t else 0) # type: ignore

    # Keep "L" mode (0..255) instead of "1" to avoid type issues downstream.
    bw = bw.convert("L")

    elapsed = time.perf_counter() - t0
    return bw, elapsed


def pil_image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def extract_output_text(resp) -> str:
    """
    Robustly pull plain text from Responses API result.
    Prefer `resp.output_text` (new SDK), else fall back to walking `resp.output`.
    """
    # New SDK convenience property:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Fallback: walk the raw structure (be tolerant to shape changes).
    try:
        out = []
        for item in getattr(resp, "output", []) or []:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    # Common shapes: {"type": "output_text", "text": "..."}
                    if isinstance(c, dict) and "text" in c:
                        out.append(c["text"])
        if out:
            return "\n".join(out)
    except Exception:
        pass

    # Absolute fallback: stringify (last resort).
    return str(resp)


def postprocess_transcript(text: str) -> str:
    """
    - Strip any code fences or lead-in labels the model might emit.
    - Merge hyphenated line-break splits: 'co-\nsię' -> 'cośię'
    - Normalize excessive whitespace.
    """
    s = text.strip()

    # Remove accidental code fences
    s = re.sub(r"^```(?:text|txt|markdown)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Remove common lead-in labels
    s = re.sub(r"^\s*(Transkrypcja|Transcription)\s*:\s*", "", s, flags=re.I)

    # Join hyphenated line breaks (be conservative)
    s = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", s, flags=re.UNICODE)

    # Squash excessive whitespace (keep paragraph breaks)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def call_openai_transcribe_image(client: OpenAI, model: str, data_uri: str) -> str:
    """
    Minimal Responses API call (no response_format, no extra knobs).
    Sends system + user messages, with the preprocessed image first.
    """
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_uri},
                    {"type": "input_text", "text": USER_PROMPT},
                ],
            }, # type: ignore
        ],
        # Intentionally no response_format to avoid TypeError in your environment.
    )
    return extract_output_text(resp)


def write_text(out_path: Path, text: str) -> Tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    with out_path.open("wb") as f:
        f.write(data)
    return len(text), len(data)


def process_one_page(
    client: OpenAI, page_num: int
) -> PageResult:
    in_name = f"Eugenia_pg{page_num:03d}.png"
    out_name = f"Eugenia_pg{page_num:03d}.txt"
    in_path = INPUT_DIR / in_name
    out_path = OUTPUT_DIR / out_name

    if not in_path.exists():
        msg = f"Missing page: {in_path}"
        logging.warning(msg)
        return PageResult(page_num, "missing", 0, 0, 0.0, 0.0, msg)

    t0 = time.perf_counter()
    try:
        img = load_image(in_path)
        pre_img, t_pre = preprocess_image(img, UPSCALE)
        data_uri = pil_image_to_data_uri(pre_img)

        text_raw = call_openai_transcribe_image(OpenAI(), MODEL_NAME, data_uri)
        text = postprocess_transcript(text_raw)

        chars, bytes_utf8 = write_text(out_path, text)
        t_total = time.perf_counter() - t0

        logging.info(
            "OK  %s -> %s | chars=%d bytes=%d | total=%.2fs preproc=%.2fs",
            in_name, out_name, chars, bytes_utf8, t_total, t_pre
        )
        return PageResult(page_num, "ok", chars, bytes_utf8, t_total, t_pre)

    except Exception as e:
        t_total = time.perf_counter() - t0
        logging.exception("ERROR processing %s: %s", in_name, e)
        return PageResult(page_num, "error", 0, 0, t_total, 0.0, str(e))


def process_range(start_num: int, count: int) -> List[PageResult]:
    results: List[PageResult] = []
    for n in range(start_num, start_num + count):
        results.append(process_one_page(OpenAI(), n))
    return results


def print_summary(results: List[PageResult]) -> None:
    ok = sum(1 for r in results if r.status == "ok")
    missing = sum(1 for r in results if r.status == "missing")
    errors = sum(1 for r in results if r.status == "error")

    logging.info(
        "Summary: ok=%d missing=%d error=%d", ok, missing, errors
    )

    # Pretty table to console
    header = f"{'page':>5} | {'status':7} | {'chars':>7} | {'bytes':>7} | {'total[s]':>8} | {'preproc[s]':>9}"
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)
    for r in results:
        print(f"{r.page_num:5d} | {r.status:7} | {r.chars:7d} | {r.bytes_utf8:7d} | {r.seconds_total:8.2f} | {r.seconds_preproc:9.2f}")
    print(sep)


def main() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Base dir:   %s", BASE_DIR)
    logging.info("Input dir:  %s", INPUT_DIR)
    logging.info("Output dir: %s", OUTPUT_DIR)
    logging.info("Model:      %s", MODEL_NAME)
    logging.info("Processing pages: %d .. %d (count=%d)", START_NUM, START_NUM + COUNT - 1, COUNT)

    ensure_dirs()

    # Basic sanity: warn if no images matching pattern exist at/after START_NUM
    sample = INPUT_DIR / f"Eugenia_pg{START_NUM:03d}.png"
    if not sample.exists():
        logging.warning("First requested page does not exist: %s", sample)

    results = process_range(START_NUM, COUNT)
    print_summary(results)


if __name__ == "__main__":
    main()
