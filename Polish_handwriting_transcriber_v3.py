#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polish_handwriting_transcriber_v4
---------------------------------
Runs from its own directory, expects two subdirs:
  - input_pages/         (Eugenia_pg###.png)
  - output_transcripts/  (Eugenia_pg###.txt)

• Processes a sequential range of pages (hardcoded START_NUM & NUM_FILES).
• Preprocessing:
    - Prefer OpenCV if available (CLAHE, denoise, unsharp, adaptive thresh).
    - Otherwise Pillow-only fallback (autocontrast, median denoise, unsharp,
      heuristic binarization).
• Calls OpenAI Responses API (GPT-5) with image input for literal Polish transcription.
• Preserves spelling/idioms; does NOT mimic layout; removes hyphenation only across line breaks.
• Logs, for each processed file: character count and elapsed time (seconds).
• No command-line arguments.

Quick start (once):
-------------------
pip install --upgrade openai pillow numpy
# Optional (better cleanup):
# pip install --upgrade opencv-python-headless

Environment:
------------
Set OPENAI_API_KEY in your environment before running.

Edit the two numbers below (START_NUM, NUM_FILES), then run:
python Polish_handwriting_transcriber_v4.py
"""

from __future__ import annotations

import os
import io
import time
import base64
import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Robust BICUBIC resolver for different Pillow versions
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC  # Pillow >= 9.1
except Exception:  # pragma: no cover
    RESAMPLE_BICUBIC = Image.BICUBIC            # type: ignore # Older Pillow

# Try OpenCV, but don’t require it.
HAS_CV2 = True
try:
    import cv2  # type: ignore
except Exception:
    HAS_CV2 = False

# OpenAI SDK (Responses API)
from openai import OpenAI

# ------------------------------
# Hardcoded configuration block
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_pages"
OUTPUT_DIR = BASE_DIR / "output_transcripts"

START_NUM: int = 118    # e.g., 118 -> starts with Eugenia_pg118.png
NUM_FILES: int = 2      # process this many pages starting at START_NUM

OPENAI_MODEL: str = "gpt-5"

# Preprocessing knobs
UPSCALE_FACTOR: float = 1.35
APPLY_CLAHE: bool = True           # OpenCV path only
APPLY_DENOISE: bool = True
APPLY_UNSHARP: bool = True
APPLY_ADAPTIVE_THRESH: bool = True # OpenCV: adaptive; Pillow: heuristic

SKIP_MISSING: bool = True

# Optional: save preprocessed images alongside outputs for debugging
DEBUG_SAVE_PREPROC: bool = False

# ------------------------------
# Prompts (strict & minimal)
# ------------------------------

SYSTEM_PROMPT = (
    "Jesteś ekspertem w wiernym przepisywaniu polskich rękopisów.\n"
    "Zadanie: przepisz DOKŁADNIE treść ze skanu.\n"
    "Zasady:\n"
    "1) Przepisuj słowo w słowo, zachowując oryginalną pisownię (błędy i regionalizmy też).\n"
    "2) Nie naśladuj układu strony (kolumn, łamań); tekst ma być ciągły.\n"
    "3) Usuwaj tylko sztuczne dzielenie wyrazów na końcu wersu (np. 'prze-\\nkazać' → 'przekazać').\n"
    "4) Używaj polskich znaków diakrytycznych.\n"
    "5) Jeśli fragment jest nieczytelny, wstaw [**].\n"
    "6) Odpowiadaj WYŁĄCZNIE pełnym tekstem – bez komentarzy, bez nagłówków."
)

USER_BRIEF = "Przepisz ten polski rękopis zgodnie z zasadami. Zwróć wyłącznie tekst przepisu."

# ------------------------------
# Helpers
# ------------------------------

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {INPUT_DIR}")

def page_png_name(n: int) -> str:
    return f"Eugenia_pg{n:03d}.png"

def img_to_data_uri(pil_img: Image.Image) -> str:
    """Encode PIL image as PNG data URI (string) for the Responses API."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

# ------------------------------
# Preprocessing (OpenCV path)
# ------------------------------

def _preprocess_opencv(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if UPSCALE_FACTOR and UPSCALE_FACTOR > 1.0:
        new_w = int(gray.shape[1] * UPSCALE_FACTOR)
        new_h = int(gray.shape[0] * UPSCALE_FACTOR)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if APPLY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if APPLY_DENOISE:
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    if APPLY_UNSHARP:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    if APPLY_ADAPTIVE_THRESH:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )

    return Image.fromarray(gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

# ------------------------------
# Preprocessing (Pillow fallback)
# ------------------------------

def _preprocess_pillow(pil_img: Image.Image) -> Image.Image:
    # Grayscale
    img = pil_img.convert("L")

    # Optional upscale (robust BICUBIC for all Pillow versions)
    if UPSCALE_FACTOR and UPSCALE_FACTOR > 1.0:
        w, h = img.size
        img = img.resize((int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)), resample=RESAMPLE_BICUBIC)

    # Contrast / normalization
    img = ImageOps.autocontrast(img)

    # Mild denoise
    if APPLY_DENOISE:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    # Unsharp
    if APPLY_UNSHARP:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))

    # Heuristic binarization (robust across pages)
    if APPLY_ADAPTIVE_THRESH:
        arr = np.asarray(img, dtype=np.uint8)
        thr = int(np.percentile(arr, 60))  # 60% percentile splits ink/paper reasonably well
        bw = (arr < thr).astype(np.uint8) * 255
        img = Image.fromarray(bw, mode="L")

    return img

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    if HAS_CV2:
        return _preprocess_opencv(pil_img)
    else:
        logging.warning("OpenCV not detected; using Pillow-only preprocessing fallback.")
        return _preprocess_pillow(pil_img)

# ------------------------------
# Text post-processing
# ------------------------------

def remove_linebreak_hyphenation(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("-\n", "")  # join hyphenated line breaks
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t.strip()

# ------------------------------
# Response text extraction
# ------------------------------

def extract_output_text(resp: Any) -> str:
    # Prefer convenience accessor if present
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Fallbacks (SDKs differ slightly in shape)
    try:
        data = resp.model_dump() if hasattr(resp, "model_dump") else getattr(resp, "__dict__", {}) or {}
    except Exception:
        data = {}

    try:
        out = data.get("output") or []
        parts = []
        for item in out:
            content = item.get("content") or []
            for c in content:
                if c.get("type") in ("output_text", "text") and "text" in c:
                    parts.append(c["text"])
                elif "text" in c:
                    parts.append(c["text"])
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass

    for key in ("content", "message", "data"):
        if key in data and isinstance(data[key], str):
            return data[key].strip()

    return (txt or "").strip() if isinstance(txt, str) else ""

# ------------------------------
# OpenAI call
# ------------------------------

def call_openai_transcribe_image(client: OpenAI, model: str, data_uri: str) -> str:
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,   # use top-level instructions for behavior
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_BRIEF},
                    {"type": "input_image", "image_url": data_uri},
                ],
            } # type: ignore
        ],
        # NOTE: do NOT pass response_format here; text is available via output_text
    )
    text = extract_output_text(resp)
    return remove_linebreak_hyphenation(text)

# ------------------------------
# Main
# ------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {INPUT_DIR}")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it once (e.g., setx OPENAI_API_KEY \"sk-...\" on Windows)."
        )

    client = OpenAI()

    start = START_NUM
    end_exclusive = START_NUM + NUM_FILES
    logging.info("Base dir: %s", BASE_DIR)
    logging.info("Input dir: %s", INPUT_DIR)
    logging.info("Output dir: %s", OUTPUT_DIR)
    logging.info("Processing pages: %03d .. %03d (count=%d)", start, end_exclusive - 1, NUM_FILES)

    total_processed = 0
    total_chars = 0
    total_seconds = 0.0

    for n in range(start, end_exclusive):
        png_name = page_png_name(n)
        in_path = INPUT_DIR / png_name
        out_path = OUTPUT_DIR / png_name.replace(".png", ".txt")

        if not in_path.exists():
            msg = f"[WARN] Missing page: {in_path}"
            if SKIP_MISSING:
                logging.warning(msg)
                continue
            else:
                raise FileNotFoundError(msg)

        t0 = time.perf_counter()
        logging.info("Reading %s", in_path)
        try:
            pil_img = Image.open(in_path).convert("RGB")
        except Exception as e:
            logging.error("Failed to open image %s: %s", in_path, e)
            continue

        try:
            pre = preprocess_for_ocr(pil_img)
        except Exception as e:
            logging.error("Preprocessing failed for %s: %s", in_path, e)
            pre = pil_img  # fail safe: send original

        if DEBUG_SAVE_PREPROC:
            try:
                dbg = out_path.with_suffix(".preproc.png")
                pre.save(dbg)
                logging.info("Saved preprocessed image: %s", dbg)
            except Exception as e:
                logging.warning("Failed to save preprocessed image for %s: %s", in_path, e)

        try:
            data_uri = img_to_data_uri(pre)
        except Exception as e:
            logging.error("Data URI encoding failed for %s: %s", in_path, e)
            continue

        try:
            text = call_openai_transcribe_image(client, OPENAI_MODEL, data_uri)
        except Exception as e:
            logging.error("OpenAI transcription failed for %s: %s", in_path, e)
            continue

        try:
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            logging.error("Failed to write %s: %s", out_path, e)
            continue

        # Stats
        elapsed = time.perf_counter() - t0
        chars = len(text)
        bytes_utf8 = len(text.encode("utf-8"))
        total_processed += 1
        total_chars += chars
        total_seconds += elapsed

        logging.info("Wrote %s | chars=%d | utf8_bytes=%d | elapsed=%.2fs",
                     out_path.name, chars, bytes_utf8, elapsed)

    # Summary
    if total_processed > 0:
        avg_time = total_seconds / total_processed
        avg_chars = total_chars // total_processed
        logging.info("Summary: files=%d | total_chars=%d | avg_chars=%d | total_time=%.2fs | avg_time=%.2fs",
                     total_processed, total_chars, avg_chars, total_seconds, avg_time)
    else:
        logging.info("No files processed.")

if __name__ == "__main__":
    main()
