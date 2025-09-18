# polish_text_batch_translate.py
# Paragraph-preserving translator WITHOUT extra blank lines between paragraphs.
# - Groups whole paragraphs (~300 words per chunk), never splits inside a paragraph
# - Translates with OpenAI Responses API (model fallback + sanity check)
# - Post-processes output to:
#     * collapse multi-blank-lines to a SINGLE newline between paragraphs,
#     * strip leading indents and trailing spaces per line
# - Recombines the final file with SINGLE newlines between paragraphs
#
# Setup:
#   pip install openai python-dotenv
#   Set OPENAI_API_KEY in env or .env

from __future__ import annotations
import os
import re
import time
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from contextlib import suppress

# Optional: .env support
with suppress(Exception):
    from dotenv import load_dotenv
    load_dotenv()

# ------------------------------ USER CONSTANTS ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "polish_input.txt"

OUT_DIR = BASE_DIR / "translation_output"
CHUNKS_PL_DIR = OUT_DIR / "chunks_pl"
CHUNKS_EN_DIR = OUT_DIR / "chunks_en"
FINAL_EN_FILE = OUT_DIR / "translated_full_en.txt"
MANIFEST_CSV = OUT_DIR / "manifest.csv"

# Target chunk size (whole paragraphs only)
TARGET_WORDS_PER_CHUNK = 300
MAX_WORDS_PER_CHUNK = 360  # soft cap

# IMPORTANT: We want EXACTLY ONE NEWLINE between paragraphs in the output.
PARAGRAPH_SEPARATOR = "\n"  # (was "\n\n" before)

# Model fallback ladder — first working model wins
MODEL_PREFERENCE = [
    "gpt-5-2025-08-07",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

'''
SYSTEM_PROMPT = """
You are a professional translator specializing in Polish-to-English translation for archival, historical, and documentary texts.
Your task is to produce literal and faithful translations of Polish text into English. Follow these rules:

1. Preserve Original Structure
   - Keep sentence order and paragraph breaks as in the source.
   - Do not combine or split sentences unless absolutely necessary for basic English readability.

2. Literal Accuracy over Style
   - Translate word-for-word as far as possible while keeping correct English grammar.
   - If the Polish is ungrammatical or fragmented, preserve this effect in English rather than smoothing it out.

3. Maintain Register and Tone
   - Preserve the formality, emotional tone, and historical “feel” of the text.
   - Do not modernize vocabulary or idioms unnecessarily.

4. Proper Names and Terms
   - Keep Polish names, places, and cultural terms untranslated (e.g., Babcia, kutia).
   - Provide a parenthetical note only if absolutely necessary for clarity.

5. Spelling and Punctuation
   - Reflect punctuation, capitalization, and stylistic quirks of the original as closely as possible.
   - Do not “correct” the original author’s style.

6. Ambiguities and Missing Words
   - If a passage is unclear, provide your best literal rendering and, if necessary, add an alternative in square brackets.

7. Output Requirement
   - Produce only the translation text.
   - Do not include commentary or explanations unless explicitly requested.
   - Preserve paragraph boundaries as received, but DO NOT insert blank empty lines.
   - Separate paragraphs with a SINGLE newline character only (no extra blank line).
   - Do not indent paragraphs; first character of each line should be non-space.
""".strip()

USER_PROMPT_PREFIX = (
    "Translate the following Polish text into English, literally and accurately. "
    "Keep exactly one newline between paragraphs (no blank empty lines) and no leading spaces:\n\n"
)
'''

SYSTEM_PROMPT = """
You are a professional translator specializing in Polish-to-English translation for historical and documentary texts.
Your task is to produce faithful translations of Polish text into English. Follow these rules:
- Keep sentence order and paragraph breaks as in the source.
- Preserve the formality, emotional tone, and historical “feel” of the text.
- Do not modernize vocabulary or idioms unnecessarily.
- Keep Polish names, places, and cultural terms untranslated (e.g., Babcia, kutia).
- Provide a parenthetical note only if absolutely necessary for clarity.
- Reflect punctuation, capitalization, and stylistic quirks of the original as closely as possible.
- Do not “correct” the original author’s style.
- If a passage is unclear, provide your best literal rendering and, if necessary, add an alternative in square brackets.
- Produce only the translation text.
- Do not include commentary or explanations unless explicitly requested.
- Preserve paragraph boundaries as received, but DO NOT insert blank empty lines.
- Separate paragraphs with a SINGLE newline character only (no extra blank line).
- Do not indent paragraphs; first character of each line should be non-space.
""".strip()

USER_PROMPT_PREFIX = (
    "Translate the following Polish text into English faithfully and accurately. "
    "Keep exactly one newline between paragraphs (no blank empty lines) and no leading spaces:\n\n"
)


TRANSLATE = True
PAUSE_BETWEEN_CALLS_S = 0.4
MAX_RETRIES = 6

# ------------------------------ LOGGING ----------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------------------ OPENAI CLIENT ----------------------------------

try:
    from openai import OpenAI
    from openai import APIStatusError, APIConnectionError, APIError, BadRequestError, RateLimitError
except Exception as e:
    raise SystemExit(
        "The 'openai' package is required. Install with: pip install openai\n"
        f"Import error: {e}"
    )

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set (env var or .env).")
    return OpenAI(api_key=api_key)

def extract_server_message(exc: Exception) -> str:
    with suppress(Exception):
        if hasattr(exc, "response") and exc.response is not None:
            data = exc.response
            if isinstance(data, dict):
                return data.get("error", {}).get("message") or str(data)
            with suppress(Exception):
                js = getattr(data, "json", None)
                if callable(js):
                    j = js()
                    return (j.get("error", {}) or {}).get("message") or str(j)
            with suppress(Exception):
                body = getattr(data, "body", None)
                if isinstance(body, dict):
                    return body.get("error", {}).get("message") or str(body)
    return str(exc)

def choose_working_model(client: OpenAI, candidates: List[str]) -> str:
    """
    Tiny sanity ping to pick a working model and avoid 400s at runtime.
    """
    test_text = "Próba. To jest krótki test.\nDrugi akapit bez pustej linii powyżej."
    for m in candidates:
        try:
            resp = client.responses.create(
                model=m,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_PREFIX + test_text},
                ],
            )
            text = getattr(resp, "output_text", None)
            if not text:
                out = getattr(resp, "output", None)
                if isinstance(out, list) and out:
                    c = getattr(out[0], "content", None) or (out[0].get("content") if isinstance(out[0], dict) else None)
                    if isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                                text = item.get("text") or item.get("value")
                                if text:
                                    break
            if text:
                logging.info(f"Model sanity check OK on: {m}")
                return m
            else:
                logging.warning(f"Model {m} responded without text; trying next.")
        except BadRequestError as e:
            logging.warning(f"BadRequest (400) on model '{m}': {extract_server_message(e)} — trying next.")
        except RateLimitError as e:
            logging.warning(f"Rate limited during sanity check on '{m}': {e}. Will still try this model.")
            return m
        except (APIStatusError, APIConnectionError, APIError, Exception) as e:
            logging.warning(f"Model '{m}' sanity check failed: {e}. Trying next.")
    raise SystemExit("No working model from MODEL_PREFERENCE. Adjust the list or check access.")

# ------------------------------ TEXT UTILITIES ---------------------------------

def _normalize_whitespace(text: str) -> str:
    """
    Preserve paragraph breaks, but avoid expanding blank lines.
    We do NOT add blank lines; we keep what's there.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Do NOT inflate blank lines; just collapse 3+ to at most 2 in the source
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize internal spaces/tabs within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Trim whitespace per line
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split into paragraphs by detecting BLANK LINE boundaries (two or more newlines).
    If your source uses single-newline paragraphs, they will be preserved within a paragraph block.
    """
    # Keep only "true" blank-line-separated paragraphs here:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts

def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))

def chunk_paragraphs(paragraphs: List[str],
                     target_words: int = TARGET_WORDS_PER_CHUNK,
                     max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    """
    Group whole paragraphs into contiguous chunks, never breaking inside a paragraph.
    Uses PARAGRAPH_SEPARATOR for joining inside a chunk (SINGLE newline).
    """
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in paragraphs:
        w = count_words(para)
        if not current:
            current = [para]
            current_words = w
            continue

        if current_words + w <= max_words:
            current.append(para)
            current_words += w
        else:
            chunks.append(PARAGRAPH_SEPARATOR.join(current).strip())
            current = [para]
            current_words = w

    if current:
        chunks.append(PARAGRAPH_SEPARATOR.join(current).strip())

    return chunks

# --------------------------- OUTPUT NORMALIZATION -------------------------------

def normalize_translated_text(s: str) -> str:
    """
    Ensure NO extra blank line before next paragraph:
      - Convert CRLF -> LF
      - Collapse 2+ newlines to exactly 1 newline
      - Remove leading indentation and trailing spaces on every line
      - Remove lines that consist only of spaces/tabs
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # NO blank empty lines between paragraphs:
    s = re.sub(r"\n{2,}", "\n", s)
    # Remove spaces on "empty" lines
    s = re.sub(r"(?m)^[ \t]+$", "", s)
    # Strip indentation at line starts and trailing spaces
    s = re.sub(r"(?m)^[ \t]+", "", s)
    s = re.sub(r"(?m)[ \t]+$", "", s)
    return s.strip()

# ------------------------------ TRANSLATION CORE -------------------------------

@dataclass
class ChunkResult:
    index: int
    pl_file: str
    en_file: str
    pl_words: int
    en_words: int
    seconds: float
    ok: bool
    error: Optional[str] = None

def translate_with_retries(client: OpenAI, model: str, polish_text: str) -> str:
    delay = 1.0
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_PREFIX + polish_text},
                ],
            )
            text = getattr(resp, "output_text", None)
            if not text:
                out = getattr(resp, "output", None)
                if isinstance(out, list) and out:
                    c = getattr(out[0], "content", None) or (out[0].get("content") if isinstance(out[0], dict) else None)
                    if isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                                text = item.get("text") or item.get("value")
                                if text:
                                    break
            if not text:
                raise RuntimeError("No text in response.")
            return text.strip()
        except BadRequestError as e:
            server_msg = extract_server_message(e)
            raise RuntimeError(f"400 Bad Request from server: {server_msg}") from e
        except RateLimitError as e:
            last_err = e
            time.sleep(delay)
            delay *= 2.0
        except (APIStatusError, APIConnectionError, APIError, Exception) as e:
            last_err = e
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"Translation failed after retries: {last_err}")

# ----------------------------------- MAIN --------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PL_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_EN_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing input file: {INPUT_FILE}")

    logging.info(f"Reading: {INPUT_FILE}")
    text = INPUT_FILE.read_text(encoding="utf-8")
    text = _normalize_whitespace(text)

    # Split & chunk by paragraphs (never break inside a paragraph)
    paragraphs = split_into_paragraphs(text)
    chunks = chunk_paragraphs(paragraphs, TARGET_WORDS_PER_CHUNK, MAX_WORDS_PER_CHUNK)

    total_words = count_words(text)
    logging.info(f"Total paragraphs: {len(paragraphs)} | Total words: {total_words}")
    logging.info(f"Planned chunks: {len(chunks)} (target ≈{TARGET_WORDS_PER_CHUNK} words each)")

    # Save staged Polish chunks (joined with SINGLE newline)
    for i, chunk in enumerate(chunks, 1):
        (CHUNKS_PL_DIR / f"chunk_{i:04d}.pl.txt").write_text(chunk, encoding="utf-8")

    client = get_openai_client() if TRANSLATE else None
    model_in_use = None
    if TRANSLATE:
        model_in_use = choose_working_model(client, MODEL_PREFERENCE)
        logging.info(f"Using model: {model_in_use}")

    results: List[ChunkResult] = []
    combined: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        pl_path = CHUNKS_PL_DIR / f"chunk_{i:04d}.pl.txt"
        en_path = CHUNKS_EN_DIR / f"chunk_{i:04d}.en.txt"
        pl_words = count_words(chunk)
        start = time.perf_counter()
        ok = True
        err = None
        out_text = ""
        try:
            if TRANSLATE:
                time.sleep(PAUSE_BETWEEN_CALLS_S)
                out_text = translate_with_retries(client, model_in_use, chunk)
            else:
                out_text = f"[DRY-RUN] {chunk}"
            # CRITICAL: remove any blank empty line before the next paragraph
            out_text = normalize_translated_text(out_text)
            en_path.write_text(out_text, encoding="utf-8")
            combined.append(out_text)
        except Exception as e:
            ok = False
            err = str(e)
            logging.error(f"Chunk {i} failed: {err}")
            en_path.write_text(f"[ERROR translating chunk {i}]\n{chunk}", encoding="utf-8")

        elapsed = time.perf_counter() - start
        results.append(ChunkResult(
            index=i,
            pl_file=pl_path.name,
            en_file=en_path.name,
            pl_words=pl_words,
            en_words=count_words(out_text),
            seconds=elapsed,
            ok=ok,
            error=err
        ))
        logging.info(f"Chunk {i}/{len(chunks)} | {pl_words}→{count_words(out_text)} words | {elapsed:.2f}s | ok={ok}")

    # Recombine using SINGLE newline between parts (no blank lines)
    FINAL_EN_FILE.write_text(PARAGRAPH_SEPARATOR.join(combined).strip() + "\n", encoding="utf-8")
    logging.info(f"Wrote combined translation: {FINAL_EN_FILE}")

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "pl_file", "en_file", "pl_words", "en_words", "seconds", "ok", "error"])
        for r in results:
            w.writerow([r.index, r.pl_file, r.en_file, r.pl_words, r.en_words, f"{r.seconds:.3f}", "yes" if r.ok else "no", r.error or ""])

    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    total_sec = sum(r.seconds for r in results)
    logging.info(f"Done: {ok_count}/{total} chunks OK in {total_sec:.1f}s "
                 f"(avg {total_sec/total:.2f}s/chunk).")

if __name__ == "__main__":
    main()
