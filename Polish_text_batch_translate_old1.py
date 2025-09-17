# polish_text_batch_translate.py
# Paragraph-preserving batch translator for a large Polish .txt file.
# - Splits by PARAGRAPHS (blank lines), never inside a paragraph
# - Groups paragraphs into ~300-word chunks
# - Translates each chunk via OpenAI Responses API
# - Recombines translation preserving paragraph structure
# - Creates per-chunk Polish/English files + CSV manifest
#
# Requirements:
#   pip install openai python-dotenv
# Environment:
#   Set OPENAI_API_KEY in your environment or in a .env file next to this script.

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

# Chunking targets: whole paragraphs only; aim for ~300 words per chunk.
TARGET_WORDS_PER_CHUNK = 300
MAX_WORDS_PER_CHUNK = 360  # soft cap for grouping paragraphs

# Model fallback ladder — script tests these at startup and picks the first that works.
MODEL_PREFERENCE = [
    "gpt-5-2025-08-07",  # if your account has access
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

# System prompt: explicitly instruct the model to preserve paragraphing.
SYSTEM_PROMPT = """
You are an expert translator from Polish to English.

Rules:
- Translate faithfully and literally with document-like accuracy.
- Preserve the original PARAGRAPH structure exactly as received: do not merge or split paragraphs.
- Keep line breaks between paragraphs (blank line = paragraph break).
- Do not summarize, omit, or add content.
- Return only the English translation.
""".strip()

USER_PROMPT_PREFIX = (
    "Translate the following Polish text into English, literally and accurately. "
    "Preserve the original paragraph structure (blank lines):\n\n"
)

# Operational toggles
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
    """Pull a concise server error message from OpenAI exceptions (for 400s, etc.)."""
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
    Try each candidate with a tiny 'sanity ping' to catch 400s/misconfigs early.
    Returns the first model that succeeds.
    """
    test_text = "Próba. To jest bardzo krótki test tłumaczenia.\n\nDrugi akapit do sprawdzenia."
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
    Preserve paragraph breaks, but normalize whitespace inside paragraphs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines to at most 2 (keeps blank-line paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize spaces/tabs within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Trim whitespace per line
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs using blank lines (double newlines) as delimiters.
    Keeps paragraph structure exactly.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras

def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))

def chunk_paragraphs(paragraphs: List[str],
                     target_words: int = TARGET_WORDS_PER_CHUNK,
                     max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    """
    Group whole paragraphs into contiguous chunks totaling ~target_words,
    never breaking inside a paragraph.
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
            # close current chunk and start a new one with this paragraph
            chunks.append("\n\n".join(current).strip())
            current = [para]
            current_words = w

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks

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
    """
    Minimal, schema-safe call (plain string content per role) with backoff.
    """
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
            # Payload/model issue — show the server’s message and abort retries
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

    paragraphs = split_into_paragraphs(text)
    chunks = chunk_paragraphs(paragraphs, TARGET_WORDS_PER_CHUNK, MAX_WORDS_PER_CHUNK)

    total_words = count_words(text)
    logging.info(f"Total paragraphs: {len(paragraphs)} | Total words: {total_words}")
    logging.info(f"Planned chunks: {len(chunks)} (target ≈{TARGET_WORDS_PER_CHUNK} words each)")

    # Save Polish paragraphs grouped per chunk (for inspection/checkpoints)
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
                out_text = translate_with_retries(client, model_in_use, chunk) # type: ignore
            else:
                out_text = f"[DRY-RUN] {chunk}"
            # Optional tiny cleanup: collapse any 3+ newlines in model output to at most 2,
            # to prevent accidental extra blank lines while preserving paragraph breaks.
            out_text = re.sub(r"\n{3,}", "\n\n", out_text).strip()
            en_path.write_text(out_text, encoding="utf-8")
            combined.append(out_text)
        except Exception as e:
            ok = False
            err = str(e)
            logging.error(f"Chunk {i} failed: {err}")
            en_path.write_text(f"[ERROR translating chunk {i}]\n\n{chunk}", encoding="utf-8")

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

    # Recombine with double newlines between chunks (i.e., natural paragraph spacing)
    FINAL_EN_FILE.write_text("\n\n".join(combined).strip() + "\n", encoding="utf-8")
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
