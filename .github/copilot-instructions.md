# Copilot Instructions for PolishDiary

Purpose: Transcribe scanned Polish handwritten diary pages into plain text using OpenAI Responses API (model alias: `gpt-5`). Single-file script; no CLI args; all config is constants at top of the file.

Architecture & data flow
- Main entry: `Polish_handwriting_transcriber_v4.py` (runs directly).
- Input images: `input_pages/Eugenia_pg###.png` → Output text: `output_transcripts/Eugenia_pg###.txt` (exact 3 digits).
- Preprocessing pipeline: grayscale → optional BICUBIC upscale → autocontrast → UnsharpMask → Otsu binarization (pure-Python; no NumPy).
- OpenAI call: `from openai import OpenAI`; Responses API with one system prompt + one user message, sending the page as a data URI (`input_image`) followed by `input_text`.
- Postprocess: strip code fences/labels; join hyphenated line breaks only; normalize whitespace; keep paragraphs.
- Logging: per-page metrics (chars, UTF‑8 bytes, total s, preproc s) and a console summary table.

Developer workflow (VS Code)
- Install deps: run task “Install requirements” or `pip install -r requirements.txt`.
- Set API key (Windows PowerShell): `setx OPENAI_API_KEY "sk-..."` then open a new terminal.
- Place files named exactly `Eugenia_pg023.png` etc. Note: files with a trailing underscore (e.g., `Eugenia_pg001_.png`) will be skipped by the strict matcher—rename to `Eugenia_pg001.png`.
- Configure in code (top constants): `START_NUM`, `COUNT`, `MODEL_NAME`, `UPSCALE`, `LOG_LEVEL`.
- Run: `python Polish_handwriting_transcriber_v4.py`.

Project-specific conventions
- Literal transcription: preserve original spelling/idioms; do not mimic layout; unreadable = `[**]` exactly; Polish diacritics preserved.
- Do not add `response_format` to Responses API calls (avoids SDK errors). The extractor prefers `resp.output_text`, with a tolerant fallback that walks `resp.output`.
- Pillow compatibility: BICUBIC enum handled via try/except for Pillow ≥10 vs older; Otsu done on `L` mode with a safe `point(lambda p, t=thr: 255 if p > t else 0)`.
- Missing inputs are warnings, not fatal; script creates `output_transcripts/` if needed.

Integration points & examples
- Exact mapping: `input_pages/Eugenia_pg023.png` → `output_transcripts/Eugenia_pg023.txt`.
- Messages payload shape (Responses API):
  - system: `{"type":"input_text","text": SYSTEM_PROMPT}`
  - user: `[ {"type":"input_image","image_url": data_uri}, {"type":"input_text","text": USER_PROMPT} ]`
- UTF‑8 outputs, written as bytes.

Pitfalls & troubleshooting
- Filenames must be `Eugenia_pg###.png` (no suffixes like `_`).
- OneDrive/Windows paths work; ensure folders are available offline and writable.
- If API errors/rate limits occur, reduce `COUNT` and retry.
- To strengthen readability, increase `UPSCALE` moderately (e.g., 1.7–2.0).

Key files/dirs
- Script: `Polish_handwriting_transcriber_v4.py`
- Inputs: `input_pages/` | Outputs: `output_transcripts/`
- Docs: `README.md` (rationale, usage). Other text files and `debug_api_payloads/` are reference materials and not used by the script.

If any part of this is unclear or incomplete (e.g., handling of current filenames with underscores, desired output formatting), please share details and I’ll refine the guidance.