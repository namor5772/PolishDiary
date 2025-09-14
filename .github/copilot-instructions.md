# Copilot Instructions for PolishDiary

## Project Overview
This repository transcribes scanned Polish handwritten diary pages into text using OpenAI's GPT-5 Responses API. The workflow is highly automated, with all configuration hardcoded in the main script. The process includes targeted image preprocessing and strict transcription rules.

## Key Components & Data Flow
- **Main script:** `Polish_handwriting_transcriber_v4.py` (all logic, no CLI args)
- **Input images:** `input_pages/Eugenia_pg###.png` (3-digit, sequential)
- **Output transcripts:** `output_transcripts/Eugenia_pg###.txt` (one per image)
- **Preprocessing:** Grayscale, BICUBIC upscale, autocontrast, unsharp mask, Otsu binarization
- **API integration:** OpenAI Responses API (GPT-5), no `response_format` parameter
- **Logging:** Per-page metrics (chars, bytes, timings), summary table

## Developer Workflow
- All configuration is at the top of `Polish_handwriting_transcriber_v4.py`:
  - `START_NUM`, `COUNT`, `MODEL_NAME`, `UPSCALE`, `LOG_LEVEL`
- No command-line arguments; run directly: `python Polish_handwriting_transcriber_v4.py`
- API key must be set in the environment (`OPENAI_API_KEY`)
- Input/output directories must exist; script will create `output_transcripts/` if missing
- Install dependencies: `pip install pillow openai`

## Project-Specific Patterns
- **Strict file naming:** Only files matching `Eugenia_pg###.png` are processed
- **Literal transcription:** Spelling mistakes and idioms preserved, layout ignored, unreadable fragments as `[**]`
- **Preprocessing logic:** Handles Pillow enum changes for BICUBIC, pure-Python Otsu thresholding
- **Error handling:** Missing files are logged as warnings, not fatal
- **No response_format:** Avoids API errors by omitting this parameter

## Integration Points
- **OpenAI API:** Uses `from openai import OpenAI` and the Responses API
- **Image processing:** Pillow (PIL)
- **Environment:** Windows/OneDrive paths supported; UTF-8 output

## Example Workflow
1. Place PNGs in `input_pages/` (named `Eugenia_pg023.png`, etc.)
2. Set API key: `setx OPENAI_API_KEY "sk-..."` (Windows)
3. Adjust constants in script as needed
4. Run: `python Polish_handwriting_transcriber_v4.py`
5. Check `output_transcripts/` for results and console for summary

## Troubleshooting
- Do not use `response_format` in API calls
- Ensure filenames match exactly
- For Pillow BICUBIC, script auto-detects enum
- For Otsu thresholding, script uses safe lambda on L mode
- API errors: reduce `COUNT` or retry

## References
- See `README.md` for full rationale, configuration, and troubleshooting details
- Main script: `Polish_handwriting_transcriber_v4.py`
- Input: `input_pages/`, Output: `output_transcripts/`

---

If any section is unclear or missing, please provide feedback for further refinement.