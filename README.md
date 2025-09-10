# What this Repository is all about
Python with OpenAI APIs was used to transcribe my polish grandmothers handwitten story of what happened to her family during WW2 and after. It was written in 1974 for my mother. It processes 157 pages scanned as *.png files and transcribes them into corresponding *.txt files.

Amazingly I used "ChatGPT 5 Thinking" with the following prompt to get working code first try.
The only thing I "needed" to do to the code was to (in the VS Code environment) do a Quick Fix on 3 sections of code, by adding "# type: ignore". This was just for cosmetics. Scanning made a big difference to the quality of the transciption, nevertheless I had to manually correct the transcription of all pages but it did reduce the time for transcription about 60%. 

```text
I would like you to create for me a python program for me with the following properties:

1. When run it has no command-line argument, any such things are hardcoded into the code.
2. It will run from some base directory (in which the *.py file will reside). Call it Polish_handwriting_transcriber_v4. In that directory there will be two subdirectories. 
3. input_pages which will contain *.png image files with sequential names of the form Eugenia_pg###.png where ### is an integer eg. 023.
4. output_transcripts which will contain *.txt files that are the result of transcribing a particular *.png file. Eg Eug_pg023.png will generate the file Eugenia_pg023.txt
5. The png image files are scans of handwritten pages of a diary written by my grandmother. They are written in polish. They are quite messy, so maybe OCR preprocessing would be quite helpful for accuracy and not just gpt-5.
6. The corresponding txt files are to be an accurate transcription of these scans.
7. The transcription must be as literal as possible. preserve spelling mistakes or local polish idioms, however do not mimic page layout. do not preserve hyphenation of words across lines.
8. processing will be specified to start at a particular hardcoded file number and be set to process a number of succeeding files (also hardcoded).
9. Include the size of the text for each file preprocessed as well as the time taken. Fix up any issues with any BICUBIC call.
10. If any fragment of handwriting is unreadable replace it with [**]
11. You will use the chat gpt-5 OpenAI new responses API.
12. Make sure to not to use the response_format parameter in the Responses API, since it causes errors when running the code.

```

## The AI response
Roman, here’s a complete, single-file Python program that matches your spec. It hard-codes the base paths and page range, runs simple but effective OCR-oriented preprocessing (grayscale → contrast → sharpen → Otsu binarization → safe BICUBIC upscale), calls the __OpenAI Responses API__ with __gpt-5__ (no `response_format` used), and writes literal Polish transcriptions to per-page `.txt` files. It also logs, for each page, (a) character count of the resulting transcript, and (b) timings (total and preprocessing). Unreadable fragments are marked as `[**]`.

### How to use
1. __"Save the file below as__ `Polish_handwriting_transcriber_v4.py` in your base folder."
1. __"Ensure subfolders__ `input_pages/` __and__ `output_transcripts/` __exist."__
1. __"Name images like__ `Eugenia_pg023.png`__."__
1. __"Set your API key in env:__ `setx OPENAI_API_KEY "sk-..."` __(Windows) or export on Linux/macOS."__
1. __"Adjust__ `START_NUM`, `COUNT`, __and (optionally)__ `MODEL_NAME`__."__
1. __"__`pip install pillow openai`__"__
1. __"Run:__ `python Polish_handwriting_transcriber_v4.py`__"__

<br><br>
<<< PLACE IN RESPONSE WHERE PYTHON CODE WAS MADE AVAILABLE FOR COPYING >>>
<br><br>

### Notes & rationale
- __No CLI args:__ All knobs are constants at the top (`START_NUM`, `COUNT`, `MODEL_NAME`, `UPSCALE`).
- __Paths:__ The script treats the folder containing the .py file as the base; uses `./input_pages` and `./output_transcripts`.
- __Preprocessing:__ Kept light but targeted for handwriting: grayscale → safe __BICUBIC__ upscale (handles the Pillow enum change), autocontrast, mild unsharp mask, __Otsu__ thresholding (pure-Python, no NumPy). Critically, the thresholding is done on L mode so the lambda sees integers—this avoids the “`>` not supported” issue you saw when someone tried to threshold in the wrong mode.
- __Responses API call:__ Minimal parameters; __no__ `response_format` (this directly avoids your `TypeError: Responses.create() got an unexpected keyword argument 'response_format'`).
- __Literal transcript:__ The system/user prompts enforce strict transcription, diacritics preserved, no layout mimicry, hyphenation joins, and`[**]`for unreadables.
- __Post-processing:__ Removes any stray code fences or “Transcription:” headers the model might emit; merges hyphenated line-break splits conservatively.
- __Per-file metrics:__ Logs character count, UTF-8 byte size, total seconds, and preprocessing seconds; a compact summary table is printed at the end.
- __Missing files:__ Gracefully warned and counted as `missing`.
- __Windows-friendly writes:__ Files are written explicitly as UTF-8 (bytes), no BOM.

If you want me to tailor the preprocessing for your exact page scans (e.g., stronger denoise, background flattening, deskew), say the word and I’ll slot it in without adding heavy dependencies, or—if you prefer—add OpenCV/Tesseract hooks as optional paths.

<br><br>
# AI generated readme for this Repository 
# Polish Handwriting Transcriber v4

A small, single-file Python utility that transcribes scanned **Polish handwritten diary pages** into plain UTF-8 text files using OpenAI’s **Responses API** (model: **gpt-5**).  
The program applies light OCR-oriented preprocessing (grayscale → optional **BICUBIC** upscale → autocontrast → unsharp mask → **Otsu** binarization), then prompts the model to produce a **literal** transcript:

- Preserve **original spelling**, mistakes, and idioms  
- **Do not** mimic page layout (no columns/frames)  
- **Do not** preserve line-end hyphenations (join split words)  
- Replace unreadable fragments with `[**]` exactly  

Per page, it records **character count** and **timings** (total and preprocessing), and writes one `.txt` output per `.png` input.

---

## Directory layout

Place the script and the two subfolders like this:

```
<your base folder>/
├─ Polish_handwriting_transcriber_v4.py
├─ input_pages/
│  ├─ Eugenia_pg023.png
│  ├─ Eugenia_pg024.png
│  └─ …
└─ output_transcripts/
   └─ (created automatically)
```

**Input naming convention**

- Files must match: `Eugenia_pg###.png` (exactly 3 digits), e.g. `Eugenia_pg023.png`.

Each input image yields a corresponding text file:

- `input_pages/Eugenia_pg023.png` → `output_transcripts/Eugenia_pg023.txt`

---

## Requirements

- Python 3.10+ (recommended 3.11 or newer)
- Pillow (PIL) for image preprocessing
- openai Python SDK (modern interface: `from openai import OpenAI`)
- An OpenAI API key with access to **gpt-5** (Responses API)

Install packages:

```bash
pip install --upgrade pillow openai
```

---

## Installation

1) Copy `Polish_handwriting_transcriber_v4.py` into your base folder.  
2) Create the `input_pages/` folder and add your PNGs.  
3) Ensure `output_transcripts/` exists (the script will also create it if missing).  
4) Set your API key:

**Windows (PowerShell)**
```powershell
setx OPENAI_API_KEY "sk-REPLACE_WITH_YOUR_KEY"
# Open a new terminal after running setx
```

**macOS / Linux (bash/zsh)**
```bash
export OPENAI_API_KEY="sk-REPLACE_WITH_YOUR_KEY"
```

---

## Configuration

All knobs are **hard-coded** at the top of the script (no CLI args):

- `MODEL_NAME = "gpt-5"` — model alias to use  
- `START_NUM = 23` — first page number (e.g., 23 → `Eugenia_pg023.png`)  
- `COUNT = 3` — how many consecutive pages to process (023, 024, 025)  
- `UPSCALE = 1.5` — optional BICUBIC upscale factor before binarization  
- `LOG_LEVEL = logging.INFO` — console verbosity

Adjust these constants to suit your run.

---

## Usage

From the base folder that contains the script:

```bash
python Polish_handwriting_transcriber_v4.py
```

No parameters are required (or accepted).

---

## What the script does

1) **Loads** each requested image (`Eugenia_pgNNN.png`).  
2) **Preprocesses** it (for legibility):
   - Convert to **grayscale**
   - Optional **BICUBIC** upscale (safe across Pillow versions)
   - **Autocontrast** and **UnsharpMask**
   - **Otsu** thresholding (pure-Python; no NumPy)
3) **Calls OpenAI Responses API** (model **gpt-5**) with:
   - A **strict system prompt** enforcing literal transcription
   - The **image** (as a data URI) followed by a short **user instruction**
   - ⚠️ **No `response_format` parameter** is used (avoids `TypeError`)
4) **Post-processes** the raw text:
   - Remove stray code fences / labels the model might emit
   - Join **hyphenated line-break splits** (`prze-\nkazać` → `przekazać`)
   - Normalize whitespace while keeping paragraph breaks
5) **Writes** the transcript to `output_transcripts/Eugenia_pgNNN.txt`  
6) **Logs** per-page metrics (character count, UTF-8 bytes, total seconds, preprocessing seconds) and prints a final summary table.

---

## Output & logging

Example console output:

```
2025-09-10 13:02:01,948 [INFO] Base dir:   C:\Users\roman\OneDrive\Python\PolishDiary
2025-09-10 13:02:01,949 [INFO] Input dir:  C:\Users\roman\OneDrive\Python\PolishDiary\input_pages
2025-09-10 13:02:01,949 [INFO] Output dir: C:\Users\roman\OneDrive\Python\PolishDiary\output_transcripts
2025-09-10 13:02:01,949 [INFO] Model:      gpt-5
2025-09-10 13:02:01,949 [INFO] Processing pages: 23 .. 25 (count=3)
2025-09-10 13:02:02,551 [INFO] OK  Eugenia_pg023.png -> Eugenia_pg023.txt | chars=1842 bytes=1883 | total=7.42s preproc=0.29s
...
Summary: ok=3 missing=0 error=0

 page | status  |   chars |   bytes | total[s] | preproc[s]
------------------------------------------------------------
   23 | ok      |    1842 |    1883 |     7.42 |      0.29
   24 | ok      |    1759 |    1794 |     7.10 |      0.27
   25 | ok      |    2013 |    2064 |     7.63 |      0.30
------------------------------------------------------------
```

Each corresponding `.txt` file will be in `output_transcripts/`.

---

## Customization

- **Page selection**: modify `START_NUM` and `COUNT`.  
- **Model pinning**: change `MODEL_NAME` (e.g., a dated `gpt-5-YYYY-MM-DD` if you want reproducibility).  
- **Preprocessing strength**:
  - Increase `UPSCALE` to `1.7–2.0` for very small or faint writing.
  - You can also adjust `UnsharpMask(radius=1.4, percent=150, threshold=2)` within the script if strokes are too soft or too harsh.
- **Prompts**: tweak `SYSTEM_PROMPT` / `USER_PROMPT` (e.g., add “treat stamps/marginalia as [**]”).
- **Logging**: set `LOG_LEVEL = logging.DEBUG` for more detail.

---

## Troubleshooting

### "Responses.create() got an unexpected keyword argument 'response_format'"
- **Cause**: Some SDK versions / environments reject `response_format` for image-based runs.  
- **Status here**: The script **does not** pass `response_format`, by design, to avoid this error.

### "operator '>' not supported" during thresholding
- **Cause**: Applying a threshold to an RGB tuple (e.g., using `"1"` mode or wrong lambda signature).  
- **Status here**: The script converts to `"L"` first and uses a safe lambda: `lambda p, t=thr: 255 if p > t else 0`.

### Pillow BICUBIC import / enum issues
- **Cause**: Pillow ≥10 moved resampling filters to `Image.Resampling`.  
- **Status here**: The script detects both:
  ```python
  try:
      RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
  except Exception:
      RESAMPLE_BICUBIC = Image.BICUBIC
  ```

### "Missing page" warnings
- Ensure your filenames exactly match `Eugenia_pg###.png` (3 digits, no suffixes like `_` or extra spaces).

### Read/Write permissions
- On Windows OneDrive paths, ensure the folder is synced/available offline and you have write permission to `output_transcripts/`.

### API errors (timeouts / rate limits)
- Re-run later, or reduce `COUNT`.
- Avoid very large images; typical scanned page PNGs at **150–300 DPI** work well.

---

## FAQ

**Q: Why not use `response_format={"type":"text"}` for guaranteed text?**  
A: In your environment, that parameter caused a hard error. The script avoids it and instead robustly extracts `resp.output_text` (or falls back to walking `resp.output`).

**Q: Will the output keep the original line breaks?**  
A: The model is asked *not* to mimic layout. It outputs plain running text (paragraphs allowed). Only line-end hyphenations are joined; otherwise spelling and wording are preserved.

**Q: How do unreadable parts appear?**  
A: Exactly as `[**]` (two asterisks inside brackets), per your requirement.

**Q: Can I process JPEGs or PDFs?**  
A: This script expects PNGs. If you have PDFs, pre-render them to PNGs first (e.g., 200–300 DPI). Adding built-in PDF rendering is possible but not included to keep dependencies minimal.

---

## Notes on privacy & costs

- Images are sent to OpenAI to produce the transcription. If your pages contain sensitive information, review OpenAI’s data handling policies for your account tier before running at scale.  
- API usage is billed per token and per image processing; costs depend on page size and content. Test on a few pages to estimate your total.

---

## Quick-start checklist

- [ ] Python environment has `pillow` and `openai`  
- [ ] `OPENAI_API_KEY` set and terminal restarted (Windows)  
- [ ] Files named `Eugenia_pg###.png` in `input_pages/`  
- [ ] `START_NUM` and `COUNT` set  
- [ ] Run the script and verify `.txt` outputs + summary table


