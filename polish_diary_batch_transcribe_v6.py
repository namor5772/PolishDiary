#!/usr/bin/env python3
# (content truncated in this cell for brevity; it's the full v6 script from earlier)
#!/usr/bin/env python3
"""
polish_diary_batch_transcribe_v6.py

Ready-to-run batch transcriber for Polish handwritten diary PNGs using the OpenAI Responses API.
- Prompt-only transcription (preserves spelling mistakes).
- Flowing text, no page-layout mimic.
- Filename filter: only Eugenia_pg001..Eugenia_pg999.png
- START_PAGE option to skip earlier pages.
"""

import base64, io, re, sys, time
from pathlib import Path
import numpy as np
from openai import OpenAI, BadRequestError
from PIL import Image, ImageOps, ImageFilter

# --------------------------
# Configuration
# --------------------------
BASE_DIR = Path(r"C:\Users\roman\OneDrive\Python\PolishDiary")
INPUT_DIR = BASE_DIR / "input_pages"
OUTPUT_DIR = BASE_DIR / "output_transcripts"
DEBUG_DIR = BASE_DIR / "debug_api_payloads"

MODEL_NAME = "gpt-5"
START_PAGE = 94       # process only pages >= this number
USE_POST_CLEAN = True
CALL_DELAY_S = 0.5
MAX_RETRIES = 4
NAME_PATTERN = re.compile(r"^Eugenia_pg(\d{1,3})\.png$")

SYSTEM_PROMPT = (
    "Jesteś ekspertem w odczytywaniu polskich rękopisów. "
    "Twoim zadaniem jest wierne przepisanie treści do pliku TXT. "
    "Wytyczne (stosuj wszystkie): "
    "1) Przepisuj DOKŁADNIE to, co widzisz, słowo po słowie. "
    "2) Zachowaj wszystkie błędy ortograficzne, gramatyczne i nietypowe formy — nie poprawiaj ich. "
    "3) Nie zmieniaj kolejności słów, nie wygładzaj tekstu. "
    "4) Jeżeli fragment jest nieczytelny, wstaw [??] " 
    "5) Usuwaj jedynie sztuczne dzielenie wyrazów na końcu wersu "
    "(np. 'prze-\\nkazać' → 'przekazać'). "
    "6) Odpowiadaj WYŁĄCZNIE pełnym tekstem — bez komentarzy i wyjaśnień."
    "7) Zachowaj polskie diakrytyki i oryginalną interpunkcję. "
    "8) Nie dodawaj nagłówków, podpisów, numerów wierszy ani komentarzy. "
)

USER_PROMPT = (
    "Przepisz dokładnie tekst z obrazu do jednego lub kilku akapitów (ciągła proza). "
    "Scal sztuczne podziały wyrazów na końcach wierszy i nie odwzorowuj układu wersów. "
    "Zwróć wyłącznie finalny tekst."
)

# --------------------------
# Image preprocessing
# --------------------------
def _score_horizontal_alignment(pil_gray):
    arr = np.asarray(pil_gray, dtype=np.float32)
    arr = 255.0 - arr
    return float(np.var(arr.sum(axis=1)))

def deskew_coarse(pil_gray, angle_range=2.0, step=0.2):
    target_w = 1200
    scale = min(1.0, target_w / pil_gray.width)
    test_img = pil_gray if scale==1.0 else pil_gray.resize((int(pil_gray.width*scale), int(pil_gray.height*scale)))
    best_score, best_angle = None, 0.0
    angle = -angle_range
    while angle <= angle_range+1e-9:
        rotated = test_img.rotate(angle, expand=True, resample=Image.BICUBIC)
        score = _score_horizontal_alignment(rotated)
        if best_score is None or score > best_score:
            best_score, best_angle = score, angle
        angle += step
    return pil_gray.rotate(best_angle, expand=True, resample=Image.BICUBIC)

def preprocess_to_png_bytes(path):
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    img = deskew_coarse(img, 2.0, 0.2)
    img = img.resize((int(img.width*1.25), int(img.height*1.25)), resample=Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()

def data_uri_from_png_bytes(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

# --------------------------
# API call
# --------------------------
def try_create_response(client, payload):
    try:
        return client.responses.create(**payload)
    except BadRequestError as e:
        msg = str(e)
        if "text.verbosity" in msg:
            payload["text"]["verbosity"]="medium"
            return client.responses.create(**payload)
        if "Unsupported parameter" in msg:
            for k in ("temperature","max_output_tokens","tool_choice","text","reasoning"):
                payload.pop(k, None)
            return client.responses.create(**payload)
        raise

def call_openai_transcribe_image(client, model, data_uri):
    payload = dict(
        model=model,
        input=[
            {"role":"system","content":[{"type":"input_text","text":SYSTEM_PROMPT}]},
            {"role":"user","content":[
                {"type":"input_text","text":USER_PROMPT},
                {"type":"input_image","image_url":data_uri},
            ]},
        ],
        reasoning={"effort":"medium"},
        text={"format":{"type":"text"},"verbosity":"medium"},
        tool_choice="none",
        max_output_tokens=8192,
    )
    resp = try_create_response(client,payload)
    txt = getattr(resp,"output_text",None)
    if txt: return txt
    return resp.model_dump_json()

def dehyphenate_soft_wraps(text):
    return re.sub(r"(\w)[-\u00AD]\s*\n(\w)", r"\1\2", text, flags=re.UNICODE)

# --------------------------
# Discover input pages
# --------------------------
def discover_pages(input_dir, start_page):
    out=[]
    for f in input_dir.glob("Eugenia_pg*.png"):
        m=NAME_PATTERN.match(f.name)
        if not m: continue
        num=int(m.group(1))
        if num>=start_page: out.append((num,f))
    return [f for _,f in sorted(out)]

# --------------------------
# Main
# --------------------------
def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Missing dir {INPUT_DIR}"); sys.exit(2)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    DEBUG_DIR.mkdir(parents=True,exist_ok=True)
    pages=discover_pages(INPUT_DIR, START_PAGE)
    if not pages:
        print(f"[ERROR] No input images found at or after page {START_PAGE}"); sys.exit(3)
    client=OpenAI()
    for i,img_path in enumerate(pages,1):
        out_txt=OUTPUT_DIR/(img_path.stem+".txt")
        print(f"[{i}/{len(pages)}] {img_path.name} -> {out_txt.name}")
        success=False
        for attempt in range(1,MAX_RETRIES+1):
            try:
                data_uri=data_uri_from_png_bytes(preprocess_to_png_bytes(img_path))
                raw=call_openai_transcribe_image(client,MODEL_NAME,data_uri)
                if USE_POST_CLEAN: raw=dehyphenate_soft_wraps(raw)
                out_txt.write_text(raw.strip()+"\n",encoding="utf-8")
                print("   ✓ wrote",out_txt); success=True; break
            except Exception as e:
                print("   attempt",attempt,"failed:",e); time.sleep(min(2**attempt,30))
        if not success: print("   ✗ skipped after retries")

if __name__=="__main__":
    main()
