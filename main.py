
from __future__ import annotations
import os, json, time, argparse, pathlib, sys
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
# pip install google-generativeai
import google.generativeai as genai
from google.api_core import exceptions as gax

load_dotenv()
# ----------------------------- Configuration -----------------------------

DEFAULT_MODEL = "gemini-1.5-flash"

SYSTEM_INSTRUCTIONS = """
You are an expert OCR+forms parser. You receive a scanned, fixed-layout form with handwritten user inputs.
Your job:
1) Read only the handwritten or user-entered parts (not boilerplate).
2) For checkboxes, return the options for corresponding fields where the box is marked with a tick or crossed out with pen.
3) Map them to a normalized JSON object using the given schema and guidelines.
4) If a field is illegible or absent, set it to null and add a brief note in `notes`.

Important:
- Return the values in the reply JSON in the same order as in request schema
- Do not invent values.
- Keep dates in ISO 8601 when possible (YYYY-MM-DD). If only partial, keep what is present and explain in `notes`.
- Normalize phone numbers to E.164 if possible; otherwise raw.
- Return strictly valid JSON that conforms to the provided schema.
- If you find fields not in the schema but clearly important (e.g., “Policy #”, “Claim #”), place them under `extra_fields` with sensible keys.
- Where handwriting is uncertain, include a short reason in `notes` and set `confidence` appropriately.
"""
SCHEMA_PATH = pathlib.Path(__file__).parent / "test.json"
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    RESPONSE_SCHEMA = json.load(f)

GEN_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "response_mime_type": "application/json",
    "response_schema": RESPONSE_SCHEMA,
}

# ----------------------------- Helpers -----------------------------

def backoff_sleep(attempt: int):
    delay = min(2 ** attempt, 30)
    time.sleep(delay)

def upload_pdf(file_path: str):
    return genai.upload_file(path=file_path)

def extract_json(model: genai.GenerativeModel, file_obj, file_name: str) -> Dict[str, Any]:
    prompt = f"""
    You will be given a scanned PDF of a fixed-layout form
    Extract ONLY the handwritten/user-entered answers for each field into the JSON schema exactly .
    If the form has boxes or labeled fields, match labels to appropriate keys.
    Strictly return values in the same order as request schema
    When you don't see any value for a field, return NULL
    """
    for attempt in range(6):
        try:
            resp = model.generate_content(
                [
                    SYSTEM_INSTRUCTIONS,
                    file_obj,
                    prompt
                ],
                generation_config=GEN_CONFIG,
                request_options={"timeout": 180},
            )
            text = resp.text
            # print(resp)
            return json.loads(text)
        except (gax.ResourceExhausted, gax.ServiceUnavailable, gax.InternalServerError) as e:
            backoff_sleep(attempt)
            continue
        except Exception as e:
            # If model returns non-JSON (shouldn’t happen with response_schema), fall back cautiously
            try:
                print("Gemini API call failed:", e)
                return {}
            except Exception:
                raise
    raise RuntimeError("Max retries exceeded while calling Gemini.")

def process_pdf(model: genai.GenerativeModel, pdf_path: pathlib.Path, out_dir: pathlib.Path) -> Tuple[pathlib.Path, Dict[str, Any]]:
    file_obj = upload_pdf(str(pdf_path))
    data = extract_json(model, file_obj, pdf_path.name)

    data["_source_file"] = pdf_path.name
    data["_source_path"] = str(pdf_path.resolve())

    out_path = out_dir / (pdf_path.stem + ".json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="pdf", required=True, help="Path to input PDF")  # 👈 change here
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY missing in .env", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: file not found {pdf_path}", file=sys.stderr)
        sys.exit(2)

    file_obj = upload_pdf(str(pdf_path))
    data = extract_json(model, file_obj, pdf_path.name)

    # data["_source_file"] = pdf_path.name
    # data["_source_path"] = str(pdf_path.resolve())

    out_path = pathlib.Path(args.out)

    if out_path.is_dir() or str(args.out).endswith("/"):
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / (pdf_path.stem + ".json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Extracted JSON written to {args.out}")

def extract_json_from_pdf(pdf_path: str, model_name: str = DEFAULT_MODEL) -> dict:
    """
    Public function to extract handwritten JSON data from a PDF.
    Can be used both by CLI (main) and Streamlit (app.py).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("ERROR: GEMINI_API_KEY missing in .env")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    file_obj = upload_pdf(pdf_path)
    return extract_json(model, file_obj, pathlib.Path(pdf_path).name)


if __name__ == "__main__":
    main()
