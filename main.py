
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

Your tasks:
1. Extract only handwritten or user-entered values (ignore pre-printed labels, headers, or boilerplate).
2. Map extracted values exactly to the corresponding JSON schema field names.
3. Return the fields in the same order as in the request schema.
4. If a field is illegible or absent, return null and add a short explanation in `notes`.
5. If a field is struck through, blacked out, or otherwise redacted, treat it as intentionally hidden and return null.
6. For checkboxes: detect if a box is ticked, crossed, or circled, and return the corresponding option(s). If multiple boxes are marked, return all as an array.
7. Do not invent or guess values. If something cannot be read, use null.
8. Normalize dates to ISO 8601 (YYYY-MM-DD) when possible.
9. Normalize phone numbers to E.164 format if possible, otherwise return raw digits.
10. Always return strictly valid JSON that conforms to the provided schema.
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
    You will be given a scanned PDF form: {file_name}.
    Your job is to extract ONLY the handwritten or user-entered answers 
    and fill them into the provided JSON schema.

    Important:
    - Match values strictly to schema field names.
    - Keep field order exactly as in the schema.
    - Return null for any illegible, missing, or redacted (struck out/blackened) fields.
    - For checkboxes: return the options where the box is visibly marked (tick, cross, or circle).
    - Do not copy boilerplate text or labels from the form.
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
    # api_key = os.getenv("GEMINI_API_KEY")
    api_key = st.secrets["GEMINI_API_KEY"]
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("ERROR: GEMINI_API_KEY missing in .env")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    file_obj = upload_pdf(pdf_path)
    return extract_json(model, file_obj, pathlib.Path(pdf_path).name)


if __name__ == "__main__":
    main()
