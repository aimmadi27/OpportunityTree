from __future__ import annotations
import os, json, time, argparse, pathlib, sys
from typing import Any, Dict, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as gax

DEFAULT_MODEL = "gemini-2.0-flash"

SYSTEM_INSTRUCTIONS = """
You are an expert OCR and form-understanding assistant.
You receive a scanned, fixed-layout PDF form that contains both printed text and handwritten answers.

Your tasks:
1. Extract ONLY the handwritten or user-entered responses (ignore printed labels and headings).
2. Match extracted responses exactly to the JSON schema field names provided.
3. Return a valid JSON strictly conforming to that schema.
4. If a field is blank, illegible, not found or redacted, set its value to null.
5. Do NOT guess or fabricate answers.
6. For checkboxes: return the marked options (tick, cross, circle). Multiple markings = array.
7. Normalize dates to YYYY-MM-DD where possible.
8. Normalize phone numbers to E.164 format where possible.
9. Do not copy form labels or instructions into the answers.
10. Always return valid JSON only.
"""

def backoff_sleep(attempt: int):
    delay = min(2 ** attempt, 30)
    print(f"Gemini request failed. Retrying in {delay}s...")
    time.sleep(delay)

def upload_pdf(file_path: str):
    print(f"Uploading {file_path} to Gemini...")
    return genai.upload_file(path=file_path)

def extract_json(model: genai.GenerativeModel, file_obj, file_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    GEN_CONFIG = {
        "temperature": 0.1,
        "top_p": 0.9,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }

    user_prompt = f"""
    You are analyzing a scanned PDF form: {file_name}.
    Fill in only the user-entered information that corresponds to the provided JSON schema.
    Do not copy any printed boilerplate text.
    """

    for attempt in range(6):
        try:
            resp = model.generate_content(
                [
                    SYSTEM_INSTRUCTIONS,
                    file_obj,
                    user_prompt
                ],
                generation_config=GEN_CONFIG,
                request_options={"timeout": 300},
            )
            text = resp.text
            return json.loads(text)
        except (gax.ResourceExhausted, gax.ServiceUnavailable, gax.InternalServerError) as e:
            backoff_sleep(attempt)
            continue
        except Exception as e:
            print(f"Gemini API error: {e}")
            try:
                text = getattr(resp, "text", "")
                start, end = text.find("{"), text.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(text[start:end + 1])
            except Exception:
                pass
            return {}
    raise RuntimeError("Max retries exceeded while calling Gemini API.")

def extract_json_from_pdf(pdf_path: str, schema_path: str, model_name: str = DEFAULT_MODEL) -> dict:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    file_obj = upload_pdf(pdf_path)
    return extract_json(model, file_obj, pathlib.Path(pdf_path).name, schema)

def main():
    parser = argparse.ArgumentParser(description="Extract handwritten form data using Gemini")
    parser.add_argument("--pdf", required=True, help="Path to input filled PDF")
    parser.add_argument("--schema", required=True, help="Path to JSON schema file")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model to use")
    args = parser.parse_args()

    #Gemini Setup
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY missing in .env", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)

    #Schema loading
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    #Calling model by giving pdf file and schema
    model = genai.GenerativeModel(args.model)
    file_obj = upload_pdf(str(pdf_path))
    data = extract_json(model, file_obj, pdf_path.name, schema)

    #Writing output file
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Extraction complete! JSON saved to: {out_path}")

if __name__ == "__main__":
    main()
