import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from pdf2image import convert_from_path
import tempfile
from json_repair import repair_json  # for safety


SYSTEM_INSTRUCTIONS = """
You are an expert OCR and form-understanding assistant.

{schema}

You receive a scanned PDF form that contains both printed labels and handwritten responses.
Your job:
1. Extract ONLY the handwritten or user-entered responses.
2. Match each extracted response exactly to the JSON schema provided.
3. If a field is blank, illegible, or missing, return null.
4. Do NOT guess or copy printed text.
5. Return only valid JSON that fits the schema structure exactly.
6. For checkboxes, return the marked options.
7. Normalize dates to YYYY-MM-DD and phone numbers to E.164 if possible.

Return strictly valid JSON. Do not include comments, trailing commas, or extra text.
"""

DEFAULT_MODEL = "gemini-2.0-flash"


def extract_page_json(model, page_image, page_num, schema_text):
    print(f"Processing page {page_num} ...")

    page_prompt = f"""
This is page {page_num} of a multi-page form.
Extract only the handwritten or user-entered responses visible on this page.
Return valid JSON according to the provided schema.
"""

    for attempt in range(3):
        try:
            response = model.generate_content(
                [
                    {"role": "user", "parts": [
                        {"text": schema_text},
                        {"text": page_prompt},
                        {"mime_type": "image/png", "data": page_image}
                    ]}
                ],
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "response_mime_type": "application/json"
                },
                request_options={"timeout": 180}
            )

            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                start, end = response.text.find("{"), response.text.rfind("}")
                if start != -1 and end != -1:
                    candidate = response.text[start:end + 1]
                    return json.loads(repair_json(candidate))
                raise

        except Exception as e:
            print(f"Gemini error on page {page_num}: {e}")
            if attempt < 2:
                delay = 2 ** attempt
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Skipping page {page_num} after repeated errors.")
                return {}


def merge_page_results(results):
    merged = {}
    for page_data in results:
        if not isinstance(page_data, dict):
            continue
        for key, value in page_data.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(description="Page-wise Gemini OCR with schema output")
    parser.add_argument("--pdf", required=True, help="Path to input filled PDF")
    parser.add_argument("--schema", required=True, help="Path to JSON schema file")
    parser.add_argument("--out", required=True, help="Path to output JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model (default: gemini-2.0-flash)")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY missing in .env")

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(args.model)

    # Load schema
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    schema_text = SYSTEM_INSTRUCTIONS.format(
        schema=json.dumps(schema, indent=2, ensure_ascii=False)
    )

    print(f"Converting {args.pdf} to images...")
    pages = convert_from_path(args.pdf, dpi=150)
    print(f"{len(pages)} pages converted.\n")

    all_page_data = []
    for i, page in enumerate(pages, start=1):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            page.save(tmp.name, "PNG")
            with open(tmp.name, "rb") as img_file:
                img_bytes = img_file.read()
            page_json = extract_page_json(model, img_bytes, i, schema_text)
            all_page_data.append(page_json)

    # Merge
    final_json = merge_page_results(all_page_data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print(f"\nExtraction complete! Combined JSON saved to {out_path}")


if __name__ == "__main__":
    main()
