import os
import json
import argparse
import pathlib
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "openai/gpt-oss-20b:nebius"
CHUNK_SIZE = 10000 

def load_ocr_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                txt = page.extract_text() or ""
                text_parts.append(f"\n--- Page {i+1} ---\n{txt}")
            except Exception as e:
                print(f"OCR failed for page {i+1}: {e}")
    return "\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def call_model(client, model: str, prompt: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
    )
    msg = completion.choices[0].message
    return msg.content if msg else None

def extract_json_only(text: str) -> str:
    if not text:
        return "{}"
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return "{}"
    return text[start:end+1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to empty form PDF")
    parser.add_argument("--out", required=True, help="Path to output schema JSON")
    args = parser.parse_args()
    load_dotenv()

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    ocr_text = load_ocr_text(args.pdf)
    chunks = chunk_text(ocr_text)

    merged_schema = {"type": "object", "properties": {}}

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}...")

        prompt = f"""
You are analyzing an EMPTY fixed-layout form (OCR text shown below).

Your task:
- Identify all fields and group them based on the blue headers or logical form sections as primary keys.
- Each SECTION must be a top-level property in the JSON schema.
- Each section must be an object with:
    "type": "object"
    "properties": {{}}

When defining fields:
- Use correct types: "string", "boolean", "integer", "array", or "object" as appropriate.
- If the field name or context implies a date → include "format": "date".
- If the field looks like an email → include "format": "email".
- If a field represents options or checkboxes → make it an "array" of "string" with an "enum" of possible values.
- If a field is a single dropdown → make it "type": "string" with an "enum".
- If a field repeats (like medication, insurance, etc.) → make it an "array" of "object" and describe the inner object fields.
- For related items (address, guardian info, etc.), group them into nested objects.
- Include "description" when it adds clarity (as in the reference JSON).
- Include enums and nested properties as required.
- Do not include "title" or "additionalProperties".
- Maintain JSON schema consistency with field nesting, array-of-object structure, and descriptive keys exactly like this example:
  - objects for grouped sections
  - arrays for repeatable fields
  - use of "format", "enum", "description" keys where meaningful.

You must output JSON with proper structure, level of nesting, and metadata richness.

OCR chunk:
{chunk}
"""
        raw = call_model(client, MODEL, prompt)
        schema_text = extract_json_only(raw)

        try:
            partial_schema = json.loads(schema_text)
            if "properties" in partial_schema:
                merged_schema["properties"].update(partial_schema["properties"])
        except Exception as e:
            print(f"Failed to parse JSON for chunk {i}: {e}")

    out_path = pathlib.Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged_schema, f, indent=2, ensure_ascii=False)

    print(f"Section-based schema written to {out_path}")

if __name__ == "__main__":
    main()
