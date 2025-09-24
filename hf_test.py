import os, sys, json, argparse, pathlib, subprocess, shutil, re
from typing import List
from dotenv import load_dotenv

from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from openai import OpenAI

SYSTEM_INSTRUCTIONS = """
You are a forms-structure analyst. You receive OCR text of part of an EMPTY form.
Your job is to infer the tree structure of fields from this section and produce a partial JSON request schema.

Rules:
- Use only: "type", "properties", "items", and optionally "required".
- Free text fields -> { "type": "string" }
- Dates -> { "type": "string" }
- Phone numbers -> { "type": "string" }
- Checkboxes / multi-select lists -> { "type": "array", "items": { "type": "string" } }
- Yes/No toggles -> { "type": "boolean" }
- Top-level must be { "type": "object", "properties": { ... } }
- Do not include enums, format, description, or additionalProperties.
- Return strictly valid JSON, nothing else.
"""

PROMPT_TEMPLATE = """
Document OCR (section):

{ocr_payload}

Task:
Analyze this section and return a JSON schema fragment with fields only from this part.
Use "type", "properties", "items", "required" only.
Output JSON object only.
"""

# ----------------------------- Helpers -----------------------------

def ensure_tools():
    if not shutil.which("tesseract"):
        sys.exit("ERROR: Tesseract not installed.")
    if not shutil.which("pdftoppm"):
        print("WARNING: poppler-utils not found; pdf2image may fail.")

def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi)

def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img)

def pdf_to_ocr_text(pdf_path: str, dpi: int = 200) -> str:
    images = pdf_to_images(pdf_path, dpi=dpi)
    parts = []
    for idx, img in enumerate(images, start=1):
        txt = ocr_image(img)
        parts.append(f"\n===== PAGE {idx} =====\n{txt.strip()}\n")
    return "\n".join(parts)

def chunk_text(text: str, max_chars: int = 10000) -> List[str]:
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind("\n", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at])
        text = text[split_at:]
    if text.strip():
        chunks.append(text)
    return chunks

def call_model(client: OpenAI, model: str, prompt: str, max_tokens: int = 8192) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    choice = completion.choices[0]
    raw = getattr(choice.message, "content", None)
    if raw is None and isinstance(choice.message, dict):
        raw = choice.message.get("content")
    if not raw:
        raise RuntimeError(f"Model did not return text. Full response: {completion}")
    return raw.strip().strip("```").replace("json\n", "").replace("JSON\n", "")

def extract_json_only(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(text[start:end+1])

def merge_schemas(schemas: List[dict]) -> dict:
    """Merge multiple partial schemas into one."""
    merged = {"type": "object", "properties": {}}
    for schema in schemas:
        if "properties" in schema:
            for k, v in schema["properties"].items():
                if k not in merged["properties"]:
                    merged["properties"][k] = v
                else:
                    if v.get("type") == "object" and merged["properties"][k].get("type") == "object":
                        merged["properties"][k]["properties"].update(v.get("properties", {}))
    return merged

# ----------------------------- Main -----------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to empty form PDF")
    parser.add_argument("--out", required=True, help="Output JSON schema file")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: set HF_TOKEN environment variable.")

    ensure_tools()

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"ERROR: file not found: {pdf_path}")

    ocr_text = pdf_to_ocr_text(str(pdf_path))

    chunks = chunk_text(ocr_text, max_chars=10000)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

    schemas = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {idx}/{len(chunks)}...")
        prompt = PROMPT_TEMPLATE.format(ocr_payload=chunk)
        raw = call_model(client, "openai/gpt-oss-20b:nebius", prompt)
        try:
            schema = extract_json_only(raw)
            schemas.append(schema)
        except Exception as e:
            print(f"Failed to parse chunk {idx}: {e}")

    final_schema = merge_schemas(schemas)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_schema, f, indent=2, ensure_ascii=False)

    print(f"Schema written to {out_path}")

if __name__ == "__main__":
    main()
