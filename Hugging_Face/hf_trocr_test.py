import os
import json
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TROCR_MODEL = "microsoft/trocr-base-handwritten"
LLM_MODEL = "openai/gpt-oss-20b:nebius"
CHUNK_SIZE = 6000
SCHEMA_KEYS_PER_CHUNK = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pdf_to_images(pdf_path, dpi=200):
    print(f"Converting {pdf_path} to images...")
    return convert_from_path(pdf_path, dpi=dpi)


def load_clients():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
    return client


def load_trocr_local():
    print(f"Loading local TrOCR model on {DEVICE} ...")
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(DEVICE)
    return processor, model


def run_trocr_local(pil_image, processor, model):
    pil_image = pil_image.convert("RGB")
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def flatten_schema_keys(schema, prefix=""):
    keys = []

    if not isinstance(schema, dict):
        return keys

    if "properties" in schema:
        for prop, subschema in schema["properties"].items():
            new_prefix = f"{prefix}{prop}" if prefix == "" else f"{prefix}.{prop}"
            keys.extend(flatten_schema_keys(subschema, new_prefix))

    elif "items" in schema:
        keys.extend(flatten_schema_keys(schema["items"], prefix))

    elif "type" in schema:
        keys.append(prefix)

    return keys


def split_schema(schema, keys_per_chunk=SCHEMA_KEYS_PER_CHUNK):
    flat_keys = flatten_schema_keys(schema)
    return [flat_keys[i:i + keys_per_chunk] for i in range(0, len(flat_keys), keys_per_chunk)]


def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks, current, current_len = [], "", 0
    for para in text.split("\n"):
        if current_len + len(para) > chunk_size:
            chunks.append(current)
            current, current_len = "", 0
        current += para + "\n"
        current_len += len(para)
    if current.strip():
        chunks.append(current)
    return chunks


def query_llm_schema_fill(client, llm_model, schema_keys, ocr_text):
    keys_str = ", ".join(schema_keys)
#     prompt = f"""
# You are an information extraction model.
# Below is OCR text from a filled handwritten form. 
# Extract values ONLY for the following field keys and return valid JSON with those keys.

# <Field Keys>
# {keys_str}
# </Field Keys>

# <OCR Text>
# {ocr_text}
# </OCR Text>

# If a value is not found, leave it as an empty string.
# Return strictly valid JSON.
# """

    prompt = f"""
You are a Hand-written form data extraction expert.
Below is the text recognized from a filled intake form.
Try to match the content to these fields, even if the labels differ slightly.
Return valid JSON mapping each field to the closest text value.

FIELDS TO FILL:
{json.dumps(keys_str, indent=2)}

FORM TEXT:
{ocr_text}

If a value cannot be found, set it to "".
Return only JSON.
"""

    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_output = response.choices[0].message.content
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            raw_output = raw_output[start:end + 1]
        data = json.loads(raw_output)
    except Exception:
        data = {"raw_output": raw_output}
    return data


def main():
    parser = argparse.ArgumentParser(description="Handwriting form extraction using local TrOCR + HF LLM")
    parser.add_argument("--pdf", required=True, help="Path to filled PDF")
    parser.add_argument("--schema", required=True, help="Path to schema JSON")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    args = parser.parse_args()

    #Schema loading
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)
    schema_chunks = split_schema(schema) #Splitting schema into chunks and flatten it 
    total_keys = sum(len(chunk) for chunk in schema_chunks)
    print(f"Flattened schema: {total_keys} total keys ({len(schema_chunks)} chunks of {SCHEMA_KEYS_PER_CHUNK})")

    #TrOCR model loading
    client = load_clients()
    processor, trocr_model = load_trocr_local()
    images = pdf_to_images(args.pdf)

    #Running TrOCR locally as GPU is required if using with Hugging Face
    page_texts = []
    for i, img in enumerate(images, 1):
        print(f"Running local TrOCR on page {i}/{len(images)}...")
        text = run_trocr_local(img, processor, trocr_model)
        page_texts.append(f"\n--- Page {i} ---\n{text}")

    all_text = "\n".join(page_texts)
    text_chunks = chunk_text(all_text)
    print(text_chunks)

    #Calling the Hugging Face API to create JSON and merge all the chunks
    merged = {}
    for i, chunk in enumerate(text_chunks, 1):
        print(f"Processing OCR chunk {i}/{len(text_chunks)}...")
        for j, keys in enumerate(schema_chunks, 1):
            print(f"Schema subset {j}/{len(schema_chunks)} ({len(keys)} keys)")
            data = query_llm_schema_fill(client, LLM_MODEL, keys, chunk)
            for k, v in data.items():
                merged[k] = v or merged.get(k, "")

    #Writing the output JSON
    out_path = Path(args.out)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"Extraction complete! Saved to {out_path}")


if __name__ == "__main__":
    main()
