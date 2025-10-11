import os
import json
import pathlib
import argparse
import pdfplumber
import pytesseract
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from pytesseract import Output
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL="openai/gpt-oss-20b:nebius"
CHUNK_SIZE=6000
ENABLE_TROCR = True

def ocr_preprocess(pil_image):
    #Preprocessing the image to convert it into format pytesseract can accept
    img = np.array(pil_image.convert("L"))
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    img = cv2.medianBlur(img, 3)
    return Image.fromarray(img).convert("RGB")

def setup_trocr():
    #TrOCR setup
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

def trocr_ocr(pil_image, processor, model):
    #TrOCR function
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def load_ocr_text(pdf_path: str, use_trocr=False, trocr_models=None) -> str:
    text_parts = []
    #Doing OCR per page inside the loop
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                pil_image = page.to_image(resolution=300).original
                processed = ocr_preprocess(pil_image)
                txt=pytesseract.image_to_string(processed,lang="eng") #pytesseract call
                # txt = image.extract_text() or ""

                #TrOCR incase tesseract didn't produce any results
                if use_trocr and trocr_models:
                    processor, model = trocr_models
                    try:
                        hw_text = trocr_ocr(processed, processor, model)
                        txt += f"\n[TrOCR detected handwriting]: {hw_text}"
                    except Exception as e:
                        print(f"TrOCR failed for page {i+1}: {e}")
                
                text_parts.append(f"\n--- Page {i+1} ---\n{txt.strip()}")
            except Exception as e:
                print(f"OCR failed for page {i+1}: {e}")
    return "\n".join(text_parts)

def query_hf_llm(client, schema, ocr_chunk):
    #Calling HuggingFace to get JSON output
    prompt = f"""
You are an information extraction model.
Below is OCR text extracted from a filled form. 
Use ONLY the information present to fill the provided JSON schema. 
If a field is not found, leave it blank. Return strictly valid JSON.

<SCHEMA>
{json.dumps(schema, indent=2)}
</SCHEMA>

<OCR_TEXT>
{ocr_chunk}
</OCR_TEXT>
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        text = response.choices[0].message.content
        return json.loads(text)
    except Exception:
        print("Failed to parse JSON response from model. Returning raw text.")
        return {"raw_output": response.choices[0].message.content}
    
def chunk_text(text, chunk_size=CHUNK_SIZE):
    #Chunk the OCR text not to exceed the Hugging Face API limit
    pages = text.split("--- Page ")
    chunks, current, current_len = [], "", 0
    for p in pages:
        if not p.strip():
            continue
        if current_len + len(p) > chunk_size:
            chunks.append(current)
            current, current_len = "", 0
        current += f"--- Page {p}"
        current_len += len(p)
    if current:
        chunks.append(current)
    return chunks

def merge_json_list(json_list):
    #Merge output JSON from all the Chunks to get final JSON response
    merged = {}
    for js in json_list:
        for k, v in js.items():
            if isinstance(v, dict):
                merged[k] = merge_json_list([merged.get(k, {}), v])
            else:
                merged[k] = v if v else merged.get(k, "")
    return merged

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to input filled PDF")
    parser.add_argument("--schema", required=True, help="Path to request schema JSON")
    parser.add_argument("--out", required=True, help="Path to output extracted JSON")
    args = parser.parse_args()
    load_dotenv()

    #Hugging Face Client creation
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    #Schema loading...
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    #TROCR setup
    trocr_models = setup_trocr() if ENABLE_TROCR else None
    
    #OCR extraction
    ocr_text = load_ocr_text(args.pdf, use_trocr=ENABLE_TROCR, trocr_models=trocr_models)

    chunks = chunk_text(ocr_text)
    print(f"Total chunks to process: {len(chunks)}")

    #Calling HuggingFace for each Chunk
    all_jsons = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Sending chunk {i}/{len(chunks)} to LLM...")
        partial_json = query_hf_llm(client, schema, chunk)
        all_jsons.append(partial_json)

    merged_json = merge_json_list(all_jsons)

    #Write JSON output file
    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(merged_json, indent=2, ensure_ascii=False))
    print(f"Extraction completed. Output saved to {out_path}")

if __name__ == "__main__":
    main()