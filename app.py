import streamlit as st
import json
import pathlib
from main import extract_json_from_pdf

st.set_page_config(page_title="Handwritten Form Extractor", page_icon="📝")

st.title("Handwritten Form Extractor")
st.write("Upload a scanned PDF form to extract handwritten information into structured JSON.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    temp_path = pathlib.Path(f"./temp_{uploaded_file.name}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Processing your file")

    try:
        data = extract_json_from_pdf(str(temp_path), model_name="gemini-1.5-flash")
    except Exception as e:
        st.error(f"Gemini extraction failed: {e}")
        st.stop()

    # Show JSON result
    st.subheader("Extracted JSON")
    st.json(data)

    # Download JSON
    st.download_button(
        label="⬇️ Download JSON",
        file_name=f"{temp_path.stem}.json",
        mime="application/json",
        data=json.dumps(data, indent=2, ensure_ascii=False)
    )
