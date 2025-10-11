import streamlit as st
import json
import pathlib
from Gemini.main import extract_json_from_pdf

st.set_page_config(page_title="Handwritten Form Extractor", page_icon="📝", layout="wide")

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

    #Radio button to choose between 2 different views
    view_mode = st.radio(
        "Choose how to view the extracted data:",
        ["JSON Output", "Form UI View"]
    )
    #JSON View
    if view_mode == "JSON Output":
        st.subheader("Extracted JSON")
        st.json(data)

        st.download_button(
            label="⬇️ Download JSON",
            file_name=f"{temp_path.stem}.json",
            mime="application/json",
            data=json.dumps(data, indent=2, ensure_ascii=False)
        )

    #UI Form View
    elif view_mode == "Form UI View":
        st.subheader("Form View (Read-only)")
        st.caption("Extracted values displayed in a form-like layout")

        for section, fields in data.items():
            if isinstance(fields, dict):
                st.markdown(f"### {section}")
                for field, value in fields.items():
                    if isinstance(value, bool):
                        st.checkbox(field, value=value, disabled=True)
                    elif isinstance(value, list):
                        st.multiselect(field, options=value, default=value, disabled=True)
                    elif isinstance(value, dict):
                        st.markdown(f"**{field}:**")
                        for subfield, subval in value.items():
                            st.text_input(f"{field} → {subfield}", value=subval or "", disabled=True)
                    else:
                        st.text_input(field, value=value or "", disabled=True)
            else:
                st.text_input(section, value=fields or "", disabled=True)
