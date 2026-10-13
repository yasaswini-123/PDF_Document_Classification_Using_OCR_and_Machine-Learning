import streamlit as st
import fitz
import joblib
import re
import shutil
import tempfile
from pathlib import Path

# --------------------------------
# Load trained machine learning model
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "document_classifier.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------
# Clean extracted text
# --------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------
# Extract text from PDF using OCR if needed
# --------------------------------
def extract_pdf_text(pdf_bytes):
    extracted_pages = []
    ocr_used = False

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:

        for page in doc:
            text = page.get_text().strip()

            # Use OCR for scanned pages
            if len(text) < 20:

                if shutil.which("tesseract") is None:
                    raise RuntimeError(
                        "Tesseract OCR is not installed."
                    )

                ocr_page = page.get_textpage_ocr(
                    language="eng",
                    dpi=200
                )

                text = page.get_text(
                    textpage=ocr_page
                ).strip()

                ocr_used = True

            extracted_pages.append(text)

    return "\n".join(extracted_pages), ocr_used


# --------------------------------
# Streamlit user interface
# --------------------------------
st.set_page_config(
    page_title="PDF Document Classification",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Document Classification")

st.write(
    "Upload a PDF document to predict whether it is "
    "an invoice, receipt, or bill."
)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Classify Document"):

        try:
            pdf_bytes = uploaded_file.getvalue()

            with st.spinner("Reading and classifying document..."):

                extracted_text, ocr_used = extract_pdf_text(
                    pdf_bytes
                )

                cleaned_text = clean_text(extracted_text)

                if not cleaned_text:
                    st.error(
                        "No readable text was found in this PDF."
                    )
                    st.stop()

                prediction = model.predict([cleaned_text])[0]

            st.subheader("Classification Result")

            st.success(
                f"Predicted Category: {prediction.upper()}"
            )

            if ocr_used:
                st.info("OCR was used to read the scanned PDF.")
            else:
                st.info("Text was extracted directly from the PDF.")

            with st.expander("View Extracted Text"):
                st.text(extracted_text[:10000])

        except Exception as error:
            st.error(f"Unable to process PDF: {error}")