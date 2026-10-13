import re
import traceback
from pathlib import Path

import fitz
import joblib
import pytesseract
import streamlit as st

from PIL import Image, ImageOps, ImageEnhance


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Document Classification",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "document_classifier.pkl"


# ==================================================
# 2. HELPER: RENDER HTML WITHOUT MARKDOWN CODE BLOCKS
# ==================================================
# Streamlit's st.markdown treats lines indented by 4+ spaces, or
# anything after a blank line, as a code block. That is why raw HTML
# appeared as text. This helper removes indentation and blank lines
# so the HTML is rendered correctly.

def render_html(html):
    cleaned = "\n".join(
        line.strip()
        for line in html.splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# ==================================================
# 3. WHITE INTERFACE DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #172033;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: #f1f6ff;
    border: 1px solid #dbe7f7;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #173b70 !important;
    font-size: 36px;
}

.hero p {
    color: #52647d !important;
    line-height: 1.8;
}

.card {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 20px;
}

.card h3 {
    color: #173b70 !important;
}

.muted {
    color: #64748b !important;
    line-height: 1.7;
}

.result-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
}

.result-label {
    color: #52647d !important;
    font-size: 13px;
    font-weight: 600;
}

.result-value {
    color: #1d4ed8 !important;
    font-size: 32px;
    font-weight: bold;
    margin-top: 10px;
}

[data-testid="stFileUploader"] {
    background: #f8fbff;
    border: 1px dashed #9db8df;
    border-radius: 12px;
    padding: 15px;
}

/* Extracted text: white background and black text */
div[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

div[data-testid="stTextArea"] label {
    color: #172033 !important;
}

.footer {
    text-align: center;
    color: #8490a3;
    border-top: 1px solid #e5e7eb;
    padding-top: 20px;
    margin-top: 35px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please save document_classifier.pkl "
        f"inside this folder:\n{BASE_DIR}"
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))

except Exception:
    st.error("Unable to load the trained model.")

    with st.expander("Technical details"):
        st.code(traceback.format_exc())

    st.stop()


# ==================================================
# 5. CLEAN EXTRACTED TEXT
# ==================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==================================================
# 6. IMAGE PREPROCESSING
# ==================================================

def prepare_image(image):

    image = ImageOps.exif_transpose(image)

    image = image.convert("L")

    if image.width < 1500:

        scale = 1500 / image.width

        new_width = 1500

        new_height = max(
            1,
            int(image.height * scale)
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    image = ImageEnhance.Contrast(image).enhance(1.5)

    return image


# ==================================================
# 7. OCR FOR IMAGES
# ==================================================

def extract_text_from_image(image):

    image = prepare_image(image)

    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 6"
    )

    return text.strip()


# ==================================================
# 8. PDF TEXT EXTRACTION WITH OCR FALLBACK
# ==================================================

def extract_text_from_pdf(pdf_bytes):

    extracted_pages = []

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            # First try extracting selectable PDF text.
            text = page.get_text(sort=True).strip()

            # Apply OCR if the page has little or no text.
            if len(text) < 30:

                text_page = page.get_textpage_ocr(
                    language="eng",
                    dpi=300,
                    full=True
                )

                text = page.get_text(
                    textpage=text_page
                ).strip()

            extracted_pages.append(text)

    return "\n\n".join(extracted_pages).strip()


# ==================================================
# 9. DISPLAY CATEGORY NAME
# ==================================================

def get_category_name(prediction):

    prediction = str(prediction).strip().lower()

    categories = {
        "invoice": "Invoice",
        "bill": "Bill",
        "bills": "Bill",
        "receipt": "Receipt"
    }

    return categories.get(
        prediction,
        prediction.title()
    )


# ==================================================
# 10. APPLICATION HEADER
# ==================================================

render_html("""
<div class="hero">
    <h1>Document Classification</h1>

    <p>
        Upload a PDF or image to identify whether it is an
        Invoice, Bill, or Receipt.
    </p>

    <p>
        Text is extracted from digital PDFs and scanned documents
        using text extraction and OCR. A trained machine-learning
        model then predicts the document category.
    </p>
</div>
""")


# ==================================================
# 11. UPLOAD SECTION
# ==================================================

left, right = st.columns(
    [1.1, 0.9],
    gap="large"
)

with left:

    render_html("""
    <div class="card">
        <h3>Upload your document</h3>

        <p class="muted">
            Supported formats: PDF, PNG, JPG, JPEG, TIF and TIFF.
        </p>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose a PDF or image",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ]
    )


with right:

    render_html("""
    <div class="card">
        <h3>Document categories</h3>

        <p class="muted">
            🧾 <b>Invoice</b><br>
            A request for payment for goods or services.
        </p>

        <p class="muted">
            🧮 <b>Bill</b><br>
            A statement of charges or an amount due.
        </p>

        <p class="muted">
            🛍️ <b>Receipt</b><br>
            Proof of a purchase or payment.
        </p>
    </div>
    """)


# ==================================================
# 12. PROCESS UPLOADED DOCUMENT
# ==================================================

if uploaded_file is not None:

    st.divider()

    st.subheader("Document analysis")

    st.caption(
        f"Selected file: {uploaded_file.name}"
    )

    try:

        file_bytes = uploaded_file.getvalue()

        # ------------------------------------------
        # Extract text
        # ------------------------------------------

        with st.spinner(
            "Extracting text from your document..."
        ):

            if uploaded_file.name.lower().endswith(".pdf"):

                extracted_text = extract_text_from_pdf(
                    file_bytes
                )

            else:

                with Image.open(uploaded_file) as image:

                    image_copy = image.copy()

                extracted_text = extract_text_from_image(
                    image_copy
                )

        # ------------------------------------------
        # Clean extracted text
        # ------------------------------------------

        cleaned_text = clean_text(
            extracted_text
        )

        # ------------------------------------------
        # Validate extracted text
        # ------------------------------------------

        if (
            not cleaned_text
            or len(cleaned_text.split()) < 3
        ):

            st.warning(
                "Not enough readable text was found. "
                "Please upload a clearer PDF or image."
            )

            with st.expander(
                "View extracted text",
                expanded=True
            ):

                st.text_area(
                    "Extracted text",
                    value=extracted_text or
                    "No readable text was extracted.",
                    height=300,
                    key="empty_extracted_text",
                    label_visibility="collapsed"
                )

        else:

            # --------------------------------------
            # Predict category
            # --------------------------------------

            with st.spinner(
                "Predicting document category..."
            ):

                # This expects a Pipeline that accepts raw text.
                prediction = model.predict(
                    [cleaned_text]
                )[0]

            label = get_category_name(
                prediction
            )

            # --------------------------------------
            # Display result
            # --------------------------------------

            render_html(f"""
            <div class="result-card">
                <div class="result-label">
                    PREDICTED DOCUMENT CATEGORY
                </div>

                <div class="result-value">
                    {label}
                </div>
            </div>
            """)

            # --------------------------------------
            # Display extracted text
            # --------------------------------------

            with st.expander(
                "View extracted text",
                expanded=True
            ):

                st.text_area(
                    "Extracted text",
                    value=extracted_text,
                    height=300,
                    key="extracted_document_text",
                    label_visibility="collapsed"
                )

            st.caption(
                "If the prediction is incorrect, inspect the extracted "
                "text for OCR errors. Model accuracy also depends on "
                "the quality and variety of the training documents."
            )

    except pytesseract.TesseractNotFoundError:

        st.error(
            "Tesseract OCR is not installed or is not available "
            "on your system PATH."
        )

    except Exception:

        st.error(
            "Unable to process this document."
        )

        with st.expander("Technical details"):

            st.code(
                traceback.format_exc()
            )


# ==================================================
# 13. FOOTER
# ==================================================

render_html("""
<div class="footer">
    Document Classification using OCR and Machine Learning
</div>
""")