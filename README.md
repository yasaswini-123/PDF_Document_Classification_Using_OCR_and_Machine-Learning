# PDF & Image Document Classification: Invoice, Bill or Receipt

Upload a PDF or an image of a document. The app extracts its text (direct PDF text extraction, with **Tesseract OCR** for scanned pages and images) and a **TF-IDF + Support Vector Machine (SVM)** model (scikit-learn `LinearSVC`) predicts whether it is an **Invoice**, a **Bill** or a **Receipt**.

> **Mini project built during my internship at SURE ProEd** (a unit of SURE Trust, Skill Upgradation for Rural-youth Empowerment).
> Mentor: `Mr. Vamsi Krishna` | Author: `Balla Yasaswini`

## What it does

- Accepts PDFs (digital or scanned) and images (PNG, JPG, JPEG, TIF, TIFF).
- Extracts the text and shows it in the interface.
- Predicts the document category: **Invoice**, **Bill** or **Receipt**.
- Shows a warning when too little readable text is found, and an error message when a file cannot be processed.

## How it works

```
        PDF or image upload
                |
      +---------+----------+
      |                    |
     PDF                 Image
      |                    |
 PyMuPDF text         Grayscale, upscale to 1500 px,
 extraction           contrast boost
      |                    |
 page has < 30 chars?      |
   yes -> OCR (300 DPI)    |
      |                    v
      +--------> Tesseract OCR (English)
                |
                v
   Clean text: lowercase, remove URLs and symbols
                |
                v
   TF-IDF (1-2 word n-grams, 5,000 features)
                |
                v
   Linear SVM classifier (LinearSVC)
                |
                v
        Invoice / Bill / Receipt
```

Digital PDFs use their embedded text. Scanned PDF pages and images go through Tesseract OCR.

## Project structure

```
pdf_document_classification/
├── datasets/                  # collected documents, one folder per class
│   ├── invoice/
│   ├── receipt/
│   └── bills/
├── notebook/
│   └── pdf_ocr.ipynb          # OCR extraction, dataset creation, training, evaluation
├── results/                   # evaluation outputs
├── csv_file.csv               # structured dataset (extracted text + category)
├── document_classifier.pkl    # trained model (TF-IDF + Linear SVM pipeline)
├── tfidf_vectorizer.pkl       # standalone TF-IDF vectorizer
├── document_app.py            # Streamlit application
├── requirements.txt
└── README.md
```

`document_classifier.pkl` is a single pipeline that already contains the TF-IDF step, so the app needs only that one file.

## Setup and run

Tested with Python 3.10 on Ubuntu.

**1. Install Tesseract OCR** (a system program, not a Python package):

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH
```

**2. Install the Python packages:**

```bash
git clone <your-repository-url>
cd pdf_document_classification

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**3. Start the app:**

```bash
streamlit run document_app.py
```

Then open http://localhost:8501, upload a PDF or image, and read the predicted category.

Notes:
- The saved model is an `imbalanced-learn` pipeline, so `imbalanced-learn` must be installed even though the app only predicts.
- The model was saved with scikit-learn 1.7.2. If loading fails with a version warning, install the version pinned in `requirements.txt`.
- To retrain, open `notebook/pdf_ocr.ipynb`, change `DATASET_PATH` and `BASE_DIR` to your own folders, and run the cells from top to bottom.

## Dataset

Documents were collected from public datasets and organised into three class folders, then converted to text with OCR and saved in `csv_file.csv` (columns: `filename`, `file_path`, `category`, `file_type`, `extracted_text`, `extraction_method`, `status`, `error`, `cleaned_text`).

| Class | Files collected | Usable after OCR and cleaning | Train | Test |
|---|---|---|---|---|
| Invoice | 860 | 860 | 688 | 172 |
| Receipt | 1,599 | 1,582 | 1,265 | 317 |
| Bill | 798 | 780 | 624 | 156 |
| **Total** | **3,257** | **3,222** | **2,577** | **645** |

32 documents produced no text, and 3 more were empty after cleaning, so 35 were removed. The remaining data was split 80/20 with stratification (`random_state=42`). The training pipeline uses random oversampling so the smaller classes are not under-learned.

### Sources

| No. | Class | Dataset | Platform | Link | Licence |
|---|---|---|---|---|---|
| 1 | Invoice | Company Documents Dataset | Kaggle | https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset | see source page |
| 2 | Invoice | High Quality Invoice Images for OCR | Hugging Face (Voxel51) | https://huggingface.co/datasets/Voxel51/high-quality-invoice-images-for-ocr | see source page |
| 3 | Invoice | RVL-CDIP small | Kaggle | https://www.kaggle.com/datasets/uditamin/rvl-cdip-small | see source page |
| 4 | Invoice | FATURA multi-layout invoice dataset | Zenodo | https://zenodo.org/records/10371464 | CC BY 4.0 |
| 5 | Receipt | Receipts | Kaggle (jenswalter) | https://www.kaggle.com/datasets/jenswalter/receipts-receipts | see source page |
| 6 | Bill | Personal Financial Dataset for India | Kaggle | https://www.kaggle.com/datasets/mehaksingal/personal-financial-dataset-for-india | see source page |
| 7 | Bill | Bills_Electricity | Roboflow Universe | https://universe.roboflow.com/electricity-bills/bills_electricity | see source page |
| 8 | Bill | gas_bills | Roboflow Universe | https://universe.roboflow.com/electricity-bills/gas_bills | see source page |

The raw documents are not included in this repository. Please check each source's licence on its own page before reusing the data.

## Model

The classifier is a **Support Vector Machine (SVM)**, specifically scikit-learn's linear SVM, `LinearSVC`. A linear SVM is a common choice for text classification because TF-IDF produces many sparse features, and a linear boundary separates them well. The whole model is one `imbalanced-learn` pipeline with three steps:

| Step | Component | Settings |
|---|---|---|
| 1. Features | `TfidfVectorizer` | `max_features=5000`, word n-grams `(1, 2)`, `min_df=1`, `sublinear_tf=True` |
| 2. Balancing | `RandomOverSampler` | `random_state=42` (applied only while training) |
| 3. Classifier | `LinearSVC` (linear SVM) | `random_state=42` |

The pipeline is trained on the cleaned OCR text and saved with `joblib` as `document_classifier.pkl`. Only this model was trained and evaluated in the notebook. Comparing it with other classifiers (for example Logistic Regression or Naive Bayes) is listed under future work.

## Results

Evaluated on the 645 held-out test documents:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Invoice | 1.00 | 1.00 | 1.00 | 172 |
| Bill | 1.00 | 0.99 | 0.99 | 156 |
| Receipt | 0.99 | 1.00 | 1.00 | 317 |

**Overall accuracy: 99.69%**

**Please read this number with care.** The test set comes from the same collection as the training set, using a random split. Template-based or generated documents (for example FATURA invoices or synthetic bills) look very similar to each other, so similar documents can end up in both sets and push the score up. Accuracy on documents from a completely different source is likely to be lower. See the limitations below.

## Limitations and future work

- **Optimistic test score.** The split is random, not by source or template. A fairer test holds out whole sources or templates, and uses new documents the model has never seen.
- **Three classes only.** Any document is forced into Invoice, Bill or Receipt, even if it is something else (a letter, a form, a contract).
- **No confidence score yet.** The app shows the predicted class only. A confidence threshold, using the SVM decision scores, would let it say "inconclusive" for uncertain documents.
- **OCR settings differ slightly.** Training used default Tesseract settings, while the app applies image preprocessing and `--psm 6` for uploaded images. Using the same OCR settings in both places would make predictions more consistent.
- **Invoice and bill overlap.** The two classes use very similar wording, and this is the most likely source of mistakes.
- **English only**, and quality drops on blurry, rotated or low-resolution scans.
- Ideas: compare the SVM with Logistic Regression and Naive Bayes, add more real bills and receipts from different regions, evaluate on a fully separate test folder, add a confidence threshold, try character n-gram features, and compare a layout-aware model such as LayoutLM.

## Libraries used

PyMuPDF, pytesseract (Tesseract OCR), Pillow, pandas, scikit-learn, imbalanced-learn, joblib, matplotlib, Streamlit.

## Acknowledgements

- **SURE ProEd** (a unit of SURE Trust) and my mentor `Mr. Vamsi Krishna` for guidance during the internship.
- Dataset authors and uploaders: FATURA, RVL-CDIP, Voxel51, the Kaggle dataset owners listed above, and the Roboflow Universe contributors.
- Libraries: PyMuPDF, Tesseract OCR, scikit-learn, imbalanced-learn, Streamlit.

## Author

`Balla Yasaswini` | [GitHub](https://github.com/yasaswini-123/) | [LinkedIn](https://linkedin.com/in/ballayasaswini)

## Licence

To be confirmed with the mentor. Dataset licences are listed above and apply to the data, not to this code.
