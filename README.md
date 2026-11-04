# TruthLens AI

**An Explainable Multilingual Multimodal System for Fake News and AI-Generated Image Detection**

- Fake / Real news classification for **English** (DeBERTa-v3-large) and **Telugu** (IndicBERT v2 / mmBERT)
- **Real vs AI-generated / manipulated** image detection (CLIP ViT-L/14 + trained head)
- **Explainable AI**: LIME word highlights for text, Grad-CAM heatmaps for images
- **Contradiction-aware claim-level evidence verification**: each claim is labelled *Support*, *Contradiction* or *Unknown*, and the labels are combined into an overall assessment
- **Batch processing**: a CSV of news or a ZIP of images, with results downloadable as CSV
- History of all checks (SQLite)

---

## 1. Install these first (one time)

| Software | Version | Download |
|---|---|---|
| **Python** | 3.10, 3.11 or **3.12** (recommended) | https://www.python.org/downloads/ (on Windows, tick **"Add python.exe to PATH"**) |
| **Node.js** | 20 LTS or newer | https://nodejs.org/ |

Also needed: about **15 GB of free disk space** (models and packages) and **8 GB of RAM** (16 GB recommended).

## 2. Get the code and the trained models

The trained models (about 2.6 GB) are too large for GitHub, so they are shared separately as one zip file.

```
git clone https://github.com/srivalli-kagitha/Fake-News-Detection.git
cd Fake-News-Detection
```

1. Download **`truthlens_models.zip`** from Google Drive: **[add the Drive link here]**
2. Extract it into `backend/models/`, so that the folder looks like this:

```
backend/models/text_en/          English model (DeBERTa-v3-large, 1.7 GB)
backend/models/text_te/          Telugu model (IndicBERT v2, 1.1 GB)
backend/models/image_detector/   Image detector (config.json + head.pt)
```

If the zip extracts into an extra folder (for example `backend/models/truthlens_models/text_en`),
move the three folders up one level.

## 3. Run the project (one command)

**Windows:** double-click **`start.bat`**, or in a terminal:

```
start.bat
```

**Linux / macOS / any OS:**

```
python start.py
```

That one command:
1. creates a Python virtual environment in `backend/.venv` and installs all packages
2. runs `npm install` for the frontend
3. starts the **backend** at http://127.0.0.1:8000 (API docs at http://127.0.0.1:8000/docs)
4. starts the **frontend** at http://localhost:5173 and opens it in the browser

Press **Ctrl+C** to stop both. Later runs skip anything already installed and start in a few seconds.

> **First run:** installing packages takes 5–15 minutes. The first claim verification also downloads
> about 4 GB of pre-trained models from HuggingFace, which takes a few minutes. Both happen only once.

Other options:

```
python start.py --setup        # only install, do not start
python start.py --no-browser   # do not open the browser automatically
```

## 4. Train the models (only if you want to retrain)

The models in `truthlens_models.zip` are already trained, so this step is not needed to run the
project. To train them again, use a free Google Colab or Kaggle GPU. Until a model is in
`backend/models/`, its page shows a "model not trained yet" message; claim verification works
without them, because it uses pre-trained NLI models.

See **[training/README.md](training/README.md)** for step-by-step Colab instructions. After training,
copy the output folders to:

```
backend/models/text_en/          English fake news model (DeBERTa-v3-large)
backend/models/text_te/          Telugu fake news model
backend/models/image_detector/   Image detector (config.json + head.pt)
```

No restart is needed: the app picks up new models on the next request.

**Report and paper:** the project report is in `report/TruthLens_Project_Report.docx` (and `.pdf`),
and the IEEE paper in `paper/truthlens_ieee.pdf`.

## 5. Project structure

```
truthlens-ai/
├── start.py / start.bat / start.sh   one-command launcher
├── backend/                          FastAPI + PyTorch
│   ├── app/
│   │   ├── main.py                   FastAPI app, CORS, error handling
│   │   ├── config.py                 settings (from backend/.env)
│   │   ├── database.py               SQLite history (SQLAlchemy)
│   │   ├── routers/                  API endpoints: text, image, batch, system
│   │   └── services/
│   │       ├── text_classifier.py    DeBERTa-v3-large / IndicBERT v2 classifier
│   │       ├── text_explainer.py     LIME
│   │       ├── image_detector.py     CLIP ViT-L/14 detector + Grad-CAM
│   │       ├── claim_extractor.py    spaCy claim extraction
│   │       ├── evidence.py           BGE-M3 + FAISS retrieval, Google Fact Check API
│   │       ├── claim_verifier.py     DeBERTa-v3 / mDeBERTa NLI + aggregation
│   │       └── language.py           English / Telugu detection
│   ├── data/evidence/                evidence corpus (*.jsonl)
│   ├── models/                       trained models go here
│   ├── scripts/                      evidence index tools
│   ├── requirements.txt
│   └── .env.example
├── frontend/                         React + Vite + Tailwind CSS
│   └── src/
│       ├── pages/                    Home, TextCheck, ImageCheck, ClaimCheck, BatchCheck, History
│       ├── components/               Navbar, verdict badges, LIME highlights, claim cards, charts
│       └── api.js                    Axios API client
└── training/                         Colab training scripts (text + image) and dataset preparation
```

## 6. Tech stack

| Layer | Technologies |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS 4, React Router, Axios, Recharts |
| Backend | Python, FastAPI, Uvicorn, SQLAlchemy + SQLite, Pandas, Pillow |
| Text models | DeBERTa-v3-large (English), IndicBERT v2 / mmBERT (Telugu), BERT-base (baseline) |
| Image model | CLIP ViT-L/14 + linear head |
| Explainability | LIME (text), Grad-CAM (images) |
| Claim verification | spaCy, BGE-M3 + FAISS, DeBERTa-v3-large-NLI, mDeBERTa-v3-XNLI, Google Fact Check API (optional) |
| Training | PyTorch, HuggingFace Transformers, Google Colab / Kaggle |

## 7. API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | System and model status |
| POST | `/api/text/analyze` | Classify text (+ LIME, + optional claim verification) |
| POST | `/api/claims/verify` | Claim-level evidence verification |
| POST | `/api/image/analyze` | Real vs AI-generated image (+ Grad-CAM) |
| POST | `/api/batch/text` | CSV batch of news texts |
| POST | `/api/batch/images` | Several images or a ZIP |
| GET | `/api/batch/{job_id}/download` | Download batch results as CSV |
| GET / DELETE | `/api/history` | List or clear history |

Interactive docs: http://127.0.0.1:8000/docs

## 8. Evidence base for claim verification

`backend/data/evidence/sample_corpus.jsonl` holds 40 verified facts for demonstration. Grow it with:

```
cd backend
.venv\Scripts\activate          (Windows)      source .venv/bin/activate   (Linux/macOS)
python -m scripts.add_wikipedia_evidence "Chandrayaan-3" "COVID-19 vaccine" "Andhra Pradesh"
python -m scripts.add_wikipedia_evidence --lang te "తెలంగాణ"
```

Any `*.jsonl` file in `backend/data/evidence/` is indexed automatically, one
`{"text": ..., "source": ..., "url": ...}` per line. For live fact-checks, put a free
Google Fact Check Tools API key in `backend/.env` as `GOOGLE_FACTCHECK_API_KEY`.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Python is not installed` / `Node.js is not installed` | Install it (section 1), then open a **new** terminal |
| `Port 8000/5173 is already in use` | Close the other TruthLens window, or restart the computer |
| Package installation fails | Use Python 3.12 (not 3.13+), check your internet connection, run `start.bat` again |
| "model not trained yet" | Train the model (section 3) and copy it to `backend/models/` |
| First claim check is very slow | It is downloading models (one time only). Later checks take seconds |
| Need a clean reinstall | Delete `backend/.venv` and `frontend/node_modules`, then run `start.bat` again |
| Have an NVIDIA GPU and want to use it | After setup: `backend\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` |
