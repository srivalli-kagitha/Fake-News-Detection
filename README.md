# TruthLens AI

**An Explainable Multilingual Multimodal System for Fake News and AI-Generated Image Detection**

- Fake / Real news classification for **English** (DeBERTa-v3-large) and **Telugu** (IndicBERT v2 / mmBERT)
- **Real vs AI-generated / manipulated** image detection (CLIP ViT-L/14 + trained head, averaged with a Swin-V2 detector for recent generators)
- **Explainable AI**: word highlights for text (occlusion; LIME optional), Grad-CAM heatmaps for images
- **Claim-level fact checking with live evidence**: an LLM (gpt-oss-120b via Groq, or a local Ollama model) searches Wikipedia, the web, Google News and fact-checks, then labels each claim *Support*, *Contradiction* or *Unknown* with a reason and cited sources. Without an LLM, a contradiction-aware NLI pipeline is used
- **Batch processing**: a CSV of news or a ZIP of images, with results downloadable as CSV
- History of all checks (SQLite), each reopening its full report

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

**Windows:** double-click **`start.bat`**, or in a terminal opened in the project folder:

```
.\start.bat
```

(PowerShell needs the `.\` in front. If it still says "not recognized", run `dir`: you must be in
the folder that contains `start.bat`, usually `cd Fake-News-Detection` after `git clone`.)

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

> **First run:** installing packages takes 5–15 minutes. When the backend starts, it downloads the
> pre-trained models it needs (about 2 GB with the LLM fact checker on) and loads them in the
> background; the app shows "Getting the AI models ready: 2 of 4" until they are ready. The download
> happens only once. Later starts load the models in under a minute, before you click anything.
>
> **Results are kept while you move around:** a check keeps running if you open another page, and the
> result is there when you come back. A browser refresh starts a clean page. Every check is saved in **History**; click one to open
> its full report again (verdict, reasons, evidence with links, highlights, heatmap).

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
│   │       ├── text_explainer.py     occlusion (default) and LIME
│   │       ├── image_detector.py     CLIP ViT-L/14 detector + Grad-CAM
│   │       ├── claim_extractor.py    spaCy claim extraction
│   │       ├── evidence.py           BGE-M3 + FAISS retrieval, Google Fact Check API
│   │       ├── live_sources.py       live Wikipedia and Google News search
│   │       ├── llm_verifier.py       LLM fact checker: queries, live search, E5 selection, verdict
│   │       ├── claim_verifier.py     LLM first, DeBERTa-v3 / mDeBERTa NLI fallback + aggregation
│   │       ├── warmup.py             loads the models in the background at start-up
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
| Image model | CLIP ViT-L/14 + linear head, averaged with a Swin-V2 detector |
| Explainability | Occlusion or LIME (text), Grad-CAM (images) |
| Claim verification | gpt-oss-120b (Groq) or Ollama, multilingual-E5-small, ddgs web search, Wikipedia, Google News, Google Fact Check API; fallback: BGE-M3 + FAISS with DeBERTa-v3 / mDeBERTa NLI |
| Training | PyTorch, HuggingFace Transformers, Google Colab / Kaggle |

## 7. API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | System and model status |
| POST | `/api/text/analyze` | Classify text (+ optional claim verification) |
| POST | `/api/text/explain` | Word highlights for a text |
| POST | `/api/claims/verify` | Claim-level evidence verification |
| POST | `/api/image/analyze` | Real vs AI-generated image (+ Grad-CAM) |
| POST | `/api/batch/text` | CSV batch of news texts |
| POST | `/api/batch/images` | Several images or a ZIP |
| GET | `/api/batch/{job_id}/download` | Download batch results as CSV |
| GET / DELETE | `/api/history` | List or clear history |
| GET | `/api/history/{id}` | One check with its full report |

Interactive docs: http://127.0.0.1:8000/docs

## 8. Where the evidence comes from

**LLM fact checker (recommended, free key).** Get a key at https://console.groq.com (API Keys >
Create API Key) and put it in `backend/.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=paste-your-key-here
```

The free tier allows 1,000 requests a day (about 500 claims) and resets daily. For each claim the LLM
writes search queries, the backend searches Wikipedia, the web (Yahoo / DuckDuckGo / Mojeek),
Google News, published fact-checks and the local corpus, multilingual-E5 keeps the 10 most relevant
passages, and the LLM gives the verdict, a one-line reason and the passages it relied on
("Cited in the verdict" in the report). To run without a key: install https://ollama.com,
`ollama pull qwen3:4b`, and set `LLM_PROVIDER=ollama` (slower on a CPU).

**Without an LLM** the NLI pipeline is used against:

1. **Local evidence corpus** - `backend/data/evidence/*.jsonl`, always searched, works offline.
   `sample_corpus.jsonl` holds 40 verified facts; this is the set used in the report's evaluation.
2. **Live Wikipedia and Google News** (needs internet, no key). Switch off with `LIVE_SEARCH=false`.
3. **Published fact-checks** (free key) - Google Fact Check Tools API: PIB, BOOM, Factly,
   Alt News, AFP and others. Get a key at https://console.cloud.google.com
   (enable "Fact Check Tools API", then Credentials > Create credentials > API key) and set
   `GOOGLE_FACTCHECK_API_KEY=...` in `backend/.env`. This also helps the LLM checker.

**Never commit `backend/.env`** - it holds your keys and is listed in `.gitignore`.

Restart the app after editing `.env`. To add your own offline evidence:

```
cd backend
.venv\Scripts\activate          (Windows)      source .venv/bin/activate   (Linux/macOS)
python -m scripts.add_wikipedia_evidence "Chandrayaan-3" "COVID-19 vaccine" "Andhra Pradesh"
python -m scripts.add_wikipedia_evidence --lang te "తెలంగాణ"
```

Any `*.jsonl` file in `backend/data/evidence/` is indexed automatically, one
`{"text": ..., "source": ..., "url": ...}` per line.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Python is not installed` / `Node.js is not installed` | Install it (section 1), then open a **new** terminal |
| `Port 8000/5173 is already in use` | Close the other TruthLens window, or restart the computer |
| Package installation fails | Use Python 3.12 (not 3.13+), check your internet connection, run `start.bat` again |
| "model not trained yet" | Train the model (section 3) and copy it to `backend/models/` |
| First start is very slow | It is downloading the models (one time only, about 2 GB). Wait until the "Getting the AI models ready" note disappears. On a CPU each claim then takes about 15-25 s |
| Laptop with 8 GB RAM is slow or freezes | Close other programs. To skip loading at start, set `PRELOAD_MODELS=false` in `backend/.env` (models then load on first use) |
| Old checks in History have no "Open" | They were saved before full reports were kept. New checks can all be opened |
| `'CLIPConfig' object has no attribute 'hidden_size'` | Fixed in the latest code: run `git pull`, then restart |
| Need a clean reinstall | Delete `backend/.venv` and `frontend/node_modules`, then run `start.bat` again |
| Have an NVIDIA GPU and want to use it | After setup: `backend\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` |
