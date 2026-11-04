This chapter describes how the design was turned into working code: the tools used, the structure of the project, the backend, the frontend, the training notebook and the launcher. Short excerpts of the real source code are included where they make a point clearer than prose.

## Technologies Used

TABLE: Technology stack
| Layer | Technology | Version | Role |
|---|---|---|---|
| Backend language | Python | 3.10 to 3.12 | All model and API code |
| Web framework | FastAPI + Uvicorn | 0.11x | REST API, request validation, automatic API docs |
| Deep learning | PyTorch | 2.x | Running and training models |
| Model library | Hugging Face Transformers | 4.4x | Loading DeBERTa, IndicBERT, CLIP and NLI models |
| Embeddings | Sentence-Transformers | 3.x | BGE-M3 evidence embeddings |
| Vector search | FAISS (CPU) | 1.x | Nearest-passage search |
| Explanations | LIME, pytorch-grad-cam | 0.2, 1.5 | Word weights, heatmaps |
| NLP utilities | spaCy (en_core_web_sm) | 3.x | Sentence splitting, named entities |
| Database | SQLite via SQLAlchemy | 2.x | Check history |
| Frontend | React, React Router | 18, 6 | Single-page application |
| Build tool | Vite | 6 | Development server, /api proxy, production build |
| Styling | Tailwind CSS | 4 | Design tokens, light and dark themes |
| HTTP client | Axios | 1.x | Calls from the browser to the API |
| Training | Kaggle / Google Colab notebook | T4 GPU | Fine-tuning and evaluation |

## Project Structure

```
truthlens-ai/
├── start.py, start.bat, start.sh     one-command launcher
├── backend/
│   ├── app/
│   │   ├── main.py                   FastAPI app, routers, error handlers
│   │   ├── config.py                 settings read from .env
│   │   ├── database.py               SQLite table and save_check()
│   │   ├── schemas.py                request models (validation)
│   │   ├── routers/                  text.py, image.py, batch.py, system.py
│   │   └── services/                 language, text_classifier, text_explainer,
│   │                                 claim_extractor, evidence, claim_verifier,
│   │                                 image_detector, common
│   ├── models/                       trained models (text_en, text_te, image_detector)
│   ├── data/evidence/                evidence corpus (*.jsonl) and FAISS index
│   └── scripts/                      build_evidence_index.py, add_wikipedia_evidence.py
├── frontend/
│   └── src/
│       ├── pages/                    Home, TextCheck, ImageCheck, ClaimCheck,
│       │                             BatchCheck, History
│       ├── components/               Sidebar, VerdictScale, MarkedText, ClaimsReport, ...
│       ├── lib/                      verdict wording, samples, theme, page memory
│       └── api.js                    all calls to the backend
├── training/                         notebooks and training scripts
└── evaluation/                       eval_claims.py and curated claim set
```

## Backend Implementation

### Configuration

Every value that a student might want to change, such as model names, the evidence folder, the similarity threshold or the number of LIME samples, is a field of one `Settings` class. Pydantic reads it from `backend/.env`, so changing a model means editing one line of a text file. Model paths starting with `models/` are resolved inside the backend folder, and anything else is treated as a Hugging Face model name and downloaded on first use.

### Lazy model loading

Loading all the models at start-up would take minutes and about 10 GB of memory. Instead, each model is created the first time it is needed. The `Lazy` wrapper uses a lock with a second check inside it, so two requests arriving together still load the model only once:

```
class Lazy(Generic[T]):
    """Builds an object the first time it is needed (thread-safe), then reuses it."""
    def get(self) -> T:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    logger.info("Loading %s ...", self.name)
                    self._value = self._factory()
        return self._value
```

If a trained model folder is missing, the classifier raises `ModelNotReadyError`, which the API returns as HTTP 503 with a message that says which training script to run. The interface shows the model as "Not trained" instead of failing.

### Language detection

```
TELUGU_CHARS = re.compile(r"[ఀ-౿]")
LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)

def detect_language(text: str) -> str:
    telugu = len(TELUGU_CHARS.findall(text))
    letters = len(LETTERS.findall(text)) or 1
    return "te" if telugu / letters > 0.3 else "en"
```

### Claim decision and fusion

The decision rule of Section 5.6 is a few lines of code. Keeping it this short makes it easy to explain in a viva and easy to test:

```
def _decide(evidence):
    if not evidence:
        return "Unknown", 0.0
    best_support = max(e["nli"]["entailment"] for e in evidence)
    best_contra = max(e["nli"]["contradiction"] for e in evidence)
    if best_contra >= DECISION_THRESHOLD and best_contra >= best_support:
        return "Contradiction", best_contra
    if best_support >= DECISION_THRESHOLD:
        return "Support", best_support
    return "Unknown", 1.0 - max(best_support, best_contra)

def combine(classification, verification):
    parts, based_on = [], []
    if classification:
        parts.append(classification["fake_probability"])
        based_on.append("classifier")
    if verification and verification["summary"]["verdict"] != "Unverified":
        parts.append(1.0 - verification["summary"]["credibility_score"])
        based_on.append("claim_evidence")
    if not parts:
        return None
    risk = sum(parts) / len(parts)
    return {"verdict": "Fake" if risk >= 0.5 else "Real",
            "fake_risk": round(risk, 4), "based_on": based_on}
```

### Evidence index

The retriever reads every `*.jsonl` file in the evidence folder, embeds the passages with BGE-M3 and builds a FAISS `IndexFlatIP` index. It stores a fingerprint of the file names, sizes and modification times with the index, so the index is rebuilt automatically when a file is added or changed and reused otherwise. Adding evidence therefore needs no command: drop a file in the folder and restart.

### Image detector and Grad-CAM

The detector class is the same in training and in the application, so the saved head loads without conversion. The CLIP embedding is normalised to unit length before the linear head:

```
class ClipDetector(nn.Module):
    def __init__(self, clip_name, num_labels):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        self.head = nn.Linear(self.clip.config.projection_dim, num_labels)

    def forward(self, pixel_values):
        emb = self.clip(pixel_values=pixel_values).image_embeds
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return self.head(emb)
```

Grad-CAM expects a feature map shaped like an image, but a vision transformer produces a sequence of tokens. The reshape function drops the class token and arranges the 256 patch tokens into a 16 by 16 grid:

```
def reshape_transform(t):
    t = t[:, 1:, :]                          # drop CLS token
    return t.reshape(t.size(0), grid, grid, t.size(2)).permute(0, 3, 1, 2)

target_layer = self.model.clip.vision_model.encoder.layers[-1].layer_norm1
with GradCAM(model=self.model, target_layers=[target_layer],
             reshape_transform=reshape_transform) as cam:
    heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_idx)])[0]
```

### API and error handling

FastAPI validates every request against a Pydantic model before the handler runs. A text shorter than 10 characters or longer than 20,000 is rejected with HTTP 422, an oversized image with HTTP 413, and an unsupported file type with HTTP 415. FastAPI also generates interactive API documentation at `/docs`, which was used throughout development to test the endpoints by hand.

## Frontend Implementation

### Pages and routing

React Router maps each tool to a path (`/`, `/text`, `/image`, `/claims`, `/batch`, `/history`). The Start page sends its input to the right tool through router state, and the tool starts the check as soon as it opens, so the user goes from pasting text to reading the report with one click.

### Verdict first, explanation second

LIME needs about 200 runs of the classifier, which on a laptop CPU takes much longer than the verdict itself. The text page therefore makes two requests. The first returns the verdict and the claim evidence and is shown at once; the second returns the word weights, and the highlights are drawn into the report already on screen:

```
// Step 1: verdict and claim evidence (fast)
data = await analyzeText({ text: input, language, explain: false,
                          verify_claims: verify });
setResult({ ...data, checkedText: input });
// Step 2: LIME highlights, filled into the report when ready
if (explain && data.classification) {
  const explanation = await explainText({ text: input, language: data.language });
  setResult((r) => r && r.checkedText === input
    ? { ...r, classification: { ...r.classification, explanation } } : r);
}
```

The check `r.checkedText === input` stops a slow explanation from an earlier check overwriting the report of a newer one.

### Marked-up article

The `MarkedText` component splits the article the same way the backend's LIME explainer does, on whitespace, so each word lines up with its weight. Each weighted word is drawn with a highlighter-style background whose strength is proportional to the absolute weight; the marks appear one after another when the report loads. Claims are found in the text by their character positions, and every word inside a claim gets an underline whose style shows the claim's status. Hovering a claim in the evidence list sets an "active" index that lights up the matching sentence.

### Theme and accessibility

All colours are design tokens defined once in CSS and redefined for dark mode, so the whole interface switches theme by changing one attribute on the page. A small script in `index.html` applies the saved theme before the first paint, so dark mode does not flash white. Every control can be reached with the keyboard and shows a visible focus ring, the verdict meters carry ARIA roles and values for screen readers, and animations are switched off when the operating system asks for reduced motion.

## Training Notebook

The training notebook writes the training scripts to disk, installs packages, downloads the data, and runs each step only if its output does not exist yet. A disconnected session can therefore be resumed with "Run all". A `QUICK_TEST` switch trains every model on a few hundred examples in about 25 minutes, which checks the whole pipeline before the long run. The Kaggle version sets `QUICK_TEST = False`, saves all outputs to the notebook's output folder, and runs as a background commit, so the laptop can be closed while it trains.

The core of the text training script uses the Hugging Face `Trainer`:

```
training_args = TrainingArguments(
    learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
    weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1",
    fp16=use_fp16, seed=args.seed)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tokenizer),
                  compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()
```

After training, each script evaluates on the test set and writes `metrics.json`, a classification report, a confusion matrix image and a training curve. The last notebook cell gathers the results into one table and packages the three application models into `truthlens_models.zip`.

## One-Command Launcher

Final-year projects are often hard to run on an examiner's computer because of missing packages or a wrong Python version. The launcher `start.py` (run by `start.bat` on Windows) removes these steps:

1. It checks that Python is version 3.10 to 3.12 and that Node.js is installed.
2. It creates the backend virtual environment if it does not exist.
3. It installs the Python packages, but only when the hash of `requirements.txt` has changed since the last install, so later starts take seconds.
4. It does the same for the frontend packages with `npm install`.
5. It checks that ports 8000 and 5173 are free, starts Uvicorn and Vite, waits for the health endpoint to answer, and opens the browser.
6. On Ctrl+C it stops both servers and their child processes, using `taskkill /T` on Windows.

```
def setup_backend():
    if not VENV_PYTHON.exists():
        run([sys.executable, "-m", "venv", VENV], ROOT,
            "Creating Python virtual environment")
    req, marker = BACKEND / "requirements.txt", VENV / ".requirements.sha256"
    wanted = file_hash(req)
    if marker.exists() and marker.read_text().strip() == wanted:
        say("Python packages already installed.")
    else:
        run([VENV_PYTHON, "-m", "pip", "install", "-r", req], BACKEND,
            "Installing Python packages")
        marker.write_text(wanted)
```
