This chapter explains how to install TruthLens AI on a new computer, how to install the trained models, and how to use each page of the application.

## Prerequisites

Install these once:

1. **Python 3.10, 3.11 or 3.12** from python.org. On Windows, tick "Add python.exe to PATH" on the first screen of the installer. Python 3.13 is not yet supported by all the machine learning packages.
2. **Node.js 18 or later** (the LTS version) from nodejs.org.
3. About **15 GB of free disk space** for the Python packages and models.
4. An internet connection for the first run, which downloads the packages and the pre-trained models.

## Installing and Starting

**Windows.** Open the `truthlens-ai` folder and double-click `start.bat`.

**Linux or macOS.** Open a terminal in the `truthlens-ai` folder and run:

```
python3 start.py
```

The first run creates the Python environment and installs the packages, which takes 5 to 15 minutes. It then starts the backend at http://127.0.0.1:8000 and the frontend at http://localhost:5173, and opens the browser. Later runs skip the installation and start in a few seconds. Press Ctrl+C in the terminal window to stop both servers.

TABLE: Launcher options
| Command | Effect |
|---|---|
| python start.py | Install what is missing, start both servers and open the browser |
| python start.py --setup | Only install; do not start |
| python start.py --no-browser | Start without opening the browser |

## Installing the Trained Models

The training notebook produces `truthlens_models.zip`. Extract it into `backend/models/` so that the folder contains:

```
backend/models/text_en/          English model (DeBERTa-v3-large)
backend/models/text_te/          Telugu model (IndicBERT v2)
backend/models/image_detector/   Image detector (config.json + head.pt)
```

No restart is needed: the application loads a model the first time it is used. The NLI models and the BGE-M3 embedding model are downloaded automatically from Hugging Face on the first claim check, about 3 GB in total.

## Configuration

All settings are in `backend/.env`, which the launcher creates from `.env.example` on the first run.

TABLE: Main configuration settings
| Setting | Default | Meaning |
|---|---|---|
| TEXT_EN_MODEL, TEXT_TE_MODEL, IMAGE_MODEL | models/... | Folders of the trained models |
| NLI_EN_MODEL, NLI_MULTI_MODEL | MoritzLaurer DeBERTa models | NLI models for English and Telugu claims |
| EMBED_MODEL | BAAI/bge-m3 | Evidence embedding model |
| EVIDENCE_DIR | data/evidence | Folder of evidence JSONL files |
| GOOGLE_FACTCHECK_API_KEY | empty | Optional free key to add published fact-checks |
| LIME_NUM_SAMPLES | 200 | Perturbed copies per text explanation; lower is faster |
| MAX_CLAIMS | 8 | Most claims taken from one article |
| EVIDENCE_TOP_K, EVIDENCE_MIN_SIMILARITY | 3, 0.45 | Passages per claim and the similarity cut-off |
| MAX_BATCH_ROWS | 500 | Most rows in one batch CSV |
| DEVICE | auto | cpu, cuda, or auto to use a GPU when present |

## Adding Evidence

To let the system check more claims, add passages to the evidence folder. Create a file such as `backend/data/evidence/my_facts.jsonl` with one passage per line (wrapped here to fit the page):

```
{"text": "ISRO's Chandrayaan-3 lander touched down near the Moon's south pole
          on 23 August 2023.", "source": "ISRO", "url": "https://www.isro.gov.in"}
```

Restart the backend. The index is rebuilt automatically because the folder changed. To add whole Wikipedia articles, run:

```
cd backend
.venv/Scripts/python scripts/add_wikipedia_evidence.py "Chandrayaan-3" "Polavaram Project"
```

On Linux or macOS the Python path is `.venv/bin/python`. The script splits each article into passages and saves them as a new evidence file.

## Using the Application

### Start page

Paste a news article or headline into the text box and press the check button (or Ctrl+Enter), or drop an image into the image area. The application opens the right tool and starts the check.

### Checking news text

1. Open "Check news text" from the left rail.
2. Paste the article, or press one of the example links under the box (rumour, news report or Telugu) to load a sample.
3. Leave the language on "Detect", or choose English or Telugu.
4. Leave "Highlight the words behind the verdict" and "Check claims against evidence" on for the full report, or switch them off for a faster verdict.
5. Press "Check this text" (or Ctrl+Enter).

The report shows the verdict and a fake-risk scale first. The highlighted article follows a little later: red marks are words that pushed the verdict toward fake, green marks toward real, and stronger colour means more influence. Each claim is underlined, solid for supported, wavy for contradicted and dotted for no evidence. The claim list below shows each claim's status, the evidence passage, its source link and the NLI scores; hovering a claim lights up its sentence in the article.

### Checking an image

Open "Check an image", drop or choose a JPG, PNG or WEBP file of up to 15 MB, and press "Check this image". The report shows the verdict, the confidence scale and the image with a slider. Drag the slider to compare the original with the Grad-CAM heatmap: warm colours mark the regions that drove the decision.

### Verifying claims only

"Verify claims" runs claim extraction and evidence checking without the classifier. It is useful for a single statement, such as a forwarded message, when the reader only wants to know whether the evidence supports it.

### Checking many at once

Open "Check many at once". For text, upload a CSV file with a column named `text` (and, optionally, a column named `language`). For images, choose several image files or one ZIP archive. After the check, the page shows a summary of the results and a table of rows; press "Download results (CSV)" to save them.

### History and saving reports

Every check is listed on the History page with its time, type, input and verdict. Use the filter to show one type of check, the delete button to remove one entry, or "Clear history" to remove everything. On any report page, "Save as PDF" opens the browser's print dialog with a print layout that hides the navigation.

### Light and dark theme

The switch at the bottom of the left rail offers Auto, Light and Dark. Auto follows the operating system's setting. The choice is remembered for the next visit.

## Troubleshooting

TABLE: Common problems and solutions
| Problem | Solution |
|---|---|
| "Python 3.10 to 3.12 is required" | Install Python 3.12 and run the launcher again |
| "npm was not found" | Install Node.js LTS and open a new terminal |
| "Port 8000 (or 5173) is already in use" | Close the other TruthLens window, or restart the computer |
| Package installation fails | Check the internet connection and run the launcher again; it resumes |
| A model shows "Not trained" | Extract truthlens_models.zip into backend/models |
| The first claim check is slow | The NLI and embedding models are downloading; later checks are faster |
| Highlights take minutes | Lower LIME_NUM_SAMPLES in backend/.env, or switch the highlighting off |
| Need a clean reinstall | Delete backend/.venv and frontend/node_modules, then run the launcher |
