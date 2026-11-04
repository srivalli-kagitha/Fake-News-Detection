This chapter describes how TruthLens AI is put together: its architecture, its modules, the UML diagrams that show how the parts interact, the data flow, the database and the programming interface between the frontend and the backend.

## System Architecture

TruthLens follows a three-tier architecture. The presentation tier is a React single-page application served by Vite on port 5173. The application tier is a FastAPI service on port 8000 that holds all the machine learning models. The data tier is a SQLite database for the check history and a folder of JSON-lines files with a FAISS index for the evidence corpus. The browser never talks to the backend directly: Vite forwards every request under `/api` to port 8000, so the frontend and backend share one origin and no cross-origin configuration is needed on the user's machine.

![Overall architecture of TruthLens AI](figures/architecture.png){w=14}

Inside the backend, the three pipelines are independent Python modules. Each model is wrapped in a `Lazy` object that loads it on the first request and keeps it in memory afterwards, guarded by a lock so that two simultaneous requests do not load the same model twice. This keeps start-up fast and means a user who only checks images never loads the 1.7 GB English text model.

## Module Description

### User interface module

The interface has six pages, all reachable from a tool rail on the left (a top bar on phones):

- **Start.** One text box and one image drop area. Text is sent to the text check, an image to the image check.
- **Check news text.** The input form on the left and the report on the right: verdict and fake-risk scale, the article with highlighted words and underlined claims, and the claim list with evidence.
- **Check an image.** Image preview, verdict scale, and a slider that wipes between the original image and its Grad-CAM heatmap.
- **Verify claims.** Claim verification on its own, without the classifier.
- **Check many at once.** Upload a CSV of texts or a set of images and download the results.
- **History.** Every past check with its result, filterable by type.

Results are kept in memory when the user moves between pages, so a report is not lost by switching tools.

### Language detection module

Telugu letters occupy the Unicode block U+0C00 to U+0C7F. The module counts the letters in the text and treats it as Telugu when more than 30% fall in that block. The threshold allows English words, numbers and names inside a Telugu article.

### Text classification module

This module loads the fine-tuned English or Telugu model, tokenises the text to at most 256 tokens and returns the probability of each class. It reads the label names from the model's own configuration file, so a model trained with a different label order still reports Fake and Real correctly.

### Explanation module

The text explainer runs LIME with 200 perturbed copies of the article and returns a weight for every word. The image explainer runs Grad-CAM on the last transformer block of CLIP and returns the heatmap blended with the original image as a PNG.

### Claim verification module

This module has four parts: the claim extractor, the evidence retriever, the NLI verifier and the aggregator. They are described as an algorithm in Section 5.6 and shown as an activity diagram in Figure 4.6.

### Image detection module

The image detector resizes and centre-crops the image to 224 by 224 pixels, passes it through the frozen CLIP ViT-L/14 encoder, and applies the trained linear head to the 768-dimensional embedding.

### Batch module

The batch module reads a CSV file with a `text` column (and an optional `language` column), or a list of images, or a ZIP archive of images. It runs the same models row by row, writes a result CSV under a random job identifier, and returns a summary with counts per label.

### History module

Every check is saved in the `checks` table with its type, a preview of the input, the verdict and the confidence. The history page lists, filters and deletes these records.

## UML Diagrams

### Use case diagram

The use case diagram shows the two kinds of actor. A user checks text, images and claims, runs batch checks and manages the history. An administrator, in practice a student maintaining the project, trains the models and adds evidence documents. Highlighting words extends the text check because it can be switched off; claim verification is included in it by default.

![Use case diagram](figures/usecase.png){w=13}

### Class diagram

The class diagram shows the main backend classes. `Settings` holds every configurable value, read from the `.env` file. `Lazy` wraps each model so it loads once. `TextClassifier`, `ImageDetector`, `NLIModel` and `EvidenceRetriever` each wrap one model, and the claim verifier module uses the last two. `CheckRecord` is the database table.

![Class diagram of the backend](figures/class.png){w=15}

### Sequence diagram: checking news text

The sequence diagram for a text check shows the two-step response. The first request returns the verdict and the claim evidence. A second request computes the LIME explanation and the highlights are added to the report already on screen, so the user is not kept waiting for the slowest step.

![Sequence diagram for checking news text](figures/sequence_text.png){w=15.5}

### Sequence diagram: checking an image

![Sequence diagram for checking an image](figures/sequence_image.png){w=13}

### Activity diagram: claim verification

The activity diagram follows one article through claim verification. The decision on contradiction comes before the decision on support, so a passage that clearly refutes a claim is not outweighed by passages that only partly agree with it.

![Activity diagram of claim verification](figures/activity_claims.png){w=11}

### Activity diagram: overall flow

![Activity diagram of the overall check](figures/activity_overall.png){w=12}

### Deployment diagram

The deployment diagram shows where each part runs. Training runs once in a cloud notebook and produces a ZIP of model files. Everything else runs on the student's laptop: the browser, the Vite server and the Python environment with the models and the database.

![Deployment diagram](figures/deployment.png){w=12}

## Data Flow Diagrams

The level 0 data flow diagram treats TruthLens as a single process. The user sends text, images or a CSV file and receives verdicts, explanations and evidence. Evidence sources and trained models are external inputs.

![Level 0 data flow diagram](figures/dfd0.png){w=12}

The level 1 diagram splits the system into its six processes. Language detection feeds both the classifier and the claim verifier. The fusion process combines their outputs and writes the result to the history store. Image detection runs on its own path.

![Level 1 data flow diagram](figures/dfd1.png){w=15}

## Database Design

The system has one relational table. Evidence and batch results are stored as files rather than tables, because evidence is maintained by hand as text files and batch results are meant to be downloaded.

![Entity relationship diagram](figures/er.png){w=10}

TABLE: Structure of the checks table
| Column | Type | Constraint | Description |
|---|---|---|---|
| id | INTEGER | Primary key, auto-increment | Record identifier |
| kind | VARCHAR(20) | Not null, indexed | text, image, claims, batch_text or batch_image |
| input_preview | TEXT | Not null | First part of the text, or the image file name |
| language | VARCHAR(8) | Nullable | en or te for text checks |
| verdict | VARCHAR(64) | Not null | Final label shown to the user |
| confidence | FLOAT | Nullable | Confidence of the verdict, 0 to 1 |
| details_json | TEXT | Default "{}" | Extra details such as probabilities and claim counts |
| created_at | DATETIME | Not null | Time of the check (UTC) |

Each evidence passage is one line of JSON with three fields, so students can add evidence with any text editor. The example below is wrapped to fit the page; in the file it is a single line:

```
{"text": "Indian 2000 rupee banknotes do not contain any GPS or nano chip; viral claims
          about a tracking chip were false.",
 "source": "Wikipedia",
 "url": "https://en.wikipedia.org/wiki/Indian_2000-rupee_note"}
```

## Interface Design

### REST API

TABLE: Backend API endpoints
| Method | Endpoint | Input | Output |
|---|---|---|---|
| POST | /api/text/analyze | text, language, explain, verify_claims | classification, verification, overall verdict |
| POST | /api/text/explain | text, language | LIME tokens and top words |
| POST | /api/claims/verify | text, language | claims with labels and evidence, summary |
| POST | /api/image/analyze | image file, explain | label, confidence, Grad-CAM overlay |
| POST | /api/batch/text | CSV file, language, verify_claims | job id, counts, results |
| POST | /api/batch/images | image files or ZIP | job id, counts, results |
| GET | /api/batch/{job_id}/download | job id | result CSV file |
| GET | /api/health | none | model status and metrics |
| GET | /api/history | limit, kind | list of past checks |
| DELETE | /api/history/{id} | record id | confirmation |
| DELETE | /api/history | none | confirmation |

### Response format

Every text check returns the same JSON structure, which the frontend uses to draw the report. The example below is a real response, shortened, for the input "The new 2000 rupee notes contain a GPS nano chip that lets the government track them, officials confirmed on Monday." with highlighting switched off:

```
{
  "language": "en",
  "classification": {"label": "Real", "confidence": 0.9998,
                     "fake_probability": 0.0002},
  "verification": {
    "claims": [
      {"claim": "The new 2000 rupee notes contain a GPS nano chip that ...",
       "label": "Contradiction", "confidence": 1.0,
       "evidence": [{"text": "Indian 2000 rupee banknotes do not contain any GPS ...",
                     "source": "Wikipedia", "similarity": 0.712,
                     "nli": {"entailment": 0.0, "neutral": 0.0,
                             "contradiction": 1.0}}]}
    ],
    "summary": {"verdict": "Likely Misleading", "credibility_score": 0.0001,
                "counts": {"Support": 0, "Contradiction": 1, "Unknown": 0}}
  },
  "overall": {"verdict": "Fake", "fake_risk": 0.5001,
              "based_on": ["classifier", "claim_evidence"]}
}
```

The classifier and the evidence disagree completely here, and Section 8.6 discusses what this example shows.

### User interface design

The interface was designed around the idea of a fact-checker's marked-up proof sheet. The article in a report is set in a serif typeface, like a newspaper. Influential words get highlighter-pen marks, red for words that push toward fake and green for words that push toward real, with stronger colour for more influence. Claims are underlined: a solid line for supported, a wavy line for contradicted and a dotted line for no evidence. Hovering a claim in the evidence list lights up its sentence in the article. Colour is never the only signal, because every verdict is also written in words.
