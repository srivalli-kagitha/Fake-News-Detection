# Training the TruthLens AI models (free Google Colab GPU)

**Everything is in one notebook: `TruthLens_Training.ipynb`.** No Kaggle account, API key or login
is needed. All datasets and models download automatically from Zenodo and Hugging Face.

## How to run (browser Colab, recommended)

1. Open https://colab.research.google.com → **Upload** → choose `TruthLens_Training.ipynb`
2. **Runtime → Change runtime type → T4 GPU → Save**
3. **Runtime → Run all**
4. When asked, allow Google Drive access. Trained models are then saved to `MyDrive/truthlens/`,
   so they survive a disconnect.

**First do a quick test.** The notebook starts with `QUICK_TEST = True` (about 25 minutes), which checks
that everything works. Then set `QUICK_TEST = False` in the CONFIG cell and **Run all** again for the
real training (about 3.5–4.5 hours in total).

**VS Code:** install the **Jupyter** and **Google Colab** extensions, open the notebook,
choose **Select Kernel → Colab → New Colab Server → GPU (T4)**, then **Run All**.
If Google Drive can't be mounted there, download `truthlens_models.zip` before you close the session.

## What gets trained

| Model | Dataset (auto-download) | Full-run time on T4 | Goes to |
|---|---|---|---|
| Image: CLIP ViT-L/14 + head | CIFAKE, 100k images (Hugging Face, MIT) | ~20–40 min | `backend/models/image_detector` |
| English: **DeBERTa-v3-large** (proposed) | WELFake, 72k articles (Zenodo, CC BY 4.0) | ~1.5–2.5 h | `backend/models/text_en` |
| English: BERT-base (baseline, comparison table only) | WELFake | ~30–60 min | not used by the app |
| Telugu: IndicBERT v2 | 10k WELFake articles translated to Telugu with NLLB-200 (built in the notebook) | ~30–40 min translation + ~20 min training | `backend/models/text_te` |

Switch any model on or off with the `TRAIN_*` settings in the CONFIG cell. Models that are already
trained are **skipped automatically**, so after a disconnect just press **Run all** again.

## After training

The last cell creates **`truthlens_models.zip`**, which also appears in `MyDrive/truthlens/`.
On your PC, unzip it into:

```
truthlens-ai/backend/models/
```

Refresh the app. The Home page should show the models as **Ready**.

## Results for the report / paper

Cell 11 prints a ready-made results table (also saved as `outputs/results_table.md`) and shows the
confusion matrices. Each model folder also contains:
- `metrics.json`: accuracy, precision, recall, F1, ROC-AUC
- `classification_report.txt`: per-class scores
- `confusion_matrix.png`: put this in the Results chapter
- `training_curve.png` (text models): loss curve

## Dataset notes (mention these in the paper)

- **WELFake** (Verma et al., 2021), DOI 10.5281/zenodo.4561253. **In the actual file 1 = fake and 0 = real**,
  the opposite of its documentation. Reuters and New York Times articles are label 0, and clickbait
  ("BREAKING", "Featured image via…") is label 1. Publisher giveaways ("(Reuters)", "- The New York Times",
  "Featured image via…", "Breitbart") are removed before training. Without this, about 60% of real articles
  start with "(Reuters)" and the model just learns that word.
- **Telugu: built by translation.** 10,000 balanced WELFake articles (headline + first sentences, ≤120 words)
  translated with **NLLB-200** (Meta, `facebook/nllb-200-distilled-1.3B`, CC BY-NC 4.0, academic use).
  - *Why not the public Telugu dataset?* In the Zenodo Telugu dataset (Patil et al., 2024,
    DOI 10.5281/zenodo.11408513), the fake class is machine-translated fact-check reports and the real class
    is native-Telugu health and lifestyle news. A simple TF-IDF word model scores **99.8%** on it, using words
    like మరియు/యొక్క (translation style), బూమ్ (the BOOM fact-check site) and విటమిన్/గుండె (health topics).
    It separates writing styles, not fake from real. Mention this in the paper's dataset section.
  - *Limitation to state honestly:* the Telugu data is translated, not written natively. A native
    Telugu fake-news dataset (e.g. DFND, via IEEE DataPort) would be the next step.
- **Bias check:** cell 6 runs the same TF-IDF test on the final English and Telugu data
  (`outputs/bias_check.md`). Put this table in the paper.
- **CIFAKE** (Bird & Lotfi, 2024), arXiv:2303.14126. Real images come from CIFAR-10, fake images from
  Stable Diffusion v1.4. 20% of the images are held out as the test set.

## Troubleshooting

| Problem | Fix |
|---|---|
| "NO GPU" | Runtime → Change runtime type → T4 GPU |
| Colab disconnected | Press **Run all** again. Finished models are skipped |
| "You cannot currently connect to a GPU" | Free GPU quota is used up. Wait a day, or use Kaggle Notebooks |
| DeBERTa loss is `nan` | Add `--no-fp16` in the cell 8 command |
| Out of memory | Lower `--batch-size` to 4 and raise `--grad-accum` to 8 in cell 8 |
| A download fails | Re-run the cell. Zenodo and Hugging Face are sometimes slow |

## Running the scripts without the notebook

The notebook writes the six files in this folder (`prepare_dataset.py`, `translate_dataset.py`,
`bias_check.py`, `train_text_classifier.py`, `train_image_detector.py`, `requirements.txt`). Each one can also be run directly. Use `--help` to see
all options.
