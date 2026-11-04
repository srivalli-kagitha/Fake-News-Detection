# Trained models go here

Copy the output folders from the training scripts (see `training/README.md`) into this folder:

```
backend/models/
├── text_en/          <- DeBERTa-v3-large fine-tuned on English fake news
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files ...
├── text_te/          <- IndicBERT v2 / mmBERT fine-tuned on Telugu fake news
│   └── ...
└── image_detector/   <- CLIP ViT-L/14 + trained head
    ├── config.json
    └── head.pt
```

The backend notices new models without a restart. Until a model is here, its page in the app
shows a "model not trained yet" message. Claim verification works right away, because it uses
pre-trained NLI models.
