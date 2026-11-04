## Conclusion

This project set out to build a fake news checker that a reader in Andhra Pradesh or Telangana could actually use: one that works in Telugu as well as English, checks images as well as text, and shows its reasons instead of asking to be trusted. TruthLens AI meets these goals in a working application that installs with one command and runs on a laptop.

The English classifier, DeBERTa-v3-large fine-tuned on cleaned WELFake, reached 99.7% accuracy on 12,727 held-out articles, against 98.8% for a BERT-base model trained the same way and 96.2% for a TF-IDF model. The Telugu classifier, IndicBERT v2 fine-tuned on a translated set built in this project, reached 93.1%, well above the 87.1% that word counts alone achieve on the same data. The image detector, a linear head on frozen CLIP ViT-L/14 features, reached 93.7% on CIFAKE.

Two findings from the data work matter as much as the scores. The WELFake labels are the reverse of their documentation, and the dataset's publisher names and credits let a model learn the source rather than the content; both had to be fixed before any number was meaningful. The public Telugu dataset turned out to be separable at 99.8% by a word-count model because its classes come from different kinds of source. Reporting a high accuracy on it would have been easy and misleading. The project instead built a translated set and measured its remaining bias openly.

The claim verification stage is the part of TruthLens that does not depend on writing style. It splits an article into claims, retrieves evidence with multilingual embeddings, and uses NLI with a contradiction-first decision rule to label each claim Supported, Contradicted or Unknown. On FEVER it reached 84.6% accuracy and 0.847 macro-F1, while a system that labels claims by similarity alone reached 40.0% and could not detect a single contradiction. This comparison is the main evidence for the project's central design choice. On the harder Climate-FEVER set, made of real claims from the web, accuracy fell to 50.9%: the system still caught 79% of the false claims but marked most true claims as having no clear evidence. For Telugu claims, the multilingual NLI model reached 74.2% on IndicXNLI, against 40.0% for the English model.

The interface puts all of this in one report: the verdict in plain words, the influential words marked in the article, each claim underlined by its status, and the evidence printed beside it.

## Limitations

- **Evidence coverage.** Claim verification can only decide claims for which the corpus holds a relevant passage. The shipped corpus has 40 passages, so most claims in a random article will be Unknown. The verification method is sound; the evidence base is small.
- **Translated Telugu data.** The Telugu model was trained on machine-translated United States news. It has not been tested on native Telugu news, whose topics, names and style differ.
- **Remaining source bias.** Even after cleaning, a TF-IDF model scores 96.2% on WELFake, so part of the classifiers' accuracy still comes from the style of the sources. The very high English score should be read with this in mind.
- **Low-resolution images.** CIFAKE images are 32 by 32 pixels from one generator. The detector has not been tested on full-resolution images or on newer generators such as Midjourney or DALL-E 3.
- **Speed on CPU.** The English classifier and its LIME explanation are slow on a laptop CPU. The interface shows the verdict first to hide this, but highlights for a long article can take several minutes.
- **Real-world claims.** On Climate-FEVER only 15% of supported claims were recognised as supported. The NLI model needs close entailment, which evidence about real-world claims often does not give.
- **Telugu verification.** The multilingual NLI model reached 74.2% on Telugu NLI pairs, so Telugu claim labels are less reliable than English ones.
- **Two languages.** Hindi, Tamil and other Indian languages are not supported by the classifiers.

## Future Work

1. **Larger evidence base.** Load Wikipedia in English and Telugu, the PIB Fact Check archive and the Google Fact Check Tools results into the index, and refresh it on a schedule so that recent claims can be checked.
2. **Native Telugu data.** Collect and label Telugu articles from fact-checking sites and regional news, with both classes drawn from comparable sources, and use it to test and retrain the Telugu model.
3. **More languages.** IndicBERT v2, BGE-M3 and the multilingual NLI model already cover Hindi, Tamil, Kannada and other languages, so adding a language needs mainly training data and a script check in the language detector.
4. **Modern image detection.** Train the CLIP head on full-resolution images from several current generators, and add checks for edited photographs as well as generated ones.
5. **Image and text together.** Check whether an image matches the claim in its caption, using CLIP's shared image and text space, since many false posts pair a real photo with a false description.
6. **Faster explanations.** Replace LIME with a gradient-based method such as integrated gradients, which needs one or two model passes instead of 200, or distil the English model into a smaller one for CPU use.
7. **Weighted fusion.** Let a strong contradiction outweigh the classifier instead of averaging the two, since a calm, well-written false claim can receive a confident Real from the classifier (Section 8.6).
8. **Browser extension and messaging bot.** Let readers check a message from inside WhatsApp or a news page without copying it into the web application.
