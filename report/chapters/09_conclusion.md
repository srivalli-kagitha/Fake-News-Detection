## Conclusion

This project set out to build a fake news checker that a reader in Andhra Pradesh or Telangana could actually use: one that works in Telugu as well as English, checks images as well as text, and shows its reasons instead of asking to be trusted. TruthLens AI meets these goals in a working application that installs with one command and runs on a laptop.

The English classifier, DeBERTa-v3-large fine-tuned on cleaned WELFake, reached 99.7% accuracy on 12,727 held-out articles, against 98.8% for a BERT-base model trained the same way and 96.2% for a TF-IDF model. The Telugu classifier, IndicBERT v2 fine-tuned on a translated set built in this project, reached 93.1%, well above the 87.1% that word counts alone achieve on the same data. The image detector, a linear head on frozen CLIP ViT-L/14 features, reached 93.7% on CIFAKE.

Two findings from the data work matter as much as the scores. The WELFake labels are the reverse of their documentation, and the dataset's publisher names and credits let a model learn the source rather than the content; both had to be fixed before any number was meaningful. The public Telugu dataset turned out to be separable at 99.8% by a word-count model because its classes come from different kinds of source. Reporting a high accuracy on it would have been easy and misleading. The project instead built a translated set and measured its remaining bias openly.

The claim verification stage is the part of TruthLens that does not depend on writing style. It splits an article into claims, retrieves evidence with multilingual embeddings, and uses NLI with a contradiction-first decision rule to label each claim Supported, Contradicted or Unknown. On FEVER it reached 84.6% accuracy and 0.847 macro-F1, while a system that labels claims by similarity alone reached 40.0% and could not detect a single contradiction. This comparison is the main evidence for the project's central design choice. On the harder Climate-FEVER set, made of real claims from the web, accuracy fell to 50.9%: the system still caught 79% of the false claims but marked most true claims as having no clear evidence. For Telugu claims, the multilingual NLI model reached 74.2% on IndicXNLI, against 40.0% for the English model.

For claims in real use, the NLI method was joined by a language-model verifier: gpt-oss-120b plans searches, live evidence is gathered from Wikipedia, the web, Google News and published fact-checks, and the model gives a verdict with a reason and the passages it relied on. On 41 single-sentence claims through the full application it was right on all 41, against 35 for the NLI method, in 13.8 seconds per claim. Two detectors are now averaged for images, which judged 50 of 61 recent and CIFAKE images correctly, and occlusion replaced LIME for the word highlights, as faithful and about ten times faster.

The interface puts all of this in one report: the verdict in plain words, the influential words marked in the article, each claim underlined by its status, and the evidence printed beside it with the passages that decided it marked. The application loads its models in the background at start-up, shows the steps of a running check, keeps results while the user moves between pages, and keeps every report in a history that can be reopened.

## Limitations

- **Outside services.** The best claim checks need the internet, the search engines and a language-model API. The free Groq tier allows about 500 claims a day; offline, the NLI method with the 40-passage corpus decides far fewer claims.
- **Small live test.** The 41 test sentences were written by the team, and the language model may know many of the facts from training. The citations let a reader check each answer, but a larger, independent test set such as AVeriTeC is needed.
- **Unknown claims in the fusion.** A claim with no evidence counts as zero in the credibility score, so an article with two refuted claims and one unverifiable one can still come out as "Likely real" (Section 8.6).
- **Translated Telugu data.** The Telugu model was trained on machine-translated United States news. It has not been tested on native Telugu news, whose topics, names and style differ.
- **Remaining source bias.** Even after cleaning, a TF-IDF model scores 96.2% on WELFake, so part of the classifiers' accuracy still comes from the style of the sources. The very high English score should be read with this in mind.
- **Images from new generators.** Even with two detectors, 7 of 24 Midjourney v6 images were called real. An image verdict is a warning, not proof.
- **Speed on CPU.** A claim check takes about 6 to 15 seconds and highlights for an 80-word article about 10 seconds on a laptop CPU; the first start downloads about 2 GB of models.
- **Real-world claims.** On Climate-FEVER only 15% of supported claims were recognised as supported. The NLI model needs close entailment, which evidence about real-world claims often does not give.
- **Telugu verification.** The multilingual NLI model reached 74.2% on Telugu NLI pairs, so Telugu claim labels are less reliable than English ones.
- **Two languages.** Hindi, Tamil and other Indian languages are not supported by the classifiers.

## Future Work

1. **Local language model.** Run a small open model such as Qwen3-4B in Ollama by default, as ClaimCheck [52] does, so that the best checker works without a key or a daily limit.
2. **Native Telugu data.** Collect and label Telugu articles from fact-checking sites and regional news, with both classes drawn from comparable sources, and use it to test and retrain the Telugu model.
3. **More languages.** IndicBERT v2, BGE-M3 and the multilingual NLI model already cover Hindi, Tamil, Kannada and other languages, so adding a language needs mainly training data and a script check in the language detector.
4. **Modern image detection.** Train the CLIP head on full-resolution images from several current generators, and add checks for edited photographs as well as generated ones.
5. **Image and text together.** Check whether an image matches the claim in its caption, using CLIP's shared image and text space, since many false posts pair a real photo with a false description.
6. **Evaluation on AVeriTeC.** Test the language-model verifier on the AVeriTeC benchmark [54] of real claims with web evidence, and on Telugu claims from Indian fact-checking sites.
7. **Better fusion.** Leave Unknown claims out of the credibility score, or let any refuted claim with high confidence decide the article, since a calm, well-written false article can still receive a confident Real from the classifier (Section 8.7).
8. **Browser extension and messaging bot.** Let readers check a message from inside WhatsApp or a news page without copying it into the web application.
