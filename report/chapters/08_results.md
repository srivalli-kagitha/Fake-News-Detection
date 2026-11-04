This chapter reports how well each part of TruthLens works. All numbers come from data the models never saw during training: the held-out test splits for the three classifiers, and public benchmarks plus a hand-written claim set for claim verification. Every figure and table can be regenerated from the notebook outputs and from `evaluation/eval_claims.py`.

## English Text Classification

TABLE: Test results of the text and image classifiers
| Model | Test set | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | WELFake (cleaned) | 0.9615 | n/a | n/a | 0.9571 | n/a |
| BERT-base (baseline) | WELFake, 12,727 articles | 0.9884 | 0.9901 | 0.9844 | 0.9872 | 0.9993 |
| DeBERTa-v3-large (proposed) | WELFake, 12,727 articles | 0.9972 | 0.9979 | 0.9960 | 0.9970 | 0.9999 |
| TF-IDF + logistic regression | WELFake-Telugu | 0.8710 | n/a | n/a | 0.8748 | n/a |
| IndicBERT v2 | WELFake-Telugu, 2,000 articles | 0.9305 | 0.9104 | 0.9550 | 0.9322 | 0.9832 |
| CLIP ViT-L/14 + linear head | CIFAKE, 10,000 images | 0.9371 | 0.9367 | 0.9376 | 0.9371 | 0.9839 |

Precision, recall and F1 are for the positive class (Fake, or AI-generated). The TF-IDF rows come from the bias check, which uses its own 80/20 split of 30,000 English or 10,000 Telugu rows, so they are a reference point rather than a matched comparison.

DeBERTa-v3-large made about 36 mistakes on 12,727 test articles, against about 148 for BERT-base, roughly a quarter as many. Of DeBERTa's errors, about 23 were fake articles it called real and about 12 were real articles it called fake. The model is slightly more likely to miss a fake article than to raise a false alarm, which is the safer direction for a tool that always shows its evidence next to the verdict.

![Confusion matrix of DeBERTa-v3-large on the English test set](figures/cm_text_en.png){w=9}

![Training and validation loss of DeBERTa-v3-large](figures/curve_text_en.png){w=12}

The training loss falls from 0.67 to below 0.05 within the first half epoch and stays near zero after that. The validation loss, measured at the end of each epoch, is about 0.02 after the first epoch and slightly lower after the second, so the model did not over-fit in the two epochs it was trained for.

The very high scores need to be read alongside the TF-IDF result. A model that only counts words reaches 96.2% on the same data, so the cleaned WELFake still contains strong style signals. DeBERTa's gain over TF-IDF, from 3.9% error to 0.3%, shows that it learns more than vocabulary. It does not show that the model would reach 99.7% on fake news written in a different style, which is why TruthLens does not rely on the classifier alone.

## Telugu Text Classification

IndicBERT v2 reached 93.1% accuracy and an F1 of 0.932 on 2,000 translated test articles, six points above the 87.1% of the TF-IDF model. The gap between the English and Telugu results has three likely causes:

1. **Less data.** The Telugu model was trained on 7,200 articles, the English model on 18,000 (the rest of each training part is held back for validation).
2. **Translation noise.** NLLB-200 sometimes drops or garbles names and numbers, and every article was cut to its first 120 words before translation, so the Telugu model sees less of each article.
3. **Smaller model.** IndicBERT v2 is a base-sized model, while DeBERTa-v3-large is a large one.

TABLE: Per-class results of the Telugu classifier
| Class | Precision | Recall | F1 | Test articles |
|---|---|---|---|---|
| Real | 0.9527 | 0.9060 | 0.9288 | 1,000 |
| Fake | 0.9104 | 0.9550 | 0.9322 | 1,000 |

The Telugu model's errors lean the other way from the English model's. It catches 95.5% of fake articles but calls about 94 of the 1,000 real ones fake (real recall 90.6%). For a reader this means a Telugu "Likely fake" verdict deserves a look at the claim evidence before it is trusted.

![Confusion matrix of IndicBERT v2 on the Telugu test set](figures/cm_text_te.png){w=9}

![Training and validation loss of IndicBERT v2](figures/curve_text_te.png){w=12}

The Telugu curve tells a different story from the English one. The validation loss is lowest after the second epoch (about 0.22) and rises after that to about 0.30 by the fourth, while the training loss keeps falling. The model starts to over-fit the 8,000 training articles after two epochs. Because the trainer keeps the checkpoint with the best validation F1 rather than the last one, the saved model is not the over-fitted final state, but a future run could stop after two or three epochs with no loss in quality.

## AI-Generated Image Detection

The CLIP linear head reached 93.7% accuracy on 10,000 CIFAKE test images, with almost equal results for both classes: 93.7% recall on real photographs and 93.8% on generated images. Only 1,538 parameters were trained, and the whole training step after feature extraction took a few minutes, which shows how much of the work CLIP's pre-trained features already do.

TABLE: Per-class results of the image detector
| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Real | 0.9375 | 0.9366 | 0.9371 | 5,000 |
| AI-generated | 0.9367 | 0.9376 | 0.9371 | 5,000 |

![Confusion matrix of the CLIP image detector on CIFAKE](figures/cm_image.png){w=9}

CIFAKE images are 32 by 32 pixels and are enlarged to 224 by 224 for CLIP, so most fine detail is lost before the model sees the image. The 93.7% is therefore a lower bound on what the approach can do with full-resolution images, but also a result for only one generator.

## Claim Verification

Claim verification was evaluated in three ways. On FEVER and Climate-FEVER the gold evidence sentences are given with each claim, so the test measures the decision stage alone: given the evidence, does the system reach the right label? On the curated set the full pipeline runs, retrieval included, over the project's own evidence corpus. Every benchmark is balanced: FEVER has 300 claims of each label, Climate-FEVER 150 of each. The results are compared with a similarity-only baseline that labels a claim Support whenever a passage with cosine similarity of at least 0.45 exists and Unknown otherwise, which is what a retrieval system without NLI can do.

TABLE: Claim verification results (threshold τ = 0.5)
| Benchmark | Method | Accuracy | Macro-F1 | F1 Support | F1 Contradiction | F1 Unknown | Claims |
|---|---|---|---|---|---|---|---|
| FEVER (dev) | Similarity only | 0.400 | 0.292 | 0.539 | 0.000 | 0.336 | 900 |
| FEVER (dev) | TruthLens | 0.846 | 0.847 | 0.895 | 0.855 | 0.791 | 900 |
| Climate-FEVER | Similarity only | 0.340 | 0.181 | 0.503 | 0.000 | 0.039 | 450 |
| Climate-FEVER | TruthLens | 0.509 | 0.467 | 0.256 | 0.663 | 0.482 | 450 |
| Curated set | Similarity only | 0.525 | 0.487 | 0.628 | 0.000 | 0.833 | 40 |
| Curated set | TruthLens | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 40 |

### FEVER

On FEVER the decision rule labels 84.6% of the 900 claims correctly, with F1 between 0.79 and 0.90 for every class. The confusion matrix shows where the errors are. Supported and refuted claims are almost never confused with each other (2 supported claims called contradicted, 5 refuted claims called supported). Most errors involve Unknown: 37 supported and 45 refuted claims were marked Unknown because the NLI model was not confident enough, and 50 NotEnoughInfo claims were given a definite label.

![Confusion matrix of claim verification on FEVER](figures/cm_claims_fever.png){w=8.5}

The similarity baseline shows why the NLI step is needed. It calls 287 of the 300 supported claims Support, but it also calls 251 of the 300 refuted claims Support, because a passage that refutes a claim is about the same subject and is just as similar to it. It finds no contradiction at all, since similarity has no notion of disagreement.

### Climate-FEVER

Climate-FEVER is much harder. Its claims were collected from the web rather than written from Wikipedia, and its evidence sentences are related to the claim without always settling it. Accuracy falls to 50.9%. The errors are lopsided: TruthLens still catches 119 of the 150 refuted claims (recall 79%), but it labels only 23 of the 150 supported claims Support (recall 15%); most of the rest come back Unknown. When TruthLens does say Support, it is right 77% of the time.

![Confusion matrix of claim verification on Climate-FEVER](figures/cm_claims_climate.png){w=8.5}

For a fact-checking aid this is the safer way to fail. A true claim marked "No evidence found" costs the reader a look at the evidence, while a false claim marked Supported would mislead them. The result also shows that the 84.6% on FEVER should not be read as the accuracy on real-world claims.

### Telugu verification

The English NLI model was never trained on Telugu, so both NLI models were tested on 1,000 Telugu premise and hypothesis pairs from IndicXNLI [25], labelled entailment, neutral or contradiction.

TABLE: Three-class NLI on IndicXNLI Telugu (1,000 pairs)
| NLI model | Accuracy | Macro-F1 |
|---|---|---|
| DeBERTa-v3-large, English only | 0.400 | 0.396 |
| mDeBERTa-v3-base, multilingual (used for Telugu) | 0.742 | 0.745 |

The English model, though larger and much stronger on English, is barely above the 33% of random guessing on Telugu. The multilingual model reaches 74.2%. This is why Telugu claims are sent to mDeBERTa, and it also shows that verification in Telugu is less reliable than in English.

![Confusion matrix of mDeBERTa-v3-base on IndicXNLI Telugu](figures/cm_te_nli.png){w=8.5}

### Curated set (full pipeline)

On the 40 curated claims (30 English, 10 Telugu), the full pipeline labelled every claim correctly, including all 17 contradicted claims, while the similarity baseline marked all 17 contradicted claims as supported. Two examples from the predictions file show the difference:

TABLE: Two curated claims and their predictions
| Claim | Gold label | TruthLens | Similarity only | Top evidence |
|---|---|---|---|---|
| India became independent from Britain in August 1947. | Support | Support | Support | India gained independence from British rule on 15 August 1947. |
| India gained independence from British rule in 1950. | Contradiction | Contradiction | Support | India gained independence from British rule on 15 August 1947. |

Both claims retrieve the same passage with high similarity. Only the NLI step notices that 1950 disagrees with 1947.

The 100% must be read carefully. The curated claims were written for this project against its own 40-passage corpus, so they test that retrieval, language routing and the decision rule work together. They are not a measure of accuracy on unseen claims, for which FEVER and Climate-FEVER are the better guide.

### Choice of threshold

![Macro-F1 of the decision rule as the threshold changes](figures/threshold_sweep.png){w=10}

The threshold τ decides how confident the NLI model must be before a claim is labelled Support or Contradiction. On FEVER the best macro-F1 is 0.859 at τ = 0.35, slightly above the 0.847 at the default of 0.5, and it falls to 0.787 at τ = 0.9 as more claims become Unknown. On Climate-FEVER the curve is flat between 0.455 and 0.467, so its weak Support recall comes from the NLI scores themselves and cannot be fixed by moving the threshold. The default of 0.5 was fixed before the evaluation and was kept, because a lower value would let weaker support through in the application.

## Sample Outputs

The screenshots in this section were taken automatically with Playwright from the running application, with the trained models installed. Each shows a real result, not a mock-up.

![Start page](figures/ss_home.png){w=13}

The Start page has a single text box and an image drop area. The left rail lists the tools and, at the bottom, the status of each model and the theme switch.

![Report for an English rumour about 5G towers](figures/ss_text_report.png){w=15}

For the 5G rumour the report gives "Likely fake" with 87% confidence. In the article, red marks show the words that pushed toward fake, among them "BREAKING", "Share" and "deleted!". The claim list below shows that the first claim, that 5G towers spread the coronavirus, is contradicted by the evidence passage, while the claim that the government ordered towers shut down has no evidence in the corpus and is marked "No evidence found". The overall verdict combines both sources.

![Hovering a claim lights up its sentence in the article](figures/ss_text_claims_hover.png){w=8}

![Report for an article that mixes true and false claims](figures/ss_text_mixed.png){w=15}

The article in this report mixes one true claim (the Chandrayaan-3 landing) with two hoaxes (the GPS chip and the UNESCO anthem award). The claim report gets all three right: two contradicted, one supported, and the call to forward the message marked as having no evidence. The headline verdict, however, is "Likely real" at 69%, because the classifier is confident that this calmly written text is real and the fused risk is the average of the two sources. This is the weakness discussed in Section 8.6, and it is the reason the claim list is shown on the same page as the verdict.

![Report for a Telugu message](figures/ss_text_telugu.png){w=15}

The Telugu message is detected as Telugu, sent to the IndicBERT v2 classifier and to the multilingual NLI model, and checked against the English evidence passages. Its two true statements about Telangana are supported and the 5G rumour is contradicted.

![Image report for a generated image, with the Grad-CAM heatmap](figures/ss_image_report.png){w=14}

![Image report for a real photograph](figures/ss_image_real.png){w=14}

The image page shows the verdict on the same scale as the text page, and a slider that wipes between the image and its Grad-CAM heatmap. The CIFAKE images are 32 by 32 pixels, so they appear blurred when enlarged.

![Verify claims page](figures/ss_claims.png){w=15}

![Batch check of three texts, with download](figures/ss_batch.png){w=15}

![History of past checks](figures/ss_history.png){w=15}

![Report in the dark theme](figures/ss_text_dark.png){w=15}

![Start page on a phone (375 px wide)](figures/ss_mobile.png){w=6}

![Interactive API documentation generated by FastAPI](figures/ss_api_docs.png){w=14}

## Discussion

The results support three conclusions and set two warnings.

**The classifiers work, but on data that is easier than real news.** DeBERTa-v3-large reached 99.7% on English and IndicBERT v2 93.1% on Telugu. Both beat their baselines, and both beat a word-count model by a clear margin. But a word-count model reaches 96.2% on the same English data, which means WELFake's fake and real articles still differ in style even after cleaning. A classifier trained on it learns that style. That is useful when a new fake article is written like the old ones, and of little use when it is written carefully.

**Explicit contradiction detection is the most important design decision.** The similarity baseline failed to find a single refuted claim on any benchmark, while TruthLens found 83% of them on FEVER and 79% on Climate-FEVER. A retrieval system that ranks evidence by similarity tells the reader what the evidence is about, not whether it agrees. Testing for contradiction first, with an NLI model, is what turns retrieved passages into a verdict.

**Claim verification is weakest on real-world support.** On Climate-FEVER the system rarely confirms a true claim. It prefers "No evidence found" to a wrong confirmation, which suits a tool meant to make readers look at evidence, but it means a true article with unusual phrasing will often get few Supported claims. Better NLI models trained on real-world evidence, or several evidence passages combined before the decision, are the most promising fixes.

**A worked example shows both strengths and a weakness.** Given the single sentence "The new 2000 rupee notes contain a GPS nano chip that lets the government track them, officials confirmed on Monday.", the English classifier answered Real with 99.98% confidence: the sentence is calm, specific and attributed to officials, which is how real reports in WELFake are written. The claim check retrieved the passage stating that the notes contain no chip, and the NLI model scored it as a contradiction with probability 1.00. The fused fake risk was 0.5001, so the final verdict was Fake, but only just. The example shows why the evidence stage matters, and it also shows the limit of simple averaging: a confident classifier can almost cancel a clear contradiction. Giving a strong contradiction more weight than the classifier, rather than averaging the two, is an obvious improvement for future work.

The first warning concerns the evidence corpus. Every claim result in the application depends on what the corpus contains. With 40 passages, most claims in an arbitrary article will be Unknown, and the fused verdict will then rest on the classifier alone. The method is sound; the coverage is small.

The second concerns Telugu. The Telugu classifier was trained and tested on translated articles about United States news, and the Telugu NLI model reaches 74.2%. The Telugu results in this chapter are therefore a measure of the method, not a promise about performance on native Telugu misinformation.

TABLE: Summary of results
| Component | Test data | Main result | Compared with |
|---|---|---|---|
| English classifier (DeBERTa-v3-large) | WELFake, 12,727 articles | 99.7% accuracy, F1 0.997 | BERT-base 98.8%, TF-IDF 96.2% |
| Telugu classifier (IndicBERT v2) | WELFake-Telugu, 2,000 articles | 93.1% accuracy, F1 0.932 | TF-IDF 87.1% |
| Image detector (CLIP + linear head) | CIFAKE, 10,000 images | 93.7% accuracy, AUC 0.984 | n/a |
| Claim verification | FEVER, 900 claims | 84.6% accuracy, macro-F1 0.847 | Similarity only 40.0% |
| Claim verification | Climate-FEVER, 450 claims | 50.9% accuracy, macro-F1 0.467 | Similarity only 34.0% |
| Telugu NLI (mDeBERTa) | IndicXNLI, 1,000 pairs | 74.2% accuracy | English NLI model 40.0% |
| Full pipeline | Curated set, 40 claims | 40 of 40 correct | Similarity only 21 of 40 |
