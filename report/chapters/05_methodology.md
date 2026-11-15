This chapter explains how each part of TruthLens works: the data the models learn from, how the data was cleaned, the models and their training, the explanation methods, the claim verification algorithm, and the measures used to evaluate everything.

## Overview

The methodology has two halves. The trained half covers three classifiers (English text, Telugu text and images) that need labelled data and a GPU to train. The pre-trained half covers claim verification, which uses a language model, an embedding model and NLI models exactly as they were released, with no training in this project. The two halves meet in the fusion step, which combines a classifier's fake probability with the credibility of the article's claims.

![Training pipeline run in the notebook](figures/training_pipeline.png){w=9.5}

## Datasets

TABLE: Datasets used in this project
| Dataset | Purpose | Source and licence | Size used |
|---|---|---|---|
| WELFake [4] | English classifier | Zenodo 4561253, CC BY 4.0 | 20,000 train, 12,727 test articles |
| WELFake-Telugu (built here) | Telugu classifier | Translated from WELFake with NLLB-200 | 8,000 train, 2,000 test articles |
| CIFAKE [19] | Image detector | Hugging Face, MIT | 40,000 train, 10,000 test images |
| FEVER [23] dev | Claim verification test | Hugging Face (gold evidence) | 900 claims |
| Climate-FEVER [24] | Claim verification test | Hugging Face | 450 claims |
| IndicXNLI [25] Telugu | Telugu NLI test | Hugging Face | 1,000 pairs |
| Curated claims (built here) | End-to-end test | Written for this project | 40 claims |

### WELFake

WELFake has 72,134 news articles with a title, a body and a binary label. It was chosen because it is large, roughly balanced and downloads without an account. Two problems had to be fixed before it could be used.

**The labels are the reverse of the documentation.** The dataset description says label 0 is fake. Reading the rows shows the opposite: articles from Reuters and The New York Times carry label 0, and articles beginning with "BREAKING" or ending with "Featured image via" carry label 1. A first quick training run that followed the documentation produced a model that called Reuters reports fake. All later runs use 1 for fake.

**Publisher names give the label away.** About 60% of the real articles begin with a dateline such as "WASHINGTON (Reuters) -". Fake articles often end with credits such as "Featured image via Getty Images" or "Follow 21WIRE on Twitter". A model can reach high accuracy by learning these strings alone and would then fail on any article without them. The cleaning step removes datelines, the words "Reuters" and "The New York Times", photo credits, "21WIRE" and "Breitbart" tags, and "Follow ... on Twitter" lines, then drops duplicate and empty rows. About 63,600 articles remain.

The cleaned data is split 80/20 with the class balance kept the same in both parts (stratified). For training, 20,000 articles are drawn from the 80% part, and 10% of those (2,000) are held back for validation. The test set is the full 20% part, 12,727 articles, which the model never sees during training.

### Telugu: why the public dataset was rejected

The obvious choice for Telugu was the dataset published on Zenodo in 2024 [13]. Before training on it, a simple test was run: a TF-IDF word-count model with logistic regression, which understands nothing about meaning, was trained to separate its two classes. It scored 99.8%.

The strongest features explained why. The fake class consists of fact-check reports machine-translated into Telugu, and the real class of native Telugu health and lifestyle articles. The words that best predicted "fake" were translation-style function words such as మరియు ("and") and యొక్క ("of"), and the name of the fact-checking site బూమ్ (BOOM). The words that best predicted "real" were health terms such as విటమిన్ (vitamin) and గుండె (heart). A model trained on this data would learn to recognise translated text and health articles, not false ones.

### Telugu: building a translated set

To give both classes the same writing style, a balanced sample of 10,000 articles (5,000 fake and 5,000 real) was taken from the WELFake training part and translated with NLLB-200 [12] (distilled, 1.3 billion parameters). Each article was cut to its headline and leading sentences, at most 120 words, and translated one sentence at a time, which NLLB handles more reliably than long passages. The script saves progress every few batches, so a disconnected session resumes where it stopped.

The same TF-IDF test on this set scores 87.1%, down from 99.8%. It does not fall to 50% because the topics of WELFake carry over: the strongest fake-class features are ట్రంప్ (Trump), హిల్లరీ (Hillary) and "(వీడియో)" ("(video)"), and the strongest real-class features are reporting verbs such as చెప్పారు ("said") and weekday names. The Telugu model therefore has to learn more than vocabulary to beat 87.1%. The set is translated, not native Telugu, and its articles are about United States politics, which is a limitation discussed in Chapter 8.

TABLE: Dataset bias check with a TF-IDF and logistic regression model
| Dataset | Rows used | Accuracy | F1 | Strongest fake-class words | Strongest real-class words |
|---|---|---|---|---|---|
| Zenodo Telugu [13] | Full set | 99.8% | n/a | మరియు, యొక్క, బూమ్ | విటమిన్, గుండె |
| WELFake (cleaned) | 30,000 | 96.2% | 0.957 | [video], hillary, october, 2016 | said, u.s., president, minister |
| WELFake-Telugu (ours) | 10,000 | 87.1% | 0.875 | ట్రంప్, హిల్లరీ, (వీడియో) | చెప్పారు, మంగళవారం, అధ్యక్షుడు |

### CIFAKE

CIFAKE contains 32 by 32 pixel images: real photographs from the CIFAR-10 dataset and images generated by Stable Diffusion v1.4 [20] to look like the same ten classes. Its training split has 50,000 images of each kind. This project samples 25,000 of each and holds out 20%, giving 40,000 training and 10,000 test images. Using half the training split keeps feature extraction inside one free GPU session.

## Preprocessing

TABLE: Preprocessing steps
| Data | Step | Reason |
|---|---|---|
| WELFake | Join title and body | Headlines carry much of the signal |
| WELFake | Set label 1 = fake | Labels in the file are reversed from the documentation |
| WELFake | Remove datelines, publisher names and credits | Prevents learning the source instead of the content |
| WELFake | Drop empty and duplicate texts | Avoids the same article in train and test |
| WELFake | Stratified 80/20 split, seed 42 | Same class balance in train and test; repeatable |
| Telugu | Cut to headline + 120 words, translate by sentence | NLLB is more accurate on short inputs |
| All text | Tokenise to at most 256 tokens | Fits a T4 GPU with batch size 8 for the large model |
| Images | Resize, centre-crop to 224 by 224, CLIP normalisation | Input format CLIP was trained on |

## Text Classification Models

### DeBERTa-v3-large for English

DeBERTa-v3-large [8] has 24 transformer layers, a hidden size of 1,024 and about 435 million parameters, of which 131 million are in its 128,000-word vocabulary embedding. A classification head (one linear layer over the first token's representation) is added and the whole network is fine-tuned.

In standard self-attention every token is a single vector that mixes what the word is and where it is. DeBERTa keeps two vectors per token, content and relative position, and computes attention from three terms: content to content, content to position and position to content. For a query token i and key token j with content vectors H and relative position embeddings P:

EQ: A(i, j) = Hᵢ Hⱼᵀ + Hᵢ Pᵢⱼᵀ + Pⱼᵢ Hⱼᵀ

The classifier turns the head's output z into class probabilities with the softmax function, and training minimises cross-entropy over the training set:

EQ: p(c | x) = exp(z_c) / Σₖ exp(z_k),   L = −(1/N) Σᵢ log p(yᵢ | xᵢ)

### BERT-base baseline

BERT-base-uncased [6] (12 layers, 110 million parameters) is fine-tuned on the same data with the same pipeline. It is not used by the application; it shows how much the larger, newer model adds.

### TF-IDF baseline

The TF-IDF model treats each article as a bag of words. Each word w in article d gets a weight

EQ: tfidf(w, d) = tf(w, d) · log(N / df(w))

where tf is the word's count in the article, N the number of articles and df the number of articles containing the word. A logistic regression model is trained on these weights. It is used only to measure how much of each dataset can be solved without understanding language.

### IndicBERT v2 for Telugu

IndicBERT v2 [11] (the MLM-only checkpoint) is a BERT-style encoder pre-trained on IndicCorp v2, a large corpus covering 23 Indian languages and English. It is fine-tuned on the translated Telugu set in the same way as the English model.

## Image Detection Model

Following Ojha et al. [17], the CLIP ViT-L/14 image encoder [14] is frozen and only a linear classifier is trained on its output. The encoder cuts a 224 by 224 image into a 16 by 16 grid of 14-pixel patches, runs 24 transformer layers over the 256 patch tokens and a class token, and projects the class token to a 768-dimensional embedding.

For an image with embedding e, the detector computes

EQ: p = softmax(W e + b),   W ∈ ℝ²ˣ⁷⁶⁸

so only 1,538 numbers are learned. The embeddings of all 50,000 images are computed once and cached, and the head is then trained for 30 epochs with AdamW in a few minutes. Because CLIP's features were learned from 400 million image and caption pairs and were never tuned to one generator, the approach is less likely to rely on artefacts specific to the training generator than a fully trained network.

CIFAKE contains only 32 by 32 images from one older generator, so the CLIP head misses many images from current generators such as Midjourney v6. The application therefore averages its AI probability with that of a public detector built on Swin Transformer V2 [58], trained on images from recent generators [59]:

EQ: p_AI = ½ (p_CLIP + p_Swin)

Neither detector was retrained for this. Section 8.3 shows that the average is better than either detector alone on a mixed test set.

## Claim Verification Algorithm

Claim verification uses no trained component from this project. It has two forms. The main one, described in Section 5.7, lets a language model search the web and judge the evidence. The fallback, described in this section, is built from an embedding model, an index and two NLI models, joined by an explicit decision rule; it runs when no language model can be reached and is also the method evaluated on FEVER and Climate-FEVER in Chapter 8.

### Step 1: claim extraction

The article is split into sentences, with spaCy for English and punctuation for Telugu (the full stop and the Devanagari danda, which Telugu text sometimes uses). A sentence is dropped if it is a question, starts with an opinion or call to action ("I think", "share this", "click"), has fewer than five words (three for Telugu) or more than 80. Each remaining sentence s gets a check-worthiness score from the cues ClaimBuster [27] found useful:

EQ: w(s) = 1 + 0.8·min(e, 3) + 1[number] + 0.3·min(c, 4) + 0.7·1[factual verb]

where e is the number of named entities, 1[number] is 1 when the sentence contains a digit, c is the number of capitalised words and 1[factual verb] is 1 when it contains words such as "announced", "confirmed", "won" or "percent". The last two terms apply to English only. The eight highest-scoring sentences, kept in article order, are the claims. A short input such as a single headline becomes one claim.

### Step 2: evidence retrieval

Every evidence passage is embedded once with BGE-M3 [31] and stored in a FAISS [32] inner-product index. Because the vectors are normalised, the inner product equals the cosine similarity

EQ: sim(q, d) = (q · d) / (‖q‖ ‖d‖)

For each claim the three most similar passages are returned, and passages with similarity below 0.45 are dropped. BGE-M3 places English and Telugu text in the same space, so a Telugu claim finds English evidence. If a Google Fact Check Tools API key is configured, published fact-checks for the claim are added to the evidence.

### Step 3: natural language inference

Each evidence passage e is paired with the claim c, the passage as premise and the claim as hypothesis, and the NLI model returns three probabilities: entailment, neutral and contradiction. English claims use DeBERTa-v3-large trained on MNLI, FEVER-NLI, ANLI, LingNLI and WANLI [30]; Telugu claims use the multilingual mDeBERTa-v3-base model. Over the claim's evidence set E the strongest support and the strongest contradiction are kept:

EQ: s⁺ = max over e in E of p_ent(e, c),   s⁻ = max over e in E of p_con(e, c)

### Step 4: decision rule

With threshold τ = 0.5, the claim's label is:

- **Contradiction** if s⁻ ≥ τ and s⁻ ≥ s⁺, with confidence s⁻;
- **Support** if s⁺ ≥ τ, with confidence s⁺;
- **Unknown** otherwise, or when no evidence was found, with confidence 1 − max(s⁺, s⁻).

Contradiction is tested first. A single passage that clearly refutes a claim outweighs passages that only partly agree, which is how a human fact-checker treats a refutation. This rule is what makes the verification contradiction-aware. A system that ranks evidence by similarity alone would find the same passage for "the notes contain a GPS chip" and "the notes do not contain a GPS chip" and could not tell the two apart.

### Step 5: aggregation

With n claims, labels yᵢ and confidences qᵢ, the article score is the average signed confidence

EQ: S = (1/n) Σᵢ vᵢ,  where vᵢ = qᵢ for Support, −qᵢ for Contradiction, 0 for Unknown

and the credibility score is C = (S + 1) / 2, between 0 and 1. The article is labelled Unverified (shown as "Can't verify") when no claim is supported or contradicted, Likely Misleading when S ≤ −0.2, Likely Credible when S ≥ 0.2, and Mixed Evidence otherwise.

```
Algorithm: VerifyArticle(text, language)
  claims  <- ExtractClaims(text, language)            # at most 8
  results <- []
  for c in claims:
      E <- [p for p in FAISS.search(BGE_M3(c), k=3) if p.similarity >= 0.45]
      E <- E + FactCheckAPI(c)                          # optional
      if E is empty: results.add(c, Unknown, 0); continue
      for e in E: e.nli <- NLI(premise=e.text, hypothesis=c)
      s_plus  <- max(e.nli.entailment for e in E)
      s_minus <- max(e.nli.contradiction for e in E)
      if s_minus >= 0.5 and s_minus >= s_plus:
          results.add(c, Contradiction, s_minus)
      else if s_plus >= 0.5:
          results.add(c, Support, s_plus)
      else:
          results.add(c, Unknown, 1 - max(s_plus, s_minus))
  S <- mean(+q for Support, -q for Contradiction, 0 for Unknown)
  return results, credibility = (S + 1) / 2
```

## Claim Verification with a Language Model

A fixed corpus of 40 passages leaves most real claims without evidence, and an NLI model reads each passage on its own, so it is easily misled by a passage about a similar but different event. The main verifier therefore follows the plan-search-judge design of recent fact-checking systems [51, 52] with an open-weight language model, gpt-oss-120b [55], served by Groq's free API [64]. The same code works with a model running locally in Ollama [65].

### Step 1: search queries

The claim is sent to the language model with a short instruction to write up to three search queries: the claim itself in neutral words, the fact that would prove it false, and a query for a published fact-check. For "Rahul Gandhi became the Prime Minister of India in 2024" the queries include "Prime Minister of India 2024" and "Narendra Modi third term".

### Step 2: live evidence

All the queries are run at the same time against five sources, with a 20-second limit for the whole search:

- Wikipedia in the claim's language: the article introductions of the best matching pages;
- a web search through the ddgs library (Yahoo first, then DuckDuckGo and Mojeek if Yahoo fails), which returns titles and snippets;
- Google News headlines with their publishers and dates;
- the Google Fact Check Tools API, when a key is set, which returns published fact-checks with their ratings;
- the local evidence corpus.

### Step 3: passage selection

Each source returns many passages, most of them unrelated. They are embedded with multilingual-E5-small [56], using the prefixes "query:" for the claim and queries and "passage:" for the evidence that the model was trained with. A passage's relevance is its highest cosine similarity to the claim or to any of the queries,

EQ: rel(d) = max over q in {c} ∪ Q of sim(q, d)

so a passage that answers the refuting query ("Narendra Modi third term") is kept even when it shares few words with the claim. Passages with rel(d) < 0.80 are dropped, at most two passages are taken from one page so that one Wikipedia article cannot fill the list, fact-checks are always kept, and the ten best passages are numbered and passed on.

### Step 4: verdict

The language model receives the claim, today's date and the numbered passages, and returns a JSON answer with the label (Supported, Refuted or Not enough evidence), a confidence, a one-sentence reason and the numbers of the passages it relied on. The instruction sets a few rules that came from testing: a claim with a wrong detail (a wrong year, place or number) is refuted; a passage about a different person or thing with a similar name (the Taj Mahal Palace hotel for the Taj Mahal) is ignored; when sources disagree, later and encyclopedic sources are preferred; a published fact-check rating counts as evidence; and a prediction about the future is "Not enough evidence". When no passage is relevant, the model may answer from its own knowledge for well-known facts, but the answer is marked as such and its confidence is capped at 0.8.

The labels map to Support, Contradiction and Unknown, and the article score and credibility are computed exactly as in Step 5 of Section 5.6. Every claim result therefore carries the reason and its cited passages, which the interface shows to the reader.

```
Algorithm: LLMVerify(claim, language)
  Q  <- LLM(plan prompt, claim)                     # up to 3 queries
  P  <- parallel search of Wikipedia, web, News, FactCheck, corpus
        for each q in {claim} ∪ Q, stop after 20 s
  P  <- [d in P if max sim_E5(q, d) >= 0.80]       # q over claim and queries
  P  <- top 10 by relevance, at most 2 per page, fact-checks kept
  (label, conf, reason, cited) <- LLM(verdict prompt, claim, date, P)
  if P is empty: conf <- min(conf, 0.8); mark knowledge_only
  return label, conf, reason, [P[i] for i in cited]
```

The free Groq tier allows 1,000 requests a day and 8,000 tokens a minute; a claim uses two requests. When the limit is reached the backend waits for the time the API asks (up to 20 seconds) and tries again, and if the model still cannot be reached it switches to the NLI verifier of Section 5.6, so a check never fails because of the language model.

## Explainability Methods

### Occlusion for text

Occlusion [61, 62] measures how much each word matters by removing it. With f(x) the classifier's fake probability for the article x and x∖w the article with every occurrence of the word w removed, the weight of w is

EQ: a(w) = f(x) − f(x∖w)

A positive weight means the word pushed the verdict toward fake. Only content words are tested: common English stop words and words beyond the 256 tokens the classifier reads are skipped, because removing them cannot change the prediction. An article of 80 words therefore needs about 50 runs of the classifier, which are made in batches of 16. Section 8.5 compares occlusion with LIME and two gradient methods; it was as faithful as LIME and about ten times faster, so it is the default.

### LIME for text (optional)

LIME [21] explains one prediction f(x) by fitting a simple model g that behaves like f near x. It creates 200 copies of the article with random subsets of words removed, runs the classifier on each copy, and weights each copy by how similar it is to the original (the kernel π). It then solves

EQ: ξ(x) = argmin over g of  L(f, g, π_x) + Ω(g)

where L measures how far g's outputs are from f's on the copies and Ω limits g to 12 words. The weight of each word in g is its explanation: positive weights push toward fake, negative toward real. The article is split on whitespace instead of an English tokenizer, so that Telugu words, whose vowel signs an English tokenizer would split off, stay whole.

### Grad-CAM for images

Grad-CAM [22] uses the activations A of a chosen layer. For the predicted class c it averages the gradient of the class score over the spatial positions to get one weight per channel k, and combines the channels:

EQ: αₖ = (1/Z) Σᵢⱼ ∂yᶜ/∂Aᵏᵢⱼ,   L = ReLU(Σₖ αₖ Aᵏ)

For CLIP the layer is the first layer normalisation of the last transformer block. Its 256 patch tokens are reshaped into a 16 by 16 grid before the formula is applied, and the map is upsampled to the image size and blended over it.

## Fusion of Classifier and Evidence

When both the classifier and the claim verification have a result, the final fake risk is a weighted average in which the evidence counts more:

EQ: r = (1 − λ) p_fake + λ (1 − C),   λ = 0.7

The article is labelled Fake when r ≥ 0.5 and Real otherwise. An earlier version used λ = 0.5, and a confident classifier could then almost cancel a clear contradiction (Section 8.7). The classifier judges only how the text is written, so when claim checking ran but found no evidence for any claim the article is labelled "Can't verify" rather than given the classifier's label; the classifier's score is still shown as a writing-style signal. If claim checking is switched off the classifier alone decides, and if the classifier is not available the claims alone decide. The report always states which sources were used.

## Training Setup

All models were trained in one notebook on a single NVIDIA T4 GPU (16 GB) in a free Kaggle session. The text models use the Hugging Face Trainer with the AdamW optimiser [41], a linear learning-rate schedule with 10% warm-up, weight decay 0.01, 16-bit mixed precision, and early stopping when validation F1 has not improved for two evaluations. The best checkpoint by validation F1 is kept. The times in Table 5.4 are the notebook's planning estimates for a T4; actual times vary with the session.

TABLE: Training hyperparameters
| Model | Learning rate | Effective batch | Epochs | Max length | Estimated time on T4 |
|---|---|---|---|---|---|
| DeBERTa-v3-large | 1e-5 | 32 (8 × 4 accumulation) | 2 | 256 tokens | 1.5 to 2.5 hours |
| BERT-base | 2e-5 | 32 (16 × 2) | 3 | 256 tokens | 30 to 60 minutes |
| IndicBERT v2 | 3e-5 | 32 (16 × 2) | 4 | 256 tokens | about 20 minutes, plus 30 to 40 minutes of translation |
| CLIP linear head | 1e-3 | 64 | 30 | 224 px | 20 to 40 minutes, mostly feature extraction |

## Evaluation Metrics

With TP, TN, FP and FN the counts of true positives, true negatives, false positives and false negatives (fake, or AI-generated, is the positive class):

EQ: Accuracy = (TP + TN) / (TP + TN + FP + FN)

EQ: Precision = TP / (TP + FP),   Recall = TP / (TP + FN)

EQ: F1 = 2 · Precision · Recall / (Precision + Recall)

ROC-AUC is the area under the curve of true positive rate against false positive rate as the decision threshold moves from 0 to 1; it measures how well the model ranks fake items above real ones regardless of threshold. Claim verification has three classes, so it is measured with accuracy and macro-F1, the unweighted mean of the F1 scores of Support, Contradiction and Unknown, which gives the rare Contradiction class the same weight as the others.
