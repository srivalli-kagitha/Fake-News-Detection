This chapter reviews the work that TruthLens builds on. It is organised by the five problems the system solves: classifying news text, handling Indian languages, detecting generated images, explaining model output, and verifying claims against evidence. The chapter ends with a comparison table and the gaps that this project addresses.

## Fake News Detection from Text

### Feature-based methods

Early detectors treated fake news as a text classification problem with hand-built features. Shu et al. [2] grouped these features into news content features (lexical, syntactic and visual) and social context features (who shared the story and how it spread), and argued that content alone is often not enough because fake stories are written to look like real ones. Rashkin et al. [42] compared the language of satire, hoaxes, propaganda and trusted news and found measurable differences in the use of first and second person pronouns, superlatives and subjective words. These findings explain why simple word-count models do well on many datasets, a point that returns in Chapter 5 of this report.

### Benchmark datasets

Progress in the field has followed the release of public datasets. LIAR [3] contains about 12,800 short political statements from PolitiFact, each labelled on a six-point truthfulness scale. FakeNewsNet [44] collects full articles from PolitiFact and GossipCop together with social context such as user engagement. WELFake [4] merges four earlier corpora (Kaggle, McIntire, Reuters and BuzzFeed Political) into 72,134 labelled articles; its authors proposed combining word embeddings with linguistic features in a voting classifier. WELFake is large, balanced and free to download, which is why it is used for training in this project, but Chapter 5 shows that it has label and source shortcuts that must be removed before training.

### Deep learning and transformers

Recurrent and convolutional networks replaced hand-built features around 2017, and transformers [34] replaced them in turn. BERT [6] is pre-trained by predicting masked words and is then fine-tuned on a labelled task. FakeBERT [5] combined BERT embeddings with parallel one-dimensional convolutional layers and reported higher accuracy than earlier neural models on a Kaggle fake news set.

DeBERTa [7] changes BERT in two ways. Each word is represented by two vectors, one for content and one for position, and attention is computed from both separately, so the model can tell that "deep learning" and "learning deep" differ in meaning. DeBERTa-v3 [8] replaces masked-word prediction with ELECTRA-style replaced-token detection, where the model learns to spot words that a smaller generator has swapped in. DeBERTa-v3-large has about 435 million parameters and is one of the strongest openly available encoders for English classification, which is why the project abstract specifies it as the English model.

## Indian-Language Resources

Labelled fake news data for Indian languages is scarce. BanFakeNews [9] provides about 50,000 Bangla news articles labelled authentic or fake, with baselines ranging from linguistic features to neural models. For Telugu, a dataset was published on Zenodo in 2024 [13]. Section 5.2 of this report examines it and shows that its two classes can be separated by writing style alone, which is why it was not used for training.

Multilingual encoders make Telugu classification possible without training a model from scratch. XLM-RoBERTa [37] was pre-trained on 100 languages. IndicBERT [10], from the AI4Bharat group, was pre-trained on Indian-language text and evaluated on the IndicGLUE benchmark. IndicBERT v2 [11] extends this to 23 Indian languages and English using the larger IndicCorp v2 corpus, and is the Telugu classifier in this project. For translation between English and Telugu, Meta's NLLB-200 [12] covers 200 languages with one model; its distilled 1.3 billion parameter version is used to build the Telugu training set.

## AI-Generated Image Detection

Generative adversarial networks [35] were the first models to produce convincing fake faces, and diffusion models [36] such as latent diffusion [20] now produce photographs of almost any scene from a text prompt.

Wang et al. [16] trained a ResNet-50 on images from one GAN (ProGAN) with blur and JPEG augmentation and found that it detected images from several other GAN architectures. Later work showed that this transfer breaks for diffusion models: Corvi et al. [18] found that detectors trained on GAN images performed poorly on diffusion images, and that diffusion models leave their own traces in the frequency spectrum.

Ojha et al. [17] took a different approach. Instead of training a network to find generator artefacts, they used the frozen image encoder of CLIP [14], a vision transformer [15] trained to match 400 million images with their captions, and trained only a nearest-neighbour or linear classifier on its features. Because CLIP's features were never tuned to one generator, this detector generalised to unseen generators, including diffusion models, better than fully trained networks. TruthLens follows this approach with CLIP ViT-L/14.

CIFAKE [19] pairs 60,000 real CIFAR-10 images with 60,000 images generated by Stable Diffusion v1.4 at the same 32 by 32 resolution. Its authors trained a convolutional network and used Grad-CAM to show that the network often relied on small details in the image background rather than the main object.

## Explainable AI

LIME [21] explains a single prediction of any classifier. It creates many copies of the input with random parts removed, records how the model's output changes, and fits a small weighted linear model to those changes. The weights of the linear model are the explanation. For text, the parts are words, so LIME can say that removing "BREAKING" lowered the fake probability by a certain amount. LIME needs only the model's output, which makes it model-agnostic, but it needs hundreds of model runs per explanation.

Grad-CAM [22] explains a convolutional or transformer image model with a single backward pass. It takes the activations of a chosen layer, weights each channel by the gradient of the target class with respect to that channel, and sums them into a heatmap that shows which regions raised the class score. TruthLens uses LIME for text and Grad-CAM for images.

## Automated Fact-Checking

Guo et al. [26] describe automated fact-checking as three steps: detecting claims worth checking, retrieving evidence, and predicting a verdict with a justification.

**Claim detection.** ClaimBuster [27] scores each sentence for check-worthiness, trained on sentences from US presidential debates labelled by human coders. Sentences with numbers, named entities and factual verbs score high and opinions score low. TruthLens uses a simpler rule-based score built on the same cues (Section 5.6).

**Verdict prediction.** FEVER [23] made verdict prediction a standard benchmark: about 185,000 claims written from Wikipedia sentences are labelled Supported, Refuted or NotEnoughInfo, with the evidence sentences that justify the label. Climate-FEVER [24] applies the same labels to 1,535 real claims about climate change collected from the web, which makes it harder and closer to real use. DeClarE [45] retrieved web articles for each claim and used attention over them to predict credibility.

**Natural language inference.** NLI asks whether a premise entails, contradicts or is neutral to a hypothesis. MNLI [28] and Adversarial NLI [29] are the main training sets. A model trained on NLI can verify a claim by treating the evidence as the premise and the claim as the hypothesis, without training on fact-checking data. Laurer et al. [30] showed that one NLI model can serve as a general classifier in this way, and released the DeBERTa-v3 NLI models used in TruthLens. For Telugu, IndicXNLI [25] provides NLI test pairs translated into Indian languages.

**Explanations for fact-checks.** Atanasova et al. [47] trained a model to produce a short written justification together with the verdict, and Kotonya and Toni [46] did the same for public health claims. TruthLens shows the evidence passage itself rather than a generated justification, so the reader sees exactly what the decision was based on.

**Evidence retrieval.** Dense retrieval embeds the claim and the passages in the same vector space and returns the nearest passages. BGE-M3 [31] produces such embeddings for more than 100 languages, including Telugu, and FAISS [32] searches millions of vectors quickly. Together they let TruthLens search the same evidence for English and Telugu claims.

## Comparison of Existing Work

TABLE: Summary of related work
| Work | Task | Method | Data | Limitation for this project |
|---|---|---|---|---|
| Shu et al. 2017 [2] | Fake news detection | Survey of content and social features | Various | Survey; no system |
| Wang 2017, LIAR [3] | Truthfulness rating | CNN with metadata | 12.8K PolitiFact statements | Short statements, English only |
| Rashkin et al. 2017 [42] | Reliability of news | Linguistic analysis, LSTM | News and PolitiFact | Style cues, not facts |
| Verma et al. 2021, WELFake [4] | Fake news detection | Word embeddings + linguistic features | 72,134 articles | Source shortcuts in data |
| Kaliyar et al. 2021, FakeBERT [5] | Fake news detection | BERT + 1D CNN | Kaggle fake news | No explanation, English only |
| Hossain et al. 2020, BanFakeNews [9] | Fake news detection | Traditional and neural models | About 50K Bangla articles | Bangla only |
| Doddapaneni et al. 2023, IndicBERT v2 [11] | Indian-language NLU | Multilingual pre-training | IndicCorp v2, 24 languages | General model, no fake news task |
| Wang et al. 2020 [16] | Generated image detection | ResNet-50 with augmentation | ProGAN and other GANs | Weak on diffusion images |
| Ojha et al. 2023 [17] | Generated image detection | Frozen CLIP + linear probe | GAN and diffusion images | Images only |
| Bird and Lotfi 2024, CIFAKE [19] | Generated image detection | CNN + Grad-CAM | 120K images, 32 by 32 | Low resolution, one generator |
| Ribeiro et al. 2016, LIME [21] | Explanation | Local linear surrogate | Model-agnostic | Slow (many model runs) |
| Selvaraju et al. 2017, Grad-CAM [22] | Explanation | Gradient-weighted activations | Image models | Images only |
| Hassan et al. 2017, ClaimBuster [27] | Claim detection | Supervised check-worthiness score | US debates | No verdict |
| Thorne et al. 2018, FEVER [23] | Fact verification | Retrieval + NLI baseline | 185K Wikipedia claims | Written claims, English only |
| Popat et al. 2018, DeClarE [45] | Credibility | Attention over web evidence | Snopes, PolitiFact | No per-claim contradiction output |
| Laurer et al. 2024 [30] | Classification via NLI | DeBERTa-v3 NLI | MNLI, ANLI, FEVER-NLI, others | General method, not a system |

## Research Gaps

The review shows five gaps that TruthLens addresses:

1. **Language.** Strong fake news classifiers exist for English, but Telugu has no public training set whose classes differ in content rather than style.
2. **Modality.** Text and image detectors are built and evaluated separately, while the misinformation readers see often combines a headline with an image.
3. **Granularity.** Article-level classifiers do not say which statement is false. Fact verification systems do, but they are evaluated on benchmark claims and are rarely joined to a classifier in one tool.
4. **Contradiction.** Retrieval-based checkers that rank evidence by similarity cannot tell a passage that supports a claim from one that refutes it, because both are about the same subject. An explicit NLI step is needed.
5. **Explanation for readers.** Explanation methods are usually shown to researchers as heatmaps or weight lists. A reader needs the explanation inside the article: the influential words marked, the false sentence underlined, and the evidence printed next to it.

The next chapter analyses the existing approaches against these gaps and sets out the requirements of the proposed system.
