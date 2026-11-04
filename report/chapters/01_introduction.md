## Overview

Soon after the 2000 rupee note was introduced in November 2016, messages spread across India claiming that it carried a GPS nano chip that let the government track black money. The note has no chip and the Reserve Bank of India never made such a claim, yet the story kept resurfacing in forwarded messages. Stories like this travel well because they are short, carry a confident tone and are passed on by people the reader trusts. A study of about 126,000 stories shared on Twitter between 2006 and 2017 found that false news reached 1,500 people roughly six times faster than true news [1].

Images now spread misinformation as easily as text. Diffusion models such as Stable Diffusion [20] produce photographs of events that never happened, and an image placed next to a false headline makes the headline more believable. A reader who wants to check a message today has to search for the claim, look for a fact-check article, and judge the picture by eye, usually in English, and usually one claim at a time.

TruthLens AI is a web application that does this checking in one place. A user pastes a news article in English or Telugu, or drops in an image, and the system returns:

- a verdict (fake or real for text, AI-generated or real for images) with the model's confidence,
- the words or image regions that led to the verdict,
- for text, each factual claim in the article marked Supported, Contradicted or No evidence found, together with the evidence and its source.

The last point is what separates this project from a plain classifier. A classifier that says "Fake, 91%" gives the reader nothing to verify. TruthLens also says which sentence is false and shows the passage that contradicts it, so the reader can judge the evidence for themselves.

## Problem Statement

Existing fake news detectors have four gaps that make them hard to use for readers in Andhra Pradesh and Telangana:

1. Most are trained and tested only on English news, while a large share of local misinformation circulates in Telugu.
2. They check text or images, but rarely both in the same tool.
3. They give one label for a whole article and do not say which statement is false.
4. They do not explain their output, so the reader must either trust the model or ignore it.

The problem addressed in this project is to build a system that classifies English and Telugu news and detects AI-generated images, explains every verdict in a form a non-expert can read, and verifies the individual claims in an article against evidence, separating claims the evidence supports from claims it contradicts.

## Motivation

Three observations shaped the design.

First, a verdict without a reason gives the reader nothing to check. If the system says only that a message is "fake", the reader has to decide whether to trust the model. If it shows the sentence that is false and the passage that contradicts it, the reader can judge the evidence directly and can see when the model is wrong. That is the reason the claim-level report is the centre of the user interface.

Second, transformer classifiers now reach very high accuracy on public fake news datasets, but much of that accuracy comes from style rather than truth. During this project a word-count model with no understanding of language scored 96.2% on the WELFake dataset after cleaning and 99.8% on a public Telugu dataset (Chapter 5). A system that only classifies style will fail on a carefully written false story. Checking claims against evidence does not depend on style.

Third, the tools needed to build such a system are now free. Every model in this project is open and downloaded from Hugging Face, all training data downloads without an account, and training runs on the free T4 GPUs in Google Colab and Kaggle. The full system runs on an ordinary laptop.

## Objectives

The objectives of the project are:

1. To fine-tune DeBERTa-v3-large for English fake news classification and IndicBERT v2 for Telugu, and compare them with a BERT-base baseline and a TF-IDF model.
2. To build a Telugu training set whose two classes cannot be separated by writing style alone.
3. To detect AI-generated images with a linear classifier on CLIP ViT-L/14 features.
4. To explain text verdicts with LIME and image verdicts with Grad-CAM.
5. To design and evaluate a contradiction-aware, claim-level verification stage that labels each claim Support, Contradiction or Unknown and combines the labels into an overall credibility score.
6. To support batch checking of CSV files and image folders, with downloadable results.
7. To package the system so that it installs and starts on Windows, Linux or macOS with one command.

## Scope of the Project

The system checks news text in English and Telugu. Other Indian languages are detected as English and are not supported by the classifiers, although the multilingual evidence retriever and NLI model would work for them with little change.

Claim verification can only confirm or contradict a claim when a passage about that claim exists in the evidence corpus or in the Google Fact Check Tools results. The project ships with a small hand-checked corpus of 40 passages on Indian current affairs, science and common health hoaxes, plus a script to add Wikipedia articles. Building a large, current evidence base is outside the scope of a final-year project and is listed as future work.

The image detector is trained on CIFAKE, which contains 32 by 32 pixel images from one generator. It shows that the approach works, but it is not a production detector for full-resolution images from current generators.

The system is a decision aid. It does not remove, block or report content, and its verdicts are meant to be read together with the evidence it shows.

## Methodology in Brief

The project was carried out in five stages:

1. **Data preparation.** WELFake was cleaned of publisher names and its labels were corrected; a Telugu set was built by machine translation; CIFAKE was sampled and split.
2. **Model training.** All models were trained in one notebook on a free T4 GPU, with results written to disk for this report.
3. **Claim verification.** The claim extractor, evidence retriever, NLI verifier and aggregation rule were built from pre-trained models and evaluated on FEVER, Climate-FEVER, IndicXNLI and a curated claim set.
4. **Application.** A FastAPI backend and a React frontend were built around the models, with a launcher that sets everything up.
5. **Testing and evaluation.** Functional test cases were run against the live application, and every model was evaluated on held-out data.

## Organisation of the Report

Chapter 2 reviews earlier work on fake news detection, Indian-language resources, AI-generated image detection, explainability and automated fact-checking. Chapter 3 analyses the existing systems, the proposed system and its requirements. Chapter 4 presents the system design with UML diagrams, data flow diagrams and the database design. Chapter 5 describes the datasets, models and algorithms in detail. Chapter 6 covers the implementation of the backend, frontend and training pipeline. Chapter 7 lists the test cases and their results. Chapter 8 reports and discusses the results of every model and of the claim verification stage. Chapter 9 concludes the report and lists future work, and Chapter 10 is a guide to installing and using the system.
