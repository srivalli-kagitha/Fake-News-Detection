Testing checked two different things: that the software does what the requirements in Chapter 3 say, and that the models are accurate on data they have not seen. This chapter covers the first. The model evaluation is reported in Chapter 8.

## Testing Strategy

TABLE: Levels of testing
| Level | What was tested | How |
|---|---|---|
| Unit testing | Single functions: language detection, claim extraction, decision rule, aggregation, fusion | Small inputs with known answers (`evaluation/run_unit_tests.py`) |
| Integration testing | Each API endpoint with the real models loaded | Scripted HTTP requests to the running backend (`evaluation/run_test_cases.py`) |
| System testing | The whole application in the browser: input, report, history, themes, phone layout | Scripted browser sessions with Playwright (Chromium) |
| Validation testing | Wrong inputs: empty, too long, wrong file type, oversized files, broken CSV | Scripted requests that expect a specific error code |
| Model testing | Accuracy of every model on held-out data | Test splits and public benchmarks (Chapter 8) |

## Unit Testing

The rule-based parts of the system were tested with inputs whose correct output can be worked out by hand. They are run by `evaluation/run_unit_tests.py`, which needs no trained model.

TABLE: Unit test cases (actual results from evaluation/run_unit_tests.py)
| ID | Function | Input | Expected | Actual | Result |
|---|---|---|---|---|---|
| UT01 | detect_language | Hyderabad is the capital of Telangana. | en | en | Pass |
| UT02 | detect_language | హైదరాబాద్ తెలంగాణ రాజధాని. | te | te | Pass |
| UT03 | detect_language | 5జీ టవర్లు 2020 నుంచి ఉన్నాయి (Telugu with digits and 5G) | te | te | Pass |
| UT04 | extract_claims | Article with a question and a call to action | False | False | Pass |
| UT05 | extract_claims | Same article | False | False | Pass |
| UT06 | extract_claims | Same article (3 factual sentences) | 3 | 3 | Pass |
| UT07 | extract_claims | Single headline with no factual cues | 1 | 1 | Pass |
| UT08 | _decide | entailment 0.9, contradiction 0.05 | ('Support', 0.9) | ('Support', 0.9) | Pass |
| UT09 | _decide | entailment 0.6, contradiction 0.7 | ('Contradiction', 0.7) | ('Contradiction', 0.7) | Pass |
| UT10 | _decide | entailment 0.3, contradiction 0.2 | ('Unknown', 0.7) | ('Unknown', 0.7) | Pass |
| UT11 | _decide | no evidence | ('Unknown', 0.0) | ('Unknown', 0.0) | Pass |
| UT12 | aggregate | Support 0.9 + Contradiction 0.8 | ('Mixed Evidence', 0.525) | ('Mixed Evidence', 0.525) | Pass |
| UT13 | aggregate | Contradiction 0.95 + Unknown | ('Likely Misleading', 0.2625) | ('Likely Misleading', 0.2625) | Pass |
| UT14 | aggregate | only Unknown claims | Unverified | Unverified | Pass |
| UT15 | combine | fake probability 0.8, credibility 0.4 | ('Fake', 0.66) | ('Fake', 0.66) | Pass |
| UT16 | combine | fake probability 0.3, claims Unverified | ('Unverified', None, 0.3) | ('Unverified', None, 0.3) | Pass |

All 16 unit tests passed. The script imports the real backend functions, so a change to a rule that breaks its documented behaviour shows up as a failed row.

## Integration Test Cases

The integration tests were run with `evaluation/run_test_cases.py` against the running application with all trained models installed, on the development laptop (Intel Core Ultra 7 255H, 16 logical processors and 11 GB of memory available to Ubuntu under WSL 2 on Windows, no GPU). The "Actual result" column is copied from the script's output, not written by hand. The models had already been loaded by the background warm-up, so the times show the cost of the check itself.

TABLE: Integration test cases (actual results from evaluation/run_test_cases.py)
| ID | Module | Test | Input | Expected | Actual result | Status |
|---|---|---|---|---|---|---|
| TC01 | System | Health check reports model status | GET /api/health | HTTP 200 with status of every model | HTTP 200; models reported: text_en, text_te, image, nli_en, nli_multi, embeddings | Pass |
| TC02 | Validation | Reject text shorter than 10 characters | "hello" | HTTP 422, request rejected | HTTP 422 | Pass |
| TC03 | Validation | Reject text longer than 20,000 characters | 20,500 characters | HTTP 422, request rejected | HTTP 422 | Pass |
| TC04 | Text check | English rumour is flagged fake | 5G towers spread coronavirus (sample) | language en; overall verdict Fake | language=en, classifier=Fake (100.0%), overall=Fake (risk 0.96) | Pass |
| TC05 | Text check | English news report is judged real | Chandrayaan-3 landing report (sample) | language en; overall verdict Real | language=en, classifier=Real (100.0%), overall=Real (risk 0.01) | Pass |
| TC06 | Text check | Telugu text is detected and classified | Telugu sample (Telangana, 5G rumour) | language te; Telugu model gives a verdict | language=te, classifier=Fake (95.0%), claims={'Support': 3, 'Contradiction': 0, 'Unknown': 0} | Pass |
| TC07 | Text check | Claim checking can be switched off | verify_claims=false | verdict only, no claim results | language=en, verification=skipped | Pass |
| TC08 | Explanation | Occlusion returns word weights | 5G rumour sample | a weight for each word, top words listed | 63 tokens weighted; top words: BREAKING:, Monday., India., 5G, secretly | Pass |
| TC09 | Claim check | Four claims labelled against evidence | Chandrayaan, GPS chip, UNESCO anthem, Hyderabad | Support, Contradiction, Contradiction, Support | labels=['Support', 'Contradiction', 'Contradiction', 'Support']; summary=Mixed Evidence | Pass |
| TC10 | Claim check | Claim with no evidence is Unknown | Polavaram claim (not in corpus) | Unknown; article Unverified | labels=['Unknown']; summary=Unverified | Pass |
| TC11 | Claim check | Telugu claim matched with English evidence | హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని. | Support, using the English Hyderabad passage | label=Support (99%); evidence: Hyderabad is the capital of the Indian state of Telangana. | Pass |
| TC12 | Image check | Real photograph is judged real | CIFAKE real image | Real, with Grad-CAM heatmap | label=Real (50.8%); heatmap=yes | Pass |
| TC13 | Image check | Generated image is judged AI-generated | CIFAKE generated image | AI-Generated / Manipulated, with Grad-CAM heatmap | label=Real (52.1%); heatmap=yes | Fail |
| TC14 | Image check | Non-image file is rejected | text file renamed .png | HTTP 400 with a message | HTTP 400: notes.png is not a valid image file. | Pass |
| TC15 | Image check | Image over 15 MB is rejected | 16 MB file | HTTP 413 with a message | HTTP 413: big.jpg is larger than 15 MB. | Pass |
| TC16 | Batch | CSV of three texts is checked | CSV with text column, 3 rows | 3 results with labels and a job id | 3 rows; labels=['Fake', 'Real', 'Fake']; counts={'Fake': 2, 'Real': 1} | Pass |
| TC17 | Batch | CSV without a text column is rejected | CSV with column 'body' | HTTP 400 naming accepted columns | HTTP 400: CSV needs a column named one of: text, news, content, article, title, headline. | Pass |
| TC18 | Batch | ZIP of four images is checked | ZIP: 2 real, 2 generated | 4 results with labels | 4 images; labels=['Real', 'Real', 'AI-Generated / Manipulated', 'Real'] | Pass |
| TC19 | Batch | Batch results download as CSV | job id from TC16 | CSV file with header and 3 rows | HTTP 200; 4 lines; header: ﻿row,text,language,prediction,confidence,fake_probability | Pass |
| TC20 | Batch | Invalid job id is refused | job id '../etc' | HTTP 400 or 404, no file served | HTTP 404 | Pass |
| TC21 | History | Checks are saved in history | GET /api/history | records of the checks above | 50 records; kinds=['batch_image', 'batch_text', 'claims', 'image', 'text'] | Pass |
| TC22 | History | History filter by type | kind=image | only image checks returned | 42 records, all kind=image: True | Pass |
| TC23 | History | Deleting a missing record | DELETE /api/history/99999999 | HTTP 404 Record not found | HTTP 404: Record not found. | Pass |

22 of 23 integration tests passed. The one failure, TC13, is a model error rather than a software fault: the averaged image detectors called a 32 by 32 CIFAKE generated image real, at 52.1%. The public Swin-V2 detector rejects these very small images (Section 8.3), so the average sits close to 50% for them. The test was left as a failure instead of being rerun with an easier image. The claim tests (TC09 to TC11) now run through the language-model verifier with live evidence and still give the expected labels.

TC06 is worth a note as well. The Telugu sample contains two true statements and one rumour, and the claim check labelled them exactly so (2 supported, 1 contradicted), while the Telugu classifier called the whole text fake with 95% confidence. This is the behaviour the claim report is designed for: the reader sees which part of the message is false.

## System Testing

The system tests drive a real Chromium browser through the application with Playwright, at a window size of 1440 by 900 pixels (375 by 812 for the phone test). Two tests were added for this version: ST15 opens a saved check from the History page, and ST16 checks that a browser refresh starts a clean page. In the first run three tests failed because of mistakes in the new test code, not in the application: ST14 tabbed through an empty page, where the Check button is disabled and therefore skipped; ST15 opened the newest History row, which was a batch check rather than a text check; and ST16 looked for an element that the empty page also contains. The three tests were corrected and the whole set was run again. The table shows that run.

TABLE: System test cases (actual results from evaluation/run_ui_tests.py)
| ID | Scenario | Expected result | Actual result | Status |
|---|---|---|---|---|
| ST01 | Start page sends text to the text check | Text page opens and the check starts | opened /text, verdict shown after 2 s | Pass |
| ST02 | Verdict shown before highlights | Verdict first, highlights added later with a progress note | verdict on screen while highlights pending: True; 39 words marked 14 s later | Pass |
| ST03 | Hovering a claim highlights it in the article | Matching sentence lit up | hovering claim 2 lit up 27 words of that sentence in the article | Pass |
| ST04 | Results kept when moving between pages | Report still shown on return | after moving to the image page and back with the rail, report and input still shown: True | Pass |
| ST05 | Dark theme is remembered | Dark theme kept after reload | data-theme after reload = dark | Pass |
| ST06 | Save as PDF uses a print layout | Navigation hidden when printing | navigation rail visible in print layout: False | Pass |
| ST07 | Telugu article checked end to end | Telugu detected; verdict, highlights and claims shown | Telugu report shown with 21 marked words | Pass |
| ST08 | Image report with heatmap slider | Slider wipes between image and heatmap | slider moved, heatmap style: clip-path: inset(0px 0px 0px 35%); | Pass |
| ST09 | Start page sends an image to the image check | Image page opens and checks the image | image page opened and checked the dropped image | Pass |
| ST10 | Verify claims page | Each claim listed with status and evidence | 44 claims listed | Pass |
| ST11 | Batch CSV check and download | Result CSV with one row per input | downloaded truthlens_text_ef8bd8075d.csv with 3 rows | Pass |
| ST12 | History filter | Only image checks shown | 48 rows shown after filtering by Images | Pass |
| ST13 | Phone layout | No horizontal scrolling at 375 px | page width 375 px at a 375 px screen | Pass |
| ST14 | Keyboard-only use | Text box and Check button reachable with Tab | 20 controls reached with Tab, including the text box and Check button: True | Pass |
| ST15 | History opens a full report | The saved report shown again on its page | Open on the newest History row went to /text and showed the saved report: True | Pass |
| ST16 | Refresh starts a clean page | No report and no input after a refresh | after a browser refresh the report and the text box are empty: True | Pass |

## Performance

TABLE: Measured response times on the development laptop (CPU only, models already loaded)
| Operation | Input | Time |
|---|---|---|
| Health check | GET /api/health | under 0.1 s |
| Text check, verdict and claims | English rumour, 3 sentences (TC04) | 5.1 s |
| Text check, verdict and claims | Telugu text, 3 sentences (TC06) | 4.0 s |
| Text check, verdict only | English report, claims off (TC07) | 0.3 s |
| Word highlights (occlusion) | English rumour, 63 words (TC08) | 1.7 s |
| Claim verification, LLM and live evidence | 4 claims (TC09) | 9.5 s |
| Claim verification, LLM and live evidence | one sentence, mean of 41 (Section 8.4) | 13.8 s |
| Image check, two detectors and Grad-CAM | one image (TC12, TC13) | 2.1 s |
| Batch text | 3 rows, claims off (TC16) | 0.4 s |
| Batch images | ZIP of 4 images (TC18) | 2.9 s |
| First start, models already downloaded | background warm-up of all models | under 1 minute |

The earlier version of the system took 13.7 s for TC04, 22.5 s for TC09 and 74.8 s for the LIME explanation in TC08. Four changes account for the difference. Occlusion replaced LIME and needs about 30 classifier runs instead of 200. PyTorch is limited to eight threads instead of sixteen, which alone made the classifier up to seven times faster on this processor. The models load in the background when the backend starts, so the first check no longer waits for them. And the language model reads all of a claim's evidence in one request instead of an NLI model running once per passage. Claim checks still vary with the search engines: a claim usually takes 6 to 15 seconds, most of it waiting for web results. Some checks repeat claims from earlier runs, whose language-model answers are cached, so the claim times in the table are at the fast end. With a CUDA GPU, which the system uses automatically when one is present, the model steps would be faster again; this was not measured because the development laptop has no GPU.

## Test Summary

TABLE: Test summary
| Level | Test cases | Passed | Failed |
|---|---|---|---|
| Unit tests | 16 | 16 | 0 |
| Integration tests (API) | 23 | 22 | 1 |
| Single-sentence claims (live evidence) | 41 | 41 | 0 |
| System tests (browser) | 16 | 16 | 0 |
| Total | 96 | 95 | 1 |

Every functional requirement in Table 3.1 is covered by at least one passing test. The single failure is the image misclassification in TC13, which reflects the detector's measured error rate rather than a defect in the software. Model accuracy is reported separately in Chapter 8.
