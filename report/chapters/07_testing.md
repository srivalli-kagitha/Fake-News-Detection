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
| UT15 | combine | fake probability 0.8, credibility 0.4 | ('Fake', 0.7) | ('Fake', 0.7) | Pass |
| UT16 | combine | fake probability 0.3, claims Unverified | ('Real', 0.3, ['classifier']) | ('Real', 0.3, ['classifier']) | Pass |

All 16 unit tests passed. The script imports the real backend functions, so a change to a rule that breaks its documented behaviour shows up as a failed row.

## Integration Test Cases

The integration tests were run with `evaluation/run_test_cases.py` against the running application with all trained models installed, on the development laptop (Intel Core Ultra 7 255H, 16 logical processors and 11 GB of memory available to Ubuntu under WSL 2 on Windows, no GPU). The "Actual result" column is copied from the script's output, not written by hand. The models had already been loaded by an earlier run, so the times show the cost of the check itself.

TABLE: Integration test cases (actual results from evaluation/run_test_cases.py)
| ID | Module | Test | Input | Expected | Actual result | Status |
|---|---|---|---|---|---|---|
| TC01 | System | Health check reports model status | GET /api/health | HTTP 200 with status of every model | HTTP 200; models reported: text_en, text_te, image, nli_en, nli_multi, embeddings | Pass |
| TC02 | Validation | Reject text shorter than 10 characters | "hello" | HTTP 422, request rejected | HTTP 422 | Pass |
| TC03 | Validation | Reject text longer than 20,000 characters | 20,500 characters | HTTP 422, request rejected | HTTP 422 | Pass |
| TC04 | Text check | English rumour is flagged fake | 5G towers spread coronavirus (sample) | language en; overall verdict Fake | language=en, classifier=Fake (100.0%), overall=Fake (risk 0.87) | Pass |
| TC05 | Text check | English news report is judged real | Chandrayaan-3 landing report (sample) | language en; overall verdict Real | language=en, classifier=Real (100.0%), overall=Real (risk 0.00) | Pass |
| TC06 | Text check | Telugu text is detected and classified | Telugu sample (Telangana, 5G rumour) | language te; Telugu model gives a verdict | language=te, classifier=Fake (95.0%), claims={'Support': 2, 'Contradiction': 1, 'Unknown': 0} | Pass |
| TC07 | Text check | Claim checking can be switched off | verify_claims=false | verdict only, no claim results | language=en, verification=skipped | Pass |
| TC08 | Explanation | LIME returns word weights | 5G rumour sample | a weight for each word, top words listed | 63 tokens weighted; top words: deleted!, Share, across, The, the | Pass |
| TC09 | Claim check | Four claims labelled against evidence | Chandrayaan, GPS chip, UNESCO anthem, Hyderabad | Support, Contradiction, Contradiction, Support | labels=['Support', 'Contradiction', 'Contradiction', 'Support']; summary=Mixed Evidence | Pass |
| TC10 | Claim check | Claim with no evidence is Unknown | Polavaram claim (not in corpus) | Unknown; article Unverified | labels=['Unknown']; summary=Unverified | Pass |
| TC11 | Claim check | Telugu claim matched with English evidence | హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని. | Support, using the English Hyderabad passage | label=Support (98%); evidence: Hyderabad is the capital of the Indian state of Telangana. | Pass |
| TC12 | Image check | Real photograph is judged real | CIFAKE real image | Real, with Grad-CAM heatmap | label=AI-Generated / Manipulated (97.7%); heatmap=yes | Fail |
| TC13 | Image check | Generated image is judged AI-generated | CIFAKE generated image | AI-Generated / Manipulated, with Grad-CAM heatmap | label=AI-Generated / Manipulated (94.5%); heatmap=yes | Pass |
| TC14 | Image check | Non-image file is rejected | text file renamed .png | HTTP 400 with a message | HTTP 400: notes.png is not a valid image file. | Pass |
| TC15 | Image check | Image over 15 MB is rejected | 16 MB file | HTTP 413 with a message | HTTP 413: big.jpg is larger than 15 MB. | Pass |
| TC16 | Batch | CSV of three texts is checked | CSV with text column, 3 rows | 3 results with labels and a job id | 3 rows; labels=['Fake', 'Real', 'Fake']; counts={'Fake': 2, 'Real': 1} | Pass |
| TC17 | Batch | CSV without a text column is rejected | CSV with column 'body' | HTTP 400 naming accepted columns | HTTP 400: CSV needs a column named one of: text, news, content, article, title, headline. | Pass |
| TC18 | Batch | ZIP of four images is checked | ZIP: 2 real, 2 generated | 4 results with labels | 4 images; labels=['Real', 'Real', 'AI-Generated / Manipulated', 'AI-Generated / Manipulated'] | Pass |
| TC19 | Batch | Batch results download as CSV | job id from TC16 | CSV file with header and 3 rows | HTTP 200; 4 lines; header: row,text,language,prediction,confidence,fake_probability | Pass |
| TC20 | Batch | Invalid job id is refused | job id '../etc' | HTTP 400 or 404, no file served | HTTP 404 | Pass |
| TC21 | History | Checks are saved in history | GET /api/history | records of the checks above | 40 records; kinds=['batch_image', 'batch_text', 'claims', 'image', 'text'] | Pass |
| TC22 | History | History filter by type | kind=image | only image checks returned | 16 records, all kind=image: True | Pass |
| TC23 | History | Deleting a missing record | DELETE /api/history/99999999 | HTTP 404 Record not found | HTTP 404: Record not found. | Pass |

22 of 23 integration tests passed. The one failure, TC12, is a model error rather than a software fault: the detector called a real CIFAKE photograph AI-generated with 97.7% confidence. The test was left as a failure instead of being rerun with a different image. To see how often this happens, all 12 sample images were checked: 9 of 12 were classified correctly (5 of 6 real photographs and 4 of 6 generated images; the two missed generated images were called Real with low confidence, 52% and 60%). Twelve images are too few to estimate accuracy, so the figure of 93.7% on 10,000 held-out images in Chapter 8 remains the measure of the detector. The sample images were taken from the public CIFAKE archive and may include images that were also in the training sample.

TC06 is worth a note as well. The Telugu sample contains two true statements and one rumour, and the claim check labelled them exactly so (2 supported, 1 contradicted), while the Telugu classifier called the whole text fake with 95% confidence. This is the behaviour the claim report is designed for: the reader sees which part of the message is false.

## System Testing

The system tests drive a real Chromium browser through the application with Playwright, at a window size of 1440 by 900 pixels (375 by 812 for the phone test). In the first run three tests failed: ST03, ST04 and ST14. All three were faults in the test script, not in the application. ST03 hovered the first numbered list on the page, which turned out to be the list of influential words rather than the claim list. ST04 and ST14 reloaded the page between steps, which clears TruthLens's in-memory results by design. The three tests were corrected to hover the claim list and to move between pages with the navigation rail, as a user does, and were run again (`evaluation/run_ui_retest.py`). The table shows the results of the corrected tests.

TABLE: System test cases (actual results from evaluation/run_ui_tests.py)
| ID | Scenario | Expected result | Actual result | Status |
|---|---|---|---|---|
| ST01 | Start page sends text to the text check | Text page opens and the check starts | opened /text, verdict shown after 31 s | Pass |
| ST02 | Verdict shown before highlights | Verdict first, highlights added later with a progress note | verdict on screen while highlights pending: True; 15 words marked 74 s later | Pass |
| ST03 | Hovering a claim highlights it in the article | Matching sentence lit up | hovering claim 2 lit up 27 words of that sentence in the article | Pass |
| ST04 | Results kept when moving between pages | Report still shown on return | after moving to the image page and back with the rail, report and input still shown: True | Pass |
| ST05 | Dark theme is remembered | Dark theme kept after reload | data-theme after reload = dark | Pass |
| ST06 | Save as PDF uses a print layout | Navigation hidden when printing | navigation rail visible in print layout: False | Pass |
| ST07 | Telugu article checked end to end | Telugu detected; verdict, highlights and claims shown | Telugu report shown with 14 marked words | Pass |
| ST08 | Image report with heatmap slider | Slider wipes between image and heatmap | slider moved, heatmap style: clip-path: inset(0px 0px 0px 35%); | Pass |
| ST09 | Start page sends an image to the image check | Image page opens and checks the image | image page opened and checked the dropped image | Pass |
| ST10 | Verify claims page | Each claim listed with status and evidence | 14 claims listed | Pass |
| ST11 | Batch CSV check and download | Result CSV with one row per input | downloaded truthlens_text_a89604b3ee.csv with 3 rows | Pass |
| ST12 | History filter | Only image checks shown | 19 rows shown after filtering by Images | Pass |
| ST13 | Phone layout | No horizontal scrolling at 375 px | page width 375 px at a 375 px screen | Pass |
| ST14 | Keyboard-only use | Text box and Check button reachable with Tab | 24 controls reached with Tab, including the text box and the Check button: True | Pass |

## Performance

TABLE: Measured response times on the development laptop (CPU only)
| Operation | Input | Time |
|---|---|---|
| Health check | GET /api/health | under 0.1 s |
| Text check, verdict and claims | English rumour, 3 sentences (TC04) | 13.7 s |
| Text check, verdict and claims | English news report, 2 sentences (TC05) | 17.9 s |
| Text check, verdict and claims | Telugu text, 3 sentences (TC06) | 10.4 s |
| Text check, verdict only | English report, claims off (TC07) | 4.7 s |
| LIME explanation | English rumour, 63 words (TC08) | 74.8 s |
| Claim verification | 4 claims (TC09) | 22.5 s |
| Image check with Grad-CAM | one image (TC12, TC13) | 10 to 11 s |
| Batch text | 3 rows, claims off (TC16) | 6.4 s |
| Batch images | ZIP of 4 images (TC18) | 10.5 s |
| Browser: verdict on screen | 4-sentence article from the Start page (ST01) | 31 s |
| Browser: highlights added | same article, after the verdict (ST02) | 74 s more |

The times come from the test runs above. The English DeBERTa-v3-large model dominates every text check: about 5 seconds for the verdict alone on a short text, and most of the rest goes to the NLI model checking each claim against three passages. LIME is the slowest step because it runs the classifier 200 times; on this CPU it took about 75 seconds for a 63-word text and grows with the length of the article. Splitting the response into a verdict request and an explanation request (Section 6.4) means the user waits for the verdict only, and the highlights appear in the open report when they are ready. With a CUDA GPU, which the system uses automatically when one is present, all of these steps are expected to be many times faster; this was not measured because the development laptop has no GPU.

## Test Summary

TABLE: Test summary
| Level | Test cases | Passed | Failed |
|---|---|---|---|
| Unit tests | 16 | 16 | 0 |
| Integration tests (API) | 23 | 22 | 1 |
| System tests (browser) | 14 | 14 | 0 |
| Total | 53 | 52 | 1 |

Every functional requirement in Table 3.1 is covered by at least one passing test. The single failure is the image misclassification in TC12, which reflects the detector's measured error rate rather than a defect in the software. Model accuracy is reported separately in Chapter 8.
