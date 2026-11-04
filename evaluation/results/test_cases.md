| ID | Module | Test | Expected | Actual | Status | Time (s) |
|---|---|---|---|---|---|---|
| TC01 | System | Health check reports model status | HTTP 200 with status of every model | HTTP 200; models reported: text_en, text_te, image, nli_en, nli_multi, embeddings | Pass | 0.0 |
| TC02 | Validation | Reject text shorter than 10 characters | HTTP 422, request rejected | HTTP 422 | Pass | 0.0 |
| TC03 | Validation | Reject text longer than 20,000 characters | HTTP 422, request rejected | HTTP 422 | Pass | 0.0 |
| TC04 | Text check | English rumour is flagged fake | language en; overall verdict Fake | language=en, classifier=Fake (100.0%), overall=Fake (risk 0.87) | Pass | 13.7 |
| TC05 | Text check | English news report is judged real | language en; overall verdict Real | language=en, classifier=Real (100.0%), overall=Real (risk 0.00) | Pass | 17.9 |
| TC06 | Text check | Telugu text is detected and classified | language te; Telugu model gives a verdict | language=te, classifier=Fake (95.0%), claims={'Support': 2, 'Contradiction': 1, 'Unknown': 0} | Pass | 10.4 |
| TC07 | Text check | Claim checking can be switched off | verdict only, no claim results | language=en, verification=skipped | Pass | 4.7 |
| TC08 | Explanation | LIME returns word weights | a weight for each word, top words listed | 63 tokens weighted; top words: deleted!, Share, across, The, the | Pass | 74.8 |
| TC09 | Claim check | Four claims labelled against evidence | Support, Contradiction, Contradiction, Support | labels=['Support', 'Contradiction', 'Contradiction', 'Support']; summary=Mixed Evidence | Pass | 22.5 |
| TC10 | Claim check | Claim with no evidence is Unknown | Unknown; article Unverified | labels=['Unknown']; summary=Unverified | Pass | 1.3 |
| TC11 | Claim check | Telugu claim matched with English evidence | Support, using the English Hyderabad passage | label=Support (98%); evidence: Hyderabad is the capital of the Indian state of Telangana. | Pass | 3.6 |
| TC12 | Image check | Real photograph is judged real | Real, with Grad-CAM heatmap | label=AI-Generated / Manipulated (97.7%); heatmap=yes | Fail | 11.3 |
| TC13 | Image check | Generated image is judged AI-generated | AI-Generated / Manipulated, with Grad-CAM heatmap | label=AI-Generated / Manipulated (94.5%); heatmap=yes | Pass | 10.4 |
| TC14 | Image check | Non-image file is rejected | HTTP 400 with a message | HTTP 400: notes.png is not a valid image file. | Pass | 0.0 |
| TC15 | Image check | Image over 15 MB is rejected | HTTP 413 with a message | HTTP 413: big.jpg is larger than 15 MB. | Pass | 0.9 |
| TC16 | Batch | CSV of three texts is checked | 3 results with labels and a job id | 3 rows; labels=['Fake', 'Real', 'Fake']; counts={'Fake': 2, 'Real': 1} | Pass | 6.4 |
| TC17 | Batch | CSV without a text column is rejected | HTTP 400 naming accepted columns | HTTP 400: CSV needs a column named one of: text, news, content, article, title, headline. | Pass | 0.0 |
| TC18 | Batch | ZIP of four images is checked | 4 results with labels | 4 images; labels=['Real', 'Real', 'AI-Generated / Manipulated', 'AI-Generated / Manipulated'] | Pass | 10.5 |
| TC19 | Batch | Batch results download as CSV | CSV file with header and 3 rows | HTTP 200; 4 lines; header: ﻿row,text,language,prediction,confidence,fake_probability | Pass | 0.0 |
| TC20 | Batch | Invalid job id is refused | HTTP 400 or 404, no file served | HTTP 404 | Pass | 0.0 |
| TC21 | History | Checks are saved in history | records of the checks above | 40 records; kinds=['batch_image', 'batch_text', 'claims', 'image', 'text'] | Pass | 0.0 |
| TC22 | History | History filter by type | only image checks returned | 16 records, all kind=image: True | Pass | 0.0 |
| TC23 | History | Deleting a missing record | HTTP 404 Record not found | HTTP 404: Record not found. | Pass | 0.0 |
