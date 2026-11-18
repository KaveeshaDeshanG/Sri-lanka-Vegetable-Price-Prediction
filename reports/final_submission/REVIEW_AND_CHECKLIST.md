# Final report: improvement checklist and independent review

Prepared 18 November 2026 for `Final_Report.docx` / `Final_Report.pdf`.

## 1. Improvement checklist (from the investigation of the earlier reports)

Sources reviewed:
- the interim report, `Kaveesha_Deshan_Interim_Report_E290292.pdf` (CS6P05ES, 38 pages);
- `Final_Project_Report/Final_Project_Report.pdf` (11 Nov, 78 pages) and `Final_Project_Report_Updated.docx` (13 Nov);
- `reports/PROJECT_REPORT.md`, `README.md` and `reports/DATA_SOURCES.md`.

The format followed is the one in the interim report:
- a CS6P05ES cover with name, ID and supervisors;
- the university's plagiarism header sheet as the declaration;
- Harvard (author, year) referencing.

No separate campus template, marking rubric or formatting guide was found on this PC.

| Item found in the earlier reports | Action in the final report |
|---|---|
| Dataset counts and dates from 9-11 Nov | Updated from `data_manifest.json` (17 Nov) |
| Said the model "has not been retrained"; no mention of the 13 Nov retraining check | Retraining decision reported with its numbers (Section 6.7.5) |
| No multilingual feature | Design, implementation, tests and screenshots added |
| Automatic updates described as "scheduling left to the user"; daily at 18:30 | Every 3 hours via Task Scheduler; lock recovery, retries, manifest, freshness |
| Planner without the current vs harvest-time comparison | Comparison documented, with its rules (±5%, likely range, when no comparison is shown) |
| Predict page described with four data sources, including sample and live HARTI options | Two sources (official data, upload) as the app now offers |
| 1,060 tests | 1,320 tests, run on 17 Nov |
| Screenshots from 11-13 Nov, old layout | 21 new screenshots captured on 18 Nov |
| Diagrams from the interim (earlier model) mixed with new ones | 11 new diagrams of the current system, with editable sources |
| Long, repetitive chapters (about 18,500 words); results split over Chapters 6 and 7 | About 11,800 words of running text in 7 chapters, plus tables |
| Placeholders for university, faculty and degree in one version | Only verified details are used (name, ID, module); the rest are left as placeholders |

## 2. Independent review findings and corrections

**Content completeness**
- All five pages, the multilingual feature, automatic updates, every data source and the planner are covered. No issues remain.

**Technical accuracy**
- **Hardware:** a hardware line written from memory named the wrong processor. It was replaced with the values read from this PC (Ryzen 5 5500, 16 GB).
- **Code size:** line counts written from memory were wrong. They were recounted (app 7,218; src 6,187; scripts 3,947; tests 11,353).
- **Freshness labels:** the statuses were named wrongly. They now use the app's own labels (`FRESHNESS_LABELS` in app/catalogue.py).
- **Behind-schedule limits:** the 3/4/21-day limits were attributed to the Home freshness panel. They are now attributed to the Predict page's source status, which is where they apply.
- **Planner diagram:** it showed every harvest price being compared with today's price. It now matches `compare_prices`: only a forecast or a seasonal reference is compared, and the latest-actual-price fallback is shown.
- **DFD text:** the text said presentation reads every data store, which did not match the diagram. The wording now matches.
- **Storage size:** "about 2 GB" for the PDF archive could not be verified. It was replaced with README's "about 700 MB".
- **Cross-references:** "Section 6.10" pointed to a section that doesn't exist. Both mentions were corrected to Section 6.7.
- **Capture dates:** the screenshots were described as taken on 17 Nov. They were taken on 18 Nov 00:15-00:24, so the captions and text now say that.

**Screenshot quality**
- **Tall pages:** they became unreadable at page width. The prediction result and its table were split, and the planner and Model Performance crops were tightened.
- **Remaining:** the Predict and Model Performance tables are wider than 1280 px, so their last columns are cut in the screenshots. This is the app's real layout at that width; the full values are in Tables 6.6 and 6.7.

**Testing quality**
- **BT-04:** the check passed even when a page did not load, because "NOT loaded" contains "loaded". The check was fixed and the full run repeated: 30 of 30 passed.
- **BT-06:** the recorded "forecast shown" value was the latest-price card. It was replaced with the forecast (Rs 463.04) from the same run's page text, with a note in `browser_tests.json`.
- **First runs:** some failures in the first browser runs were script problems (wrong labels, waiting for a section that only appears when an optional report exists). These were fixed and every test was re-run.
- **Automated catalogue:** all 52 cases are generated from the JUnit results. The build stops if a named test is missing.

**Academic quality**
- **Citations:** every in-text citation has a reference and every reference is cited (checked by script).
- **DOIs and arXiv IDs:** all were checked against doi.org, Crossref and arXiv. The details changed or added during that check:
  - AGRICAF: arXiv ID added (arXiv:2410.20363);
  - Hathurusingha et al.: DOI added (10.1142/S0219649219500060);
  - FAO GIEWS: a citation added for a claim that had none;
  - Navaratnalingam et al.: an unverified place of publication removed.

**Document quality**
- **Declaration:** two lines overflowed onto a separate page. The extracts now fit on one page.
- **Page numbers:** the appendices restarted at page 1, because new sections inherited the start number. They now continue.
- **Table widths:** WPS ignored the cell widths, so columns came out equal and the evidence column was unreadable. An explicit grid with a fixed layout fixed this.
- **Narrow columns:** ID columns wrapped (for example "FR" / "1"). A minimum width from the longest word in each column fixed this.
- **Small tables:** they split across pages. Tables of up to 14 rows are now kept together.
- **Sinhala and Tamil text:** checked in the PDF and renders correctly, using Nirmala UI.

## 3. Items left for the student

- **Personal details:** supervisor names, module leader, deadline, submission date, signature, the acknowledgement text, and the AI-use statement (Appendix E), which must match the university's policy.
- **Translations:** the Sinhala and Tamil wording has not yet been reviewed by native speakers.
- **App text:** the Home page still mentions a synthetic sample option that the Predict page no longer offers. This is an app wording issue; the app was not changed for the report.
- **Final layout check:** the PDF was produced by WPS Office through the Word interface (Microsoft Word is not installed). Open the DOCX in Microsoft Word before submission, update all fields (Ctrl+A, F9) and check the layout.
- **Campus requirements:** check any formatting rules the campus may have that were not available here, such as word limits, fonts or a required chapter list.
