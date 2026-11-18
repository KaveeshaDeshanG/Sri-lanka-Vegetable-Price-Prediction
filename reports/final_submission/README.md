# Final submission: report and evidence

| Path | What it is |
|---|---|
| `Final_Report.docx` | Editable report (contents, lists of figures and tables filled in) |
| `Final_Report.pdf` | PDF export of the same document (86 pages) |
| `Final_Report.md` | The same content as one Markdown file |
| `REVIEW_AND_CHECKLIST.md` | Improvement checklist, independent review findings and corrections, items left for the student |
| `assets/screenshots/` | 21 screenshots cropped from captures of the running app (18 Nov 2026); `crop_screenshots.py` shows each crop |
| `assets/diagrams/source/` | Editable diagram sources: Mermaid `.mmd` files and `structural_diagrams.py` (architecture, context, use case, DFD 1, languages) |
| `assets/diagrams/rendered/` | Rendered diagrams (`.png`, and `.svg` for the hand-laid-out ones) |
| `assets/charts/` | Result and data charts, drawn by `make_charts.py` from the saved metrics and datasets |
| `test_evidence/` | JUnit XML and console output of the full pytest run (17 Nov), browser test results, script and full-page captures (18 Nov) |
| `source/` | Report text (Markdown), references, automated test catalogue and the build scripts |

**Rebuilding the report.** The build needs python-docx and Pillow, which are not project requirements; use a separate environment.

```
python -m pip install python-docx pillow
cd reports/final_submission/source
python build_report.py ../test_evidence/browser_tests.json ../test_evidence/pytest_junit_2026-11-17.xml
powershell -ExecutionPolicy Bypass -File update_fields.ps1
```

`update_fields.ps1` needs Microsoft Word or WPS Office (the Word.Application interface). It fills in the contents and the lists of figures and tables, then exports the PDF.
