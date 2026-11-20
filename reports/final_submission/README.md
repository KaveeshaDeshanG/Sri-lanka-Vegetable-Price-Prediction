# Final submission: report and evidence

| Path | What it is |
|---|---|
| `Final_Report.docx` | The report to submit (editable; contents and lists of figures and tables filled in) |
| `Final_Report.pdf` | PDF export of the same document |
| `update_fields.ps1` | Refreshes the contents and lists after an edit and exports the PDF |
| `assets/screenshots/` | 22 screenshots cropped from captures of the running app (18 Nov 2026); `crop_screenshots.py` shows each crop |
| `assets/diagrams/source/` | Editable diagram sources: Mermaid `.mmd` files (rendered by `render_mermaid.py`) and `structural_diagrams.py` (architecture, context, use case, DFD 1, languages; PNGs made with `svg_to_png.py`) |
| `assets/diagrams/rendered/` | Rendered diagrams (`.png`, and `.svg`) |
| `assets/charts/` | Result and data charts, drawn by `make_charts.py` from the saved metrics and datasets |
| `test_evidence/` | JUnit XML and console output of the full pytest run (17 Nov 2026); browser test results, script, upload inputs and full-page captures (18 Nov 2026) |

To refresh the contents and the PDF after editing the DOCX:

```
powershell -ExecutionPolicy Bypass -File reports/final_submission/update_fields.ps1
```

`update_fields.ps1` needs Microsoft Word or WPS Office (the Word.Application interface). The Python scripts in `assets/` need Pillow; the browser and diagram scripts also need Microsoft Edge and the websockets package.
