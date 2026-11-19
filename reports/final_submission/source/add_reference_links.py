"""Edits Final_Report.docx in place: makes every reference URL and DOI a clickable hyperlink, adds verified links to
references that had none, and inserts the 'Project Source Code Repository' section before References.
Nothing else in the document is changed. Every added URL was verified on 19 November 2026 (see the report summary).

Usage: python add_reference_links.py <path to Final_Report.docx>
"""
import copy
import re
import sys

import docx
from docx.enum.text import WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

PATH = sys.argv[1]
REPO_NAME = "KaveeshaDeshanG/Sri-lanka-Vegetable-Price-Prediction"
REPO_URL = "https://github.com/KaveeshaDeshanG/Sri-lanka-Vegetable-Price-Prediction"
LINK_COLOUR = RGBColor(0x1F, 0x4E, 0x9A)

# Text edits to individual references (exact, verified details only).
ADDITIONS = {
    "Tran, N.-Q.": ("arXiv preprint arXiv:2310.18646.", "arXiv preprint arXiv:2310.18646. Available at: https://arxiv.org/abs/2310.18646."),
    "Weerasekara, R.": ("arXiv preprint arXiv:2606.29248.", "arXiv preprint arXiv:2606.29248. Available at: https://arxiv.org/abs/2606.29248."),
    "Zelingher, R.": ("arXiv preprint arXiv:2410.20363.", "arXiv preprint arXiv:2410.20363. Available at: https://arxiv.org/abs/2410.20363."),
    "McKinney, W.": ("pp. 56-61.", "pp. 56-61. doi:10.25080/Majora-92bf1922-00a."),
    "Pedregosa, F.": ("pp. 2825-2830.", "pp. 2825-2830. Available at: https://jmlr.org/papers/v12/pedregosa11a.html (Accessed: 19 November 2026)."),
    "Hastie, T.": ("New York: Springer.", "New York: Springer. Available at: https://hastie.su.domains/ElemStatLearn/ (Accessed: 19 November 2026)."),
    "Zippenfenig, P.": ("Weather API*. Zenodo.", "Weather API* [Computer software]. Zenodo."),
}
LINK_RE = re.compile(r"(https?://[^\s)]+?)(?=[.,]?(?:\s|$|\)))|doi:(10\.[^\s]+?)(?=\.?(?:\s|$))")


def add_hyperlink_run(paragraph, before_run, text, url, template_rpr):
    """Insert a w:hyperlink holding `text` right before `before_run` (or at the end when None)."""
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink"); link.set(qn("r:id"), r_id); link.set(qn("w:history"), "1")
    run = OxmlElement("w:r")
    rpr = copy.deepcopy(template_rpr) if template_rpr is not None else OxmlElement("w:rPr")
    for tag in ("w:i", "w:color", "w:u"):
        for el in rpr.findall(qn(tag)): rpr.remove(el)
    color = OxmlElement("w:color"); color.set(qn("w:val"), str(LINK_COLOUR)); rpr.append(color)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
    run.append(rpr)
    t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = text; run.append(t)
    link.append(run)
    if before_run is None:
        paragraph._p.append(link)
    else:
        before_run._r.addprevious(link)
    return link


def plain_run_like(paragraph, before_run, text, template_rpr):
    r = OxmlElement("w:r")
    if template_rpr is not None: r.append(copy.deepcopy(template_rpr))
    t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = text; r.append(t)
    before_run._r.addprevious(r)


def link_paragraph(paragraph):
    """Replace URLs and doi: identifiers inside the paragraph's runs by hyperlinks, keeping run formatting."""
    made = []
    for run in list(paragraph.runs):
        text = run.text
        if not LINK_RE.search(text):
            continue
        rpr = run._r.find(qn("w:rPr"))
        pos = 0
        for m in LINK_RE.finditer(text):
            if m.start() > pos:
                plain_run_like(paragraph, run, text[pos:m.start()], rpr)
            if m.group(1):
                url = m.group(1); shown = url
            else:
                url = "https://doi.org/" + m.group(2); shown = url
                plain_run_like(paragraph, run, "Available at: ", rpr)
            add_hyperlink_run(paragraph, run, shown, url, rpr)
            made.append(url)
            pos = m.end()
        tail = text[pos:]
        if tail:
            plain_run_like(paragraph, run, tail, rpr)
        run._r.getparent().remove(run._r)
    return made


def apply_addition(paragraph, old, new):
    """Apply an exact text edit inside one run (reference runs keep italics on titles only)."""
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new, 1); return True
    # the edit spans runs (e.g. italic title followed by plain text): edit the plain run holding the tail
    if old.startswith("Weather API*"):
        for run in paragraph.runs:
            if run.text.startswith(". Zenodo."):
                run.text = run.text.replace(". Zenodo.", " [Computer software]. Zenodo.", 1); return True
    return False


doc = docx.Document(PATH)
paras = doc.paragraphs
ref_idx = next(i for i, p in enumerate(paras) if p.text.strip() == "References" and p.style.name.startswith("Heading"))
end_idx = next((i for i in range(ref_idx + 1, len(paras)) if paras[i].style.name.startswith("Heading")), len(paras))
refs = [p for p in paras[ref_idx + 1:end_idx] if p.text.strip()]

edited, links = [], []
for p in refs:
    for key, (old, new) in ADDITIONS.items():
        if p.text.startswith(key):
            if not apply_addition(p, old, new):
                raise SystemExit(f"could not apply the edit for {key}: {p.text[:80]}")
            edited.append(key)
    links += link_paragraph(p)

# ---- Project Source Code Repository, on its own page before References
heading = paras[ref_idx]
anchor = heading
prev = heading._p.getprevious()
if prev is not None and prev.tag == qn("w:p") and "w:br" in prev.xml and 'w:type="page"' in prev.xml:
    anchor_p = prev  # the page-break paragraph that starts the References page
else:
    anchor_p = heading._p


def new_para(style, text_parts, before):
    p = doc.add_paragraph(style=style)
    for part in text_parts:
        if isinstance(part, tuple) and part[0] == "link":
            add_hyperlink_run(p, None, part[1], part[1], None)
        elif isinstance(part, tuple) and part[0] == "bold":
            r = p.add_run(part[1]); r.bold = True
        else:
            p.add_run(part)
    before.addprevious(p._p)
    return p


pb = doc.add_paragraph(); pb.add_run().add_break(WD_BREAK.PAGE); anchor_p.addprevious(pb._p)
new_para("Heading 1", ["Project Source Code Repository"], anchor_p)
new_para(None, ["The complete source code of this project is kept in a Git repository on GitHub, together with the data and documentation needed to run, test and maintain it."], anchor_p)
new_para(None, [("bold", "Repository: "), REPO_NAME], anchor_p)
new_para(None, [("bold", "URL: "), ("link", REPO_URL)], anchor_p)
new_para(None, [("bold", "Contents: "),
                "the Streamlit application (app/); the data collection, preprocessing, feature engineering, model and prediction code (src/); "
                "the update, training, evaluation and validation scripts (scripts/); the automated tests and their real document fixtures (tests/); "
                "the configuration (config/); the stored price, weather and fuel datasets and the Department of Agriculture reference data (data/); "
                "the trained models and evaluation results (models/, reports/metrics/); and the documentation, including the README, the data "
                "source and data quality reports, and this report with its screenshots, diagrams and test evidence (reports/). The original PDF "
                "bulletins are not stored in the repository because of their size; every stored price keeps the address of its source bulletin."],
         anchor_p)
new_para(None, [("bold", "Running it: "), "the setup steps are in README.md and in Appendix C."], anchor_p)
new_para(None, ["[Note: the repository is currently private. Make it public, or give your supervisor and examiner access, before submission, then delete this note.]"], anchor_p)

doc.save(PATH)
print("references:", len(refs), "| edited:", edited, "| hyperlinks:", len(links))
for u in links: print("  ", u)
