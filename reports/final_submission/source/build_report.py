"""Builds reports/final_submission/Final_Report.docx from the Markdown sources in this folder.

Run with a Python that has python-docx:  python build_report.py <capture.json> <junit.xml>
The values that come from test runs (browser results, forecasts shown in screenshots, automated test results) are read
from those two files, never typed in. Fields (contents, lists of figures and tables) are filled in afterwards by
update_fields.ps1, which also exports the PDF.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Inches, Pt, RGBColor
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # reports/final_submission
OUT = ROOT / "Final_Report.docx"
CAPTURE = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
JUNIT = ET.parse(sys.argv[2]).getroot()
sys.path.insert(0, str(HERE))
from test_catalogue import AUTOMATED  # noqa: E402

TITLE = "AI-Powered Sri Lankan Vegetable Cultivation Planning and Market Price Prediction System"
SHORT_TITLE = "AI-Powered Vegetable Cultivation Planning and Price Prediction System"
AUTHOR = "Kaveesha Deshan"
BODY_FONT, CODE_FONT, LOCAL_FONT = "Times New Roman", "Consolas", "Nirmala UI"
TEXT_WIDTH_IN = 6.27
LANDSCAPE_WIDTH_IN = 9.69
GREEN = RGBColor(0x1F, 0x5F, 0x3A)

# --------------------------------------------------------------------------- values from the test runs
tests = CAPTURE["tests"]
bt_total, bt_pass = len(tests), sum(t["status"] == "Pass" for t in tests)


def find(pattern, text, default="(not found)"):
    m = re.search(pattern, text or "", flags=re.S)
    return m.group(1) if m else default


predict_text = CAPTURE.get("predict_text", "")
plan_text = CAPTURE.get("planner_within_text", "")
TOKENS = {
    "BT_TOTAL": str(bt_total),
    "BT_DATE": CAPTURE["run_started"][:10],
    "BEANS_7D": find(r"Forecast[^\n]*\n?\s*(Rs\. ?[\d,]+\.\d\d) ?/ ?kg", predict_text),
    "BEANS_LATEST": find(r"starting from (Rs [\d,]+\.\d\d)", predict_text),
    "PLAN_FORECAST": find(r"Expected price at harvest\s*\n?\s*(Rs\. ?[\d,]+\.\d\d)", plan_text),
}


def junit_results():
    res = defaultdict(list)
    for tc in JUNIT.iter("testcase"):
        cls = tc.get("classname").replace(".", "/") + ".py"
        name = re.sub(r"\[.*\]$", "", tc.get("name"))
        ok = tc.find("failure") is None and tc.find("error") is None and tc.find("skipped") is None
        res[f"{cls}::{name}"].append((ok, float(tc.get("time", 0))))
    return res


JRES = junit_results()
SUITE = next(JUNIT.iter("testsuite"))


def automated_rows():
    rows = []
    for tid, feature, node, pre, steps, expected in AUTOMATED:
        runs = JRES.get(node)
        if not runs:
            raise SystemExit(f"{tid}: {node} is not in the JUnit results")
        ok = all(r[0] for r in runs)
        actual = f"{'Passed' if ok else 'Failed'}: {sum(r[0] for r in runs)} of {len(runs)} case(s), {sum(r[1] for r in runs):.2f} s"
        rows.append([tid, feature, pre, steps, expected, actual, "Pass" if ok else "Fail", node.replace("tests/", "")])
    return rows


AUTO_ROWS = automated_rows()


def md_table(header, rows, caption, label):
    esc = lambda s: str(s).replace("|", "/").replace("\n", " ")
    lines = [f"Table: {caption} {{#{label}}}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def browser_rows():
    order = sorted(tests, key=lambda t: int(t["id"].split("-")[1]))
    return [[t["id"], t["area"], t["preconditions"].replace("App running at http://localhost:8501 (start_app.bat); data updated to 17 Nov 2026; saved models present", "App running; data to 17 Nov 2026"),
             t["steps"], t["expected"], t["actual"][:330], t["status"], t["evidence"]] for t in order]


def browser_summary():
    areas = defaultdict(lambda: [0, 0])
    for t in tests:
        a = t["area"].split(" /")[0]
        areas[a][0] += 1; areas[a][1] += t["status"] == "Pass"
    rows = [f"| {a} | {n} | {p} | {n - p} |" for a, (n, p) in areas.items()]
    rows.append(f"| **Total** | **{bt_total}** | **{bt_pass}** | **{bt_total - bt_pass}** |")
    return "\n".join(rows)


auto_pass = sum(r[6] == "Pass" for r in AUTO_ROWS)
TOKENS["BROWSER_SUMMARY_ROWS"] = browser_summary()
HEAD8 = ["ID", "Feature", "Preconditions", "Steps", "Expected result", "Actual result", "Status", "Evidence"]
TOKENS["AUTOMATED_TABLE"] = md_table(HEAD8, AUTO_ROWS, "Automated test cases (pytest; actual results from the run of 17 November 2026)", "tab:autocases")
TOKENS["BROWSER_TABLE"] = md_table(HEAD8, browser_rows(), "Browser test cases on the running application (headless Edge)", "tab:browsercases")
TOKENS["TOTALS_TABLE"] = md_table(
    ["Test type", "Executed", "Passed", "Failed", "Blocked", "Command or tool"],
    [["Automated (full pytest suite)", SUITE.get("tests"), int(SUITE.get("tests")) - int(SUITE.get("failures")) - int(SUITE.get("errors")) - int(SUITE.get("skipped")),
      int(SUITE.get("failures")) + int(SUITE.get("errors")), SUITE.get("skipped"), "python -m pytest"],
     ["of which documented above", len(AUTO_ROWS), auto_pass, len(AUTO_ROWS) - auto_pass, 0, "Appendix A.2"],
     ["Browser (scripted, running app)", bt_total, bt_pass, bt_total - bt_pass, 0, "headless Microsoft Edge"],
     ["Manual tests with users", 0, 0, 0, 0, "not carried out"]],
    "Test totals", "tab:totals")

for key, value in TOKENS.items():
    if "(not found)" in value:
        print(f"WARNING: token {key} not found in the capture results")


def fill(text):
    for key, value in TOKENS.items():
        text = text.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    if left:
        raise SystemExit(f"unfilled tokens: {left}")
    return text


# --------------------------------------------------------------------------- Markdown parsing
def parse(md):
    blocks, lines, i = [], md.splitlines(), 0
    md_comment = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("<!--"):
            md_comment = True
        if md_comment:
            md_comment = "-->" not in line; i += 1; continue
        if not line.strip():
            i += 1; continue
        if line.startswith("```"):
            code = []
            i += 1
            while not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            blocks.append(("code", code)); i += 1; continue
        m = re.match(r"(#{1,3}) (.+?)(?: \{#([\w:-]+)\})?$", line)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2), m.group(3))); i += 1; continue
        m = re.match(r"Table: (.+?) \{#([\w:-]+)\}$", line)
        if m:
            caption, label = m.group(1), m.group(2)
            i += 1
            while not lines[i].strip(): i += 1
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", caption, label, rows)); continue
        m = re.match(r"!\[(.+?)\]\((.+?)\)\{#([\w:-]+) width=([\d.]+)\}$", line)
        if m:
            blocks.append(("fig", m.group(1), m.group(2), m.group(3), float(m.group(4)))); i += 1; continue
        if re.match(r"- ", line):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:]); i += 1
            blocks.append(("ul", items)); continue
        if re.match(r"\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                items.append(re.sub(r"^\d+\. ", "", lines[i])); i += 1
            blocks.append(("ol", items)); continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"(#|- |\d+\. |Table: |!\[|```|\|)", lines[i]):
            para.append(lines[i].strip()); i += 1
        blocks.append(("p", " ".join(para)))
    return blocks


# --------------------------------------------------------------------------- numbering
def number(blocks):
    labels, chapter, sec, sub, fig, tab = {}, "0", 0, 0, 0, 0
    out = []
    n_ch = 0
    for b in blocks:
        if b[0] == "h":
            level, text = b[1], b[2]
            if level == 1:
                if text == "References":
                    out.append(("h", 1, text, None)); continue
                if text.startswith("Appendix"):
                    chapter = text.split()[1].rstrip(":")
                    title = text
                else:
                    n_ch += 1; chapter = str(n_ch); title = f"{n_ch}. {text}"
                sec = sub = fig = tab = 0
                out.append(("h", 1, title, b[3]))
            elif level == 2:
                sec += 1; sub = 0
                title = text if chapter.isalpha() else f"{chapter}.{sec} {text}"
                out.append(("h", 2, title, b[3]))
            else:
                sub += 1
                title = text if chapter.isalpha() else f"{chapter}.{sec}.{sub} {text}"
                out.append(("h", 3, title, b[3]))
        elif b[0] == "fig":
            fig += 1; num = f"{chapter}.{fig}"; labels[b[3]] = f"Figure {num}"
            out.append(("fig", f"Figure {num}: {b[1]}", b[2], b[3], b[4]))
        elif b[0] == "table":
            tab += 1; num = f"{chapter}.{tab}"; labels[b[2]] = f"Table {num}"
            out.append(("table", f"Table {num}: {b[1]}", b[2], b[3]))
        else:
            out.append(b)
    return out, labels


# --------------------------------------------------------------------------- docx helpers
doc = Document()


def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_fonts(run, name, east=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), east if (east and attr in ("w:cs",)) else name)


LOCAL_RE = re.compile(r"([඀-෿஀-௿‍]+(?:[\s‍]*[඀-෿஀-௿]+)*)")


def add_text(paragraph, text, bold=False, italic=False, code=False, size=None, color=None):
    for k, seg in enumerate(LOCAL_RE.split(text)):
        if not seg: continue
        run = paragraph.add_run(seg)
        run.bold, run.italic = bold or None, italic or None
        if size: run.font.size = Pt(size)
        if color: run.font.color.rgb = color
        if k % 2 == 1:
            set_fonts(run, LOCAL_FONT, LOCAL_FONT)
        elif code:
            set_fonts(run, CODE_FONT); run.font.size = Pt((size or 11) - 1.5)


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`)")


def add_inline(paragraph, text, size=None, bold=False):
    for part in INLINE.split(text):
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            add_text(paragraph, part[2:-2], bold=True, size=size)
        elif part.startswith("`") and part.endswith("`"):
            add_text(paragraph, part[1:-1], code=True, size=size, bold=bold)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            add_text(paragraph, part[1:-1], italic=True, size=size, bold=bold)
        else:
            add_text(paragraph, part, size=size, bold=bold)


def resolve_refs(text, labels):
    def rep(m):
        if m.group(1) not in labels:
            raise SystemExit(f"unknown reference {m.group(1)}")
        return labels[m.group(1)]
    return re.sub(r"\[@([\w:-]+)\]", rep, text)


def field(paragraph, instr, placeholder=""):
    r = paragraph.add_run(); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), "begin"); f.set(qn("w:dirty"), "true"); r._r.append(f)
    r = paragraph.add_run(); t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = f" {instr} "; r._r.append(t)
    r = paragraph.add_run(); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), "separate"); r._r.append(f)
    paragraph.add_run(placeholder)
    r = paragraph.add_run(); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), "end"); r._r.append(f)


def keep_with_next(p, on=True):
    p.paragraph_format.keep_with_next = on


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# --------------------------------------------------------------------------- styles
st = doc.styles
normal = st["Normal"]
normal.font.name = BODY_FONT; normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for name, size, before, after, italic in (("Heading 1", 16, 0, 18, False), ("Heading 2", 13.5, 16, 6, False), ("Heading 3", 12, 12, 4, True)):
    s = st[name]
    s.font.name = BODY_FONT; s.font.size = Pt(size); s.font.bold = True; s.font.italic = italic; s.font.color.rgb = GREEN
    s.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if False else None
    rf = s.element.rPr.find(qn("w:rFonts"))
    for attr in list(rf.attrib):
        if "Theme" in attr: del rf.attrib[attr]
    rf.set(qn("w:ascii"), BODY_FONT); rf.set(qn("w:hAnsi"), BODY_FONT); rf.set(qn("w:cs"), BODY_FONT)
    pf = s.paragraph_format
    pf.space_before = Pt(before); pf.space_after = Pt(after); pf.keep_with_next = True; pf.line_spacing = 1.15
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
for name in ("Figure Caption", "Table Caption", "Front Title", "Code Block", "Table Text", "Cover", "List Para"):
    st.add_style(name, 1)
for name, italic in (("Figure Caption", False), ("Table Caption", False)):
    s = st[name]; s.base_style = normal
    s.font.size = Pt(10.5); s.font.bold = False; s.font.italic = italic
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.paragraph_format.line_spacing = 1.15; s.paragraph_format.space_before = Pt(4); s.paragraph_format.space_after = Pt(12)
st["Table Caption"].paragraph_format.space_after = Pt(4)
st["Table Caption"].paragraph_format.keep_with_next = True
s = st["Front Title"]; s.base_style = normal; s.font.size = Pt(16); s.font.bold = True; s.font.color.rgb = GREEN
s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT; s.paragraph_format.space_after = Pt(18)
s = st["Code Block"]; s.base_style = normal; s.font.name = CODE_FONT; s.font.size = Pt(8.5)
s.paragraph_format.line_spacing = 1.0; s.paragraph_format.space_after = Pt(0); s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
s = st["Table Text"]; s.base_style = normal; s.font.size = Pt(10)
s.paragraph_format.line_spacing = 1.08; s.paragraph_format.space_after = Pt(0); s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
s = st["List Para"]; s.base_style = normal; s.paragraph_format.left_indent = Cm(0.9); s.paragraph_format.first_line_indent = Cm(-0.6)
s.paragraph_format.space_after = Pt(3)
for lvl in ("TOC 1", "TOC 2", "TOC 3"):
    try:
        s = st[lvl]
    except KeyError:
        s = st.add_style(lvl, 1)
    s.base_style = normal; s.paragraph_format.line_spacing = 1.15; s.paragraph_format.space_after = Pt(2)
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
st["TOC 2"].paragraph_format.left_indent = Cm(0.6); st["TOC 3"].paragraph_format.left_indent = Cm(1.2)

# settings: update fields when the file is opened
settings = doc.settings.element
upd = OxmlElement("w:updateFields"); upd.set(qn("w:val"), "true"); settings.append(upd)


def set_section(section, landscape=False, number_format=None, start=None, header=True, title_page=False):
    section.page_width, section.page_height = (Cm(29.7), Cm(21.0)) if landscape else (Cm(21.0), Cm(29.7))
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    section.left_margin = section.right_margin = Cm(2.54)
    section.top_margin = section.bottom_margin = Cm(2.3)
    section.header_distance = section.footer_distance = Cm(1.1)
    section.different_first_page_header_footer = title_page
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType"); sectPr.append(pg)
    if number_format: pg.set(qn("w:fmt"), number_format)
    if start is not None:
        pg.set(qn("w:start"), str(start))
    elif pg.get(qn("w:start")) is not None:
        del pg.attrib[qn("w:start")]  # continue numbering from the previous section
    for part in (section.header, section.footer):
        part.is_linked_to_previous = False
        for p in part.paragraphs:
            p.clear()
    if header:
        hp = section.header.paragraphs[0]
        hp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        tab_pos = Cm(29.7 - 5.08) if landscape else Cm(21.0 - 5.08)
        hp.paragraph_format.tab_stops.add_tab_stop(tab_pos, alignment=2)
        add_text(hp, SHORT_TITLE, size=9, color=RGBColor(0x55, 0x60, 0x6B))
        hp.add_run("\t")
        add_text(hp, AUTHOR, size=9, color=RGBColor(0x55, 0x60, 0x6B))
        pPr = hp._p.get_or_add_pPr(); bdr = OxmlElement("w:pBdr"); b = OxmlElement("w:bottom")
        b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:space"), "1"); b.set(qn("w:color"), "A0A8B0"); bdr.append(b); pPr.append(bdr)
        fp = section.footer.paragraphs[0]; fp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field(fp, "PAGE", "1")
        for r in fp.runs: r.font.size = Pt(10)


# --------------------------------------------------------------------------- rendering
def col_widths(rows, total):
    """Share the width by content length, but never narrower than the longest word in the column."""
    n = len(rows[0])
    clean = lambda t: re.sub(r"[*`]", "", t)
    lens = [max(min(len(clean(r[c])) if c < len(r) else 0, 60) for r in rows) for c in range(n)]
    lens = [max(l, 6) for l in lens]
    floor = [max(len(w) for r in rows for w in (clean(r[c]).split() if c < len(r) else [""]) or [""]) * 0.072 + 0.16 for c in range(n)]
    widths = [total * l / sum(lens) for l in lens]
    for _ in range(4):
        short = [i for i in range(n) if widths[i] < floor[i]]
        if not short: break
        for i in short: widths[i] = floor[i]
        rest = [i for i in range(n) if i not in short]
        spare = total - sum(widths[i] for i in short)
        tot = sum(lens[i] for i in rest) or 1
        for i in rest: widths[i] = spare * lens[i] / tot
    return widths


def fix_grid(table, widths):
    """Explicit grid and fixed layout, so every word processor keeps the column widths."""
    tbl = table._tbl
    grid = tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w * 1440)))
    tblPr = tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    tw = tblPr.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW"); tblPr.append(tw)
    tw.set(qn("w:w"), str(int(sum(widths) * 1440))); tw.set(qn("w:type"), "dxa")


def render_table(caption, rows, labels, landscape=False, small=False):
    cp = doc.add_paragraph(style="Table Caption")
    add_inline(cp, caption, size=10.5)
    total = LANDSCAPE_WIDTH_IN if landscape else TEXT_WIDTH_IN
    widths = [0.55, 0.95, 1.05, 1.2, 1.45, 1.75, 0.45, 2.3] if rows[0] == HEAD8 else col_widths(rows, total)
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"; table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = table._tbl.tblPr; mar = OxmlElement("w:tblCellMar")
    for side, v in (("top", 25), ("bottom", 25), ("left", 70), ("right", 70)):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); mar.append(e)
    tblPr.append(mar)
    table.autofit = False
    size = 8 if small else 9.5
    for r_i, row in enumerate(rows):
        tr = table.rows[r_i]._tr
        trPr = tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if r_i == 0:
            th = OxmlElement("w:tblHeader"); trPr.append(th)
        for c_i in range(len(rows[0])):
            text = resolve_refs(row[c_i] if c_i < len(row) else "", labels)
            cell = table.cell(r_i, c_i)
            cell.width = Inches(widths[c_i])
            p = cell.paragraphs[0]; p.style = "Table Text"
            add_inline(p, text, size=size, bold=(r_i == 0))
            if r_i == 0:
                set_cell_shading(cell, "DCEBE1")
            if len(rows) <= 14 and r_i < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
    for c_i, w in enumerate(widths):
        for cell in table.columns[c_i].cells:
            cell.width = Inches(w)
    fix_grid(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def render_figure(caption, path, width_in):
    img = ROOT / path
    if not img.exists():
        raise SystemExit(f"missing image {img}")
    with Image.open(img) as im:
        w, h = im.size
    width_in = min(width_in, TEXT_WIDTH_IN)
    if width_in * h / w > 8.4:
        width_in = 8.4 * w / h
    p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(0); keep_with_next(p)
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(str(img), width=Inches(width_in))
    cp = doc.add_paragraph(style="Figure Caption")
    add_inline(cp, caption, size=10.5)


def render(blocks, labels, landscape_tables=False):
    first_h1 = True
    for b in blocks:
        kind = b[0]
        if kind == "h":
            level, title = b[1], b[2]
            if level == 1 and not first_h1:
                page_break()
            first_h1 = False
            doc.add_paragraph(title, style=f"Heading {level}")
        elif kind == "p":
            p = doc.add_paragraph(); add_inline(p, resolve_refs(b[1], labels))
        elif kind == "ul":
            for item in b[1]:
                p = doc.add_paragraph(style="List Para"); p.paragraph_format.tab_stops.add_tab_stop(Cm(0.9))
                add_text(p, "•\t"); add_inline(p, resolve_refs(item, labels))
        elif kind == "ol":
            for n, item in enumerate(b[1], 1):
                p = doc.add_paragraph(style="List Para"); p.paragraph_format.tab_stops.add_tab_stop(Cm(0.9))
                add_text(p, f"{n}.\t"); add_inline(p, resolve_refs(item, labels))
        elif kind == "code":
            for k, line in enumerate(b[1]):
                p = doc.add_paragraph(style="Code Block")
                add_text(p, line if line else " ", code=True, size=10)
                pPr = p._p.get_or_add_pPr(); shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F4F6"); pPr.append(shd)
                if k < len(b[1]) - 1: keep_with_next(p)
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
        elif kind == "table":
            wide = landscape_tables and len(b[3][0]) >= 7
            render_table(b[1], b[3], labels, landscape=wide, small=wide)
        elif kind == "fig":
            render_figure(b[1], b[2], b[4])


# --------------------------------------------------------------------------- read and number all content
body_md = fill((HERE / "report_body.md").read_text(encoding="utf-8"))
app_md = fill((HERE / "appendices.md").read_text(encoding="utf-8"))
front_md = fill((HERE / "front_matter.md").read_text(encoding="utf-8"))
body_blocks = parse(body_md)
app_blocks = parse(app_md)
numbered, labels = number(body_blocks + [("h", 1, "References", None)] + app_blocks)
split = next(i for i, b in enumerate(numbered) if b[0] == "h" and b[2] == "References")
# References is not a numbered chapter
body_numbered, app_numbered = numbered[:split], numbered[split + 1:]
labels_all = labels

# --------------------------------------------------------------------------- cover page (section 1)
sec = doc.sections[0]
set_section(sec, header=False)
for _ in range(3):
    doc.add_paragraph()
for text, size, bold in (("CS6P05ES Project", 20, True), ("Final Report", 18, True)):
    p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_text(p, text, bold=bold, size=size)
doc.add_paragraph()
p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.line_spacing = 1.2
add_text(p, TITLE, bold=True, size=22, color=GREEN)
for _ in range(3):
    doc.add_paragraph()
cover = doc.add_table(rows=5, cols=2); cover.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (k, v) in enumerate((("Name:", AUTHOR), ("ID Number:", "E - 290292"), ("Date:", "[Submission date]"),
                            ("First Supervisor:", "[First Supervisor's Name]"), ("Second Supervisor:", "[Second Supervisor's Name]"))):
    for j, text in enumerate((k, v)):
        c = cover.cell(i, j); c.width = Inches(2.0 if j == 0 else 3.6)
        pp = c.paragraphs[0]; pp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_text(pp, text, bold=(j == 0), size=13)

# --------------------------------------------------------------------------- front matter (section 2, roman)
sec = doc.add_section(WD_SECTION.NEW_PAGE)
set_section(sec, number_format="lowerRoman", start=1)
doc.add_paragraph("Declaration", style="Heading 1")
for line in ["Module: CS6P05ES", "Deadline: [Final deadline]", "Module Leader: [Module Leader's Name]", "Student ID: E - 290292"]:
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(0); add_text(p, line)
p = doc.add_paragraph(); add_text(p, "PLAGIARISM", bold=True)
doc.add_paragraph("You are reminded that there exist regulations concerning plagiarism. Extracts from these regulations are printed below. "
                  "Please sign below to say that you have read and understand these extracts:")
doc.add_paragraph("(signature:) ______________________          Date: [Date]")
doc.add_paragraph("This header sheet should be attached to the work you submit. No work will be accepted without it.")
p = doc.add_paragraph(); add_text(p, "Extracts from University Regulations on Cheating, Plagiarism and Collusion", bold=True)
REG = [
    "Section 2.3: \"The following broad types of offence can be identified and are provided as indicative examples…",
    "(i) Cheating: including taking unauthorised material into an examination; consulting unauthorised material outside the examination hall during the examination; obtaining an unseen examination paper in advance of the examination; copying from another examinee; using an unauthorised calculator during the examination or storing unauthorised material in the memory of a programmable calculator which is taken into the examination; copying coursework.",
    "(ii) Falsifying data in experimental results.",
    "(iii) Personation, where a substitute takes an examination or test on behalf of the candidate. Both candidate and substitute may be guilty of an offence under these Regulations.",
    "(iv) Bribery or attempted bribery of a person thought to have some influence on the candidate's assessment.",
    "(v) Collusion to present joint work as the work solely of one individual.",
    "(vi) Plagiarism, where the work or ideas of another are presented as the candidate's own.",
    "(vii) Other conduct calculated to secure an advantage on assessment.",
    "(viii) Assisting in any of the above.",
    "Some notes on what this means for students:",
    "1. Copying another student's work is an offence, whether from a copy on paper or from a computer file, and in whatever form the intellectual property being copied takes, including text, mathematical notation and computer programs.",
    "2. Taking extracts from published sources without attribution is an offence. To quote ideas, sometimes using extracts, is generally to be encouraged. Quoting ideas is achieved by stating an author's argument and attributing it, perhaps by quoting, immediately in the text, his or her name and year of publication, e.g. \"e = mc2 (Einstein 1905)\". A references section at the end of your work should then list all such references in alphabetical order of authors' surnames. (There are variations on this referencing system which your tutors may prefer you to use.) If you wish to quote a paragraph or so from published work then indent the quotation on both left and right margins, using an italic font where practicable, and introduce the quotation with an attribution.",
]
for line in REG:
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(2)
    add_text(p, line, size=10)

front = parse(front_md)
for b in front:
    if b[0] == "h":
        page_break(); doc.add_paragraph(b[2], style="Heading 1")
    elif b[0] == "p":
        p = doc.add_paragraph(); add_inline(p, b[1])

page_break()
doc.add_paragraph("Table of Contents", style="Front Title")
field(doc.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u', "Right-click and choose Update Field to build the contents.")
page_break()
doc.add_paragraph("List of Figures", style="Front Title")
field(doc.add_paragraph(), 'TOC \\h \\z \\t "Figure Caption,1"', "Update field to build the list of figures.")
page_break()
doc.add_paragraph("List of Tables", style="Front Title")
field(doc.add_paragraph(), 'TOC \\h \\z \\t "Table Caption,1"', "Update field to build the list of tables.")
page_break()
doc.add_paragraph("List of Abbreviations", style="Front Title")
ABBR = [("Abbreviation", "Meaning"), ("API", "Application Programming Interface"), ("ARIMA", "Autoregressive Integrated Moving Average"),
        ("CBSL", "Central Bank of Sri Lanka"), ("CEYPETCO", "Ceylon Petroleum Corporation"), ("CI", "Confidence Interval"),
        ("CSV", "Comma-Separated Values"), ("DCS", "Department of Census and Statistics"), ("DEC", "Dedicated Economic Centre"),
        ("DFD", "Data Flow Diagram"), ("DOA", "Department of Agriculture"), ("HARTI", "Hector Kobbekaduwa Agrarian Research and Training Institute"),
        ("HORDI", "Horticultural Crops Research and Development Institute"), ("JSON", "JavaScript Object Notation"), ("LSTM", "Long Short-Term Memory"),
        ("MAE", "Mean Absolute Error"), ("MAPE", "Mean Absolute Percentage Error"), ("ML", "Machine Learning"), ("PDF", "Portable Document Format"),
        ("R²", "Coefficient of Determination"), ("RF", "Random Forest"), ("RMSE", "Root Mean Square Error"), ("SHA-256", "Secure Hash Algorithm, 256-bit"),
        ("UML", "Unified Modelling Language"), ("XGBoost", "Extreme Gradient Boosting")]
t = doc.add_table(rows=len(ABBR), cols=2); t.style = "Table Grid"
for i, (a, m) in enumerate(ABBR):
    for j, text in enumerate((a, m)):
        c = t.cell(i, j); c.width = Inches(1.6 if j == 0 else 4.6)
        pp = c.paragraphs[0]; pp.style = "Table Text"; add_text(pp, text, bold=(i == 0), size=10.5)
        if i == 0: set_cell_shading(c, "DCEBE1")
fix_grid(t, [1.6, 4.6])

# --------------------------------------------------------------------------- main body (section 3, arabic)
sec = doc.add_section(WD_SECTION.NEW_PAGE)
set_section(sec, number_format="decimal", start=1)
render(body_numbered, labels_all)

# References
page_break()
doc.add_paragraph("References", style="Heading 1")
for line in sorted((l for l in (HERE / "references.md").read_text(encoding="utf-8").splitlines() if l.strip()), key=str.lower):
    p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Cm(1.0); p.paragraph_format.first_line_indent = Cm(-1.0); p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    add_inline(p, line)

# Appendix A in landscape (wide tables), the rest portrait
app_a_end = next(i for i, b in enumerate(app_numbered) if b[0] == "h" and b[1] == 1 and b[2].startswith("Appendix B"))
sec = doc.add_section(WD_SECTION.NEW_PAGE)
set_section(sec, landscape=True)
render(app_numbered[:app_a_end], labels_all, landscape_tables=True)
sec = doc.add_section(WD_SECTION.NEW_PAGE)
set_section(sec)
render(app_numbered[app_a_end:], labels_all)

def export_markdown(path):
    """The same numbered content as one Markdown file (front matter, body, references, appendices)."""
    out = [f"# {TITLE}", "", f"CS6P05ES Project, Final Report. {AUTHOR} (E - 290292). Date: [Submission date].", ""]
    for b in parse(front_md):
        out += [f"## {b[2]}" if b[0] == "h" else b[1], ""]

    def emit(blocks):
        for b in blocks:
            if b[0] == "h":
                out.extend(["#" * (b[1] + 1) + " " + b[2], ""])
            elif b[0] == "p":
                out.extend([resolve_refs(b[1], labels_all), ""])
            elif b[0] == "ul":
                out.extend([f"- {resolve_refs(i, labels_all)}" for i in b[1]] + [""])
            elif b[0] == "ol":
                out.extend([f"{n}. {resolve_refs(i, labels_all)}" for n, i in enumerate(b[1], 1)] + [""])
            elif b[0] == "code":
                out.extend(["```"] + b[1] + ["```", ""])
            elif b[0] == "table":
                rows = b[3]
                out.extend([f"**{b[1]}**", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                           + ["| " + " | ".join(resolve_refs(c, labels_all) for c in r) + " |" for r in rows[1:]] + [""])
            elif b[0] == "fig":
                out.extend([f"![{b[1]}]({b[2]})", "", f"*{b[1]}*", ""])
    emit(body_numbered)
    out += ["## References", ""] + [l + "  " for l in sorted((l for l in (HERE / "references.md").read_text(encoding="utf-8").splitlines() if l.strip()), key=str.lower)] + [""]
    emit(app_numbered)
    path.write_text("\n".join(out), encoding="utf-8")


export_markdown(ROOT / "Final_Report.md")
doc.core_properties.title = TITLE
doc.core_properties.author = AUTHOR
doc.core_properties.subject = "CS6P05ES Final Report, E-290292"
doc.core_properties.comments = ""
doc.save(OUT)
print("written", OUT, "| figures:", sum(1 for v in labels_all.values() if v.startswith("Figure")), "| tables:", sum(1 for v in labels_all.values() if v.startswith("Table")),
      "| tokens:", {k: v for k, v in TOKENS.items() if len(v) < 40})
