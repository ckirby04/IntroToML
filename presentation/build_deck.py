"""Builds IntroToML_Progress.pptx with python-pptx.

Run: python build_deck.py   (needs python-pptx; images come from ./assets and the repo reports)
All numbers come from E01_det_vs_seg/report/*.md and E02_det_vs_seg_transfer/report/*.md.
"""
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
OUT = HERE / "IntroToML_Progress.pptx"

# palette: near-black "MRI" slate dominates; A blue, B orange, cascade green
INK = RGBColor(0x1B, 0x24, 0x30)
MUTED = RGBColor(0x5B, 0x6B, 0x80)
CARD = RGBColor(0xF1, 0xF4, 0xF8)
LINE = RGBColor(0xD5, 0xDC, 0xE5)
DARK = RGBColor(0x12, 0x18, 0x21)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xB8, 0xC4, 0xD3)
A_C = RGBColor(0x2F, 0x7B, 0xD8)
B_C = RGBColor(0xE8, 0x67, 0x3A)
C_C = RGBColor(0x17, 0xA6, 0x73)
HEAD = "Cambria"
BODY = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
_count = {"n": 0}


# ---------- helpers ----------
def new_slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    _count["n"] += 1
    fill = s.background.fill
    fill.solid()
    fill.fore_color.rgb = DARK if dark else WHITE
    if not dark:
        tb = s.shapes.add_textbox(Inches(12.2), Inches(6.95), Inches(0.6), Inches(0.35))
        p = tb.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.RIGHT
        r = p.add_run()
        r.text = str(_count["n"])
        style(r, 11, MUTED)
    return s


def style(run, size, color=INK, bold=False, font=BODY, italic=False):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def text(slide, x, y, w, h, paras, size=16, color=INK, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=6, bold=False, margin=0.0, line_spacing=None):
    """paras: str | list of (str | list of (text, {opts}))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            style(r, o.get("size", size), o.get("color", color), o.get("bold", bold),
                  o.get("font", font), o.get("italic", False))
    return tb


def title(slide, t, dark=False):
    text(slide, 0.6, 0.4, 12.1, 0.9, t, size=36, color=WHITE if dark else INK, font=HEAD,
         bold=True, anchor=MSO_ANCHOR.MIDDLE)


def box(slide, x, y, w, h, fill=CARD, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, radius=0.08):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.text_frame.text = ""
    return sh


def badge(slide, x, y, label, color, d=0.55, size=18):
    c = box(slide, x, y, d, d, fill=color, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    style(r, size, WHITE, True, HEAD)
    return c


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75, dash=False):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = color
    ln.line.width = Pt(width)
    if dash:
        from pptx.enum.dml import MSO_LINE
        ln.line.dash_style = MSO_LINE.DASH
    else:
        end = ln.line._get_or_add_ln()
        from pptx.oxml.ns import qn
        tail = end.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        end.append(tail)
    return ln


def picture(slide, path, x, y, w=None, h=None, name=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    if name:
        pic.name = name
    return pic


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def style_chart(chart, title_text=None, legend=False, size=12):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = MUTED
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.color.rgb = INK
    if title_text:
        chart.has_title = True
        tf = chart.chart_title.text_frame
        tf.text = title_text
        r = tf.paragraphs[0].runs[0]
        style(r, 14, INK, True)
    else:
        chart.has_title = False


def quiet_axes(chart, vmax=None, vmin=0, fmt=None, gridlines=True, major=None):
    va = chart.value_axis
    va.has_major_gridlines = gridlines
    if gridlines:
        va.major_gridlines.format.line.color.rgb = LINE
        va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    if vmax is not None:
        va.maximum_scale = vmax
    va.minimum_scale = vmin
    if major:
        va.major_unit = major
    if fmt:
        va.tick_labels.number_format = fmt
        va.tick_labels.number_format_is_linked = False
    va.tick_labels.font.color.rgb = MUTED
    ca = chart.category_axis
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.color.rgb = INK
    ca.has_major_gridlines = False


def color_series(chart, colors, labels=True, fmt="0.00", label_size=12, pos=XL_LABEL_POSITION.OUTSIDE_END):
    plot = chart.plots[0]
    for s, col in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = col
        s.invert_if_negative = False
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.position = pos
        dl.font.size = Pt(label_size)
        dl.font.color.rgb = INK


def point_colors(chart, colors):
    s = chart.plots[0].series[0]
    for i, col in enumerate(colors):
        pt = s.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col


# ---------- 1. Title ----------
s = new_slide(dark=True)
text(s, 0.8, 1.35, 6.8, 0.4, "INTRO TO ML  ·  PROJECT PROGRESS", size=14, color=SOFT, bold=True)
text(s, 0.8, 1.9, 6.9, 1.3, "Detect or segment?", size=54, color=WHITE, font=HEAD, bold=True)
text(s, 0.8, 3.15, 6.6, 1.2, "Finding brain metastases on MRI with one nnU-Net, trained two ways",
     size=22, color=SOFT)
text(s, 0.8, 5.35, 6.0, 0.9, [[("Clark Kirby", {"bold": True, "color": WHITE, "size": 20})],
                             [("November 30, 2026", {"color": SOFT, "size": 16})]], space_after=2)
picture(s, ASSETS / "schem_image.png", 8.15, 1.25, h=2.4)
picture(s, ASSETS / "schem_mask.png", 8.15, 3.85, h=2.4)
picture(s, ASSETS / "schem_heat.png", 10.75, 3.85, h=2.4)
text(s, 10.75, 1.25, 2.4, 2.4, [[("Same scan,", {"color": WHITE, "bold": True})],
                                [("two targets:", {"color": WHITE, "bold": True})],
                                [("A  ", {"color": A_C, "bold": True}), ("a mask", {"color": SOFT})],
                                [("B  ", {"color": B_C, "bold": True}), ("a peak per lesion", {"color": SOFT})]],
     size=18, anchor=MSO_ANCHOR.MIDDLE, space_after=4)
notes(s, "Hi, I'm Clark. My project asks a simple question about a hard medical problem: if you want a "
         "neural network to find brain metastases, should you train it to outline them (segment) or to point "
         "at them (detect)? Same network, same data, two training targets. (~20 s)")

# ---------- 2. Problem ----------
s = new_slide()
title(s, "Brain metastases: many, small, easy to miss")
text(s, 0.6, 1.6, 6.9, 4.8, [
    [("What they are. ", {"bold": True}), ("Tumors that spread to the brain from cancer elsewhere, most often lung, breast and melanoma. Patients often have several at once.", {})],
    [("How they are treated. ", {"bold": True}), ("Targeted radiation (stereotactic radiosurgery) aims a dose at each lesion, so every lesion has to be found and outlined first.", {})],
    [("Why it is hard. ", {"bold": True}), ("On a contrast MRI, a 2 mm metastasis is only a few bright voxels and looks a lot like a blood vessel. A missed lesion goes untreated.", {})],
], size=18, space_after=16)
for i, (big, small, col) in enumerate([("866", "lesions in 97 test scans", INK),
                                       ("26%", "are smaller than 3 mm", B_C),
                                       ("116", "lesions in a single scan", INK)]):
    y = 1.6 + i * 1.62
    box(s, 8.1, y, 4.6, 1.38)
    text(s, 8.4, y + 0.12, 1.9, 1.1, big, size=46, color=col, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.35, y + 0.12, 2.25, 1.1, small, size=16, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Brain metastases are cancer that has spread to the brain. They're treated with radiosurgery, which "
         "needs every lesion located. The hard part is the tiny ones: in my test set a quarter of lesions are "
         "under 3 mm, and one scan has 116 lesions. (~45 s)")

# ---------- 3. Why it matters ----------
s = new_slide()
title(s, "Why it matters: a pointer before a mask")
steps = [("MRI scan", "contrast T1, 3D"), ("Model flags spots", "candidate points + scores"),
         ("Radiologist reviews", "keeps or dismisses each"), ("Plan treatment", "outline and dose each lesion")]
for i, (h, sub) in enumerate(steps):
    x = 0.6 + i * 3.1
    fill = B_C if i == 1 else CARD
    box(s, x, 1.65, 2.6, 1.35, fill=fill)
    text(s, x + 0.2, 1.75, 2.2, 0.55, h, size=18, bold=True, color=WHITE if i == 1 else INK,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.2, 2.3, 2.2, 0.6, sub, size=14, color=WHITE if i == 1 else MUTED)
    if i < 3:
        arrow(s, x + 2.65, 2.32, x + 3.05, 2.32)
box(s, 0.6, 3.5, 5.95, 2.9)
badge(s, 0.9, 3.8, "+", INK, d=0.5)
text(s, 1.55, 3.78, 4.8, 0.55, "Clinical angle", size=20, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.9, 4.45, 5.45, 1.9, ["Radiologists are not ready to let a model outline tumors on its own. A model "
                              "that points at suspicious spots, as a pre-read or a second check, fits the "
                              "workflow today.",
                              "A missed lesion costs more than a false alarm a radiologist can dismiss."],
     size=16, space_after=8)
box(s, 6.75, 3.5, 5.95, 2.9)
badge(s, 7.05, 3.8, "?", INK, d=0.5)
text(s, 7.7, 3.78, 4.8, 0.55, "Machine learning angle", size=20, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
text(s, 7.05, 4.45, 5.45, 1.9, ["Finding lesions is what matters here, yet most models are trained to segment "
                               "and points are read off the mask afterwards.",
                               "Does the training objective change which lesions get found?"],
     size=16, space_after=8)
notes(s, "Clinically, radiologists aren't ready to hand outlining over to a model, but a model that points out "
         "candidates for them to check is realistic. That makes detection the real task. Most published models "
         "are segmenters, though, so I wanted to test whether training for detection directly actually helps. (~45 s)")

# ---------- 4. Prior approaches ----------
s = new_slide()
title(s, "How people have tackled it")
cols = [("Manual reading", "Non-ML", MUTED,
         "Radiologists scroll thin-slice contrast MRI slice by slice. Reliable for large lesions; slow, and small ones get missed."),
        ("Classical CAD", "Non-ML", MUTED,
         "Hand-built blob and template-matching filters (e.g. Laplacian of Gaussian) plus hand-made features and rules. Vessels cause many false alarms."),
        ("Deep segmentation", "ML", A_C,
         "Voxel-wise CNNs: DeepMedic, 3D U-Net, nnU-Net (BraTS-METS challenge). Lesions are read off the predicted mask."),
        ("Deep detection", "ML", B_C,
         "Object detectors (Faster R-CNN, RetinaNet) and CenterNet-style heatmaps predict one box or point per lesion.")]
for i, (h, tag, col, body) in enumerate(cols):
    x = 0.6 + i * 3.08
    box(s, x, 1.6, 2.85, 3.75)
    tagbox = box(s, x + 0.25, 1.85, 1.0 if tag == "ML" else 1.25, 0.38, fill=col, radius=0.5)
    text(s, x + 0.25, 1.85, 1.0 if tag == "ML" else 1.25, 0.38, tag, size=12, color=WHITE, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    text(s, x + 0.25, 2.35, 2.45, 0.9, h, size=20, bold=True, font=HEAD, anchor=MSO_ANCHOR.BOTTOM, space_after=0)
    text(s, x + 0.25, 3.3, 2.4, 2.0, body, size=15, color=INK)
box(s, 0.6, 5.65, 12.1, 0.95, fill=DARK)
text(s, 0.9, 5.65, 11.6, 0.95, [[("My question: ", {"bold": True, "color": B_C}),
                                 ("with the same network, data and training recipe, does a detection objective "
                                  "find more lesions than a segmentation objective?", {"color": WHITE})]],
     size=18, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
notes(s, "Before deep learning, it was manual reading or classical computer-aided detection with hand-built blob "
         "filters, which fire on vessels a lot. Deep learning went two ways: segmentation networks like nnU-Net, "
         "and detection networks like RetinaNet or CenterNet. Papers usually change the architecture too, so you "
         "can't tell whether the objective or the network made the difference. I hold everything fixed except the "
         "objective. (~50 s)")

# ---------- 5. Data ----------
s = new_slide()
title(s, "Data: 454 contrast-enhanced brain MRIs")
rows = [("Sources", "Stanford BrainMetShare + UCSF brain metastases (UCSF-BMSR), public research datasets"),
        ("Collected", "Clinical MRI of patients with brain metastases; every lesion outlined by experts"),
        ("Data type", "3D volume, 1 channel (post-contrast T1), resampled to 1 mm voxels, median 172 × 137 × 137"),
        ("Labels", "Binary voxel mask; each connected blob counts as one lesion"),
        ("Split", "By patient. Train 357 scans / 268 patients; test 97 scans / 66 patients, 866 lesions")]
for i, (k, v) in enumerate(rows):
    y = 1.6 + i * 0.98
    text(s, 0.6, y, 1.5, 0.9, k, size=15, bold=True, color=MUTED)
    text(s, 2.05, y, 4.65, 0.9, v, size=15)
for i, (img, cap) in enumerate([("wiki_mets_t1ce.png", "Image: contrast T1 slice"),
                                ("wiki_mets_label.png", "Label: lesion mask")]):
    x = 7.35 + i * 2.95
    picture(s, ASSETS / img, x, 1.6, h=2.6)
    text(s, x - 0.1, 4.28, 2.54, 0.4, cap, size=14, bold=True, color=INK if i == 0 else A_C, align=PP_ALIGN.CENTER)
arrow(s, 9.78, 2.9, 10.22, 2.9)
box(s, 7.05, 4.85, 5.7, 1.6)
text(s, 7.3, 4.92, 5.3, 1.48, [
    [("Example image–label pair. ", {"bold": True}),
     ("A public brain MRI with one ring-enhancing metastasis (patient scans from our dataset aren't shown). "
      "The mask is drawn for illustration; in our data, experts outline every lesion in 3D.", {})],
    [("Image: Nevit Dilmen, Wikimedia Commons, CC BY-SA 3.0", {"size": 11, "color": MUTED})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=4)
notes(s, "Two public datasets: Stanford's BrainMetShare and UCSF's brain metastases set. I use the post-contrast "
         "T1 MRI, a 3D grayscale volume at 1 mm resolution, with a voxel mask of every lesion as the label. 454 "
         "scans from 334 patients, split by patient so no person is in both train and test. The test set has 97 "
         "scans and 866 lesions. The picture is a public example, not from our data: a contrast MRI with one ring-shaped metastasis, and its mask. (~50 s)")

# ---------- 6. Data visualizations ----------
s = new_slide()
title(s, "Lesion sizes and counts vary a lot")
cd = CategoryChartData()
cd.categories = ["< 3 mm", "3–10 mm", "> 10 mm"]
cd.add_series("Lesions", (225, 549, 92))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(5.9), Inches(4.0), cd)
ch = gf.chart
style_chart(ch, "Test lesions by size (equivalent sphere diameter)")
quiet_axes(ch, vmax=650, major=200)
color_series(ch, [INK], fmt="0", label_size=14)
point_colors(ch, [B_C, INK, A_C])
ch.plots[0].gap_width = 70
cd = CategoryChartData()
cd.categories = ["1", "2–3", "4–5", "6–10", "11–20", "21–50", "> 50"]
cd.add_series("Scans", (14, 28, 15, 13, 20, 5, 2))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.85), Inches(1.5), Inches(5.9), Inches(4.0), cd)
ch = gf.chart
style_chart(ch, "Test scans by number of lesions")
quiet_axes(ch, vmax=32, major=10)
color_series(ch, [MUTED], fmt="0", label_size=14)
ch.plots[0].gap_width = 50
text(s, 0.8, 5.75, 5.6, 0.8, [[("1 in 4 lesions is under 3 mm", {"bold": True}),
                               (": the hard case. Most are 3–10 mm.", {})]], size=16)
text(s, 7.05, 5.75, 5.6, 0.8, [[("Median 4 lesions per scan", {"bold": True}),
                                (", but 2 scans have more than 50 (up to 116).", {})]], size=16)
notes(s, "Two views of the test data. Left: lesion size. Most are 3 to 10 mm, but a quarter are under 3 mm, which "
         "is where models struggle. Right: lesions per scan. The median is 4, but the tail is long, up to 116. "
         "(~35 s)")

# ---------- 7. Framing ----------
s = new_slide()
title(s, "Supervised learning, framed two ways")
for i, (img, cap, col) in enumerate([("schem_image.png", "Input: MRI patch", INK),
                                     ("schem_mask.png", "A target: voxel mask", A_C),
                                     ("schem_heat.png", "B target: Gaussian peaks", B_C)]):
    x = 0.6 + i * 2.05
    picture(s, ASSETS / img, x, 1.6, w=1.85)
    text(s, x, 3.5, 1.85, 0.4, cap, size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
text(s, 0.6, 4.05, 5.95, 0.4, "Schematic, not real data. Both targets come from the same expert masks.",
     size=12, color=MUTED)
text(s, 0.6, 4.6, 5.95, 1.8, [
    [("Supervised", {"bold": True}), (": every voxel has a label.", {})],
    [("A", {"bold": True, "color": A_C}), (" is voxel-wise ", {}), ("classification", {"bold": True}), (" (lesion or not).", {})],
    [("B", {"bold": True, "color": B_C}), (" is dense heatmap ", {}), ("regression", {"bold": True}),
     (" (how close to a lesion center), trained with a focal loss.", {})]], size=16, space_after=8)
for i, (lab, col, head, lines) in enumerate([
        ("A", A_C, "Segmenter", [("Output", "P(lesion) for every voxel"), ("Loss", "Dice + cross-entropy (nnU-Net default)"),
                                 ("To points", "one point per blob, scored by its max probability")]),
        ("B", B_C, "Heatmap detector", [("Target", "Gaussian at each lesion center, σ 1–4 mm"),
                                        ("Loss", "CenterNet focal loss"),
                                        ("To points", "local maxima after 3 mm non-max suppression")])]):
    y = 1.6 + i * 2.45
    box(s, 6.95, y, 5.8, 2.2)
    badge(s, 7.2, y + 0.25, lab, col)
    text(s, 7.95, y + 0.25, 4.5, 0.55, head, size=20, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    for j, (k, v) in enumerate(lines):
        text(s, 7.25, y + 0.95 + j * 0.4, 1.2, 0.4, k, size=14, bold=True, color=MUTED)
        text(s, 8.45, y + 0.95 + j * 0.4, 4.15, 0.4, v, size=14)
notes(s, "Both models are supervised; the labels come from the same expert masks. Model A is ordinary "
         "segmentation: classify every voxel as lesion or not. To compare it as a detector, each blob in its mask "
         "becomes one point. Model B is a heatmap detector: it learns to output a small Gaussian bump at the center "
         "of each lesion, like CenterNet, and its detections are the local peaks. So this is classification versus "
         "heatmap regression. (~60 s)")

# ---------- 8. nnU-Net ----------
s = new_slide()
title(s, "Backbone: nnU-Net, a self-configuring U-Net")
chans = [32, 64, 128, 256, 320, 320]
bw, step = 0.42, 0.5
x0, ytop = 0.75, 2.2
pos = {}
for i, c in enumerate(chans):  # encoder going down
    h = 2.3 - i * 0.32
    x = x0 + i * step
    y = ytop + i * 0.42
    box(s, x, y, bw, h, fill=A_C if i < 5 else INK, shape=MSO_SHAPE.RECTANGLE)
    text(s, x - 0.1, y + h + 0.02, bw + 0.2, 0.3, str(c), size=11, color=MUTED, align=PP_ALIGN.CENTER)
    pos[("e", i)] = (x, y, h)
for k, i in enumerate(range(4, -1, -1)):  # decoder going up
    h = 2.3 - i * 0.32
    x = x0 + (6 + k) * step
    y = ytop + i * 0.42
    box(s, x, y, bw, h, fill=C_C, shape=MSO_SHAPE.RECTANGLE)
    ex, ey, eh = pos[("e", i)]
    arrow(s, ex + bw, ey + 0.2, x, y + 0.2, color=MUTED, width=1.25, dash=True)
text(s, 0.6, 1.55, 3.0, 0.4, "Input patch 160 × 128 × 112", size=13, color=MUTED)
text(s, 4.6, 1.55, 2.2, 0.4, "Per-voxel output", size=13, color=MUTED, align=PP_ALIGN.RIGHT)
text(s, 0.6, 5.35, 6.2, 1.1, [
    [("■ ", {"color": A_C}), ("Encoder: 3×3×3 convs + downsampling, learns context  ", {}),],
    [("■ ", {"color": C_C}), ("Decoder: upsamples back to full resolution", {})],
    [("- - ", {"color": MUTED, "bold": True}), ("Skip connections carry fine detail across", {})]], size=13, space_after=3)
text(s, 7.2, 1.6, 5.5, 4.9, [
    [("What it is. ", {"bold": True}), ("A U-Net whose patch size, depth, resampling and normalization are set "
                                       "automatically from the dataset (Isensee et al., Nature Methods 2021).", {})],
    [("Why it. ", {"bold": True}), ("The standard off-the-shelf baseline for medical segmentation; it has won or "
                                   "placed near the top of many challenges.", {})],
    [("Plan used here. ", {"bold": True}), ("3D, 6 stages, 32 → 320 channels, instance norm + LeakyReLU, deep "
                                           "supervision.", {})],
    [("Fair comparison. ", {"bold": True, "color": B_C}), ("A and B share data, plans, augmentation, schedule and seed. The output layer, target and loss differ "
                                                          "(B also needed bf16 and a small output-layer init to train stably).", {})]],
     size=16, space_after=14)
notes(s, "Both models use nnU-Net. A U-Net has an encoder that shrinks the image to understand context, a decoder "
         "that grows it back, and skip connections that keep fine detail, which matters for tiny lesions. nnU-Net "
         "configures all of this automatically from the data and is the go-to baseline in medical imaging. I chose "
         "it because it's strong and because it let me keep the two arms nearly identical: only the output layer, target and loss differ, plus two stability fixes B needed. (~50 s)")

# ---------- 9. Pipeline ----------
s = new_slide()
title(s, "From scan to prediction")
pipe = [("1", "Preprocess", "Resample to 1 mm, crop away empty background, z-score intensities inside the brain"),
        ("2", "Sample", "Random 160×128×112 patches, batch of 2, lesion-containing patches oversampled"),
        ("3", "Augment", "Rotation, scaling, noise, blur, brightness, contrast, gamma, mirroring"),
        ("4", "Train", "250 epochs, SGD with poly learning-rate decay, same seed; ~6.5 h per model on one GPU"),
        ("5", "Predict", "Sliding window + mirrored test-time augmentation, then map → scored points")]
for i, (n, h, body) in enumerate(pipe):
    x = 0.6 + i * 2.47
    box(s, x, 1.65, 2.25, 3.2)
    badge(s, x + 0.22, 1.88, n, INK, d=0.5, size=16)
    text(s, x + 0.22, 2.5, 1.9, 0.45, h, size=19, bold=True, font=HEAD)
    text(s, x + 0.22, 3.0, 1.85, 1.8, body, size=14)
    if i < 4:
        arrow(s, x + 2.27, 3.25, x + 2.45, 3.25)
box(s, 0.6, 5.2, 5.95, 1.25)
text(s, 0.85, 5.25, 5.5, 1.15, [[("B's target is built on the fly ", {"bold": True, "color": B_C}),
                                 ("from the augmented mask, so peaks always sit on the lesion after rotation or scaling.", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
box(s, 6.75, 5.2, 5.95, 1.25)
text(s, 7.0, 5.25, 5.5, 1.15, [[("Pre-registered: ", {"bold": True}),
                                ("every threshold, the matching rule and the metrics were fixed in writing before "
                                 "the models were trained.", {})]], size=15, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
notes(s, "Preprocessing is nnU-Net's standard recipe: resample, crop, normalize intensities. Training samples random "
         "3D patches with heavy augmentation, for 250 epochs, about six and a half hours per model. For model B "
         "the heatmap target is generated after augmentation so it stays aligned. I wrote down every threshold and "
         "metric before training, so I couldn't tune my way to a result. (~45 s)")

# ---------- 10. Metrics ----------
s = new_slide()
title(s, "Metrics: both models scored as detectors")
mets = [("Hit", "a predicted point within 2 mm of a labeled lesion"),
        ("Sensitivity", "share of lesions found (recall)"),
        ("FP / scan", "false alarms a radiologist would have to dismiss"),
        ("FROC curve", "sensitivity vs. FP/scan as the score threshold moves"),
        ("CPM (primary)", "mean sensitivity at 0.5, 1, 2 and 3 FP/scan; one number, 0 to 1"),
        ("Dice", "mask overlap, 2|P ∩ G| / (|P| + |G|): outline quality"),
        ("95% CI", "bootstrap over patients, 2,000 resamples")]
for i, (k, v) in enumerate(mets):
    y = 1.6 + i * 0.68
    text(s, 0.6, y, 1.85, 0.6, k, size=16, bold=True, color=B_C if i == 4 else INK)
    text(s, 2.45, y, 3.9, 0.65, v, size=15)
cd = XyChartData()
sa = cd.add_series("A: segmenter")
for x, y in [(0.1, 0.0), (0.25, 0.545), (0.5, 0.694), (0.75, 0.741), (1.0, 0.775), (1.5, 0.801)]:
    sa.add_data_point(x, y)
sb = cd.add_series("B: heatmap detector")
for x, y in [(0.1, 0.388), (0.25, 0.545), (0.5, 0.647), (0.75, 0.704), (1.0, 0.730), (1.5, 0.796),
             (2.0, 0.837), (2.5, 0.852), (3.0, 0.876), (4.0, 0.901)]:
    sb.add_data_point(x, y)
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(6.75), Inches(1.5), Inches(6.0),
                        Inches(4.55), cd)
ch = gf.chart
style_chart(ch, "FROC, all 866 test lesions", legend=True)
quiet_axes(ch, vmax=1.0, major=0.2, fmt="0.0")
ch.category_axis.maximum_scale = 4.0
ch.category_axis.minimum_scale = 0
ch.category_axis.major_unit = 0.5
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "False positives per scan"
style(ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0], 12, MUTED)
ch.value_axis.has_title = True
ch.value_axis.axis_title.text_frame.text = "Sensitivity"
style(ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0], 12, MUTED)
for ser, col in zip(ch.plots[0].series, [A_C, B_C]):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(3)
    ser.smooth = False
text(s, 6.95, 6.15, 5.8, 0.5, "A runs out of candidates at 1.5 FP/scan; B keeps climbing.", size=14, color=MUTED)
notes(s, "To compare fairly, both models are scored as detectors. A point counts as a hit if it lands within 2 mm "
         "of a lesion. Sweeping the score threshold traces an FROC curve: sensitivity against false positives per "
         "scan. My primary metric, CPM, averages sensitivity at 0.5, 1, 2 and 3 false positives per scan. Dice "
         "measures outline quality, and confidence intervals come from resampling patients. On the right is the "
         "actual result for all lesions. (~60 s)")

# ---------- 11. Results ----------
s = new_slide()
title(s, "Overall a tie, but they miss different lesions")
for i, (lab, col, val) in enumerate([("A", A_C, "0.768"), ("B", B_C, "0.773")]):
    y = 1.65 + i * 1.35
    box(s, 0.6, y, 3.9, 1.15)
    badge(s, 0.85, y + 0.3, lab, col)
    text(s, 1.6, y + 0.05, 1.6, 1.05, val, size=38, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 3.25, y + 0.05, 1.15, 1.05, "CPM", size=16, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 4.4, 3.9, 0.9, [[("B − A = +0.005", {"bold": True}), ("  [−0.036, 0.055]", {"color": MUTED})],
                             [("No overall difference.", {})]], size=16, space_after=2)
cd = CategoryChartData()
cd.categories = ["All lesions", "< 3 mm", "3–10 mm", "> 10 mm"]
cd.add_series("A: segmenter", (0.775, 0.520, 0.845, 0.978))
cd.add_series("B: heatmap detector", (0.730, 0.533, 0.811, 0.728))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.85), Inches(1.5), Inches(7.9), Inches(3.9), cd)
ch = gf.chart
style_chart(ch, "Sensitivity at 1 false positive per scan, by lesion size", legend=True)
quiet_axes(ch, vmax=1.0, major=0.25, fmt="0.00")
color_series(ch, [A_C, B_C], fmt="0.00", label_size=13)
ch.plots[0].gap_width = 60
ch.plots[0].overlap = -5
for i, (big, col, small) in enumerate([
        ("−0.25", A_C, "Large lesions (> 10 mm): A finds far more. CI [−0.35, −0.12]"),
        ("+0.10", B_C, "Small lesions (< 3 mm): B's CPM is higher. CI [0.05, 0.18]"),
        ("74 vs 29", INK, "lesions found only by B vs only by A, at fixed thresholds")]):
    x = 0.6 + i * 4.1
    text(s, x, 5.65, 1.6 if i < 2 else 1.9, 0.8, big, size=26, bold=True, font=HEAD, color=col, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + (1.6 if i < 2 else 1.9), 5.65, 2.3 if i < 2 else 2.0, 0.85, small, size=13, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Headline: on the primary metric it's a tie, 0.768 versus 0.773. But the size breakdown tells a different "
         "story. On large lesions the segmenter finds 98% and the detector only 73%. On small lesions the detector "
         "does better once you allow more false positives. And their errors only partly overlap: 74 lesions only B "
         "found, 29 only A found. That suggests combining them. (~60 s)")

# ---------- 12. Cascade ----------
s = new_slide()
title(s, "Combining them beats both: B finds, A scores")
flow = [("B", B_C, "B proposes", "every heatmap peak is a candidate"),
        ("A", A_C, "A re-scores", "score = average of B's peak and A's blob probability"),
        ("C1", C_C, "A outlines", "A's mask gives each kept lesion its outline")]
for i, (lab, col, h, sub) in enumerate(flow):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.7, 1.25)
    badge(s, x + 0.2, 1.95, lab, col, size=15 if len(lab) > 1 else 18)
    text(s, x + 0.95, 1.68, 2.6, 0.45, h, size=18, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.95, 2.12, 2.65, 0.7, sub, size=13, color=MUTED)
    if i < 2:
        arrow(s, x + 3.73, 2.22, x + 4.07, 2.22)
cd = CategoryChartData()
cd.categories = ["A: segmenter", "B: detector", "C1: cascade"]
cd.add_series("CPM", (0.768, 0.773, 0.802))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(3.15), Inches(6.6), Inches(3.3), cd)
ch = gf.chart
style_chart(ch, "CPM, all lesions (higher is better)", size=13)
quiet_axes(ch, vmax=0.85, vmin=0.6, major=0.05, fmt="0.00")
color_series(ch, [INK], fmt="0.000", label_size=14)
point_colors(ch, [A_C, B_C, C_C])
ch.plots[0].gap_width = 45
ch.category_axis.reverse_order = True
for i, (big, small) in enumerate([("+0.029", "CPM over B  [0.005, 0.045]"),
                                  ("0.957", "large-lesion sensitivity at 1 FP/scan (B: 0.728)"),
                                  ("0.764", "per-scan Dice; outlines as good as A's (0.769)")]):
    y = 3.2 + i * 1.05
    text(s, 7.6, y, 1.9, 0.9, big, size=30, bold=True, font=HEAD, color=C_C, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 9.55, y, 3.2, 0.9, small, size=15, anchor=MSO_ANCHOR.MIDDLE)
text(s, 7.6, 6.3, 5.15, 0.5, "Exploratory: designed after seeing the A/B results, so it needed a fresh test.",
     size=12, color=MUTED)
notes(s, "So I tried a two-stage cascade with no extra training. B proposes candidates, which helps small lesions. "
         "Each candidate is re-scored using A's confidence, which rescues the large lesions B was unsure about. "
         "And A's mask provides the outline. CPM goes up to 0.802, above both; the gain over B is statistically significant, and segmentation "
         "quality is unchanged. Caveat: I designed it after seeing the results, so it needed an independent "
         "test. (~50 s)")

# ---------- 13. Transfer ----------
s = new_slide()
title(s, "Does it transfer? Not to crowded cells")
picture(s, ASSETS / "cells_dense.png", 0.6, 1.6, w=4.4)
text(s, 0.6, 5.35, 4.4, 1.2, [[("LIVECell", {"bold": True}),
                               (": phase-contrast microscopy, 1,460 test images, 444,600 cells, 8 cell lines. "
                                "White = true outlines, blue = A's mask, orange = B's points. LIVECell (Edlund et al. 2021), CC BY-NC 4.0.", {})]], size=13, color=MUTED)
cd = CategoryChartData()
cd.categories = ["A: segmenter", "B: detector", "C1: cascade"]
cd.add_series("CPM", (0.093, 0.943, 0.945))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(5.35), Inches(1.5), Inches(7.4), Inches(2.6), cd)
ch = gf.chart
style_chart(ch, "CPM on the cell test set (same design, pre-registered)", size=13)
quiet_axes(ch, vmax=1.0, major=0.25, fmt="0.00")
color_series(ch, [INK], fmt="0.000", label_size=14)
point_colors(ch, [A_C, B_C, C_C])
ch.plots[0].gap_width = 45
ch.category_axis.reverse_order = True
box(s, 5.35, 4.35, 3.55, 2.1)
text(s, 5.6, 4.45, 3.1, 1.95, [[("Why A collapses", {"bold": True, "color": A_C, "size": 17})],
                               [("Its masks are good (Dice 0.906), but touching cells merge: ~32 blobs per "
                                 "image vs ~304 cells, and one blob = one point.", {})]], size=14, space_after=6)
box(s, 9.2, 4.35, 3.55, 2.1)
text(s, 9.45, 4.45, 3.1, 1.95, [[("Cascade not replicated", {"bold": True, "color": C_C, "size": 17})],
                                [("B is already near its ceiling. C1 adds only +0.003 [0.000, 0.004] CPM.", {})]],
     size=14, space_after=6)
notes(s, "To test the cascade properly, I pre-registered a second experiment on a very different dataset: cell "
         "microscopy, with hundreds of cells per image. The detector transfers great, 0.94 CPM. The segmenter "
         "collapses to 0.09, not because its masks are bad, but because touching cells merge into one blob and one "
         "blob can only count as one cell. With A that weak, the cascade had nothing to add, so the main hypothesis "
         "did not replicate. (~55 s)")

# ---------- 14. Discussion ----------
s = new_slide()
title(s, "Takeaways and future work")
text(s, 0.6, 1.55, 5.6, 0.5, "What I learned", size=22, bold=True, font=HEAD)
take = ["The training objective changes which lesions are found more than how many.",
        "Reading points off a mask only works when objects rarely touch.",
        "Detector + segmenter is promising for sparse lesions like brain metastases, but still needs confirming on new MRI data."]
for i, t in enumerate(take):
    y = 2.2 + i * 1.4
    badge(s, 0.6, y, str(i + 1), INK, d=0.5, size=16)
    text(s, 1.3, y - 0.05, 4.9, 1.3, t, size=16)
box(s, 6.75, 1.55, 6.0, 4.95)
text(s, 7.05, 1.7, 5.5, 0.5, "Future work", size=22, bold=True, font=HEAD)
fut = [("Size-aware heatmaps", "let B's Gaussian grow with lesion size, or add a size head"),
       ("Instance-aware read-out", "split touching objects (e.g. watershed) before counting"),
       ("Calibrated scores", "A's probabilities saturate at 1.0, so it ranks poorly"),
       ("Confirm the cascade", "new MRI data, more seeds and folds"),
       ("Reader study", "are some 'false positives' real but unlabeled lesions?")]
for i, (k, v) in enumerate(fut):
    y = 2.35 + i * 0.8
    text(s, 7.05, y, 5.5, 0.8, [[(k + ": ", {"bold": True, "color": B_C}), (v, {})]], size=15)
notes(s, "Three takeaways. The objective changes which lesions you find. Turning a segmentation into detections "
         "only works when objects don't touch. And combining a detector with a segmenter looks promising for brain "
         "metastases, but needs another test. Next steps: size-aware heatmap targets to fix B on large lesions, "
         "splitting touching objects, better-calibrated scores, more seeds, and having a reader check whether some "
         "false positives are real lesions. (~50 s)")

# ---------- 15. Contributions ----------
s = new_slide()
title(s, "Contributions")
box(s, 0.6, 1.6, 12.1, 4.85)
badge(s, 1.0, 2.0, "CK", INK, d=1.1, size=26)
text(s, 2.4, 2.05, 6.0, 0.55, "Clark Kirby", size=24, bold=True, font=HEAD)
text(s, 2.4, 2.6, 6.0, 0.45, "Individual project: all work", size=15, color=MUTED)
contrib = ["Problem framing and background research",
           "Data preparation and patient-level train/test split",
           "Pre-registration of both experiments",
           "Heatmap trainer, point decoding and evaluation code, with unit tests",
           "Training all models (A, B, crop segmenter) on a local GPU",
           "Two-stage cascade, LIVECell transfer study, analysis and slides"]
for i, c in enumerate(contrib):
    col, row = divmod(i, 3)
    x = 2.4 + col * 5.0
    y = 3.4 + row * 0.85
    box(s, x, y + 0.12, 0.18, 0.18, fill=B_C, shape=MSO_SHAPE.OVAL)
    text(s, x + 0.35, y, 4.5, 0.8, c, size=16)
notes(s, "This was an individual project, so I did all of it. (~10 s)")

# ---------- 16. Thanks ----------
s = new_slide(dark=True)
text(s, 0.8, 2.3, 11.7, 1.3, "Thank you", size=54, color=WHITE, font=HEAD, bold=True)
text(s, 0.8, 3.6, 11.7, 0.8, "Questions?", size=26, color=SOFT)
text(s, 0.8, 5.6, 11.7, 0.5, "Code and full results: github.com/ckirby04/IntroToML", size=16, color=SOFT)
notes(s, "Thanks, happy to take questions.")

prs.save(OUT)
print("saved", OUT)
