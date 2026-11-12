from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_DIR / "report_assets"
OUTPUT_PATH = PROJECT_DIR / "ELG5163_Method_Comparison_Summary.docx"

TRAD_PATH = (
    PROJECT_DIR
    / "traditional_method/results/enhanced_traditional_evaluation.csv"
)
AI_PATH = PROJECT_DIR / "ai_method/results/rtmpose_evaluation.csv"
AI_FEATURES_PATH = PROJECT_DIR / "ai_method/results/rtmpose_features.csv"
OVERLAY_PATH = (
    PROJECT_DIR
    / "ai_method/results/keypoint_overlays/R_ABD_F_NONE_02_rtmpose.jpg"
)
CHART_PATH = ASSET_DIR / "method_comparison.png"

NAVY = "19365F"
BLUE = "DDEBF7"
PALE = "F3F6FA"
GRAY = "666666"
LIGHT_BORDER = "D9D9D9"


def set_cell_shading(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_border(cell, color=LIGHT_BORDER, size="6"):
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in {
        "top": top,
        "start": start,
        "bottom": bottom,
        "end": end,
    }.items():
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def set_col_widths(table, widths):
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)


def style_table(table, header=True, first_col_left=True, font_size=9.5):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row_index, row in enumerate(table.rows):
        for col_index, cell in enumerate(row.cells):
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if header and row_index == 0:
                set_cell_shading(cell, NAVY)
            elif row_index % 2 == 0:
                set_cell_shading(cell, PALE)
            for paragraph in cell.paragraphs:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT
                    if first_col_left and col_index == 0
                    else WD_ALIGN_PARAGRAPH.CENTER
                )
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(font_size)
                    run._element.get_or_add_rPr().rFonts.set(
                        qn("w:eastAsia"), "Arial"
                    )
                    if header and row_index == 0:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
    if header:
        set_repeat_table_header(table.rows[0])


def add_table(doc, headers, rows, widths, font_size=9.5):
    table = doc.add_table(rows=1, cols=len(headers))
    for index, text in enumerate(headers):
        table.rows[0].cells[index].text = str(text)
    for row in rows:
        cells = table.add_row().cells
        for index, text in enumerate(row):
            cells[index].text = str(text)
    set_col_widths(table, widths)
    style_table(table, font_size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.add_run(item)
        paragraph.paragraph_format.space_after = Pt(3)


def percentage(value):
    return f"{100 * value:.2f}%"


def metric(evaluation, method, protocol):
    row = evaluation[
        (evaluation["method"] == method)
        & (evaluation["protocol"] == protocol)
    ]
    return float(row.iloc[0]["accuracy"])


def cross_subject_average(evaluation, method):
    rows = evaluation[
        (evaluation["method"] == method)
        & evaluation["protocol"].str.startswith("train_other_test_subject_")
    ]
    return float(rows["accuracy"].mean())


def create_chart(traditional, ai):
    methods = [
        ("Baseline\nSVM", traditional, "baseline_fixed_svm"),
        ("Traditional\nRandom Forest", traditional, "enhanced_geometry_random_forest"),
        ("Traditional\nGradient Boosting", traditional, "enhanced_geometry_gradient_boosting"),
        ("RTMPose\nSVM", ai, "rtmpose_keypoints_svm"),
        ("RTMPose\nRandom Forest", ai, "rtmpose_keypoints_random_forest"),
    ]
    cv_values = [metric(table, method, "stratified_6fold") for _, table, method in methods]
    cross_values = [cross_subject_average(table, method) for _, table, method in methods]
    labels = [label for label, _, _ in methods]

    figure, axis = plt.subplots(figsize=(10, 4.8))
    x = range(len(methods))
    width = 0.36
    axis.bar(
        [value - width / 2 for value in x],
        [100 * value for value in cv_values],
        width,
        label="Stratified 6-fold",
        color="#4C78A8",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [100 * value for value in cross_values],
        width,
        label="Cross-subject average",
        color="#F28E2B",
    )
    for offset, values in [(-width / 2, cv_values), (width / 2, cross_values)]:
        for index, value in enumerate(values):
            axis.text(
                index + offset,
                100 * value + 1.2,
                f"{100 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(list(x), labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Traditional and RTMPose comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(CHART_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.extend([field_begin, instruction, field_end])


def configure_document(doc):
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    add_page_number(section.footer.paragraphs[0])

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    title = doc.styles["Title"]
    title.font.name = "Arial"
    title.font.size = Pt(22)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")

    for style_name, size in [("Heading 1", 15), ("Heading 2", 12)]:
        style = doc.styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.keep_with_next = True


def build_document():
    traditional = pd.read_csv(TRAD_PATH)
    ai = pd.read_csv(AI_PATH)
    ai_features = pd.read_csv(AI_FEATURES_PATH)
    baseline_cv = metric(traditional, "baseline_fixed_svm", "stratified_6fold")
    hgb_cv = metric(
        traditional,
        "enhanced_geometry_gradient_boosting",
        "stratified_6fold",
    )
    rf_cross = cross_subject_average(
        traditional, "enhanced_geometry_random_forest"
    )
    ai_svm_cv = metric(ai, "rtmpose_keypoints_svm", "stratified_6fold")
    ai_svm_cross = cross_subject_average(ai, "rtmpose_keypoints_svm")

    doc = Document()
    configure_document(doc)

    doc.add_paragraph(
        "Hand Pose Classification Method Comparison",
        style="Title",
    )
    subtitle = doc.add_paragraph(
        "Traditional image processing and RTMPose Hands on the 108-image ELG5163 dataset"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.LEFT
    subtitle.runs[0].font.size = Pt(11)
    subtitle.runs[0].font.color.rgb = RGBColor.from_string(GRAY)
    subtitle.paragraph_format.space_after = Pt(12)

    intro = doc.add_paragraph()
    intro.add_run("Main conclusion. ").bold = True
    intro.add_run(
        f"The improved traditional pipeline raised the strongest ordinary "
        f"six-fold result from {percentage(baseline_cv)} to {percentage(hgb_cv)}. "
        f"Random Forest was the strongest traditional choice for cross-subject "
        f"testing at {percentage(rf_cross)}. The proposal-aligned RTMPose keypoint "
        f"method performed substantially better, reaching {percentage(ai_svm_cv)} "
        f"in stratified six-fold validation and {percentage(ai_svm_cross)} on "
        f"average when training on one participant and testing on the other."
    )

    doc.add_heading("Dataset and evaluation", level=1)
    add_table(
        doc,
        ["Item", "Value"],
        [
            ["Images", "108 total, 54 left-hand and 54 right-hand"],
            ["Participants", "2 participants, 54 images each"],
            ["Classes", "Abduction, adduction and opposition"],
            ["Views", "Front, 45 degrees and 90 degrees"],
            ["Occlusion", "None, paper and another hand"],
            ["Primary evaluation", "Stratified 6-fold and cross-subject testing"],
        ],
        [1.7, 5.1],
    )
    paragraph = doc.add_paragraph()
    paragraph.add_run("Why cross-subject testing matters. ").bold = True
    paragraph.add_run(
        "Random image splits may place visually similar images of the same person "
        "in both training and test folds. Training on one participant and testing "
        "on the other gives a stricter estimate of generalization to a new hand."
    )

    doc.add_page_break()
    doc.add_heading("Traditional method", level=1)
    doc.add_paragraph(
        "The baseline follows the proposal: YCrCb and HSV skin segmentation, "
        "morphological cleanup, contour selection, convex hull measurements, "
        "Hu moments and an RBF SVM. The enhanced experiment kept the same masks "
        "and added the following compact improvements."
    )
    add_bullets(
        doc,
        [
            "Mirrored right hands into a common orientation and normalized every silhouette to a 128 by 128 canvas.",
            "Added convexity-defect depth, palm-radius, skeleton length, endpoint and branch counts, and width profiles.",
            "Tested tuned SVM, Random Forest, gradient boosting and HOG features.",
            "Used both ordinary six-fold validation and participant-to-participant testing.",
        ],
    )

    traditional_methods = [
        ("Baseline SVM", "baseline_fixed_svm"),
        ("Enhanced SVM", "enhanced_geometry_svm"),
        ("Enhanced Random Forest", "enhanced_geometry_random_forest"),
        ("Enhanced Gradient Boosting", "enhanced_geometry_gradient_boosting"),
        ("Enhanced HOG SVM", "enhanced_geometry_hog_svm"),
    ]
    traditional_rows = []
    for label, method in traditional_methods:
        traditional_rows.append(
            [
                label,
                percentage(metric(traditional, method, "stratified_6fold")),
                percentage(cross_subject_average(traditional, method)),
                percentage(metric(traditional, method, "holdout_90deg")),
            ]
        )
    add_table(
        doc,
        ["Traditional method", "6-fold", "Cross-subject", "90 degree holdout"],
        traditional_rows,
        [2.7, 1.25, 1.45, 1.45],
        font_size=9,
    )
    paragraph = doc.add_paragraph()
    paragraph.add_run("Interpretation. ").bold = True
    paragraph.add_run(
        "Gradient boosting gave the best ordinary six-fold score, while Random "
        "Forest generalized best across participants and improved the difficult "
        "90-degree condition from 30.56% to 44.44%. HOG reduced the six-fold score "
        "to 36.11%, indicating overfitting on this small dataset."
    )

    doc.add_page_break()
    doc.add_heading("AI keypoint method", level=1)
    doc.add_paragraph(
        "The AI workflow follows the proposal's RTMPose Hands benchmark. An "
        "official pre-trained RTMDet hand detector and RTMPose-m Hand5 model were "
        "run through ONNX Runtime on CPU. The model produced 21 hand keypoints for "
        "all 108 images; the RTMPose network itself was not fine-tuned."
    )
    add_bullets(
        doc,
        [
            "Selected the upright target hand when more than one hand instance was detected.",
            "Translated keypoints to the wrist, rotated them to a common palm direction, normalized by palm size and mirrored right hands.",
            "Computed normalized coordinates, confidence values, thumb joint angles and thumb-to-fingertip distances.",
            "Compared SVM, Random Forest, gradient boosting and a shallow decision tree.",
        ],
    )
    ai_methods = [
        ("RTMPose SVM", "rtmpose_keypoints_svm"),
        ("RTMPose Random Forest", "rtmpose_keypoints_random_forest"),
        ("RTMPose Gradient Boosting", "rtmpose_keypoints_gradient_boosting"),
        ("RTMPose Decision Tree", "rtmpose_keypoints_decision_tree"),
    ]
    ai_rows = []
    for label, method in ai_methods:
        ai_rows.append(
            [
                label,
                percentage(metric(ai, method, "stratified_6fold")),
                percentage(cross_subject_average(ai, method)),
                percentage(metric(ai, method, "holdout_90deg")),
            ]
        )
    add_table(
        doc,
        ["AI method", "6-fold", "Cross-subject", "90 degree holdout"],
        ai_rows,
        [2.7, 1.25, 1.45, 1.45],
        font_size=9,
    )

    picture = doc.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(OVERLAY_PATH), width=Inches(2.7))
    caption = doc.add_paragraph(
        "Example RTMPose output showing the selected 21 hand keypoints"
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(9)
    caption.paragraph_format.space_after = Pt(2)

    doc.add_page_break()
    doc.add_heading("Method comparison and interpretation", level=1)
    chart_paragraph = doc.add_paragraph()
    chart_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    chart_paragraph.add_run().add_picture(str(CHART_PATH), width=Inches(6.65))

    confidence_by_view = ai_features.groupby("view")["mean_score"].mean()
    confidence_by_occlusion = ai_features.groupby("occlusion")["mean_score"].mean()
    add_table(
        doc,
        ["Condition", "Mean RTMPose keypoint confidence"],
        [
            ["Front", f"{confidence_by_view['front']:.3f}"],
            ["45 degrees", f"{confidence_by_view['45deg']:.3f}"],
            ["90 degrees", f"{confidence_by_view['90deg']:.3f}"],
            ["No occlusion", f"{confidence_by_occlusion['none']:.3f}"],
            ["Paper occlusion", f"{confidence_by_occlusion['paper']:.3f}"],
            ["Hand occlusion", f"{confidence_by_occlusion['hand']:.3f}"],
        ],
        [3.7, 3.1],
    )
    paragraph = doc.add_paragraph()
    paragraph.add_run("What the comparison shows. ").bold = True
    paragraph.add_run(
        "RTMPose keypoints separate the three thumb poses much better than global "
        "silhouette statistics. The remaining weakness is the side view: its mean "
        "keypoint confidence is 0.473, compared with 0.762 for the front view. "
        "When another hand overlaps the target, RTMPose can also place some joints "
        "on the assisting hand. Therefore, the AI numbers are promising but should "
        "not be presented as a finished clinical measurement system."
    )

    doc.add_heading("Limitations and next steps", level=2)
    add_bullets(
        doc,
        [
            "Only two participants are available, so accuracy estimates remain sensitive to individual images.",
            "The project currently uses still images; the proposal's Lucas-Kanade tracking and temporal voting remain future video work.",
            "Manual keypoint ground truth is not yet available, so keypoint quality was checked visually rather than with pixel error.",
            "The next data collection should add participants and more 90-degree and occluded examples before further model tuning.",
        ],
    )

    doc.add_page_break()
    doc.add_heading("Suggested presentation structure", level=1)
    add_table(
        doc,
        ["Slide", "Main content"],
        [
            ["1", "Problem, three thumb poses and project objective"],
            ["2", "Dataset design: two participants, views and occlusion conditions"],
            ["3", "Traditional segmentation and geometric feature pipeline"],
            ["4", "Traditional improvements and cross-subject result"],
            ["5", "RTMPose detector, 21 keypoints and normalized keypoint features"],
            ["6", "Main comparison chart and strongest quantitative results"],
            ["7", "Failure cases: 90-degree view and overlapping hands"],
            ["8", "Conclusion and next steps for video and more participants"],
        ],
        [0.75, 6.05],
    )
    doc.add_heading("Presentation takeaway", level=2)
    doc.add_paragraph(
        "The traditional pipeline is lightweight and interpretable, and the enhanced "
        "Random Forest improves cross-person robustness. RTMPose provides a much "
        "stronger representation for fine thumb motion, but side views and hand-on-hand "
        "occlusion remain difficult. The comparison supports the proposal's central "
        "claim that learned keypoints are more robust than silhouette-only features, "
        "while also showing why the traditional method remains useful as an explainable baseline."
    )

    doc.add_heading("Reproducible outputs", level=2)
    doc.add_paragraph(
        "Traditional results are stored in traditional_method/results, and RTMPose "
        "features, predictions and overlays are stored in ai_method/results. The "
        "scripts enhanced_traditional.py and run_rtmpose_comparison.py reproduce the experiments."
    )
    doc.add_heading("References", level=2)
    references = [
        "[1] ELG5163 project proposal, Comparing Skeleton-Based and Machine Learning Methods for Fine-Grained Hand Pose Classification.",
        "[2] OpenMMLab MMPose, 2D Hand Keypoint Demo and RTMPose Hands model documentation.",
        "[3] T. Jiang et al., RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose, 2023.",
    ]
    for reference in references:
        paragraph = doc.add_paragraph(reference)
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.first_line_indent = Inches(-0.18)
        paragraph.paragraph_format.space_after = Pt(3)

    doc.core_properties.title = "Hand Pose Classification Method Comparison"
    doc.core_properties.subject = "ELG5163 traditional and RTMPose experiment summary"
    doc.core_properties.author = "Mengyuan Wang"
    doc.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_document()
