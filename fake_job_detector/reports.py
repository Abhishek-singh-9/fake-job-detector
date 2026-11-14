"""
Audit Report Generation Module for Fake Job Posting Detector.

Generates structured downloadable PDF audit reports containing risk score,
risk level breakdown, warning signs, and keyword analysis using ReportLab.

Key Architecture & Viva Points:
- In-Memory Document Assembly:
  Uses `io.BytesIO` to construct the PDF binary stream entirely in memory.
  No temporary files are created or left on disk, ensuring clean execution
  and avoiding race conditions or file permission issues in multi-user settings.
- Security & Injection Defense:
  ReportLab Paragraphs interpret XML-like tags. Any untrusted text (e.g. job titles,
  job descriptions, user inputs) is strictly escaped with `xml.sax.saxutils.escape`.
  Characters outside Latin-1 are replaced with '?' to prevent Type 1 font encoder
  crashes (`UnicodeEncodeError`).
- Privacy & Role Boundaries:
  Passwords and sensitive user metadata are never exposed or rendered into reports.
  Users may only download reports for their own evaluations.
- Resilient Schema Handling:
  Gracefully handles checks with missing fields (e.g. missing text, empty warnings,
  or omitted explainability word weights) without raising exceptions.
"""

from datetime import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.sax.saxutils as saxutils

from predictor import get_risk_label, label_for_level

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def sanitize_text(text: Any) -> str:
    """
    Sanitizes arbitrary user input for safe rendering inside ReportLab Paragraphs.

    Why this matters for viva:
    1. Built-in Helvetica/Times fonts in ReportLab use Latin-1 (WinAnsi) encoding.
       Characters from other scripts (Cyrillic, Devanagari, Chinese, emoji) raise
       a UnicodeEncodeError unless replaced with a safe fallback ('?').
    2. Paragraph objects parse XML markup. Escaping XML entities (&, <, >) prevents
       malformed tags or injection attempts (e.g. <script>) from breaking the layout.
    """
    if text is None:
        return ""
    raw = str(text)
    # Replace non-Latin-1 characters with '?'
    safe_chars = raw.encode("latin-1", errors="replace").decode("latin-1")
    # Escape XML markup characters
    return saxutils.escape(safe_chars)


class ReportGenerator:
    """
    Builds professional PDF audit reports from job check dictionaries.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the generator with optional model metadata.

        Args:
            metadata: Optional dictionary loaded from models/metadata.json.
        """
        self.metadata = metadata or {}

    @staticmethod
    def build_pdf(check: Dict[str, Any], owner_username: str) -> bytes:
        """
        Generates a PDF report in memory for a given scan check.

        Args:
            check: Dictionary containing evaluation results.
            owner_username: Username of the evaluation owner.

        Returns:
            bytes: Binary content of the generated PDF document.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#1e3d59"),
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#555555"),
        )
        h2_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#1e3d59"),
            spaceBefore=8,
            spaceAfter=4,
        )
        cell_bold = ParagraphStyle(
            "CellBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#222222"),
        )
        cell_norm = ParagraphStyle(
            "CellNorm",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#333333"),
        )
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#4a5568"),
        )
        body_style = ParagraphStyle(
            "JobBodyText",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#2d3748"),
        )
        note_style = ParagraphStyle(
            "TruncNote",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#e63946"),
        )

        story = []

        # ---------------------------------------------------------------------
        # 1. Document Header
        # ---------------------------------------------------------------------
        story.append(Paragraph("Fake Job Posting Detector — Evaluation Report", title_style))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Decision Support & Heuristic Audit Log | Machine Learning & Pattern Analysis",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 6))
        story.append(
            HRFlowable(
                width="100%",
                thickness=1.5,
                color=colors.HexColor("#1e3d59"),
                spaceAfter=10,
            )
        )

        # ---------------------------------------------------------------------
        # 2. Key-Value Evaluation Summary
        # ---------------------------------------------------------------------
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_date = check.get("created_at") or now_str
        job_title = check.get("job_title") or "Untitled Position"
        model_name = check.get("model_name") or "Logistic Regression"
        raw_level = str(check.get("risk_level", "Unknown")).upper()

        # Risk score calculation and display
        prob_val = check.get("fraud_probability")
        if prob_val is not None:
            try:
                score_pct = int(round(float(prob_val) * 100))
                score_text = f"{score_pct}% ({float(prob_val):.4f})"
            except (ValueError, TypeError):
                score_text = "N/A"
        else:
            score_text = "N/A"

        # Risk level badge styling
        if raw_level == "LOW":
            level_color = "#2a9d8f"
        elif raw_level == "MEDIUM":
            level_color = "#f4a261"
        elif raw_level == "HIGH":
            level_color = "#e63946"
        else:
            level_color = "#555555"

        level_display = f'<font color="{level_color}"><b>{raw_level}</b></font>'

        # Classification label resolution (aligned with risk level in predictor.py)
        # Always derive dynamically from risk level to ensure consistent 3-tier classification:
        # LOW    -> "Likely genuine"
        # MEDIUM -> "Borderline - review carefully"
        # HIGH   -> "Potentially fraudulent"
        label_text = label_for_level(raw_level)

        # Cutoffs resolution (from check or metadata fallback)
        cutoffs = check.get("risk_cutoffs")
        if not cutoffs:
            try:
                meta_file = Path(__file__).resolve().parent / "models" / "metadata.json"
                if meta_file.is_file():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                        cutoffs = meta_data.get("risk_cutoffs", {})
            except Exception:
                cutoffs = {}

        low_max = float(cutoffs.get("low_max", 0.40)) if cutoffs else 0.40
        high_min = float(cutoffs.get("high_min", 0.50)) if cutoffs else 0.50
        low_pct = int(round(low_max * 100))
        high_pct = int(round(high_min * 100))
        med_max_pct = high_pct - 1
        cutoffs_str = f"Low &lt; {low_pct}%, Medium {low_pct}%–{med_max_pct}%, High &ge; {high_pct}%"

        summary_data = [
            [
                Paragraph("Job Title:", cell_bold),
                Paragraph(sanitize_text(job_title), cell_norm),
            ],
            [
                Paragraph("Evaluated By (Owner):", cell_bold),
                Paragraph(sanitize_text(owner_username or "Unknown"), cell_norm),
            ],
            [
                Paragraph("Evaluation Date:", cell_bold),
                Paragraph(sanitize_text(check_date), cell_norm),
            ],
            [
                Paragraph("Report Generated At:", cell_bold),
                Paragraph(sanitize_text(now_str), cell_norm),
            ],
            [
                Paragraph("Production Classifier:", cell_bold),
                Paragraph(sanitize_text(model_name), cell_norm),
            ],
            [
                Paragraph("Estimated Risk Score:", cell_bold),
                Paragraph(score_text, cell_norm),
            ],
            [
                Paragraph("Assigned Risk Level:", cell_bold),
                Paragraph(level_display, cell_norm),
            ],
            [
                Paragraph("Classification Output:", cell_bold),
                Paragraph(sanitize_text(label_text), cell_norm),
            ],
            [
                Paragraph("Cutoffs Used:", cell_bold),
                Paragraph(cutoffs_str, cell_norm),
            ],
        ]

        summary_table = Table(summary_data, colWidths=[140, 392])
        summary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(summary_table)
        story.append(Spacer(1, 8))

        # ---------------------------------------------------------------------
        # 3. Mandatory Advisory Disclaimer
        # ---------------------------------------------------------------------
        disclaimer_text = (
            "<b>Disclaimer:</b> This score is a model estimate of how similar this ad is "
            "to known fake ads. It is not proof, and it is not a true probability."
        )
        disc_table = Table(
            [[Paragraph(disclaimer_text, disclaimer_style)]],
            colWidths=[532],
        )
        disc_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf2f7")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(disc_table)
        story.append(Spacer(1, 8))

        # ---------------------------------------------------------------------
        # 4. Heuristic Warning Signs
        # ---------------------------------------------------------------------
        raw_warnings = check.get("warnings")
        if isinstance(raw_warnings, str):
            try:
                raw_warnings = json.loads(raw_warnings)
            except Exception:
                raw_warnings = []

        story.append(Paragraph("Heuristic Warning Signs", h2_style))

        if raw_warnings and isinstance(raw_warnings, list) and len(raw_warnings) > 0:
            warn_rows = [
                [
                    Paragraph("<b>Severity</b>", cell_bold),
                    Paragraph("<b>Warning Sign Description</b>", cell_bold),
                    Paragraph("<b>Detected Phrases</b>", cell_bold),
                ]
            ]
            for w in raw_warnings:
                if isinstance(w, dict):
                    sev = str(w.get("severity", "low")).upper()
                    if sev == "HIGH":
                        sev_disp = '<font color="#e63946"><b>HIGH</b></font>'
                    elif sev == "MEDIUM":
                        sev_disp = '<font color="#f4a261"><b>MEDIUM</b></font>'
                    else:
                        sev_disp = '<font color="#2a9d8f"><b>LOW</b></font>'

                    msg = sanitize_text(w.get("message", "Rule triggered"))
                    matched = w.get("matched", [])
                    matched_str = sanitize_text(", ".join(matched) if matched else "N/A")

                    warn_rows.append([
                        Paragraph(sev_disp, cell_norm),
                        Paragraph(msg, cell_norm),
                        Paragraph(matched_str, cell_norm),
                    ])

            warn_table = Table(warn_rows, colWidths=[70, 292, 170])
            warn_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(warn_table)
        else:
            story.append(
                Paragraph(
                    "No heuristic warning signs were triggered. The ad passed all rule-based pattern checks.",
                    cell_norm,
                )
            )

        story.append(Spacer(1, 8))

        # ---------------------------------------------------------------------
        # 5. Top Influential Words (Local Feature Attribution)
        # ---------------------------------------------------------------------
        top_words = check.get("top_words")
        if isinstance(top_words, str):
            try:
                top_words = json.loads(top_words)
            except Exception:
                top_words = None

        if top_words and isinstance(top_words, dict):
            fraud_words = top_words.get("fraud", [])
            genuine_words = top_words.get("genuine", [])

            if fraud_words or genuine_words:
                story.append(Paragraph("Influential Words (Feature Attribution)", h2_style))

                words_data = [
                    [
                        Paragraph("<b>Pushing Toward Fraud (+)</b>", cell_bold),
                        Paragraph("<b>Pushing Toward Genuine (-)</b>", cell_bold),
                    ]
                ]

                max_len = max(len(fraud_words), len(genuine_words))
                for i in range(max_len):
                    fw_cell = ""
                    if i < len(fraud_words):
                        fw_item = fraud_words[i]
                        w_txt = sanitize_text(fw_item.get("word", ""))
                        c_val = fw_item.get("contribution", 0.0)
                        fw_cell = f"{w_txt} (+{c_val:.4f})"

                    gw_cell = ""
                    if i < len(genuine_words):
                        gw_item = genuine_words[i]
                        w_txt = sanitize_text(gw_item.get("word", ""))
                        c_val = gw_item.get("contribution", 0.0)
                        gw_cell = f"{w_txt} ({c_val:.4f})"

                    words_data.append([
                        Paragraph(fw_cell, cell_norm),
                        Paragraph(gw_cell, cell_norm),
                    ])

                words_table = Table(words_data, colWidths=[266, 266])
                words_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ])
                )
                story.append(words_table)
                story.append(Spacer(1, 8))

        # ---------------------------------------------------------------------
        # 6. Pasted Job Advertisement Text (Truncated to first 1500 chars)
        # ---------------------------------------------------------------------
        job_text_raw = check.get("job_text")
        if job_text_raw is not None and str(job_text_raw).strip():
            story.append(Paragraph("Analyzed Job Advertisement Content", h2_style))
            full_text = str(job_text_raw)
            is_trunc = len(full_text) > 1500
            display_text = full_text[:1500]

            if is_trunc:
                story.append(
                    Paragraph(
                        f"[Notice: Content truncated. Displaying first 1,500 of {len(full_text):,} characters.]",
                        note_style,
                    )
                )
                story.append(Spacer(1, 4))

            safe_ad_html = sanitize_text(display_text).replace("\n", "<br/>")
            text_box = Table(
                [[Paragraph(safe_ad_html, body_style)]],
                colWidths=[532],
            )
            text_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ])
            )
            story.append(text_box)

        # Build document in memory
        doc.build(story)
        return buffer.getvalue()


# -----------------------------------------------------------------------------
# Module Self-Test (Task 3)
# -----------------------------------------------------------------------------

def run_self_tests() -> bool:
    """
    Executes automated verification tests for PDF generation, security escaping,
    and schema resilience.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    generator = ReportGenerator()
    all_passed = True

    # Test 1: Nasty input and non-Latin characters
    print("\n--- TEST 1: Nasty Input & Non-Latin Character Handling ---")
    nasty_check = {
        "id": 101,
        "job_title": "Senior Engineer <b>Urgent</b> & <script>alert('xss')</script>",
        "created_at": "2026-11-14 10:30:00",
        "fraud_probability": 0.88,
        "risk_level": "high",
        "model_name": "Logistic Regression",
        "job_text": (
            "<b>bold</b> & <script>x</script> and a long line with special & "
            "non-Latin characters: Привет, नमस्ते, 日本語, 🛡️, 🚨, and "
            "quotes \"test\" and 'apos' across multiple paragraphs.\n\n"
            "Payment requires registration fee of $150 via Western Union."
        ),
        "warnings": [
            {
                "id": "fee",
                "severity": "high",
                "message": "Fee requested <script>injection</script>",
                "matched": ["registration fee", "Western Union"],
            },
            {
                "id": "urgency",
                "severity": "low",
                "message": "Urgent recruitment",
                "matched": ["urgent"],
            },
        ],
        "top_words": {
            "fraud": [{"word": "fee", "contribution": 0.42}],
            "genuine": [{"word": "engineer", "contribution": -0.31}],
        },
        "risk_cutoffs": {"low_max": 0.40, "high_min": 0.50},
    }

    try:
        pdf_bytes_1 = generator.build_pdf(nasty_check, owner_username="student_tester")
        has_header_1 = pdf_bytes_1.startswith(b"%PDF")
        is_large_1 = len(pdf_bytes_1) > 1000

        if has_header_1 and is_large_1:
            print(f"PASS: Nasty input PDF generated successfully ({len(pdf_bytes_1):,} bytes, starts with %PDF).")
        else:
            print(f"FAIL: Nasty input PDF validation failed (header={has_header_1}, size={len(pdf_bytes_1)}).")
            all_passed = False
    except Exception as exc:
        print(f"FAIL: Exception during Test 1: {exc}")
        all_passed = False

    # Test 2: Missing optional fields
    print("\n--- TEST 2: Missing Optional Fields Handling ---")
    minimal_check = {
        "id": 102,
        "fraud_probability": 0.15,
        "risk_level": "low",
        # Missing: job_title, job_text, warnings, top_words, model_name, created_at, risk_cutoffs
    }

    try:
        pdf_bytes_2 = generator.build_pdf(minimal_check, owner_username="minimal_user")
        has_header_2 = pdf_bytes_2.startswith(b"%PDF")
        is_large_2 = len(pdf_bytes_2) > 1000

        if has_header_2 and is_large_2:
            print(f"PASS: Minimal check PDF generated successfully ({len(pdf_bytes_2):,} bytes, starts with %PDF).")
        else:
            print(f"FAIL: Minimal check PDF validation failed (header={has_header_2}, size={len(pdf_bytes_2)}).")
            all_passed = False
    except Exception as exc:
        print(f"FAIL: Exception during Test 2: {exc}")
        all_passed = False

    # Test 3: Long job text truncation (> 1500 chars)
    print("\n--- TEST 3: Long Job Text Truncation Handling ---")
    long_text = "This is a detailed job posting description sentence. " * 50  # ~2650 chars
    long_check = {
        "id": 103,
        "job_title": "Full Stack Lead",
        "job_text": long_text,
        "fraud_probability": 0.45,
        "risk_level": "medium",
        "warnings": [],
    }

    try:
        pdf_bytes_3 = generator.build_pdf(long_check, owner_username="analyst")
        has_header_3 = pdf_bytes_3.startswith(b"%PDF")
        is_large_3 = len(pdf_bytes_3) > 1000

        if has_header_3 and is_large_3:
            print(f"PASS: Long text check PDF generated successfully ({len(pdf_bytes_3):,} bytes, truncated safely).")
        else:
            print(f"FAIL: Long text check PDF validation failed (header={has_header_3}, size={len(pdf_bytes_3)}).")
            all_passed = False
    except Exception as exc:
        print(f"FAIL: Exception during Test 3: {exc}")
        all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_self_tests()
    if success:
        print("\nAll ReportGenerator self-tests PASSED!")
    else:
        print("\nSome ReportGenerator self-tests FAILED!")
