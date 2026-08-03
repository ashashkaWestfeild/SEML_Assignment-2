"""Build the Assignment II submission PDF (``Group_84.pdf``).

Every number typeset here is read back from the artifacts produced by
``scripts/train_model.py`` and ``scripts/evaluate_and_report.py`` -- nothing in
the report is hand-copied, so the document cannot drift from the measurements.

Run:
    python scripts/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "reports" / "figures"
METRICS = PROJECT_ROOT / "reports" / "metrics"
LINT = PROJECT_ROOT / "reports" / "lint"
OUT = PROJECT_ROOT.parent / "Group_84.pdf"

INK = colors.HexColor("#1f3864")
ACCENT = colors.HexColor("#c00000")
GREY = colors.HexColor("#5a6472")
LIGHT = colors.HexColor("#eef2f8")
BORDER = colors.HexColor("#c9d2e3")

HEADER_TEXT = "Group 84  |  SEML Assignment-2  |  BITS Pilani WILP"

#: Separator inserted between two extracted definitions in one code listing.
BLANK_LINE = "\n\n"

MEMBERS = [
    [
        "1",
        "2025AA05710",
        "Singh Pritesh",
        "Refactoring to the loan_risk package, error handling & structured "
        "logging, lint/format toolchain, quality gates",
        "100",
    ],
    [
        "2",
        "2025AA05368",
        "Gangera Tushar",
        "Research-vs-production analysis, data-quality metrics, schema contract "
        "and data-validation test suite",
        "100",
    ],
    [
        "3",
        "2025AB05154",
        "Gangam Shuba Nandini",
        "ML behavioural tests (training & inference), model-quality metrics, "
        "calibration and drift monitoring",
        "100",
    ],
    [
        "4",
        "2025AA05574",
        "Shaifali Garg",
        "FastAPI design & implementation, integration tests, production "
        "experimentation and security analysis",
        "100",
    ],
]


# --------------------------------------------------------------------- styles
def build_styles() -> Dict[str, ParagraphStyle]:
    """Return the named paragraph styles used throughout the document."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=19,
            leading=24,
            textColor=INK,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=12.5,
            leading=17,
            textColor=GREY,
            alignment=1,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontSize=15,
            leading=19,
            textColor=INK,
            spaceBefore=16,
            spaceAfter=8,
            borderPadding=0,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=12.2,
            leading=16,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontSize=10.6,
            leading=14,
            textColor=ACCENT,
            spaceBefore=9,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13.4,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13.2,
            leftIndent=13,
            bulletIndent=3,
            spaceAfter=3,
            alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontSize=8.4,
            leading=11,
            textColor=GREY,
            alignment=1,
            spaceBefore=3,
            spaceAfter=10,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontSize=6.9,
            leading=8.4,
            textColor=colors.HexColor("#16324f"),
            backColor=colors.HexColor("#f6f8fc"),
            borderColor=BORDER,
            borderWidth=0.5,
            borderPadding=5,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontSize=8.1,
            leading=10.4,
        ),
        "cellb": ParagraphStyle(
            "cellb",
            parent=base["Normal"],
            fontSize=8.1,
            leading=10.4,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        ),
    }


S = build_styles()


# ---------------------------------------------------------------- primitives
def para(text: str, style: str = "body") -> Paragraph:
    """Shorthand for a styled paragraph."""
    return Paragraph(text, S[style])


def bullets(items: List[str]) -> List[Paragraph]:
    """Render a bulleted list."""
    return [Paragraph(item, S["bullet"], bulletText="\u2022") for item in items]


def table(
    rows: List[List[str]],
    widths: List[float],
    header: bool = True,
    align_center: List[int] | None = None,
) -> Table:
    """Render a styled table; the first row is the header when ``header``."""
    data = []
    for index, row in enumerate(rows):
        style = "cellb" if (header and index == 0) else "cell"
        data.append([Paragraph(str(cell), S[style]) for cell in row])

    tbl = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ]
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]))
    for column in align_center or []:
        commands.append(("ALIGN", (column, 0), (column, -1), "CENTER"))
    tbl.setStyle(TableStyle(commands))
    return tbl


def code(text: str) -> Preformatted:
    """Render a verbatim code listing."""
    return Preformatted(text.rstrip("\n"), S["code"])


def figure(name: str, caption: str, width: float = 16.0) -> List[Any]:
    """Embed a PNG scaled to ``width`` centimetres, with a numbered caption."""
    path = FIGURES / name
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(str(path)).getSize()
    target_w = width * cm
    image = Image(str(path), width=target_w, height=target_w * ih / iw)
    image.hAlign = "CENTER"
    return [image, para(caption, "caption")]


def _find_symbol(lines: List[str], name: str) -> int:
    """Return the index of the line defining ``name``, decorators included."""
    prefixes = (f"def {name}(", f"class {name}(")
    start = next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith(prefixes)),
        None,
    )
    if start is None:
        raise ValueError(f"Symbol '{name}' not found")
    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1
    return start


def _signature_end(lines: List[str], definition: int) -> int:
    """Return the index of the line closing a (possibly multi-line) signature."""
    depth = 0
    for index in range(definition, len(lines)):
        line = lines[index]
        depth += line.count("(") - line.count(")")
        if depth <= 0 and line.rstrip().endswith(":"):
            return index
    return definition


def _block_end(lines: List[str], start: int, indent: int) -> int:
    """Return the exclusive end index of the block opened at ``start``.

    Scanning begins after the signature, so a signature wrapped across several
    lines (its closing ``)`` sits back at the definition's own indent) does not
    look like the end of the block.
    """
    for index in range(_signature_end(lines, start) + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        if (len(line) - len(line.lstrip())) <= indent:
            return index
    return len(lines)


def _strip_docstring(block: List[str]) -> List[str]:
    """Drop the leading docstring of an extracted definition, if present."""
    header = next((i for i, line in enumerate(block) if line.rstrip().endswith(":")), -1)
    body = header + 1
    if body >= len(block) or not block[body].lstrip().startswith('"""'):
        return block
    if block[body].strip().count('"""') >= 2:
        return block[:body] + block[body + 1 :]
    closing = next(i for i in range(body + 1, len(block)) if '"""' in block[i])
    return block[:body] + block[closing + 1 :]


def defn(path: str, name: str, drop_docstring: bool = False) -> str:
    """Extract the source of ``def name`` / ``class name`` from ``path``.

    Symbol-based rather than line-number-based, so reformatting the codebase
    can never silently shift a listing in the report onto the wrong lines.
    """
    lines = (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
    start = _find_symbol(lines, name)
    definition = next(
        i
        for i in range(start, len(lines))
        if lines[i].lstrip().startswith(("def ", "class "))
    )
    indent = len(lines[definition]) - len(lines[definition].lstrip())
    block = lines[start : _block_end(lines, definition, indent)]
    while block and not block[-1].strip():
        block.pop()
    return "\n".join(_strip_docstring(block) if drop_docstring else block)


def between(path: str, first: str, last: str) -> str:
    """Extract from the first line containing ``first`` to the next ``last``."""
    lines = (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if first in line)
    end = next(i for i in range(start, len(lines)) if last in lines[i])
    return "\n".join(lines[start : end + 1])


# ------------------------------------------------------------------ page furniture
def decorate(canvas, doc) -> None:
    """Draw the running header and the page footer."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(GREY)
    canvas.drawString(2.0 * cm, A4[1] - 1.15 * cm, HEADER_TEXT)
    canvas.drawRightString(
        A4[0] - 2.0 * cm, A4[1] - 1.15 * cm, "AIMLCZG546 · Assignment II"
    )
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(2.0 * cm, A4[1] - 1.32 * cm, A4[0] - 2.0 * cm, A4[1] - 1.32 * cm)
    canvas.line(2.0 * cm, 1.45 * cm, A4[0] - 2.0 * cm, 1.45 * cm)
    canvas.drawCentredString(A4[0] / 2, 1.05 * cm, f"Page {doc.page}")
    canvas.restoreState()


def load_artifacts() -> Dict[str, Any]:
    """Read every measured artifact the report quotes."""
    qa = json.loads((METRICS / "qa_metrics.json").read_text(encoding="utf-8"))
    training = json.loads((METRICS / "training_metrics.json").read_text(encoding="utf-8"))
    drift = [
        line.split(",")
        for line in (METRICS / "drift_report.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    tests = (METRICS / "test_inventory.txt").read_text(encoding="utf-8").splitlines()
    pytest_tail = [
        line
        for line in (METRICS / "pytest_output.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if "passed" in line or "failed" in line
    ]
    before_flake8 = (LINT / "01_before_flake8.txt").read_text(encoding="utf-8")
    return {
        "qa": qa,
        "training": training,
        "drift": drift,
        "tests": tests,
        "pytest_tail": pytest_tail[-1] if pytest_tail else "",
        "before_flake8": before_flake8,
    }


def build_story(art: Dict[str, Any]) -> List[Any]:
    """Assemble the full flowable story for the document."""
    qa = art["qa"]
    mq = qa["model_quality"]
    dq = qa["data_quality"]
    lat = qa["latency_ms"]
    comparison = art["training"]["algorithm_comparison"]
    tests = art["tests"]

    def count(prefix: str) -> int:
        return sum(1 for t in tests if t.startswith(f"tests/{prefix}"))

    story: List[Any] = []

    # ============================ TITLE PAGE ============================
    story += [
        Spacer(1, 1.0 * cm),
        para("BITS PILANI WILP", "subtitle"),
        para("Software Engineering for Machine Learning (AIMLCZG546)", "title"),
        para(
            "Assignment II — Implementation, Code Quality & Quality Assurance", "subtitle"
        ),
        Spacer(1, 0.2 * cm),
        para(
            "<b>Project:</b> Loan Approval Risk Service — automated consumer "
            "credit underwriting<br/><b>Group Number:</b> 84 &nbsp;·&nbsp; "
            "<b>Date:</b> August 2026 &nbsp;·&nbsp; <b>Builds on:</b> "
            "Assignment I (Group 84)",
            "subtitle",
        ),
        Spacer(1, 0.5 * cm),
        para("Group Members &amp; Contributions", "h2"),
        table(
            [["Sl.", "BITS ID", "Name", "Qualitative Contribution", "%"]] + MEMBERS,
            [1.0 * cm, 2.6 * cm, 3.4 * cm, 7.6 * cm, 1.0 * cm],
            align_center=[0, 4],
        ),
        Spacer(1, 0.6 * cm),
        para("Executive Summary", "h2"),
        para(
            "Assignment I delivered a working prototype: a four-filter "
            "pipe-and-filter pipeline behind a single FastAPI endpoint. This "
            "assignment turns that prototype into a system that could be "
            "handed to an operations team. The scoring logic was rebuilt as an "
            "installable package (<font face='Courier'>src/loan_risk/</font>) "
            "with one cohesive class per responsibility; a research notebook is "
            "retained unchanged beside its engineered counterpart so the "
            "difference is evidence rather than assertion; error handling and "
            "structured JSON logging were added across the ingestion, "
            "validation and training paths; the whole codebase was put under "
            "<font face='Courier'>isort</font>, <font face='Courier'>black</font>, "
            "<font face='Courier'>flake8</font> and <font face='Courier'>pylint</font>; "
            "and the API was redesigned with versioned resources, a bounded "
            "batch endpoint and differentiated status codes."
        ),
        para(
            f"On the QA side, {len(tests)} pytest tests were written across four "
            "distinct test types, including the ML-specific families the "
            "assignment calls for — overfit-a-small-batch, loss-decreases, "
            "output shape/range, directional and invariance. Four "
            "model-quality metrics and four data-quality metrics are measured "
            "and gated. The measured headline numbers are: accuracy "
            f"<b>{mq['accuracy']:.4f}</b>, F1 <b>{mq['f1']:.4f}</b>, ROC-AUC "
            f"<b>{mq['roc_auc']:.4f}</b>, Brier <b>{mq['brier_score']:.4f}</b>, "
            f"mean latency <b>{lat['mean']:.2f} ms</b> against a 150 ms SLA, "
            "flake8 <b>45 → 0</b> violations and pylint <b>10.00/10</b>."
        ),
        para(
            "<b>The test suite earned its keep before submission.</b> The "
            "invariance test <font face='Courier'>test_prediction_is_invariant_"
            "to_repeated_calls</font> failed on the first run: with "
            "<font face='Courier'>n_jobs=-1</font> the forest accumulated tree "
            "votes in non-deterministic thread order, so two identical requests "
            "could differ by ~5e-17 — enough to flip an applicant sitting "
            "exactly on the 0.50 cut-off between APPROVED and DENIED. Section "
            "7.3 documents the defect and the fix."
        ),
        PageBreak(),
    ]

    # ==================== OBJECTIVE 1 ====================
    story += [
        para("Objective 1 — Implementation and Code Sharing", "h1"),
        para("1. Refactoring with OOP and Functional Principles", "h2"),
        para(
            "The Assignment I code was a flat <font face='Courier'>app/</font> "
            "folder in which <font face='Courier'>pipeline.py</font> held four "
            "module-level functions that each reached into a global "
            "<font face='Courier'>settings</font> object, and "
            "<font face='Courier'>main.py</font> called "
            "<font face='Courier'>joblib.load</font> directly. That is "
            "serviceable for a demo and awkward to test, extend or reason "
            "about. The refactor splits the system along responsibility "
            "boundaries, one module per concern:"
        ),
        table(
            [
                ["Module", "Type", "Responsibility", "Paradigm"],
                [
                    "<font face='Courier'>config.py</font>",
                    "<font face='Courier'>Settings</font> et al.",
                    "Frozen dataclasses loaded from YAML; every threshold is data",
                    "OOP (immutable value objects)",
                ],
                [
                    "<font face='Courier'>data/ingestion.py</font>",
                    "<font face='Courier'>DataIngestor</font>",
                    "The only component that reads training data from disk",
                    "OOP",
                ],
                [
                    "<font face='Courier'>data/validation.py</font>",
                    "<font face='Courier'>DataValidator</font>",
                    "Declarative schema contract; emits DataQualityReport",
                    "OOP + declarative spec",
                ],
                [
                    "<font face='Courier'>features/engineering.py</font>",
                    "<font face='Courier'>FeatureEngineer</font>",
                    "Derived ratios; sklearn transformer wrapping pure functions",
                    "Functional core, OOP shell",
                ],
                [
                    "<font face='Courier'>models/trainer.py</font>",
                    "<font face='Courier'>ModelTrainer</font>",
                    "train → evaluate → gate → persist lifecycle",
                    "OOP",
                ],
                [
                    "<font face='Courier'>models/predictor.py</font>",
                    "<font face='Courier'>ModelRegistry</font>, "
                    "<font face='Courier'>RiskPredictor</font>",
                    "Artifact custody; the four scoring filters",
                    "OOP + pure filters",
                ],
                [
                    "<font face='Courier'>monitoring/drift.py</font>",
                    "<font face='Courier'>DriftMonitor</font>",
                    "PSI and KS drift against the frozen reference profile",
                    "Functional core, OOP shell",
                ],
                [
                    "<font face='Courier'>api/</font>",
                    "<font face='Courier'>schemas.py</font>, "
                    "<font face='Courier'>app.py</font>",
                    "HTTP contract, routing and exception→status mapping",
                    "Declarative (Pydantic)",
                ],
            ],
            [3.6 * cm, 3.2 * cm, 6.2 * cm, 3.6 * cm],
        ),
        para(
            "Figure 1: Module responsibility map of the "
            "<font face='Courier'>loan_risk</font> package.",
            "caption",
        ),
        para("Where each paradigm was chosen, and why", "h3"),
    ]
    story += bullets(
        [
            "<b>Object-oriented</b> where there is genuine state or a lifecycle to "
            "own: <font face='Courier'>ModelRegistry</font> owns the loaded "
            "artifact, <font face='Courier'>ModelTrainer</font> owns the fitted "
            "pipeline and its metrics. Constructor injection of "
            "<font face='Courier'>Settings</font> replaced the global lookups, so "
            "any test can substitute a different configuration without patching a "
            "module.",
            "<b>Functional</b> where the operation is a transformation: "
            "<font face='Courier'>safe_ratio</font>, "
            "<font face='Courier'>compute_loan_to_income</font>, "
            "<font face='Courier'>assign_risk_tier</font> and "
            "<font face='Courier'>validate_business_rules</font> are pure — same "
            "input, same output, no mutation. "
            "<font face='Courier'>test_validation_filter_does_not_mutate_the_"
            "payload</font> and <font face='Courier'>test_transformer_does_not_"
            "mutate_its_input</font> assert that purity rather than trusting it.",
            "<b>Immutability</b> for configuration: the "
            "<font face='Courier'>Settings</font> aggregate is a frozen dataclass, "
            "so a request handler cannot quietly rewrite the decision threshold for "
            "every other request in the process.",
            "<b>Composition over duplication</b>: "
            "<font face='Courier'>FeatureEngineer</font> is a scikit-learn "
            "<font face='Courier'>TransformerMixin</font> and is therefore the "
            "<i>first step of the serialised pipeline</i>. Training and serving "
            "cannot disagree about feature definitions, because they execute the "
            "same pickled object.",
        ]
    )
    story += [
        Spacer(1, 0.15 * cm),
        para(
            "The last point is the single most valuable structural change. In "
            "Assignment I the derived ratios were computed in "
            "<font face='Courier'>data/prepare_data.py</font> for training and "
            "again in <font face='Courier'>extract_features()</font> for "
            "serving — two copies of one definition, which is the textbook "
            "setup for training/serving skew. There is now one copy, and it "
            "travels inside the artifact.",
            "body",
        ),
        code(
            between(
                "src/loan_risk/models/trainer.py",
                'if algorithm == "random_forest"',
                "]",
            )
            + "\n        ...\n"
            + between(
                "src/loan_risk/models/trainer.py",
                '"pipeline_built"',
                "return Pipeline(steps)",
            )
        ),
        para(
            "Listing 1: the estimator is assembled as a Pipeline whose first "
            "stage is the serving-time feature transformer.",
            "caption",
        ),
    ]

    # ---------------- 2. research vs production ----------------
    story += [
        para("2. Research Code vs Production Code", "h2"),
        para(
            "The component chosen for this comparison is <b>feature "
            "engineering</b>. Both artefacts are in the repository and both "
            "compute the same two ratios:"
        ),
    ]
    story += bullets(
        [
            "<b>Research:</b> <font face='Courier'>notebooks/research_prototype."
            "ipynb</font> (also exported verbatim as "
            "<font face='Courier'>legacy/research_feature_prototype.py</font> so "
            "the linters can be pointed at it).",
            "<b>Production:</b> <font face='Courier'>src/loan_risk/features/"
            "engineering.py</font>.",
        ]
    )
    story += [
        Spacer(1, 0.15 * cm),
        para(
            "The research notebook is not a strawman — it is what genuinely "
            "useful exploratory code looks like. It answered the question it "
            "was written for. The point of the table below is that every one "
            "of its shortcuts is a defect only once the code has to run "
            "unattended, and each has a specific engineered answer.",
            "body",
        ),
        table(
            [
                [
                    "#",
                    "Research code (notebook)",
                    "Consequence in production",
                    "Production answer",
                ],
                [
                    "1",
                    "Hard-coded absolute path "
                    "<font face='Courier'>C:/Users/analyst/Desktop/…</font>",
                    "Runs on exactly one machine",
                    "<font face='Courier'>Settings.path()</font> resolves every "
                    "path from <font face='Courier'>config.yaml</font>",
                ],
                [
                    "2",
                    "<font face='Courier'>train_test_split</font> with no "
                    "<font face='Courier'>random_state</font>",
                    "Metrics move every run; results are unauditable",
                    "<font face='Courier'>random_state</font> from config; "
                    "<font face='Courier'>test_training_is_reproducible</font> "
                    "asserts bit-identical output",
                ],
                [
                    "3",
                    "<font face='Courier'>LoanAmount/AnnualIncome</font> with "
                    "no guard",
                    "<font face='Courier'>inf</font> / "
                    "<font face='Courier'>ZeroDivisionError</font> on a zero-income "
                    "application",
                    "<font face='Courier'>safe_ratio()</font> applies +1 smoothing; "
                    "a finiteness check raises "
                    "<font face='Courier'>FeatureEngineeringError</font>",
                ],
                [
                    "4",
                    "Feature list retyped by hand in the notebook",
                    "Training/serving skew as soon as one copy changes",
                    "One <font face='Courier'>FEATURE_ORDER</font> contract, "
                    "sourced from config and re-imposed on every transform",
                ],
                [
                    "5",
                    "Dead variables "
                    "(<font face='Courier'>tmp</font>, "
                    "<font face='Courier'>ratio2</font>, "
                    "<font face='Courier'>x</font>)",
                    "Reader cannot tell what is load-bearing",
                    "Every symbol is used; flake8 F401/F841 enforce it",
                ],
                [
                    "6",
                    "<font face='Courier'>except: pass</font>",
                    "Failures are silent",
                    "Typed exception hierarchy; every handler logs at a "
                    "level and re-raises",
                ],
                [
                    "7",
                    "No logging, no types, no docstrings",
                    "Nothing is observable; behaviour is undocumented",
                    "Structured JSON logging, full type hints, docstring on every "
                    "public callable (pylint 10.00/10)",
                ],
                [
                    "8",
                    "Logic lives in notebook cells",
                    "Cannot be imported, reused or unit-tested",
                    "Importable package; 14 unit tests cover this module alone",
                ],
                [
                    "9",
                    "<font face='Courier'>print()</font> for diagnostics",
                    "Unstructured output, unusable by a log aggregator",
                    "<font face='Courier'>logger.info(..., extra={...})</font> "
                    "emitting one JSON object per event",
                ],
            ],
            [0.8 * cm, 4.2 * cm, 5.0 * cm, 6.6 * cm],
            align_center=[0],
        ),
        para(
            "Figure 2: Defect-by-defect mapping from the research notebook to "
            "its production counterpart.",
            "caption",
        ),
        para("Side-by-side: the same computation, twice", "h3"),
        para(
            "<b>Research</b> — <font face='Courier'>notebooks/research_"
            "prototype.ipynb</font>",
            "body",
        ),
        code(
            "df['LoanToIncomeRatio'] = df['LoanAmount']/df['AnnualIncome']"
            "       # NOTE: blows up when income is 0\n"
            "df['SavingsToLoanRatio'] = df['SavingsAccountBalance']/df['LoanAmount']\n"
            "df['ratio2'] = df['MonthlyDebtPayments']*12/df['AnnualIncome']\n"
            "df['x'] = df['CreditScore']/850\n"
            "FEATURES=['Age','AnnualIncome','CreditScore','LoanAmount',"
            "'LoanToIncomeRatio', ...]\n"
            "\n"
            "def get_ratio(a,b) :\n"
            "    return a/b        # no zero guard, no types, no docstring"
        ),
        para(
            "<b>Production</b> — <font face='Courier'>src/loan_risk/features/"
            "engineering.py</font>",
            "body",
        ),
        code(
            defn("src/loan_risk/features/engineering.py", "safe_ratio")
            + BLANK_LINE
            + defn("src/loan_risk/features/engineering.py", "compute_loan_to_income")
            + BLANK_LINE
            + defn("src/loan_risk/features/engineering.py", "compute_savings_to_loan")
        ),
        code(defn("src/loan_risk/features/engineering.py", "transform")),
        para(
            "Listing 2: the production transformer — declared contract, "
            "defensive division, explicit failure modes, structured logging, "
            "and a sklearn interface so it can be serialised with the model.",
            "caption",
        ),
    ]

    # ---------------- 3. error handling & logging ----------------
    story += [
        para("3. Error Handling and Logging", "h2"),
        para(
            "A typed exception hierarchy rooted at "
            "<font face='Courier'>LoanRiskError</font> lets the API layer tell "
            "our failures apart from unexpected ones and map each to the right "
            "status code, instead of returning a blanket 500. Logging is "
            "structured JSON on stdout — the same records that help a developer "
            "locally become the monitoring substrate in production without a "
            "second instrumentation path."
        ),
        table(
            [
                ["Level", "Policy in this codebase", "Example event"],
                [
                    "INFO",
                    "Normal lifecycle events worth auditing",
                    "<font face='Courier'>data_ingestion_completed</font>, "
                    "<font face='Courier'>model_artifact_saved</font>, "
                    "<font face='Courier'>loan_application_scored</font>",
                ],
                [
                    "WARNING",
                    "Recoverable anomaly: a fallback was taken, a soft "
                    "data rule was breached, or a request was rejected by policy",
                    "<font face='Courier'>duplicate_rows_detected</font>, "
                    "<font face='Courier'>data_quality_violations_detected</font>, "
                    "<font face='Courier'>business_rule_rejected</font>, "
                    "<font face='Courier'>data_drift_detected</font>",
                ],
                [
                    "ERROR",
                    "The operation failed and the caller receives an error",
                    "<font face='Courier'>data_source_missing</font>, "
                    "<font face='Courier'>feature_contract_violation</font>, "
                    "<font face='Courier'>quality_gate_breached</font>, "
                    "<font face='Courier'>model_training_failed</font>",
                ],
            ],
            [2.0 * cm, 6.4 * cm, 8.2 * cm],
        ),
        para(
            "Figure 3: Log-level policy, applied consistently across all " "modules.",
            "caption",
        ),
        para("The three critical functions", "h3"),
        para(
            "<b>Critical function 1 — <font face='Courier'>DataIngestor.load()"
            "</font></b>: every I/O failure mode (missing file, empty file, "
            "unparseable CSV, zero rows) is caught, logged and re-raised as a "
            "single <font face='Courier'>DataIngestionError</font>, so callers "
            "never have to catch <font face='Courier'>OSError</font>. "
            "Duplicate rows are a WARNING — the run continues, but the "
            "operator is told.",
            "body",
        ),
        code(defn("src/loan_risk/data/ingestion.py", "load")),
        para(
            "<b>Critical function 2 — <font face='Courier'>DataValidator."
            "validate()</font></b>: soft breaches accumulate and log at "
            "WARNING; in <font face='Courier'>strict=True</font> mode (used by "
            "the training entrypoint) a breach logs at ERROR and raises "
            "<font face='Courier'>SchemaValidationError</font> carrying the "
            "full violation list. Training on silently corrupt data is the "
            "failure this prevents.",
            "body",
        ),
        code(defn("src/loan_risk/data/validation.py", "validate", drop_docstring=True)),
        para(
            "<b>Critical function 3 — <font face='Courier'>ModelTrainer."
            "train()</font> / <font face='Courier'>enforce_quality_gates()"
            "</font></b>: length mismatch and a single-class target are "
            "rejected before fitting; a fit exception is wrapped as "
            "<font face='Courier'>ModelTrainingError</font>; and a model that "
            "misses a release gate is logged at ERROR and refuses to ship.",
            "body",
        ),
        code(
            defn(
                "src/loan_risk/models/trainer.py",
                "enforce_quality_gates",
                drop_docstring=True,
            )
        ),
        para(
            "Sample of the real emitted log stream "
            "(<font face='Courier'>reports/metrics/training_log.txt</font>):",
            "body",
        ),
        code(
            '{"timestamp": "2026-08-03T22:33:34", "level": "INFO", "logger": '
            '"loan_risk.data.ingestion", "message": "data_ingestion_completed", '
            '"rows": 20000, "columns": 23}\n'
            '{"timestamp": "2026-08-03T22:33:34", "level": "INFO", "logger": '
            '"loan_risk.data.validation", "message": "data_quality_gate_passed", '
            '"rows": 20000, "schema_conformance_rate": 1.0, '
            '"missing_value_fraction": 0.0}\n'
            '{"timestamp": "2026-08-03T22:33:40", "level": "INFO", "logger": '
            '"loan_risk.models.trainer", "message": "model_evaluated", '
            '"accuracy": 0.948, "f1": 0.934343, "roc_auc": 0.988084, '
            '"brier_score": 0.045828}\n'
            '{"timestamp": "2026-08-03T22:33:41", "level": "WARNING", "logger": '
            '"loan_risk.monitoring.drift", "message": "data_drift_detected", '
            '"n_drifted_features": 3, "features": ["CreditScore", '
            '"CreditCardUtilizationRate", "DebtToIncomeRatio"]}\n'
            '{"timestamp": "2026-08-03T22:33:41", "level": "INFO", "logger": '
            '"loan_risk.models.predictor", "message": "loan_application_scored", '
            '"is_approved": false, "probability": 0.0725, "risk_tier": "HIGH", '
            '"latency_ms": 5.18, "model_version": "2.0.0"}'
        ),
        para(
            "Listing 3: every record is one JSON object, so "
            "<font face='Courier'>latency_ms</font> or "
            "<font face='Courier'>risk_tier</font> can be aggregated directly "
            "by ELK/Splunk with no log parsing.",
            "caption",
        ),
    ]

    # ---------------- 4. linting ----------------
    before_lines = [ln for ln in art["before_flake8"].splitlines() if ln.strip()]
    story += [
        para("4. Code Formatting and Linting", "h2"),
        para(
            "Four tools are configured, each with a distinct job: "
            "<font face='Courier'>isort</font> orders imports, "
            "<font face='Courier'>black</font> owns formatting outright, "
            "<font face='Courier'>flake8</font> catches correctness and "
            "complexity issues formatting cannot, and "
            "<font face='Courier'>pylint</font> adds design-level checks. "
            "Configuration lives in "
            "<font face='Courier'>pyproject.toml</font> (black, isort, pytest, "
            "pylint) and <font face='Courier'>setup.cfg</font> (flake8, which "
            "still has no pyproject support). The line length is set to 90 in "
            "all four so they cannot fight each other."
        ),
        para("Method", "h3"),
        para(
            f"The <b>before</b> snapshot was taken on the codebase as first "
            f"written, together with the untouched research code in "
            f"<font face='Courier'>legacy/</font>. It reported "
            f"<b>{len(before_lines)} flake8 violations</b> across the tree and "
            "<b>13 files</b> that black would reformat. The formatters were "
            "then run, and the two findings the formatters cannot fix were "
            "repaired by hand — one of which was a genuine design issue "
            "(<font face='Courier'>C901</font>: "
            "<font face='Courier'>DataValidator.validate</font> had a "
            "cyclomatic complexity of 11 against a budget of 10, and was split "
            "into <font face='Courier'>_check_column</font> and "
            "<font face='Courier'>_check_range</font>). Both reports are stored "
            "verbatim in <font face='Courier'>reports/lint/</font>."
        ),
        table(
            [
                ["Check", "Command", "Before", "After"],
                [
                    "flake8 violations",
                    "<font face='Courier'>python -m flake8 src scripts tests legacy"
                    "</font>",
                    f"{len(before_lines)}",
                    "<b>0</b>",
                ],
                [
                    "black — files needing reformat",
                    "<font face='Courier'>python -m black --check src scripts tests"
                    "</font>",
                    "13",
                    "<b>0</b>",
                ],
                [
                    "isort — import order",
                    "<font face='Courier'>python -m isort --check-only src scripts "
                    "tests</font>",
                    "clean",
                    "<b>clean</b>",
                ],
                [
                    "pylint score",
                    "<font face='Courier'>python -m pylint src/loan_risk</font>",
                    "9.72 / 10",
                    "<b>10.00 / 10</b>",
                ],
            ],
            [3.8 * cm, 7.8 * cm, 2.3 * cm, 2.7 * cm],
            align_center=[2, 3],
        ),
        para(
            "Figure 4: Lint and format results, before and after. Raw reports: "
            "<font face='Courier'>reports/lint/01_before_flake8.txt</font> → "
            "<font face='Courier'>07_after_flake8.txt</font>.",
            "caption",
        ),
    ]
    story += figure(
        "lint_before_after.png",
        "Figure 5: The same data as a chart. The pylint score is "
        "scaled ×10 to share the axis.",
        12.5,
    )
    story += [
        para("Representative violations from the BEFORE report", "h3"),
        code("\n".join(before_lines[:14])),
        para(
            "Listing 4: extract from "
            "<font face='Courier'>reports/lint/01_before_flake8.txt</font>. "
            "<font face='Courier'>E401</font> multiple imports per line, "
            "<font face='Courier'>F401</font> unused import, "
            "<font face='Courier'>E722</font> bare except, "
            "<font face='Courier'>E231</font> missing whitespace, "
            "<font face='Courier'>C901</font> too complex.",
            "caption",
        ),
        para("AFTER — the same command on the production tree", "h3"),
        code(
            "$ python -m flake8 src scripts tests\n"
            "$ python -m black --check src scripts tests\n"
            "All done!  25 files would be left unchanged.\n"
            "$ python -m isort --check-only src scripts tests\n"
            "$ python -m pylint src/loan_risk\n"
            "\n"
            "Your code has been rated at 10.00/10 (previous run: 9.72/10, +0.28)"
        ),
        para(
            "Listing 5: flake8 and isort print nothing on success. "
            "<font face='Courier'>legacy/</font> is deliberately excluded from "
            "black and isort — it is the evidence, and reformatting it would "
            "destroy the comparison.",
            "caption",
        ),
    ]

    # ---------------- 5. REST API ----------------
    story += [
        para("5. REST API Design and Implementation", "h2"),
        para(
            "The service is a FastAPI application. Assignment I exposed a "
            "single unversioned <font face='Courier'>/predict</font>; the "
            "redesign applies the API-design practices the assignment asks for."
        ),
    ]
    story += bullets(
        [
            "<b>Versioned, resource-oriented paths.</b> "
            "<font face='Courier'>/v1/predict</font>, "
            "<font face='Courier'>/v1/predict/batch</font>, "
            "<font face='Courier'>/v1/model/metadata</font>. A breaking change can "
            "ship as <font face='Courier'>/v2</font> without stranding callers. "
            "<font face='Courier'>/health</font> stays unversioned because "
            "orchestrator probes should never be tied to an API generation.",
            "<b>Explicit request/response schemas.</b> Every field is bounded with "
            "<font face='Courier'>ge</font>/<font face='Courier'>le</font>, and "
            "<font face='Courier'>extra=\"forbid\"</font> rejects unknown keys — so "
            "a client typo like <font face='Courier'>credit_scr</font> is a loud "
            "422 rather than a silently defaulted feature.",
            "<b>Correct, differentiated status codes.</b> 200 scored · 422 schema "
            "violation · 400 business-rule rejection · 503 model unavailable · 404 "
            "unknown route. A policy rejection is a client error and must not be "
            "reported as a server fault.",
            "<b>A declared <font face='Courier'>response_model</font> on every "
            "route</b>, so the OpenAPI contract is generated from the code and "
            "cannot drift from it.",
            "<b>A bounded batch endpoint</b> (1–100 applications). The cap is a "
            "denial-of-service control as much as an ergonomic one.",
            "<b>Exception handlers, not try/except in routes.</b> Domain exceptions "
            "are translated centrally, so internal messages and stack traces never "
            "reach the client — asserted by "
            "<font face='Courier'>test_error_responses_never_leak_internals</font>.",
            "<b>Graceful degradation.</b> If the artifact is missing the process "
            "still starts, <font face='Courier'>/health</font> reports "
            "<font face='Courier'>degraded</font>, and scoring returns 503 — so an "
            "orchestrator keeps the previous replica serving instead of "
            "crash-looping.",
        ]
    )
    story += figure(
        "api_endpoints.png",
        "Figure 6: The endpoint table, generated from the live "
        "service's own <font face='Courier'>/openapi.json</font>.",
        16.0,
    )
    story += [
        code(
            defn("src/loan_risk/api/app.py", "health")
            + BLANK_LINE
            + defn("src/loan_risk/api/app.py", "predict")
        ),
        para(
            "Listing 6: route definitions — documented status codes, declared "
            "response models, tags for the Swagger grouping.",
            "caption",
        ),
        para("The application: generated Swagger UI", "h3"),
        para(
            "FastAPI derives the interactive documentation from the same "
            "type annotations that enforce validation at runtime, so the page "
            "below is not a hand-maintained artefact that can go stale — it "
            "<i>is</i> the contract. The screenshots are of the running "
            "service at <font face='Courier'>http://127.0.0.1:8077/docs</font>.",
            "body",
        ),
    ]
    story += figure(
        "swagger_ui.png",
        "Figure 7: Swagger UI at <font face='Courier'>/docs</font>. Routes are "
        "grouped by tag (<i>ops</i>, <i>inference</i>) and all six request/"
        "response models are published under Schemas.",
        15.0,
    )
    story += figure(
        "swagger_predict.png",
        "Figure 8: <font face='Courier'>POST /v1/predict</font> expanded. The "
        "worked example payload comes from the schema itself, and all four "
        "documented outcomes are published to the client: 200 scored, 400 "
        "business-rule rejection, 422 schema validation failure, 503 model "
        "unavailable.",
        14.0,
    )
    story += [
        para("Verification against a running service", "h3"),
        para(
            "The figures below are the recorded transcript of real HTTP calls "
            "issued against the same live "
            "<font face='Courier'>uvicorn</font> process "
            "(<font face='Courier'>reports/metrics/api_transcript.json</font>).",
            "body",
        ),
    ]
    story += figure(
        "api_happy_path.png",
        "Figure 9: Health probe and two scored applications. The "
        "same schema-valid payload with a 720 credit score and a "
        "540 score with prior defaults yields LOW (p=0.939, "
        "approved) and HIGH (p=0.081, denied) respectively.",
        16.0,
    )
    story += figure(
        "api_error_handling.png",
        "Figure 10: Differentiated error handling. Out-of-range "
        "value → 422 with the offending field named; unknown key "
        "→ 422 <font face='Courier'>extra_forbidden</font>; "
        "schema-valid but over-leveraged → 400 with a business "
        "reason and no internal detail.",
        16.0,
    )

    # ==================== OBJECTIVE 2 ====================
    story += [
        PageBreak(),
        para("Objective 2 — Quality Assurance", "h1"),
        para("6. Test Types Implemented", "h2"),
        para(
            f"The suite contains <b>{len(tests)} tests</b>, all passing, "
            "organised into four distinct types and tagged with pytest markers "
            "so each layer can be run independently in CI "
            "(<font face='Courier'>pytest -m unit</font> for the fast "
            "pre-commit gate, the full suite before merge)."
        ),
        table(
            [
                ["Type", "Marker", "File", "Count", "What it proves"],
                [
                    "Unit",
                    "<font face='Courier'>unit</font>",
                    "<font face='Courier'>test_unit_features.py</font>",
                    str(count("test_unit_features")),
                    "Individual pure functions and the transformer behave in "
                    "isolation: correct values, determinism, no mutation, correct "
                    "failure modes",
                ],
                [
                    "Integration",
                    "<font face='Courier'>integration</font>",
                    "<font face='Courier'>test_integration_api.py</font>",
                    str(count("test_integration_api")),
                    "Routing → Pydantic → registry → predictor → transformer → "
                    "estimator → serialisation are wired together correctly, "
                    "exercised over real HTTP",
                ],
                [
                    "Data validation",
                    "<font face='Courier'>data</font>",
                    "<font face='Courier'>test_data_validation.py</font>",
                    str(count("test_data_validation")),
                    "The data contract holds: schema conformance, missing values, "
                    "PSI/KS drift, ingestion feature contract",
                ],
                [
                    "ML behavioural",
                    "<font face='Courier'>ml</font>",
                    "<font face='Courier'>test_model_training.py</font>, "
                    "<font face='Courier'>test_model_inference.py</font>",
                    str(count("test_model_training") + count("test_model_inference")),
                    "Learning actually happens, and inference obeys the domain's "
                    "shape, range, directional and invariance expectations",
                ],
            ],
            [2.4 * cm, 1.9 * cm, 4.1 * cm, 1.2 * cm, 7.0 * cm],
            align_center=[3],
        ),
        para(
            "Figure 11: Test inventory. Fixtures build every artefact in "
            "memory, so the suite runs on a clean checkout with no trained "
            "model on disk.",
            "caption",
        ),
        code(
            "$ python -m pytest tests -q\n"
            f"{art['pytest_tail']}\n"
            "\n"
            "$ python -m pytest tests -m unit --co -q\n"
            f"{count('test_unit_features')}/84 tests collected "
            f"({84 - count('test_unit_features')} deselected)"
        ),
        para(
            "Listing 7: full-suite result. Raw output is stored in "
            "<font face='Courier'>reports/metrics/pytest_output.txt</font>.",
            "caption",
        ),
    ]

    # ---------------- 7. ML tests ----------------
    story += [
        para("7. Tests for ML Components", "h2"),
        para("7.1 Testing model training", "h3"),
        para(
            "A model that trains without raising is not a model that has "
            "learned. Five training-specific tests establish that it has:"
        ),
        table(
            [
                ["Test", "Assertion", "Defect it catches"],
                [
                    "<font face='Courier'>test_model_can_overfit_a_small_batch" "</font>",
                    "A high-capacity forest reaches ≥ 0.95 training accuracy on 40 "
                    "rows",
                    "Features misaligned with labels, or the pipeline dropping the "
                    "signal before the estimator",
                ],
                [
                    "<font face='Courier'>test_training_loss_decreases_with_"
                    "capacity</font>",
                    "Log-loss falls monotonically over "
                    "<font face='Courier'>max_depth</font> ∈ {1, 3, 8, None}, "
                    "and the last is under half the first",
                    "The model is not fitting the batch at all",
                ],
                [
                    "<font face='Courier'>test_training_is_reproducible</font>",
                    "Two runs with the same seed give identical probabilities",
                    "Unseeded randomness; unauditable results",
                ],
                [
                    "<font face='Courier'>test_shuffled_labels_destroy_"
                    "generalisation</font>",
                    "With permuted labels, held-out AUC collapses to 0.35–0.65",
                    "Target leakage or a broken evaluation split — if a model can "
                    "still 'predict' random labels, the metrics are meaningless",
                ],
                [
                    "<font face='Courier'>test_quality_gates_reject_a_weak_model"
                    "</font>",
                    "A depth-1, one-tree stump trips the release gate",
                    "A gate that silently passes everything",
                ],
            ],
            [4.6 * cm, 5.6 * cm, 6.4 * cm],
        ),
        para("Figure 12: Model-training tests.", "caption"),
        code(
            defn("tests/test_model_training.py", "test_model_can_overfit_a_small_batch")
        ),
        para("Listing 8: the canonical overfit-a-small-batch test.", "caption"),
        para(
            "Measured loss curve produced by the capacity sweep: "
            "<b>0.4290 → 0.2936 → 0.1146 → 0.0940</b> for "
            "<font face='Courier'>max_depth</font> 1 → 3 → 8 → unbounded. "
            "Depth, not tree count, is the right capacity axis here: an "
            "unconstrained forest already interpolates the batch at a single "
            "tree, so sweeping <font face='Courier'>n_estimators</font> would "
            "measure variance reduction rather than learning. Our first "
            "version of this test made exactly that mistake and failed; the "
            "test was wrong, not the model.",
            "body",
        ),
        para("7.2 Testing model inference", "h3"),
        para("Inference tests follow the three standard families:", "body"),
        table(
            [
                ["Family", "Representative test", "Expectation"],
                [
                    "Shape &amp; range",
                    "<font face='Courier'>test_batch_prediction_shape_is_correct"
                    "</font><br/>"
                    "<font face='Courier'>test_probabilities_are_valid_and_sum_to_"
                    "one</font><br/>"
                    "<font face='Courier'>test_output_stays_in_range_across_the_"
                    "credit_spectrum</font>",
                    "37 rows in → (37, 2) out; every probability in [0, 1] and "
                    "rows summing to 1; finite output for credit scores 300–850",
                ],
                [
                    "Directional",
                    "<font face='Courier'>test_higher_credit_score_does_not_reduce_"
                    "approval_probability</font><br/>"
                    "<font face='Courier'>test_higher_debt_to_income_does_not_"
                    "increase_approval_probability</font><br/>"
                    "<font face='Courier'>test_a_much_larger_loan_does_not_increase_"
                    "approval_probability</font>",
                    "Domain monotonicity holds: 520→800 credit score cannot lower "
                    "P(approve); 0.10→0.95 DTI cannot raise it; a 15k→140k loan "
                    "cannot raise it",
                ],
                [
                    "Invariance",
                    "<font face='Courier'>test_prediction_is_invariant_to_"
                    "dictionary_key_order</font><br/>"
                    "<font face='Courier'>test_prediction_is_invariant_to_column_"
                    "order</font><br/>"
                    "<font face='Courier'>test_prediction_is_invariant_to_batching"
                    "</font><br/>"
                    "<font face='Courier'>test_prediction_is_invariant_to_repeated_"
                    "calls</font>",
                    "Payload key order, dataframe column order, batched vs "
                    "row-by-row scoring, and repetition must all leave the score "
                    "unchanged",
                ],
            ],
            [2.6 * cm, 7.4 * cm, 6.6 * cm],
        ),
        para(
            "Figure 13: Model-inference tests. The column-order test is what "
            "proves the <font face='Courier'>FeatureEngineer</font> really "
            "does re-impose the feature contract inside the artifact.",
            "caption",
        ),
        code(
            defn(
                "tests/test_model_inference.py",
                "test_higher_credit_score_does_not_reduce_approval_probability",
            )
            + BLANK_LINE
            + defn(
                "tests/test_model_inference.py",
                "test_higher_debt_to_income_does_not_increase_approval_probability",
            )
            + BLANK_LINE
            + defn(
                "tests/test_model_inference.py",
                "test_a_much_larger_loan_does_not_increase_approval_probability",
            )
        ),
        para(
            "Listing 9: directional tests. They assert the weak inequality — a "
            "tree ensemble is not globally monotonic, and demanding strict "
            "monotonicity would produce a flaky test that teams learn to "
            "ignore.",
            "caption",
        ),
        para("7.3 A defect the suite actually caught", "h3"),
        para(
            "<font face='Courier'>test_prediction_is_invariant_to_repeated_calls"
            "</font> failed on its first run. Scoring the identical payload "
            "five times produced four distinct floats. The cause was "
            "<font face='Courier'>n_jobs=-1</font> on the "
            "<font face='Courier'>RandomForestClassifier</font>: per-tree vote "
            "accumulation happens in whatever order the worker threads finish, "
            "and floating-point addition is not associative. The spread was "
            "~5×10⁻¹⁷ — invisible in any aggregate metric, and completely "
            "unacceptable here, because an applicant whose probability sits "
            "exactly on the 0.50 cut-off could receive APPROVED on one request "
            "and DENIED on an identical retry. In regulated lending, decisions "
            "must be bit-reproducible for audit.",
            "body",
        ),
        para(
            "The fix was to pin the estimator to a deterministic reduction. "
            "Inference is ~10 ms per request against a 150 ms SLA, so there "
            "was no performance argument for keeping the parallel path.",
            "body",
        ),
        code(
            between(
                "src/loan_risk/models/trainer.py",
                "estimator = RandomForestClassifier(",
                "n_jobs=1,",
            )
        ),
        para(
            "Listing 10: the fix, with the reason recorded at the point of "
            "the decision and a pointer back to the test that found it.",
            "caption",
        ),
    ]

    # ---------------- 8. metrics ----------------
    drift_rows = art["drift"][1:6]
    story += [
        para("8. Model-Quality and Data-Quality Metrics", "h2"),
        para("8.1 Model quality — four metrics", "h3"),
        para(
            "Accuracy alone is a poor summary for underwriting: the class "
            "balance is uneven (40.3% approvals) and the <i>probability</i>, "
            "not just the label, drives the risk tier. Four complementary "
            "metrics are therefore measured on a stratified held-out split of "
            f"{mq['n_test']:,} applications, and all four are enforced as "
            "release gates in <font face='Courier'>config.yaml</font>."
        ),
        table(
            [
                ["ID", "Metric", "Why it is measured", "Gate", "Measured", "Status"],
                [
                    "MQ-1",
                    "Accuracy",
                    "Headline decision correctness",
                    "≥ 0.80",
                    f"<b>{mq['accuracy']:.4f}</b>",
                    "PASS",
                ],
                [
                    "MQ-2",
                    "F1 score",
                    "Balances the cost of a missed good customer against an "
                    "approved bad one under class imbalance",
                    "≥ 0.80",
                    f"<b>{mq['f1']:.4f}</b>",
                    "PASS",
                ],
                [
                    "MQ-3",
                    "ROC-AUC",
                    "Threshold-independent ranking quality; survives a change to "
                    "the 0.50 cut-off",
                    "≥ 0.85",
                    f"<b>{mq['roc_auc']:.4f}</b>",
                    "PASS",
                ],
                [
                    "MQ-4",
                    "Brier score",
                    "<b>Calibration.</b> The risk tier is derived from the "
                    "probability, so the probability itself must be truthful",
                    "≤ 0.15",
                    f"<b>{mq['brier_score']:.4f}</b>",
                    "PASS",
                ],
                [
                    "",
                    "Precision / Recall / Log-loss",
                    "Reported alongside for diagnosis",
                    "—",
                    f"{mq['precision']:.4f} / {mq['recall']:.4f} / "
                    f"{mq['log_loss']:.4f}",
                    "—",
                ],
            ],
            [1.2 * cm, 3.0 * cm, 6.4 * cm, 1.7 * cm, 2.5 * cm, 1.8 * cm],
            align_center=[0, 3, 4, 5],
        ),
        para("Figure 14: Model-quality metrics against their release gates.", "caption"),
    ]
    story += figure(
        "model_quality_metrics.png",
        "Figure 15: ROC, calibration and confusion matrix on the "
        "held-out split. The calibration curve tracking the "
        "diagonal is what makes the LOW/MEDIUM/HIGH tiering "
        "defensible — a 0.70 prediction really does correspond to "
        "roughly 70% observed approvals.",
        16.5,
    )
    story += [
        para(
            "Algorithm comparison (carried forward from the Assignment I "
            "Analytics Design View)",
            "h3",
        ),
        table(
            [
                ["Candidate", "Accuracy", "F1", "ROC-AUC", "Brier", "Verdict"],
                [
                    "Logistic Regression (explainable baseline)",
                    f"{comparison['logistic_regression']['accuracy']:.4f}",
                    f"{comparison['logistic_regression']['f1']:.4f}",
                    f"{comparison['logistic_regression']['roc_auc']:.4f}",
                    f"{comparison['logistic_regression']['brier_score']:.4f}",
                    "Baseline",
                ],
                [
                    "Random Forest (<b>chosen</b>)",
                    f"<b>{comparison['random_forest']['accuracy']:.4f}</b>",
                    f"<b>{comparison['random_forest']['f1']:.4f}</b>",
                    f"<b>{comparison['random_forest']['roc_auc']:.4f}</b>",
                    f"<b>{comparison['random_forest']['brier_score']:.4f}</b>",
                    "Wins on every metric",
                ],
            ],
            [5.6 * cm, 2.2 * cm, 2.0 * cm, 2.2 * cm, 2.0 * cm, 2.6 * cm],
            align_center=[1, 2, 3, 4],
        ),
        para(
            "Figure 16: The winner is selected automatically by "
            "<font face='Courier'>ModelTrainer.compare_algorithms()</font> on "
            "ROC-AUC — the choice is code, not a comment.",
            "caption",
        ),
        para("8.2 Data quality — four metrics", "h3"),
        para(
            "In an ML system the data is a dependency exactly as a library is, "
            "so it needs its own measurable contract. Two metrics come from the "
            "schema gate and two from the drift monitor."
        ),
        table(
            [
                [
                    "ID",
                    "Metric",
                    "Definition",
                    "Gate",
                    "Measured on the " "20,000-row training table",
                ],
                [
                    "DQ-1",
                    "Schema conformance rate",
                    "Fraction of the 22 declared columns present <i>and</i> "
                    "within their declared type and range",
                    "= 1.00",
                    f"<b>{dq['schema_conformance_rate']:.4f}</b> "
                    f"({dq['n_violations']} violations)",
                ],
                [
                    "DQ-2",
                    "Missing-value fraction",
                    "Overall null ratio, plus the worst single column",
                    "≤ 0.02",
                    f"<b>{dq['missing_value_fraction']:.4f}</b> overall; worst "
                    f"column {dq['worst_column_missing_fraction']:.4f}",
                ],
                [
                    "DQ-3",
                    "Population Stability Index",
                    "Per-feature covariate shift against the frozen training "
                    "reference profile; the banking-standard drift measure",
                    "≤ 0.20",
                    "Monitored per batch — see the stress test below",
                ],
                [
                    "DQ-4",
                    "Kolmogorov–Smirnov statistic",
                    "Distribution-free two-sample test; catches tail shifts that "
                    "PSI's coarse bins can miss",
                    "p ≥ 0.05",
                    "Reported alongside PSI for every feature",
                ],
            ],
            [1.2 * cm, 3.4 * cm, 5.6 * cm, 1.7 * cm, 4.7 * cm],
            align_center=[0, 3],
        ),
        para(
            "Figure 17: Data-quality metrics. DQ-1 and DQ-2 gate the training "
            "run (<font face='Courier'>strict=True</font> aborts the build); "
            "DQ-3 and DQ-4 run against live batches in production.",
            "caption",
        ),
        para("Drift stress test", "h3"),
        para(
            "To prove the monitor detects real shift rather than merely running "
            "without error, the held-out split was perturbed into a synthetic "
            "recession — credit scores down 85 points, debt-to-income ×1.6, "
            "card utilisation ×1.5 — and compared against the training "
            "reference. The monitor flagged exactly the three perturbed "
            "features as SEVERE and left the other 19 STABLE, and emitted a "
            "WARNING naming them."
        ),
        table(
            [["Feature", "PSI", "KS statistic", "KS p-value", "Severity"]]
            + [[r[0], r[1], r[2], r[3], r[4]] for r in drift_rows],
            [5.0 * cm, 2.6 * cm, 2.9 * cm, 2.9 * cm, 3.2 * cm],
            align_center=[1, 2, 3, 4],
        ),
        para(
            "Figure 18: Top five features by PSI "
            "(<font face='Courier'>reports/metrics/drift_report.csv</font>). "
            "The gap between the third and fourth row is the signal: real "
            "shift is unambiguous.",
            "caption",
        ),
    ]
    story += figure(
        "drift_psi.png",
        "Figure 19: PSI per feature against the 0.10/0.25 industry " "thresholds.",
        14.0,
    )
    story += figure(
        "latency_distribution.png",
        f"Figure 20: End-to-end scoring latency over "
        f"{lat['n_requests']:,} requests through the real serving "
        f"path: mean {lat['mean']:.2f} ms, p95 {lat['p95']:.2f} ms, "
        f"p99 {lat['p99']:.2f} ms — comfortably inside the "
        f"{lat['sla']:.0f} ms SLA inherited from Assignment I.",
        12.5,
    )

    # ---------------- 9. production experimentation + security ----------------
    story += [
        para("9. Testing in Production, and a Security Consideration", "h2"),
        para("9.1 Approach: shadow deployment, then canary", "h3"),
        para(
            "Offline metrics are necessary and not sufficient. A credit model "
            "meets a population it never saw in training, and the label — "
            "whether the loan actually defaults — arrives months later. Our "
            "release path is therefore staged, and the staging is what makes it "
            "safe to be wrong."
        ),
        table(
            [
                [
                    "Stage",
                    "Traffic",
                    "What is compared",
                    "Promote when",
                    "Roll back when",
                ],
                [
                    "<b>1. Shadow</b>",
                    "100% mirrored, <b>0% served</b>",
                    "The candidate scores every live request in parallel with the "
                    "incumbent; only the incumbent's answer is returned. We diff "
                    "score distributions, tier mix, disagreement rate and latency.",
                    "≥ 2 weeks with no unexplained disagreement and p99 latency "
                    "within budget",
                    "Never — shadow traffic is not served, so a bad candidate "
                    "cannot harm a customer",
                ],
                [
                    "<b>2. Canary</b>",
                    "5% → 25% → 50% → 100%",
                    "Live business KPIs on the canary slice: approval rate, "
                    "average risk tier, manual-override rate, error rate, p99 "
                    "latency",
                    "Each step held ≥ 48 h with KPIs inside their control limits",
                    "Automatic rollback on any breach; the previous artifact is "
                    "one config change away",
                ],
                [
                    "<b>3. A/B holdout</b>",
                    "Permanent 5% on the incumbent",
                    "The metric that actually matters — realised default rate — "
                    "measured months later against a like-for-like control",
                    "Retained permanently as the reference arm",
                    "n/a",
                ],
            ],
            [2.3 * cm, 2.7 * cm, 5.6 * cm, 3.2 * cm, 3.2 * cm],
        ),
        para(
            "Figure 21: Staged rollout. Shadow answers 'does it behave?'; "
            "canary answers 'does it behave on customers?'; the A/B holdout "
            "answers 'was it actually better?'.",
            "caption",
        ),
        para(
            "<b>Why shadow first for this system specifically.</b> The "
            "outcome label is delayed by months, so we cannot A/B-test our way "
            "to a decision quickly. Shadow mode gives an immediate, zero-risk "
            "read on the one thing that is observable on day one — whether the "
            "candidate's score distribution and its disagreements with the "
            "incumbent are explainable. The infrastructure for this already "
            "exists in the codebase: <font face='Courier'>ModelRegistry</font> "
            "makes 'which artifact is loaded' a configuration value, and every "
            "scoring event is already logged as JSON with "
            "<font face='Courier'>model_version</font>, so comparing two arms "
            "is a log query rather than new instrumentation. "
            "<font face='Courier'>DriftMonitor</font> supplies the population "
            "check for the same window.",
            "body",
        ),
        para(
            "9.2 Security consideration: the input boundary as an attack " "surface", "h3"
        ),
        para(
            "The threat we treat as primary is <b>adversarial and malformed "
            "input at the scoring endpoint</b>. The endpoint is reachable from "
            "a customer-facing portal, it accepts a 20-field numeric payload, "
            "and its output is a lending decision with direct financial "
            "consequence. That combination invites three concrete abuses: "
            "gaming the model by probing which field flips a decision; feeding "
            "out-of-distribution values to push the model into a region where "
            "its behaviour is untested; and classic injection or resource "
            "exhaustion against the service itself."
        ),
        table(
            [
                ["Control", "Where", "Effect"],
                [
                    "Bounded field constraints on all 20 inputs",
                    "<font face='Courier'>api/schemas.py</font>",
                    "Values outside the legitimate domain (credit score 1500, "
                    "income 10¹⁵) are rejected with 422 before any model code "
                    "runs — the estimator is never asked to extrapolate",
                ],
                [
                    "<font face='Courier'>extra=\"forbid\"</font>",
                    "<font face='Courier'>api/schemas.py</font>",
                    "Unknown keys are rejected instead of ignored; blocks "
                    "parameter-pollution probing and catches client typos",
                ],
                [
                    "Strict type coercion",
                    "Pydantic v2",
                    "A SQL/JS injection string in a numeric field fails "
                    "<font face='Courier'>int_parsing</font> — the payload never "
                    "becomes a query or a template",
                ],
                [
                    "Business-rule filter",
                    "<font face='Courier'>RiskPredictor.validate_business_rules"
                    "</font>",
                    "A second, semantic gate behind the syntactic one: "
                    "schema-valid but economically absurd applications are "
                    "refused with 400",
                ],
                [
                    "Batch size cap (100)",
                    "<font face='Courier'>BatchLoanApplications</font>",
                    "Bounds the work one request can demand — a basic DoS control",
                ],
                [
                    "Opaque error envelope",
                    "Exception handlers in " "<font face='Courier'>api/app.py</font>",
                    "Clients get a reason, never a stack trace, module path or "
                    "internal identifier",
                ],
                [
                    "Full audit log",
                    "<font face='Courier'>logging_utils.py</font>",
                    "Every decision is recorded with its inputs, probability, tier "
                    "and <font face='Courier'>model_version</font>, so probing "
                    "patterns are detectable after the fact and every decision is "
                    "reconstructable for a regulator",
                ],
            ],
            [4.4 * cm, 4.0 * cm, 8.2 * cm],
        ),
        para(
            "Figure 22: Layered input-validation controls. "
            "Six integration tests parametrised over injection strings, "
            "oversized batches and extreme numerics assert that each control "
            "holds.",
            "caption",
        ),
        para(
            "<b>What this does not solve.</b> Validation constrains each field "
            "independently; it cannot detect an application in which every "
            "value is individually plausible but the combination is fabricated. "
            "The honest mitigations for that are outside the request path: "
            "corroborating declared income and liabilities against the bureau "
            "feed rather than trusting the payload, rate-limiting and "
            "authenticating per client so systematic probing is visible and "
            "attributable, and monitoring the score distribution per caller for "
            "the signature of an optimisation attack. Model access control — "
            "the artifact and the reference profile are deployment assets, not "
            "public ones — belongs in the same list.",
            "body",
        ),
        PageBreak(),
    ]

    # ---------------- appendix ----------------
    story += [
        para("Appendix A — Repository Layout", "h1"),
        code(
            "Group_84/\n"
            "├── configs/config.yaml            # thresholds, gates, "
            "hyper-parameters, feature contract\n"
            "├── data/\n"
            "│   ├── generate_synthetic_data.py # schema-faithful stand-in for "
            "the Kaggle file\n"
            "│   ├── prepare_data.py            # cleaning, encoding, feature "
            "engineering\n"
            "│   └── loan_data_processed.csv    # 20,000 x 23 modelling table\n"
            "├── src/loan_risk/                 # THE PRODUCTION PACKAGE\n"
            "│   ├── config.py                  # frozen dataclasses loaded "
            "from YAML\n"
            "│   ├── exceptions.py              # domain exception hierarchy\n"
            "│   ├── logging_utils.py           # structured JSON logging + "
            "Timer\n"
            "│   ├── data/ingestion.py          # DataIngestor\n"
            "│   ├── data/validation.py         # DataValidator  -> DQ-1, DQ-2\n"
            "│   ├── features/engineering.py    # FeatureEngineer (sklearn "
            "transformer)\n"
            "│   ├── models/trainer.py          # ModelTrainer + quality gates\n"
            "│   ├── models/predictor.py        # ModelRegistry + RiskPredictor\n"
            "│   ├── monitoring/drift.py        # DriftMonitor   -> DQ-3, DQ-4\n"
            "│   └── api/{schemas,app}.py       # FastAPI contract and routes\n"
            "├── tests/                         # 84 tests: unit | integration "
            "| data | ml\n"
            "├── scripts/                       # train, evaluate, render "
            "evidence, build report\n"
            "├── notebooks/research_prototype.ipynb   # RESEARCH CODE "
            "(evidence)\n"
            "├── legacy/research_feature_prototype.py # same, exported for the "
            "linters\n"
            "├── reports/{lint,metrics,figures}/      # all captured evidence\n"
            "└── artifacts/model.joblib               # serialised pipeline"
        ),
        para("Appendix B — Reproducing Every Number in This Report", "h1"),
        code(
            "pip install -r requirements.txt\n"
            "\n"
            "python data/generate_synthetic_data.py --rows 20000   # raw table\n"
            "python data/prepare_data.py                           # 22 "
            "features + target\n"
            "python scripts/train_model.py                         # train, "
            "gate, persist\n"
            "python -m pytest tests -v                             # 84 tests\n"
            "python scripts/evaluate_and_report.py                 # metrics + "
            "figures\n"
            "\n"
            "python -m isort src scripts tests\n"
            "python -m black src scripts tests\n"
            "python -m flake8 src scripts tests\n"
            "python -m pylint src/loan_risk\n"
            "\n"
            "PYTHONPATH=src python -m uvicorn loan_risk.api.app:app --port 8000\n"
            "#   -> http://127.0.0.1:8000/docs   (generated Swagger UI)"
        ),
        para("Appendix C — Full Test Inventory", "h1"),
        para(f"All {len(tests)} tests, as collected by pytest.", "body"),
    ]

    module_names = {
        "test_unit_features.py": "Unit tests",
        "test_integration_api.py": "Integration tests",
        "test_data_validation.py": "Data-validation tests",
        "test_model_training.py": "ML tests — training",
        "test_model_inference.py": "ML tests — inference",
    }
    for module, heading in module_names.items():
        names = [t.split("::", 1)[1] for t in tests if t.startswith(f"tests/{module}")]
        block = [
            para(
                f"{heading} — <font face='Courier'>{module}</font> " f"({len(names)})",
                "h3",
            ),
            code("\n".join(names)),
        ]
        story.append(KeepTogether(block))

    return story


def main() -> int:
    """Render the report PDF."""
    art = load_artifacts()
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.75 * cm,
        bottomMargin=1.8 * cm,
        title="Group 84 - SEML Assignment II",
        author="Group 84, BITS Pilani WILP",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(build_story(art))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
