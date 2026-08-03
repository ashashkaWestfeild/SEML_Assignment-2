"""Render the recorded live-API session as report figures.

The transcript in ``reports/metrics/api_transcript.json`` was captured by
issuing real HTTP calls against a running uvicorn process
(``127.0.0.1:8077``); this script only typesets it.

Run:
    python scripts/render_api_evidence.py
"""

from __future__ import annotations

import json
import textwrap
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "reports" / "figures"
METRICS = PROJECT_ROOT / "reports" / "metrics"

INK = "#1f3864"
STATUS_COLOR = {2: "#2e7d32", 4: "#c00000", 5: "#7b1fa2"}
MONO = {"family": "DejaVu Sans Mono", "size": 7.4}


def _status_color(status: int) -> str:
    return STATUS_COLOR.get(status // 100, "#555555")


REQUEST_WRAP = 118
MAX_RESPONSE_LINES = 11


def _card_lines(entry: dict) -> tuple[list[str], list[str]]:
    """Return the wrapped request lines and the trimmed response lines."""
    request = entry.get("request")
    if request is None:
        request_lines = ["(no request body)"]
    else:
        request_lines = textwrap.wrap(json.dumps(request), width=REQUEST_WRAP)[:3]

    response_lines = json.dumps(entry["response"], indent=1).splitlines()
    if len(response_lines) > MAX_RESPONSE_LINES:
        response_lines = response_lines[:MAX_RESPONSE_LINES] + ["  ... (truncated)"]
    return request_lines, response_lines


def render_transcript(
    transcript: list[dict], calls: list[int], out: Path, title: str
) -> None:
    """Typeset a subset of the recorded calls as request/response cards.

    Layout uses a single axes with a top-down line cursor (one unit = one text
    line) rather than fractional per-card coordinates, so blocks of different
    lengths can never overlap.
    """
    selected = [transcript[i] for i in calls]
    blocks = [_card_lines(entry) for entry in selected]

    # 5 chrome lines per card (header, label, 2 section labels, padding).
    card_heights = [len(req) + len(resp) + 6 for req, resp in blocks]
    total_lines = sum(card_heights) + 2 * len(selected) + 2

    line_height_in = 0.155
    fig, ax = plt.subplots(figsize=(11.4, total_lines * line_height_in))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(total_lines, 0)  # inverted: y counts lines downward

    cursor = 1.0
    for entry, (request_lines, response_lines), height in zip(
        selected, blocks, card_heights
    ):
        colour = _status_color(entry["status"])
        ax.add_patch(
            FancyBboxPatch(
                (0.6, cursor - 0.4),
                98.8,
                height - 0.4,
                boxstyle="round,pad=0.2,rounding_size=0.6",
                linewidth=1.1,
                edgecolor="#c9d2e3",
                facecolor="#f7f9fc",
                clip_on=False,
            )
        )
        y = cursor + 0.6
        ax.text(
            3,
            y,
            f"{entry['method']}  {entry['path']}",
            fontweight="bold",
            fontsize=10.5,
            color=INK,
            va="center",
        )
        ax.text(
            97,
            y,
            f"HTTP {entry['status']}",
            fontweight="bold",
            fontsize=10.5,
            color=colour,
            ha="right",
            va="center",
        )
        y += 1.3
        ax.text(
            3,
            y,
            entry["label"],
            fontsize=8.8,
            style="italic",
            color="#4a5568",
            va="center",
        )

        y += 1.5
        ax.text(
            3, y, "request", fontsize=7.6, fontweight="bold", color="#7a869a", va="center"
        )
        for line in request_lines:
            y += 1.0
            ax.text(3, y, line, fontdict=MONO, color="#243b53", va="center")

        y += 1.6
        ax.text(
            3,
            y,
            "response",
            fontsize=7.6,
            fontweight="bold",
            color="#7a869a",
            va="center",
        )
        for line in response_lines:
            y += 1.0
            ax.text(3, y, line, fontdict=MONO, color=colour, va="center")

        cursor += height + 2

    fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_endpoint_table(schema: dict, out: Path) -> None:
    """Typeset the generated OpenAPI contract as an endpoint table."""
    rows = []
    for path, methods in sorted(schema["paths"].items()):
        for method, spec in methods.items():
            codes = ", ".join(sorted(spec.get("responses", {})))
            rows.append(
                [
                    method.upper(),
                    path,
                    spec.get("summary", ""),
                    ", ".join(spec.get("tags", [])),
                    codes,
                ]
            )

    fig, ax = plt.subplots(figsize=(11.6, 0.52 * len(rows) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Method", "Path", "Summary", "Tag", "Documented status codes"],
        colWidths=[0.08, 0.20, 0.30, 0.10, 0.32],
        cellLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.7)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c9d2e3")
        if row == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f5fa")
    ax.set_title(
        "REST API contract generated from the code (GET /openapi.json)",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    """Render both API evidence figures from the captured artifacts."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    transcript = json.loads((METRICS / "api_transcript.json").read_text(encoding="utf-8"))

    schema_path = METRICS / "openapi.json"
    if not schema_path.exists():
        with urllib.request.urlopen("http://127.0.0.1:8077/openapi.json") as resp:
            schema_path.write_text(resp.read().decode("utf-8"), encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    render_endpoint_table(schema, FIGURES / "api_endpoints.png")
    render_transcript(
        transcript,
        [0, 2, 3],
        FIGURES / "api_happy_path.png",
        "Live service: health probe and two scored applications (HTTP 200)",
    )
    render_transcript(
        transcript,
        [5, 6, 8],
        FIGURES / "api_error_handling.png",
        "Live service: differentiated error handling (422 schema, 422 typo, "
        "400 business rule)",
    )
    print(f"Rendered API evidence figures into {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
