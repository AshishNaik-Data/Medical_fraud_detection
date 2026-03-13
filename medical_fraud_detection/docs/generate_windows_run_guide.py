from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "Windows_Run_Guide.pdf"


PAGES = [
    {
        "title": "Medical Fraud Detection\nWindows Run Guide",
        "lines": [
            "Project folder:",
            str(PROJECT_ROOT),
            "",
            "This guide explains how to run the project on a fresh Windows laptop.",
            "The project now auto-checks Python, virtual environment health, and missing libraries.",
            "",
            "Recommended launch command:",
            r"start_project.cmd",
            "",
            "PowerShell equivalent:",
            r"powershell -ExecutionPolicy Bypass -File .\start_project.ps1",
        ],
    },
    {
        "title": "Fresh Machine Setup",
        "lines": [
            "1. Open Command Prompt or PowerShell in the project folder.",
            "2. Run the launcher below.",
            "",
            r"start_project.cmd",
            "",
            "What it does automatically:",
            "- checks for Python 3",
            "- downloads and installs Python 3.12 if needed",
            "- creates or repairs .venv",
            "- installs required dependencies from requirements.txt",
            "- starts the Streamlit dashboard",
            "",
            "If PowerShell script execution is restricted, use:",
            r"powershell -ExecutionPolicy Bypass -File .\start_project.ps1",
        ],
    },
    {
        "title": "Optional Packages And Pipeline",
        "lines": [
            "To install optional ML and Neo4j packages too:",
            r"start_project.cmd -InstallOptional",
            "",
            "To force a clean reinstall of the environment:",
            r"start_project.cmd -ForceReinstall",
            "",
            "To rerun the full end-to-end pipeline:",
            r"powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -InstallOptional",
            "",
            "To rerun the pipeline and load Neo4j tables into a live Neo4j database:",
            r"powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -InstallOptional -LoadNeo4j",
        ],
    },
    {
        "title": "Useful URLs And Files",
        "lines": [
            "Streamlit dashboard:",
            "http://localhost:8501",
            "",
            "Neo4j Browser:",
            "http://localhost:7474",
            "",
            "Main scripts:",
            "- start_project.cmd",
            "- start_project.ps1",
            "- setup_local.ps1",
            "- run_dashboard.ps1",
            "- run_pipeline.ps1",
            "",
            "If needed, reopen the terminal inside the project folder before rerunning commands.",
        ],
    },
]


def add_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.9])
    ax.axis("off")

    y = 0.97
    ax.text(0.0, y, title, fontsize=20, fontweight="bold", va="top", family="DejaVu Sans")
    y -= 0.08

    for line in lines:
        if not line:
            y -= 0.03
            continue
        wrapped = wrap(line, width=84) or [""]
        for wrapped_line in wrapped:
            ax.text(0.0, y, wrapped_line, fontsize=11.5, va="top", family="DejaVu Sans Mono" if "\\" in wrapped_line or ".cmd" in wrapped_line or ".ps1" in wrapped_line else "DejaVu Sans")
            y -= 0.028
        y -= 0.004

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        for page in PAGES:
            add_page(pdf, page["title"], page["lines"])
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
