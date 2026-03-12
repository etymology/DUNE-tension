from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _wrap_markdown(text: str, width: int = 100) -> list[str]:
    wrapped: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            wrapped.append("")
            continue
        if line.startswith("|") or line.startswith("![") or line.startswith("```"):
            wrapped.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        prefix = " " * indent
        chunks = textwrap.wrap(
            line.strip(),
            width=width - indent,
            replace_whitespace=False,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not chunks:
            wrapped.append(prefix)
            continue
        wrapped.extend(prefix + chunk for chunk in chunks)
    return wrapped


def _add_text_pages(pdf: PdfPages, markdown_path: Path) -> None:
    lines = _wrap_markdown(markdown_path.read_text(encoding="utf-8"))
    lines_per_page = 42
    for start in range(0, len(lines), lines_per_page):
        page_lines = lines[start : start + lines_per_page]
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        fig.text(
            0.06,
            0.96,
            "\n".join(page_lines),
            ha="left",
            va="top",
            family="monospace",
            fontsize=9.5,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path) -> None:
    img = mpimg.imread(image_path)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(image_path.name, fontsize=12, pad=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a markdown report and image attachments into a PDF."
    )
    parser.add_argument("markdown", type=Path, help="Markdown report path")
    parser.add_argument("output_pdf", type=Path, help="Output PDF path")
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        default=[],
        type=Path,
        help="Image path to append as a full PDF page. Repeat for multiple images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output_pdf) as pdf:
        _add_text_pages(pdf, args.markdown)
        for image_path in args.images:
            _add_image_page(pdf, image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
