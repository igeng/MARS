"""
Simple Markdown → LaTeX converter for MARS survey output.

Converts the generated review Markdown to a basic LaTeX article suitable
for arXiv submission or conference templates.  Handles the most common
Markdown constructs; complex tables or figures may need manual touch-up.

Usage::

    from mars.utils.latex_converter import md_to_latex

    latex = md_to_latex(markdown_text, title="My Survey", author="MARS")
    Path("output.tex").write_text(latex)
"""

from __future__ import annotations

import re


def md_to_latex(
    md_text: str,
    title: str = "Automated Literature Review",
    author: str = "MARS (Multi-Agent Research System)",
    date: str = r"\today",
) -> str:
    """Convert a Markdown survey to a LaTeX article.

    Args:
        md_text: Markdown survey content.
        title: Document title.
        author: Author field.
        date: Date field (LaTeX command).

    Returns:
        Complete LaTeX document source.
    """
    lines = md_text.split("\n")
    latex_lines: list[str] = []

    # State
    in_reference = False
    in_enum = False
    in_itemize = False
    in_table = False
    table_rows: list[str] = []

    def _flush_table() -> str:
        nonlocal table_rows
        if not table_rows:
            return ""
        col_count = max(
            len(re.findall(r"(?<!\\)\|", row)) - 1 for row in table_rows
        )
        col_count = max(col_count, 2)
        cols = "|" + "c|" * col_count
        header = table_rows[0]
        body = "\\\\\n\\hline\n".join(table_rows[1:]) if len(table_rows) > 1 else ""
        result = f"\\begin{{tabular}}{{{cols}}}\n\\hline\n{header}\\\\\n\\hline\n{body}\n\\end{{tabular}}"
        table_rows = []
        return result

    for line in lines:
        stripped = line.strip()

        # References section
        if re.match(r"^##\s*References?", stripped, re.IGNORECASE):
            in_reference = True
            latex_lines.append(r"\begin{thebibliography}{99}")
            continue

        if in_reference:
            ref_match = re.match(r"^\[(\d+)\]\s+(.+)", stripped)
            if ref_match:
                num, text = ref_match.groups()
                latex_lines.append(r"\bibitem{ref" + num + "} " + text)
            continue

        # Headings
        if stripped.startswith("# "):
            title_text = _escape(stripped[2:])
            latex_lines.append(r"\title{" + title_text + "}")
            continue
        if stripped.startswith("## "):
            latex_lines.append(r"\section{" + _escape(stripped[3:]) + "}")
            continue
        if stripped.startswith("### "):
            latex_lines.append(r"\subsection{" + _escape(stripped[4:]) + "}")
            continue
        if stripped.startswith("#### "):
            latex_lines.append(r"\subsubsection{" + _escape(stripped[5:]) + "}")
            continue

        # Table
        if "|" in stripped and stripped.startswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            if "---" not in stripped:  # skip separator rows
                row = " & ".join(
                    _escape(cell.strip()) for cell in stripped.split("|")[1:-1]
                )
                table_rows.append(row)
            continue
        elif in_table:
            latex_lines.append(_flush_table())
            in_table = False

        # Lists
        if re.match(r"^\d+\.\s", stripped):
            latex_lines.append(r"\begin{enumerate}")
            in_enum = True
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_itemize:
                latex_lines.append(r"\begin{itemize}")
                in_itemize = True
            latex_lines.append(r"\item " + _escape(stripped[2:]))
            continue
        elif not stripped and in_itemize:
            latex_lines.append(r"\end{itemize}")
            in_itemize = False
        elif not stripped and in_enum:
            latex_lines.append(r"\end{enumerate}")
            in_enum = False

        # Emphasis
        line = _escape(stripped)
        line = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", line)
        line = re.sub(r"\*(.+?)\*", r"\\textit{\1}", line)

        if line:
            latex_lines.append(line)
        else:
            latex_lines.append("")

    # Cleanup
    if in_table:
        latex_lines.append(_flush_table())
    if in_itemize:
        latex_lines.append(r"\end{itemize}")
    if in_enum:
        latex_lines.append(r"\end{enumerate}")
    if in_reference:
        latex_lines.append(r"\end{thebibliography}")

    body = "\n\n".join(latex_lines)

    return f"""\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{hyperref}}
\\usepackage{{booktabs}}
\\usepackage[margin=1in]{{geometry}}

\\title{{{_escape(title)}}}
\\author{{{_escape(author)}}}
\\date{{{date}}}

\\begin{{document}}
\\maketitle

{body}

\\end{{document}}
"""


def _escape(text: str) -> str:
    """Escape LaTeX special characters."""
    for char, repl in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\^{}"),
    ]:
        text = text.replace(char, repl)
    return text
