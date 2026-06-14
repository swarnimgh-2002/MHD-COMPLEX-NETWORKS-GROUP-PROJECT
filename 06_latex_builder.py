"""
06_latex_builder.py
Compiles the manuscript (pdflatex x2 + bibtex) and copies the result to
output/final_pdf/. Also builds the self-contained Overleaf bundle under
latex/final_latex_code/ by inlining every \\input and copying the figures.
"""

import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, "latex")
TABLES = os.path.join(ROOT, "tables")
FIGURES = os.path.join(ROOT, "figures")
BUNDLE = os.path.join(LATEX, "final_latex_code")
OUT_PDF = os.path.join(ROOT, "output", "final_pdf")
LOGS = os.path.join(ROOT, "output", "logs")
for d in (OUT_PDF, LOGS, BUNDLE, os.path.join(BUNDLE, "figures")):
    os.makedirs(d, exist_ok=True)

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\graphicspath\{(?:\{[^}]*\})+\}")


def run(cmd, cwd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def compile_dir(directory, jobname="main"):
    logs = []
    for cmd in ([["pdflatex", "-interaction=nonstopmode", "main.tex"],
                 ["bibtex", jobname],
                 ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                 ["pdflatex", "-interaction=nonstopmode", "main.tex"]]):
        r = run(cmd, directory)
        logs.append((r.stdout or "") + (r.stderr or ""))
    with open(os.path.join(LOGS, f"build_{os.path.basename(directory)}.log"),
              "w", encoding="utf-8") as f:
        f.write("\n\n".join(logs))


def resolve(token):
    cand = token if token.endswith(".tex") else token + ".tex"
    for base in (LATEX, ROOT):
        p = os.path.normpath(os.path.join(base, cand))
        if os.path.exists(p):
            return p
    fname = os.path.basename(cand)
    for sub in ("main_text", "appendix"):
        p = os.path.join(TABLES, sub, fname)
        if os.path.exists(p):
            return p
    return None


def inline(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    out, last = [], 0
    for m in INPUT_RE.finditer(text):
        out.append(text[last:m.start()])
        tgt = resolve(m.group(1))
        if tgt:
            out.append("\n" + inline(tgt) + "\n")
        else:
            out.append(m.group(0))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def build_bundle():
    merged = inline(os.path.join(LATEX, "main.tex"))
    merged = GRAPHICS_RE.sub(lambda _m: r"\graphicspath{{figures/}}", merged, 1)
    with open(os.path.join(BUNDLE, "main.tex"), "w", encoding="utf-8") as f:
        f.write(merged)
    shutil.copy2(os.path.join(LATEX, "references.bib"),
                 os.path.join(BUNDLE, "references.bib"))
    n = 0
    for root, _, files in os.walk(FIGURES):
        for fn in files:
            if fn.lower().endswith((".pdf", ".png")):
                shutil.copy2(os.path.join(root, fn),
                             os.path.join(BUNDLE, "figures", fn))
                n += 1
    print(f"Overleaf bundle: inlined main.tex + {n} figure files", flush=True)


def main():
    print("Compiling local multi-file build ...", flush=True)
    compile_dir(LATEX)
    src = os.path.join(LATEX, "main.pdf")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT_PDF, "MHD_final_report.pdf"))
        print("Copied local PDF to output/final_pdf/MHD_final_report.pdf",
              flush=True)

    print("Building self-contained Overleaf bundle ...", flush=True)
    build_bundle()
    compile_dir(BUNDLE)
    bsrc = os.path.join(BUNDLE, "main.pdf")
    if os.path.exists(bsrc):
        shutil.copy2(bsrc, os.path.join(OUT_PDF, "MHD_final_report_single.pdf"))
        print("Bundle PDF built and copied.", flush=True)
    # tidy bundle aux files
    for ext in (".aux", ".bbl", ".blg", ".log", ".out", ".toc"):
        p = os.path.join(BUNDLE, "main" + ext)
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
