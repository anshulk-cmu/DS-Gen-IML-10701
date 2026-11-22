"""List the fonts embedded in every fig*.pdf of a directory; fail on Type 3 or unembedded fonts.

Usage (from the repository root):  python paper_figures/check_fonts.py [DIR]
DIR defaults to paper_figures/out. Requires pypdf.
"""
import sys
from pathlib import Path

from pypdf import PdfReader

FIG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out"
if not any(FIG.glob("fig*.pdf")):
    sys.exit(f"no fig*.pdf in {FIG}")


def fonts(pdf):
    out = {}
    for page in PdfReader(pdf).pages:
        res = page.get("/Resources") or {}
        stack = [res]
        while stack:  # also walk Form XObjects
            r = stack.pop().get_object()
            for f in (r.get("/Font") or {}).values():
                f = f.get_object()
                sub = f.get("/Subtype")
                desc = f.get("/FontDescriptor")
                if sub == "/Type0":
                    d = f["/DescendantFonts"][0].get_object()
                    desc, sub = d.get("/FontDescriptor"), f"Type0/{d.get('/Subtype')}"
                desc = desc.get_object() if desc else {}
                emb = next((k for k in ("/FontFile", "/FontFile2", "/FontFile3") if k in desc), None)
                out[str(f.get("/BaseFont"))] = (str(sub), emb)
            for x in (r.get("/XObject") or {}).values():
                x = x.get_object()
                if x.get("/Subtype") == "/Form" and "/Resources" in x:
                    stack.append(x["/Resources"])
    return out


bad = 0
for pdf in sorted(FIG.glob("fig*.pdf")):
    fs = fonts(pdf)
    issues = [n for n, (t, e) in fs.items() if t == "/Type3" or e is None]
    bad += bool(issues)
    desc = "; ".join(f"{n.lstrip('/')} [{t.lstrip('/')}, {e.lstrip('/') if e else 'NOT EMBEDDED'}]"
                     for n, (t, e) in sorted(fs.items()))
    print(f"{pdf.name}\t{pdf.stat().st_size/1024:.1f} KB\t{'OK' if not issues else 'BAD ' + str(issues)}\t{desc}")
sys.exit(1 if bad else 0)
