#!/usr/bin/env python3
"""CLI: PDF papers -> classified, context-resolved xy datasets."""
import argparse, glob, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper2xy.pipeline import run

DEFAULT_DIR = "/Users/mac/Documents/MOF/AIMATX - img_papers_db"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=[DEFAULT_DIR])
    ap.add_argument("-o", "--outdir", default="out_papers")
    ap.add_argument("-n", "--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", default="", help="comma-separated categories to keep")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    pdfs = []
    for p in a.inputs:
        pdfs += sorted(glob.glob(os.path.join(p, "*.pdf"))) if os.path.isdir(p) else [p]
    if a.limit and len(pdfs) > a.limit:
        random.seed(a.seed)
        pdfs = sorted(random.sample(pdfs, a.limit))
    only = {s.strip() for s in a.only.split(",") if s.strip()} or None
    recs = run(pdfs, a.outdir, only=only)
    if a.report:
        from paper2xy.report import build
        print("report:", build(recs, a.outdir))

if __name__ == "__main__":
    main()
