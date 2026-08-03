#!/usr/bin/env python3
"""CLI: digitize PXRD figures -> xy data + visual verification report."""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pxrd2xy.pipeline import run_batch, digitize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=["/Users/mac/Documents/MOF/img"])
    ap.add_argument("-o", "--outdir", default="/Users/mac/Documents/MOF/pxrd2xy/out")
    ap.add_argument("--cache", default="/Users/mac/Documents/MOF/pxrd2xy/.ocrcache")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--report", action="store_true", help="build the HTML dashboard")
    a = ap.parse_args()
    paths = []
    for p in a.inputs:
        if os.path.isdir(p):
            paths += sorted(glob.glob(os.path.join(p, "**", "*.png"), recursive=True))
            paths += sorted(glob.glob(os.path.join(p, "**", "*.jpg"), recursive=True))
        else:
            paths.append(p)
    paths = [p for p in paths if "__" not in os.path.basename(p)]
    recs = run_batch(paths, a.outdir, a.cache, verbose=a.verbose)
    if a.report:
        from pxrd2xy.report import build_report
        print("report:", build_report(recs, a.outdir))

if __name__ == "__main__":
    main()
