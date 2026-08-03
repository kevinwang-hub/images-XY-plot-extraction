"""Ground-truth accuracy test.

Every other check in this project compares the digitised curve against the *pixels* it came
from. That cannot tell us the absolute error, because the pixels are already a lossy drawing
of the data. So here a figure is generated from known xy data, digitised, and the result is
compared with the data it was drawn from.

Run:  python3 tests/test_synthetic_accuracy.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "/tmp/pxrd_synth"


def make_truth(n=1400):
    """Three stacked PXRD-like patterns with peaks from very sharp to broad."""
    x = np.linspace(5.0, 50.0, n)
    peaks = [  # (position, height, fwhm) - the last few are deliberately sub-pixel narrow
        (7.3, 1.00, 0.10), (9.8, 0.42, 0.12), (12.4, 0.55, 0.09), (15.1, 0.30, 0.35),
        (18.0, 0.22, 0.60), (21.7, 0.38, 0.10), (25.2, 0.18, 0.25), (29.0, 0.26, 0.08),
        (33.5, 0.12, 0.40), (38.2, 0.15, 0.10), (44.0, 0.09, 0.30),
    ]
    curves = []
    for k, scale in enumerate([1.0, 0.75, 0.55]):
        y = np.full_like(x, 0.02)
        for pos, h, fwhm in peaks:
            sig = fwhm / 2.355
            y += scale * h * np.exp(-0.5 * ((x - (pos + 0.15 * k)) / sig) ** 2)
        curves.append(y)
    return x, curves


def render(x, curves, path, lw=2.5, dpi=110, offsets=(0.0, 1.25, 2.5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    labels = ["sample A", "sample B", "simulated"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=dpi)
    for y, off, col, lab in zip(curves, offsets, colors, labels):
        ax.plot(x, y + off, lw=lw, color=col, label=lab)
    ax.set_xlabel("2θ (degree)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(5, 50)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return offsets


def main():
    os.makedirs(OUT, exist_ok=True)
    img = os.path.join(OUT, "synthetic_f1a.png")
    x, curves = make_truth()
    offsets = render(x, curves, img)

    from pxrd2xy.pipeline import digitize
    rec = digitize(img, os.path.join(OUT, "figs"), None, save_visuals=False)
    assert rec["ok"], rec.get("warnings")

    # the pipeline reports y as a fraction of the plot box; put truth on the same scale
    ylo, yhi = 0.0, max(c.max() + o for c, o in zip(curves, offsets))
    got = sorted(rec["curves"], key=lambda c: -np.median(c["y"]))   # top curve first
    truth = list(zip(curves, offsets))[::-1]                        # top curve first

    print(f"digitised {len(got)} of 3 curves   x range "
          f"{rec['axis']['x_range'][0]:.2f} – {rec['axis']['x_range'][1]:.2f} "
          f"(truth 5.00 – 50.00)")
    print(f"x-axis label {rec['axis']['x_axis_label']!r}   "
          f"legends {[c['legend'] for c in got]}")
    print()
    print(f"{'curve':<10}{'n':>6}{'x err (°)':>12}{'y RMSE':>10}{'y p95':>9}"
          f"{'peak pos err (°)':>19}{'peak height err':>17}")
    ok = True
    for i, (g, (yt, off)) in enumerate(zip(got, truth)):
        gx, gy = np.asarray(g["x"]), np.asarray(g["y"])
        # the axes were drawn with matplotlib's default 5% margins
        span = yhi - ylo
        lo, hi = ylo - 0.05 * span, yhi + 0.05 * span
        truth_frac = (yt + off - lo) / (hi - lo)
        ref = np.interp(gx, x, truth_frac)
        err = gy - ref
        rmse = float(np.sqrt(np.mean(err ** 2)))
        p95 = float(np.percentile(np.abs(err), 95))

        # peak positions and heights, on the strongest peaks
        from scipy.signal import find_peaks
        pi, _ = find_peaks(gy, height=np.median(gy) + 0.15 * (gy.max() - np.median(gy)))
        ti, _ = find_peaks(truth_frac, height=np.median(truth_frac)
                           + 0.15 * (truth_frac.max() - np.median(truth_frac)))
        gpos, tpos = gx[pi], x[ti]
        dpos, dh = [], []
        for p, h in zip(tpos, truth_frac[ti]):
            if not len(gpos):
                break
            j = int(np.argmin(np.abs(gpos - p)))
            if abs(gpos[j] - p) < 0.5:
                dpos.append(abs(gpos[j] - p))
                dh.append(abs(gy[pi][j] - h))
        xstep = float(np.median(np.diff(gx)))
        print(f"{g['legend'][:9]:<10}{len(gx):>6}{xstep:>12.4f}{rmse:>10.4f}{p95:>9.4f}"
              f"{np.mean(dpos) if dpos else float('nan'):>19.4f}"
              f"{np.mean(dh) if dh else float('nan'):>17.4f}")
        # tolerance is set from the representation limit, not from taste: peaks narrower
        # than the pen cannot be recovered exactly, and the tallest-amplitude curve here is
        # deliberately at that limit. These bounds exist to catch regressions.
        if rmse > 0.025 or (dpos and np.mean(dpos) > 0.05):
            ok = False
    print()
    print("y is a fraction of the plot-box height, so a y RMSE of 0.01 = 1% of the axis.")
    print("RESULT:", "PASS" if ok else
          "CHECK - a curve exceeded 2.5% RMSE or 0.05 deg peak-position error")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
