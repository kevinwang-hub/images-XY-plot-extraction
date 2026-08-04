"""Decide what each detected panel actually is.

Geometry gets us panels; it cannot tell a diffractogram from an NMR spectrum, and it
still lets through the occasional crystal structure whose bonds happened to line up.
A vision model settles both questions at once — but only ever as a *classifier*. It is
never asked for coordinates, panel boxes, or data values: those are exactly the readings
vision models get plausibly and precisely wrong, and they come from pixels instead.

The caption travels with the image, because the panel alone often cannot say whether a
curve is an N2 isotherm or a CO2 one.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re

import cv2
import numpy as np

from . import llmcache

MODEL = os.environ.get("PAPER2XY_MODEL", "claude-opus-5")

# Types worth digitising as xy data, plus the ones we deliberately exclude.
PLOT_TYPES = [
    "pxrd", "gas_adsorption_isotherm", "tga", "dsc", "uv_vis_absorption",
    "photoluminescence", "ir_spectrum", "raman_spectrum", "nmr_spectrum",
    "cyclic_voltammetry", "magnetic_susceptibility", "impedance", "kinetics",
    "pore_size_distribution", "other_xy_plot",
]
NON_PLOT = ["crystal_structure", "molecular_diagram", "photograph", "scheme",
            "table", "bar_chart", "not_a_plot"]

# How the data is *drawn*. This decides which tracer runs, and it is a question about
# appearance, which is what a vision model is reliable for -- unlike anything positional.
RENDER_STYLES = ["lines", "markers", "markers_joined_by_lines", "mixed"]

SCHEMA = {
    "type": "object",
    "properties": {
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string", "enum": PLOT_TYPES + NON_PLOT},
                    "digitizable": {"type": "boolean"},
                    "n_curves": {"type": "integer"},
                    "render_style": {"type": "string", "enum": RENDER_STYLES},
                    "x_quantity": {"type": "string"},
                    "y_quantity": {"type": "string"},
                    "series_labels": {"type": "array", "items": {"type": "string"}},
                    "panel_letter": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "category", "digitizable", "n_curves", "x_quantity",
                             "y_quantity", "series_labels", "panel_letter", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["panels"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are classifying panels cropped from figures in inorganic-chemistry
papers. For each panel image, decide what it shows.

`digitizable` means: it is a 2-D plot with a quantitative x axis, drawn as continuous
curves or a scatter of points that could be traced back into numbers. Set it false for
crystal structures, molecular diagrams, photographs, schemes, tables, bar charts, and for
any crop that is not actually a plot (the panel detector is geometric and occasionally
mistakes a molecular structure for a set of axes).

`category` — use `pxrd` for powder X-ray diffraction (2-theta on x),
`gas_adsorption_isotherm` for uptake vs pressure or P/P0, `tga` for mass loss vs
temperature, `pore_size_distribution` for dV/dD vs pore width. Use `other_xy_plot` for a
genuine plot that fits none of the listed types.

`x_quantity` / `y_quantity` — copy the axis labels as printed, including units, e.g.
"2-theta (degree)" or "Volume adsorbed (cm3/g STP)". Use "" if an axis is unlabelled.

`series_labels` — the legend entries or curve labels visible *in this panel*, verbatim,
including bare codes like "1a" or "as-synthesized". Empty list if none are shown.

`panel_letter` — the sub-panel letter printed on the crop ("a", "b", ...), or "" if none.

`n_curves` — how many distinct data curves or series you can see.

`render_style` — how the data is drawn, which decides how it gets traced. Look closely:
  • `lines` — continuous strokes with no individual point symbols. Diffractograms, TGA
    traces and most spectra are drawn this way.
  • `markers` — discrete symbols (filled or open circles, squares, triangles, diamonds)
    with no line through them. Common for isotherms and magnetic data.
  • `markers_joined_by_lines` — symbols *and* a line connecting them. If you can see both,
    choose this rather than guessing which dominates.
  • `mixed` — different series in the same panel are drawn differently, e.g. an
    experimental scatter with a fitted line through it.
Judge by what the ink looks like, not by what the plot type usually is.

Answer for every panel you are given, keyed by the id printed before each image."""


def _b64(rgb: np.ndarray, max_px: int = 1100) -> str:
    h, w = rgb.shape[:2]
    if max(h, w) > max_px:
        s = max_px / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def classify(items: list[dict], client=None, batch: int = 5,
             model: str = MODEL) -> dict:
    """items: [{id, rgb, caption, label}] -> {id: classification dict}"""
    import anthropic
    client = client or anthropic.Anthropic()
    out: dict[str, dict] = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        imgs = [_b64(it["rgb"]) for it in chunk]
        # sha1, not hash(): Python randomises string hashing per process, so hash()
        # would silently miss the cache on every new run
        ck = llmcache.key("classify3", model, [it["id"] for it in chunk],
                          [hashlib.sha1(b.encode()).hexdigest() for b in imgs],
                          [(it.get("caption") or "")[:700] for it in chunk])
        hit = llmcache.get(ck)
        if hit is not None:
            for p in hit.get("panels", []):
                out[str(p.get("id"))] = p
            continue
        content: list = [{"type": "text", "text": INSTRUCTIONS}]
        for n, it in enumerate(chunk):
            cap = (it.get("caption") or "").strip().replace("\n", " ")[:700]
            content.append({"type": "text",
                            "text": f"\n--- panel id: {it['id']} ---\n"
                                    f"figure: {it.get('label', '?')}\n"
                                    f"caption: {cap or '(none available)'}"})
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": "image/png",
                                       "data": imgs[n]}})
        resp = client.messages.create(
            model=model, max_tokens=4000,
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        if resp.stop_reason == "refusal":
            continue
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        llmcache.put(ck, data)
        for p in data.get("panels", []):
            out[str(p.get("id"))] = p
    return out


def is_target(cls: dict, only: set | None = None) -> bool:
    """Should this panel be digitised?"""
    if not cls or not cls.get("digitizable"):
        return False
    if cls.get("category") in NON_PLOT:
        return False
    if only and cls.get("category") not in only:
        return False
    return True
