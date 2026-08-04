"""Recover what a curve actually is, from the paper's own words.

Figures label their curves the way the authors think, not the way a database needs:
"1a", "complex 2", "as-synthesized", or nothing at all. The compound behind "1" is
defined once, in the abstract or the synthesis section, and never repeated on the axes.
So the caption alone is not enough — the resolution has to read the surrounding text.

This is a text-only task, which is why it is separated from the vision classifier: the
model gets the caption, the paragraphs that cite the figure, and the paper's opening
(where compound numbers are almost always defined), and returns a name per series.
"""
from __future__ import annotations

import json
import os
import re

from . import llmcache

MODEL = os.environ.get("PAPER2XY_MODEL", "claude-opus-5")

SCHEMA = {
    "type": "object",
    "properties": {
        "material": {"type": "string"},
        "technique_detail": {"type": "string"},
        "panel_subject": {"type": "string"},
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label_in_figure": {"type": "string"},
                    "resolved_name": {"type": "string"},
                    "role": {"type": "string",
                             "enum": ["experimental", "simulated", "calculated",
                                      "reference", "background", "unknown"]},
                    "conditions": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "string",
                                   "enum": ["high", "medium", "low"]},
                },
                "required": ["label_in_figure", "resolved_name", "role", "conditions",
                             "evidence", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["material", "technique_detail", "panel_subject", "series"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are reading one figure panel from an inorganic-chemistry paper and
resolving what each plotted curve represents, using the paper's own text.

The curve labels printed in a figure are usually shorthand — a bare compound number
("1", "2a"), a condition ("as-synthesized", "after 5 cycles"), or nothing. The compound
those numbers refer to is defined elsewhere in the paper, typically in the abstract or
the synthesis section. Your job is to connect them.

Rules:
- `resolved_name`: the chemical identity the label refers to — a formula or compound name
  as the paper writes it, e.g. "[Cd(succ)(H2O)2]" or "MOF-5". If the label is a bare
  number or code, resolve it. If the text does not define it, repeat the label verbatim
  and set confidence "low". Never invent a formula.
- `evidence`: the phrase from the provided text that justifies the resolution, quoted.
  Leave "" if you could not resolve it from the text.
- `conditions`: measurement conditions specific to this curve (temperature, solvent,
  cycle number, activation state), "" if none.
- `material`: the compound the whole panel is about, if it is about a single one.
- `technique_detail`: measurement conditions for the panel as a whole — adsorbate and
  temperature for an isotherm ("N2, 77 K"), heating rate for TGA, radiation for PXRD.
  Take these only from the provided text; "" if not stated.
- `panel_subject`: one short phrase describing what the panel shows.

Return one `series` entry per curve. If the number of curves you are told about exceeds
the labels available, still return that many entries, using "" for unknown labels."""


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:n]


def resolve(caption: str, context: list, series_labels: list, n_curves: int,
            category: str, paper_head: str = "", client=None,
            model: str = MODEL) -> dict:
    """Resolve one panel's series to chemical identities using the paper text."""
    import anthropic
    client = client or anthropic.Anthropic()
    ck = llmcache.key("context2", model, caption[:1200], context[:6], series_labels,
                      n_curves, category, paper_head[:2500])
    hit = llmcache.get(ck)
    if hit is not None:
        return hit

    parts = [f"PLOT TYPE: {category}",
             f"NUMBER OF CURVES DETECTED: {n_curves}",
             f"LABELS READ FROM THE FIGURE: {series_labels if series_labels else '(none)'}",
             f"\nCAPTION:\n{_clip(caption, 1200)}"]
    if paper_head:
        parts.append(f"\nPAPER OPENING (title / abstract / early text):\n"
                     f"{_clip(paper_head, 2500)}")
    if context:
        joined = "\n\n".join(_clip(c, 900) for c in context[:6])
        parts.append(f"\nPARAGRAPHS THAT CITE THIS FIGURE:\n{_clip(joined, 5000)}")

    resp = client.messages.create(
        model=model, max_tokens=3000,
        output_config={"effort": "low",
                       "format": {"type": "json_schema", "schema": SCHEMA}},
        system=INSTRUCTIONS,
        messages=[{"role": "user", "content": "\n".join(parts)}],
    )
    if resp.stop_reason == "refusal":
        return {}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    llmcache.put(ck, data)
    return data


def attach(curves: list, resolved: dict) -> None:
    """Match resolved series onto digitised curves, by label first then by order."""
    series = list(resolved.get("series", []))
    used = set()
    for c in curves:
        lab = _clip(c.get("legend", ""), 80).lower()
        hit = None
        if lab:
            for i, s in enumerate(series):
                if i in used:
                    continue
                sl = _clip(s.get("label_in_figure", ""), 80).lower()
                if sl and (sl == lab or sl in lab or lab in sl):
                    hit = i
                    break
        if hit is None:
            for i, s in enumerate(series):
                if i not in used:
                    hit = i
                    break
        if hit is None:
            continue
        used.add(hit)
        s = series[hit]
        c["resolved_name"] = s.get("resolved_name", "")
        c["role"] = s.get("role", "unknown")
        c["conditions"] = s.get("conditions", "")
        c["name_confidence"] = s.get("confidence", "low")
        c["name_evidence"] = s.get("evidence", "")
        if not c.get("legend"):
            c["legend"] = s.get("label_in_figure", "")
