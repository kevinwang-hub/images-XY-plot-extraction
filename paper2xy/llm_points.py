"""Ask the model to read a scatter plot's data points directly.

This is the thing the rest of the pipeline deliberately does not do. Everywhere else the
division is that models are asked what things *are* and pixels are asked where things
*are*, on the grounds -- stated by the source paper and repeated in our own notes -- that
vision models return coordinates that are roughly right and precisely wrong.

That is a claim, and claims should be measured. So this reader exists to be compared
against the pixel readers on the same panels, scored the same way: map what it returns
back into the image and ask how many of its points land on ink.

It is given every advantage. It sees the panel at full resolution, it is told the axis
ranges our OCR already recovered, and it answers in a schema that forces one list of
points per series. If it still misses, the miss is the model's and not the prompt's.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os

import cv2
import numpy as np

from . import llmcache, usage

MODEL = os.environ.get("PAPER2XY_MODEL", "claude-opus-5")

SCHEMA = {
    "type": "object",
    "properties": {
        "x_min": {"type": "number"},
        "x_max": {"type": "number"},
        "y_min": {"type": "number"},
        "y_max": {"type": "number"},
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "marker": {"type": "string"},
                    "color": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "number"},
                                           "y": {"type": "number"}},
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["label", "marker", "color", "points"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["x_min", "x_max", "y_min", "y_max", "series"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are reading a scatter plot from a chemistry paper and returning its
data points as numbers.

Each plotted symbol is one measurement. Return one entry per series, and inside it one
{x, y} pair per symbol you can see, in the axis units printed on the plot — not in
pixels, and not normalised.

First report the value at each edge of the plot box: `x_min` at its left edge, `x_max` at
its right, `y_min` at its bottom, `y_max` at its top. These are the values *at the box
edges*, which are not always the outermost tick labels — read where the box actually
falls relative to the ticks.

Rules:
- Read the axis scales from the tick labels and use them. If an axis is logarithmic, say
  so in the series label and still return the values in axis units.
- A series is a set of symbols sharing one colour AND one shape. Filled and open symbols
  of the same colour are different series, as are circles and triangles of one colour.
- Return every symbol you can distinguish, in increasing x. Do not thin, round, smooth or
  interpolate them, and do not add points between symbols.
- Ignore legend keys, axis ticks, grid lines and anything drawn inside an inset.
- `marker` is a short description like "filled circle" or "open square"; `color` is a
  plain colour name.

Accuracy of the values matters more than how many you return."""


def read_points(rgb: np.ndarray, axis: dict | None = None, caption: str = "",
                client=None, model: str = MODEL, effort: str = "high") -> dict:
    """Returns {"series": [{label, marker, color, points: [[x, y], ...]}]}."""
    import anthropic

    client = client or anthropic.Anthropic()
    h, w = rgb.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    img = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf.tobytes()).decode()

    hint = ""
    if axis:
        xr, yr = axis.get("x_range"), axis.get("y_range")
        if xr:
            hint += f"\nThe x axis runs from {xr[0]:.4g} to {xr[1]:.4g}"
            hint += f" ({axis.get('x_axis_label') or 'unlabelled'})."
        if yr:
            hint += f"\nThe y axis runs from {yr[0]:.4g} to {yr[1]:.4g}"
            hint += f" ({axis.get('y_axis_label') or 'unlabelled'})."
    if caption:
        hint += f"\nCaption: {caption.strip()[:500]}"

    ck = llmcache.key("llmpoints1", model, effort,
                      hashlib.sha1(b64.encode()).hexdigest(), hint)
    hit = llmcache.get(ck)
    if hit is not None:
        return hit

    resp = client.messages.create(
        model=model, max_tokens=16000,
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTIONS + hint},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": b64}},
        ]}],
    )
    usage.record(resp)
    if resp.stop_reason == "refusal":
        return {"series": []}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"series": []}
    llmcache.put(ck, data)
    return data
