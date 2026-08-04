"""Disk cache for model calls, keyed by the exact inputs.

Re-running the pipeline after an engine change would otherwise re-pay for every
classification and every name resolution, even though neither input changed.
"""
from __future__ import annotations

import hashlib
import json
import os

DIR = os.environ.get("PAPER2XY_CACHE", "")


def key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()


def get(k: str):
    if not DIR:
        return None
    p = os.path.join(DIR, k + ".json")
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def put(k: str, value) -> None:
    if not DIR:
        return
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, k + ".json"), "w") as fh:
        json.dump(value, fh)
