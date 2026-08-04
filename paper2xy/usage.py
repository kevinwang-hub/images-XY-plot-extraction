"""Token and cost accounting for the model calls, with a hard ceiling.

Every API response passes through `record`, so the running cost is a measurement rather
than an estimate. `check` raises once the ceiling is reached, which stops a long run at
the limit instead of discovering the overspend afterwards.

Cached calls cost nothing and are not counted, so a re-run after an engine change is
free -- which is also why the figure reported at the end is the true spend for the run,
not the spend the work would have taken from cold.
"""
from __future__ import annotations

import os
import threading

# USD per million tokens, by model. Fable 5 is twice Opus 5, so a run that mixes them
# cannot be costed at one rate -- which is the whole reason this is a table.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
IN_PER_M, OUT_PER_M = PRICES["claude-opus-5"]
CACHE_WRITE_PER_M = 6.25
CACHE_READ_PER_M = 0.50

_lock = threading.Lock()
_tot = dict(input=0, output=0, cache_write=0, cache_read=0, calls=0)
_by_model: dict = {}
LIMIT = float(os.environ.get("PAPER2XY_BUDGET", "0") or 0)


class BudgetExceeded(RuntimeError):
    pass


def record(resp, model: str | None = None) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    model = model or getattr(resp, "model", None) or "claude-opus-5"
    with _lock:
        m = _by_model.setdefault(model, dict(input=0, output=0, calls=0))
        m["input"] += int(getattr(u, "input_tokens", 0) or 0)
        m["output"] += int(getattr(u, "output_tokens", 0) or 0)
        m["calls"] += 1
        _tot["input"] += int(getattr(u, "input_tokens", 0) or 0)
        _tot["output"] += int(getattr(u, "output_tokens", 0) or 0)
        _tot["cache_write"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        _tot["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        _tot["calls"] += 1


def cost() -> float:
    with _lock:
        total = (_tot["cache_write"] * CACHE_WRITE_PER_M
                 + _tot["cache_read"] * CACHE_READ_PER_M)
        for name, m in _by_model.items():
            # an unknown model is priced at the Opus rate rather than at zero: a silent
            # under-count is the one failure a budget ceiling must not have
            pin, pout = PRICES.get(name, PRICES["claude-opus-5"])
            total += m["input"] * pin + m["output"] * pout
        return total / 1e6


def by_model() -> dict:
    with _lock:
        out = {}
        for name, m in _by_model.items():
            pin, pout = PRICES.get(name, PRICES["claude-opus-5"])
            out[name] = dict(m, cost_usd=(m["input"] * pin + m["output"] * pout) / 1e6)
        return out


def totals() -> dict:
    with _lock:
        d = dict(_tot)
    d["cost_usd"] = cost()
    return d


def check() -> None:
    """Raise if the ceiling has been reached. Called between papers, not mid-paper, so a
    run stops on a whole paper rather than half of one."""
    if LIMIT and cost() >= LIMIT:
        raise BudgetExceeded(
            f"spent ${cost():.2f} of the ${LIMIT:.2f} ceiling after {_tot['calls']} calls")


def summary() -> str:
    d = totals()
    return (f"{d['calls']} model calls · {d['input']:,} in / {d['output']:,} out tokens"
            f" · ${d['cost_usd']:.2f}")
