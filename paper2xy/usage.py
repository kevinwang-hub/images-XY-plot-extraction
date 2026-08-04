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

# Claude Opus 5, USD per million tokens.
IN_PER_M = 5.0
OUT_PER_M = 25.0
CACHE_WRITE_PER_M = 6.25
CACHE_READ_PER_M = 0.50

_lock = threading.Lock()
_tot = dict(input=0, output=0, cache_write=0, cache_read=0, calls=0)
LIMIT = float(os.environ.get("PAPER2XY_BUDGET", "0") or 0)


class BudgetExceeded(RuntimeError):
    pass


def record(resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    with _lock:
        _tot["input"] += int(getattr(u, "input_tokens", 0) or 0)
        _tot["output"] += int(getattr(u, "output_tokens", 0) or 0)
        _tot["cache_write"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        _tot["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        _tot["calls"] += 1


def cost() -> float:
    with _lock:
        return (_tot["input"] * IN_PER_M + _tot["output"] * OUT_PER_M
                + _tot["cache_write"] * CACHE_WRITE_PER_M
                + _tot["cache_read"] * CACHE_READ_PER_M) / 1e6


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
