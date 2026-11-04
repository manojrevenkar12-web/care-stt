"""v2 release gates over results/v2/metrics.json and results/v2/gate.json (gates_v2.yaml).

Two kinds of gate: the release candidate must be at least as good as stock Whisper on every check
that matters in the chart (relative gates), and must stay above absolute floors set just under its
first measured values, so a later change that loses ground fails CI.

  python -m care_stt.gates_v2            (exit 1 on any failure)
"""
from __future__ import annotations

import json
import sys

import yaml

from .config import ROOT


def value(metrics: dict, system: str, test_set: str, condition: str, metric: str):
    return metrics[system][test_set][condition][metric]


def check(metrics: dict, gate: dict | None, spec: dict) -> list[dict]:
    out = []
    for g in spec.get("relative", []):
        try:
            a = value(metrics, g["candidate"], g["set"], g.get("condition", "all"), g["metric"])
            b = value(metrics, g["baseline"], g["set"], g.get("condition", "all"), g["metric"])
        except KeyError:
            out.append({**g, "passed": False, "detail": "metric missing"})
            continue
        better = a <= b if g.get("lower_is_better") else a >= b
        out.append({**g, "passed": better, "detail": f"{g['candidate']} {a} vs {g['baseline']} {b}"})
    for g in spec.get("absolute", []):
        try:
            v = gate[g["metric"]] if g.get("source") == "gate" else value(metrics, g["system"], g["set"], g.get("condition", "all"), g["metric"])
        except (KeyError, TypeError):
            out.append({**g, "passed": False, "detail": "metric missing"})
            continue
        ok = ("min" not in g or v >= g["min"]) and ("max" not in g or v <= g["max"])
        bound = f">= {g['min']}" if "min" in g else f"<= {g['max']}"
        out.append({**g, "passed": ok, "detail": f"{v} (needs {bound})"})
    return out


def main() -> int:
    metrics = json.loads((ROOT / "results/v2/metrics.json").read_text())
    gp = ROOT / "results/v2/gate.json"
    gate = json.loads(gp.read_text()) if gp.exists() else None
    spec = yaml.safe_load((ROOT / "gates_v2.yaml").read_text())
    res = check(metrics, gate, spec)
    for r in res:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['name']}: {r['detail']}")
    failed = [r for r in res if not r["passed"]]
    print(f"{len(res) - len(failed)}/{len(res)} v2 gates passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
