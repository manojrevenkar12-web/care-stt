"""Release gates: block a model/config change that regresses what matters.

gates.yaml lists checks like

  - name: numbers survive clean audio
    model: base-int8
    condition: clean
    metric: raw.number_utt_acc
    min: 0.80

Run: python -m care_stt.gates results/metrics.json gates.yaml   (exit 1 on failure)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def lookup(results: dict, model: str, condition: str, metric: str):
    node = results["test"][model][condition]
    for k in metric.split("."):
        node = node[k]
    return node


def check(results: dict, gates: list[dict]) -> list[dict]:
    out = []
    for g in gates:
        try:
            v = lookup(results, g["model"], g["condition"], g["metric"])
        except KeyError:
            out.append({**g, "value": None, "passed": False, "reason": "metric missing"})
            continue
        ok = True
        if "min" in g and not v >= g["min"]:
            ok = False
        if "max" in g and not v <= g["max"]:
            ok = False
        out.append({**g, "value": v, "passed": ok})
    return out


def main(argv: list[str]) -> int:
    metrics = json.loads(Path(argv[0]).read_text())
    gates = yaml.safe_load(Path(argv[1]).read_text())["gates"]
    res = check(metrics, gates)
    for r in res:
        bound = f">= {r['min']}" if "min" in r else f"<= {r['max']}"
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['name']}: {r['model']}/{r['condition']} {r['metric']} = {r['value']} (needs {bound})")
    failed = [r for r in res if not r["passed"]]
    print(f"{len(res) - len(failed)}/{len(res)} gates passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["results/metrics.json", "gates.yaml"]))
