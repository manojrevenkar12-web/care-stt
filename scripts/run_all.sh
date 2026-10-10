#!/usr/bin/env bash
# Full benchmark: dev transcripts (for tuning) then the test grid.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/transcribe.py --split dev  --models tiny-int8 base-int8 --conditions clean babble_10
python scripts/transcribe.py --split test --models tiny-int8 base-int8 small-int8 --conditions all
python scripts/transcribe.py --split test --models tiny-fp32 base-fp32 small-fp32 --conditions clean
