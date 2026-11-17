#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e .
optionlab demo --n-paths 100000 --seed 42 --out results/demo
