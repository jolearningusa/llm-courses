#!/usr/bin/env bash
# Lab 14, Part C (folder lab3_google_harness2): quickstart for Google Research's Harness² (recursive agent harnessing).
#
#   Paper: Xu et al., "Harness²: Recursive Agent Harnessing for an Open World", arXiv:2610.14224
#   Repo:  https://github.com/google-research/harness2   (Apache-2.0, "not an officially supported Google product")
#
# Commands below follow the upstream README as of Oct 2026.
#
# Usage:
#   ./run_harness2_demo.sh          # offline demo: harness2-demo with a recursion budget of k=2
#   ./run_harness2_demo.sh --lab    # also run one LAB benchmark task end to end (needs model credentials)
set -euo pipefail

cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }

if [ ! -d harness2 ]; then
  git clone https://github.com/google-research/harness2.git
fi
cd harness2

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e '.[render]'

# --- 1. Demo: task-level recursion (propose -> probe -> compose) with k=2 ---
harness2-demo --output runs/demo --k 2
echo "Demo outputs written to $(pwd)/runs/demo"

# --- 2. Optional: one real benchmark task ---
if [ "${1:-}" = "--lab" ]; then
  # Model-backed runs read credentials from .env and Google Cloud ADC.
  [ -f .env ] || cp .env.example .env
  echo "Edit harness2/.env, then make sure ADC is set: gcloud auth application-default login"

  harness2-setup lab
  harness2-check --bench lab
  harness2 --bench lab --k 1 --mode parallel --only-domains antitrust-competition --max-tasks 1
  harness2-judge --bench lab --k 1 --mode parallel
  harness2-report --bench lab --k 1 --mode parallel
fi

# Useful flags (see `harness2 --help`):
#   --dry-run          preview task selection without model calls
#   --substrate cx     use Codex instead of the default OpenCode executor
#   HARNESS2_RUNS_DIR  set a fresh runs dir whenever you change models or task selection
