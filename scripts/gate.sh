#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
what="${1:-all}"
case "$what" in
  lint|test|all) ;;
  *) echo "scripts/gate.sh: unknown step '$what'; one of lint, test, all"; echo "GATE RED"; exit 1 ;;
esac
templates=(couch-serve)
steps=""
[[ "$what" == all || "$what" == lint ]] && steps+="ruff check . && ruff format --check . && "
[[ "$what" == all || "$what" == test ]] && steps+="python -m pytest -q && "
failed=0
for template in "${templates[@]}"; do
  echo "=== $template"
  git archive HEAD "$template" ruff.toml | MSYS_NO_PATHCONV=1 docker run --rm -i -v windlet-templates-pip:/root/.cache/pip python:3.14-slim bash -c "
    mkdir /work && cd /work && tar -xf - && cd $template &&
    pip install -q --index-url https://download.pytorch.org/whl/cpu \$(grep '^torch==' requirements.txt) &&
    pip install -q pytest==9.1.1 ruff &&
    ${steps}true"
  if [ "$?" -ne 0 ]; then echo "FAILED: $template"; failed=1; fi
done

if [ "$failed" -ne 0 ]; then echo "GATE RED"; exit 1; else echo "GATE GREEN"; fi
