#!/usr/bin/env bash
# Upload this repo's training code to Kaggle and start a GPU training run.
# Run from training/:  kaggle/push.sh
# Needs ~/.kaggle/kaggle.json and the data dataset uploaded once (see training/README.md).
set -euo pipefail

kaggle() { uv run --with kaggle --with six kaggle "$@"; }

stage=data/kaggle_code
rm -rf "$stage" && mkdir -p "$stage"
cp -R clean_sweep_data configs "$stage/"
rm -rf "$stage/clean_sweep_data/__pycache__"
cat > "$stage/dataset-metadata.json" <<'EOF'
{"title": "Clean Sweep Code", "id": "audrey518/clean-sweep-code", "licenses": [{"name": "other"}]}
EOF

if kaggle datasets status audrey518/clean-sweep-code >/dev/null 2>&1; then
  kaggle datasets version -p "$stage" --dir-mode zip -m "code update"
else
  kaggle datasets create -p "$stage" --dir-mode zip
fi

# A just-uploaded dataset can't be attached until Kaggle finishes processing it.
for dataset in audrey518/clean-sweep-litter audrey518/clean-sweep-code; do
  until [ "$(kaggle datasets status "$dataset" 2>/dev/null)" = "ready" ]; do
    echo "waiting for $dataset to be ready..."; sleep 20
  done
done

kaggle kernels push -p kaggle
echo "Started. Check with: uv run --with kaggle --with six kaggle kernels status audrey518/clean-sweep-train"
