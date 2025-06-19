#!/bin/bash
set -e

PROJECT_ROOT=$(pwd)
run_id_file="wandb_runids/wandb_run_id.txt"
run_id_path="$PROJECT_ROOT/$run_id_file"

run_id=$(cat "$run_id_path")
cd "$PROJECT_ROOT/wandb"

run_folder=$(find . -maxdepth 1 -type d -name "*${run_id}" | head -n 1)

if [ -n "$run_folder" ]; then
    echo "Found W&B run folder: $run_folder"
    wandb sync "$run_folder"
else
    echo "Could not find run folder for run_id: $run_id"
fi
