#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_dir="$repo_root/yolo26/cpp/detect"
build_dir="$task_dir/build"

image_dir="${1:-../images}"
model="${2:-yolo26m}"
plan_file="${3:-}"

mkdir -p "$build_dir"
cd "$build_dir"

cmake ..
make -j"$(nproc)"

cmd=( "./detect" "$image_dir" "$model" )
if [[ -n "$plan_file" ]]; then
  cmd+=( "$plan_file" )
fi

"${cmd[@]}"
