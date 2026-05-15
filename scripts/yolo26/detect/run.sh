#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_dir="$repo_root/yolo26/cpp/detect"
build_dir="$task_dir/build"

mkdir -p "$build_dir"
cd "$build_dir"

cmake ..
cmake --build .

echo "yolo26/cpp/detect is still a scaffold in this repo; no runnable executable is defined yet." >&2
exit 1
