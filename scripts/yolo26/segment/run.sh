#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_dir="$repo_root/yolo26/cpp/segment"
build_dir="$task_dir/build"

image_dir="${1:-../images}"
model="${2:-yolo26m-seg}"
plan_file="${3:-}"
precision=""

usage() {
  echo "Usage: bash scripts/yolo26/segment/run.sh [image dir] [onnx file|model name] [plan file optional] [--precision fp16|fp32]"
  echo "Example: bash scripts/yolo26/segment/run.sh ../images yolo26m-seg --precision fp32"
  echo "Example: bash scripts/yolo26/segment/run.sh ../images ../onnx_model/yolo26m-seg.onnx ./yolo26m-seg.plan --precision fp16"
}

positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --precision)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      case "$2" in
        fp16|fp32) precision="$2" ;;
        *)
          usage
          exit 1
          ;;
      esac
      shift 2
      ;;
    --*)
      usage
      exit 1
      ;;
    *)
      positionals+=( "$1" )
      shift
      ;;
  esac
done

if (( ${#positionals[@]} > 3 )); then
  usage
  exit 1
fi

image_dir="${positionals[0]:-../images}"
model="${positionals[1]:-yolo26m-seg}"
plan_file="${positionals[2]:-}"

mkdir -p "$build_dir"
cd "$build_dir"

cmake ..
make -j"$(nproc)"

cmd=( "./segment" "$image_dir" "$model" )
if [[ -n "$plan_file" ]]; then
  cmd+=( "$plan_file" )
fi
if [[ -n "$precision" ]]; then
  cmd+=( "--precision" "$precision" )
fi

"${cmd[@]}"
