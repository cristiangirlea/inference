#!/usr/bin/env bash
# Serves the Phase 1 model on this PC with the settings measured in PLAN.md (Phase 1).
# Expects llama.cpp's Windows Vulkan build unpacked in .bin/vulkan and the model in .models/
# (both git-ignored). OpenAI-compatible endpoint: http://127.0.0.1:8081/v1
#
#   -dev Vulkan0       the RX 7900 XT; Vulkan1 is the CPU's integrated GPU
#   --n-cpu-moe 8      8 expert layers on the CPU: the rest fits the ~13 GiB of VRAM the desktop
#                      leaves free, so nothing spills into shared memory
#   --parallel 4       4 slots of 8k context each; 4 concurrent requests double total throughput
#   -ctk/-ctv q8_0     KV cache at 8 bits, half the VRAM of f16
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
exec "$root/.bin/vulkan/llama-server.exe" \
  -m "$root/.models/gemma-4-26B-A4B-it-Q4_0.gguf" \
  -dev Vulkan0 -ngl 99 --n-cpu-moe "${N_CPU_MOE:-8}" -fa on -t 16 \
  -c 32768 --parallel 4 -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port "${PORT:-8081}" "$@"
