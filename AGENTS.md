# inference

Self-hosted model serving for hrn (the owner's agent harness, `cristiangirlea/harness`): the
serving stack, model and quantization choices, and the measurements behind them. hrn reaches a
model here only through an OpenAI-compatible endpoint, so the harness never depends on how it is
served. Today it is llama.cpp's `llama-server` (Vulkan) on the owner's Windows PC.

`PLAN.md` is the work queue and the record: phases, hardware facts, every measured number and the
choice it justified. Read the phase you are working on before changing anything.

## Layout

| Path | Holds |
|---|---|
| `PLAN.md` | Phases 0-2, the measurement tables, what is left out on purpose and why. |
| `scripts/serve.sh` | Launches the Phase 1 Gemma 4 26B server on `127.0.0.1:8081` with the measured settings (`N_CPU_MOE`, `PORT` override). |
| `scripts/throughput.py` | Total decode throughput of a running server at 1, 4 and 8 concurrent requests. Standard library only. |
| `.bin/`, `.models/` | Git-ignored, only in the owner's main checkout: llama.cpp builds (`.bin/vulkan`, `.bin/rocm-10.0`), GGUF files, `SHA256SUMS`, server logs. A worktree has neither, so `serve.sh` cannot run there. |

Related repos: launch settings for the multi-model sweep live in `cristiangirlea/llm-agent-benchmark`
(`sweeps/local-7900xt.json`); quality runs in `hrn-evals`; the cluster in `infra`. Agent loop,
eval suites and cluster manifests do not belong here.

## Checks

There is no CI. Before a PR (`/check` runs these), with Python 3 and nothing installed:

```bash
bash -n scripts/serve.sh
python -m py_compile scripts/throughput.py
git diff --check origin/main
```

Never download model weights, unpack llama.cpp builds or start `llama-server` as part of a check.
Those are measurement runs: the owner's GPU is shared with their desktop and the numbers depend
on what else is running.

## Measurements are evidence

- Never change a recorded number, table row or "Chosen:" line in `PLAN.md` without re-measuring.
  A new run gets its own dated section or row, with the model file, llama.cpp build, backend,
  flags and tool (`llama-bench`, `scripts/throughput.py`, `bench sweep`) that produced it.
- Keep each choice next to the numbers that justify it. If a script's flags change, the
  comment in it and the matching `PLAN.md` text change in the same PR, with the measurement.
- Facts that decide every sizing: RX 7900 XT, 20 GiB VRAM of which about 13 GiB is free with
  the desktop running; always `-dev Vulkan0` (Vulkan1 is the CPU's integrated GPU); ROCm on
  Windows sees no device. Measure with nothing on the CPU first; move expert layers off
  (`--n-cpu-moe`) only when shared memory grows past the desktop's 2.5 GiB baseline.
- A model file is identified by name and SHA-256 (`.models/SHA256SUMS`). Check its licence
  before proposing it; weights and binaries are never committed (`.gitignore`).

## Working rules

- One branch per change, one PR; the owner merges. Never `gh pr merge`, never push to `main`,
  never tag, release or deploy: hand the owner the command.
- Commit messages: plain sentences with a lowercase area first, as in the log
  (`plan: six smaller models measured with quality, ...`). No conventional-commit prefixes, no
  emoji, no AI attribution lines.
- Any CI added later runs on the owner's runners (`vars.CI_RUNNER`, `vars.CI_RUNNER_WINDOWS`);
  this private repo spends no GitHub Actions minutes. Never add a hosted-runner fallback.
- Secrets (`.env`, API keys, Hugging Face tokens) are never read, printed or committed.
- Small diffs: don't restyle what you didn't change. Not sure? Ask, with one proposal.
