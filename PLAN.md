# inference: self-hosted model serving for hrn

This file is the work queue for the `inference` repository. Written 2026-09-30. Nothing here
is built yet, on purpose: the first phase answers questions that decide everything after it.

## Why this repository exists

hrn runs on hosted models today: Claude through the API or Claude Code, and Codex. That is
the right default. A local model earns its place only for work where one of these matters
more than quality: cost at volume (evals repeated nightly, bulk classification), data that
must not leave the machine, or availability when a provider is down or a subscription limit
is reached. This repository is where serving such a model lives, so the harness never has to
know how a model is served, only where its endpoint is.

## Boundaries

- **Inside this repository:** the serving stack, model choices and their quantization,
  launch configuration, deployment manifests for the local cluster, and the measurements
  that justify each choice.
- **Not here:** the agent loop, tools, budgets and records (hrn); the cluster itself
  (`infra`); eval suites (`hrn-evals`, which gains a local-model variant once a model runs).
- **Contract with hrn:** an OpenAI-compatible HTTP endpoint. opencode already speaks it, so
  hrn can use a local model through its opencode backend with no harness change; a native
  hrn backend comes only if that path proves too limited.

## Phase 0: facts first (no code)

Answer and record here:

1. Which machine serves, and which GPU and how much VRAM. The model size and the
   quantization follow from that. AMD cards need ROCm, and ROCm under WSL on Windows is
   limited, so a Linux host is the likely target.
2. Which workloads move off hosted models, with their current monthly cost from
   `hrn report`, so the saving can be compared with the hardware and power it takes.
3. The quality floor: which `hrn-evals` suites a local model must pass before it is used
   for anything.

## Phase 1: one model, measured

- Serve one model with an OpenAI-compatible server: `llama.cpp`'s `llama-server` for GGUF
  quantizations (simplest, runs on ROCm, Vulkan and CPU), or vLLM where the hardware
  supports it (continuous batching and paged attention for throughput when several agents
  call at once).
- Measure, and write the numbers here: time to first token (prefill) and tokens per second
  (decode) at the context lengths hrn uses, memory at load and under load, and the
  `hrn-evals` pass rates next to the hosted models.
- Compare two quantizations of the same model (for example Q8_0 against Q4_K_M) on the same
  suites. A quantization is accepted only when the suite results say the quality loss does
  not matter for the workload; tool-calling and structured-output tasks usually suffer
  first.

## Phase 2: in the cluster

A deployment for the local cluster in `infra` (GPU access, model files on a persistent
volume, a readiness check that sends a real request), and a variant in `hrn-evals` that
points at it.

## Left out on purpose

Fine-tuning and distillation: without a measured gap that prompting, retrieval and a better
hosted model cannot close, they cost more than they return. Speculative decoding: a
throughput optimisation to reach for after Phase 1's numbers show decode speed is the
bottleneck, not before.
