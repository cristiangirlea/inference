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

### Phase 0 answers (2026-10-01)

1. **This PC serves.** AMD Radeon RX 7900 XT (gfx1100, 20 GiB VRAM), Ryzen 9 9950X (16 cores),
   64 GB RAM, Windows 11, driver 32.0.31041. **The desktop already holds about 6.6 GiB of that
   VRAM** with no model loaded (compositor, Chrome, Parsec, Discord, Steam), so about 13 GiB is
   really free. Every sizing decision below starts from 13, not 20.
   - Vulkan (`llama-b11321-bin-win-vulkan-x64`) sees the card and works natively on Windows.
   - ROCm for Windows (`llama-b11321-bin-win-rocm-10.0-x64`) lists **no devices**, with its
     bundled HIP runtime and with the driver's own (10.0.3679). gfx1100 is compiled in, so the
     cause is the runtime and driver pairing, not the target. Not pursued: Vulkan is good enough
     (below), and ROCm belongs with a Linux host if one comes.
   - Vulkan also lists the CPU's integrated GPU (`Vulkan1`). Always pass `-dev Vulkan0`, or the
     model may be split onto the iGPU.
2. **First workload: `datagen`** (bulk synthetic data for evals and fixtures). There is no
   hosted API key yet, so there is no monthly cost to compare against; the saving is "possible
   at all" against "not possible".
3. **Quality floor:** still open. It needs an `hrn-evals` variant that points at this endpoint
   (through hrn's opencode backend), and a datagen calibration set (datagen P2).

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

### Phase 1 measurements (2026-10-01)

Model: `ggml-org/gemma-4-26B-A4B-it-GGUF`, file `gemma-4-26B-A4B-it-Q4_0.gguf` (13.6 GiB,
Apache-2.0, mixture of experts with about 4B parameters active per token, SHA-256 in
`.models/SHA256SUMS`). llama.cpp build b11321, Vulkan. Launch: `scripts/serve.sh`.
Throughput: `scripts/throughput.py` (fixed 256-token completions, `ignore_eos`).

**Everything on the GPU spills.** 13.6 GiB of weights into about 13 GiB free: Windows pages the
rest into shared system memory (observed 6.1 GiB shared), and every token then reads it over
PCIe. Moving some expert layers to the CPU on purpose (`--n-cpu-moe`) is much faster than
letting the driver spill at random:

`llama-bench`, prefill 2048 tokens / decode 128, flash attention on, 16 threads:

| expert layers on CPU | prefill t/s | decode t/s |
|---:|---:|---:|
| 0 (all on GPU, spilling) | 286 | 45.5 |
| 4 | 312 | 70.3 |
| 5 | 289 | 66.8 |
| **6** | **1055** | **65.0** |
| 8 | 909 | 58.1 |
| 12 | 675 | 48.2 |
| 18 | 512 | 38.1 |

Below 6 the prefill buffers still spill (the cliff between 5 and 6). The server also holds the
KV cache, so its best value is higher. `llama-server`, 4 slots, 32k context (8k per slot), KV
cache q8_0:

| expert layers on CPU | 1 request, t/s | 4 concurrent, total t/s | VRAM dedicated / shared |
|---:|---:|---:|---|
| 0, f16 KV, 64k context (first try) | 30.5 | 64.2 | 19.0 / 6.1 GiB (spilling) |
| 6 | 43.7 | 82.6 | n/a |
| **8** | **44–51** | **97–114** | 18.5 / 2.5 GiB (2.5 is the desktop's own) |
| 10 | 40.4 | 85.5 | 17.8 / 2.5 GiB |
| 12 | 42.8 | 81.8 | 17.1 / 2.5 GiB |

**Chosen: `--n-cpu-moe 8`, 4 slots, q8_0 KV.** Four concurrent requests give about twice the
total throughput of one; 8 concurrent gain nothing more with 4 slots. These numbers move with
whatever else is using the GPU: Parsec streaming this desktop showed about 19% video-encode
load during the runs, and closing the browser frees about 1.6 GiB.

What that means for `datagen`: a QA record of about 150 output tokens at about 100 t/s total is
roughly 0.6 records per second, about 2,000 generated records an hour before filtering.

Behaviour that clients must know:

- **Gemma 4 thinks before answering.** With a 400-token limit, a structured-output request
  came back with empty content: the whole budget went to reasoning (`finish_reason: length`).
  Send `"chat_template_kwargs": {"enable_thinking": false}` for generation work, or give a
  budget that covers the reasoning.
- **JSON Schema output works** (`response_format` with `json_schema`): valid, schema-shaped
  output on every try.
- **Same seed, same output** was observed for two sequential identical requests. It is not
  promised under concurrency: requests batched together can change the numerics. Clients that
  need reproducibility (datagen) should rely on their own cache, not on the seed.

Not measured yet: quality (the `hrn-evals` pass rate next to the hosted models), Q8_0 against
Q4_0 (Q8_0 is 26.9 GiB, far over the free VRAM, so the comparison needs the CPU or a smaller
model), and a judge model from another family for datagen (planned: Qwen3.6-35B-A3B, 20.4 GiB at
Q4_K_M, so more of it on the CPU).

## Phase 2: in the cluster

A deployment for the local cluster in `infra` (GPU access, model files on a persistent
volume, a readiness check that sends a real request), and a variant in `hrn-evals` that
points at it.

On this PC, containers in k3d (Docker Desktop, WSL 2) cannot use the 7900 XT, and the Vulkan
path above is a Windows process. Until there is a Linux host, the server stays on Windows and the
cluster reaches it as an external service at `host.k3d.internal:8081`: a Service without a
selector, plus the readiness check. Moving the model into the cluster comes with the Linux host.

## Left out on purpose

Fine-tuning and distillation: without a measured gap that prompting, retrieval and a better
hosted model cannot close, they cost more than they return. Speculative decoding: a
throughput optimisation to reach for after Phase 1's numbers show decode speed is the
bottleneck, not before.
