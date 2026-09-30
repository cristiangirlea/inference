# inference

Self-hosted model serving for [hrn](https://github.com/cristiangirlea/harness): the serving
stack, model and quantization choices, and the measurements behind them. hrn reaches a model
here through an OpenAI-compatible endpoint, so the harness never depends on how it is
served.

Nothing is built yet. [PLAN.md](PLAN.md) is the work queue; its first phase collects the
facts (hardware, workloads, quality floor) that decide whether a local model is worth
running at all.
