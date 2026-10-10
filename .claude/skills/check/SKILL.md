---
name: check
description: Run the repository's checks (script syntax, Python compile, whitespace) before a PR
---
From the repository root, with Python 3 and nothing installed. Run in this order and stop at the
first failure:

```bash
bash -n scripts/serve.sh
python -m py_compile scripts/throughput.py
git diff --check origin/main
```

Report one line per step (passed, or the error). Never download model weights, unpack llama.cpp
builds, start `llama-server` or run `scripts/throughput.py` as part of a check: those are
measurement runs on the owner's shared GPU. If the change touches `PLAN.md`, confirm that no
recorded number changed without a new measurement behind it (`git diff origin/main -- PLAN.md`).
Never merge, never push to `main`.
