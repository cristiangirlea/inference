"""Total decode throughput of a running llama-server at 1, 4 and 8 concurrent requests.

Each request asks for exactly 256 tokens (ignore_eos) with prompt caching off, so runs compare.
Usage: python scripts/throughput.py [base-url]   (default http://127.0.0.1:8081)
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"


def one(i):
    body = json.dumps({
        "prompt": f"Write a long technical story number {i} about distributed systems.",
        "n_predict": 256, "ignore_eos": True, "temperature": 0.8, "seed": i, "cache_prompt": False,
    }).encode()
    request = urllib.request.Request(BASE + "/completion", body, {"Content-Type": "application/json"})
    timings = json.load(urllib.request.urlopen(request, timeout=600))["timings"]
    return timings["predicted_n"], timings["predicted_per_second"]


for n in (1, 4, 8):
    start = time.time()
    with cf.ThreadPoolExecutor(n) as pool:
        results = list(pool.map(one, range(n)))
    wall = time.time() - start
    tokens = sum(r[0] for r in results)
    print(f"concurrent={n}: {tokens} tokens in {wall:.1f}s -> total {tokens / wall:.1f} t/s, "
          f"per request {sum(r[1] for r in results) / n:.1f} t/s")
