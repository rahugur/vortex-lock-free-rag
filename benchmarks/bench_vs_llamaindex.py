#!/usr/bin/env python3
"""
Head-to-head benchmark: Vortex vs LlamaIndex RAG pipeline.

Both systems use the same mock embedding + LLM servers,
so the gap is entirely orchestration and retrieval overhead.

Usage:
    # Terminal 1: Start Vortex server
    ./build/vortex-server --serve --embed-url http://localhost:9998 -k unused

    # Terminal 2: Start mock servers
    python3 benchmarks/mock_servers.py --embed-port 9998 --llm-port 9999

    # Terminal 3: Run comparison
    python3 benchmarks/bench_vs_llamaindex.py \
        --vortex-url http://localhost:8081 \
        --embed-port 9998 --llm-port 9999 \
        --queries 100 --concurrent 10
"""

import argparse
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError

def query_vortex(url, query_text):
    """Send a RAG query to Vortex server."""
    data = json.dumps({"query": query_text}).encode()
    req = Request(f"{url}/v1/query", data=data,
                  headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        elapsed = time.perf_counter() - start
        return {
            "elapsed_ms": elapsed * 1000,
            "search_us": result.get("metrics", {}).get("search_us", 0),
            "overhead_us": result.get("metrics", {}).get("pipeline_overhead_us", 0),
        }
    except URLError as e:
        return {"elapsed_ms": -1, "error": str(e)}


def query_llamaindex(index, query_text):
    """Run a RAG query through LlamaIndex."""
    start = time.perf_counter()
    response = index.as_query_engine().query(query_text)
    elapsed = time.perf_counter() - start
    return {"elapsed_ms": elapsed * 1000}


def run_benchmark(fn, queries, concurrent):
    """Run benchmark with given concurrency."""
    latencies = []
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {executor.submit(fn, q): q for q in queries}
        for future in as_completed(futures):
            result = future.result()
            if result.get("elapsed_ms", -1) > 0:
                latencies.append(result["elapsed_ms"])
            else:
                errors += 1

    if not latencies:
        return {"error": "All queries failed"}

    latencies.sort()
    return {
        "count": len(latencies),
        "errors": errors,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p99_ms": latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0],
        "qps": len(latencies) / (sum(latencies) / 1000 / concurrent),
    }


def main():
    parser = argparse.ArgumentParser(description="Vortex vs LlamaIndex benchmark")
    parser.add_argument("--vortex-url", default="http://localhost:8081")
    parser.add_argument("--embed-port", type=int, default=9998)
    parser.add_argument("--llm-port", type=int, default=9999)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--concurrent", type=int, default=10)
    args = parser.parse_args()

    queries = [f"Query about topic {i % 50}" for i in range(args.queries)]

    print("=" * 60)
    print("Vortex vs LlamaIndex RAG Benchmark")
    print("=" * 60)
    print(f"Queries: {args.queries}, Concurrent: {args.concurrent}")
    print()

    # --- Vortex ---
    print("--- Vortex ---")
    try:
        vortex_results = run_benchmark(
            lambda q: query_vortex(args.vortex_url, q),
            queries, args.concurrent)
        print(f"  Mean:   {vortex_results['mean_ms']:.1f} ms")
        print(f"  Median: {vortex_results['median_ms']:.1f} ms")
        print(f"  p99:    {vortex_results['p99_ms']:.1f} ms")
        print(f"  QPS:    {vortex_results['qps']:.0f}")
        if vortex_results['errors']:
            print(f"  Errors: {vortex_results['errors']}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  (Is the Vortex server running?)")
        vortex_results = None

    # --- LlamaIndex ---
    print("\n--- LlamaIndex ---")
    try:
        from llama_index.core import VectorStoreIndex, Document, Settings
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.llms.openai_like import OpenAILike
        from llama_index.embeddings.openai import OpenAIEmbedding

        # Configure to use mock servers
        Settings.llm = OpenAILike(
            api_base=f"http://localhost:{args.llm_port}/v1",
            api_key="test", model="mock", timeout=30)
        Settings.embed_model = OpenAIEmbedding(
            api_base=f"http://localhost:{args.embed_port}/v1",
            api_key="test", model_name="mock")

        # Build index with same documents
        docs = [Document(text=f"Document about topic {i}. Some content here.")
                for i in range(100)]
        index = VectorStoreIndex.from_documents(docs)

        llamaindex_results = run_benchmark(
            lambda q: query_llamaindex(index, q),
            queries, args.concurrent)
        print(f"  Mean:   {llamaindex_results['mean_ms']:.1f} ms")
        print(f"  Median: {llamaindex_results['median_ms']:.1f} ms")
        print(f"  p99:    {llamaindex_results['p99_ms']:.1f} ms")
        print(f"  QPS:    {llamaindex_results['qps']:.0f}")
    except ImportError:
        print("  LlamaIndex not installed. pip install llama-index")
        llamaindex_results = None
    except Exception as e:
        print(f"  Error: {e}")
        llamaindex_results = None

    # --- Comparison ---
    if vortex_results and llamaindex_results:
        print("\n--- Comparison ---")
        speedup = llamaindex_results["mean_ms"] / vortex_results["mean_ms"]
        qps_ratio = vortex_results["qps"] / llamaindex_results["qps"]
        print(f"  Latency: Vortex is {speedup:.1f}x faster")
        print(f"  QPS:     Vortex is {qps_ratio:.1f}x higher throughput")

        print(f"\n{'Metric':<25} {'Vortex':>12} {'LlamaIndex':>12} {'Ratio':>10}")
        print("-" * 60)
        print(f"{'Mean latency (ms)':<25} {vortex_results['mean_ms']:>12.1f} "
              f"{llamaindex_results['mean_ms']:>12.1f} {speedup:>9.1f}x")
        print(f"{'p99 latency (ms)':<25} {vortex_results['p99_ms']:>12.1f} "
              f"{llamaindex_results['p99_ms']:>12.1f} "
              f"{llamaindex_results['p99_ms']/vortex_results['p99_ms']:>9.1f}x")
        print(f"{'QPS':<25} {vortex_results['qps']:>12.0f} "
              f"{llamaindex_results['qps']:>12.0f} {qps_ratio:>9.1f}x")


if __name__ == "__main__":
    main()
