#!/usr/bin/env python3
"""
Mock embedding and LLM servers for benchmarking.
Returns deterministic responses with configurable latency.
"""

import argparse
import json
import random
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class MockEmbeddingHandler(BaseHTTPRequestHandler):
    latency_ms = 2
    dim = 128

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        time.sleep(self.latency_ms / 1000.0)

        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        data = []
        for i, text in enumerate(inputs):
            # Deterministic embedding based on text hash
            random.seed(hash(text) % 2**32)
            embedding = [random.gauss(0, 1) for _ in range(self.dim)]
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x / norm for x in embedding]
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding
            })

        response = {
            "object": "list",
            "data": data,
            "model": "mock",
            "usage": {"prompt_tokens": len(inputs) * 10, "total_tokens": len(inputs) * 10}
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # Suppress logs


class MockLLMHandler(BaseHTTPRequestHandler):
    latency_ms = 50

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        time.sleep(self.latency_ms / 1000.0)

        response = {
            "id": "mock-1",
            "object": "chat.completion",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response based on the provided context."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Mock servers for benchmarking")
    parser.add_argument("--embed-port", type=int, default=9998)
    parser.add_argument("--embed-latency-ms", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--llm-port", type=int, default=9999)
    parser.add_argument("--llm-latency-ms", type=int, default=50)
    args = parser.parse_args()

    MockEmbeddingHandler.latency_ms = args.embed_latency_ms
    MockEmbeddingHandler.dim = args.embed_dim
    MockLLMHandler.latency_ms = args.llm_latency_ms

    embed_server = HTTPServer(("127.0.0.1", args.embed_port), MockEmbeddingHandler)
    llm_server = HTTPServer(("127.0.0.1", args.llm_port), MockLLMHandler)

    print(f"Mock embedding server: http://127.0.0.1:{args.embed_port} "
          f"(dim={args.embed_dim}, latency={args.embed_latency_ms}ms)")
    print(f"Mock LLM server:      http://127.0.0.1:{args.llm_port} "
          f"(latency={args.llm_latency_ms}ms)")

    t1 = threading.Thread(target=embed_server.serve_forever, daemon=True)
    t2 = threading.Thread(target=llm_server.serve_forever, daemon=True)
    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        embed_server.shutdown()
        llm_server.shutdown()


if __name__ == "__main__":
    main()
