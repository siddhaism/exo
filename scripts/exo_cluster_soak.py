#!/usr/bin/env python3
# pyright: reportAny=false
"""Repeat small-to-large Exo requests and capture diagnostics on first failure."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, *, timeout: float, data: bytes | None = None) -> bytes:
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:52415")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pause", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--model", default="mlx-community/medgemma-27b-text-it-4bit")
    parser.add_argument("--prompt-sizes", default="128,2048,8192,24000")
    parser.add_argument("--output", type=Path, default=Path("cluster-soak-results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_sizes = [int(value) for value in args.prompt_sizes.split(",")]

    for iteration in range(args.iterations):
        started = time.monotonic()
        event: dict[str, object] = {"iteration": iteration, "ok": False}
        try:
            health = json.loads(
                request(f"{args.base_url}/v1/cluster/health", timeout=10)
            )
            event["health"] = health
            if not health.get("dataPlane", {}).get("ready"):
                raise RuntimeError("cluster data plane is not ready")

            prompt_size = prompt_sizes[iteration % len(prompt_sizes)]
            payload = json.dumps(
                {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Reply with exactly OK. Context: "
                                + ("x " * (prompt_size // 2))
                            ),
                        }
                    ],
                    "stream": False,
                    "options": {"num_predict": 1, "temperature": 0},
                }
            ).encode()
            response = json.loads(
                request(
                    f"{args.base_url}/ollama/api/chat",
                    timeout=args.timeout,
                    data=payload,
                )
            )
            event["promptSize"] = prompt_size
            event["done"] = response.get("done")
            if response.get("done") is not True:
                raise RuntimeError(f"inference did not complete: {response}")
            event["ok"] = True
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            event["error"] = repr(error)
            try:
                bundle = request(f"{args.base_url}/v1/diagnostics/bundle", timeout=30)
                path = args.output / f"failure-{iteration}.zip"
                path.write_bytes(bundle)
                event["diagnosticBundle"] = str(path)
            except OSError as bundle_error:
                event["diagnosticError"] = repr(bundle_error)
        finally:
            event["elapsedSeconds"] = time.monotonic() - started
            print(json.dumps(event, sort_keys=True), flush=True)

        if not event["ok"]:
            raise SystemExit(1)
        time.sleep(args.pause)


if __name__ == "__main__":
    main()
