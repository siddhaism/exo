#!/usr/bin/env python3
# pyright: reportAny=false
"""Exercise Exo with increasing prompt sizes and record progress snapshots."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid


def request_json(
    url: str,
    *,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> object:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:52415")
    parser.add_argument("--model", default="mlx-community/medgemma-27b-text-it-4bit")
    parser.add_argument(
        "--sizes",
        default="128,512,1024,2048,4096,8192,24000",
        help="Approximate prompt token counts",
    )
    parser.add_argument("--stall-timeout", type=float, default=90)
    args = parser.parse_args()

    sizes = [int(item) for item in args.sizes.split(",")]
    unit = "clinical record observation without additional findings. "

    for approximate_tokens in sizes:
        request_id = str(uuid.uuid4()).upper()
        # This intentionally overestimates English tokens only slightly; Exo's
        # progress endpoint reports the authoritative tokenizer count.
        prompt = (unit * ((approximate_tokens * 4 // len(unit)) + 1))[
            : approximate_tokens * 4
        ]
        started = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "request_begin",
                    "request_id": request_id,
                    "approximate_tokens": approximate_tokens,
                }
            ),
            flush=True,
        )
        try:
            request_json(
                f"{args.base_url}/ollama/api/chat",
                payload={
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 1, "temperature": 0},
                },
                headers={"x-exo-request-id": request_id},
                timeout=args.stall_timeout + 60,
            )
            status = "complete"
        except (TimeoutError, urllib.error.URLError) as error:
            status = "failed"
            print(
                json.dumps(
                    {
                        "event": "request_error",
                        "request_id": request_id,
                        "error": str(error),
                    }
                ),
                flush=True,
            )

        try:
            progress = request_json(
                f"{args.base_url}/v1/tasks/progress/{request_id}", timeout=5
            )
        except urllib.error.URLError as error:
            progress = {"error": str(error)}
        print(
            json.dumps(
                {
                    "event": "request_end",
                    "request_id": request_id,
                    "status": status,
                    "elapsed_seconds": time.monotonic() - started,
                    "progress": progress,
                },
                default=str,
            ),
            flush=True,
        )
        if status != "complete":
            break


if __name__ == "__main__":
    main()
