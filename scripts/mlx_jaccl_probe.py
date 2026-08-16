#!/usr/bin/env python3
# pyright: reportAny=false
"""Two-rank MLX/JACCL transport probe with operation-level JSONL tracing.

Run the same command on both Macs, changing only ``--rank``. The device matrix
uses the same format as ``MLX_IBV_DEVICES`` in Exo.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx


@dataclass
class ActiveOperation:
    sequence: int = -1
    operation: str = "initializing"
    byte_count: int = 0
    started_at: float = 0.0
    completed: bool = False


ACTIVE = ActiveOperation()


def emit(event: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "timestamp": time.time(),
                "event": event,
                "rank": int(os.environ.get("MLX_RANK", "-1")),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def watchdog(timeout_seconds: float) -> None:
    while True:
        time.sleep(0.1)
        if ACTIVE.completed or ACTIVE.started_at == 0:
            continue
        elapsed = time.monotonic() - ACTIVE.started_at
        if elapsed < timeout_seconds:
            continue
        emit(
            "operation_timeout",
            sequence=ACTIVE.sequence,
            operation=ACTIVE.operation,
            bytes=ACTIVE.byte_count,
            elapsed_seconds=elapsed,
        )
        os._exit(124)


def evaluated(array: mx.array) -> mx.array:
    mx.eval(array)
    return array


def run_operation(
    sequence: int,
    operation: str,
    byte_count: int,
    callback: Any,
) -> mx.array:
    ACTIVE.sequence = sequence
    ACTIVE.operation = operation
    ACTIVE.byte_count = byte_count
    ACTIVE.started_at = time.monotonic()
    ACTIVE.completed = False
    emit(
        "operation_begin",
        sequence=sequence,
        operation=operation,
        bytes=byte_count,
    )
    result = evaluated(callback())
    elapsed = time.monotonic() - ACTIVE.started_at
    ACTIVE.completed = True
    emit(
        "operation_end",
        sequence=sequence,
        operation=operation,
        bytes=byte_count,
        elapsed_seconds=elapsed,
        checksum=float(mx.sum(result).item()),
    )
    return result


def parse_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated bytes")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument(
        "--devices",
        default='[[null,"rdma_en0"],["rdma_en0",null]]',
        help="JACCL device matrix JSON",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=parse_sizes(
            "1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--operation-timeout", type=float, default=30)
    args = parser.parse_args()

    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    devices = json.loads(args.devices)
    with tempfile.TemporaryDirectory(prefix="exo-jaccl-probe-") as directory:
        device_file = Path(directory) / "devices.json"
        device_file.write_text(json.dumps(devices))
        os.environ["MLX_IBV_DEVICES"] = str(device_file)
        os.environ["MLX_JACCL_COORDINATOR"] = args.coordinator
        os.environ["MLX_RANK"] = str(args.rank)

        threading.Thread(
            target=watchdog, args=(args.operation_timeout,), daemon=True
        ).start()

        emit(
            "probe_start",
            coordinator=args.coordinator,
            devices=devices,
            sizes=args.sizes,
            repetitions=args.repetitions,
        )
        group = mx.distributed.init(backend="jaccl", strict=True)
        if group.size() != 2:
            raise RuntimeError(f"expected two ranks, got {group.size()}")

        sequence = 0
        for repetition in range(args.repetitions):
            for byte_count in args.sizes:
                element_count = max(1, byte_count // 4)
                payload = mx.full((element_count,), args.rank + 1, dtype=mx.float32)

                run_operation(
                    sequence,
                    "all_sum",
                    byte_count,
                    lambda p=payload: mx.distributed.all_sum(p, group=group),
                )
                sequence += 1

                if args.rank == 0:
                    operation = "send_0_to_1"

                    def callback(p: mx.array = payload) -> mx.array:
                        return mx.distributed.send(p, 1, group=group)

                else:
                    operation = "recv_0_to_1"

                    def callback(p: mx.array = payload) -> mx.array:
                        return mx.distributed.recv_like(p, 0, group=group)

                run_operation(sequence, operation, byte_count, callback)
                sequence += 1

                if args.rank == 1:
                    operation = "send_1_to_0"

                    def callback(p: mx.array = payload) -> mx.array:
                        return mx.distributed.send(p, 0, group=group)

                else:
                    operation = "recv_1_to_0"

                    def callback(p: mx.array = payload) -> mx.array:
                        return mx.distributed.recv_like(p, 1, group=group)

                run_operation(sequence, operation, byte_count, callback)
                sequence += 1

            emit("repetition_complete", repetition=repetition)

        emit("probe_complete", operations=sequence)


if __name__ == "__main__":
    main()
