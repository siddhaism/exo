#!/usr/bin/env python3
"""Print and optionally verify the complete MLX runtime build identity."""

from __future__ import annotations

import argparse
import json
from typing import cast

from exo.worker.engines.mlx.build_info import collect_mlx_build_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-build-id",
        help="fail unless the installed runtime matches this build ID",
    )
    args = parser.parse_args()
    expected_build_id = cast(str | None, args.expected_build_id)

    build_info = collect_mlx_build_info()
    print(json.dumps(build_info, indent=2, sort_keys=True))
    actual = str(build_info["buildId"])
    if expected_build_id and actual != expected_build_id:
        raise SystemExit(
            f"MLX runtime mismatch: expected {expected_build_id}, got {actual}"
        )


if __name__ == "__main__":
    main()
