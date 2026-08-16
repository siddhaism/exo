"""Fingerprint the complete MLX runtime loaded by an Exo runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_mlx_build_info(site_packages: Path | None = None) -> dict[str, object]:
    if site_packages is None:
        site_packages = Path(
            str(importlib.metadata.distribution("mlx").locate_file(""))
        ).resolve()

    mlx_root = site_packages if site_packages.name == "mlx" else site_packages / "mlx"
    core_candidates = sorted(mlx_root.glob("core.*.so"))
    components = {
        "core": (
            core_candidates[0]
            if core_candidates
            else mlx_root / "core.extension-not-found"
        ),
        "libmlx": mlx_root / "lib" / "libmlx.dylib",
        "libjaccl": mlx_root / "lib" / "libjaccl.dylib",
        "metallib": mlx_root / "lib" / "mlx.metallib",
    }
    fingerprints: dict[str, dict[str, object]] = {}
    for name, path in components.items():
        fingerprints[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "size": path.stat().st_size if path.is_file() else None,
        }

    versions: dict[str, str | None] = {}
    for package in ("mlx", "mlx-metal", "mlx-lm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    portable_fingerprints = {
        name: {
            "exists": fingerprint["exists"],
            "sha256": fingerprint["sha256"],
            "size": fingerprint["size"],
        }
        for name, fingerprint in fingerprints.items()
    }
    canonical = json.dumps(
        {"versions": versions, "components": portable_fingerprints},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "buildId": hashlib.sha256(canonical.encode()).hexdigest(),
        "versions": versions,
        "components": fingerprints,
    }
