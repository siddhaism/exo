from pathlib import Path
from typing import cast

from exo.worker.engines.mlx.build_info import collect_mlx_build_info, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"exo mlx identity")

    assert sha256_file(artifact) == (
        "f411013bfcda6fcff2a910fbafaab78f2e9df831652a9aa166bc69f8e5638669"
    )


def test_collect_mlx_build_info_changes_when_component_changes(
    tmp_path: Path,
) -> None:
    package = tmp_path / "mlx"
    package.mkdir()
    core = package / "core.cpython-test-darwin.so"
    core.write_bytes(b"core-v1")

    first = collect_mlx_build_info(package)
    core.write_bytes(b"core-v2")
    second = collect_mlx_build_info(package)

    assert first["buildId"] != second["buildId"]
    components = first["components"]
    assert isinstance(components, dict)
    core_component = cast(dict[str, object], components["core"])
    assert isinstance(core_component, dict)
    assert core_component["exists"] is True
