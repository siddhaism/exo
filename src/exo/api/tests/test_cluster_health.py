import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

import exo.api.main
from exo.api.main import API
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.state import State
from exo.shared.types.worker.runners import RunnerId, RunnerReady


def test_cluster_health_reports_control_and_data_planes() -> None:
    node_id = NodeId("node-a")
    state = State(
        last_seen={node_id: datetime.now(timezone.utc)},
    )
    state.topology.add_node(node_id)

    api = object.__new__(API)
    api.state = state

    def network_health() -> dict[str, object]:
        return {
            "peers": {
                "node-b": {
                    "connected": True,
                    "changedAt": datetime.now(timezone.utc).isoformat(),
                }
            },
            "reconnectCount": 2,
            "lastError": None,
        }

    api._network_health_provider = network_health  # pyright: ignore[reportPrivateUsage]

    health = api.get_cluster_health()

    control_plane = health["controlPlane"]
    assert isinstance(control_plane, dict)
    assert control_plane["reconnectCount"] == 2

    nodes = cast(dict[str, object], health["nodes"])
    node_health = cast(dict[str, object], nodes[str(node_id)])
    assert node_health["healthy"] is True

    data_plane = health["dataPlane"]
    assert isinstance(data_plane, dict)
    assert data_plane["ready"] is False


def test_cluster_health_rejects_mixed_mlx_builds() -> None:
    state = State(
        runners={
            RunnerId("runner-a"): RunnerReady(
                runtime_build_id="build-a",
                data_plane_generation="generation-a",
            ),
            RunnerId("runner-b"): RunnerReady(
                runtime_build_id="build-b",
                data_plane_generation="generation-a",
            ),
        }
    )
    api = object.__new__(API)
    api.state = state
    api._network_health_provider = dict  # pyright: ignore[reportPrivateUsage]

    data_plane = cast(dict[str, object], api.get_cluster_health()["dataPlane"])

    assert data_plane["ready"] is False
    assert data_plane["runtimeBuildIds"] == ["build-a", "build-b"]


@pytest.mark.anyio
async def test_diagnostic_bundle_contains_health_progress_state_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_id = NodeId("node-a")
    state = State(last_seen={node_id: datetime.now(timezone.utc)})
    state.topology.add_node(node_id)
    (tmp_path / "rank-0.log").write_text("pipeline trace\n")
    monkeypatch.setattr(exo.api.main, "EXO_LOG_DIR", tmp_path)

    api = object.__new__(API)
    api.node_id = node_id
    api.state = state
    api._task_progress = {  # pyright: ignore[reportPrivateUsage]
        CommandId("request-a"): {"phase": "prefill"}
    }
    api._network_health_provider = dict  # pyright: ignore[reportPrivateUsage]

    response = api.get_diagnostic_bundle()
    chunks = [chunk async for chunk in response.body_iterator]
    payload = b"".join(
        chunk.encode() if isinstance(chunk, str) else chunk for chunk in chunks
    )

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert {
            "manifest.json",
            "cluster-health.json",
            "task-progress.json",
            "state.json",
            "logs/rank-0.log",
        }.issubset(archive.namelist())
        progress = cast(
            dict[str, dict[str, str]],
            json.loads(archive.read("task-progress.json")),
        )
        assert progress["request-a"]["phase"] == "prefill"
