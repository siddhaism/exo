from exo.master.main import (
    _has_in_flight_generation,  # pyright: ignore[reportPrivateUsage]
    _recovery_placement,  # pyright: ignore[reportPrivateUsage]
)
from exo.shared.apply import apply_instance_deleted
from exo.shared.types.common import CommandId
from exo.shared.types.events import InstanceDeleted
from exo.shared.types.state import State
from exo.shared.types.tasks import DownloadModel, TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.runners import RunnerFailed, RunnerReady
from exo.shared.types.worker.shards import Sharding
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    INSTANCE_2_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def _distributed_instance():
    shard_0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard_1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    return get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard_0, RUNNER_2_ID: shard_1},
    )


def test_instance_deletion_purges_entire_runner_generation() -> None:
    instance = _distributed_instance()
    state = State(
        instances={INSTANCE_1_ID: instance},
        runners={
            RUNNER_1_ID: RunnerFailed(error_message="boom", diagnostics=[]),
            RUNNER_2_ID: RunnerReady(),
        },
        prefill_server_ports={RUNNER_1_ID: 5001, RUNNER_2_ID: 5002},
    )

    recovered = apply_instance_deleted(
        InstanceDeleted(instance_id=INSTANCE_1_ID), state
    )

    assert INSTANCE_1_ID not in recovered.instances
    assert RUNNER_1_ID not in recovered.runners
    assert RUNNER_2_ID not in recovered.runners
    assert RUNNER_1_ID not in recovered.prefill_server_ports
    assert RUNNER_2_ID not in recovered.prefill_server_ports


def test_recovery_placement_preserves_model_sharding_and_node_count() -> None:
    command = _recovery_placement(_distributed_instance())

    assert command.model_card.model_id == MODEL_A_ID
    assert command.sharding == Sharding.Pipeline
    assert command.instance_meta == InstanceMeta.MlxRing
    assert command.min_nodes == 2


def _text_generation_task(status: TaskStatus) -> TextGeneration:
    return TextGeneration(
        instance_id=INSTANCE_1_ID,
        task_status=status,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=MODEL_A_ID,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


def _download_task(status: TaskStatus) -> DownloadModel:
    return DownloadModel(
        instance_id=INSTANCE_1_ID,
        task_status=status,
        shard_metadata=get_pipeline_shard_metadata(
            MODEL_A_ID, device_rank=0, world_size=2
        ),
    )


def test_running_generation_blocks_the_retiring_drain() -> None:
    state = State(tasks={TaskId(): _text_generation_task(TaskStatus.Running)})
    assert _has_in_flight_generation(state, INSTANCE_1_ID)


def test_pending_generation_blocks_the_retiring_drain() -> None:
    state = State(tasks={TaskId(): _text_generation_task(TaskStatus.Pending)})
    assert _has_in_flight_generation(state, INSTANCE_1_ID)


def test_completed_generation_does_not_block_the_drain() -> None:
    state = State(tasks={TaskId(): _text_generation_task(TaskStatus.Complete)})
    assert not _has_in_flight_generation(state, INSTANCE_1_ID)


def test_stale_lifecycle_task_does_not_block_the_drain() -> None:
    """A worker task stuck short of a terminal status must not pin the instance.

    Counting these meant a stale DownloadModel — whose weights had already arrived — kept a
    retiring instance in the retiring set forever, and every request was refused with a
    transient-sounding "being recycled" message that only a restart cleared.
    """
    state = State(tasks={TaskId(): _download_task(TaskStatus.Running)})
    assert not _has_in_flight_generation(state, INSTANCE_1_ID)


def test_generation_for_another_instance_does_not_block_the_drain() -> None:
    task = _text_generation_task(TaskStatus.Running).model_copy(
        update={"instance_id": INSTANCE_2_ID}
    )
    state = State(tasks={TaskId(): task})
    assert not _has_in_flight_generation(state, INSTANCE_1_ID)


def test_generation_alongside_a_stale_lifecycle_task_still_blocks() -> None:
    state = State(
        tasks={
            TaskId(): _download_task(TaskStatus.Running),
            TaskId(): _text_generation_task(TaskStatus.Running),
        }
    )
    assert _has_in_flight_generation(state, INSTANCE_1_ID)
