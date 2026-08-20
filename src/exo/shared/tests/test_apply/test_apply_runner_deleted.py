from exo.shared.apply import apply_runner_status_updated
from exo.shared.types.events import RunnerStatusUpdated
from exo.shared.types.state import State
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerShutdown,
    RunnerShuttingDown,
)
from exo.worker.tests.constants import INSTANCE_1_ID


def test_apply_runner_shutdown_removes_runner():
    runner_id = RunnerId()
    state = State(runners={runner_id: RunnerIdle()})

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()), state
    )

    assert runner_id not in new_state.runners


def test_apply_runner_status_updated_adds_runner():
    runner_id = RunnerId()
    state = State()

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerIdle()), state
    )

    assert runner_id in new_state.runners


def test_late_runner_shutting_down_does_not_resurrect_deleted_generation():
    runner_id = RunnerId()
    state = State(
        runners={runner_id: RunnerIdle()}, prefill_server_ports={runner_id: 5001}
    )

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(
            runner_id=runner_id,
            runner_status=RunnerShuttingDown(
                data_plane_generation=str(INSTANCE_1_ID)
            ),
        ),
        state,
    )

    assert runner_id not in new_state.runners
    assert runner_id not in new_state.prefill_server_ports
