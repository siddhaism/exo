from datetime import datetime, timedelta, timezone

import anyio
from loguru import logger

from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.master.placement_utils import find_ip_prioritised
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.apply import apply
from exo.shared.constants import (
    EXO_DISTRIBUTED_RECYCLE_COOLDOWN_SECONDS,
    EXO_EVENT_LOG_DIR,
    EXO_RECYCLE_DISTRIBUTED_MLX_AFTER_GENERATION,
    EXO_TRACING_ENABLED,
)
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    SetInstanceLink,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceCreated,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
)
from exo.shared.types.worker.runners import RunnerFailed
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    Sharding,
    TensorShardMetadata,
)
from exo.utils.channels import Receiver, Sender
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.task_group import TaskGroup


def _has_in_flight_generation(state: State, instance_id: InstanceId) -> bool:
    """Whether a generation is still assigned to this instance.

    Used to decide when a retiring instance has drained and can be torn down. Only
    generations are counted, because only a generation makes teardown unsafe.

    Worker lifecycle tasks must not count. One that never reaches a terminal status — a
    stale DownloadModel, for instance, whose weights have long since arrived — would
    otherwise pin the instance in the retiring set indefinitely. The master excludes
    retiring instances from eligibility, so every subsequent request is then refused with
    "a distributed instance is being recycled or replaced; retry in a few seconds", a
    transient-sounding message for a condition that only a restart clears.
    """
    return any(
        task.instance_id == instance_id
        and task.task_status in {TaskStatus.Pending, TaskStatus.Running}
        and isinstance(
            task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
        )
        for task in state.tasks.values()
    )


def _prefill_endpoint_for(state: State, decode_instance_id: InstanceId) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


def _recovery_placement(instance: Instance) -> PlaceInstance:
    shards = list(instance.shard_assignments.runner_to_shard.values())
    if not shards:
        raise ValueError(f"Instance {instance.instance_id} has no shards")
    first_shard = shards[0]
    if isinstance(first_shard, TensorShardMetadata):
        sharding = Sharding.Tensor
    elif isinstance(first_shard, PipelineShardMetadata):
        sharding = Sharding.Pipeline
    else:
        raise ValueError(
            "Automatic distributed recovery does not support "
            f"{first_shard.__class__.__name__}"
        )
    return PlaceInstance(
        model_card=first_shard.model_card,
        sharding=sharding,
        instance_meta=(
            InstanceMeta.MlxJaccl
            if isinstance(instance, MlxJacclInstance)
            else InstanceMeta.MlxRing
        ),
        min_nodes=len(instance.shard_assignments.node_to_runner),
    )


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        download_command_sender: Sender[ForwarderDownloadCommand],
    ):
        self.node_id = node_id
        self.session_id = session_id
        self.state = State()
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(EXO_EVENT_LOG_DIR / "master")
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}
        self._recovering_instances: dict[InstanceId, tuple[float, PlaceInstance]] = {}
        # A retiring instance is never selected for a new request.  It is
        # removed as soon as its already-assigned requests have drained.
        self._retiring_instances: dict[InstanceId, PlaceInstance] = {}

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._plan)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            # set-difference => prefill-only nodes
                            prefill_only: set[InstanceId] = set()
                            for link in self.state.instance_links.values():
                                prefill_only.update(link.prefill_instances)
                            for link in self.state.instance_links.values():
                                prefill_only.difference_update(link.decode_instances)

                            for instance in self.state.instances.values():
                                # NON-prefill-only instances matching the model ID
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                    and instance.instance_id not in prefill_only
                                    and instance.instance_id
                                    not in self._retiring_instances
                                ):
                                    # count in-flight tasks of that instance
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            # there are no NON-prefill-only instances matching this model ID
                            if not instance_task_counts:
                                # Raising here went into the command processor's
                                # catch-all and emitted nothing, so the client
                                # learned nothing and waited out the full task
                                # stall timeout for a condition known instantly.
                                # `_retiring_instances` is master-local, so the
                                # API's validation cannot see it and will happily
                                # publish a command the master then refuses.
                                logger.warning(
                                    "Rejecting generation: no eligible instance for "
                                    f"model={command.task_params.model} "
                                    f"instances={len(self.state.instances)} "
                                    f"retiring={len(self._retiring_instances)} "
                                    f"prefill_only={len(prefill_only)}"
                                )
                                generated_events.append(
                                    ChunkGenerated(
                                        command_id=command.command_id,
                                        chunk=ErrorChunk(
                                            model=command.task_params.model,
                                            error_message=(
                                                "No instance is currently able to serve "
                                                f"model {command.task_params.model}. A "
                                                "distributed instance is being recycled "
                                                "or replaced; retry in a few seconds."
                                            ),
                                        ),
                                    )
                                )
                            else:
                                available_instance_ids = sorted(
                                    instance_task_counts.keys(),
                                    key=lambda instance_id: instance_task_counts[
                                        instance_id
                                    ],
                                )

                                decode_instance_id = available_instance_ids[0]
                                task_id = TaskId()
                                params = command.task_params.model_copy(
                                    update={
                                        "prefill_endpoint": _prefill_endpoint_for(
                                            self.state, decode_instance_id
                                        ),
                                    }
                                )
                                generated_events.append(
                                    TaskCreated(
                                        task_id=task_id,
                                        task=TextGenerationTask(
                                            task_id=task_id,
                                            command_id=command.command_id,
                                            instance_id=decode_instance_id,
                                            task_status=TaskStatus.Pending,
                                            task_params=params,
                                        ),
                                    )
                                )
                                self.command_task_mapping[command.command_id] = task_id
                        case ImageGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ImageEdits():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                self.state.node_memory,
                                self.state.node_network,
                                self.state.node_backends,
                                download_status=self.state.downloads,
                                node_rdma_ctl=self.state.node_rdma_ctl,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            if (
                                task_id := self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                            ) is not None:
                                task = self.state.tasks.get(task_id)
                                generated_events.append(TaskDeleted(task_id=task_id))
                                if isinstance(task, TextGenerationTask):
                                    instance = self.state.instances.get(
                                        task.instance_id
                                    )
                                    if (
                                        EXO_RECYCLE_DISTRIBUTED_MLX_AFTER_GENERATION
                                        and isinstance(instance, MlxJacclInstance)
                                        and len(
                                            instance.shard_assignments.node_to_runner
                                        )
                                        > 1
                                    ):
                                        self._retiring_instances[
                                            instance.instance_id
                                        ] = _recovery_placement(instance)
                                        logger.info(
                                            "Distributed MLX generation completed; "
                                            "retiring the JACCL instance before reuse "
                                            f"instance_id={instance.instance_id} "
                                            f"task_id={task_id}"
                                        )
                            else:
                                logger.warning(
                                    f"Finished command {command.finished_command_id} finished"
                                )

                        case AddCustomModelCard():
                            generated_events.append(
                                CustomModelCardAdded(model_card=command.model_card)
                            )
                        case DeleteCustomModelCard():
                            generated_events.append(
                                CustomModelCardDeleted(model_id=command.model_id)
                            )
                        case SetInstanceLink():
                            link = InstanceLink(
                                link_id=command.link_id,
                                prefill_instances=list(
                                    dict.fromkeys(command.prefill_instances)
                                ),
                                decode_instances=list(
                                    dict.fromkeys(command.decode_instances)
                                ),
                            )
                            generated_events.append(InstanceLinkCreated(link=link))
                        case DeleteInstanceLink():
                            generated_events.append(
                                InstanceLinkDeleted(link_id=command.link_id)
                            )
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            # rate limit to 1000 at a time
                            end = min(command.since_idx + 1000, len(self._event_log))
                            for i, event in enumerate(
                                self._event_log.read_range(command.since_idx, end),
                                start=command.since_idx,
                            ):
                                await self._send_indexed_event(
                                    IndexedEvent(idx=i, event=event)
                                )
                    for event in generated_events:
                        await self.event_sender.send(event)
                except Exception as e:
                    logger.opt(exception=e).warning("Error in command processor")

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            # A completed distributed generation may leave native JACCL/Metal
            # state that cannot safely serve another large request.  Drain any
            # requests that were already assigned, then remove both ranks as a
            # single failure domain.  Placement below recreates fresh runners.
            for instance_id, recovery_command in list(
                self._retiring_instances.items()
            ):
                instance = self.state.instances.get(instance_id)
                if instance is None:
                    self._retiring_instances.pop(instance_id, None)
                    continue
                in_flight = _has_in_flight_generation(self.state, instance_id)
                if in_flight:
                    continue

                self._recovering_instances[instance_id] = (
                    anyio.current_time()
                    + EXO_DISTRIBUTED_RECYCLE_COOLDOWN_SECONDS,
                    recovery_command,
                )
                self._retiring_instances.pop(instance_id, None)
                logger.info(
                    "Distributed MLX instance drained; shutting down all ranks "
                    f"instance_id={instance_id} "
                    "reason=post_generation_recycle "
                    f"cooldown_seconds={EXO_DISTRIBUTED_RECYCLE_COOLDOWN_SECONDS}"
                )
                target_instances = dict(self.state.instances)
                del target_instances[instance_id]
                for event in get_transition_events(
                    self.state.instances, target_instances, self.state.tasks
                ):
                    await self.event_sender.send(event)

            # A distributed backend is one failure domain. Reusing one side of
            # a failed JACCL generation leaves stale queue pairs on its peer,
            # so tear down the whole instance and later place a fresh one.
            for instance_id, instance in list(self.state.instances.items()):
                if (
                    len(instance.shard_assignments.node_to_runner) <= 1
                    or instance_id in self._recovering_instances
                    or instance_id in self._retiring_instances
                ):
                    continue
                failed_runners = [
                    runner_id
                    for runner_id in instance.shard_assignments.runner_to_shard
                    if isinstance(self.state.runners.get(runner_id), RunnerFailed)
                ]
                if not failed_runners:
                    continue

                recovery_command = _recovery_placement(instance)
                self._recovering_instances[instance_id] = (
                    anyio.current_time() + 30,
                    recovery_command,
                )
                logger.error(
                    "Distributed instance failure detected; deleting entire "
                    f"generation instance_id={instance_id} "
                    f"failed_runners={failed_runners} cooldown_seconds=30"
                )
                target_instances = dict(self.state.instances)
                del target_instances[instance_id]
                for event in get_transition_events(
                    self.state.instances, target_instances, self.state.tasks
                ):
                    await self.event_sender.send(event)

            for old_instance_id, (
                ready_at,
                recovery_command,
            ) in list(self._recovering_instances.items()):
                if old_instance_id in self.state.instances:
                    continue
                if anyio.current_time() < ready_at:
                    continue
                try:
                    placement = place_instance(
                        recovery_command,
                        self.state.topology,
                        self.state.instances,
                        self.state.node_memory,
                        self.state.node_network,
                        self.state.node_backends,
                        download_status=self.state.downloads,
                        node_rdma_ctl=self.state.node_rdma_ctl,
                    )
                    transition_events = get_transition_events(
                        self.state.instances, placement, self.state.tasks
                    )
                    created = next(
                        (
                            event
                            for event in transition_events
                            if isinstance(event, InstanceCreated)
                        ),
                        None,
                    )
                    if created is None:
                        raise RuntimeError(
                            "recovery placement did not create a fresh instance"
                        )
                    logger.info(
                        "Distributed instance cooldown complete; creating fresh "
                        f"generation old_instance_id={old_instance_id} "
                        f"new_instance_id={created.instance.instance_id}"
                    )
                    for event in transition_events:
                        await self.event_sender.send(event)
                    del self._recovering_instances[old_instance_id]
                except Exception:
                    logger.opt(exception=True).warning(
                        "Unable to place fresh distributed generation yet; "
                        f"old_instance_id={old_instance_id}"
                    )

            # kill broken instances
            connected_node_ids = set(self.state.topology.list_nodes())
            for instance_id, instance in self.state.instances.items():
                for node_id in instance.shard_assignments.node_to_runner:
                    if node_id not in connected_node_ids:
                        await self.event_sender.send(
                            InstanceDeleted(instance_id=instance_id)
                        )
                        break

            # time out dead nodes
            for node_id, time in self.state.last_seen.items():
                now = datetime.now(tz=timezone.utc)
                if now - time > timedelta(seconds=30):
                    logger.info(f"Manually removing node {node_id} due to inactivity")
                    await self.event_sender.send(NodeTimedOut(node_id=node_id))

            await anyio.sleep(10)

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    logger.debug(f"Master indexing event: {str(event)[:100]}")

                    event = event.model_copy(
                        update={"_master_time_stamp": datetime.now(tz=timezone.utc)}
                    )
                    if isinstance(event, NodeGatheredInfo):
                        event = event.model_copy(
                            update={"when": str(datetime.now(tz=timezone.utc))}
                        )

                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    self._event_log.append(event)
                    await self._send_indexed_event(indexed)

    # This function is re-entrant, take care!
    async def _send_indexed_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self.event_sender.send(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]
