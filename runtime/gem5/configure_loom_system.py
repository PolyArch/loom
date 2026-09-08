#!/usr/bin/env python3
"""Instantiate one exact Loom RISC-V machine-mode gem5 projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import resource
import subprocess
import time

import m5
from m5.params import NULL
from m5.objects import (
    AddrRange,
    Cache,
    CommMonitor,
    LoomMemoryServiceProbe,
    FUPool,
    FUDesc,
    IQUnit,
    PMAChecker,
    LoomRiscvDeploymentWorkload,
    LoomSpatialBridge,
    LoomSpatialEngineSession,
    LoomThreadDispatch,
    OpDesc,
    RiscvO3CPU,
    RiscvSystem,
    RiscvTimingSimpleCPU,
    Root,
    SimpleMemory,
    SrcClockDomain,
    SystemXBar,
    VoltageDomain,
)


CONFIG_SCHEMA = "loom.gem5_system_projection.16"
PERFORMANCE_PROFILE_SCHEMA = "loom.gem5_system_performance_profile.6"
STATISTICS_BEGIN = "---------- Begin Simulation Statistics ----------"
STATISTICS_END = "---------- End Simulation Statistics   ----------"

BRIDGE_STAT_SUFFIXES = {
    "bridge_callback_cpu_nanoseconds": ".loomPerformance.callbackCpuNanoseconds",
    "bridge_engine_wait_nanoseconds": ".loomPerformance.engineWaitNanoseconds",
    "bridge_message_count": ".loomPerformance.messageCount",
    "accelerator_invocation_count": ".loomPerformance.invocationCount",
    "bridge_clock_failure_count": ".loomPerformance.clockFailureCount",
}


def load_timeout_seconds(tier: str) -> int:
    bundled_path = pathlib.Path(__file__).with_name("timeout-budgets.json")
    repository_path = (
        pathlib.Path(__file__).resolve().parents[2] / "config" / "timeout-budgets.json"
    )
    budget_path = bundled_path if bundled_path.is_file() else repository_path
    document = json.loads(budget_path.read_text(encoding="utf-8"))
    if document.get("schema") != "loom.timeout_budgets":
        raise ValueError("timeout budget document has the wrong schema")
    value = document.get("tiers", {}).get(tier, {}).get("seconds")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{tier} timeout budget is invalid")
    return value


XLONG_TIMEOUT_SECONDS = load_timeout_seconds("xlong")
FAST_TIMEOUT_SECONDS = load_timeout_seconds("fast")

PROCESSOR_FIELDS = {
    "cpu_id",
    "model",
    "num_threads",
    "caches",
    "execution_units",
    "pipeline",
}

CACHE_FIELDS = {
    "capacity_bytes",
    "line_bytes",
    "associativity",
    "hit_latency_cycles",
    "miss_status_entries",
}

# gem5 charges each miss-status entry a separate target list. The Fabric record
# owns the outstanding-miss count; this is the per-entry coalescing depth, which
# the architecture contract does not distinguish.
CACHE_TARGETS_PER_MSHR = 8

O3_PIPELINE_FIELDS = {
    "fetch_width",
    "decode_width",
    "rename_width",
    "dispatch_width",
    "issue_width",
    "writeback_width",
    "commit_width",
    "reorder_buffer_entries",
    "issue_queue_entries",
    "load_queue_entries",
    "store_queue_entries",
    "physical_integer_registers",
    "physical_float_registers",
    "physical_vector_registers",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--performance-profile")
    return parser.parse_args()


def require_keys(value: dict, expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} fields do not match the projection schema")


def load_projection(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gem5 system projection must be an object")
    require_keys(
        value,
        {
            "schema",
            "gem5_binary_sha256",
            "clock",
            "memory",
            "host",
            "instruction_images",
            "runtime_images",
            "system_memory",
            "dispatch",
            "processors",
            "bridges",
            "maximum_ticks",
        },
        "gem5 system projection",
    )
    if value["schema"] != CONFIG_SCHEMA:
        raise ValueError("gem5 system projection has the wrong schema")
    if not isinstance(value["processors"], list) or not value["processors"]:
        raise ValueError("gem5 system projection requires processors")
    if not isinstance(value["bridges"], list):
        raise ValueError("gem5 system projection bridges must be an array")
    if not isinstance(value["instruction_images"], list):
        raise ValueError("gem5 instruction images must be an array")
    if not isinstance(value["runtime_images"], list):
        raise ValueError("gem5 runtime images must be an array")
    if not isinstance(value["system_memory"], dict):
        raise ValueError("gem5 System memory projection must be an object")
    return value


def verify_running_binary(expected_sha256: str) -> None:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("gem5 binary digest is invalid")
    digest = hashlib.sha256()
    with pathlib.Path("/proc/self/exe").open("rb") as binary:
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("running gem5 binary differs from the exact binding")


def cpu_seconds(usage: resource.struct_rusage) -> float:
    return usage.ru_utime + usage.ru_stime


def elapsed_cpu_nanoseconds(
    before: resource.struct_rusage, after: resource.struct_rusage
) -> int:
    elapsed = cpu_seconds(after) - cpu_seconds(before)
    if elapsed < 0.0:
        raise RuntimeError("process CPU accounting moved backwards")
    return round(elapsed * 1_000_000_000)


def read_bridge_statistics(path: pathlib.Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    block_begin = text.rfind(STATISTICS_BEGIN)
    if block_begin < 0:
        raise RuntimeError("gem5 statistics have no complete dump")
    block_begin += len(STATISTICS_BEGIN)
    block_end = text.find(STATISTICS_END, block_begin)
    if block_end < 0:
        raise RuntimeError("gem5 statistics have no complete final dump")
    totals = {field: 0 for field in BRIDGE_STAT_SUFFIXES}
    counts = {field: 0 for field in BRIDGE_STAT_SUFFIXES}
    for line in text[block_begin:block_end].splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[1]
        for field, suffix in BRIDGE_STAT_SUFFIXES.items():
            if not name.endswith(suffix):
                continue
            parsed = int(value)
            if parsed < 0:
                raise RuntimeError(f"gem5 bridge statistic {field} is negative")
            totals[field] += parsed
            counts[field] += 1
    bridge_count = counts["bridge_callback_cpu_nanoseconds"]
    if bridge_count == 0 or any(count != bridge_count for count in counts.values()):
        raise RuntimeError("gem5 bridge statistics are not structurally total")
    totals["bridge_count"] = bridge_count
    return totals


def start_engines(
    bridges: list[dict], dispatch_target_count: int
) -> list[subprocess.Popen]:
    processes: list[subprocess.Popen] = []
    claimed_targets: set[int] = set()
    try:
        for ordinal, bridge in enumerate(bridges):
            require_keys(
                bridge,
                {
                    "dispatch_target_ordinals",
                    "acc_core_ref",
                    "execution_context_keys",
                    "spatial_workloads",
                    "cache",
                    "pio_address",
                    "pio_size",
                    "pio_latency",
                    "session_ordinal",
                    "engine_socket",
                    "engine_command",
                    "result_path",
                    "maximum_message_bytes",
                },
                f"bridge {ordinal}",
            )
            targets = bridge["dispatch_target_ordinals"]
            contexts = bridge["execution_context_keys"]
            workloads = bridge["spatial_workloads"]
            if (
                not isinstance(targets, list)
                or targets != sorted(set(targets))
                or any(
                    not isinstance(target, int)
                    or target < 0
                    or target >= dispatch_target_count
                    for target in targets
                )
                or not isinstance(contexts, list)
                or len(contexts) != len(targets)
                or not isinstance(workloads, list)
                or len(workloads) != len(targets)
            ):
                raise ValueError(f"bridge {ordinal} target table is invalid")
            if claimed_targets.intersection(targets):
                raise ValueError("dispatch target is claimed by multiple bridges")
            claimed_targets.update(targets)
            encoded_references = [bridge["acc_core_ref"], *contexts]
            for encoded in encoded_references:
                if (
                    not isinstance(encoded, str)
                    or not encoded
                    or len(encoded) % 2 != 0
                    or any(character not in "0123456789abcdef" for character in encoded)
                ):
                    raise ValueError(f"bridge {ordinal} reference is invalid")
            if any(
                not isinstance(workload, str)
                or len(workload) != 64
                or any(character not in "0123456789abcdef" for character in workload)
                for workload in workloads
            ):
                raise ValueError(f"bridge {ordinal} Spatial workload is invalid")
            if bridge["session_ordinal"] != ordinal:
                raise ValueError(f"bridge {ordinal} session ordinal is invalid")
            command = bridge["engine_command"]
            if not isinstance(command, list) or not all(
                isinstance(item, str) and item for item in command
            ):
                raise ValueError(f"bridge {ordinal} engine command is invalid")
            if bridge["engine_socket"] == "":
                if targets or command:
                    raise ValueError(f"idle bridge {ordinal} has an engine or targets")
                continue
            socket_path = pathlib.Path(bridge["engine_socket"])
            if not command:
                if not socket_path.is_socket():
                    raise RuntimeError(
                        f"bridge {ordinal} external engine socket is unavailable"
                    )
                continue
            socket_path.unlink(missing_ok=True)
            processes.append(subprocess.Popen(command))
            deadline = time.monotonic() + XLONG_TIMEOUT_SECONDS
            while not socket_path.exists():
                if processes[-1].poll() is not None:
                    raise RuntimeError(
                        f"bridge {ordinal} engine exited before publishing its socket"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"bridge {ordinal} engine did not publish its socket"
                    )
                time.sleep(0.01)
        if claimed_targets != set(range(dispatch_target_count)):
            raise ValueError("bridge sessions do not cover every dispatch target")
        return processes
    except BaseException:
        stop_engines(processes)
        raise


def stop_engines(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=FAST_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def finish_engines(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        status = process.poll()
        if status is not None and status != 0:
            raise RuntimeError(f"Spatial engine exited with status {status}")


def build_o3_execution_units(records: list[dict]) -> FUPool:
    units = []
    for ordinal, record in enumerate(records):
        require_keys(
            record,
            {
                "operation_classes",
                "count",
                "latency_cycles",
                "initiation_interval",
            },
            f"execution unit {ordinal}",
        )
        op_classes = record["operation_classes"]
        if (
            not isinstance(op_classes, list)
            or not op_classes
            or not all(isinstance(value, str) and value for value in op_classes)
            or len(set(op_classes)) != len(op_classes)
        ):
            raise ValueError("O3 execution-unit operation classes are invalid")
        latency = record["latency_cycles"]
        interval = record["initiation_interval"]
        if not all(
            isinstance(value, int) and value > 0
            for value in (record["count"], latency, interval)
        ):
            raise ValueError("O3 execution-unit parameters must be positive")
        if interval not in (1, latency):
            raise ValueError("O3 execution-unit initiation interval is unsupported")
        operations = [
            OpDesc(opClass=op_class, opLat=latency, pipelined=interval == 1)
            for op_class in op_classes
        ]
        units.append(FUDesc(opList=operations, count=record["count"]))
    if not units:
        raise ValueError("O3 processor requires execution units")
    return FUPool(FUList=units)


def parse_cache(cache: dict, context: str) -> dict:
    require_keys(cache, CACHE_FIELDS, context)
    for field, value in cache.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{context}.{field} must be positive")
    line = cache["line_bytes"]
    if line & (line - 1):
        raise ValueError(f"{context}.line_bytes must be a power of two")
    if cache["capacity_bytes"] % (line * cache["associativity"]):
        raise ValueError(
            f"{context}.capacity_bytes is not a multiple of line times associativity"
        )
    return cache


def build_cache(cache: dict, read_only: bool, addr_ranges=None) -> Cache:
    # gem5 splits a hit across tag lookup, data access, and the response path.
    # The Fabric record declares one architectural hit latency, so each stage
    # carries it and a hit occupies exactly `hit_latency_cycles` per stage.
    hit_latency = cache["hit_latency_cycles"]
    parameters = dict(
        size=f"{cache['capacity_bytes']}B",
        assoc=cache["associativity"],
        tag_latency=hit_latency,
        data_latency=hit_latency,
        response_latency=hit_latency,
        mshrs=cache["miss_status_entries"],
        tgts_per_mshr=CACHE_TARGETS_PER_MSHR,
        is_read_only=read_only,
    )
    if addr_ranges is not None:
        parameters["addr_ranges"] = addr_ranges
    return Cache(**parameters)


def build_processor(processor: dict, ordinal: int):
    require_keys(processor, PROCESSOR_FIELDS, f"processor {ordinal}")
    cpu_id = processor["cpu_id"]
    num_threads = processor["num_threads"]
    if not isinstance(cpu_id, int) or cpu_id < 0:
        raise ValueError(f"processor {ordinal} cpu_id is invalid")
    if not isinstance(num_threads, int) or num_threads <= 0:
        raise ValueError(f"processor {ordinal} num_threads is invalid")
    if not isinstance(processor["execution_units"], list):
        raise ValueError(f"processor {ordinal} execution units are invalid")
    caches = processor["caches"]
    if not isinstance(caches, dict) or set(caches) != {"instruction", "data"}:
        raise ValueError(f"processor {ordinal} private caches are invalid")
    for role, cache in caches.items():
        parse_cache(cache, f"processor {ordinal} {role} cache")
    pipeline = processor["pipeline"]
    if not isinstance(pipeline, dict):
        raise ValueError(f"processor {ordinal} pipeline is invalid")
    if processor["model"] == "timing_simple":
        if num_threads != 1 or pipeline:
            raise ValueError("TimingSimpleCPU requires one in-order thread")
        return RiscvTimingSimpleCPU(cpu_id=cpu_id, numThreads=num_threads)
    if processor["model"] != "o3":
        raise ValueError(f"processor {ordinal} model is unsupported")
    require_keys(pipeline, O3_PIPELINE_FIELDS, f"processor {ordinal} O3 pipeline")
    if not all(isinstance(value, int) and value > 0 for value in pipeline.values()):
        raise ValueError(f"processor {ordinal} O3 pipeline fields must be positive")
    instruction_queue = IQUnit(
        numEntries=pipeline["issue_queue_entries"],
        fuPool=build_o3_execution_units(processor["execution_units"]),
    )
    return RiscvO3CPU(
        cpu_id=cpu_id,
        numThreads=num_threads,
        fetchWidth=pipeline["fetch_width"],
        decodeWidth=pipeline["decode_width"],
        renameWidth=pipeline["rename_width"],
        dispatchWidth=pipeline["dispatch_width"],
        issueWidth=pipeline["issue_width"],
        wbWidth=pipeline["writeback_width"],
        commitWidth=pipeline["commit_width"],
        numROBEntries=pipeline["reorder_buffer_entries"],
        instQueues=[instruction_queue],
        LQEntries=pipeline["load_queue_entries"],
        SQEntries=pipeline["store_queue_entries"],
        numPhysIntRegs=pipeline["physical_integer_registers"],
        numPhysFloatRegs=pipeline["physical_float_registers"],
        numPhysVecRegs=pipeline["physical_vector_registers"],
    )


def build_system(
    projection: dict, collect_performance: bool
) -> tuple[RiscvSystem, list[LoomSpatialEngineSession]]:
    memory = projection["memory"]
    require_keys(memory, {"base", "size", "latency", "service_ticks_per_byte"}, "memory")
    host = projection["host"]
    require_keys(
        host,
        {
            "elf",
            "cpu_id",
            "entry_symbol",
            "result_address",
            "result_size",
            "value_table_address",
            "value_table_entries",
            "return_address",
        },
        "host",
    )
    dispatch = projection["dispatch"]
    require_keys(
        dispatch,
        {
            "pio_address",
            "pio_size",
            "pio_latency",
            "stack_base",
            "stack_stride",
            "logical_target_count",
            "endpoint_target_offsets",
            "endpoint_dispatch_enabled",
            "root_event_control_path",
            "root_event_trace_path",
            "targets",
        },
        "dispatch",
    )
    if not isinstance(dispatch["root_event_control_path"], str):
        raise ValueError("root event control path must be a string")
    logical_target_count = dispatch["logical_target_count"]
    endpoint_target_offsets = dispatch["endpoint_target_offsets"]
    endpoint_dispatch_enabled = dispatch["endpoint_dispatch_enabled"]
    if not isinstance(logical_target_count, int) or logical_target_count < 0:
        raise ValueError("logical dispatch target count must be nonnegative")
    if (
        not isinstance(endpoint_target_offsets, list)
        or not endpoint_target_offsets
        or not all(
            isinstance(offset, int) and offset >= 0
            for offset in endpoint_target_offsets
        )
    ):
        raise ValueError("endpoint target offsets are invalid")
    if (
        not isinstance(endpoint_dispatch_enabled, list)
        or len(endpoint_dispatch_enabled) != len(endpoint_target_offsets)
        or not all(enabled in (0, 1) for enabled in endpoint_dispatch_enabled)
    ):
        raise ValueError("endpoint dispatch flags are invalid")
    instruction_images = projection["instruction_images"]
    if not all(isinstance(path, str) and path for path in instruction_images):
        raise ValueError("instruction image paths are invalid")
    runtime_image_paths = []
    runtime_image_addresses = []
    for ordinal, image in enumerate(projection["runtime_images"]):
        require_keys(image, {"path", "address"}, f"runtime image {ordinal}")
        runtime_image_paths.append(image["path"])
        runtime_image_addresses.append(image["address"])

    system_memory = projection["system_memory"]
    require_keys(
        system_memory,
        {
            "interface_table_address",
            "interface_table_entries",
            "observation_path",
            "observations",
        },
        "system memory",
    )
    observation_addresses = []
    observation_sizes = []
    if not isinstance(system_memory["observations"], list):
        raise ValueError("System memory observations must be an array")
    for ordinal, observation in enumerate(system_memory["observations"]):
        require_keys(observation, {"address", "size"}, f"observation {ordinal}")
        observation_addresses.append(observation["address"])
        observation_sizes.append(observation["size"])

    target_cpu_ids = []
    target_image_ordinals = []
    target_entry_symbols = []
    target_bridge_addresses = []
    target_launch_addresses = []
    target_launch_sizes = []
    if not isinstance(dispatch["targets"], list):
        raise ValueError("Thread Dispatch targets must be an array")
    if not dispatch["targets"] and (
        logical_target_count != 0 or endpoint_target_offsets != [0]
        or endpoint_dispatch_enabled != [0] or dispatch["root_event_control_path"]
    ):
        raise ValueError("host-only Thread Dispatch has an active endpoint")
    for ordinal, target in enumerate(dispatch["targets"]):
        require_keys(
            target,
            {
                "cpu_id",
                "image_ordinal",
                "entry_symbol",
                "bridge_address",
                "launch_address",
                "launch_size",
            },
            f"dispatch target {ordinal}",
        )
        target_cpu_ids.append(target["cpu_id"])
        target_image_ordinals.append(target["image_ordinal"])
        target_entry_symbols.append(target["entry_symbol"])
        target_bridge_addresses.append(target["bridge_address"])
        target_launch_addresses.append(target["launch_address"])
        target_launch_sizes.append(target["launch_size"])
    if logical_target_count > len(target_cpu_ids) or any(
        offset + logical_target_count > len(target_cpu_ids)
        for offset in endpoint_target_offsets
    ):
        raise ValueError("endpoint target range exceeds the dispatch table")

    line_sizes = {
        processor["caches"][role]["line_bytes"]
        for processor in projection["processors"]
        for role in ("instruction", "data")
    } | {bridge["cache"]["line_bytes"] for bridge in projection["bridges"]}
    if len(line_sizes) != 1:
        raise ValueError("System caches disagree on one physical line size")

    system = RiscvSystem()
    system.cache_line_size = line_sizes.pop()
    system.clk_domain = SrcClockDomain(
        clock=projection["clock"], voltage_domain=VoltageDomain()
    )
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange(start=memory["base"], size=memory["size"])]
    system.workload = LoomRiscvDeploymentWorkload(
        bootloader=host["elf"],
        host_cpu_id=host["cpu_id"],
        host_entry_symbol=host["entry_symbol"],
        host_dispatch_address=dispatch["pio_address"],
        host_memory_table_address=system_memory["interface_table_address"],
        host_memory_table_entries=system_memory["interface_table_entries"],
        host_result_address=host["result_address"],
        host_result_size=host["result_size"],
        host_value_table_address=host["value_table_address"],
        host_value_table_entries=host["value_table_entries"],
        host_return_address=host["return_address"],
        stack_base=dispatch["stack_base"],
        stack_stride=dispatch["stack_stride"],
        instruction_images=instruction_images,
        runtime_images=runtime_image_paths,
        runtime_image_addresses=runtime_image_addresses,
        memory_observation_path=system_memory["observation_path"],
        memory_observation_addresses=observation_addresses,
        memory_observation_sizes=observation_sizes,
        target_cpu_ids=target_cpu_ids,
        target_image_ordinals=target_image_ordinals,
        target_entry_symbols=target_entry_symbols,
        target_bridge_addresses=target_bridge_addresses,
        target_launch_addresses=target_launch_addresses,
        target_launch_sizes=target_launch_sizes,
    )
    system.membus = SystemXBar()
    system.system_port = system.membus.cpu_side_ports

    # Device apertures are physically uncacheable: the private L1 caches must
    # never absorb a Thread Dispatch or Spatial bridge access.
    device_ranges = [
        AddrRange(start=dispatch["pio_address"], size=dispatch["pio_size"])
    ] + [
        AddrRange(start=bridge["pio_address"], size=bridge["pio_size"])
        for bridge in projection["bridges"]
    ]

    processors = []
    for ordinal, processor in enumerate(projection["processors"]):
        cpu = build_processor(processor, ordinal)
        cpu.createInterruptController()
        cpu.createThreads()
        cpu.mmu.pma_checker = PMAChecker(uncacheable=device_ranges)
        cpu.icache = build_cache(processor["caches"]["instruction"], True)
        cpu.dcache = build_cache(processor["caches"]["data"], False)
        cpu.icache_port = cpu.icache.cpu_side
        cpu.dcache_port = cpu.dcache.cpu_side
        cpu.icache.mem_side = system.membus.cpu_side_ports
        cpu.dcache.mem_side = system.membus.cpu_side_ports
        processors.append(cpu)
    system.cpu = processors

    service_cost = memory["service_ticks_per_byte"]
    if not isinstance(service_cost, int) or isinstance(service_cost, bool) or service_cost <= 0:
        raise ValueError("shared System memory has no finite acceptance service cost")
    bandwidth = 1e12 / service_cost
    system.memory = SimpleMemory(range=system.mem_ranges[0], latency=memory["latency"],
                                 bandwidth=f"{bandwidth:.17g}B/s")
    # This zero-delay monitor exposes the native accepted-request probe. Disable
    # its unrelated histograms; the sole maintained observer integrates service.
    system.memory_monitor = CommMonitor(
        disable_burst_length_hists=True, disable_bandwidth_hists=True,
        disable_latency_hists=True, disable_itt_dists=True,
        disable_outstanding_hists=True, disable_transaction_hists=True,
        disable_addr_dists=True,
    )
    system.memory_monitor.cpu_side_port = system.membus.mem_side_ports
    system.memory_monitor.mem_side_port = system.memory.port
    system.memory_service = LoomMemoryServiceProbe(
        manager=[system.memory_monitor], service_ticks_per_byte=service_cost,
    )

    system.loom_thread_dispatch = LoomThreadDispatch(
        pio_addr=dispatch["pio_address"],
        pio_latency=dispatch["pio_latency"],
        workload=system.workload,
        memory_service=system.memory_service,
        root_event_control_path=dispatch["root_event_control_path"],
        root_event_trace_path=dispatch["root_event_trace_path"],
        logical_target_count=logical_target_count,
        endpoint_target_offsets=endpoint_target_offsets,
        endpoint_dispatch_enabled=endpoint_dispatch_enabled,
    )
    system.loom_thread_dispatch.pio = system.membus.mem_side_ports

    engine_members = {}
    for bridge in projection["bridges"]:
        if bridge["engine_socket"]:
            engine_members.setdefault(bridge["engine_socket"], []).append(bridge)
    engine_sessions = {
        socket: LoomSpatialEngineSession(engine_socket=socket, bridge_count=len(members))
        for socket, members in engine_members.items()
    }
    sessions = list(engine_sessions.values())
    if sessions:
        system.loom_engine_sessions = sessions
    bridges = []
    for bridge in projection["bridges"]:
        device = LoomSpatialBridge(
            pio_addr=bridge["pio_address"],
            pio_size=bridge["pio_size"],
            pio_latency=bridge["pio_latency"],
            session_ordinal=bridge["session_ordinal"],
            engine_session=engine_sessions.get(bridge["engine_socket"], NULL),
            result_path=bridge["result_path"],
            max_message_bytes=bridge["maximum_message_bytes"],
            collect_performance=collect_performance,
        )
        device.pio = system.membus.mem_side_ports
        device.cache = build_cache(
            parse_cache(bridge["cache"], "Spatial bridge cache"),
            False,
            addr_ranges=system.mem_ranges,
        )
        device.dma = device.cache.cpu_side
        device.cache.mem_side = system.membus.cpu_side_ports
        bridges.append(device)
    system.loom_bridges = bridges
    return system, sessions


def main() -> None:
    arguments = parse_arguments()
    projection = load_projection(pathlib.Path(arguments.projection))
    verify_running_binary(projection["gem5_binary_sha256"])
    diagnostics = arguments.performance_profile is not None
    configuration_started = time.monotonic_ns() if diagnostics else None
    has_external_engines = any(
        bridge.get("engine_socket") for bridge in projection["bridges"]
    ) and not any(bridge.get("engine_command") for bridge in projection["bridges"])
    has_managed_engines = any(
        bridge.get("engine_command") for bridge in projection["bridges"]
    )
    engine_cpu_before = (
        resource.getrusage(resource.RUSAGE_CHILDREN)
        if diagnostics and has_managed_engines
        else None
    )
    engine_readiness_cpu_before = (
        resource.getrusage(resource.RUSAGE_SELF) if diagnostics else None
    )
    engine_readiness_started = time.monotonic_ns() if diagnostics else None
    engines = start_engines(
        projection["bridges"], len(projection["dispatch"]["targets"])
    )
    engine_readiness_finished = time.monotonic_ns() if diagnostics else None
    engine_readiness_cpu_after = (
        resource.getrusage(resource.RUSAGE_SELF) if diagnostics else None
    )
    performance = None
    try:
        m5.ticks.setGlobalFrequency("1ps")
        system, engine_sessions = build_system(projection, diagnostics)
        Root(full_system=True, system=system)
        m5.instantiate()
        memory_capacity = projection["memory"]
        if system.memory.bandwidth.getValue() != memory_capacity["service_ticks_per_byte"]:
            raise RuntimeError("native SimpleMemory bandwidth differs from its capacity contract")
        configuration_finished = time.monotonic_ns() if diagnostics else None
        entry_tick = int(m5.curTick())
        system.memory_service.beginWindow()
        simulation_cpu_before = (
            resource.getrusage(resource.RUSAGE_SELF) if diagnostics else None
        )
        simulation_started = time.monotonic_ns() if diagnostics else None
        deadline = entry_tick + projection["maximum_ticks"]
        while True:
            remaining = max(0, deadline - int(m5.curTick()))
            event = m5.simulate(remaining)
            if not any(
                session.isAdvanceExit(event.getCause())
                for session in engine_sessions
            ):
                break
            for session in engine_sessions:
                session.advance()
            if int(m5.curTick()) >= deadline:
                event = m5.simulate(0)
                break
        simulation_finished = time.monotonic_ns() if diagnostics else None
        simulation_cpu_after = (
            resource.getrusage(resource.RUSAGE_SELF) if diagnostics else None
        )
        finish_engines(engines)
        observation_cpu_before = (
            resource.getrusage(resource.RUSAGE_SELF) if diagnostics else None
        )
        observation_started = time.monotonic_ns() if diagnostics else None
        bridge_statistics = None
        if diagnostics:
            m5.stats.dump()
            bridge_statistics = read_bridge_statistics(
                pathlib.Path(m5.options.outdir, "stats.txt")
            )
        system.workload.writeMemoryObservations()
        memory_activity = {"occupied_ticks": int(system.memory_service.occupiedTicks())}
        result = {
            "schema": "loom.gem5_system_attempt.2",
            "memory_activity": memory_activity,
            "entry_tick": entry_tick,
            "exit_tick": int(m5.curTick()),
            "cause": event.getCause(),
        }
        pathlib.Path(arguments.result).write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        if diagnostics:
            observation_finished = time.monotonic_ns()
            observation_cpu_after = resource.getrusage(resource.RUSAGE_SELF)
            if None in (
                configuration_started,
                configuration_finished,
                engine_readiness_cpu_before,
                engine_readiness_started,
                engine_readiness_finished,
                engine_readiness_cpu_after,
                simulation_cpu_before,
                simulation_started,
                simulation_finished,
                simulation_cpu_after,
                observation_cpu_before,
                observation_started,
                bridge_statistics,
            ):
                raise RuntimeError("gem5 performance accounting is incomplete")
            engine_readiness = {
                "wall_nanoseconds": (
                    engine_readiness_finished - engine_readiness_started
                ),
                "self_cpu_nanoseconds": elapsed_cpu_nanoseconds(
                    engine_readiness_cpu_before, engine_readiness_cpu_after
                ),
            }
            performance = {
                "schema": PERFORMANCE_PROFILE_SCHEMA,
                "configuration_wall_nanoseconds": (
                    configuration_finished - configuration_started
                ),
                "managed_engine_startup": (
                    engine_readiness if has_managed_engines else None
                ),
                "external_engine_socket_readiness": (
                    engine_readiness if has_external_engines else None
                ),
                "simulation_wall_nanoseconds": (
                    simulation_finished - simulation_started
                ),
                "gem5_simulation_cpu_nanoseconds": elapsed_cpu_nanoseconds(
                    simulation_cpu_before, simulation_cpu_after
                ),
                "observation_wall_nanoseconds": (
                    observation_finished - observation_started
                ),
                "observation_cpu_nanoseconds": elapsed_cpu_nanoseconds(
                    observation_cpu_before, observation_cpu_after
                ),
                **bridge_statistics,
            }
    finally:
        stop_engines(engines)
    if diagnostics:
        if performance is None:
            raise RuntimeError("gem5 attempt produced no performance observation")
        performance["engine_process_cpu_nanoseconds"] = (
            elapsed_cpu_nanoseconds(
                engine_cpu_before, resource.getrusage(resource.RUSAGE_CHILDREN)
            )
            if engine_cpu_before is not None
            else None
        )
        pathlib.Path(arguments.performance_profile).write_text(
            json.dumps(performance, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__m5_main__":
    main()
