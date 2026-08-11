#!/usr/bin/env python3
"""Instantiate one exact Loom RISC-V machine-mode gem5 projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time

import m5
from m5.objects import (
    AddrRange,
    LoomRiscvDeploymentWorkload,
    LoomSpatialBridge,
    LoomThreadDispatch,
    RiscvSystem,
    RiscvTimingSimpleCPU,
    Root,
    SimpleMemory,
    SrcClockDomain,
    SystemXBar,
    VoltageDomain,
)


CONFIG_SCHEMA = "loom.gem5_system_projection.2"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", required=True)
    parser.add_argument("--result", required=True)
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


def start_engines(bridges: list[dict]) -> list[subprocess.Popen]:
    processes: list[subprocess.Popen] = []
    try:
        for ordinal, bridge in enumerate(bridges):
            require_keys(
                bridge,
                {
                    "pio_address",
                    "pio_size",
                    "pio_latency",
                    "engine_socket",
                    "engine_command",
                    "result_path",
                    "maximum_message_bytes",
                },
                f"bridge {ordinal}",
            )
            command = bridge["engine_command"]
            if not isinstance(command, list) or not all(
                isinstance(item, str) and item for item in command
            ):
                raise ValueError(f"bridge {ordinal} engine command is invalid")
            socket_path = pathlib.Path(bridge["engine_socket"])
            if not command:
                if not socket_path.is_socket():
                    raise RuntimeError(
                        f"bridge {ordinal} external engine socket is unavailable"
                    )
                continue
            socket_path.unlink(missing_ok=True)
            processes.append(subprocess.Popen(command))
            deadline = time.monotonic() + 10.0
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
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def finish_engines(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        try:
            status = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Spatial engine did not exit after completion") from error
        if status != 0:
            raise RuntimeError(f"Spatial engine exited with status {status}")


def build_system(projection: dict) -> RiscvSystem:
    memory = projection["memory"]
    require_keys(memory, {"base", "size", "latency"}, "memory")
    host = projection["host"]
    require_keys(host, {"elf", "cpu_id", "entry_symbol"}, "host")
    dispatch = projection["dispatch"]
    require_keys(
        dispatch,
        {
            "pio_address",
            "pio_latency",
            "stack_base",
            "stack_stride",
            "targets",
        },
        "dispatch",
    )
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
    if not isinstance(dispatch["targets"], list) or not dispatch["targets"]:
        raise ValueError("Thread Dispatch requires at least one target")
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

    system = RiscvSystem()
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

    processors = []
    for ordinal, processor in enumerate(projection["processors"]):
        require_keys(processor, {"cpu_id"}, f"processor {ordinal}")
        cpu = RiscvTimingSimpleCPU(cpu_id=processor["cpu_id"])
        cpu.createInterruptController()
        cpu.createThreads()
        cpu.icache_port = system.membus.cpu_side_ports
        cpu.dcache_port = system.membus.cpu_side_ports
        processors.append(cpu)
    system.cpu = processors

    system.memory = SimpleMemory(
        range=system.mem_ranges[0], latency=memory["latency"]
    )
    system.memory.port = system.membus.mem_side_ports

    system.loom_thread_dispatch = LoomThreadDispatch(
        pio_addr=dispatch["pio_address"],
        pio_latency=dispatch["pio_latency"],
        workload=system.workload,
    )
    system.loom_thread_dispatch.pio = system.membus.mem_side_ports

    bridges = []
    for bridge in projection["bridges"]:
        device = LoomSpatialBridge(
            pio_addr=bridge["pio_address"],
            pio_size=bridge["pio_size"],
            pio_latency=bridge["pio_latency"],
            engine_socket=bridge["engine_socket"],
            result_path=bridge["result_path"],
            max_message_bytes=bridge["maximum_message_bytes"],
        )
        device.pio = system.membus.mem_side_ports
        device.dma = system.membus.cpu_side_ports
        bridges.append(device)
    system.loom_bridges = bridges
    return system


def main() -> None:
    arguments = parse_arguments()
    projection = load_projection(pathlib.Path(arguments.projection))
    verify_running_binary(projection["gem5_binary_sha256"])
    engines = start_engines(projection["bridges"])
    try:
        system = build_system(projection)
        Root(full_system=True, system=system)
        m5.instantiate()
        entry_tick = int(m5.curTick())
        event = m5.simulate(projection["maximum_ticks"])
        finish_engines(engines)
        system.workload.writeMemoryObservations()
        result = {
            "schema": "loom.gem5_system_attempt.1",
            "entry_tick": entry_tick,
            "exit_tick": int(m5.curTick()),
            "cause": event.getCause(),
        }
        pathlib.Path(arguments.result).write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    finally:
        stop_engines(engines)


if __name__ == "__m5_main__":
    main()
