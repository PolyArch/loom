#!/usr/bin/env python3
"""Run the real Loom Host-to-Spatial gem5 bridge path."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import socket
import struct
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_SCRIPT = REPOSITORY_ROOT / "runtime" / "gem5" / "configure_loom_system.py"
M5OP_SOURCE = REPOSITORY_ROOT / "externals" / "gem5" / "util" / "m5" / "src" / "abi" / "riscv" / "m5op.S"
M5_INCLUDE = REPOSITORY_ROOT / "externals" / "gem5" / "include"
TEST_RUN_ROOT = REPOSITORY_ROOT / "build" / "test-runs"

WIRE_MAGIC = b"LGB1"
RESULT_MAGIC = b"LGR1"
RESULT_COLLECTION_MAGIC = b"LGC1"
SPATIAL_LAUNCH_MAGIC = b"LGL2"
INVOCATION_RESULT_MAGIC = b"LGX3"
WIRE_HEADER = struct.Struct(">4sIQQ")
RESULT_HEADER = struct.Struct(">4sIQQQ")
RESULT_COLLECTION_HEADER = struct.Struct(">4sQ")
SPATIAL_LAUNCH_HEADER = struct.Struct(">4sQQQ")
INVOCATION_RESULT_HEADER = struct.Struct("<4sQQQQ32s")
MEMORY_REQUEST_HEADER = struct.Struct(">IQQQQ")
MEMORY_RESPONSE_HEADER = struct.Struct(">QIQ")
COMPLETION_HEADER = struct.Struct(">QIQ")

SPATIAL_LAUNCH = 0
MEMORY_REQUEST = 1
MEMORY_RESPONSE = 2
COMPLETION = 4
MEMORY_READ = 0
MEMORY_WRITE = 1

HOST_LOAD_ADDRESS = 0x80000000
INSTRUCTION_LOAD_ADDRESS = 0x80100000
MEMORY_BASE = 0x80000000
MEMORY_SIZE = 0x04000000
LAUNCH_ADDRESS = 0x82000000
SECOND_LAUNCH_ADDRESS = 0x82004000
EXTERNAL_VALUE_ADDRESS = 0x82001000
SYSTEM_MEMORY_ADDRESS = 0x82002000
MEMORY_TABLE_ADDRESS = 0x82003000
BRIDGE_ADDRESS = 0x10000000
SECOND_BRIDGE_ADDRESS = 0x10001000
DISPATCH_ADDRESS = 0x10002000
STACK_BASE = 0x83F00000
STACK_STRIDE = 0x00010000
EXPECTED_VALUE = 0x1122334455667788
EXPECTED_SYSTEM_MEMORY = 0x8877665544332211
ACC_CORE_OCCURRENCE_KIND = 9


def acc_core_reference(entity_id: int) -> str:
    return struct.pack(">IQ", ACC_CORE_OCCURRENCE_KIND, entity_id).hex()


def spatial_execution_context_key(entity_id: int) -> str:
    acc_core = bytes.fromhex(acc_core_reference(entity_id))
    mapping_identity = bytes([entity_id + 1]) * 32
    return (
        struct.pack(">I", 1)
        + struct.pack(">Q", len(acc_core))
        + acc_core
        + struct.pack(">Q", len(mapping_identity))
        + mapping_identity
    ).hex()
EXPECTED_RESULT = b"loom-gem5-system-smoke"
EXPECTED_LAUNCH = b"loom-spatial-launch-v1"
INITIAL_SYSTEM_MEMORY = b"loommem0"

HOST_SOURCE = f"""
.section .text,"ax",@progbits
.align 2
.globl loom_host_entry
.type loom_host_entry,@function
loom_host_entry:
  mv s0, a0
  mv s1, a2
  mv s2, a3
  mv s3, a1
  li t0, 2
  bne s3, t0, 3f
  li t0, 1
  bne s2, t0, 3f
  lw t0, 0(s1)
  li t1, 0x494d474c
  bne t0, t1, 3f
  lw t0, 4(s1)
  li t1, 1
  bne t0, t1, 3f
  ld t0, 8(s1)
  bne t0, t1, 3f
  ld t0, 32(s1)
  li t1, 8
  bltu t0, t1, 3f
  lw t0, 40(s1)
  andi t0, t0, 2
  beqz t0, 3f
  ld t0, 24(s1)
  li t1, {EXPECTED_SYSTEM_MEMORY}
  sd t1, 0(t0)
  li s4, 0
4:
  bgeu s4, s3, 6f
  sw s4, 0(s0)
  sw zero, 4(s0)
  li t0, 2
  sw t0, 8(s0)
  li t0, 1
  fence iorw, iorw
  sw t0, 8(s0)
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, 3f
  addi s4, s4, 1
  j 4b
6:
  li s4, 0
7:
  bgeu s4, s3, 5f
  sw s4, 0(s0)
  sw zero, 4(s0)
1:
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, 3f
  andi t1, t0, 2
  beqz t1, 1b
  fence iorw, iorw
  addi s4, s4, 1
  j 7b
5:
  li t2, {EXTERNAL_VALUE_ADDRESS}
  ld t3, 0(t2)
  li t4, {EXPECTED_VALUE}
  bne t3, t4, 3f
  li a0, 0
  call m5_exit
2:
  wfi
  j 2b
3:
  li a0, 0
  li a1, 1
  call m5_fail
  j 2b
.size loom_host_entry, .-loom_host_entry
"""

INSTRUCTION_SOURCE = """
.section .text,"ax",@progbits
.align 2
.globl __loom_thread_entry_0
.type __loom_thread_entry_0,@function
__loom_thread_entry_0:
  srli t0, a1, 32
  sw a1, 20(a0)
  sw t0, 24(a0)
  sw a2, 28(a0)
  li t0, 1
  fence iorw, iorw
  sw t0, 4(a0)
1:
  lw t0, 0(a0)
  andi t1, t0, 4
  bnez t1, 3f
  andi t1, t0, 2
  beqz t1, 1b
  fence iorw, iorw
  li t0, 1
  sw t0, 0(a3)
2:
  wfi
  j 2b
3:
  lw t0, 8(a0)
  bnez t0, 4f
  li t0, 1
4:
  sw t0, 4(a3)
  j 2b
.size __loom_thread_entry_0, .-__loom_thread_entry_0
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gem5", type=pathlib.Path)
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--socket", type=pathlib.Path)
    parser.add_argument("--expected-launch", type=pathlib.Path, action="append")
    parser.add_argument("--trace", type=pathlib.Path, action="append")
    parser.add_argument("--bridge-ordinal", type=int, action="append")
    return parser.parse_args()


def read_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) != size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise RuntimeError("bridge connection closed before the message completed")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_message(connection: socket.socket) -> tuple[int, int, bytes]:
    magic, kind, sequence, payload_size = WIRE_HEADER.unpack(
        read_exact(connection, WIRE_HEADER.size)
    )
    if magic != WIRE_MAGIC:
        raise RuntimeError("bridge message has the wrong magic")
    return kind, sequence, read_exact(connection, payload_size)


def decode_spatial_launch_envelope(payload: bytes) -> tuple[int, bytes, bytes]:
    if len(payload) < SPATIAL_LAUNCH_HEADER.size:
        raise RuntimeError("Spatial launch envelope is truncated")
    magic, bridge_ordinal, static_size, invocation_size = (
        SPATIAL_LAUNCH_HEADER.unpack_from(payload)
    )
    if magic != SPATIAL_LAUNCH_MAGIC:
        raise RuntimeError("Spatial launch envelope has the wrong magic")
    if static_size + invocation_size != len(payload) - SPATIAL_LAUNCH_HEADER.size:
        raise RuntimeError("Spatial launch envelope lengths are not canonical")
    static_end = SPATIAL_LAUNCH_HEADER.size + static_size
    return (
        bridge_ordinal,
        payload[SPATIAL_LAUNCH_HEADER.size : static_end],
        payload[static_end:],
    )


def invocation_result(invocation: bytes, boundary_result: bytes) -> bytes:
    return (
        INVOCATION_RESULT_HEADER.pack(
            INVOCATION_RESULT_MAGIC,
            0,
            len(invocation),
            0,
            len(boundary_result),
            bytes(32),
        )
        + invocation
        + boundary_result
    )


def decode_invocation_result(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) < INVOCATION_RESULT_HEADER.size:
        raise RuntimeError("Spatial invocation result is truncated")
    (
        magic,
        session_entry_ordinal,
        invocation_size,
        runtime_input_size,
        boundary_size,
        runtime_input_identity,
    ) = INVOCATION_RESULT_HEADER.unpack_from(payload)
    if magic != INVOCATION_RESULT_MAGIC:
        raise RuntimeError("Spatial invocation result has the wrong magic")
    if (
        session_entry_ordinal != 0
        or runtime_input_size != 0
        or runtime_input_identity != bytes(32)
        or invocation_size + runtime_input_size + boundary_size
        != len(payload) - INVOCATION_RESULT_HEADER.size
    ):
        raise RuntimeError("Spatial invocation result lengths are not canonical")
    invocation_end = INVOCATION_RESULT_HEADER.size + invocation_size
    return payload[INVOCATION_RESULT_HEADER.size : invocation_end], payload[invocation_end:]


def send_message(
    connection: socket.socket, kind: int, sequence: int, payload: bytes
) -> None:
    connection.sendall(WIRE_HEADER.pack(WIRE_MAGIC, kind, sequence, len(payload)))
    connection.sendall(payload)


def require_memory_response(
    connection: socket.socket,
    sequence: int,
    request_id: int,
    expected_data: bytes,
) -> None:
    kind, response_sequence, payload = receive_message(connection)
    if kind != MEMORY_RESPONSE or response_sequence != sequence:
        raise RuntimeError("bridge returned the wrong memory response envelope")
    if len(payload) < MEMORY_RESPONSE_HEADER.size:
        raise RuntimeError("bridge returned a truncated memory response")
    response_id, success, data_size = MEMORY_RESPONSE_HEADER.unpack_from(payload)
    data = payload[MEMORY_RESPONSE_HEADER.size :]
    if (
        response_id != request_id
        or success != 1
        or data_size != len(data)
        or data != expected_data
    ):
        raise RuntimeError("bridge returned a noncanonical memory response")


def run_engine(arguments: argparse.Namespace) -> int:
    if (
        arguments.socket is None
        or not arguments.expected_launch
        or len(arguments.expected_launch) != len(arguments.trace or [])
        or len(arguments.expected_launch) != len(arguments.bridge_ordinal or [])
        or len(set(arguments.bridge_ordinal)) != len(arguments.bridge_ordinal)
    ):
        raise RuntimeError("engine mode requires one unique launch table per bridge")
    entries = {
        ordinal: (launch, trace)
        for ordinal, launch, trace in zip(
            arguments.bridge_ordinal,
            arguments.expected_launch,
            arguments.trace,
            strict=True,
        )
    }
    arguments.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(arguments.socket))
    server.listen(len(entries))
    try:
        pending = dict(entries)
        while pending:
            connection, _ = server.accept()
            with connection:
                kind, sequence, payload = receive_message(connection)
                bridge_ordinal, launch, invocation = decode_spatial_launch_envelope(
                    payload
                )
                entry = pending.pop(bridge_ordinal, None)
                if (
                    kind != SPATIAL_LAUNCH
                    or sequence != 0
                    or entry is None
                    or launch != entry[0].read_bytes()
                    or invocation
                ):
                    raise RuntimeError("bridge launch differs from the expected payload")

                value = EXPECTED_VALUE.to_bytes(8, byteorder="little")
                write_payload = MEMORY_REQUEST_HEADER.pack(
                    MEMORY_WRITE, 7, 1, EXTERNAL_VALUE_ADDRESS, len(value)
                ) + value
                send_message(connection, MEMORY_REQUEST, sequence, write_payload)
                require_memory_response(connection, sequence, 1, b"")

                read_payload = MEMORY_REQUEST_HEADER.pack(
                    MEMORY_READ, 11, 2, EXTERNAL_VALUE_ADDRESS, len(value)
                )
                send_message(connection, MEMORY_REQUEST, sequence, read_payload)
                require_memory_response(connection, sequence, 2, value)

                result = invocation_result(invocation, EXPECTED_RESULT)
                completion = COMPLETION_HEADER.pack(13, 0, len(result))
                send_message(connection, COMPLETION, sequence, completion + result)
                entry[1].write_text(
                    json.dumps(
                        {
                            "bridge_ordinal": bridge_ordinal,
                            "launch_sha256": hashlib.sha256(launch).hexdigest(),
                            "memory_address": EXTERNAL_VALUE_ADDRESS,
                            "memory_value": EXPECTED_VALUE,
                            "sequence": sequence,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        server.close()
        arguments.socket.unlink(missing_ok=True)
    return 0


def compile_image(
    compiler: str,
    source_path: pathlib.Path,
    output_path: pathlib.Path,
    load_address: int,
    entry_symbol: str,
    include_m5ops: bool,
) -> None:
    linker_script = output_path.with_suffix(".ld")
    linker_script.write_text(
        f"""
OUTPUT_ARCH(riscv)
ENTRY({entry_symbol})
SECTIONS
{{
  . = 0x{load_address:x};
  .text : {{ *(.text .text.*) }}
  .rodata : {{ *(.rodata .rodata.*) }}
  .data : {{ *(.data .data.*) }}
  .bss : {{ *(.bss .bss.* COMMON) }}
}}
""",
        encoding="ascii",
    )
    sources = [str(source_path)]
    if include_m5ops:
        sources.append(str(M5OP_SOURCE))
    command = [
        compiler,
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-nostdlib",
        "-static",
        "-fuse-ld=lld",
        f"-Wl,-T,{linker_script}",
        "-Wl,--build-id=none",
        "-Wl,--no-relax",
        "-Wl,-z,max-page-size=4096",
        "-I",
        str(M5_INCLUDE),
        *sources,
        "-o",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def binary_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_bridge_results(path: pathlib.Path) -> list[tuple[int, int, int, bytes]]:
    data = path.read_bytes()
    if len(data) < RESULT_COLLECTION_HEADER.size:
        raise RuntimeError("bridge result collection is truncated")
    magic, count = RESULT_COLLECTION_HEADER.unpack_from(data)
    if magic != RESULT_COLLECTION_MAGIC:
        raise RuntimeError("bridge result collection has the wrong magic")
    offset = RESULT_COLLECTION_HEADER.size
    results = []
    for _ in range(count):
        if len(data) - offset < RESULT_HEADER.size:
            raise RuntimeError("bridge result collection member is truncated")
        magic, status, completion_tick, sequence, payload_size = (
            RESULT_HEADER.unpack_from(data, offset)
        )
        offset += RESULT_HEADER.size
        payload = data[offset : offset + payload_size]
        offset += payload_size
        if magic != RESULT_MAGIC or len(payload) != payload_size:
            raise RuntimeError("bridge result collection member is not canonical")
        results.append((status, completion_tick, sequence, payload))
    if offset != len(data):
        raise RuntimeError("bridge result collection has trailing bytes")
    return results


def run_smoke(arguments: argparse.Namespace) -> int:
    if arguments.gem5 is None:
        raise RuntimeError("runtime smoke requires --gem5")
    gem5 = arguments.gem5.resolve()
    if not gem5.is_file():
        raise RuntimeError(f"gem5 binary is absent: {gem5}")
    compiler = shutil.which("clang")
    if compiler is None:
        raise RuntimeError("clang is unavailable")

    TEST_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gem5-runtime-", dir=TEST_RUN_ROOT) as directory:
        root = pathlib.Path(directory)
        host_source = root / "host.S"
        instruction_source = root / "instruction.S"
        host_image = root / "host.elf"
        instruction_image = root / "instruction.elf"
        launch_paths = [
            root / f"spatial-launch-{ordinal}.bin" for ordinal in range(2)
        ]
        memory_object_path = root / "system-memory.bin"
        memory_table_path = root / "system-memory-table.bin"
        memory_observation_path = root / "system-memory.result"
        socket_path = root / "spatial-bridge-session.sock"
        bridge_result_paths = [
            root / f"spatial-result-{ordinal}.bin" for ordinal in range(2)
        ]
        system_result_path = root / "system-result.json"
        engine_trace_paths = [
            root / f"engine-trace-{ordinal}.json" for ordinal in range(2)
        ]
        projection_path = root / "projection.json"
        gem5_log_path = root / "gem5.log"

        host_source.write_text(HOST_SOURCE, encoding="ascii")
        instruction_source.write_text(INSTRUCTION_SOURCE, encoding="ascii")
        for launch_path in launch_paths:
            launch_path.write_bytes(EXPECTED_LAUNCH)
        memory_object_path.write_bytes(INITIAL_SYSTEM_MEMORY)
        memory_table_path.write_bytes(
            struct.pack(
                "<4sIQQQQII",
                b"LGMI",
                1,
                1,
                0,
                SYSTEM_MEMORY_ADDRESS,
                len(INITIAL_SYSTEM_MEMORY),
                3,
                0,
            )
        )
        compile_image(
            compiler,
            host_source,
            host_image,
            HOST_LOAD_ADDRESS,
            "loom_host_entry",
            True,
        )
        compile_image(
            compiler,
            instruction_source,
            instruction_image,
            INSTRUCTION_LOAD_ADDRESS,
            "__loom_thread_entry_0",
            False,
        )

        engine_command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--engine",
            "--socket",
            str(socket_path),
        ]
        for ordinal in range(2):
            engine_command.extend(
                [
                    "--expected-launch",
                    str(launch_paths[ordinal]),
                    "--trace",
                    str(engine_trace_paths[ordinal]),
                    "--bridge-ordinal",
                    str(ordinal),
                ]
            )
        engine_commands = [engine_command, []]
        projection = {
            "schema": "loom.gem5_system_projection.10",
            "gem5_binary_sha256": binary_digest(gem5),
            "clock": "1GHz",
            "memory": {"base": MEMORY_BASE, "size": MEMORY_SIZE, "latency": "20ns"},
            "host": {
                "elf": str(host_image),
                "cpu_id": 0,
                "entry_symbol": "loom_host_entry",
                "result_address": 0,
                "result_size": 0,
                "return_address": HOST_LOAD_ADDRESS,
            },
            "instruction_images": [str(instruction_image)],
            "runtime_images": [
                {"path": str(launch_paths[0]), "address": LAUNCH_ADDRESS},
                {"path": str(launch_paths[1]), "address": SECOND_LAUNCH_ADDRESS},
                {"path": str(memory_object_path), "address": SYSTEM_MEMORY_ADDRESS},
                {"path": str(memory_table_path), "address": MEMORY_TABLE_ADDRESS},
            ],
            "system_memory": {
                "interface_table_address": MEMORY_TABLE_ADDRESS,
                "interface_table_entries": 1,
                "observation_path": str(memory_observation_path),
                "observations": [
                    {
                        "address": SYSTEM_MEMORY_ADDRESS,
                        "size": len(INITIAL_SYSTEM_MEMORY),
                    }
                ],
            },
            "dispatch": {
                "pio_address": DISPATCH_ADDRESS,
                "pio_latency": "10ns",
                "stack_base": STACK_BASE,
                "stack_stride": STACK_STRIDE,
                "targets": [
                    {
                        "cpu_id": 1,
                        "image_ordinal": 0,
                        "entry_symbol": "__loom_thread_entry_0",
                        "bridge_address": BRIDGE_ADDRESS,
                        "launch_address": LAUNCH_ADDRESS,
                        "launch_size": len(EXPECTED_LAUNCH),
                    },
                    {
                        "cpu_id": 2,
                        "image_ordinal": 0,
                        "entry_symbol": "__loom_thread_entry_0",
                        "bridge_address": SECOND_BRIDGE_ADDRESS,
                        "launch_address": SECOND_LAUNCH_ADDRESS,
                        "launch_size": len(EXPECTED_LAUNCH),
                    },
                ],
            },
            "processors": [
                {
                    "cpu_id": cpu_id,
                    "model": "timing_simple",
                    "num_threads": 1,
                    "execution_units": [
                        {
                            "operation_classes": ["IntAlu"],
                            "count": 1,
                            "latency_cycles": 1,
                            "initiation_interval": 1,
                        }
                    ],
                    "pipeline": {},
                }
                for cpu_id in range(3)
            ],
            "bridges": [
                {
                    "dispatch_target_ordinals": [0],
                    "acc_core_ref": acc_core_reference(0),
                    "execution_context_keys": [spatial_execution_context_key(0)],
                    "spatial_workloads": [hashlib.sha256(EXPECTED_LAUNCH).hexdigest()],
                    "pio_address": BRIDGE_ADDRESS,
                    "pio_size": 4096,
                    "pio_latency": "10ns",
                    "session_ordinal": 0,
                    "engine_socket": str(socket_path),
                    "engine_command": engine_commands[0],
                    "result_path": str(bridge_result_paths[0]),
                    "maximum_message_bytes": 1048576,
                    "maximum_invocations": 16,
                },
                {
                    "dispatch_target_ordinals": [1],
                    "acc_core_ref": acc_core_reference(1),
                    "execution_context_keys": [spatial_execution_context_key(1)],
                    "spatial_workloads": [hashlib.sha256(EXPECTED_LAUNCH).hexdigest()],
                    "pio_address": SECOND_BRIDGE_ADDRESS,
                    "pio_size": 4096,
                    "pio_latency": "10ns",
                    "session_ordinal": 1,
                    "engine_socket": str(socket_path),
                    "engine_command": engine_commands[1],
                    "result_path": str(bridge_result_paths[1]),
                    "maximum_message_bytes": 1048576,
                    "maximum_invocations": 16,
                },
            ],
            "maximum_ticks": 100000000,
        }
        projection_path.write_text(
            json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        command = [
            str(gem5),
            f"--outdir={root / 'gem5-output'}",
            str(CONFIG_SCRIPT),
            "--projection",
            str(projection_path),
            "--result",
            str(system_result_path),
        ]
        with gem5_log_path.open("w", encoding="utf-8") as gem5_log:
            completed = subprocess.run(
                command,
                text=True,
                stdout=gem5_log,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
        if completed.returncode != 0:
            sys.stderr.write(gem5_log_path.read_text(encoding="utf-8"))
            raise RuntimeError(f"gem5 runtime smoke exited with {completed.returncode}")

        system_result = json.loads(system_result_path.read_text(encoding="utf-8"))
        if system_result["schema"] != "loom.gem5_system_attempt.1":
            raise RuntimeError("gem5 system result has the wrong schema")
        if "m5_exit instruction encountered" not in system_result["cause"]:
            raise RuntimeError(f"guest did not retire normally: {system_result['cause']}")
        completion_ticks = []
        for ordinal, (bridge_result_path, engine_trace_path) in enumerate(
            zip(bridge_result_paths, engine_trace_paths, strict=True)
        ):
            bridge_results = decode_bridge_results(bridge_result_path)
            if len(bridge_results) != 1:
                raise RuntimeError("Spatial bridge did not publish one invocation")
            status, completion_tick, sequence, result = bridge_results[0]
            invocation, boundary_result = decode_invocation_result(result)
            if (
                status != 0
                or sequence != 0
                or invocation
                or boundary_result != EXPECTED_RESULT
            ):
                raise RuntimeError(
                    "Spatial bridge result differs from the engine completion"
                )
            if not (0 < completion_tick <= system_result["exit_tick"]):
                raise RuntimeError(
                    "Spatial completion is outside the gem5 execution interval"
                )
            completion_ticks.append(completion_tick)
            trace = json.loads(engine_trace_path.read_text(encoding="utf-8"))
            if (
                trace["bridge_ordinal"] != ordinal
                or trace["launch_sha256"]
                != hashlib.sha256(EXPECTED_LAUNCH).hexdigest()
                or trace["memory_value"] != EXPECTED_VALUE
            ):
                raise RuntimeError("engine trace differs from its exact bridge entry")
        expected_memory = EXPECTED_SYSTEM_MEMORY.to_bytes(8, byteorder="little")
        expected_observation = (
            b"LGM1"
            + struct.pack(">Q", 1)
            + struct.pack(
                ">QQ", SYSTEM_MEMORY_ADDRESS, len(expected_memory)
            )
            + expected_memory
        )
        if memory_observation_path.read_bytes() != expected_observation:
            raise RuntimeError("System memory observation differs from the guest write")

        print(
            json.dumps(
                {
                    "bridge_completion_ticks": completion_ticks,
                    "cause": system_result["cause"],
                    "exit_tick": system_result["exit_tick"],
                    "memory_value": EXPECTED_VALUE,
                    "system_memory_value": EXPECTED_SYSTEM_MEMORY,
                },
                sort_keys=True,
            )
        )
    return 0


def main() -> int:
    arguments = parse_arguments()
    return run_engine(arguments) if arguments.engine else run_smoke(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
