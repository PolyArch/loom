#!/usr/bin/env python3

import argparse
import json
import pathlib
import struct
import subprocess
import sys


def write_meta(path: pathlib.Path, result: dict) -> None:
    lines = [
        f"success={1 if result.get('success') else 0}",
        f"cycle_count={int(result.get('cycle_count', 0))}",
        f"error_message={str(result.get('error_message', ''))}",
        f"trace_path={str(result.get('trace_path', ''))}",
        f"stat_path={str(result.get('stat_path', ''))}",
    ]
    for output in result.get("outputs", []):
        lines.append(f"output_slot={int(output['slot'])}")
    for memory in result.get("memory_regions", []):
        lines.append(f"memory_slot={int(memory['slot'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_output_files(reply_dir: pathlib.Path, result: dict) -> None:
    for output in result.get("outputs", []):
        slot = int(output["slot"])
        data = [int(x) for x in output.get("data", [])]
        tags = [int(x) for x in output.get("tags", [])]
        data_path = reply_dir / f"output.slot{slot}.data.bin"
        tag_path = reply_dir / f"output.slot{slot}.tags.bin"
        with data_path.open("wb") as f:
            for value in data:
                f.write(struct.pack("<Q", value))
        with tag_path.open("wb") as f:
            for tag in tags:
                f.write(struct.pack("<H", tag))

    for memory in result.get("memory_regions", []):
        slot = int(memory["slot"])
        bytes_path = reply_dir / f"memory.slot{slot}.bin"
        bytes_path.write_bytes(bytes(int(x) & 0xFF for x in memory.get("bytes", [])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcc", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--reply-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    reply_dir = pathlib.Path(args.reply_dir)
    work_dir = pathlib.Path(args.work_dir)
    reply_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    result_json = work_dir / "runtime-result.json"
    trace_path = work_dir / "runtime.trace"
    stat_path = work_dir / "runtime.stat"

    cmd = [
        args.fcc,
        "-o",
        str(work_dir),
        "--runtime-manifest",
        args.runtime_manifest,
        "--runtime-request",
        args.request,
        "--runtime-result",
        str(result_json),
        "--runtime-trace",
        str(trace_path),
        "--runtime-stat",
        str(stat_path),
    ]
    completed = subprocess.run(cmd, check=False)
    if not result_json.exists():
        return completed.returncode or 1

    result = json.loads(result_json.read_text(encoding="utf-8"))
    write_meta(reply_dir / "reply.meta", result)
    write_output_files(reply_dir, result)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
