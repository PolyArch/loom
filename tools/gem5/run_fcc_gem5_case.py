#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEM5_BIN = REPO_ROOT / "build/RISCV/gem5.opt"
GEM5_REBUILD_INPUTS = [
    REPO_ROOT / "src/gem5dev",
    REPO_ROOT / "lib/fcc/Simulator",
    REPO_ROOT / "include/fcc/Simulator",
]


def load_single(path_glob: str, base_dir: pathlib.Path) -> pathlib.Path:
    matches = sorted(base_dir.glob(path_glob))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one match for {path_glob}, got {matches}")
    return matches[0]


def pack_values(values, elem_size):
    out = bytearray()
    for value in values:
        out.extend(int(value).to_bytes(elem_size, "little", signed=False))
    return list(out)


def parse_gem5_stats(stats_path: pathlib.Path):
    stats = {}
    for line in stats_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        if stripped.startswith("Begin Simulation Statistics"):
            continue
        if stripped.startswith("End Simulation Statistics"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        value = parts[1]
        stats[key] = value
    return stats


def build_cpu_perf_summary(stats: dict, report: dict):
    summary = {
        "tick": int(report.get("tick", 0)),
    }
    keys = [
        "simTicks",
        "simInsts",
        "hostSeconds",
        "system.cpu.numCycles",
        "system.cpu.cpi",
        "system.cpu.ipc",
        "system.cpu.exec_context.thread_0.numBusyCycles",
        "system.cpu.exec_context.thread_0.numIdleCycles",
        "system.mem_ctrl.numReads::total",
        "system.mem_ctrl.numWrites::total",
        "system.mem_ctrl.bytesRead::total",
        "system.mem_ctrl.bytesWritten::total",
    ]
    for key in keys:
        if key in stats:
            summary[key] = stats[key]
    return summary


def parse_accel_stats(stat_path: pathlib.Path):
    stats = {}
    for line in stat_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        stats[key.strip()] = value.strip()
    return stats


def read_case_metadata(case_dir: pathlib.Path):
    runtime_manifest = load_single("*.runtime.json", case_dir)
    sim_bundle = case_dir / "sim.bundle.json"
    if not sim_bundle.exists():
        raise RuntimeError(f"missing {sim_bundle}")
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    bundle = json.loads(sim_bundle.read_text(encoding="utf-8"))
    sim_image_entry = runtime.get("sim_image_bin", "")
    if sim_image_entry:
        sim_image = pathlib.Path(sim_image_entry)
        if not sim_image.is_absolute():
            sim_image = (case_dir / sim_image).resolve()
    else:
        sim_image = load_single("*.simimage.bin", case_dir).resolve()
    if not sim_image.exists():
        raise RuntimeError(f"missing sim image {sim_image}")
    sim_image_json_entry = runtime.get("sim_image_json", "")
    if sim_image_json_entry:
        sim_image_json_path = pathlib.Path(sim_image_json_entry)
        if not sim_image_json_path.is_absolute():
            sim_image_json_path = (case_dir / sim_image_json_path).resolve()
    else:
        sim_image_json_path = load_single("*.simimage.json", case_dir).resolve()
    if not sim_image_json_path.exists():
        raise RuntimeError(f"missing sim image json {sim_image_json_path}")
    sim_image_json = json.loads(sim_image_json_path.read_text(encoding="utf-8"))
    map_json_path = load_single("*.map.json", case_dir).resolve()
    map_json = json.loads(map_json_path.read_text(encoding="utf-8"))
    return (
        runtime_manifest,
        runtime,
        sim_image,
        sim_image_json_path,
        sim_image_json,
        map_json_path,
        map_json,
        sim_bundle,
        bundle,
    )


def build_slot_maps(runtime: dict):
    scalar_slot_by_arg = {
        int(entry["arg_index"]): int(entry["slot"])
        for entry in runtime.get("scalar_args", [])
    }
    memory_slot_by_arg = {}
    for entry in runtime.get("memory_regions", []):
        memory_slot_by_arg.setdefault(int(entry["memref_arg_index"]), int(entry["slot"]))
    output_slot_by_result = {
        int(entry["result_index"]): int(entry["slot"])
        for entry in runtime.get("outputs", [])
    }
    return scalar_slot_by_arg, memory_slot_by_arg, output_slot_by_result


def build_config_patch_entries(runtime: dict, sim_image_json: dict, map_json: dict):
    slice_by_hw_node = {}
    for entry in sim_image_json.get("config_image", {}).get("config_slices", []):
        if entry.get("kind") not in ("extmemory", "memory"):
            continue
        slice_by_hw_node[int(entry["hw_node"])] = entry

    region_layout_by_memref = {}
    for memory in map_json.get("memory_regions", []):
        hw_node_id = int(memory["hw_node"])
        for region in memory.get("regions", []):
            memref_arg_index = int(region["memref_arg_index"])
            region_layout_by_memref[memref_arg_index] = {
                "hw_node_id": hw_node_id,
                "region_index": int(region["region_index"]),
            }

    patch_entries = []
    for entry in runtime.get("memory_regions", []):
        memref_arg_index = int(entry["memref_arg_index"])
        slot = int(entry["slot"])
        region_id = int(entry["region_id"])
        region_layout = region_layout_by_memref.get(memref_arg_index)
        if region_layout is None:
            raise RuntimeError(
                f"missing map memory layout for memref arg {memref_arg_index}"
            )
        hw_node_id = int(region_layout["hw_node_id"])
        config_slice = slice_by_hw_node.get(hw_node_id)
        if config_slice is None:
            raise RuntimeError(f"missing config slice for memory hw node {hw_node_id}")
        local_index = int(region_layout["region_index"])
        word_index = int(config_slice["word_offset"]) + local_index * 5 + 3
        patch_entries.append(
            {
                "memref_arg_index": memref_arg_index,
                "slot": slot,
                "region_id": region_id,
                "hw_node_id": hw_node_id,
                "word_index": word_index,
            }
        )
    return sorted(patch_entries, key=lambda entry: (entry["slot"], entry["region_id"]))


def emit_host_source(case_dir: pathlib.Path, gem5_dir: pathlib.Path, runtime: dict,
                     sim_image_json: dict, map_json: dict, bundle: dict) -> pathlib.Path:
    scalar_slot_by_arg, memory_slot_by_arg, output_slot_by_result = build_slot_maps(runtime)
    config_words = [
        int(word)
        for word in sim_image_json.get("config_image", {}).get("config_words", [])
    ]
    config_patch_entries = build_config_patch_entries(runtime, sim_image_json, map_json)

    memory_defs = []
    memory_inits = []
    memory_binds = []
    memory_checks = []

    for memory in bundle.get("memory_regions", []):
        arg_index = int(memory["memref_arg_index"])
        slot = memory_slot_by_arg[arg_index]
        elem_size = int(memory["elem_size_bytes"])
        bytes_list = pack_values(memory["values"], elem_size)
        array_name = f"mem_region_{slot}"
        init_name = f"init_mem_region_{slot}"
        memory_defs.append(
            f"static unsigned char {array_name}[{len(bytes_list)}];"
        )
        memory_defs.append(
            f"static const unsigned char {init_name}[] = "
            f"{{{', '.join(str(x) for x in bytes_list)}}};"
        )
        memory_inits.extend([
            f"  for (size_t i = 0; i < sizeof({array_name}); ++i) {{",
            f"    {array_name}[i] = {init_name}[i];",
            "  }",
        ])
        memory_binds.append(
            f"  fcc_accel_set_mem_region({slot}, {array_name}, sizeof({array_name}));"
        )

    expected_memory_by_slot = {}
    for memory in bundle.get("expected_memory_regions", []):
        arg_index = int(memory["memref_arg_index"])
        slot = memory_slot_by_arg[arg_index]
        elem_size = int(memory["elem_size_bytes"])
        bytes_list = pack_values(memory["values"], elem_size)
        expected_name = f"expected_mem_region_{slot}"
        memory_defs.append(
            f"static const unsigned char {expected_name}[] = "
            f"{{{', '.join(str(x) for x in bytes_list)}}};"
        )
        expected_memory_by_slot[slot] = (expected_name, len(bytes_list))

    for slot, (expected_name, count) in sorted(expected_memory_by_slot.items()):
        memory_checks.append(
            "\n".join([
                f"  for (size_t i = 0; i < {count}; ++i) {{",
                f"    if (mem_region_{slot}[i] != {expected_name}[i])",
                "      { fcc_uart_puts(\"FAIL\\n\");",
                f"        return 0x300 + {slot}; }}",
                "  }",
            ])
        )

    arg_lines = []
    for scalar in bundle.get("scalar_inputs", []):
        arg_index = int(scalar["arg_index"])
        slot = scalar_slot_by_arg[arg_index]
        data = scalar.get("data", [])
        if len(data) != 1:
            raise RuntimeError("gem5 baremetal host only supports one scalar value per arg slot")
        arg_lines.append(f"  fcc_accel_set_arg({slot}, {int(data[0])}ULL);")

    output_defs = []
    output_checks = []
    for expected in bundle.get("expected_outputs", []):
        result_index = int(expected["result_index"])
        slot = output_slot_by_result[result_index]
        values = [int(x) for x in expected.get("data", [])]
        tags = [int(x) for x in expected.get("tags", [0] * len(values))]
        data_name = f"expected_output_{slot}"
        tag_name = f"expected_output_tags_{slot}"
        output_defs.append(
            f"static const unsigned long long {data_name}[] = "
            f"{{{', '.join(str(x) + 'ULL' for x in values)}}};"
        )
        output_defs.append(
            f"static const unsigned short {tag_name}[] = "
            f"{{{', '.join(str(x) for x in tags)}}};"
        )
        output_checks.extend([
            f"  if (fcc_accel_output_count({slot}) != {len(values)})",
            "    { fcc_uart_puts(\"FAIL\\n\");",
            f"      return 0x200 + {slot}; }}",
            f"  for (unsigned i = 0; i < {len(values)}; ++i) {{",
            "    unsigned short tag = 0;",
            f"    unsigned long long value = fcc_accel_read_output({slot}, i, &tag);",
            f"    if (value != {data_name}[i] || tag != {tag_name}[i])",
            "      { fcc_uart_puts(\"FAIL\\n\");",
            f"        return 0x220 + {slot}; }}",
            "  }",
        ])

    lines = [
        "#include <stddef.h>",
        "#include <stdint.h>",
        '#include "fcc_accel.h"',
        "",
        "#define FCC_UART0_BASE 0x10000000UL",
        "#define FCC_UART_THR 0x0UL",
        "#define FCC_UART_LSR 0x5UL",
        "#define FCC_UART_LSR_THRE 0x20U",
        "",
        "static void fcc_uart_putc(char ch) {",
        "  volatile unsigned char *uart =",
        "      (volatile unsigned char *)(uintptr_t)FCC_UART0_BASE;",
        "  while ((uart[FCC_UART_LSR] & FCC_UART_LSR_THRE) == 0) {",
        "  }",
        "  uart[FCC_UART_THR] = (unsigned char)ch;",
        "}",
        "",
        "static void fcc_uart_puts(const char *text) {",
        "  while (*text) {",
        "    if (*text == '\\n')",
        "      fcc_uart_putc('\\r');",
        "    fcc_uart_putc(*text++);",
        "  }",
        "}",
        "",
    ]
    if config_words:
        lines.append(
            "static uint32_t fcc_runtime_config_words[] = "
            f"{{{', '.join(str(word) + 'u' for word in config_words)}}};"
        )
        lines.append(
            "static const unsigned fcc_runtime_config_word_count = "
            f"{len(config_words)};"
        )
        lines.append("")
    lines.extend(memory_defs)
    lines.extend(output_defs)
    lines.extend([
        "",
    ])
    if config_patch_entries:
        lines.extend([
            "static void patch_runtime_config_memory_bases(void) {",
        ])
        for entry in config_patch_entries:
            lines.append(
                "  fcc_runtime_config_words[{word_index}] = "
                "(uint32_t)(uintptr_t)mem_region_{slot};".format(
                    word_index=entry["word_index"], slot=entry["slot"]
                )
            )
        lines.extend([
            "}",
            "",
        ])
    lines.extend([
        "int main(void) {",
    ])
    lines.extend(memory_inits)
    lines.extend([
        "  fcc_accel_init();",
    ])
    if config_patch_entries:
        lines.append("  patch_runtime_config_memory_bases();")
    if config_words:
        lines.extend([
            "  if (fcc_runtime_config_word_count > 0)",
            "    fcc_accel_load_config(fcc_runtime_config_words,",
            "                          fcc_runtime_config_word_count);",
        ])
    lines.extend(memory_binds)
    lines.extend(arg_lines)
    lines.extend([
        "  fcc_accel_launch();",
        "  if (fcc_accel_wait() != 0)",
        "    { fcc_uart_puts(\"FAIL\\n\"); return 0x100 + (int)fcc_accel_error_code(); }",
    ])
    lines.extend(output_checks)
    lines.extend(memory_checks)
    lines.extend([
        "  fcc_uart_puts(\"PASS\\n\");",
        "  return 0;",
        "}",
        "",
    ])

    host_c = gem5_dir / "host.c"
    host_c.write_text("\n".join(lines), encoding="utf-8")
    return host_c


def build_host_elf(case_dir: pathlib.Path, gem5_dir: pathlib.Path, host_c: pathlib.Path):
    host_elf = gem5_dir / "host.elf"
    cmd = [
        "riscv64-linux-gnu-gcc",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-nostdlib",
        "-ffreestanding",
        "-fno-pie",
        "-no-pie",
        "-mcmodel=medany",
        "-msmall-data-limit=0",
        "-O2",
        "-I",
        str(case_dir),
        "-I",
        str(REPO_ROOT / "externals/gem5/include"),
        "-T",
        str(REPO_ROOT / "runtime/baremetal/fcc_baremetal.ld"),
        str(REPO_ROOT / "runtime/baremetal/crt0.S"),
        str(REPO_ROOT / "externals/gem5/util/m5/src/abi/riscv/m5op.S"),
        str(host_c),
        str(case_dir / "fcc_accel.c"),
        "-o",
        str(host_elf),
    ]
    subprocess.run(cmd, check=True)
    return host_elf


def newest_input_mtime(paths):
    newest = 0.0
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            newest = max(newest, root.stat().st_mtime)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            newest = max(newest, path.stat().st_mtime)
    return newest


def gem5_needs_rebuild(gem5_bin: pathlib.Path):
    if os.environ.get("FCC_GEM5_FORCE_REBUILD", "") not in ("", "0"):
        return True, "FCC_GEM5_FORCE_REBUILD is set"
    if not gem5_bin.exists():
        return True, f"missing {gem5_bin}"
    binary_mtime = gem5_bin.stat().st_mtime
    newest_input = newest_input_mtime(GEM5_REBUILD_INPUTS)
    if newest_input > binary_mtime:
        return True, "local gem5/simulator sources are newer than gem5.opt"
    return False, "existing gem5.opt is up to date"


def build_gem5(force_rebuild: bool):
    gem5_bin = GEM5_BIN
    if not force_rebuild:
        needs_rebuild, reason = gem5_needs_rebuild(gem5_bin)
        if not needs_rebuild:
            print(f"reusing gem5 binary: {gem5_bin} ({reason})")
            return gem5_bin
    else:
        reason = "requested by --rebuild-gem5"

    print(f"rebuilding gem5 binary: {reason}")
    jobs = max(os.cpu_count() or 1, 1)
    cmd = (
        "source /etc/profile.d/modules.sh && "
        "module load scons && "
        f"scons -C {REPO_ROOT / 'externals/gem5'} "
        f"EXTRAS={REPO_ROOT / 'src/gem5dev'} -j{jobs} build/RISCV/gem5.opt"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, cwd=REPO_ROOT)
    return gem5_bin


def run_gem5(case_dir: pathlib.Path, gem5_dir: pathlib.Path, host_elf: pathlib.Path,
             runtime_manifest: pathlib.Path, sim_image: pathlib.Path,
             extra_accel_count: int, force_rebuild_gem5: bool):
    gem5_bin = build_gem5(force_rebuild_gem5)
    report_path = gem5_dir / "gem5.report.json"
    accel_work_dir = gem5_dir / "accel-work"
    m5out_dir = gem5_dir / "m5out"
    cpu_trace_name = "cpu.exec.trace"
    config_script = REPO_ROOT / "configs/fcc_cgra.py"

    cmd = [
        str(gem5_bin),
        "-d",
        str(m5out_dir),
        f"--debug-file={cpu_trace_name}",
        "--debug-flags=Exec",
        str(config_script),
        "--kernel",
        str(host_elf),
        "--accel-sim-image",
        str(sim_image),
        "--accel-runtime-manifest",
        str(runtime_manifest),
        "--accel-work-dir",
        str(accel_work_dir),
        "--report",
        str(report_path),
    ]
    if extra_accel_count > 0:
        cmd.extend(["--extra-accel-count", str(extra_accel_count)])
    subprocess.run(cmd, check=False)
    if not report_path.exists():
        raise RuntimeError("gem5 did not produce gem5.report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("pass"):
        raise RuntimeError(f"gem5 run failed: {report}")

    latest_meta = None
    for meta in sorted(accel_work_dir.glob("invoke-*/reply/reply.meta")):
        latest_meta = meta
    if latest_meta is None:
        raise RuntimeError("gem5 run did not produce accelerator reply.meta")

    meta = {}
    for line in latest_meta.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta.setdefault(key, []).append(value)

    trace_path = pathlib.Path(meta["trace_path"][0])
    stat_path = pathlib.Path(meta["stat_path"][0])
    stat_json_path = pathlib.Path(meta["stat_json_path"][0]) if meta.get("stat_json_path") else None
    trace_dst = gem5_dir / f"{case_dir.name}.gem5.trace"
    stat_dst = gem5_dir / f"{case_dir.name}.gem5.stat"
    stat_json_dst = gem5_dir / f"{case_dir.name}.accel.stat.json"
    shutil.copyfile(trace_path, trace_dst)
    shutil.copyfile(stat_path, stat_dst)
    if stat_json_path and stat_json_path.exists():
        shutil.copyfile(stat_json_path, stat_json_dst)

    cpu_trace_src = m5out_dir / cpu_trace_name
    cpu_trace_dst = gem5_dir / f"{case_dir.name}.cpu.trace"
    if cpu_trace_src.exists():
        shutil.copyfile(cpu_trace_src, cpu_trace_dst)
    cpu_stats_src = m5out_dir / "stats.txt"
    cpu_stats_dst = gem5_dir / f"{case_dir.name}.cpu.stat.txt"
    cpu_stats_json = gem5_dir / f"{case_dir.name}.cpu.stat.json"
    if cpu_stats_src.exists():
        shutil.copyfile(cpu_stats_src, cpu_stats_dst)
        cpu_stats = parse_gem5_stats(cpu_stats_src)
        cpu_summary = build_cpu_perf_summary(cpu_stats, report)
        cpu_stats_json.write_text(
            json.dumps(cpu_summary, indent=2) + "\n", encoding="utf-8"
        )

    accel_stats = {}
    if stat_json_path and stat_json_path.exists():
        accel_stats = json.loads(stat_json_dst.read_text(encoding="utf-8"))
    else:
        accel_stats = parse_accel_stats(stat_dst)
        accel_stats = {
            "cycle_count": int(meta.get("cycle_count", ["0"])[0]),
            "trace_path": str(trace_dst),
            "stat_path": str(stat_dst),
            "memory_slots": [int(slot) for slot in meta.get("memory_slot", [])],
            "output_slots": [int(slot) for slot in meta.get("output_slot", [])],
            **accel_stats,
        }
        if meta.get("error_message"):
            accel_stats["error_message"] = meta["error_message"][0]
        stat_json_dst.write_text(
            json.dumps(accel_stats, indent=2) + "\n", encoding="utf-8"
        )
    accel_stats["cycle_count"] = int(meta.get("cycle_count", ["0"])[0])
    accel_stats["trace_path"] = str(trace_dst)
    accel_stats["stat_path"] = str(stat_dst)
    accel_stats["memory_slots"] = [int(slot) for slot in meta.get("memory_slot", [])]
    accel_stats["output_slots"] = [int(slot) for slot in meta.get("output_slot", [])]
    stat_json_dst.write_text(json.dumps(accel_stats, indent=2) + "\n",
                             encoding="utf-8")
    for memory_slot in meta.get("memory_slot", []):
        slot = int(memory_slot)
        src = latest_meta.parent / f"memory.slot{slot}.bin"
        if src.exists():
            dst = gem5_dir / f"{case_dir.name}.gem5.memory.slot{slot}.bin"
            shutil.copyfile(src, dst)

    report["cpu_trace"] = str(cpu_trace_dst) if cpu_trace_src.exists() else ""
    report["cpu_stat_txt"] = str(cpu_stats_dst) if cpu_stats_src.exists() else ""
    report["cpu_stat_json"] = str(cpu_stats_json) if cpu_stats_src.exists() else ""
    report["accel_trace"] = str(trace_dst)
    report["accel_stat"] = str(stat_dst)
    report["accel_stat_json"] = str(stat_json_dst)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path, trace_dst, stat_dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--extra-accel-count", type=int, default=0)
    parser.add_argument("--rebuild-gem5", action="store_true")
    args = parser.parse_args()

    case_dir = pathlib.Path(args.case_dir).resolve()
    (
        runtime_manifest,
        runtime,
        sim_image,
        _,
        sim_image_json,
        _,
        map_json,
        _,
        bundle,
    ) = read_case_metadata(case_dir)
    gem5_dir = case_dir / "gem5"
    gem5_dir.mkdir(parents=True, exist_ok=True)

    host_c = emit_host_source(case_dir, gem5_dir, runtime, sim_image_json, map_json, bundle)
    host_elf = build_host_elf(case_dir, gem5_dir, host_c)
    report_path, trace_path, stat_path = run_gem5(
        case_dir, gem5_dir, host_elf, runtime_manifest, sim_image,
        args.extra_accel_count, args.rebuild_gem5
    )

    print(f"host_c={host_c}")
    print(f"host_elf={host_elf}")
    print(f"report={report_path}")
    print(f"trace={trace_path}")
    print(f"stat={stat_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
