#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


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


def read_case_metadata(case_dir: pathlib.Path):
    runtime_manifest = load_single("*.runtime.json", case_dir)
    sim_bundle = case_dir / "sim.bundle.json"
    if not sim_bundle.exists():
        raise RuntimeError(f"missing {sim_bundle}")
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    software_stem = pathlib.Path(runtime["dfg_mlir"]).stem
    if software_stem.endswith(".dfg"):
        software_stem = software_stem[:-4]
    config_header = case_dir / f"{software_stem}.config.h"
    if not config_header.exists():
        raise RuntimeError(f"missing {config_header}")
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
    return runtime_manifest, runtime, sim_image, sim_bundle, bundle, config_header


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


def emit_host_source(case_dir: pathlib.Path, gem5_dir: pathlib.Path, runtime: dict,
                     bundle: dict, config_header: pathlib.Path) -> pathlib.Path:
    scalar_slot_by_arg, memory_slot_by_arg, output_slot_by_result = build_slot_maps(runtime)

    memory_defs = []
    memory_binds = []
    memory_checks = []

    for memory in bundle.get("memory_regions", []):
        arg_index = int(memory["memref_arg_index"])
        slot = memory_slot_by_arg[arg_index]
        elem_size = int(memory["elem_size_bytes"])
        bytes_list = pack_values(memory["values"], elem_size)
        array_name = f"mem_region_{slot}"
        memory_defs.append(
            f"static unsigned char {array_name}[] = {{{', '.join(str(x) for x in bytes_list)}}};"
        )
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
                f"      return 0x300 + {slot};",
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
            f"    return 0x200 + {slot};",
            f"  for (unsigned i = 0; i < {len(values)}; ++i) {{",
            "    unsigned short tag = 0;",
            f"    unsigned long long value = fcc_accel_read_output({slot}, i, &tag);",
            f"    if (value != {data_name}[i] || tag != {tag_name}[i])",
            f"      return 0x220 + {slot};",
            "  }",
        ])

    lines = [
        "#include <stddef.h>",
        "#include <stdint.h>",
        '#include "fcc_accel.h"',
        f'#include "{config_header.name}"',
        "",
    ]
    lines.extend(memory_defs)
    lines.extend(output_defs)
    lines.extend([
        "",
        "int main(void) {",
        "  fcc_accel_init();",
        "  if (fcc_accel_config_word_count > 0)",
        "    fcc_accel_load_config(fcc_accel_config_words,",
        "                          fcc_accel_config_word_count);",
    ])
    lines.extend(memory_binds)
    lines.extend(arg_lines)
    lines.extend([
        "  fcc_accel_launch();",
        "  if (fcc_accel_wait() != 0)",
        "    return 0x100 + (int)fcc_accel_error_code();",
    ])
    lines.extend(output_checks)
    lines.extend(memory_checks)
    lines.extend([
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


def build_gem5():
    jobs = max(os.cpu_count() or 1, 1)
    cmd = (
        "source /etc/profile.d/modules.sh && "
        "module load scons && "
        f"scons -C {REPO_ROOT / 'externals/gem5'} "
        f"EXTRAS={REPO_ROOT / 'src/gem5dev'} -j{jobs} build/RISCV/gem5.opt"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, cwd=REPO_ROOT)
    return REPO_ROOT / "build/RISCV/gem5.opt"


def run_gem5(case_dir: pathlib.Path, gem5_dir: pathlib.Path, host_elf: pathlib.Path,
             runtime_manifest: pathlib.Path, sim_image: pathlib.Path):
    gem5_bin = build_gem5()
    report_path = gem5_dir / "gem5.report.json"
    accel_work_dir = gem5_dir / "accel-work"
    m5out_dir = gem5_dir / "m5out"
    bridge_script = REPO_ROOT / "tools/gem5/fcc_runtime_bridge.py"
    fcc_binary = REPO_ROOT / "build/bin/fcc"
    config_script = REPO_ROOT / "configs/fcc_cgra.py"

    cmd = [
        str(gem5_bin),
        "-d",
        str(m5out_dir),
        str(config_script),
        "--kernel",
        str(host_elf),
        "--accel-sim-image",
        str(sim_image),
        "--accel-runtime-manifest",
        str(runtime_manifest),
        "--fcc-binary",
        str(fcc_binary),
        "--bridge-script",
        str(bridge_script),
        "--accel-work-dir",
        str(accel_work_dir),
        "--report",
        str(report_path),
    ]
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
    trace_dst = gem5_dir / f"{case_dir.name}.gem5.trace"
    stat_dst = gem5_dir / f"{case_dir.name}.gem5.stat"
    shutil.copyfile(trace_path, trace_dst)
    shutil.copyfile(stat_path, stat_dst)
    return report_path, trace_dst, stat_dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args()

    case_dir = pathlib.Path(args.case_dir).resolve()
    runtime_manifest, runtime, sim_image, _, bundle, config_header = read_case_metadata(case_dir)
    gem5_dir = case_dir / "gem5"
    gem5_dir.mkdir(parents=True, exist_ok=True)

    host_c = emit_host_source(case_dir, gem5_dir, runtime, bundle, config_header)
    host_elf = build_host_elf(case_dir, gem5_dir, host_c)
    report_path, trace_path, stat_path = run_gem5(
        case_dir, gem5_dir, host_elf, runtime_manifest, sim_image
    )

    print(f"host_c={host_c}")
    print(f"host_elf={host_elf}")
    print(f"report={report_path}")
    print(f"trace={trace_path}")
    print(f"stat={stat_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
