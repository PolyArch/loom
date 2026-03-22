#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import subprocess


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


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


def emit_host_source(out_dir: pathlib.Path, mmio_base: int) -> pathlib.Path:
    host_c = out_dir / "dma_spike_host.c"
    host_c.write_text(
        f"""#include <stdint.h>

#define MMIO_BASE 0x{mmio_base:x}ULL
#define REG32(off) (*(volatile uint32_t *)(uintptr_t)(MMIO_BASE + (off)))

enum {{
  REG_STATUS = 0x00,
  REG_CONTROL = 0x04,
  REG_SRC_LO = 0x08,
  REG_SRC_HI = 0x0c,
  REG_DST_LO = 0x10,
  REG_DST_HI = 0x14,
  REG_SIZE = 0x18,
  REG_CHECKSUM_LO = 0x1c,
  REG_CHECKSUM_HI = 0x20,
  REG_ERROR = 0x24,
  STATUS_BUSY = 1u << 0,
  STATUS_DONE = 1u << 1,
  STATUS_ERROR = 1u << 2,
  CTRL_START = 1u << 0,
  CTRL_RESET = 1u << 1
}};

static uint8_t src_buf[64] = {{
  1, 2, 3, 4, 5, 6, 7, 8,
  9, 10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 23, 24,
  25, 26, 27, 28, 29, 30, 31, 32,
  33, 34, 35, 36, 37, 38, 39, 40,
  41, 42, 43, 44, 45, 46, 47, 48,
  49, 50, 51, 52, 53, 54, 55, 56,
  57, 58, 59, 60, 61, 62, 63, 64
}};

static uint8_t dst_buf[64];

int main(void) {{
  uint64_t checksum = 0;
  for (unsigned i = 0; i < sizeof(src_buf); ++i)
    checksum += src_buf[i];

  REG32(REG_CONTROL) = CTRL_RESET;
  REG32(REG_SRC_LO) = (uint32_t)((uintptr_t)src_buf & 0xffffffffu);
  REG32(REG_SRC_HI) = (uint32_t)(((uint64_t)(uintptr_t)src_buf) >> 32);
  REG32(REG_DST_LO) = (uint32_t)((uintptr_t)dst_buf & 0xffffffffu);
  REG32(REG_DST_HI) = (uint32_t)(((uint64_t)(uintptr_t)dst_buf) >> 32);
  REG32(REG_SIZE) = sizeof(src_buf);
  REG32(REG_CONTROL) = CTRL_START;

  while ((REG32(REG_STATUS) & (STATUS_DONE | STATUS_ERROR)) == 0) {{ }}

  if (REG32(REG_STATUS) & STATUS_ERROR)
    return 0x100 + (int)REG32(REG_ERROR);

  for (unsigned i = 0; i < sizeof(src_buf); ++i) {{
    if (dst_buf[i] != src_buf[i])
      return 0x200 + (int)i;
  }}

  uint64_t observed_checksum =
      ((uint64_t)REG32(REG_CHECKSUM_HI) << 32) | REG32(REG_CHECKSUM_LO);
  if (observed_checksum != checksum)
    return 0x300;

  return 0;
}}
""",
        encoding="utf-8",
    )
    return host_c


def build_host_elf(out_dir: pathlib.Path, host_c: pathlib.Path) -> pathlib.Path:
    host_elf = out_dir / "dma_spike_host.elf"
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
        str(REPO_ROOT / "externals/gem5/include"),
        "-T",
        str(REPO_ROOT / "runtime/baremetal/loom_baremetal.ld"),
        str(REPO_ROOT / "runtime/baremetal/crt0.S"),
        str(REPO_ROOT / "externals/gem5/util/m5/src/abi/riscv/m5op.S"),
        str(host_c),
        "-o",
        str(host_elf),
    ]
    subprocess.run(cmd, check=True)
    return host_elf


def run_gem5(out_dir: pathlib.Path, host_elf: pathlib.Path, mmio_base: int):
    gem5_bin = build_gem5()
    report_path = out_dir / "dma_spike.report.json"
    m5out_dir = out_dir / "m5out"
    config_script = REPO_ROOT / "configs/loom_dma_spike.py"
    cmd = [
        str(gem5_bin),
        "-d",
        str(m5out_dir),
        str(config_script),
        "--kernel",
        str(host_elf),
        "--report",
        str(report_path),
        "--mmio-base",
        hex(mmio_base),
    ]
    subprocess.run(cmd, check=False)
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mmio-base", type=lambda x: int(x, 0), default=0x10010000)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    host_c = emit_host_source(out_dir, args.mmio_base)
    host_elf = build_host_elf(out_dir, host_c)
    report = run_gem5(out_dir, host_elf, args.mmio_base)
    print(f"host_c={host_c}")
    print(f"host_elf={host_elf}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
