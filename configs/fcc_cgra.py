#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys

import m5
from m5.objects import *


class MemBus(SystemXBar):
    badaddr_responder = BadAddr()
    default = Self.badaddr_responder.pio


class FccHiFive(HiFive):
    def _off_chip_devices(self):
        devices = list(super()._off_chip_devices())
        for child in self._children.values():
            if isinstance(child, FccCgraDevice) and child not in devices:
                devices.append(child)
        return devices


def build_accel(args, mmio_base, work_dir):
    accel = FccCgraDevice(
        pio_addr=mmio_base,
        pio_latency="100ns",
        pio_size=0x1000,
        accel_cycle_latency="1ns",
        max_batch_cycles=1024,
        max_inflight_mem_reqs=8,
        sim_image=args.accel_sim_image,
        runtime_manifest=args.accel_runtime_manifest,
        work_dir=work_dir,
    )
    return accel


def build_system(args):
    system = RiscvSystem()
    system.mem_mode = "atomic"
    system.mem_ranges = [AddrRange(start=0x80000000, size="512MiB")]

    system.voltage_domain = VoltageDomain()
    system.clk_domain = SrcClockDomain(
        clock="1GHz", voltage_domain=system.voltage_domain
    )
    system.cpu_voltage_domain = VoltageDomain()
    system.cpu_clk_domain = SrcClockDomain(
        clock="1GHz", voltage_domain=system.cpu_voltage_domain
    )

    system.workload = RiscvBareMetal()
    system.workload.bootloader = args.kernel

    system.iobus = IOXBar()
    system.membus = MemBus()
    system.system_port = system.membus.cpu_side_ports

    system.platform = FccHiFive()
    system.platform.rtc = RiscvRTC(frequency=Frequency("100MHz"))
    system.platform.clint.int_pin = system.platform.rtc.int_pin
    system.platform.accel = build_accel(args, args.mmio_base, args.accel_work_dir)
    system.platform.accel.dma = system.membus.cpu_side_ports
    for accel_idx in range(args.extra_accel_count):
        mmio_base = args.mmio_base + ((accel_idx + 1) * args.extra_accel_mmio_stride)
        work_dir = f"{args.accel_work_dir}.accel{accel_idx + 1}"
        extra = build_accel(args, mmio_base, work_dir)
        extra.dma = system.membus.cpu_side_ports
        setattr(system.platform, f"accel_extra_{accel_idx + 1}", extra)

    system.iobus.cpu_side_ports = system.platform.pci_host.up_request_port()
    system.iobus.mem_side_ports = system.platform.pci_host.up_response_port()
    system.platform.pci_bus.cpu_side_ports = (
        system.platform.pci_host.down_request_port()
    )
    system.platform.pci_bus.default = (
        system.platform.pci_host.down_response_port()
    )
    system.platform.pci_bus.config_error_port = (
        system.platform.pci_host.config_error.pio
    )

    system.bridge = Bridge(delay="50ns")
    system.bridge.mem_side_port = system.iobus.cpu_side_ports
    system.bridge.cpu_side_port = system.membus.mem_side_ports
    system.bridge.ranges = system.platform._off_chip_ranges()

    system.platform.attachOnChipIO(system.membus)
    system.platform.attachOffChipIO(system.iobus)
    system.platform.attachPlic()
    system.platform.setNumCores(1)

    system.cpu = [RiscvAtomicSimpleCPU(cpu_id=0, clk_domain=system.cpu_clk_domain)]
    cpu = system.cpu[0]
    cpu.icache_port = system.membus.cpu_side_ports
    cpu.dcache_port = system.membus.cpu_side_ports
    cpu.createInterruptController()
    cpu.createThreads()

    uncacheable = [*system.platform._on_chip_ranges(), *system.platform._off_chip_ranges()]
    cpu.mmu.pma_checker = PMAChecker(uncacheable=uncacheable)

    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports

    return system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--accel-sim-image", default="")
    parser.add_argument("--accel-runtime-manifest", required=True)
    parser.add_argument("--accel-work-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mmio-base", type=lambda x: int(x, 0), default=0x10010000)
    parser.add_argument("--extra-accel-count", type=int, default=0)
    parser.add_argument(
        "--extra-accel-mmio-stride",
        type=lambda x: int(x, 0),
        default=0x1000,
    )
    args = parser.parse_args()

    system = build_system(args)
    root = Root(full_system=True, system=system)
    m5.instantiate()
    exit_event = m5.simulate()

    cause = exit_event.getCause()
    code = exit_event.getCode()
    report = {
        "cause": cause,
        "code": code,
        "tick": int(m5.curTick()),
        "pass": cause == "m5_exit instruction encountered",
    }
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"Exiting @ tick {m5.curTick()} because {cause} ({code})")

    if cause == "m5_exit instruction encountered":
        return 0
    if cause == "m5_fail instruction encountered":
        return code or 1
    return 1


main()
