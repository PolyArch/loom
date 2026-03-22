#!/usr/bin/env python3

import argparse
import json
import pathlib

import m5
from m5.objects import *


class MemBus(SystemXBar):
    badaddr_responder = BadAddr()
    default = Self.badaddr_responder.pio


class LoomHiFive(HiFive):
    def _off_chip_devices(self):
        devices = list(super()._off_chip_devices())
        if hasattr(self, "dma_spike"):
            devices.append(self.dma_spike)
        return devices


def build_system(args):
    system = RiscvSystem()
    system.mem_mode = "timing"
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

    system.platform = LoomHiFive()
    system.platform.rtc = RiscvRTC(frequency=Frequency("100MHz"))
    system.platform.clint.int_pin = system.platform.rtc.int_pin
    system.platform.dma_spike = LoomDmaSpikeDevice(
        pio_addr=args.mmio_base,
        pio_size=0x1000,
    )
    system.platform.dma_spike.dma = system.membus.cpu_side_ports

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

    system.cpu = [RiscvTimingSimpleCPU(cpu_id=0, clk_domain=system.cpu_clk_domain)]
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
    parser.add_argument("--report", required=True)
    parser.add_argument("--mmio-base", type=lambda x: int(x, 0), default=0x10010000)
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
