# LOOM Host-Accelerator Interface Specification

## Overview

LOOM targets a split execution model:

- the host program runs on the CPU side
- the selected kernel region is executed by the mapped accelerator

The host-accelerator contract must work in both standalone simulation and gem5.

## MVP Strategy

LOOM may generate a separate host-side file that replaces the original direct
kernel call with an accelerator invocation sequence.

The host-side flow is:

1. prepare input buffers and scalar arguments
2. load accelerator configuration
3. bind external memory regions
4. launch execution
5. wait for completion
6. read back outputs or memory side effects

## SimSession-Level Contract

The runtime-facing simulation contract is centered on a session abstraction with
the following responsibilities:

- build from mapped DFG and ADG state
- load configuration words
- inject scalar or stream inputs
- bind a backing memory region for external memories
- invoke execution
- collect outputs, traces, and statistics

The exact C++ surface may evolve, but these responsibilities are stable.

## Standalone vs gem5

The same simulation core should serve both:

- standalone LOOM simulation
- gem5 device-backed execution

The difference is only where inputs, memory, and completion handling come from.
The mapped accelerator semantics must stay identical across the two environments.

## Memory-Centric Kernels

LOOM must treat memory side effects as first-class outputs. A kernel that
produces its final result by storing into external memory is not fully validated
by checking return tokens alone.

Therefore the host contract must support:

- output-port comparison
- post-execution memory comparison

## Relationship to Other Specs

- Top-level pipeline: [spec-loom.md](./spec-loom.md)
- Exploration and validation loop: [spec-dse.md](./spec-dse.md)
- Mapper outputs: [spec-mapper-output.md](./spec-mapper-output.md)
