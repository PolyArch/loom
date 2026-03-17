# FCC Validation Specification

## Overview

This document defines the end-to-end validation contract for FCC, including
standalone simulation checks, gem5-backed checks, and regression expectations.

## Validation Scope

A conforming FCC validation flow should be able to demonstrate:

- successful graph lowering and mapping
- successful configuration generation and loading
- successful accelerator execution
- correct output or memory side effects
- consistent trace and performance data when enabled

## Required Validation Styles

### Standalone Validation

The standalone simulator must support:

- synthetic or generated inputs
- external-memory prefill
- CPU-reference comparison
- trace and statistics generation

### gem5 Validation

The gem5-backed flow must support:

- baremetal host execution
- MMIO-based configuration and launch
- DMA-based memory interaction
- result checking by host code and/or offline inspection

## Acceptance Matrix

The FCC target validation matrix includes at least:

1. full lowering from source to DFG artifacts
2. ADG generation
3. mapping report generation
4. standalone simulation with trace and stat output
5. correctness on a representative vecadd-style memory-writing kernel
6. gem5 end-to-end host-plus-accelerator execution
7. visualization rendering with mapping and, when available, trace playback
8. focused unit tests for temporal hardware, tagged memory topologies, and
   decomposable switches

## Compare Policies

Validation must support:

- output-port comparison
- memory-region comparison

Memory comparison is mandatory for kernels whose final observable results are
written to memory instead of being returned on output ports.

## Determinism Expectations

When deterministic settings are enabled:

- mapping results should be reproducible under the same seed and inputs
- standalone simulation results should be reproducible
- trace ordering should remain stable

## Deliverable-Oriented Checks

The intended FCC validation family covers:

- IR-stage artifact generation
- mapping JSON and text reports
- visualization HTML generation
- trace and stat generation
- host-runtime or gem5 integration success

## Relationship to Project Planning

This document captures the validation contract and acceptance intent.
Implementation batches and project scheduling remain planning concerns and do
not need to be part of the normative runtime behavior.

## Related Documents

- [spec-runtime-mmio.md](./spec-runtime-mmio.md)
- [spec-gem5.md](./spec-gem5.md)
- [spec-trace.md](./spec-trace.md)
- [spec-simulation.md](./spec-simulation.md)
