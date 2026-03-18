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
9. container-config decoding checks for `spatial_pe` and `temporal_pe`
   slices, including mux or demux fields and selected internal FU config bits
10. temporal register encoding checks when `num_register > 0`, including
    `result -> reg` and `reg -> operand` cases
11. expected-fail temporal tests for incompatible `function_unit`
    configuration reuse
12. temporal operand-buffer hardware-parameter checks for
    `enable_share_operand_buffer`, `operand_buffer_size`, `num_instruction`,
    `num_register`, and `reg_fifo_depth`
13. tagged `spatial_sw` positive and negative tests, including the rule that
    tagged spatial switches cannot be decomposable
14. `temporal_sw` structural validation, including:
    all ports tagged and same type, positive `num_route_table`, and valid
    `connectivity_table` row dimensions
15. focused tagged-path tests that distinguish:
    source tags that remain distinct after width adaptation, and
    source tags that collapse to one observed hardware tag and must be rejected
    even when the conflicting shared resource only appears after expanding a
    memory-bridge suffix or prefix
16. focused tagged memory and extmemory ingress tests in which tagged
    `spatial_sw` performs tag-agnostic request merging while egress uses
    `temporal_sw` for tag-dependent response separation
17. application-level tagged memory regressions in which a frontend-generated
    DFG reaches one shared `fabric.extmemory` through tagged route-stage
    boundaries and still completes mapping, visualization, and standalone
    simulation
18. focused direct-boundary tests for both `fabric.memory` and
    `fabric.extmemory`, where the shared memory bridge terminates at already
    tagged route-stage ports on the compute side without requiring an explicit
    `fabric.add_tag` or `fabric.del_tag` at that boundary
19. focused tagged-memory egress tests that combine `fabric.map_tag`,
    width-adapting tagged `fabric.spatial_sw`, and one `fabric.temporal_sw`
    split, including negative cases where distinct source tags collapse to one
    observed tag before the temporal split
20. structural validation of Fabric definition and instantiation placement,
    including:
    `fabric.function_unit` visibility and instantiation hosts,
    inline-only `fabric.mux` and tag-boundary ops,
    module-level component inline placement restricted to `fabric.module`,
    lexical same-host duplicate-definition rejection across operation kinds,
    and rejection of `fabric.instance` in unsupported hosts or with PE-local
    SSA operands or results

## Compare Policies

Validation must support:

- output-port comparison
- memory-region comparison
- config-slice decoding for temporal register fields
- negative tests that assert mapper failure by exit code and diagnostic text

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
