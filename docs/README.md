# LOOM Documentation Index

LOOM is the successor design to the legacy design for the current Fabric Compiler effort.
This directory contains the implementation-anchor specifications for LOOM.

When the legacy design and LOOM differ, LOOM documents in this directory take
precedence for future work under this repository.

## Top Level

| Spec | Description |
|------|-------------|
| [spec-loom.md](./spec-loom.md) | End-to-end LOOM architecture and stage boundaries |
| [spec-loom-vs-legacy.md](./spec-loom-vs-legacy.md) | LOOM changes relative to the legacy design and the intended compatibility boundary |
| [spec-cli.md](./spec-cli.md) | CLI modes, inputs, outputs, and artifact naming |
| [spec-compilation.md](./spec-compilation.md) | Frontend, SCF, DFG, and compilation-stage contracts |
| [spec-dse.md](./spec-dse.md) | Four-layer exploration and DFG-domain selection |
| [spec-host-accel-interface.md](./spec-host-accel-interface.md) | Host-side invocation model and accelerator interface |

## ADG and Construction

| Spec | Description |
|------|-------------|
| [spec-adg-builder.md](./spec-adg-builder.md) | ADG Builder responsibilities and fabric-generation contract |
| [spec-dataflow.md](./spec-dataflow.md) | LOOM dataflow dialect semantics and mapping expectations |
| [spec-dataflow-compilation.md](./spec-dataflow-compilation.md) | SCF-to-handshake+dataflow lowering rules, raw-vs-gated control, and nested-SCF flattening |
| [spec-dataflow-memory.md](./spec-dataflow-memory.md) | Memory-op lowering, root-memref grouping, and serial-recursive ctrl-done ordering |

## Fabric Dialect

| Spec | Description |
|------|-------------|
| [spec-fabric.md](./spec-fabric.md) | Fabric dialect overview and operation taxonomy |
| [spec-fabric-config_mem.md](./spec-fabric-config_mem.md) | Unified config bitstream model, slice ordering, and per-resource packing |
| [spec-fabric-function_unit.md](./spec-fabric-function_unit.md) | `fabric.function_unit` and `fabric.mux` semantics |
| [spec-fabric-fifo.md](./spec-fabric-fifo.md) | `fabric.fifo` structure, legality, and runtime bypass config |
| [spec-fabric-tag.md](./spec-fabric-tag.md) | `fabric.add_tag`, `fabric.map_tag`, and `fabric.del_tag` |
| [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md) | Extmemory-facing routing, tagged multi-port memory, and memory region contract |
| [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md) | `fabric.spatial_pe` container, mux/demux model, config layout |
| [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md) | `fabric.spatial_sw`, decomposable routing, route-table semantics |
| [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md) | `fabric.temporal_pe` structure, slot encoding, and register model |
| [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md) | `fabric.temporal_sw` tag-matching route-table semantics |

## Mapper

| Spec | Description |
|------|-------------|
| [spec-mapper.md](./spec-mapper.md) | Mapper scope, responsibilities, and stage boundaries |
| [spec-mapper-config.md](./spec-mapper-config.md) | Mapper base-config merge rules, promoted CLI surface, and configuration ownership |
| [spec-mapper-model.md](./spec-mapper-model.md) | Graph model, flattening contract, hard constraints, route semantics |
| [spec-mapper-output.md](./spec-mapper-output.md) | Mapping reports, config fragments, and visualization payloads |

## Visualization

| Spec | Description |
|------|-------------|
| [spec-viz.md](./spec-viz.md) | Visualization architecture and mapping-on rendering requirements |

## Simulation

| Spec | Description |
|------|-------------|
| [spec-simulation.md](./spec-simulation.md) | Standalone simulation model, trace, and validation contract |
| [spec-runtime-mmio.md](./spec-runtime-mmio.md) | SimSession lifecycle, host driver API, and MMIO control model |
| [spec-gem5.md](./spec-gem5.md) | gem5 integration, baremetal host execution, and device boundary |
| [spec-trace.md](./spec-trace.md) | Trace and performance schema for standalone and gem5-backed runs |
| [spec-validation.md](./spec-validation.md) | Acceptance matrix and end-to-end validation contract |

## Notes

- This is the first LOOM-native spec set. It intentionally focuses on the
  currently active architecture and the most important implementation anchors.
- Backend realization, standalone simulation, and gem5 integration specs
  are included above and continue to evolve.
- Several currently observed mapper and visualization gaps are intentionally
  captured here as normative requirements so later implementation work can
  converge against explicit specs.
- Pure project-management content such as implementation batches remains better
  suited to planning notes than to normative spec documents.
