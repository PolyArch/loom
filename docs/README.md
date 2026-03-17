# FCC Documentation Index

FCC is the successor design to Loom for the current Fabric Compiler effort.
This directory contains the implementation-anchor specifications for FCC.

These documents are aligned primarily with:

- `temp/plan-rebuild-0315-0.md`
- `temp/design-four-layer-dse.md`
- `temp/design-host-accel-interface.md`

When Loom and FCC differ, FCC documents in this directory take precedence for
future work under this repository.

## Top Level

| Spec | Description |
|------|-------------|
| [spec-fcc.md](./spec-fcc.md) | End-to-end FCC architecture and stage boundaries |
| [spec-fcc-vs-loom.md](./spec-fcc-vs-loom.md) | FCC changes relative to Loom and the intended compatibility boundary |
| [spec-cli.md](./spec-cli.md) | CLI modes, inputs, outputs, and artifact naming |
| [spec-compilation.md](./spec-compilation.md) | Frontend, SCF, DFG, and compilation-stage contracts |
| [spec-dse.md](./spec-dse.md) | Four-layer exploration and DFG-domain selection |
| [spec-host-accel-interface.md](./spec-host-accel-interface.md) | Host-side invocation model and accelerator interface |

## ADG and Construction

| Spec | Description |
|------|-------------|
| [spec-adg-builder.md](./spec-adg-builder.md) | ADG Builder responsibilities and fabric-generation contract |

## Fabric Dialect

| Spec | Description |
|------|-------------|
| [spec-fabric.md](./spec-fabric.md) | Fabric dialect overview and operation taxonomy |
| [spec-fabric-function_unit.md](./spec-fabric-function_unit.md) | `fabric.function_unit` and `fabric.static_mux` semantics |
| [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md) | Extmemory-facing routing, tagged multi-port memory, and memory region contract |
| [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md) | `fabric.spatial_pe` container, mux/demux model, config layout |
| [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md) | `fabric.spatial_sw`, decomposable routing, route-table semantics |
| [spec-fabric-temporal.md](./spec-fabric-temporal.md) | `fabric.temporal_pe` and `fabric.temporal_sw` summary |

## Mapper

| Spec | Description |
|------|-------------|
| [spec-mapper.md](./spec-mapper.md) | Mapper scope, responsibilities, and stage boundaries |
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

- This is the first FCC-native spec set. It intentionally focuses on the
  currently active architecture and the most important implementation anchors.
- Backend realization, standalone simulation, and gem5 integration are still
  primarily anchored by `temp/plan-rebuild-0315-0.md` and related design notes.
- Several currently observed mapper and visualization gaps are intentionally
  captured here as normative requirements so later implementation work can
  converge against explicit specs.
- Pure project-management content such as implementation batches remains better
  suited to planning notes than to normative spec documents.
- Future documents should extend this structure instead of adding new
  ad-hoc design notes under `temp/`.
