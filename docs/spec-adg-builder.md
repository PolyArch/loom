# FCC ADG Builder Specification

## Overview

The ADG Builder is responsible for producing Fabric MLIR that accurately
describes the available hardware architecture before mapping.

For topologies whose geometry is already known at construction time, the
builder may also emit a visualization sidecar and bind it from the generated
`fabric.module`.

## Responsibilities

The builder must define:

- the top-level `fabric.module`
- instances of `spatial_pe`, `temporal_pe`, `spatial_sw`, `temporal_sw`, FIFO,
  and memory resources as needed
- module boundary ports
- inter-instance connectivity

When helper APIs already know intended component positions, they should prefer
emitting explicit layout metadata instead of forcing the visualization layer to
infer one later.

## FCC-Specific Construction Rules

The ADG Builder must respect FCC's newer hardware structure:

- PEs are containers of `function_unit`
- switch descriptions may use decomposable spatial switches
- memory and stream resources should be emitted in a form that later allows
  flattening and mapping without losing configuration meaning

## Builder Output Contract

The ADG Builder's Fabric MLIR must carry enough information for later stages to:

- flatten placeable resources
- recover PE containment
- recover switch port identities
- generate visualization that still understands the structural hierarchy

Builder-emitted ADGs and hand-authored ADGs consumed by the mapper are subject
to the same `verifyFabricModule` contract.

At minimum, a conforming `fabric.module` must satisfy:

- no dangling hardware outputs inside the graph
- no dangling module input ports
- no instance or inline-graph input-count mismatch against the declared module
  type
- no non-tagged to tagged hardware connection, nor tagged to non-tagged
  connection
- module-result connections whose tag-kind is compatible with the declared
  `fabric.module` function type

## Flattening Boundary

The builder itself does not perform mapping-oriented flattening.
However, it must emit a hierarchy that the flattener can transform without
losing semantic ownership information.

This is particularly important for:

- PE exterior ports versus FU-local ports
- switch-local connectivity versus module-level wiring

## Visualization Sidecar Contract

When a builder emits explicit layout metadata, it should:

- write a sidecar JSON file adjacent to the generated `fabric.mlir`
- attach `attributes {viz_file = "relative-or-local-name.json"}` to the
  generated `fabric.module`

The current sidecar schema is:

- top-level `version`
- top-level `components`
- each component entry contains:
  - `name`
  - `kind`
  - `center_x`
  - `center_y`
  - optional `grid_row`
  - optional `grid_col`

Component identity is currently name-based. Builders that intend to bind
explicit visualization metadata should therefore emit stable component names.

This mechanism is intended for topology-aware layouts such as chess meshes,
lattice meshes, torus-like fabrics, or other regular structures where the
builder already knows the desired geometry.

## Relationship to Other Specs

- [spec-fabric.md](./spec-fabric.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-viz.md](./spec-viz.md)
