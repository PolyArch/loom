# FCC ADG Builder Specification

## Overview

The ADG Builder is responsible for producing Fabric MLIR that accurately
describes the available hardware architecture before mapping.

## Responsibilities

The builder must define:

- the top-level `fabric.module`
- instances of `spatial_pe`, `temporal_pe`, `spatial_sw`, `temporal_sw`, FIFO,
  and memory resources as needed
- module boundary ports
- inter-instance connectivity

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

## Flattening Boundary

The builder itself does not perform mapping-oriented flattening.
However, it must emit a hierarchy that the flattener can transform without
losing semantic ownership information.

This is particularly important for:

- PE exterior ports versus FU-local ports
- switch-local connectivity versus module-level wiring

## Relationship to Other Specs

- [spec-fabric.md](./spec-fabric.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-viz.md](./spec-viz.md)
