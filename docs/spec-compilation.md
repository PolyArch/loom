# FCC Compilation Pipeline Specification

## Overview

This document captures the compilation-side contract that turns source code or
an SCF-level candidate region into a DFG suitable for mapping.

## Stage Family

FCC compilation is organized into these conceptual stages:

1. source to LLVM IR
2. LLVM dialect MLIR
3. CF-stage MLIR
4. SCF-stage MLIR
5. DFG-domain selection
6. DFG-stage lowering

## Canonical Artifact Flow

The intended artifact family is:

```text
<name>.ll
<name>.llvm.mlir
<name>.cf.mlir
<name>.scf.mlir
<name>.dfg.mlir
```

Exact emission policy may vary by mode, but these stage boundaries are
architecturally significant.

## DFG-Domain Selection

FCC does not hardcode "convert the whole function" as the only model.
Instead:

- a candidate region is selected from SCF-level IR
- the candidate must respect SCF hierarchy boundaries
- the chosen candidate is what proceeds into DFG lowering

The exploration logic itself is specified in [spec-dse.md](./spec-dse.md).

## DFG-Stage Responsibilities

The DFG stage must produce a graph suitable for mapping and later validation.

Important FCC-side requirements include:

- unsupported fanout or merge patterns should be normalized away by the
  lowering pipeline when required by the mapper model
- kernels with memory side effects must preserve enough structure for extmemory
  reconstruction and later validation
- unused outputs are handled by discard semantics rather than by introducing
  `handshake.sink` as the primary model

## Multi-Port Memory and Tagging

FCC retains Loom's broad direction that memory-related routing may involve tag
mechanisms and explicit memory interface structure, but the exact MVP contract
is currently centered on the FCC rebuild plan:

- one extmemory per array in the MVP
- tagged memory interaction remains part of the architecture, especially for
  temporal or multi-port expansion

This area should expand into dedicated memory specs as implementation grows.

## Relationship to Other Specs

- [spec-dse.md](./spec-dse.md)
- [spec-fcc.md](./spec-fcc.md)
- [spec-host-accel-interface.md](./spec-host-accel-interface.md)
