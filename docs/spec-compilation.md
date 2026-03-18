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

The software-only artifact family is:

```text
<name>.ll
<name>.llvm.mlir
<name>.cf.mlir
<name>.scf.mlir
<name>.dfg.mlir
```

Exact emission policy may vary by mode, but these stage boundaries are
architecturally significant.

When compilation continues into mapping against a selected ADG, later artifacts
that contain both software and hardware meaning must switch to mixed naming:

```text
<dfg>.<adg>.map.json
<dfg>.<adg>.map.txt
<dfg>.<adg>.config.json
<dfg>.<adg>.config.bin
<dfg>.<adg>.config.h
<dfg>.<adg>.viz.html
```

Hardware-only artifacts such as the ADG itself continue to use `<adg>.*`
names, for example:

```text
<adg>.fabric.mlir
<adg>.fabric.viz.json
```

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

### Join Fan-In Legalization

The DFG stage must also respect the hardware join capacity exported by the ADG.

Normative rules:

- the ADG exporter annotates the selected hardware with a maximum supported
  `handshake.join` fan-in
- FCC lowers this limit into the compilation pipeline as
  `fcc.adg_max_join_fanin`
- if the software DFG contains a `handshake.join` whose fan-in is less than or
  equal to that limit, it may remain unchanged
- if the software DFG contains a `handshake.join` whose fan-in exceeds that
  limit, FCC must rewrite it into a tree of smaller joins whose fan-in does
  not exceed the hardware limit
- the current mapper/config encoding supports hardware join fan-in up to `64`
- if the hardware limit is less than `2`, FCC must reject any software join
  whose fan-in is greater than `1`

This legalization happens before mapping. The mapper therefore only needs to
handle software joins whose fan-in is within the ADG-advertised hardware
capacity.

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
