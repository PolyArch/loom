# DFG Visualization Conventions

## Overview

This document defines DOT/Graphviz visual conventions for software
dataflow graphs (DFGs) extracted from `handshake.func`.

This is the DFG counterpart to
[spec-viz-adg.md](./spec-viz-adg.md) (hardware visualization).

## Graph Direction

`rankdir=TB` (top-to-bottom): data flows from inputs at top to outputs
at bottom, following dependency order.

## Node Styles

| Operation | Shape | Fill Color | Text Color |
|-----------|-------|------------|------------|
| `arith.*` | box | lightblue | black |
| `handshake.constant` | ellipse | gold | black |
| `handshake.cond_br` | diamond | lightyellow | black |
| `handshake.mux` | invtriangle | lightyellow | black |
| `handshake.join` | triangle | lightyellow | black |
| `handshake.load` | box | skyblue | black |
| `handshake.store` | box | lightsalmon | black |
| `handshake.memory` | cylinder | skyblue | black |
| `handshake.extmemory` | hexagon | gold | black |
| `handshake.sink` | point | gray | - |
| `dataflow.carry` | octagon | lightgreen | black |
| `dataflow.gate` | octagon | palegreen | black |
| `dataflow.invariant` | octagon | mintcream | black |
| `dataflow.stream` | doubleoctagon | lightgreen | black |
| `math.*` | box | plum | black |
| `ModuleInputNode` | invhouse | lightpink | black |
| `ModuleOutputNode` | house | lightcoral | black |
| Unknown/fallback | star | red | white |

## Edge Styles

| Edge Type | Line Style | Color | Width |
|-----------|------------|-------|-------|
| Data dependency | solid | black | 2.0 |
| Control token (`none` type) | dashed | gray | 1.0 |

Fan-out edges (one output port to multiple consumers) use the same
style as regular data edges. Each consumer edge is drawn separately.

## Label Templates

### Node Labels

```
<operation_name>
<result_type_summary>
<source_location>
```

- `operation_name`: MLIR operation name (e.g., `arith.addi`).
- `result_type_summary`: abbreviated result types (e.g., `i32`, `f32`).
- `source_location`: from MLIR `loc` attribute, format `<file>:<line>`.

Operation-specific additions:

- `handshake.constant`: include constant value.
- `dataflow.stream`: include `step_op` and loop annotations.
- `handshake.memory`/`extmemory`: include `ldCount`/`stCount`.
- `arith.cmpi`/`arith.cmpf`: include predicate name.

### Edge Labels

- No label by default.
- When ambiguity exists (multiple edges between same node pair), show
  port indices: `out0->in1`.

## Port Rendering

### Compact Mode (default)

No explicit port fields. Ports inferred from edges.

### Detailed Mode (optional)

Record-style labels with explicit input/output ports:

- Inputs: `in0`, `in1`, ...
- Outputs: `out0`, `out1`, ...
- Memory ports: `ldaddr`, `lddata`, `staddr`, `stdata`, `lddone`, `stdone`

## Source Location Extraction

MLIR operations carry `loc` attributes:

```mlir
%0 = arith.addi %a, %b : i32 loc("vecadd.cpp":15:3)
```

The DOT exporter extracts location info and embeds it as:

- A line in the node label (compact: `vecadd.cpp:15`).
- A `data-loc` attribute on the DOT node (for viewer interaction).

Fused locations are resolved to the first concrete file location.

## Sentinel Nodes

Sentinel nodes (`ModuleInputNode`, `ModuleOutputNode`) use the same
styles as ADG sentinel nodes for visual consistency.

Labels include the block argument index or return operand index.

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-viz-adg.md](./spec-viz-adg.md)
- [spec-viz-gui.md](./spec-viz-gui.md)
