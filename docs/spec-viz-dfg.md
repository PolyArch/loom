# DFG Visualization Conventions

## Overview

This document defines the layout and visual conventions for software dataflow
graph (DFG) visualization. The DFG is rendered using the Graphviz `dot` engine
via viz.js (WebAssembly) for automatic hierarchical top-to-bottom layout.

This is the DFG counterpart to
[spec-viz-adg.md](./spec-viz-adg.md) (hardware visualization).

## Layout

### Engine

The DFG is rendered by generating a DOT graph description from the DFG `Graph`
and passing it to the viz.js `dot` engine. This produces an SVG element with
automatic hierarchical layout.

### Graph Direction

`rankdir=TB` (top-to-bottom). Data flows from inputs at the top to outputs at
the bottom.

### Rank Constraints

- `handshake.func` block arguments (`ModuleInputNode` sentinels) use
  `rank=source` (topmost rank).
- `handshake.return` operands (`ModuleOutputNode` sentinels) use
  `rank=sink` (bottommost rank).

This ensures function arguments always appear at the top and return values
always appear at the bottom, regardless of graph topology.

## Node Styles

| Operation | Shape | Fill Color | Text Color |
|-----------|-------|-----------|-----------|
| `arith.*` | box | #add8e6 (light blue) | black |
| `handshake.constant` | ellipse | #ffd700 (gold) | black |
| `handshake.cond_br` | diamond | #ffffe0 (light yellow) | black |
| `handshake.mux` | invtriangle | #ffffe0 | black |
| `handshake.join` | triangle | #ffffe0 | black |
| `handshake.load` | box | #87ceeb (sky blue) | black |
| `handshake.store` | box | #ffa07a (light salmon) | black |
| `handshake.memory` | cylinder | #87ceeb | black |
| `handshake.extmemory` | hexagon | #ffd700 | black |
| `handshake.sink` | point | #999999 | - |
| `dataflow.carry` | octagon | #90ee90 (light green) | black |
| `dataflow.gate` | octagon | #98fb98 (pale green) | black |
| `dataflow.invariant` | octagon | #f5fffa (mint cream) | black |
| `dataflow.stream` | doubleoctagon | #90ee90 | black |
| `math.*` | box | #dda0dd (plum) | black |
| ModuleInputNode | invhouse | #ffb6c1 (light pink) | black |
| ModuleOutputNode | house | #f08080 (light coral) | black |
| Unknown/fallback | star | #ff0000 (red) | white |

## Edge Styles

| Edge Type | Line Style | Color | Width |
|-----------|-----------|-------|-------|
| Data dependency | solid | #333333 | 2.0 |
| Control token (`none` type) | dashed | #999999 | 1.0 |

Fan-out edges (one output port to multiple consumers) use the same style as
regular data edges. Each consumer edge is drawn separately.

## Node Labels

Template:

```
<operation_name>
<result_type_summary>
```

Operation-specific additions:

- `handshake.constant`: include constant value.
- `dataflow.stream`: include `step_op` annotation.
- `handshake.memory`/`extmemory`: include `ldCount`/`stCount`.
- `arith.cmpi`/`arith.cmpf`: include predicate name.

Source location (`loc` attribute) is stored in `swNodeMetadata` for display in
the detail panel but is not shown in the node label to keep labels compact.

### Metadata Source

The mapper's DFG `Graph` only stores `op_name`, `loc`, and port types. The
operation-specific attributes listed above (constant values, predicates,
`step_op`, `ldCount`/`stCount`) are **not** preserved in the `Graph`. The
C++ exporter (`VizHTMLExporter`) therefore receives the original MLIR
`handshake.func` module in addition to the mapper `Graph`, and extracts
these attributes directly from MLIR operations by correlating each `Graph`
node back to its source MLIR op via the `loc` attribute or a node-to-op
map built during DFG extraction.

## Edge Labels

No labels by default. When multiple edges exist between the same node pair,
port indices are shown: `out0->in1`.

## Stable DOM Identities

The DOT exporter must emit stable `id` attributes on all nodes and edges so
that the browser-side JS can locate SVG elements for cross-highlighting:

- Node IDs: `id="sw_<sw_node_id>"` (e.g., `id="sw_0"` for DFG node 0).
- Edge IDs: `id="swedge_<sw_edge_id>"` (e.g., `id="swedge_2"`).

After viz.js renders the DOT to SVG, the renderer JS uses these IDs to attach
hover/click handlers and to look up `mappingData` for cross-highlighting.

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-viz-adg.md](./spec-viz-adg.md)
- [spec-viz-gui.md](./spec-viz-gui.md)
