# Mapped Visualization Conventions

## Overview

This document defines DOT/Graphviz visual conventions for mapped
(DFG-on-ADG) visualizations. It covers both overlay and side-by-side
display modes.

This is the mapped-level counterpart to
[spec-viz-adg.md](./spec-viz-adg.md) (hardware) and
[spec-viz-dfg.md](./spec-viz-dfg.md) (software).

## Overlay Mode

### Graph Direction

`rankdir` inherits from the ADG base graph (typically `LR` for
Structure mode).

### Mapped Node Coloring

Mapped HW nodes are recolored based on the SW operation dialect of the
mapped DFG operation:

| SW Dialect | Fill Color | Notes |
|------------|------------|-------|
| `arith.*` | lightblue | Integer and float arithmetic |
| `dataflow.*` | lightgreen | Carry, gate, invariant, stream |
| `handshake.cond_br`, `handshake.mux`, `handshake.join` | lightyellow | Control flow |
| `handshake.load`, `handshake.store` | lightsalmon | Memory access |
| `handshake.memory`, `handshake.extmemory` | skyblue | Memory modules |
| `handshake.constant` | gold | Constants |
| `math.*` | plum | Math operations |
| Multiple dialects (temporal PE) | gradient or striped | Uses primary dialect color |

### Unmapped Node Style

HW nodes without any mapped SW operation:

- Fill color: white
- Border style: dashed
- Border color: original node color (preserved from ADG style)

### Mapped Node Labels

```
<hw_name>
<- <sw_op_name>
```

For temporal PEs with multiple mapped operations:

```
<hw_name>
<- <sw_op_1> [slot 0, tag=0]
<- <sw_op_2> [slot 1, tag=1]
Reg: <used>/<available>
```

### Routing Path Coloring

Each mapped SW edge is rendered as a colored overlay on the ADG edges
it uses. Colors cycle through a 12-color palette to distinguish
different routes:

```
Route palette (12 colors):
  #e6194b, #3cb44b, #4363d8, #f58231,
  #911eb4, #42d4f4, #f032e6, #bfef45,
  #fabed4, #469990, #dcbeff, #9A6324
```

Assignment: SW edges are sorted by source node ID, then assigned
colors in order, wrapping at 12.

### Color Legend

A color legend is auto-generated and placed at the bottom of the graph.
It lists each mapped SW dialect and its corresponding fill color, plus
a sample of route colors with their SW edge labels.

### Temporal Annotations

For temporal PE nodes, the label includes:

- Slot assignments: which SW operation is in which FU slot.
- Tag values: the tag assigned to each operation for temporal routing.
- Register usage: `<used> / <available>` from MappingState.

## Side-by-Side Mode

### Left Panel (DFG)

Renders the DFG using conventions from [spec-viz-dfg.md](./spec-viz-dfg.md)
with the following additions:

- Each node carries a `data-hw-id` attribute (set via post-render DOM
  annotation, see [spec-viz-gui.md](./spec-viz-gui.md)).
- Mapped nodes have a subtle colored border matching the overlay
  dialect color, so the user can see mapping status without hovering.

### Right Panel (ADG)

Renders the ADG using conventions from [spec-viz-adg.md](./spec-viz-adg.md)
with the following additions:

- Each node carries a `data-sw-ids` attribute (JSON array).
- Mapped nodes are recolored using the same dialect-based scheme as
  overlay mode.
- Unmapped nodes use white fill + dashed border.
- Node labels include the mapped SW operation name(s) below the HW
  name.

### Cross-Highlighting

See [spec-viz-gui.md](./spec-viz-gui.md) for the cross-highlighting
protocol (data attributes, highlight sequence, CSS classes).

## Viewer Data Schema

The HTML viewer embeds three JSON data structures for mapped
visualizations. These are serialized from `MappingState` by the
`DOTExporterMapped` and `HTMLViewer` exporters.

### `mappingData`

```json
{
  "swToHw": {
    "<sw_node_id>": "<hw_node_id>",
    ...
  },
  "hwToSw": {
    "<hw_node_id>": ["<sw_node_id>", ...],
    ...
  },
  "routes": {
    "<sw_edge_id>": [
      {"src": "<hw_port_id>", "dst": "<hw_port_id>"},
      ...
    ],
    ...
  }
}
```

Field mapping from `MappingState`:

| JSON Field | MappingState Source |
|------------|--------------------|
| `swToHw` | `swNodeToHwNode` (SW Node ID -> HW Node ID) |
| `hwToSw` | `hwNodeToSwNodes` (reverse mapping) |
| `routes` | `swEdgeToHwPaths` (SW Edge ID -> ordered list of HW port pairs) |

Route entries are ordered lists of `{src, dst}` port-pair hops,
matching the `HwPath` representation in `MappingState.swEdgeToHwPaths`.

### `nodeMetadata`

```json
{
  "<node_id>": {
    "op": "arith.addi",
    "types": "(i32, i32) -> i32",
    "loc": "vecadd.cpp:15",
    "attrs": {"overflow": "none"},
    "hwName": "compute_tile_3_2",
    "hwType": "TemporalPE",
    "slots": [
      {"slot": 0, "op": "arith.addi", "tag": 0},
      {"slot": 1, "op": "arith.muli", "tag": 1}
    ],
    "registers": {"used": 1, "available": 4}
  },
  ...
}
```

Fields are populated by the exporter from:

| JSON Field | Source |
|------------|--------|
| `op` | MLIR operation name |
| `types` | MLIR result type summary |
| `loc` | MLIR `loc` attribute (file:line) |
| `attrs` | MLIR operation attributes (filtered to relevant ones) |
| `hwName` | ADG node symbol name (mapped nodes only) |
| `hwType` | ADG node type: PE, TemporalPE, Switch, etc. |
| `slots` | Temporal slot assignments from MappingState (temporal PEs only) |
| `registers` | Register usage from MappingState (temporal PEs only) |

### `dotSources`

```json
{
  "overlay": "digraph { ... }",
  "dfg": "digraph { ... }",
  "adg": "digraph { ... }"
}
```

Three DOT strings: one for overlay mode, one each for the side-by-side
DFG and ADG panels.

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-viz-adg.md](./spec-viz-adg.md)
- [spec-viz-dfg.md](./spec-viz-dfg.md)
- [spec-viz-gui.md](./spec-viz-gui.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
