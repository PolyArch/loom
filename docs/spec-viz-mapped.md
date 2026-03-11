# Mapped Visualization Conventions

## Overview

This document defines visual conventions for mapped (DFG-on-ADG) visualization.
It covers both side-by-side and overlay display modes.

This is the mapped-level counterpart to
[spec-viz-adg.md](./spec-viz-adg.md) (hardware) and
[spec-viz-dfg.md](./spec-viz-dfg.md) (software).

## Temporal PE Aggregation

The mapper maps SW operations to FU sub-nodes (not the temporal PE container
node). However, the visualization renders each `fabric.temporal_pe` as a
**single visual node** on the ADG grid. The C++ exporter aggregates:

- All FU-level mappings (`swNodeToHwNode` entries targeting FU sub-nodes) are
  grouped under the parent temporal PE container node.
- `adgGraph.nodes` contains only the container node (virtual node); FU sub-
  nodes are not emitted as separate visual nodes.
- `mappingData.hwToSw` is keyed by the container node ID, listing all SW
  nodes mapped to any of its FUs.
- FU-level detail (which FU, slot, tag) is preserved in
  `mappingData.temporal` for display in the detail panel.
- FU sub-nodes lack a `sym_name` in the flattened ADG. The exporter
  synthesizes display names as `<parent_name>/fu_<index>`.

## Side-by-Side Mode

### Left Panel (ADG)

Renders the ADG using conventions from [spec-viz-adg.md](./spec-viz-adg.md)
with mapping annotations:

- Mapped nodes are recolored based on the SW operation dialect (see Dialect
  Color Table below).
- Unmapped nodes use white fill with dashed border (original border color
  preserved).
- Node labels append the mapped SW operation name(s) below the HW name.

### Right Panel (DFG)

Renders the DFG using conventions from [spec-viz-dfg.md](./spec-viz-dfg.md)
with mapping annotations:

- Each mapped DFG node has a subtle colored left-border indicator matching
  the ADG node's dialect color, showing placement status.
- Unmapped DFG nodes (mapping failure) are shown with red dashed border.

### Cross-Highlighting

Cross-highlighting uses `mappingData.swToHw` and `mappingData.hwToSw`.
For temporal PEs, the HW ID in these maps is the container node ID (after
aggregation). See [spec-viz-gui.md](./spec-viz-gui.md) for the interaction
protocol.

## Overlay Mode

### Base Graph

The ADG grid layout serves as the base. All ADG nodes remain at their grid
positions from [spec-viz-adg.md](./spec-viz-adg.md).

### Mapped Node Recoloring

Mapped hardware nodes are recolored based on the mapped software operation:

| SW Dialect | Fill Color | Notes |
|------------|-----------|-------|
| `arith.*` | #add8e6 (light blue) | Integer and float arithmetic |
| `dataflow.*` | #90ee90 (light green) | Carry, gate, invariant, stream |
| `handshake.cond_br/mux/join` | #ffffe0 (light yellow) | Control flow |
| `handshake.load` | #87ceeb (sky blue) | Load access |
| `handshake.store` | #ffa07a (light salmon) | Store access |
| `handshake.memory/extmemory` | #87ceeb (sky blue) | Memory modules |
| `handshake.constant` | #ffd700 (gold) | Constants |
| `math.*` | #dda0dd (plum) | Math operations |

When a temporal PE has multiple mapped operations from different dialects,
the node displays the primary dialect color (highest count) with a striped
secondary color indicator.

### Unmapped Node Style

Hardware nodes without any mapped software operation:

- Fill: white
- Border: dashed, original border color preserved
- Opacity: 0.4

### Overlay Labels

Mapped hardware nodes display:

```
<hw_name>
<- <sw_op_name>
```

For temporal PEs with multiple mapped operations:

```
<hw_name>
<- <sw_op_1> [slot 0, tag 0]
<- <sw_op_2> [slot 1, tag 1]
Reg: <used>/<available>
```

### Routing Path Visualization

Each mapped DFG edge is drawn as a colored overlay path on the ADG edges it
traverses. Colors cycle through a 12-color palette:

```
#e6194b, #3cb44b, #4363d8, #f58231,
#911eb4, #42d4f4, #f032e6, #bfef45,
#fabed4, #469990, #dcbeff, #9A6324
```

SW edges are sorted by source node ID, then assigned colors in order,
wrapping at 12. Route paths are semi-transparent (opacity 0.6) and drawn
above ADG edges but below overlay labels.

Route overlay stroke width follows the same bit-width-proportional model
as base ADG edges (see [spec-viz-adg.md](./spec-viz-adg.md)).

## Viewer Data Schema

The HTML viewer embeds JSON data structures for mapped visualization. These
are **exporter-derived views**, not direct serializations of the internal
mapper data model. The C++ exporter (`VizHTMLExporter`) transforms the
internal `Graph` (which uses global port IDs) and `MappingState` (which uses
flat port-pair sequences) into the viewer-friendly structures below.

### `adgGraph`

```json
{
  "nodes": [
    {
      "id": "hw_<hw_node_id>",
      "name": "<symbol_name or synthesized name>",
      "type": "<fabric.pe|fabric.switch|fabric.temporal_pe|...>",
      "class": "<functional|routing|memory|boundary>",
      "gridCol": null,
      "gridRow": null,
      "areaCost": 1,
      "areaW": 1,
      "areaH": 1,
      "bitWidth": 32,
      "params": { "body_ops": [...], "num_instruction": 8 }
    }
  ],
  "edges": [
    {
      "id": "hwedge_<hw_edge_id>",
      "srcNode": "hw_<src_node_id>",
      "dstNode": "hw_<dst_node_id>",
      "srcPort": "<port_index>",
      "dstPort": "<port_index>",
      "edgeType": "native|tagged|memref|control",
      "bitWidth": 32,
      "valueBitWidth": 32,
      "tagBitWidth": 0
    }
  ]
}
```

Notes:
- All IDs are prefixed with `hw_` to avoid collision with DFG IDs.
- `gridCol`/`gridRow` are null when the exporter cannot resolve coordinates.
- `bitWidth` on edges is the total bit width (value + tag for tagged types).
- `valueBitWidth` is the value-only bit width; `tagBitWidth` is the tag bit
  width (0 for non-tagged edges). The renderer uses `valueBitWidth` for the
  piecewise stroke width lookup, then adds 0.5px for tagged edges (see
  spec-viz-adg.md).
- `bitWidth` on nodes is the primary port width (for area heuristic).
- `areaW`/`areaH` are the grid cell dimensions (width x height) of the node.
  Default values per type are defined in spec-viz-adg.md (e.g., PE=1x1,
  temporal_pe=2x2, memory=1x2). The exporter computes these from the node
  type and complexity heuristics.
- Temporal PE container nodes represent the entire temporal PE; FU sub-nodes
  are not emitted (see Temporal PE Aggregation above).
- `name` for nodes without `sym_name` (e.g., flattened switches) uses the
  internal node ID formatted as `node_<id>`.

### `mappingData`

```json
{
  "swToHw": { "sw_<sw_node_id>": "hw_<hw_node_id>" },
  "hwToSw": { "hw_<hw_node_id>": ["sw_<sw_node_id>"] },
  "routes": {
    "sw_<sw_edge_id>": {
      "hwPath": [
        {
          "src": "<port_id>",
          "dst": "<port_id>",
          "hwEdgeId": "hwedge_<edge_id>"
        }
      ],
      "color": "<hex_color>"
    }
  },
  "temporal": {
    "sw_<sw_node_id>": {
      "container": "hw_<container_node_id>",
      "fuName": "<parent_name>/fu_<index>",
      "slot": 0,
      "tag": 0
    }
  },
  "registers": {
    "sw_<sw_edge_id>": { "registerIndex": 0 }
  }
}
```

Notes:
- All keys use `sw_` or `hw_` prefixes matching `adgGraph` node IDs.
- For temporal PE mappings, `swToHw` maps to the **container** node ID
  (not the FU sub-node), consistent with the aggregation model.
- `temporal` entries provide FU-level detail for the detail panel.

Fields populated from `MappingState`:

| JSON Field | MappingState Source |
|------------|--------------------|
| `swToHw` | `swNodeToHwNode` (FU IDs resolved to container IDs) |
| `hwToSw` | `hwNodeToSwNodes` (aggregated to container IDs) |
| `routes` | `swEdgeToHwPaths` with resolved `hwEdgeId` and palette color |
| `temporal` | Temporal PE slot/tag assignments with FU identity |
| `registers` | Internal temporal register assignments |

### `swNodeMetadata`

```json
{
  "sw_<node_id>": {
    "op": "<mlir_op_name>",
    "types": "<type_summary>",
    "loc": "<file>:<line>",
    "attrs": {},
    "hwTarget": "hw_<hw_node_id>"
  }
}
```

### `hwNodeMetadata`

```json
{
  "hw_<node_id>": {
    "name": "<symbol_name>",
    "type": "<fabric.pe|fabric.temporal_pe|...>",
    "body_ops": ["arith.addi", "arith.muli"],
    "ports": { "in": 4, "out": 4 },
    "mappedSw": ["sw_0", "sw_3"]
  }
}
```

Node metadata is split into two separate maps (`swNodeMetadata` and
`hwNodeMetadata`) because the DFG and ADG use independent ID spaces
(both are `IdIndex` starting from 0). The `sw_`/`hw_` prefixed keys
prevent collision.

### `dfgDot`

A single DOT string for the DFG graph, ready to pass to
`viz.renderSVGElement()`. Generated by the C++ exporter following
[spec-viz-dfg.md](./spec-viz-dfg.md) conventions.

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-viz-adg.md](./spec-viz-adg.md)
- [spec-viz-dfg.md](./spec-viz-dfg.md)
- [spec-viz-gui.md](./spec-viz-gui.md)
- [spec-mapper-output.md](./spec-mapper-output.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
