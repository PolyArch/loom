# ADG Visualization Conventions

## Overview

This document defines the grid-based layout algorithm and visual conventions
for ADG (hardware architecture) visualization. The ADG is rendered using D3.js
with a custom grid layout engine (not Graphviz `dot`).

This is the ADG counterpart to
[spec-viz-dfg.md](./spec-viz-dfg.md) (software visualization).

## Grid Layout Model

### Unit Cell

The layout is based on a uniform grid. Each grid cell has a fixed pixel size
(`cellSize`, default 120px). All node positions and sizes are quantized to
integer multiples of the unit cell size.

### Coordinate System

The grid uses a 2D coordinate system `(gridCol, gridRow)` with origin at the
top-left. Coordinates increase rightward (col) and downward (row).

The scheme uses 2x spacing so that switches occupy even grid positions and PEs
occupy odd positions (cell centers), matching the physical lattice topology:

```
  col0    col1    col2    col3    col4
   |       |       |       |       |
   SW------+------SW------+------SW   -- row0
   |       |       |       |       |
   +      [PE]     +      [PE]    +   -- row1
   |       |       |       |       |
   SW------+------SW------+------SW   -- row2
   |       |       |       |       |
   +      [PE]     +      [PE]    +   -- row3
   |       |       |       |       |
   SW------+------SW------+------SW   -- row4
```

### Node Placement

Each node occupies a rectangular region of `w x h` grid cells, where `w` and
`h` are positive integers derived from the node's area cost. Node center
position in pixels:

```
center_x = (gridCol + w/2) * cellSize
center_y = (gridRow + h/2) * cellSize
```

### Coordinate Sources (Priority Order)

Grid coordinates for each node are determined by the first available source:

1. **Explicit attributes** (`viz_row`, `viz_col`): If the ADG node carries
   `viz_row` and `viz_col` integer attributes, these are used directly as
   `(gridCol, gridRow) = (viz_col, viz_row)`. This is the preferred method.
   ADGBuilder automatically sets these for lattice mesh topologies:
   switches at (even, even), PEs at (odd, odd), west FIFOs at (even, odd),
   north FIFOs at (odd, even). The attributes are emitted in MLIR by
   ADGBuilderGen and copied through by ADGFlattener.

2. **Name-based extraction** (current primary method): The C++ exporter
   attempts regex extraction from node names. The following patterns reflect
   the actual naming conventions in ADGGen, ADGBuilder, ADGGenCube3D, and
   ADGGenTemporal:

   | Pattern | Grid Position (col, row) | Source |
   |---------|--------------------------|--------|
   | `sw_w<W>_<R>_<C>` | (C\*2, R\*2) per width band | ADGGen (most common) |
   | `l<L>_sw_<R>_<C>` | (C\*2, R\*2) per lattice band | ADGBuilder.latticeMesh |
   | `sw_<R>_<C>` | (C\*2, R\*2) | ADGBuilder basic mesh |
   | `<type>_r<R>_c<C>` | (C\*2+1, R\*2+1) | ADGGen PEs (e.g., `pe_addi_i32_r0_c1`) |
   | `load_pe_r<R>_c<C>` | (C\*2+1, R\*2+1) | ADGGen load PEs |
   | `store_pe_r<R>_c<C>` | (C\*2+1, R\*2+1) | ADGGen store PEs |
   | `pe_<R>_<C>` | (C\*2+1, R\*2+1) | ADGBuilder basic mesh, hand-written tests |
   | `tsw_<R>_<C>` | (C\*2, R\*2) in temporal band | ADGGenTemporal |
   | `tpe_r<R>_c<C>` | (C\*2+1, R\*2+1) in temporal band | ADGGenTemporal |
   | `sw_w<W>_<D>_<R>_<C>` | (C\*2, R\*2) per depth layer | ADGGenCube3D |

   Width-prefixed (`w<W>`) patterns group nodes by width plane. Multiple
   width planes or lattice IDs are placed in vertically separated bands.
   Temporal switches/PEs (`tsw_`, `tpe_`) are placed in a separate band
   below the spatial lattice.

3. **Topology inference**: FIFO nodes are positioned at the midpoint of their
   two connected switch nodes, computed from ADG edge topology.

Module input sentinels are placed in a row above the top grid boundary.
Module output sentinels are placed in a row below the bottom boundary.

Coordinate extraction is performed once in the C++ exporter
(`VizHTMLExporter`) and serialized as `gridCol`/`gridRow` in the `adgGraph`
JSON. The browser-side JS reads these values directly and does not re-extract.

### Fallback Layout (Crossing-Minimized Grid Placement)

When any non-sentinel node has null `gridCol`/`gridRow`, the browser-side JS
computes a layout for those nodes, optimized for **regularity** and **minimum
edge crossings**. Nodes with valid coordinates are pinned in place. If fewer
than 50% of non-sentinel nodes have coordinates, the fallback runs on the
entire graph (ignoring all partial coordinates) for better global quality. The algorithm is based on the **Sugiyama method** (layered
graph drawing), adapted for grid-aligned output.

#### Algorithm: Sugiyama-Based Grid Layout

The Sugiyama method is the standard algorithm for hierarchical graph drawing
with crossing minimization. It is the same method used by Graphviz's `dot`
engine internally. It proceeds in four phases:

**Phase 1 -- Cycle Removal**: ADG graphs may contain feedback cycles (e.g.,
bidirectional mesh edges). Detect cycles via DFS and temporarily reverse the
minimum set of back-edges to produce a DAG. After layout, restore original
edge directions.

**Phase 2 -- Layer Assignment**: Assign each node to a discrete integer layer
(vertical rank). Use the **longest-path** heuristic: nodes with no
predecessors go to layer 0; each other node goes to `max(predecessors) + 1`.
Nodes with pre-extracted grid coordinates are pinned to their layer. For
multi-cell nodes (area > 1x1), reserve adjacent layers.

**Phase 3 -- Crossing Minimization**: Within each layer, reorder nodes to
minimize edge crossings with adjacent layers. Use the **barycenter heuristic**:

```
for each sweep direction (top-down, then bottom-up):
  for each layer L:
    for each node v in L:
      barycenter(v) = average position of v's neighbors in adjacent layer
    sort nodes in L by their barycenter values
```

Run alternating top-down and bottom-up sweeps (typically 4-8 iterations)
until the crossing count stabilizes. This is an O(|V| * |E|) heuristic that
does not guarantee optimality but produces good results in practice. It is
the same heuristic used by Graphviz `dot`.

**Phase 4 -- Coordinate Assignment and Grid Snapping**: Assign each node a
grid cell `(gridCol, gridRow)` where `gridRow` equals its layer and
`gridCol` equals its position within the layer. Use the **Brandes-Kopf**
algorithm for balanced horizontal coordinate assignment (minimizes total
edge length while respecting ordering). Then snap all positions to the
nearest integer grid cell, resolving overlaps by shifting.

#### Implementation Notes

A suitable JS library is `d3-dag` (provides Sugiyama layout with
`d3.sugiyama()` including configurable layer assignment, crossing
minimization, and coordinate assignment). If `d3-dag` is insufficient or
too heavy, a self-contained implementation of the four phases above
(approximately 300-500 lines of JS) is tractable.

The goal is that even an arbitrary-topology ADG produces a visually regular,
grid-aligned layout with minimal edge crossings, not a tangled spider-web.

## Node Area Model

Each ADG node has an **area cost** measured in grid cells. Area determines the
visual size of the node on the grid. The area should reflect the relative
physical complexity of the hardware unit: a PE supporting floating-point
operations should appear visually larger than one supporting only integer add.

### Default Area Rules

| Node Type | Default Area | Visual Shape |
|-----------|-------------|-------------|
| `fabric.pe` | 1x1 | rounded square |
| `fabric.temporal_pe` | 2x2 | rounded square |
| `fabric.switch` | 1x1 | diamond (rotated square) |
| `fabric.temporal_sw` | 1x1 | diamond |
| `fabric.memory` | 1x2 | rectangle |
| `fabric.extmemory` | 1x2 | rectangle |
| `fabric.fifo` | 0.5x0.5 | small circle |
| `fabric.add_tag` | 0.5x0.5 | small triangle up |
| `fabric.map_tag` | 0.5x0.5 | small triangle up |
| `fabric.del_tag` | 0.5x0.5 | small triangle down |
| Module I/O | 0.5x1 | pentagon |

### Complexity-Based Area Scaling

For `fabric.pe` nodes, the exporter scales the default area based on the
body operation complexity:

- Multi-operation PE body: area = `max(1, body_op_count)` cells.
- Floating-point operations (`arith.addf`, `arith.mulf`, `math.*`):
  area multiplied by 1.5x compared to integer-only PEs.
- 64-bit operations: area multiplied by 1.5x compared to 32-bit.

These heuristics produce a visually meaningful size ordering without
requiring a full cost model. The actual scale factors are tunable
constants in the exporter.

### Future: `viz_area` Attribute Override

Area can be overridden by a `viz_area` attribute on the ADG node. When
present, the node is rendered as a square with side = `ceil(sqrt(viz_area))`
grid cells. This extensibility point enables future cost models to directly
drive the visual sizing with precise silicon area estimates. The `viz_area`
attribute does not exist in the current codebase and is a planned addition.

## Node Visual Styles

| Node Type | Fill Color | Border Color | Text Color |
|-----------|-----------|-------------|-----------|
| `fabric.pe` | #2d7d2d (dark green) | #1a5c1a | white |
| `fabric.temporal_pe` | #551a8b (purple) | #3d1266 | white |
| `fabric.switch` | #d3d3d3 (light gray) | #888888 | black |
| `fabric.temporal_sw` | #708090 (slate gray) | #4a5568 | white |
| `fabric.memory` | #87ceeb (sky blue) | #4a90a4 | black |
| `fabric.extmemory` | #ffd700 (gold) | #b8960f | black |
| `fabric.fifo` | #e8e8e8 | #aaaaaa | black |
| `fabric.add_tag` | #e0ffff (light cyan) | #66aaaa | black |
| `fabric.map_tag` | #da70d6 (orchid) | #9b4d96 | black |
| `fabric.del_tag` | #e0ffff | #66aaaa | black |
| Module input | #ffb6c1 (light pink) | #cc8899 | black |
| Module output | #f08080 (light coral) | #cc6666 | black |

## Edge Visual Styles

### Line Style and Color

| Connection Type | Line Style | Color |
|-----------------|-----------|-------|
| Native value | solid | #333333 |
| Tagged value | dashed | #7b2d8b |
| Memref | dotted | #2255cc |
| Control (none type) | dashed | #999999 |

### Bit-Width-Proportional Edge Width

Edge stroke width reflects the physical bit width of the connection. This
gives an immediate visual sense of data bandwidth across the fabric:

| Bit Width | Stroke Width | Example Types |
|-----------|-------------|---------------|
| 1 bit (`none`, control) | 1px | control tokens |
| 8-16 bits | 1.5px | `bits<8>`, `bits<16>` |
| 32 bits | 2px | `bits<32>`, `i32`, `f32` |
| 64 bits | 3px | `bits<64>`, `i64`, `f64` |
| Tagged (value + tag) | value width + 0.5px | `tagged<bits<32>, i3>` = 2.5px |

The table above is the canonical reference. The piecewise mapping is:

```
if totalBits <= 1:   strokeWidth = 1.0
elif totalBits <= 16: strokeWidth = 1.5
elif totalBits <= 32: strokeWidth = 2.0
elif totalBits <= 64: strokeWidth = 3.0
else:                 strokeWidth = 4.0
```

For tagged types, first compute the base stroke width from the **value**
bit width using the table above, then add 0.5px. This visually
distinguishes tagged edges from native edges of the same data width
(e.g., `tagged<bits<32>, i3>` = 2.0 + 0.5 = 2.5px).

### Edge Routing

Edges between adjacent grid nodes are drawn as straight lines. Edges between
non-adjacent nodes use orthogonal (Manhattan) polyline routing with right-angle
turns, following grid lines. This gives a clean, PCB-like visual appearance
that matches the physical mesh topology.

For dense meshes, parallel edges between the same node pair use lane offsets
(small perpendicular displacement) to avoid overlap. In overlay mode, base
ADG edges are drawn at low opacity (0.2) so that route overlays remain
visually distinct.

### Edge Crossing Bridge/Hop Convention

When two edges unavoidably cross (e.g., non-adjacent routes in a dense mesh),
the renderer applies the standard EDA/schematic **hop-over** convention: one
edge draws a small semicircular arc at the intersection point, visually
"hopping over" the other edge. This makes crossings immediately
distinguishable from connected junctions.

**Convention details:**

- For orthogonal (Manhattan) routing, intersections occur at grid-line
  crossings where a horizontal segment meets a vertical segment. The
  **horizontal edge hops over the vertical edge** (horizontal takes
  priority), matching the standard EDA convention. When both edges run in
  the same direction (rare, from lane offsets), the edge with the higher
  source node ID draws the hop arc.
- Arc radius: 6px (tunable constant). The arc is a 180-degree semicircle
  centered on the intersection point, with its bulge perpendicular to the
  hopping edge's direction.
- When more than two edges cross at the same point, each hopping edge
  offsets its arc by an additional radius increment to avoid overlapping
  arcs.
- Hop arcs inherit the stroke color and width of their parent edge.

## Node Labels

Each node displays a compact label:

```
<node_name>
<key_params>
```

Key parameters per node type:

- PE: body operation name (e.g., `arith.addi`)
- Temporal PE: `N instr, M reg`
- Switch: `Xin/Yout`
- Memory: `Lld/Sst, C elem`
- FIFO: depth (e.g., `d=2`)
- Tag ops: omitted (shape conveys type)

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-viz-dfg.md](./spec-viz-dfg.md)
- [spec-viz-gui.md](./spec-viz-gui.md)
- [spec-adg.md](./spec-adg.md)
