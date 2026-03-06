# Browser Viewer Specification

## Overview

This document specifies the self-contained HTML viewer used by all Loom
visualization levels (DFG, ADG, Mapped). The viewer renders DOT graphs
in-browser using Graphviz WASM and provides interactive exploration.

This is the GUI counterpart to the graph conventions defined in
[spec-viz-adg.md](./spec-viz-adg.md) and
[spec-viz-dfg.md](./spec-viz-dfg.md).

## Technology

### Graphviz WASM

Rendering uses [viz-js](https://viz-js.com/) (Graphviz compiled to
WebAssembly).

No server-side rendering or native Graphviz installation is required.

### Self-Contained HTML (Inline Bundling)

Each visualization output is a **truly self-contained** single `.html`
file. The viz.js WASM engine is inlined directly into the HTML as a
base64-encoded `<script>` block. This adds approximately 3 MB per file
but guarantees full offline operation with zero external dependencies.

The build system vendors `viz-standalone.js` at
`lib/loom/Viz/assets/viz-standalone.js` (fetched once at build time).
The `HTMLViewer.cpp` exporter reads this file and base64-encodes it
into the output HTML.

Each HTML file contains:

1. Embedded DOT source(s) as JavaScript string literals.
2. Embedded mapping metadata (Mapped level only) as JSON.
3. Embedded node metadata (attributes, source locations) as JSON.
4. Inlined viz.js WASM engine (base64-encoded `<script>` block).
5. Inline JavaScript for rendering and interaction.
6. Inline CSS for layout, toolbar, and detail panel.

The file opens in any modern browser with no network access required.

## HTML Structure

```
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Loom Visualization</title>
  <style>/* inline CSS */</style>
</head>
<body>

  <div id="toolbar">
    <!-- Zoom controls (all levels) -->
    <button id="btn-zoom-in">+</button>
    <button id="btn-zoom-out">-</button>
    <button id="btn-fit">Fit</button>

    <!-- Mode toggle (Mapped level only) -->
    <button id="btn-mode-toggle" style="display:none">
      Overlay | Side-by-Side
    </button>
  </div>

  <div id="graph-area">
    <div id="graph-left"></div>   <!-- primary graph panel -->
    <div id="graph-right"></div>  <!-- secondary panel (side-by-side) -->
  </div>

  <div id="detail-panel">
    <div id="detail-content"></div>
    <button id="detail-close">Close</button>
  </div>

  <script>
    // Embedded data (populated by DOT exporter)
    const vizLevel = "dfg";  // or "adg" or "mapped"
    const dotSources = {
      primary: "digraph { ... }",
      // Mapped level adds:
      overlay: "digraph { ... }",
      dfg: "digraph { ... }",
      adg: "digraph { ... }"
    };
    const mappingData = {
      swToHw: {},    // SW node ID -> HW node ID
      hwToSw: {},    // HW node ID -> [SW node IDs]
      routes: {}     // SW edge ID -> [HW edge IDs]
    };
    const nodeMetadata = {
      // "node_id": {
      //   op: "arith.addi",
      //   types: "(i32, i32) -> i32",
      //   loc: "vecadd.cpp:15",
      //   attrs: { overflow: "none" },
      //   hwName: "compute_tile_3_2",  // mapped only
      //   slots: [...]                 // temporal only
      // }
    };
  </script>
  <script>
    // viz.js WASM engine (inlined as base64 by HTMLViewer.cpp)
    // Decoded and evaluated at load time
  </script>
  <script>/* inline renderer + interaction JS */</script>

</body>
</html>
```

## Rendering Pipeline

1. On page load, initialize viz.js WASM engine.
2. Select the appropriate DOT source based on `vizLevel` and current
   display mode.
3. Call `viz.renderSVGElement(dotSource)` to produce an SVG DOM element.
4. Insert the SVG into `#graph-left` (and `#graph-right` for
   side-by-side mode).
5. Apply pan/zoom transform to the SVG container.
6. Attach event listeners for interaction.

Mode toggle (Mapped level) re-runs the rendering from the appropriate
DOT source without reloading the page.

## Interaction Model

### All Levels (DFG, ADG, Mapped)

| Feature | Trigger | Action |
|---------|---------|--------|
| Zoom in | Mouse wheel up / pinch out / toolbar button | Scale SVG around cursor position |
| Zoom out | Mouse wheel down / pinch in / toolbar button | Scale SVG around cursor position |
| Pan | Click + drag on background | Translate SVG viewport |
| Fit to view | Toolbar button / double-click background | Reset zoom and center graph |
| Hover highlight | Mouse enter node | Bold node border (3px), highlight connected edges (color + width increase) |
| Hover unhighlight | Mouse leave node | Restore original border and edge styles |
| Click detail | Click node | Populate and show detail panel |
| Close detail | Click close button / click background | Hide detail panel |

### Mapped Level Additional

| Feature | Trigger | Action |
|---------|---------|--------|
| Mode toggle | Toolbar button click | Switch between overlay and side-by-side rendering |
| Cross-highlight | Hover node (side-by-side) | Highlight corresponding mapped node(s) in opposite panel |
| Route trace | Click mapped edge (overlay) | Highlight full routing path through ADG |
| Scroll-to-view | Cross-highlight activation | Smooth-scroll opposite panel to bring highlighted node into view |

## Detail Panel

### DFG Node Detail

When a DFG node is clicked, the detail panel shows:

```
Operation: arith.addi
Source:    vecadd.cpp:15
Types:     (i32, i32) -> i32
Attributes:
  overflow: none
```

Fields are populated from `nodeMetadata[nodeId]`.

### ADG Node Detail

```
Module:    compute_tile_3_2
Type:      TemporalPE
Params:
  num_instruction: 16
  num_register: 4
FU Types:  int_arith, float_arith, compare, select
Ports:     4 in, 4 out
```

### Mapped Node Detail (Overlay Mode)

```
Hardware:  compute_tile_3_2 (TemporalPE)
Mapped SW: arith.addi (vecadd.cpp:15)
           arith.muli (vecadd.cpp:14)
Slot 0:    arith.addi, tag=0
Slot 1:    arith.muli, tag=1
Registers: 1 used / 4 available
```

### Mapped Node Detail (Side-by-Side, DFG Panel)

Same as DFG node detail, plus:

```
Mapped to: compute_tile_3_2
```

### Mapped Node Detail (Side-by-Side, ADG Panel)

Same as ADG node detail, plus:

```
Mapped SW: arith.addi (vecadd.cpp:15)
           arith.muli (vecadd.cpp:14)
```

## Source Traceability

MLIR operations carry `loc` attributes encoding source location:

```mlir
%0 = arith.addi %a, %b : i32 loc("vecadd.cpp":15:3)
```

The DOT exporter extracts location info and embeds it in
`nodeMetadata` as the `loc` field. The detail panel displays
this as `Source: vecadd.cpp:15`.

Fused locations (`loc(fused[...])`) are resolved to the first
concrete `FileLineCol` location in the fusion chain.

## Cross-Highlighting Protocol (Side-by-Side Mode)

### Data Attributes

Each SVG node carries custom data attributes for cross-linking:

- DFG panel nodes: `data-hw-id="<hw_node_id>"` (the mapped HW node).
- ADG panel nodes: `data-sw-ids='["<sw_id1>","<sw_id2>"]'` (JSON array
  of mapped SW node IDs).

### Post-Render DOM Annotation

Graphviz WASM renders DOT to SVG, but SVG nodes do not automatically
carry custom `data-*` attributes. The renderer performs a post-render
annotation step:

1. The DOT exporter embeds mapping IDs in DOT node `id` fields using
   a convention: `id="sw_<sw_node_id>"` for DFG nodes,
   `id="hw_<hw_node_id>"` for ADG nodes.
2. After `viz.renderSVGElement()` produces the SVG DOM, the renderer
   JavaScript walks all SVG `<g>` elements with `id` attributes.
3. For each DFG node (`id` starting with `sw_`), look up
   `mappingData.swToHw[swId]` and set `data-hw-id` on the element.
4. For each ADG node (`id` starting with `hw_`), look up
   `mappingData.hwToSw[hwId]` and set `data-sw-ids` (JSON array) on
   the element.
5. For mapped edges, set `data-sw-edge` from edge `id` fields
   following the convention `id="edge_<sw_edge_id>"`.

### Highlight Sequence

**Hover on DFG node:**

1. Add `highlight` CSS class to the hovered DFG node and its edges.
2. Read `data-hw-id` from the hovered node.
3. Find the corresponding ADG node in `#graph-right` by matching
   `data-hw-id` to the node's DOM id.
4. Add `cross-highlight` CSS class to the matched ADG node.
5. Smooth-scroll `#graph-right` to center the highlighted ADG node.

**Hover on ADG node:**

1. Add `highlight` CSS class to the hovered ADG node.
2. Read `data-sw-ids` from the hovered node (parse JSON array).
3. For each SW ID, find the corresponding DFG node in `#graph-left`.
4. Add `cross-highlight` CSS class to all matched DFG nodes and
   their edges.
5. Smooth-scroll `#graph-left` to center the first highlighted
   DFG node.

**Mouse leave:**

1. Remove `highlight` and `cross-highlight` classes from all nodes
   and edges in both panels.

### CSS Classes

```css
.highlight polygon,
.highlight ellipse,
.highlight path {
  stroke: #ff6600;
  stroke-width: 3px;
}

.cross-highlight polygon,
.cross-highlight ellipse,
.cross-highlight path {
  stroke: #0066ff;
  stroke-width: 3px;
  stroke-dasharray: 5,3;
}
```

## Overlay Mode Route Tracing

In overlay mode, clicking a mapped edge highlights the full routing
path through the ADG:

1. On edge click, look up the SW edge ID from the edge's `data-sw-edge`
   attribute.
2. Retrieve the HW path from `mappingData.routes[swEdgeId]`.
3. For each HW edge in the path, add `route-trace` CSS class.
4. On background click or next edge click, remove all `route-trace`
   classes.

```css
.route-trace path {
  stroke: #ff3300;
  stroke-width: 4px;
  stroke-dasharray: none;
}
```

## Layout

### Single-Panel Layout (DFG, ADG, Overlay)

```
+--[toolbar]--------------------+
|                               |
|         #graph-left           |
|       (full viewport)         |
|                               |
+-------------------------------+
|  #detail-panel (collapsible)  |
+-------------------------------+
```

### Side-by-Side Layout (Mapped)

```
+--[toolbar]--------------------+
|               |               |
|  #graph-left  | #graph-right  |
|    (DFG)      |    (ADG)      |
|               |               |
+-------------------------------+
|  #detail-panel (collapsible)  |
+-------------------------------+
```

The two panels have equal width (50% each) with a vertical divider.
Each panel has independent pan/zoom state.

## Toolbar

| Button | Availability | Action |
|--------|-------------|--------|
| Zoom In (+) | All levels | Increase zoom by 20% |
| Zoom Out (-) | All levels | Decrease zoom by 20% |
| Fit | All levels | Reset zoom to fit graph in viewport |
| Overlay / Side-by-Side | Mapped only | Toggle display mode |

The mode toggle button displays the name of the *other* mode
(clicking "Side-by-Side" switches to side-by-side and the button
text changes to "Overlay").

## Browser Compatibility

Target browsers: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+.

Required features: ES2020, WebAssembly, CSS Grid, SVG 1.1.

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-viz-adg.md](./spec-viz-adg.md)
- [spec-viz-dfg.md](./spec-viz-dfg.md)
- [spec-viz-mapped.md](./spec-viz-mapped.md)
