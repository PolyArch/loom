# Browser Viewer Specification

## Overview

This document specifies the self-contained HTML viewer for Loom visualization.
The viewer combines D3.js-based ADG rendering with Graphviz WASM-based DFG
rendering in an interactive dual-panel interface.

This is the GUI counterpart to the graph conventions defined in
[spec-viz-adg.md](./spec-viz-adg.md) and
[spec-viz-dfg.md](./spec-viz-dfg.md).

## Self-Contained HTML

Each `.viz.html` file is fully self-contained. Embedded assets:

| Asset | Format | Approximate Size |
|-------|--------|-----------------|
| viz-standalone.js | base64-encoded `<script>` | ~3 MB |
| d3.min.js (v7) | inline `<script>` | ~300 KB |
| Renderer JS | inline `<script>` | ~20 KB |
| CSS styles | inline `<style>` | ~5 KB |
| Graph data | inline JSON/DOT string literals | variable |

The build system vendors `viz-standalone.js` and `d3.min.js` and converts them
to compiled-in C string literals at build time (e.g., via `xxd -i` or CMake
file embedding). The C++ exporter writes them directly into the HTML without
runtime file I/O, eliminating install-path dependencies.

## HTML Structure

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Loom: <dfg_name> on <adg_name></title>
  <style>/* all CSS inline */</style>
</head>
<body>

  <div id="toolbar">
    <span id="title">...</span>
    <div id="mode-buttons">
      <button id="btn-sidebyside" class="active">Side-by-Side</button>
      <button id="btn-overlay">Overlay</button>
    </div>
    <button id="btn-fit">Fit</button>
    <span id="status-bar"></span>
  </div>

  <div id="graph-area">
    <div id="panel-adg">
      <div class="panel-header">Hardware (ADG)</div>
      <svg id="svg-adg"></svg>
    </div>
    <div id="panel-divider"></div>
    <div id="panel-dfg">
      <div class="panel-header">Software (DFG)</div>
      <svg id="svg-dfg"></svg>
    </div>
  </div>

  <div id="detail-panel">
    <div id="detail-content"></div>
    <button id="detail-close">Close</button>
  </div>

  <script>
    // Embedded data (populated by C++ exporter)
    const adgGraph = { nodes: [...], edges: [...] };
    const dfgDot = "digraph { ... }";
    const mappingData = { swToHw: {}, hwToSw: {}, routes: {}, temporal: {} };
    const swNodeMetadata = { "sw_0": { op: "...", ... } };
    const hwNodeMetadata = { "hw_24": { name: "...", ... } };
  </script>
  <script>/* viz-standalone.js (base64, decoded at load) */</script>
  <script>/* d3.min.js (inline) */</script>
  <script>/* renderer.js (inline) */</script>

</body>
</html>
```

## Panel Layout

### Side-by-Side (Default)

```
+--[toolbar]-------------------------------------------+
|                    |          |                       |
|   #panel-adg       | divider |   #panel-dfg          |
|   Hardware (ADG)   |         |   Software (DFG)      |
|   [D3 grid render] |         |   [Graphviz SVG]      |
|                    |          |                       |
+------------------------------------------------------+
|              #detail-panel (collapsible)              |
+------------------------------------------------------+
```

ADG panel occupies the left 55%, DFG panel the right 45%. The divider is
draggable for resizing. Each panel has independent pan/zoom state managed
by `d3.zoom()`.

### Overlay Mode

```
+--[toolbar]-------------------------------------------+
|                                                      |
|   #panel-adg (full width)                            |
|   Hardware grid with mapped SW ops overlaid          |
|                                                      |
+------------------------------------------------------+
|              #detail-panel (collapsible)              |
+------------------------------------------------------+
```

DFG panel is hidden. ADG panel expands to full width. Mapped DFG operations
are rendered as colored annotations on their assigned hardware nodes.

### Single-Panel Maximize

Either panel can be maximized to fill the viewport for standalone inspection.
Double-clicking a panel header toggles between split view and full-width
single-panel mode. The non-maximized panel is hidden (same as overlay hides
DFG). The toolbar shows a "Restore" button to return to split view. Keyboard
shortcut: `1` maximizes ADG, `2` maximizes DFG, `0` restores split.

## Rendering Pipeline

### ADG Panel (D3.js)

1. Parse `adgGraph` JSON: read nodes, edges, pre-computed coordinates, areas.
2. Run grid layout algorithm (see [spec-viz-adg.md](./spec-viz-adg.md)):
   a. Use `gridCol`/`gridRow` from JSON (set by C++ exporter).
   b. Compute pixel positions and node sizes from grid coordinates.
   c. Apply crossing-minimized fallback layout (Sugiyama method, see
      spec-viz-adg.md) for nodes with null coordinates. If fewer than
      50% of non-sentinel nodes have coordinates, re-layout the entire
      graph (ignoring partial coordinates) for better global quality.
3. Render SVG elements using D3 data joins:
   a. Draw subtle grid lines as background reference.
   b. Draw nodes as shaped SVG elements at grid positions.
   c. Draw edges as SVG paths between node borders.
   d. Add text labels.
4. Apply `d3.zoom()` behavior to `#svg-adg`.

### DFG Panel (viz.js)

1. Initialize viz.js WASM engine (async).
2. Call `viz.renderSVGElement(dfgDot)` to produce an SVG DOM element.
3. Insert SVG into `#panel-dfg`.
4. Apply `d3.zoom()` behavior to `#svg-dfg`.

## Interaction Model

### All Modes

| Feature | Trigger | Action |
|---------|---------|--------|
| Zoom | Mouse wheel / pinch | Scale SVG around cursor |
| Pan | Click + drag background | Translate SVG viewport |
| Fit | Toolbar button / double-click background | Reset zoom to fit graph |
| Hover highlight | Mouse enter node | Bold border, dim non-adjacent elements |
| Click detail | Click node | Show detail panel with attributes |
| Close detail | Click close / Escape / background | Hide detail panel |

### Side-by-Side Cross-Highlighting

| Feature | Trigger | Action |
|---------|---------|--------|
| SW->HW highlight | Hover DFG node | Highlight mapped ADG node(s) with blue glow |
| HW->SW highlight | Hover ADG node | Highlight mapped DFG node(s) with blue glow |
| Scroll-to-view | Cross-highlight activation | Smooth-scroll opposite panel to center target |
| Route preview | Hover mapped DFG edge | Highlight routing path on ADG panel |

### Overlay Mode

| Feature | Trigger | Action |
|---------|---------|--------|
| Route trace | Click mapped edge | Highlight full routing path on ADG |
| Clear trace | Click background | Remove route trace highlight |

## Cross-Highlighting Protocol

Cross-highlighting uses `mappingData.swToHw` and `mappingData.hwToSw`:

**Hover on DFG node:**
1. Read SW node ID from the hovered element.
2. Look up `mappingData.swToHw[swId]` to find the HW node.
3. Add `.cross-highlight` class to the matched ADG node in `#panel-adg`.
4. Smooth-scroll ADG panel to center the highlighted node.

**Hover on ADG node:**
1. Read HW node ID from the hovered element.
2. Look up `mappingData.hwToSw[hwId]` to find SW node(s).
3. Add `.cross-highlight` class to all matched DFG nodes in `#panel-dfg`.
4. Smooth-scroll DFG panel to center the first highlighted node.

**Mouse leave:** Remove `.highlight` and `.cross-highlight` from all elements
in both panels.

## Detail Panel Content

### DFG Node

```
Operation: arith.addi
Types:     (i32, i32) -> i32
Source:    vecadd.cpp:15
Mapped to: pe_0_0
```

### ADG Node

```
Name:      pe_0_0
Type:      fabric.pe
Grid:      (1, 1)
Area:      1 cell
Body:      arith.addi
Ports:     2 in, 1 out
Mapped SW: arith.addi
```

### Temporal PE Node (Overlay or Side-by-Side ADG)

```
Hardware:  tpe_r0_c0 (fabric.temporal_pe)
Grid:      (1, 1)
Area:      4 cells
FU 0 (int_arith):  arith.addi (vecadd.cpp:15)  [slot 0, tag 0]
FU 1 (compare):    arith.cmpi (vecadd.cpp:16)  [slot 1, tag 1]
Instructions: 2 / 8
Registers:    1 / 4
```

## CSS Highlight Classes

```css
.node-highlight  { stroke: #ff6600; stroke-width: 3px; }
.cross-highlight { stroke: #0066ff; stroke-width: 3px; stroke-dasharray: 5,3; }
.route-trace     { stroke: #ff3300; stroke-width: 4px; }
.node-dimmed     { opacity: 0.3; }
.edge-dimmed     { opacity: 0.15; }
```

## Browser Compatibility

Target: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+.
Required features: ES2020, WebAssembly, CSS Grid/Flexbox, SVG 1.1.

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-viz-adg.md](./spec-viz-adg.md)
- [spec-viz-dfg.md](./spec-viz-dfg.md)
- [spec-viz-mapped.md](./spec-viz-mapped.md)
