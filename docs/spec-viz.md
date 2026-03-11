# Loom Visualization Specification

## Overview

This document is the top-level specification for Loom visualization. It defines
the technology stack, output format, and visualization modes.

Detailed conventions for each component are in separate documents:

- [spec-viz-adg.md](./spec-viz-adg.md): ADG (hardware) grid layout and visual conventions
- [spec-viz-dfg.md](./spec-viz-dfg.md): DFG (software) hierarchical layout and visual conventions
- [spec-viz-mapped.md](./spec-viz-mapped.md): Mapped (DFG-on-ADG) overlay and side-by-side views
- [spec-viz-gui.md](./spec-viz-gui.md): Self-contained HTML viewer, interaction model, panel layout

## Technology Stack

The visualization system uses two complementary rendering engines:

| Engine | Role | Use Case |
|--------|------|----------|
| **viz.js** (Graphviz WASM) | Hierarchical graph layout and SVG rendering | DFG rendering via `dot` engine |
| **D3.js** | Grid-based custom layout, SVG rendering, interaction | ADG grid rendering, pan/zoom, cross-highlighting |

Both engines are embedded inline in the output HTML file. No external CDN,
server, or network access is required.

Approximate file overhead: ~3.3 MB per HTML file (viz.js ~3 MB, D3.js ~300 KB).

## Output Format

All visualization outputs are **self-contained single HTML files**. Each file
embeds:

1. ADG graph data as JSON (nodes, edges, extracted grid coordinates, area costs).
2. DFG graph as a DOT string (for viz.js rendering).
3. Mapping data as JSON (placement, routing, temporal assignments).
4. Node metadata as JSON (operation names, types, source locations).
5. viz.js WASM engine (base64-encoded inline script).
6. D3.js library (minified inline script).
7. Renderer JavaScript and all CSS (inline).

The file opens in any modern browser (Chrome 90+, Firefox 88+, Safari 14+,
Edge 90+) with zero network access required.

## Output File Naming

The visualization HTML is generated alongside other mapper output artifacts:

| File | Description |
|------|-------------|
| `<name>.config.bin` | Binary config image |
| `<name>_addr.h` | C header with addresses |
| `<name>.map.json` | Machine-readable mapping report |
| `<name>.map.txt` | Human-readable mapping report |
| `<name>.viz.html` | **Self-contained visualization** |

where `<name>` is derived from the `-o` base path provided to the `loom` CLI
(the same base path used for `.map.json` and other artifacts).

## Visualization Modes

The viewer supports three display modes, all within a single HTML file:

### Side-by-Side (Default)

Left panel shows the ADG (hardware) with D3-based grid layout. Right panel
shows the DFG (software) with Graphviz hierarchical layout. Mapping
relationships are shown via cross-highlighting on hover.

### Overlay

DFG operations are overlaid on the ADG grid. Mapped hardware nodes are
recolored by software operation dialect. Routing paths are drawn as colored
overlays on the ADG edges. Unmapped hardware nodes are dimmed.

### Single-Panel Maximize

Either panel can be maximized to fill the viewport for standalone inspection.

## Integration with Mapper Pipeline

The visualization file is produced **only when mapping succeeds**
(`status == "success"`). On failure, no `.viz.html` is generated (only
`.map.txt` and optional `.log` are written).

When mapping succeeds, the C++ exporter (`VizHTMLExporter`) serializes the
`MappingState`, ADG `Graph`, DFG `Graph`, and the original MLIR module into
the embedded JSON/DOT data structures. The original MLIR module is needed
because the mapper `Graph` does not preserve all operation attributes
(constant values, compare predicates, `step_op`, `ldCount`/`stCount`, etc.);
the exporter extracts these directly from MLIR operations.

The vendored assets (`viz-standalone.js`, `d3.min.js`) are converted to
compiled-in C string literals at build time, so the exporter embeds them
without runtime file I/O. This eliminates install-path dependencies.

## Related Documents

- [spec-viz-adg.md](./spec-viz-adg.md)
- [spec-viz-dfg.md](./spec-viz-dfg.md)
- [spec-viz-mapped.md](./spec-viz-mapped.md)
- [spec-viz-gui.md](./spec-viz-gui.md)
- [spec-mapper-output.md](./spec-mapper-output.md)
- [spec-adg.md](./spec-adg.md)
