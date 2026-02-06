# Hardware Visualization Conventions

## Overview

This document is the **single source of truth** for DOT/Graphviz visual
conventions used in ADG-exported hardware diagrams. Both
[spec-adg.md](./spec-adg.md) and [spec-adg-api.md](./spec-adg-api.md)
reference this document.

## Node Styles

| Operation | Shape | Fill Color | Text Color | Notes |
|-----------|-------|------------|------------|-------|
| `fabric.pe` | Msquare | darkgreen | white | |
| `fabric.temporal_pe` | Msquare | purple4 | white | Larger size |
| `fabric.switch` | diamond | lightgray | black | |
| `fabric.temporal_sw` | diamond | slategray | white | |
| `fabric.memory` | cylinder | skyblue | black | |
| `fabric.extmemory` | hexagon | gold | black | |
| `fabric.add_tag` | trapezium | lightcyan | black | |
| `fabric.map_tag` | trapezium | orchid | black | |
| `fabric.del_tag` | invtrapezium | lightcyan | black | |
| `fabric.instance` | box | wheat | black | Uses referenced module color with dashed border |
| Module input ports | invhouse | lightpink | black | |
| Module output ports | house | lightcoral | black | |
| Unknown/fallback | star | red | white | Indicates error |

## Edge Styles

| Connection Type | Line Style | Color | Width |
|-----------------|------------|-------|-------|
| Native value | solid | black | 2.0 |
| Tagged value | dashed | purple | 2.0 |
| Memref | dotted | blue | 2.0 |
| Control (none type) | dashed | gray | 1.0 |

## Unmapped Elements

In `DOTMode::Detailed`, elements without runtime configuration are shown with:

- Fill color: white
- Border style: dashed
- Original border color preserved

## Related Documents

- [spec-adg.md](./spec-adg.md): ADG overall design
- [spec-adg-api.md](./spec-adg-api.md): ADGBuilder API reference
