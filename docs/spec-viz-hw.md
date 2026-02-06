# Hardware Visualization Conventions

## Overview

This document is the **single source of truth** for DOT/Graphviz visual
conventions used in ADG-exported hardware diagrams. Both
[spec-adg.md](./spec-adg.md) and [spec-adg-api.md](./spec-adg-api.md)
reference this document.

## Global Graph Conventions

### Graph Direction (`rankdir`)

- `DOTMode::Structure`: `rankdir=LR`
- `DOTMode::Detailed`: `rankdir=TB`

The exporter must emit the selected mode explicitly in the DOT header.

### Naming and IDs

- Node IDs must be stable and unique within one graph.
- Preferred format: `<kind>_<ordinal>` (for example, `pe_3`, `switch_5`).
- If a symbol name exists, include it in the label (not in the ID).

### Label Escaping

- Use DOT-safe escaped strings for all labels.
- New lines use `\n`.
- Type strings and MLIR-like syntax must be escaped as plain text.

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

## Label Templates

### Node Label Template

`DOTMode::Structure` label template:

- `<op_kind>\n<symbol_or_instance_name>`

`DOTMode::Detailed` label template:

- `<op_kind>\n<symbol_or_instance_name>\n<type_summary>\n<key_params>`

`<key_params>` should include only mode-relevant hardware/runtime fields (for
example `num_instruction`, `num_register`, `ldCount/stCount`, `table_size`).

### Edge Label Template

- Structure mode: optional short port hint (`out0->in1`) when ambiguity exists.
- Detailed mode: include explicit source and destination port names.

## Port Visualization Rules

### Structure Mode

- Do not materialize per-port record fields.
- Show one node per operation and infer ports from edges.
- Use compact labels to prioritize topology readability.

### Detailed Mode

- Use record-style node labels with explicit input/output ports.
- Port field naming:
  - Inputs: `in0`, `in1`, ...
  - Outputs: `out0`, `out1`, ...
  - Memory ports: `ldaddr[k]`, `lddata[k]`, `staddr[k]`, `stdata[k]`,
    `lddone[k]`, `stdone[k]`
- Edge endpoints should connect to record fields (`node:out1 -> other:in0`).

## Cluster Conventions for `fabric.module` Hierarchy

- Each `fabric.module` emits one DOT `subgraph cluster_<module_name>`.
- Cluster label format: `module: <module_name>`.
- Nested module instances (from `fabric.instance`) may emit nested clusters in
  `DOTMode::Detailed`.
- Cluster visual style:
  - Border: solid for top-level module cluster
  - Border: dashed for instance-derived nested clusters
  - Background: transparent

## Unmapped Elements

In `DOTMode::Detailed`, elements without runtime configuration are shown with:

- Fill color: white
- Border style: dashed
- Original border color preserved

## DOT Mode Differences

| Aspect | `DOTMode::Structure` | `DOTMode::Detailed` |
|--------|-----------------------|---------------------|
| Purpose | Fast topology inspection | Debug-level structural inspection |
| `rankdir` | `LR` | `TB` |
| Node label density | Minimal (`kind + name`) | Expanded (`kind + name + type + params`) |
| Port rendering | Implicit | Explicit record ports |
| Cluster nesting | Optional shallow | Full module/instance hierarchy |
| Edge labeling | Minimal, ambiguity-only | Explicit per-port endpoints |
| Runtime-config visibility | Optional summary | Explicit key fields |

## Related Documents

- [spec-adg.md](./spec-adg.md): ADG overall design
- [spec-adg-api.md](./spec-adg-api.md): ADGBuilder API reference
