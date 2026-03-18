# FCC Fabric Dialect Specification

## Overview

Fabric MLIR is FCC's hardware architecture IR. It describes the modules,
containers, routing resources, and memory endpoints that form the ADG.

FCC retains the overall Fabric role from Loom, but changes the internal PE and
switch model substantially.

## Key FCC Differences from Loom

- `fabric.spatial_pe` and `fabric.temporal_pe` contain explicit
  `fabric.function_unit` instances.
- `fabric.function_unit` may contain a configurable internal DAG via
  `fabric.mux`.
- `fabric.spatial_sw` may be decomposable, so routing may operate at sub-lane
  granularity.
- `fabric.spatial_sw` may also carry tagged payloads, but tagged spatial
  routing remains tag-agnostic and may not be decomposable.
- DFG-domain exploration replaces the older pragma-centric control model.

## Operation Families

| Family | Operations |
|--------|------------|
| Top level | `fabric.module`, `fabric.instance`, `fabric.yield` |
| Compute containers | `fabric.spatial_pe`, `fabric.temporal_pe` |
| Compute bodies | `fabric.function_unit`, `fabric.mux` |
| Routing | `fabric.spatial_sw`, `fabric.temporal_sw`, `fabric.fifo` |
| Tag boundary | `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag` |
| Memory | `fabric.memory`, `fabric.extmemory`, and related memory resources |

## Type Model

At module boundaries and inter-module connections, FCC uses structural bit
types, not native arithmetic types.

Typical rules:

- module and switch ports use `!fabric.bits<N>`
- `fabric.spatial_sw` may also use `!fabric.tagged<!fabric.bits<N>, iK>` when
  routing itself does not depend on tag
- tagged temporal routing uses `!fabric.tagged<!fabric.bits<N>, iK>`
- native types such as `i32`, `f32`, `index`, and `none` live inside
  `function_unit` boundaries

For tagged Fabric ports, two concepts must stay separate:

- hardware tag parameters:
  whether a port is tagged at all, and if so, its `tagWidth`
- runtime tag values:
  the concrete tag numbers carried by one mapped software flow or written into
  one runtime-config structure

Hardware tag parameters come directly from the ADG's port types. They are not
inferred by the mapper. The mapper only validates compatibility and computes
runtime tag values where configuration requires them.

FCC hardware connections allow width mismatch as long as tag-kind matches:

- non-tagged may connect to non-tagged
- tagged may connect to tagged
- non-tagged may not connect directly to tagged

When widths differ on one hardware connection:

- value bits are LSB-aligned to value bits
- tag bits are LSB-aligned to tag bits
- wide to narrow truncates high bits
- narrow to wide zero-extends high bits

This means runtime tag values may change implicitly along one tagged hardware
path even when no operation changes tagged shape. A runtime tag value observed
on `!fabric.tagged<!fabric.bits<N>, i4>` and then carried into
`!fabric.tagged<!fabric.bits<N>, i3>` is observed as the low 3 bits only.

Only three Fabric operations may intentionally change tagged shape at a port
boundary:

- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`

All other Fabric operations are structural carriers or routers. They may route
tagged values, but they do not create, remove, or resize the tag field.

For `fabric.map_tag`, the value payload type must remain unchanged. `map_tag`
may rewrite runtime tag values and may also change the tag width between its
input and result tagged types.

## Definition and Instantiation Rules

FCC distinguishes named definitions from inline instantiations by structural
syntax, not by whether an op happens to have SSA results on the left-hand side.

For module-level Fabric components:

- a named definition has no operand list and contributes reusable structure to
  its host scope
- an inline instantiation carries an operand list and contributes one graph
  node directly to the enclosing `fabric.module`

Current parser and verifier materialize this distinction through the
operation-local `inline_instantiation` marker. This is an implementation detail,
but the normative semantic distinction is:

- definitions establish symbols
- inline instantiations do not establish instantiable definition targets

### `fabric.function_unit`

- a `fabric.function_unit` definition may appear directly inside:
  - the top-level `builtin.module`
  - `fabric.module`
  - `fabric.spatial_pe`
  - `fabric.temporal_pe`
- `fabric.function_unit` may only be instantiated inside:
  - `fabric.spatial_pe`
  - `fabric.temporal_pe`
- inside one PE, a function unit may be provided either:
  - by a direct local `fabric.function_unit` body
  - by `fabric.instance` targeting a visible `fabric.function_unit`

An instance inside one PE must target `fabric.function_unit`, and it must not
carry SSA operands or SSA results. PE-local function-unit binding is structural,
not a routed Fabric edge.

### `fabric.mux`

- `fabric.mux` may appear only directly inside `fabric.function_unit`
- it is inline-only and is not instantiable through `fabric.instance`

### Module-Level Components

The following operations are module-level Fabric components:

- `fabric.spatial_pe`
- `fabric.temporal_pe`
- `fabric.spatial_sw`
- `fabric.temporal_sw`
- `fabric.memory`
- `fabric.extmemory`
- `fabric.fifo`

For these operations:

- named definitions may appear directly inside:
  - the top-level `builtin.module`
  - `fabric.module`
- inline instantiations may appear directly only inside `fabric.module`
- `fabric.instance` targeting one of these definitions may appear directly only
  inside `fabric.module`

### Inline-Only Boundary Operations

The following operations are inline-only graph nodes:

- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`

They may appear directly only inside `fabric.module`.

### Name Resolution and Host Scope

Named definitions are resolved lexically by host scope. The relevant host scopes
are:

- the top-level `builtin.module`
- `fabric.module`
- `fabric.spatial_pe`
- `fabric.temporal_pe`

Two named definitions in the same host scope may not share the same name, even
if they have different Fabric operation kinds. This conflict space is shared
across module types so that `fabric.instance` resolution remains unambiguous.

## Hardware Parameters vs Runtime Configuration

Each operation separates:

- hardware parameters: physical structure, fixed for an instance
- runtime configuration: values programmed by the mapper

For FCC, this split is especially important for:

- `spatial_sw` connectivity versus route tables
- `spatial_pe` structure versus opcode, mux, demux, and FU config selections
- `function_unit` static structure versus selected `mux` settings

When an ADG is given to the mapper, pre-populated runtime-config fields are
treated as hints unless a more specific spec says otherwise. The mapping output
is the authoritative source of final runtime configuration.

## Textual Assembly Convention

FCC follows one textual convention across the Fabric dialect:

- hardware parameters are serialized in square brackets `[...]`
- runtime-configurable state is serialized in braces, either as bare
  attribute-dict braces `{...}` or as `attributes {...}` for ops with custom
  assembly forms

Examples:

- `fabric.function_unit ... [latency = 1, interval = 1]`
- `fabric.mux ... {sel = 0 : i64, discard = false, disconnect = false}`
- `fabric.temporal_pe ... [num_register = 0 : i64, num_instruction = 4 : i64,
  reg_fifo_depth = 0 : i64] attributes {instruction_mem = [...] }`
- `fabric.map_tag ... [table_size = 4 : i64] attributes {table = [[1, 0, 3], ...] }`
- `fabric.memory` or `fabric.extmemory` use `[]` for fixed memory-interface
  structure and `attributes {}` for region programming such as
  `addr_offset_table`

Structural attributes are not part of this split. Names and type metadata such
as `sym_name`, `module` symbol references, and `function_type` remain part of
the operation's structural syntax.

One additional structural attribute is currently standardized for
visualization:

- `fabric.module` may carry `attributes {viz_file = "..."}` to reference a
  sidecar JSON file that provides explicit visualization layout metadata

`viz_file` is neither a hardware parameter nor runtime configuration. It is a
structural pointer consumed by visualization tooling.

## Related Documents

- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-fifo.md](./spec-fabric-fifo.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)
- [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
