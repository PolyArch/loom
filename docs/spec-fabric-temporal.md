# FCC Temporal Fabric Specification

## Overview

FCC keeps temporal hardware as a first-class part of the Fabric design, even
when MVP flows focus mainly on `spatial_pe` and `spatial_sw`.

This document summarizes the temporal model without repeating every low-level
encoding detail from the rebuild plan.

## `fabric.temporal_pe`

A temporal PE contains:

- multiple `function_unit` instances
- instruction slots
- operand routing and result routing state
- optional register storage
- per-FU persistent internal configuration
- one operand-buffer hardware mode selection

Textual assembly follows the Fabric-wide split:

- hardware parameters live in `[]`
- runtime configuration such as `instruction_mem` lives in `attributes {}`

Canonical syntax:

```mlir
fabric.temporal_pe @name(...) -> (...)
  [num_register = R : i64,
   num_instruction = I : i64,
   reg_fifo_depth = F : i64,
   enable_share_operand_buffer = false]
  attributes {instruction_mem = [...] } {
  ...
}
```

The temporal PE container slice is organized low-to-high as:

- instruction slot `0`
- instruction slot `1`
- ...
- instruction slot `num_instruction - 1`
- global per-function_unit internal config payloads

Opcode numbering is derived from `function_unit` definition order inside the
`temporal_pe` body, starting from `0`.

Each instruction slot chooses:

- opcode
- tag match
- input mux selections
- output demux selections
- register read and write behavior

Here, `tag` and `res_tag` are runtime tag values used by temporal execution
and temporal switching. They are not hardware tag-parameter inference. The
hardware tag width of a `temporal_pe` interface comes directly from the tagged
port types declared by the ADG.

The temporal PE may execute different instructions over time, but each
`function_unit`'s internal `mux` configuration is not reselected per
instruction in the base FCC design. Those FU-internal config bits live in the
container-global suffix above the instruction-memory region.

The operand buffer mode is a hardware parameter, not a runtime config choice:

- `enable_share_operand_buffer = false` selects per-instruction operand buffers
- `enable_share_operand_buffer = true` selects one shared operand buffer with
  per-tag FIFO behavior
- `operand_buffer_size` is meaningful only in shared-buffer mode

These fields are validated when parsing or verifying `fabric.temporal_pe`.
They do not contribute bits to `instruction_mem` or other `config_mem`
payloads.

Within one instruction slot, the low-to-high bit layout is:

- `valid`
- `tag`
- `opcode`
- operand configs
- input mux controls
- output demux controls
- result configs

Sizing rules:

- operand, input-mux, output-demux, and result-config counts use
  `max_fu_inputs` and `max_fu_outputs` across all FUs in the temporal PE
- mux and demux control fields use the low-to-high order
  `[sel | discard | disconnect]`
- when `num_register > 0`, each operand field is packed low-to-high as
  `[reg_idx | is_reg]`
- when `num_register > 0`, each result field is packed low-to-high as
  `[res_tag | reg_idx | is_reg]`

Current FCC config generation interprets those fields as follows:

- if one operand is sourced from an internal temporal register, its operand
  field sets `is_reg = 1`, its `reg_idx` names the allocated register, and the
  corresponding input mux is emitted in `disconnect` mode
- if one result is written to an internal temporal register, its result field
  sets `is_reg = 1`, its `reg_idx` names the allocated register, and its
  `res_tag` is forced to `0`
- if one result exits the temporal PE, its result field leaves `is_reg = 0`
  and its `res_tag` defaults to the instruction tag

FCC currently models internal temporal dependencies as mapper-generated
`temporal_reg` edges. One register is allocated per writer software output
port, so one writer may feed multiple readers through the same register.

The global per-function_unit config suffix is concatenated in two nested orders:

- `function_unit` order follows the `temporal_pe` body definition order
- within one `function_unit`, `fabric.mux` fields follow body occurrence
  order

For result configs:

- when an instruction result writes a register, the emitted `res_tag` is `0`
- when an instruction result exits the temporal PE, the default `res_tag`
  inherits the instruction tag

Current implementation note:

- the base FCC mapper emits at most one instruction slot per physical
  `function_unit`
- it explicitly rejects forced conflicting reuse of one physical
  `function_unit` when different slot candidates require incompatible
  `fabric.mux` settings
- same-config repeated reuse is not yet materialized as multiple slots in the
  current mapper

Hardware-parameter legality rules:

- `num_instruction > 0`
- `num_register >= 0`
- `reg_fifo_depth = 0` when `num_register = 0`
- `reg_fifo_depth >= 1` when `num_register > 0`
- if `enable_share_operand_buffer = false`, `operand_buffer_size` must be
  absent
- if `enable_share_operand_buffer = true`, `operand_buffer_size` must be
  present and lie in `[1, 8192]`

For an unused instruction slot, the default serialized state is:

- `valid = 0`
- all remaining bits in that slot are `0`

## `fabric.temporal_sw`

A temporal switch uses tag-indexed route tables.

Key properties:

- route selection depends on tag
- per-output routing remains mux-like
- one observed runtime tag may select at most one input-to-output transition
  within one `fabric.temporal_sw`
- arbitration and temporal correctness are part of the switch semantics
- every port must be `!fabric.tagged<...>` with exactly the same tagged type
- `num_route_table >= 1`
- if `connectivity_table` is present, it has one binary row per output and one
  binary column per input
- other Fabric operations may route tagged payloads, but only
  `fabric.add_tag`, `fabric.map_tag`, and `fabric.del_tag` may change tagged
  shape at a boundary

## Relationship to Spatial Components

The temporal model reuses the same broad concepts as spatial hardware:

- mux or demux controlled ingress and egress
- route-table driven switching
- explicit separation between hardware structure and runtime configuration

The main difference is that temporal hardware introduces time, tags, and
instruction state into the legality model.

## Mapper Implications

Temporal mapping must respect:

- slot capacity
- tag uniqueness where required
- register legality
- FU configuration consistency across all uses of one physical FU

These constraints are part of the mapper model rather than this summary spec.
