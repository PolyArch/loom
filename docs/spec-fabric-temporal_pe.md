# FCC Temporal PE Specification

`fabric.temporal_pe` contains:

- `function_unit` instances
- instruction slots
- operand routing and result routing state
- optional internal registers
- persistent per-function_unit internal configuration

Placement rules:

- definitions may appear directly in the top-level module or in `fabric.module`
- inline instantiations may appear directly only in `fabric.module`

Textual assembly follows the Fabric-wide split:

- hardware parameters in `[]`
- runtime configuration in `attributes {}`

The temporal PE container slice is organized low-to-high as:

- instruction slot `0`
- instruction slot `1`
- ...
- instruction slot `num_instruction - 1`
- global per-function_unit internal config payloads

Opcode numbering follows `function_unit` definition order inside the
`temporal_pe` body, starting from `0`.

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
  `max_fu_inputs` and `max_fu_outputs`
- mux and demux controls use `[sel | discard | disconnect]`
- operand fields use `[reg_idx | is_reg]` when `num_register > 0`
- result fields use `[res_tag | reg_idx | is_reg]` when `num_register > 0`

Global per-function_unit configuration follows:

- `function_unit` definition order in the `temporal_pe` body
- within one `function_unit`, body occurrence order of configurable fields

The operand-buffer mode is a hardware parameter, not runtime config:

- `enable_share_operand_buffer = false`: per-instruction operand buffers
- `enable_share_operand_buffer = true`: one shared operand buffer with per-tag
  FIFO behavior
- `operand_buffer_size` is only meaningful in shared-buffer mode

Unlike `fabric.spatial_pe`, a `fabric.temporal_pe` may activate multiple
distinct physical `function_unit` instances across its instruction slots, as
long as:

- the number of occupied slots does not exceed `num_instruction`
- per-slot operand and result routing is legal
- repeated use of one physical `function_unit` does not require incompatible
  persistent internal configuration

`function_unit` instances inside a `temporal_pe` are compute resources, not
generic routing resources. Routed inter-component edges may terminate at FU
inputs or originate at FU outputs, but they must not traverse through
unrelated FU instances as intermediate transit hops.

Temporal PE operand routing and result routing are selectors, not generic
mixing or broadcasting structures:

- one operand path selects one PE-side ingress for one FU input
- one result path selects one PE-side egress for one FU output
- they must not be used to merge multiple distinct software flows before a FU
  input
- they must not be used to broadcast one software flow to multiple PE-side
  outputs
- flow mixing and dataflow broadcast are legal only inside
  `fabric.spatial_sw` or `fabric.temporal_sw`
