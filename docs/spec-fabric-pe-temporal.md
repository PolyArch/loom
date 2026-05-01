# Fabric PE (temporal schedule)

This document specifies the temporal-schedule branch of `fabric.pe`. The
spatial branch is documented in `spec-fabric-pe.md`. Both branches share
the same op (`Fabric_PeOp` in `include/Fabric/IR/FabricOps.td`); the
verifier dispatches on the mandatory `schedule` predicate. The temporal
verifier lives in `lib/Fabric/IR/FabricPeTemporalOps.cpp`.

## Schedule predicate dispatch

`fabric.pe [temporal]` selects the time-multiplexed branch. A single PE
holds one or more inner `fabric.fu` instances and a per-PE instruction
memory of length `num_instruction`. Each instruction slot may fire one
inner FU per cycle with operand routing across PE input ports and a
local register-FIFO bank.

Both anonymous and named-template forms are accepted:

```mlir
%out = fabric.pe [temporal]
           (%pa = %a : !fabric.bits_tag<32, 4> to !fabric.bits<32>)
           -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 4 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } { ... }

fabric.pe @TempPe [temporal] (!fabric.bits_tag<32, 4>)
                              -> (!fabric.bits_tag<32, 4>)
     attributes { ... } {
^bb0(%pa: !fabric.bits_tag<32, 4>):
  fabric.fu(%fa = %pa : !fabric.bits_tag<32, 4> to !fabric.bits<32>)
            -> (!fabric.bits<32>) { ... }
  fabric.yield %pa : !fabric.bits_tag<32, 4>
}
```

## Boundary type rule

Every PE input port and every PE output port has type
`!fabric.bits_tag<W, T>` with the same `W >= 0` and the same `T >= 1`.
The verifier extracts `(W, T)` from PE input #0 and rejects any other
port with a different shape. The `tag_width` hardware attribute must
equal `T`.

The anonymous form supports a kind-changing `to` clause on the PE
operand list: outer `!fabric.bits_tag<W, T>` may be paired with inner
`!fabric.bits<W>` (the tag is stripped at the PE-to-FU boundary). This
is the only fabric-kind transition permitted at the PE boundary; spatial
PEs continue to require uniform `bits<W>`.

Inner `fabric.fu` ops follow the same width relaxation rule as in the
spatial branch: each FU input may use the FU-level `to <inner-type>`
clause to narrow its inner block-arg width. A temporal-PE-parented FU
may additionally accept `!fabric.bits_tag<W, T>` on its outer input
side, which the FU's own boundary then strips down to `!fabric.bits<W>`
(or narrower) using the FU `to` clause. FU output ports remain strict
`!fabric.bits<W>`.

## Hardware parameters

All eight parameters are op-level attributes. They are present only on
`fabric.pe [temporal]`; the verifier rejects any of them on a
`fabric.pe [spatial]` (spatial PEs must not carry temporal-only
attributes).

| attribute             | type             | requirement                                                     |
| --------------------- | ---------------- | --------------------------------------------------------------- |
| `tag_width`           | `I32Attr`        | required, `>= 1`, must equal boundary `T`                       |
| `num_instruction`     | `I32Attr`        | required, `>= 1`                                                |
| `num_reg_fifo`        | `I32Attr`        | optional (default 0), `>= 0`                                    |
| `reg_fifo_depth`      | `I32Attr`        | required iff `num_reg_fifo > 0`; absent or `0` otherwise        |
| `reg_fifo_ports`      | `I32Attr`        | optional (default 1); must be `1` or `2`                        |
| `fu_config_mode`      | `StrAttr`        | required, `"per_instruction_fu_config"` or `"per_fu_config"`    |
| `operand_buffer_mode` | `StrAttr`        | required, `"per_instruction"` / `"per_input_port"` / `"all_fu_share"` |
| `operand_buffer_size` | `I32Attr`        | required iff `operand_buffer_mode != "per_instruction"`; absent otherwise |

The `K = numInputs()` and `L = numOutputs()` shape parameters are
read from the op signature (anonymous form) or the `function_type`
attribute (named form). Both must be `>= 1`.

The implicit shape parameters `num_fu`, `max_fu_inputs`,
`max_fu_outputs` are derived from the body: counting `fabric.fu` ops
plus `fabric.instantiate` ops, and taking the per-FU max of input and
output counts.

## Software configuration

When the PE is "programmed" (carries software configuration), three
attributes must be present. They obey an all-or-nothing rule: either all
present or all absent. When `fu_config_mode == "per_instruction_fu_config"`,
`per_fu_sw_configs` is replaced by per-instruction `fu_sw_configs`
embedded inside each `instruction_mem` entry, and the trio reduces to
`{ pe_enable, instruction_mem }` (no top-level `per_fu_sw_configs`).

* `pe_enable : BoolAttr`. PE-level enable. `false` gates the PE off; per-
  instruction enable bits then determine which slots fire.
* `instruction_mem : ArrayAttr`. Length must equal `num_instruction`.
  Each entry is a `DictionaryAttr` (see "Per-instruction format" below).
* `per_fu_sw_configs : ArrayAttr`. Required only when
  `fu_config_mode == "per_fu_config"`. Length equals the body's
  `num_fu`; entries are FU-specific `DictionaryAttr` blobs (loose
  validation in this iteration).

When the PE is "hw-only" (no software configuration), the trio is
absent.

## Per-instruction format

Each `instruction_mem` entry is a `DictionaryAttr` with the following
keys.

```
{
  enable      : BoolAttr,            // 1 bit, LSB of the instruction word
  opcode      : IntegerAttr,         // log2Ceil(num_fu) bits
  operand_sel : ArrayAttr (length max_fu_inputs)
                of DictionaryAttr {
                  src_sel    : IntegerAttr,
                  tag        : IntegerAttr,
                  is_port    : BoolAttr,
                  discard    : BoolAttr,
                  disconnect : BoolAttr
                },
  result_sel  : ArrayAttr (length max_fu_outputs)
                of DictionaryAttr {
                  dst_sel    : IntegerAttr,
                  tag        : IntegerAttr,
                  is_port    : BoolAttr,
                  discard    : BoolAttr,
                  disconnect : BoolAttr
                },
  fu_sw_configs : DictionaryAttr     // present iff fu_config_mode is
                                     // "per_instruction_fu_config"
}
```

Validation rules (per entry `i ∈ [0, num_instruction)`):

* `opcode` must be in `[0, num_fu)`.
* For each `operand_sel[j]` (`j ∈ [0, max_fu_inputs)`):
  * If `is_port == true`, `src_sel ∈ [0, K)`.
  * If `is_port == false`, then `num_reg_fifo > 0` and `src_sel ∈ [0,
    num_reg_fifo)`. Setting `is_port == false` while
    `num_reg_fifo == 0` is rejected.
  * `discard && disconnect` is rejected.
* For each `result_sel[j]` (`j ∈ [0, max_fu_outputs)`):
  * Same rules as `operand_sel[j]`, with `dst_sel` replacing `src_sel`
    and `L` replacing `K`.

The bit layout for each operand-sel and result-sel field, low-to-high,
is:

```
[ src_sel|dst_sel | tag | is_port | discard | disconnect ]
```

The verifier does not pack the IR into bits; this layout is fixed for
the configuration generator.

## Width formulas

```
opcode_width        = log2Ceil(num_fu)
src_sel_width       = max(log2Ceil(K), log2Ceil(num_reg_fifo))
dst_sel_width       = max(log2Ceil(L), log2Ceil(num_reg_fifo))
operand_field_width = src_sel_width + T + 1 (is_port) + 1 (discard) + 1 (disconnect)
result_field_width  = dst_sel_width + T + 1 (is_port) + 1 (discard) + 1 (disconnect)
fu_cfg_width_max    = max(fu_config_bitwidth(fu_i)) over inner FUs (per_instruction_fu_config mode)
                    = sum(fu_config_bitwidth(fu_i)) (per_fu_config mode, stored separately)

instruction_word_width =
    1                                        // enable
  + opcode_width
  + max_fu_inputs  * operand_field_width
  + max_fu_outputs * result_field_width
  + (per_instruction_fu_config ? fu_cfg_width_max : 0)
```

The verifier does not numerically enforce these widths; the IR carries
structured `DictionaryAttr`s and the configuration generator emits the
bit-packed layout.

## Reg FIFO semantics

When `num_reg_fifo > 0`, the PE owns a bank of `num_reg_fifo` register
FIFOs, each of depth `reg_fifo_depth` and with `reg_fifo_ports` ports
(`1` for single-ported, `2` for separate read/write). Each register
slot stores a `(data, tag)` pair, identical in shape to a single
`!fabric.bits_tag<W, T>` token. Writing to a register pushes one such
pair (`result_sel.tag` is the tag value); reading pops the head.

The `is_port == false` form on `operand_sel`/`result_sel` selects a reg
FIFO instead of a PE port; `src_sel`/`dst_sel` is the FIFO index in
`[0, num_reg_fifo)`.

## Operand buffer modes

* `per_instruction`: each instruction slot owns a depth-1 operand buffer
  per FU input (no `operand_buffer_size`).
* `per_input_port`: one operand buffer per FU input port, shared across
  instructions. Depth `operand_buffer_size` per buffer.
* `all_fu_share`: a single shared buffer of total depth
  `operand_buffer_size` serves all FU inputs.

All three are ordered-dataflow models: tokens leave the buffer in
arrival order; an instruction fires when the first token at each
selected operand source carries the matching tag.

## Trigger condition

An instruction at slot `i` fires when:

1. Slot `i` is enabled (`pe_enable && instruction_mem[i].enable`).
2. For every `operand_sel[j]` with `is_port == true` and
   `discard|disconnect == false`, the head token of the selected source
   (PE input port `src_sel` or reg FIFO `src_sel`) is available with a
   tag matching `operand_sel[j].tag`.
3. The selected FU (`opcode`) is ready to consume.

`discard` drains the input slot regardless of consumption; `disconnect`
treats the source as unconnected for that slot. `discard && disconnect`
is forbidden.

## Body whitelist

Identical to spatial: only `fabric.fu` and `fabric.instantiate`. The
named-template form additionally requires a closing `fabric.yield`. No
other op kind is permitted.

## Cross-reference

* Spatial branch and the spatial instruction word format:
  `spec-fabric-pe.md`.
* PE op IR shape (operand types, attribute schema): `Fabric_PeOp` in
  `include/Fabric/IR/FabricOps.td`.
* Verifier rules: `lib/Fabric/IR/FabricPeTemporalOps.cpp`.
* Per-FU runtime config catalogue:
  `spec-fabric-reconfigurable-op.md`.
* Boundary ops bridging spatial and temporal domains:
  `spec-fabric-boundary.md`.
