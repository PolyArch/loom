# FCC Fabric config_mem and Bitstream Specification

## Overview

`config_mem` is FCC's unified runtime-configuration image for one mapped
`fabric.module`.

This document defines:

- the logical `config_mem` model
- the serialized word-stream format used by FCC artifacts
- the slice ordering and field packing rules used by current FCC `ConfigGen`
- the per-resource configuration layouts that contribute to the final
  bitstream

This document plays the same role in FCC that
`spec-fabric-config_mem.md` played in Loom, but FCC's slice structure is
different because compute containers are now built around:

- `fabric.spatial_pe`
- `fabric.temporal_pe`
- `fabric.function_unit`

Related documents:

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)
- [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md)
- [spec-mapper-output.md](./spec-mapper-output.md)
- [spec-runtime-mmio.md](./spec-runtime-mmio.md)

## Role of `config_mem`

FCC uses one unified configuration image to capture all runtime-programmable
state needed by one mapped accelerator instance.

This includes:

- routing configuration
- tag-boundary configuration
- memory-region configuration
- PE opcode and mux or demux selections
- FU-internal runtime-config fields
- temporal instruction-slot routing state

The same logical word stream is emitted in three artifact views:

- `<mixed>.config.bin`
- `<mixed>.config.json`
- `<mixed>.config.h`

These are three representations of the same configuration image:

- `config.bin`: raw little-endian bytes
- `config.json`: structured summary with slice metadata
- `config.h`: C header embedding the same 32-bit words as
  `fcc_accel_config_words[]`

## Physical Word Model

FCC currently uses a fixed 32-bit configuration word width.

Normative properties:

- word width is `32` bits
- word index order is ascending from the low end of the bitstream
- binary serialization is little-endian per word
- host upload is word-oriented

Current exported summary fields in `config.json` are:

- `word_width_bits`
- `word_count`
- `byte_size`
- `complete`
- `words`
- `slices`

The raw binary payload in `config.bin` is exactly the `words` array serialized
as consecutive little-endian `uint32_t` values.

## Slice Model

FCC does not treat the configuration image as one monolithic anonymous bit
array. Instead, it is assembled from ordered slices.

Each slice has:

- a `name`
- a `kind`
- an optional `hw_node` id
- a `word_offset`
- a `word_count`
- a `complete` flag

`config.json` is the authoritative slice table.

## Current Slice Ordering

Current FCC `ConfigGen` emits slices in two stages.

### Stage 1: Primitive and Storage Slices

First, FCC walks flattened ADG nodes in flattened node order and emits slices
for configurable primitive resources.

Current primitive slice families are:

- `spatial_sw`
- `temporal_sw`
- `add_tag`
- `map_tag`
- `fifo` when bypassable
- `memory`
- `extmemory`

Only nodes that actually produce one non-empty serialized payload appear in
the final slice table.

### Stage 2: PE Container Slices

After primitive slices, FCC emits one slice per PE container in PE-containment
order.

Current PE slice families are:

- `spatial_pe`
- `temporal_pe`

This means the current FCC bitstream order is:

1. configurable routing, tag, FIFO, and memory primitive slices
2. then PE container slices

This differs from Loom's older `fabric.pe`-centric layout because FCC models
PE routing and FU state at the container level.

## Alignment and Packing Rules

### Slice Alignment

Every slice is word-aligned.

Normative consequences:

- two different slices never share a 32-bit word
- a slice may occupy one or more complete words
- unused high bits in the final word of one slice are zero

This preserves simple host-side per-slice updates and avoids cross-slice
read-modify-write.

### Intra-Slice Packing

Inside one slice, fields are packed low-to-high unless that slice family is
defined as a word-table structure.

General bit-packing rules:

- earlier fields occupy lower bit positions
- later fields occupy higher bit positions
- array-like fields use ascending logical index order
- a field may legally straddle a 32-bit word boundary within the same slice

The core helper rule used by current FCC `ConfigGen` is:

- `packBits(...)` appends fields LSB-first
- `packMuxField(...)` appends `[sel | discard | disconnect]` low-to-high

### Choice Width Rule

Whenever FCC needs to encode a choice among `N` alternatives, the current
selection width is:

- `0` bits when `N <= 1`
- `ceil(log2(N))` bits when `N > 1`

This rule is used for:

- PE opcode fields
- PE input-mux `sel`
- PE output-demux `sel`
- temporal register index fields

## Primitive Slice Layouts

This section defines the current FCC slice layouts used by `ConfigGen`.

### `fabric.spatial_sw`

Current generated `spatial_sw` slices use one route bit per connected
`input -> output` position.

Bit count:

- `route_bits = popcount(connectivity_table)`
- if `connectivity_table` is omitted, `route_bits = num_inputs * num_outputs`

Bit order:

- output-major
- within one output, input-major
- only connected positions contribute bits

Bit value:

- `1` means this connected transition is active in the current mapped result
- `0` means it is inactive

Current implementation note:

- `spec-fabric-spatial_sw.md` defines both route selection and discard
  semantics
- current `ConfigGen` artifact export serializes the route-bitmap portion
  only

### `fabric.temporal_sw`

`temporal_sw` uses a multi-slot route-table image.

Per-slot low-to-high field order:

- `valid`
- `tag`
- route transition bits

Here:

- `tag` width equals the tagged-port tag width of the switch
- route transition bit order uses the same connected-position ordinal order as
  `spatial_sw`

Bit count:

- `slot_width = 1 + tag_width + route_bits`
- `total_bits = num_route_table * slot_width`

Semantic rule:

- the table is matched by tag
- tag is not used as a direct array index

### `fabric.add_tag`

Current generated `add_tag` slices occupy one 32-bit word.

Low bits hold:

- the configured runtime tag value

Unused upper bits are zero.

### `fabric.map_tag`

`map_tag` is packed entry-by-entry.

Per-entry low-to-high field order:

- `valid`
- `src_tag`
- `dst_tag`

Bit count:

- `table_size * (1 + in_tag_width + out_tag_width)`

Entries are emitted in ascending table index order.

### `fabric.fifo`

Only bypassable FIFOs contribute one runtime bit.

Low bit:

- `0`: buffered mode
- `1`: bypass mode

Non-bypassable FIFOs contribute no slice.

### `fabric.memory` and `fabric.extmemory`

FCC memory-facing nodes contribute region-table configuration through
`addr_offset_table`.

Current generated memory-family slices are word tables, not bit-packed record
streams.

Each region contributes exactly five 32-bit words in this order:

1. `valid`
2. `start_lane`
3. `end_lane`
4. `addr_offset`
5. `elem_size_log2`

Total word count:

- `5 * num_region`

Semantics:

- `valid`: whether this region entry is occupied
- `start_lane`, `end_lane`: logical lane range served by this region
- `addr_offset`: base offset of the region
- `elem_size_log2`: AXI-style element-size code

Important FCC distinction:

- `fabric.memory` may carry concrete mapped base offsets
- `fabric.extmemory` is emitted by the mapper with `addr_offset = 0`
- host runtime is responsible for patching or programming the actual backing
  base for `extmemory`

This is a major FCC difference from Loom's older assumption that memory or
extmemory contributed no runtime configuration.

## PE Container Slice Layouts

PE container slices are the most important FCC difference relative to Loom.

FCC does not serialize one old-style `fabric.pe` record. It serializes
container-level state for `spatial_pe` and `temporal_pe`.

### `fabric.spatial_pe`

One `spatial_pe` contributes one container-local packed record.

Low-to-high field order:

- `spatial_pe_enable`
- `opcode`
- input mux controls
- output demux controls
- FU internal config payload

Field widths:

- `spatial_pe_enable`: `1`
- `opcode_bits = choice_width(num_function_units)`
- `input_mux_sel_bits = choice_width(num_pe_inputs)`
- `output_demux_sel_bits = choice_width(num_pe_outputs)`
- one input control field per `max_fu_inputs`
- one output control field per `max_fu_outputs`
- `fu_payload_bits = max(fu_config_bits among FUs in this PE)`

Each input or output control field uses:

- `sel`
- `discard`
- `disconnect`

That is, width:

- `input_mux_width = input_mux_sel_bits + 2`
- `output_demux_width = output_demux_sel_bits + 2`

Important FCC-specific rules:

- exactly zero or one physical FU is active in one `spatial_pe`
- only the selected FU's internal payload is emitted into the shared FU-config
  region
- that shared region is sized by the maximum FU config width in the PE
- if the selected FU uses fewer bits than the shared region width, the high
  remainder is zero-padded

This is one of the biggest FCC differences from Loom.

### `fabric.temporal_pe`

One `temporal_pe` contributes one packed container slice consisting of:

- instruction-slot region
- persistent per-FU config region

#### Slot Region

For each instruction slot, low-to-high field order is:

- `valid`
- `tag`
- `opcode`
- operand configs
- input mux controls
- output demux controls
- result configs

Field widths:

- `valid`: `1`
- `tag`: `pe.tag_width`
- `opcode_bits = choice_width(num_function_units)`
- `reg_idx_bits = choice_width(num_register)`
- `operand_cfg_width = 0` when `num_register == 0`, else
  `reg_idx_bits + 1`
- `input_mux_width = choice_width(num_pe_inputs) + 2`
- `output_demux_width = choice_width(num_pe_outputs) + 2`
- `result_cfg_width = tag_width + operand_cfg_width`

Counts:

- one operand-config field per `max_fu_inputs`
- one input-mux field per `max_fu_inputs`
- one output-demux field per `max_fu_outputs`
- one result-config field per `max_fu_outputs`

Operand-config low-to-high order:

- `reg_idx`
- `is_reg`

Result-config low-to-high order:

- `result_tag`
- `reg_idx`
- `is_reg`

Important temporal semantics:

- `opcode` chooses which FU fires in that slot
- input mux or output demux state is per slot
- operand/result register selection is per slot
- the slot `tag` is the routed temporal tag used for matching and result
  bookkeeping

#### Persistent FU-Config Region

After all instruction slots, FCC appends persistent FU-internal config
payloads.

Concatenation order:

- FU definition order inside the `temporal_pe`
- within one FU, body occurrence order of configurable fields

Width:

- `sum(fu_config_bits of all FUs in this temporal_pe)`

This is another major FCC difference from Loom:

- temporal PE instruction slots do not carry duplicated copies of FU-internal
  config
- FU-internal config is persistent per FU, not per instruction slot

## FU-Internal Payload Layout

The FU-internal payload referenced by PE slices is defined by
[spec-fabric-function_unit.md](./spec-fabric-function_unit.md).

Current configurable FU-body field kinds are:

- `fabric.mux`
- `handshake.constant`
- `handshake.join`
- `arith.cmpi`
- `arith.cmpf`
- `dataflow.stream`

Within one FU body, the payload is concatenated in body occurrence order.

Per-field encodings are:

- `fabric.mux`: `[sel | discard | disconnect]`
- `handshake.constant`: literal value bits
- `handshake.join`: `join_mask`
- `arith.cmpi`: 4-bit predicate value
- `arith.cmpf`: 4-bit predicate value
- `dataflow.stream`: 5-bit one-hot `cont_cond`

FCC-specific consequence:

- `spatial_pe` stores only the active FU payload in one shared max-width slot
- `temporal_pe` stores all FU payloads persistently in FU-definition order

## Artifact Semantics

### `config.bin`

`config.bin` is the raw serialized word stream.

Properties:

- words are emitted in slice order
- each word is written little-endian
- no extra framing bytes exist

### `config.h`

`config.h` embeds the same word stream as:

- `fcc_accel_config_words[]`
- `fcc_accel_config_word_count`
- `fcc_accel_config_complete`

Host code may load this array directly through `fcc_accel_load_config(...)`.

### `config.json`

`config.json` is the structured and most inspectable representation.

It includes:

- the flattened `words`
- slice metadata
- total completeness status

The `complete` flags are important:

- one slice may be marked incomplete when the current mapper/config generator
  could not fully recover all fields for that slice
- the top-level `complete` bit is the conjunction across emitted slices

This lets FCC distinguish:

- architectural config layout
- current implementation coverage

## Current Coverage Boundary

The current FCC `ConfigGen` models and emits slices for:

- routing primitives
- tag primitives
- memory and extmemory region tables
- `spatial_pe`
- `temporal_pe`
- bypassable FIFOs

Therefore the current exported bitstream already covers the main
configuration-bearing structures of FCC's mapped accelerator view.

At the same time, `config.json` completeness flags remain authoritative for
whether every slice was fully reconstructed for one concrete mapping result.

## FCC Differences Relative to Loom

The most important FCC differences are:

- `fabric.pe` is no longer the central configuration carrier
- `fabric.spatial_pe` adds:
  - `spatial_pe_enable`
  - active FU `opcode`
  - PE input mux controls
  - PE output demux controls
  - shared active-FU internal config payload
- `fabric.temporal_pe` adds:
  - per-slot `opcode`
  - per-slot input mux controls
  - per-slot output demux controls
  - per-slot operand/result register steering
  - persistent per-FU internal config payloads
- `fabric.function_unit` contributes explicit configurable body fields through
  `fabric.mux`, `handshake.constant`, `handshake.join`, `arith.cmpi`,
  `arith.cmpf`, and `dataflow.stream`
- `fabric.memory` and `fabric.extmemory` now contribute region-table config via
  `addr_offset_table`
- FCC's exported bitstream is assembled from primitive slices first and PE
  container slices second

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)
- [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md)
- [spec-mapper-output.md](./spec-mapper-output.md)
- [spec-runtime-mmio.md](./spec-runtime-mmio.md)
