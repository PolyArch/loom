# Fabric config_mem Specification

## Overview

`config_mem` is the unified configuration memory interface for all runtime
configurable hardware modules in the Fabric dialect. This document is the
authoritative definition of config_mem.

## Definition

`config_mem` is a register array that holds all runtime configuration for a
`fabric.module`. It provides:

- Unified memory-mapped access via AXI-Lite (RTL) or TLM 2.0 (SystemC)
- Simple software programming model
- Deterministic configuration sequence

### Physical Properties

- **Word width**: 32 bits (fixed)
- **Depth**: Determined by the total configuration bits of all modules with
  `CONFIG_WIDTH > 0` within a `fabric.module` (see [Depth Calculation](#depth-calculation))
- **Access model**: Read-write from host, read-only from accelerator
- **Reset behavior**: config_mem registers retain their values across
  accelerator reset. Reset halts dataflow execution but does not clear
  configuration state.

### Depth Calculation

The config_mem depth (number of 32-bit words) is calculated as follows:

1. For each module with `CONFIG_WIDTH > 0` within the `fabric.module`,
   take its `CONFIG_WIDTH` (see [CONFIG_WIDTH](#config_width)) as the number
   of configuration bits
2. Round each module's `CONFIG_WIDTH` up to the nearest multiple of 32
3. Sum all rounded values
4. Divide the total by 32

In other words:

```
depth = sum(ceil(CONFIG_WIDTH / 32)) for each module with CONFIG_WIDTH > 0
```

Modules with `CONFIG_WIDTH = 0` do not participate in depth calculation or
address allocation; they occupy no config_mem space.

Each module's configuration bits occupy a contiguous range of 32-bit words.
No two modules share a 32-bit word.

### ADDR_WIDTH

`ADDR_WIDTH` is the byte-address width of the config_mem interface, derived
from depth:

```
ADDR_WIDTH = ceil(log2(depth * 4))    // depth > 0
```

The factor of 4 accounts for 4 bytes per 32-bit word. `ADDR_WIDTH` determines
the width of AXI-Lite address buses and TLM address fields.

When `depth = 0` (all modules have `CONFIG_WIDTH = 0`), no config_mem is
generated and `ADDR_WIDTH` is not applicable.

### CONFIG_WIDTH

`CONFIG_WIDTH` is a **derived parameter** for each module. It is not an
independently settable parameter; it is determined by the module's hardware
parameters:

| Module Type | CONFIG_WIDTH Formula |
|-------------|---------------------|
| `fabric.pe` (tagged, non-constant non-load/store) | `NUM_OUTPUTS * TAG_WIDTH` |
| `fabric.pe` (constant, native) | `bitwidth(constant_value_type)` |
| `fabric.pe` (constant, tagged, `NUM_OUTPUTS = 1`) | `bitwidth(constant_value_type) + TAG_WIDTH` |
| `fabric.pe` (dataflow.stream, native) | `5` (`cont_cond_sel`, one-hot [`<`, `<=`, `>`, `>=`, `!=`]) |
| `fabric.pe` (dataflow.stream, tagged) | `NUM_OUTPUTS * TAG_WIDTH + 5` |
| `fabric.pe` (load/store, TagOverwrite + tagged) | `TAG_WIDTH` (single shared `output_tag` for all outputs) |
| `fabric.pe` (load/store, TagOverwrite + native) | 0 |
| `fabric.pe` (load/store, TagTransparent) | 0 |
| `fabric.pe` (compute, native) | 0 (no config) |
| `fabric.add_tag` | `TAG_WIDTH` |
| `fabric.map_tag` | `TABLE_SIZE * (1 + IN_TAG_WIDTH + OUT_TAG_WIDTH)` |
| `fabric.switch` | `K` (number of connected positions in connectivity_table) |
| `fabric.temporal_pe` | `NUM_INSTRUCTIONS * INSTRUCTION_WIDTH` |
| `fabric.temporal_sw` | `NUM_ROUTE_TABLE * SLOT_WIDTH` |
| `fabric.memory` | 0 (no runtime config) |
| `fabric.extmemory` | 0 (no runtime config) |
| `fabric.fifo` (not bypassable) | 0 (no config) |
| `fabric.fifo` (bypassable) | 1 (`bypassed` flag) |
| `fabric.del_tag` | 0 (no config) |

See the individual spec-fabric-*.md documents for detailed bit width formulas.

## Address Allocation

Configuration addresses are allocated sequentially based on MLIR operation
order within `fabric.module`. Only operations with `CONFIG_WIDTH > 0`
participate in address allocation; operations with `CONFIG_WIDTH = 0` are
skipped and occupy no config_mem space.

1. Traverse operations in definition order
2. For each operation with `CONFIG_WIDTH > 0`, allocate space starting at the
   next 32-bit aligned boundary
3. The operation's config bits occupy consecutive 32-bit words
4. Unused bits within a word are tied to zero
5. Repeat until all operations with `CONFIG_WIDTH > 0` are processed

Each node's bits are isolated within its allocated words. No two nodes share
a 32-bit word.

### Design Rationale: Per-Module Word Alignment

config_mem deliberately aligns each module's configuration to a 32-bit word
boundary rather than packing bits contiguously across modules. For example, if
module A uses bits `[6:0]` of word 0, module B starts at word 1 rather than at
bit 7 of word 0.

This design is motivated by two considerations:

1. **Atomic per-module reconfiguration.** With word-aligned allocation, any
   module's configuration can be updated by writing one or more complete 32-bit
   words. No read-modify-write cycle is required because adjacent modules never
   share a word. A compact bit-packed layout would force the host to read the
   current word, mask and merge the target bits, and write the result back --
   adding complexity and potential race conditions in partial reconfiguration
   scenarios.

2. **No real hardware cost.** Unused upper bits within a word are tied to zero
   and are never read by logic. Synthesis and place-and-route tools eliminate
   the corresponding flip-flops during optimization, so the apparent storage
   waste does not translate into additional area or power. The only cost is
   address space, which scales exponentially with address width (each
   additional address bit doubles the addressable range) and is effectively
   free for the small configuration memories typical of accelerator fabrics.

## Bit Layout

Configuration bits are packed starting from the LSB of each word:

```
Word N:   [31:B] = 0 (tied low)         // B = min(CONFIG_WIDTH, 32)
          [B-1:0] = config data

Word N+1: [31:R] = 0 (tied low)         // R = CONFIG_WIDTH - 32
          [R-1:0] = overflow config data
```

Here `B` is the number of config bits that fit in the first word, and `R` is
the number of remaining bits that overflow into subsequent words.

For multi-word configurations, bits flow from LSB of word 0 to MSB of word 0,
then LSB of word 1, and so on.

Fields within a module's config_mem allocation are packed continuously
(LSB-first). The 32-bit word alignment applies to the entire module's total
config bits, not individual fields. A single field may straddle a 32-bit word
boundary within the same module's allocation; only module boundaries are
word-aligned.

General field packing rule:

- Fields are packed in spec-defined field order, LSB-first.
- Array fields are packed by ascending index (index 0 in lower bits).
- Later fields occupy higher bit positions than earlier fields.

### Layout Example

The following example shows two modules packed into four 32-bit words.
Module A has `CONFIG_WIDTH = 38` with fields `a` (20 bits) and `b` (18 bits).
Module B has `CONFIG_WIDTH = 40` with fields `c` (12 bits) and `d` (28 bits).

```
          31                                      0
          ┌──────────────┬────────────────────────┐
Word 0    │   b[11:0]    │         a[19:0]        │  Module A
          ├──────────────┴─────┬──────────────────┤
Word 1    │  (0)               │   b[17:12]       │  Module A
          ╞════════════════════╧══════════════════╡
Word 2    │    d[19:0]     │       c[11:0]        │  Module B
          ├────────────────┴─┬────────────────────┤
Word 3    │  (0)             │     d[27:20]       │  Module B
          └──────────────────┴────────────────────┘
```

Key observations:

- Field `b` (18 bits) straddles the Word 0 / Word 1 boundary within Module A.
  Its lower 12 bits occupy `Word 0[31:20]`, its upper 6 bits occupy
  `Word 1[5:0]`. This is permitted because word alignment is enforced at
  module boundaries, not at field boundaries.
- Field `d` (28 bits) similarly straddles Word 2 / Word 3 within Module B.
- `(0)` denotes unused bits tied low. These bits are not backed by flip-flops
  after synthesis optimization.
- The `╞══╡` line marks a word-aligned module boundary. Module B starts at
  Word 2 regardless of how many bits Module A leaves unused in Word 1.

### Field Packing Examples

- `fabric.pe` constant tagged (`NUM_OUTPUTS = 1`):
  - Lower bits: `constant_value`
  - Upper bits: `output_tag`
- `fabric.pe` tagged non-constant non-load/store:
  - `output_tag[0]` is lowest, then `output_tag[1]`, ..., `output_tag[N-1]`
- `fabric.pe` dataflow.stream tagged:
  - Lower bits: `output_tag` array (ascending output index)
  - Upper bits: 5-bit `cont_cond_sel`

## Access Protocol

### RTL (SystemVerilog)

config_mem is accessed via AXI4-Lite slave interface. `ADDR_WIDTH` is defined
in [ADDR_WIDTH](#addr_width).

- Write: `cfg_awaddr[ADDR_WIDTH-1:2]` selects word index
- Read: `cfg_araddr[ADDR_WIDTH-1:2]` selects word index
- Single-cycle read latency (registered)
- Single-cycle write (combinational address phase, registered data)

### SystemC

config_mem is accessed via TLM 2.0 `simple_target_socket`:

- Blocking transport (`b_transport`)
- Word-addressed (address / 4 = word index)

## Configuration Sequence

### Full Reconfiguration

Software configures the accelerator by:

1. Assert reset
2. Write all config_mem words via AXI-Lite (RTL) or TLM (SystemC)
3. De-assert reset
4. Begin dataflow execution

### Partial Reconfiguration

Per-module partial reconfiguration is supported. Because each module's
configuration is word-aligned and no two modules share a 32-bit word, any
individual module can be reconfigured by writing only its allocated words.
The host does not need to read-modify-write or coordinate with other modules'
configuration state.

Partial reconfiguration requires a reset-reconfigure-restart cycle:

1. Halt dataflow execution
2. Assert reset
3. Write only the target module's config_mem words (all other words are
   retained; see [Reset behavior](#physical-properties))
4. De-assert reset
5. Resume dataflow execution

Updating config_mem words while the accelerator is executing (without reset)
is not supported.

## Related Documents

- [spec-fabric.md](./spec-fabric.md): Fabric dialect overview
- [spec-mapper.md](./spec-mapper.md): Mapper stage that emits runtime configuration values
- [spec-adg.md](./spec-adg.md): ADG specification (config_mem principles)
- [spec-adg-sv.md](./spec-adg-sv.md): SystemVerilog config_mem controller
- [spec-adg-sysc.md](./spec-adg-sysc.md): SystemC config_mem controller
- [spec-fabric-pe.md](./spec-fabric-pe.md): PE configuration
- [spec-fabric-switch.md](./spec-fabric-switch.md): Switch configuration
- [spec-fabric-tag.md](./spec-fabric-tag.md): Tag operations configuration
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md): Temporal PE instruction memory
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md): Temporal switch route tables
- [spec-fabric-error.md](./spec-fabric-error.md): Error code definitions
