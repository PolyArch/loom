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
- **Depth**: Determined by the total configuration bits of all configurable
  modules within a `fabric.module`
- **Access model**: Read-write from host, read-only from accelerator

### Depth Calculation

The config_mem depth (number of 32-bit words) is calculated as follows:

1. For each configurable module within the `fabric.module`, calculate the
   total number of configuration bits required
2. Round each module's config bits up to the nearest multiple of 32
3. Sum all rounded values
4. Divide the total by 32

In other words:

```
depth = sum(ceil(module_config_bits / 32)) for each configurable module
```

Each module's configuration bits occupy a contiguous range of 32-bit words.
No two modules share a 32-bit word.

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
order within `fabric.module`:

1. Traverse operations in definition order
2. For each configurable operation, allocate space starting at the next 32-bit
   aligned boundary
3. The operation's config bits occupy consecutive 32-bit words
4. Unused bits within a word are tied to zero
5. Repeat until all configurable operations are processed

Each configurable node's bits are isolated within its allocated words. No two
nodes share a 32-bit word.

## Bit Layout

Configuration bits are packed starting from the LSB of each word:

```
Word N:   [31:bits_used] = 0 (tied low)
          [bits_used-1:0] = config data

Word N+1: [31:remaining] = 0 (tied low)
          [remaining-1:0] = overflow config data
```

For multi-word configurations, bits flow from LSB of word 0 to MSB of word 0,
then LSB of word 1, and so on.

Fields within a module's config_mem allocation are packed continuously
(LSB-first). The 32-bit word alignment applies to the entire module's total
config bits, not individual fields.

General field packing rule:

- Fields are packed in spec-defined field order, LSB-first.
- Array fields are packed by ascending index (index 0 in lower bits).
- Later fields occupy higher bit positions than earlier fields.

Examples:

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

config_mem is accessed via AXI4-Lite slave interface:

- Write: `cfg_awaddr[ADDR_WIDTH-1:2]` selects word index
- Read: `cfg_araddr[ADDR_WIDTH-1:2]` selects word index
- Single-cycle read latency (registered)
- Single-cycle write (combinational address phase, registered data)

### SystemC

config_mem is accessed via TLM 2.0 `simple_target_socket`:

- Blocking transport (`b_transport`)
- Word-addressed (address / 4 = word index)

## Configuration Sequence

Software configures the accelerator by:

1. Assert reset
2. Write all config_mem words via AXI-Lite (RTL) or TLM (SystemC)
3. De-assert reset
4. Begin dataflow execution

Partial reconfiguration (updating some nodes while others execute) is not
supported in the base design. Full reconfiguration requires reset.

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
