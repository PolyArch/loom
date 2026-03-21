# FCC RTL Generation Constraints Specification

## Overview

This document is the normative specification for what `--gen-sv` can and
cannot produce. It defines the supported operation set, parameter ranges,
latency and interval rules, module-level constraints, config interface
requirements, synthesis targeting rules, and the complete error reporting
contract.

All constraints below apply to the C++ `SVGen` library that converts
fabric MLIR to self-contained synthesizable SystemVerilog collateral.

Related documents:

- [spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md)
- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md)
- [spec-runtime-mmio.md](./spec-runtime-mmio.md)

## Tier Classification

RTL generation classifies every FU body operation into one of three tiers:

- **Tier 1** -- Integer, logic, and bitwise operations. Fully synthesizable
  from behavioral SystemVerilog with no vendor IP dependency. Included in the
  first generation milestone.

- **Tier 2** -- Basic floating-point operations. Synthesizable using behavioral
  models for simulation. Synthesis uses `ifdef SYNTH_FP_IP` guards for vendor
  IP instantiation (e.g. Synopsys DesignWare). Included in the second
  generation milestone.

- **Tier 3** -- Transcendental floating-point operations. No portable
  synthesizable implementation exists. Generation is rejected unless the user
  provides an explicit `--fp-ip-profile` flag selecting a vendor IP library.

## Supported Operations by Dialect

### `arith` Dialect

#### Tier 1 -- Integer and Logic (Combinational)

These operations have zero intrinsic latency unless stated otherwise.

| Operation | RTL Module | Intrinsic Latency | Notes |
|-----------|-----------|-------------------|-------|
| `arith.addi` | `fu_op_addi.sv` | 0 | |
| `arith.subi` | `fu_op_subi.sv` | 0 | |
| `arith.andi` | `fu_op_andi.sv` | 0 | |
| `arith.ori` | `fu_op_ori.sv` | 0 | |
| `arith.xori` | `fu_op_xori.sv` | 0 | |
| `arith.shli` | `fu_op_shli.sv` | 0 | |
| `arith.shrsi` | `fu_op_shrsi.sv` | 0 | |
| `arith.shrui` | `fu_op_shrui.sv` | 0 | |
| `arith.cmpi` | `fu_op_cmpi.sv` | 0 | 4-bit predicate config |
| `arith.extsi` | `fu_op_extsi.sv` | 0 | |
| `arith.extui` | `fu_op_extui.sv` | 0 | |
| `arith.trunci` | `fu_op_trunci.sv` | 0 | |
| `arith.select` | `fu_op_select.sv` | 0 | |
| `arith.index_cast` | `fu_op_index_cast.sv` | 0 | |
| `arith.index_castui` | `fu_op_index_castui.sv` | 0 | |

#### Tier 1 -- Integer (Multi-Cycle)

These operations have non-zero intrinsic latency. The exact intrinsic latency
is implementation-dependent and parameterized by data width.

| Operation | RTL Module | Intrinsic Latency | Notes |
|-----------|-----------|-------------------|-------|
| `arith.muli` | `fu_op_muli.sv` | >= 1 | Width-dependent |
| `arith.divsi` | `fu_op_divsi.sv` | >= 1 | Width-dependent |
| `arith.divui` | `fu_op_divui.sv` | >= 1 | Width-dependent |
| `arith.remsi` | `fu_op_remsi.sv` | >= 1 | Width-dependent |
| `arith.remui` | `fu_op_remui.sv` | >= 1 | Width-dependent |

#### Tier 2 -- Floating-Point

| Operation | RTL Module | Intrinsic Latency | Notes |
|-----------|-----------|-------------------|-------|
| `arith.addf` | `fu_op_addf.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `arith.subf` | `fu_op_subf.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `arith.mulf` | `fu_op_mulf.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `arith.divf` | `fu_op_divf.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `arith.cmpf` | `fu_op_cmpf.sv` | 0 | 4-bit predicate config |
| `arith.negf` | `fu_op_negf.sv` | 0 | Sign-bit inversion |
| `arith.fptosi` | `fu_op_fptosi.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `arith.fptoui` | `fu_op_fptoui.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `arith.sitofp` | `fu_op_sitofp.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `arith.uitofp` | `fu_op_uitofp.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |

### `math` Dialect

#### Tier 2 -- Basic FP Math

| Operation | RTL Module | Intrinsic Latency | Notes |
|-----------|-----------|-------------------|-------|
| `math.absf` | `fu_op_absf.sv` | 0 | Sign-bit clear |
| `math.fma` | `fu_op_fma.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |
| `math.sqrt` | `fu_op_sqrt.sv` | >= 1 | Behavioral + `ifdef SYNTH_FP_IP` |

#### Tier 3 -- Transcendental FP (Rejected Without IP Profile)

| Operation | RTL Module | Intrinsic Latency | Notes |
|-----------|-----------|-------------------|-------|
| `math.cos` | `fu_op_cos.sv` | N/A | Requires `--fp-ip-profile` |
| `math.sin` | `fu_op_sin.sv` | N/A | Requires `--fp-ip-profile` |
| `math.exp` | `fu_op_exp.sv` | N/A | Requires `--fp-ip-profile` |
| `math.log2` | `fu_op_log2.sv` | N/A | Requires `--fp-ip-profile` |

Generation rejects any FU body containing a Tier 3 operation unless the
`--fp-ip-profile` flag is provided. See Error Reporting for the exact error
message format.

### `llvm` Dialect

#### Tier 1

| Operation | RTL Module | Intrinsic Latency | Notes |
|-----------|-----------|-------------------|-------|
| `llvm.intr.bitreverse` | `fu_op_bitreverse.sv` | 0 | Pure wiring |

No other `llvm.*` operation is supported for RTL generation.

### `dataflow` Dialect

All dataflow operations are dedicated fixed-behavior state-machine FUs.
They require `latency = -1` and `interval = -1`. They do not use the
pipeline wrapper or interval throttle.

| Operation | RTL Module | Timing Class |
|-----------|-----------|-------------|
| `dataflow.stream` | `fu_op_stream.sv` | Dataflow state machine |
| `dataflow.gate` | `fu_op_gate.sv` | Dataflow state machine |
| `dataflow.carry` | `fu_op_carry.sv` | Dataflow state machine |
| `dataflow.invariant` | `fu_op_invariant.sv` | Dataflow state machine |

Dataflow operations must each occupy an exclusive FU body. They must not be
mixed with any other non-terminator operation.

### `handshake` Dialect

All handshake operations belong to the single-fire single-result-set timing
class and use `latency >= 0` and `interval >= 1`.

| Operation | RTL Module | Intrinsic Latency | Notes |
|-----------|-----------|-------------------|-------|
| `handshake.cond_br` | `fu_op_cond_br.sv` | 0 | |
| `handshake.constant` | `fu_op_constant.sv` | 0 | Config: literal value |
| `handshake.join` | `fu_op_join.sv` | 0 | Config: join_mask |
| `handshake.load` | `fu_op_load.sv` | 0 | |
| `handshake.store` | `fu_op_store.sv` | 0 | |
| `handshake.mux` | `fu_op_mux.sv` | 0 | Runtime selector operand |

### `fabric` Dialect (FU-Internal)

| Operation | RTL Module | Notes |
|-----------|-----------|-------|
| `fabric.mux` | `fabric_mux.sv` | Configuration-time structural selector |
| `fabric.yield` | N/A | Terminator; no RTL module |

No other `fabric.*` operation is allowed inside an FU body.

### Unsupported Operations

Any operation not listed above is unsupported for RTL generation. If an FU
body contains an unsupported operation, generation is rejected. This includes
but is not limited to:

- `func.*`, `cf.*`, `scf.*`, `affine.*` control-flow operations
- `arith.constant` (use `handshake.constant` instead)
- `handshake.sink` (unused outputs handled by PE-side discard)
- Nested `fabric.spatial_pe`, `fabric.temporal_pe`, `fabric.spatial_sw`,
  `fabric.temporal_sw`, `fabric.memory`, `fabric.extmemory`, `fabric.fifo`,
  `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag`

## Latency and Interval Constraints

### Intrinsic Latency Model

Every supported operation has an intrinsic latency that represents the minimum
number of cycles from input acceptance to result availability inside the FU
body itself (before any retiming added by the PE slot wrapper).

The three-layer separation for latency handling is:

1. **FU body** -- owns intrinsic compute behavior
2. **Slot wrapper** -- adds extra retiming registers and interval throttling
3. **PE container** -- owns mandatory FU-local output registers and
   round-robin arbitration (temporal PE only)

### Latency Rules

For FUs with single-fire single-result-set behavior:

- `latency >= 0` is required
- `latency >= intrinsic_latency` is legal; the slot wrapper adds
  `(latency - intrinsic_latency)` retiming shift-register stages after the
  FU output
- `latency == 0` means the FU output is combinational (no extra registers
  in the slot wrapper); this is only legal when `intrinsic_latency == 0`
- `latency < intrinsic_latency` when `intrinsic_latency > 0` is a
  specification violation and generation is rejected

For FUs with dedicated dataflow state-machine behavior:

- `latency == -1` is required
- Any other value is a specification violation and generation is rejected

### Interval Rules

For FUs with single-fire single-result-set behavior:

- `interval >= 1` is required
- `interval == 1` means fully pipelined; no throttle counter is instantiated
- `interval > 1` means the slot wrapper adds a countdown counter that blocks
  re-firing until the interval has elapsed

For FUs with dedicated dataflow state-machine behavior:

- `interval == -1` is required
- Any other value is a specification violation and generation is rejected

### Latency/Interval Interaction

The refire condition in a `temporal_pe` is the conjunction of:

- the FU's intrinsic `interval` constraint (countdown counter)
- the temporal scheduler selecting that FU
- all required operands being ready
- the FU not being busy (all FU-local output registers drained)

Observable egress time from a `temporal_pe` is:

- FU fire time + FU-local `latency` + arbitration wait in FU-local output
  register

### Compound FU Body Latency

For compound FU bodies containing multiple operations, the intrinsic latency
of the whole body is the critical-path latency through the internal DAG.

The declared `latency` must be greater than or equal to this critical-path
intrinsic latency.

## Module-Level Constraints

### Switch Port Limits

| Parameter | Constraint | Notes |
|-----------|-----------|-------|
| `spatial_sw` input count | 1..32 | Defined in `fabric_pkg.sv` `MAX_SWITCH_PORTS` |
| `spatial_sw` output count | 1..32 | |
| `temporal_sw` input count | 1..32 | |
| `temporal_sw` output count | 1..32 | |

Port counts outside this range cause a generation error.

### Tag Width

| Parameter | Constraint |
|-----------|-----------|
| Minimum tag width | 1 bit |
| Maximum tag width | 16 bits (`MAX_TAG_WIDTH` in `fabric_pkg.sv`) |

Tag width of zero means the port is non-tagged. Non-tagged ports do not
carry a tag field. Tag widths above 16 are rejected.

### Data Width

| Parameter | Constraint |
|-----------|-----------|
| Minimum data width | 1 bit |
| Maximum data width | 4096 bits |

These limits apply to `!fabric.bits<N>` port widths on switches, PEs, FIFOs,
tag operations, and memory interfaces.

### FIFO Parameters

| Parameter | Constraint |
|-----------|-----------|
| `depth` | >= 1 |
| `bypassable` | Optional; when present, adds 1 config bit |

A FIFO with `depth < 1` is rejected.

### PE Parameters

| Parameter | Constraint |
|-----------|-----------|
| Spatial PE FU count | >= 1 |
| Temporal PE FU count | >= 1 |
| Temporal PE `num_instruction` | >= 1 |
| Temporal PE `num_register` | >= 0 |
| Temporal PE `reg_fifo_depth` | >= 1 when `num_register > 0` |
| `max_fu_inputs` | >= 1 |
| `max_fu_outputs` | >= 1 |
| `handshake.join` fan-in | 1..64 |

### Memory Parameters

| Parameter | Constraint |
|-----------|-----------|
| `ldCount` | >= 0 |
| `stCount` | >= 0 |
| `ldCount + stCount` | >= 1 |
| `numRegion` | >= 1 |
| `lsqDepth` | >= 1 |
| `tagWidth` for multi-port | >= `ceil(log2(max(ldCount, stCount)))` |

### Spatial Switch Decomposition

| Parameter | Constraint |
|-----------|-----------|
| `decomposable_bits` | 0 (disabled) or positive divisor of all port widths |
| Tagged + decomposable | Not allowed |

When `decomposable_bits > 0`, every port width must be evenly divisible by
`decomposable_bits`. Tagged `spatial_sw` must not be decomposable.

### Connectivity Table

When a `connectivity_table` is present on a switch:

- It must have dimensions `[num_outputs][num_inputs]`
- Each entry is a single bit (0 = not connected, 1 = connected)
- At least one connection must exist per output

When omitted, full connectivity is assumed.

## Config Interface Constraints

### Word Width

The configuration interface uses a fixed 32-bit word width.

- `cfg_wdata` is always 32 bits wide
- This matches the `word_width_bits` in `config.json`
- No other word width is supported

### Word-Serial Protocol

Configuration loading uses a word-serial protocol with `valid`/`ready`/`last`
handshake signals:

- One 32-bit word is transferred per handshake cycle
- The `fabric_config_ctrl` module auto-increments an internal address counter
- Configuration loading occurs only during reset or explicit quiescent mode
- Loading during active dataflow execution is not supported

### Slice Alignment

Every configuration slice is word-aligned:

- Two different slices never share a 32-bit word
- A slice occupies one or more complete 32-bit words
- Unused high bits in the final word of a slice are zero-padded

### Intra-Slice Packing

Inside one slice, fields are packed low-to-high (LSB-first):

- Earlier fields occupy lower bit positions
- Later fields occupy higher bit positions
- A field may straddle a 32-bit word boundary within the same slice
- Array-like fields use ascending logical index order

### Slice Ordering

The RTL config controller routes words to per-module config ports based on a
slice offset table. The ordering follows the FCC `ConfigGen` convention:

1. Primitive and storage slices in flattened node order: `spatial_sw`,
   `temporal_sw`, `add_tag`, `map_tag`, bypassable `fifo`, `memory`,
   `extmemory`
2. PE container slices in PE-containment order: `spatial_pe`, `temporal_pe`

### Config Layout Consistency

The RTL config register unpacking must exactly match the packing rules defined
in `spec-fabric-config_mem.md` and implemented by `ConfigGenConfig.cpp`. The
`SVGenConfigLayout` module mirrors these field-width computation functions.

Drift between SVGen and ConfigGen config layouts is verified by config-decode
equivalence tests.

### Choice Width Rule

Selection fields use this encoding width:

- 0 bits when the number of alternatives N <= 1
- `ceil(log2(N))` bits when N > 1

This applies to: opcode fields, input-mux `sel`, output-demux `sel`, and
register index fields.

## Synthesis Constraints

### Target Technology

RTL generation targets ASIC synthesis using:

- **Synthesis tool**: Synopsys Design Compiler (DC)
- **Standard cell library**: saed32 (SAED 32nm EDK), specifically
  `saed32rvt_dlvl_ff0p85v25c`
- **Library path**: `/mnt/nas0/eda.libs/saed32/EDK_08_2025`

### Structural Rules

The generated RTL must adhere to these synthesizable-subset rules:

- **No latches**: All storage elements must be edge-triggered flip-flops. No
  inferred latches are permitted. Synthesis warnings about inferred latches
  are treated as errors.

- **No tri-state**: No `tri`, `wand`, `wor`, or `trireg` net types. No
  `bufif0`/`bufif1`/`notif0`/`notif1` primitives. All outputs must be
  actively driven.

- **No initial blocks**: `initial` blocks are permitted only in testbench
  code, never in design modules.

- **No delay annotations**: `#delay` constructs are permitted only in
  testbench code.

- **Synchronous resets**: All reset paths must be synchronous
  (or explicitly gated by clock). Asynchronous reset is not used in the
  generated RTL.

- **Single clock domain**: All generated modules operate in a single clock
  domain.

- **No black-box IP without profile**: Tier 2 FP operations use behavioral
  models by default. Vendor IP instantiation requires `ifdef SYNTH_FP_IP`.
  Tier 3 operations are not generated without `--fp-ip-profile`.

### Lint Requirements

All generated SV must pass `verilator --lint-only` without errors. Lint
warnings in the synthesizable subset are treated as generation defects.

### Named Begin-End Blocks

Every `begin`-`end` block in generated SV must have a named label
(`: label_name`). This is a project-wide SystemVerilog style requirement.

### Loop Variable Declaration

Loop variables must be declared at the top of the enclosing procedural block
(`always`, `initial`, `function`), not inline in `for` statements. Loop
variables use the naming convention `iter_var0`, `iter_var1`, etc., numbered
by nesting depth.

## Error Reporting

### Error Message Format

All generation errors use the format:

```
gen-sv error: <category>: <message>
```

The `<category>` identifies the class of constraint violation. The `<message>`
provides specific details including the offending operation, parameter values,
and the applicable constraint.

### Error Categories and Conditions

#### Category: `unsupported-op`

Reported when an FU body contains an operation not in the supported tier set.

| Condition | Message |
|-----------|---------|
| Unlisted operation in FU body | `gen-sv error: unsupported-op: operation '<op_name>' is not supported for RTL generation` |
| Tier 3 FP op without profile | `gen-sv error: unsupported-op: transcendental FP op '<op_name>' requires --fp-ip-profile; no portable synthesizable implementation available` |

#### Category: `latency`

Reported when the declared FU latency violates the intrinsic latency
constraint.

| Condition | Message |
|-----------|---------|
| Latency below intrinsic | `gen-sv error: latency: FU '<fu_name>' declares latency=<N> but intrinsic latency is <M>; latency must be >= intrinsic` |
| Non-negative latency on dataflow FU | `gen-sv error: latency: FU '<fu_name>' contains dataflow op '<op_name>' but declares latency=<N>; dataflow FUs require latency=-1` |
| Negative latency on non-dataflow FU | `gen-sv error: latency: FU '<fu_name>' declares latency=-1 but contains no dataflow op; only dataflow FUs may use latency=-1` |

#### Category: `interval`

Reported when the declared FU interval violates constraints.

| Condition | Message |
|-----------|---------|
| Interval < 1 on non-dataflow FU | `gen-sv error: interval: FU '<fu_name>' declares interval=<N>; non-dataflow FUs require interval >= 1` |
| Non-negative interval on dataflow FU | `gen-sv error: interval: FU '<fu_name>' contains dataflow op '<op_name>' but declares interval=<N>; dataflow FUs require interval=-1` |
| Negative interval on non-dataflow FU | `gen-sv error: interval: FU '<fu_name>' declares interval=-1 but contains no dataflow op; only dataflow FUs may use interval=-1` |

#### Category: `timing-class`

Reported when the FU body timing classification is inconsistent.

| Condition | Message |
|-----------|---------|
| Dataflow op mixed with other ops | `gen-sv error: timing-class: FU '<fu_name>' mixes dataflow op '<op_name>' with other non-terminator ops; dataflow ops must be body-exclusive` |

#### Category: `param-range`

Reported when a hardware parameter is outside the supported range.

| Condition | Message |
|-----------|---------|
| Switch port count out of range | `gen-sv error: param-range: '<module_name>' has <N> input ports; supported range is 1..32` |
| Switch port count out of range | `gen-sv error: param-range: '<module_name>' has <N> output ports; supported range is 1..32` |
| Tag width out of range | `gen-sv error: param-range: '<module_name>' has tag_width=<N>; supported range is 1..16` |
| Data width out of range | `gen-sv error: param-range: '<module_name>' has data_width=<N>; supported range is 1..4096` |
| FIFO depth < 1 | `gen-sv error: param-range: FIFO '<name>' has depth=<N>; minimum depth is 1` |
| Join fan-in out of range | `gen-sv error: param-range: handshake.join in FU '<fu_name>' has <N> operands; supported range is 1..64` |
| Zero FU count in PE | `gen-sv error: param-range: PE '<pe_name>' contains 0 function units; minimum is 1` |
| Memory ldCount+stCount < 1 | `gen-sv error: param-range: memory '<name>' has ldCount=<L> stCount=<S>; at least one must be positive` |
| Memory tag width insufficient | `gen-sv error: param-range: memory '<name>' has tag_width=<T> but needs >= <M> for max(ldCount,stCount)=<N>` |

#### Category: `decomposition`

Reported when decomposable switch parameters are invalid.

| Condition | Message |
|-----------|---------|
| Port width not divisible | `gen-sv error: decomposition: '<module_name>' port width <W> is not divisible by decomposable_bits=<D>` |
| Tagged and decomposable | `gen-sv error: decomposition: '<module_name>' is both tagged and decomposable; this combination is not supported` |

#### Category: `config-layout`

Reported when the config layout computation encounters an inconsistency.

| Condition | Message |
|-----------|---------|
| Slice overflow | `gen-sv error: config-layout: slice for '<module_name>' exceeds maximum config address space` |
| Layout mismatch detected | `gen-sv error: config-layout: computed layout for '<module_name>' does not match ConfigGen reference` |

#### Category: `fu-body`

Reported when FU body structural rules are violated.

| Condition | Message |
|-----------|---------|
| Empty FU body | `gen-sv error: fu-body: FU '<fu_name>' has no non-terminator operations` |
| Passthrough detected | `gen-sv error: fu-body: FU '<fu_name>' yield operand is a direct block argument; passthrough FUs are illegal` |
| Unused block argument | `gen-sv error: fu-body: FU '<fu_name>' block argument <idx> is not consumed by any body operation` |
| Prohibited nested op | `gen-sv error: fu-body: FU '<fu_name>' contains prohibited operation '<op_name>'; container/routing/tag ops are not allowed inside FU bodies` |

#### Category: `connectivity`

Reported when connectivity table dimensions are inconsistent.

| Condition | Message |
|-----------|---------|
| Dimension mismatch | `gen-sv error: connectivity: '<module_name>' connectivity_table has dimensions [<R>][<C>] but switch has <O> outputs and <I> inputs` |
| No connections for output | `gen-sv error: connectivity: '<module_name>' output <idx> has no connected inputs in connectivity_table` |

### Error Behavior

When any generation error is reported:

- The error message is printed to stderr
- Generation halts after collecting all errors for the current module
  (errors are not immediately fatal; all independent checks run first)
- The exit code is non-zero
- No partial RTL output is written to the output directory

Multiple errors may be reported in a single generation run. Each error
appears on its own line with the `gen-sv error:` prefix.

## Output Collateral

When generation succeeds, the output directory contains:

| Path | Content |
|------|---------|
| `<outdir>/rtl/*.sv` | All generated and copied SystemVerilog files |
| `<outdir>/rtl/filelist.f` | Compilation-order file list |

The `filelist.f` lists files in dependency order:

1. Packages (`fabric_pkg.sv`)
2. Interfaces (`fabric_handshake_if.sv`, `fabric_cfg_if.sv`)
3. Common primitives (`fabric_rr_arbiter.sv`, `fabric_fifo_mem.sv`, etc.)
4. Leaf modules (FU ops, tag ops, FIFOs, switches)
5. PE containers
6. Top module (`fabric_top.sv`)

## Limitations and Future Work

The following are known limitations of the current RTL generation:

- Tier 3 transcendental FP operations are not generated without
  `--fp-ip-profile`. No default IP profile is bundled.

- Multi-clock-domain designs are not supported. All modules share one
  clock and one synchronous reset.

- Generated RTL targets ASIC synthesis only. FPGA-specific primitives
  (block RAMs, DSP slices, carry chains) are not instantiated. FPGA
  targeting may work via inference but is not validated.

- The maximum tested design size is bounded by the EDA tool capacity,
  not by the generator itself. Very large fabrics may encounter
  synthesis runtime or memory limits.

- AXI-MM interface parameters (address width, data width, ID width) are
  derived from the fabric MLIR and must be consistent across all
  extmemory modules in one design.

- The config controller assumes a single config chain. Hierarchical or
  parallel config distribution is not currently supported.

## Related Documents

- [spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md)
- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-runtime-mmio.md](./spec-runtime-mmio.md)
