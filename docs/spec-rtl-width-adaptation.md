# RTL Width Adaptation Specification

This document is the **single source of truth** for all allowed width mismatches
in generated and prewritten RTL. Every width-mismatch relaxation in the Loom RTL
flow must reference one of the scenarios below by its unique identifier (WA-N).

## Governing Principle

Width mismatches in RTL are generally bad practice. The Fabric spec
(`docs/spec-fabric.md`) makes narrow, explicit exceptions for specific connection
patterns. This specification enumerates every such exception with precise rules.

**Strict by default, relaxed only with spec backing.**

- The Verilator/VCS lint check runs with maximum strictness for width warnings.
- No global `-Wno-WIDTHTRUNC` or `-Wno-WIDTHEXPAND` flags are permitted.
- Only the exact connection patterns listed below receive local lint waivers.
- Each waiver is scoped to the exact `assign` statement and annotated with the
  WA-identifier and a path back to this document.
- Any width mismatch NOT covered by a local waiver is a hard lint error.

## Adaptation Rules

All adaptations follow the Fabric spec's LSB-aligned rule:

- **Truncation** (wide to narrow): retain the lower `min(src, dst)` bits;
  discard upper bits. RTL idiom: `dst = src[dst_width-1:0];`
- **Zero-extension** (narrow to wide): pad upper bits with zero.
  RTL idiom: `dst = {{pad{1'b0}}, src};`
- **Tagged connections**: data and tag portions are adapted independently,
  each following the rules above.

---

## Allowed Scenarios

### WA-1: PE boundary data path

**Context**: PE exterior ports use `bits<N>` (structural width); FU operands
use native types (`i32`, `f32`, `none`, etc.) whose bit width may differ from N.

**Rules**:
- Input: truncate PE data bus to FU operand width (take lower bits).
- Output: zero-extend FU result to PE data bus width (pad upper bits with 0).
- NoneType (width 0) maps to 1-bit in SystemVerilog.

**Applies in**: `SVGenPE` spatial and temporal PE FU instantiation.

### WA-2: Inter-module data connection

**Context**: When modules with different DATA_WIDTH are connected (e.g., a
64-bit PE output to a 32-bit switch input), the connecting net may differ in
width from the destination port.

**Rules**:
- Truncation (wide to narrow): take lower `min(src, dst)` bits.
- Zero-extension (narrow to wide): pad upper bits with 0.

**Applies in**: `SVGenTop` module interconnect (non-tagged paths).

### WA-3: Tagged connection data/tag adaptation

**Context**: When two tagged ports have different data widths or different tag
widths, each portion is adapted independently.

**Rules**:
- Data bits: LSB-aligned truncation or zero-extension per WA-2 rules.
- Tag bits: LSB-aligned truncation or zero-extension per WA-2 rules.
- Tag-kind must match: tagged-to-tagged only.

**Applies in**: `SVGenTop` tagged module interconnect.

### WA-4: Config bit extraction

**Context**: Configuration words are 32 bits (`CONFIG_WORD_WIDTH` in
`fabric_pkg.sv`). Extracted fields (opcode, mux select, tag value, route
table entries, etc.) may be narrower.

**Rules**:
- Truncation via bit-select is always intentional.
- Each extraction site gets a local lint waiver scoped to that statement.

**Applies in**: Prewritten RTL modules under `src/rtl/design/`:
`fabric_add_tag`, `fabric_map_tag`, `fabric_spatial_pe`, `fabric_temporal_pe`,
`fabric_temporal_sw`, `fabric_spatial_sw`, `fabric_mux`, `fabric_fifo`,
`fabric_config_ctrl`, and any other module that unpacks config words.

### WA-5: Testbench width adaptation

**Context**: The testbench uses max-width wire arrays for uniform driver/monitor
infrastructure. DUT ports may be narrower or use different tag/data
decomposition than the max-width arrays.

**Rules**:
- Input adaptation: slice max-width driver array to DUT input port width.
- Output adaptation: assign DUT output bits into lower portion of max-width
  monitor array.
- For tagged ports: pack `{tag, data}` from separate max-width arrays into
  DUT input, and unpack DUT output into separate arrays.
- All adaptation assigns use explicit bit-select or concatenation.

**Applies in**: Generated `dut_inst.svh`.

---

## Lint Waiver Format

Every width adaptation in RTL must use this annotation pattern:

```systemverilog
// Fabric width adaptation (WA-N): Xb -> Yb
// See docs/spec-rtl-width-adaptation.md
/* verilator lint_off WIDTHTRUNC */   // or WIDTHEXPAND
assign dst = src[Y-1:0];             // explicit bit-select
/* verilator lint_on WIDTHTRUNC */    // or WIDTHEXPAND
```

The `WA-N` identifier, source/destination widths, and spec path must all be
present so any reviewer can trace the relaxation back to this document.

---

## Safety Invariants

1. **Mapper responsibility**: Tagged-width mismatch is only safe when runtime
   tag values remain representable at every tagged port on the routed path.
   This is enforced by the mapper (`canMapSoftwareTypeToHardware` in
   `TypeCompat.h`), not by SVGen.

2. **Uniform width per instance**: Switches, FIFOs, and memory modules use a
   single DATA_WIDTH/TAG_WIDTH for all their ports. Width adaptation happens
   at the boundary between different module types, not within a single
   module instance.

3. **No implicit adaptation**: Every width mismatch must use explicit RTL
   syntax (bit-select for truncation, concatenation for zero-extension).
   Verilog implicit truncation/extension is never relied upon.
