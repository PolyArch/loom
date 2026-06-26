# Fabric Mem

This document specifies `fabric.mem`, the leaf-level memory tile of the
fabric dialect. The canonical IR source is `Fabric_MemOp` in
`include/Fabric/IR/FabricOps.td`; verifier and parser/printer live in
`lib/Fabric/IR/FabricMemOp.cpp`.

## Identity

* Mnemonic: `mem`. C++ class: `::fabric::MemOp`.
* The op wraps `dataflow.load` / `dataflow.store` semantics into a
  fabric-domain leaf op. See the dataflow load/store specs for the
  semantic origin.
* Vector lane masks and their software semantics are specified in
  `docs/spec-dataflow-vectorization.md`. This document owns the Fabric
  projection from those masks to memory-port structure and byte enables.
* Variadic SSA operands and variadic SSA results (anonymous form) or a
  zero-operand / zero-result template (named form) whose port signature
  is captured in a `function_type` attribute.
* Carries a mandatory `Fabric_ScheduleAttr` predicate `[spatial]` or
  `[temporal]` (reused from `fabric.pe`).
* Optional `sym_name` so a memory tile can be referenced by
  `fabric.instantiate` (mirrors `fabric.pe`, `fabric.fu`,
  `fabric.switch`).
* The op has no body region; behavior is fully described by attributes.
* Allowed only inside a `fabric.module` body; the module body
  whitelist was extended to admit `fabric.mem`.

## Schedule predicate and port types

The schedule predicate selects the port type kind of every per-port
input and output of the op. The two cases are mutually exclusive.

| Schedule    | Per-port type              | Notes                                      |
|-------------|----------------------------|--------------------------------------------|
| `spatial`   | `!fabric.bits<W>`          | `ctrl`/`done` use `bits<0>`.               |
| `temporal`  | `!fabric.bits_tag<W, T>`   | `ctrl`/`done` use `bits_tag<0, T>`.        |

Spatial ports may not use `bits_tag`; temporal ports may not use
`bits`. The verifier emits a "schedule mismatch with port kind"
diagnostic on violation.

## Memref interfaces (M2.A model)

`fabric.mem` exposes two memref interfaces:

* `memref_mgr` (Manager-side, **required**): the first SSA operand. It
  is the memref this fabric.mem actively manages on its inward side --
  every internal load and store port goes through `memref_mgr`. The
  element type **must** be `!fabric.bits<W_mgr>`. `W_mgr` is derived
  from the element type and is the canonical bus-width of the
  internal load/store data ports.
* `memref_sub` (Subordinate-side, **optional**): when present, it is
  the **first SSA result** (detected by checking the first result's
  type for a memref). It exposes a memref to downstream consumers as
  an independent bypass. Its element type **must** be
  `!fabric.bits<W_sub>`. `W_sub` MAY differ from `W_mgr`.

The internal load/store ports are independent of `memref_sub`. The
M2.A decision is: load/store ports always operate against
`memref_mgr`; `memref_sub` is exposed inward as an independent bypass
for downstream consumers and does not affect port type rules.

The verifier requires `!fabric.bits<W>`-element memrefs only;
`memref<?xiN>` and other element types are rejected.

## Operand layout (anonymous form)

In order:

1. `memref_mgr : memref<?x!fabric.bits<W_mgr>>`
2. For each load port `i in [0, load_group_size)`:
   `(addr_i, ctrl_i)` -- two SSA operands.
3. For each store port `j in [0, store_group_size)`:
   `(addr_j, data_j, ctrl_j)` -- three SSA operands.

Total operand count: `1 + 2 * load_group_size + 3 * store_group_size`.

## Result layout (anonymous form)

In order:

1. Optional `memref_sub : memref<?x!fabric.bits<W_sub>>` (when present
   it is the first result; presence is detected from the result type
   alone; no syntactic marker is needed).
2. For each load port `i`: `(data_i, done_i)` -- two SSA results.
3. For each store port `j`: `done_j` -- one SSA result.

Total result count: `(0 or 1) + 2 * load_group_size + store_group_size`.

## Per-port type rules

Let `index_width = loom::getIndexWidth()` (existing helper at
`lib/Common/IndexWidth.{h,cpp}`).

For each load port:

| Operand/result | Spatial type           | Temporal type                |
|----------------|------------------------|------------------------------|
| `addr_i`       | `bits<index_width>`    | `bits_tag<index_width, T>`   |
| `ctrl_i`       | `bits<0>`              | `bits_tag<0, T>`             |
| `data_i`       | `bits<W_mgr>`          | `bits_tag<W_mgr, T>`         |
| `done_i`       | `bits<0>`              | `bits_tag<0, T>`             |

For each store port:

| Operand/result | Spatial type           | Temporal type                |
|----------------|------------------------|------------------------------|
| `addr_j`       | `bits<index_width>`    | `bits_tag<index_width, T>`   |
| `data_j`       | `bits<W_mgr>`          | `bits_tag<W_mgr, T>`         |
| `ctrl_j`       | `bits<0>`              | `bits_tag<0, T>`             |
| `done_j`       | `bits<0>`              | `bits_tag<0, T>`             |

`W_mgr` is the inferred element width of `memref_mgr`; `T` is
`tag_width` (temporal only).

## Hardware parameters

Hardware parameters live in `hw_params`, an ArrayAttr of length 1
wrapping a DictionaryAttr (the same `[ ... ]` convention used by
other fabric ops). `fabric.mem` requires `hw_params` to be present.

```
[{load_group_size = N : i32, store_group_size = M : i32}]                                 // spatial
[{load_group_size = N : i32, store_group_size = M : i32,
  tag_width = T : i32, addr_table_size = K : i32}]                                        // temporal
```

Constraints:

* `load_group_size >= 0`, `store_group_size >= 0`,
  `load_group_size + store_group_size >= 1`.
* Temporal: `tag_width >= 1`, `addr_table_size >= 1`.
* Spatial: `tag_width` and `addr_table_size` MUST be absent.

## Software configuration

Software parameters live in `addr_table` (ArrayAttr of DictionaryAttr)
and `mem_enable` (BoolAttr, the PE/clock/power-gating equivalent),
printed in `{ ... }`. They follow the fabric all-or-nothing rule:
both present (programmed) or both absent (hw-only).

`addr_table.base_addr` is a physical address or a physical-region
offset after mapping/runtime binding. It never encodes a virtual
address. Region identity and host pointer translation belong to
`fabric.system`, mapping artifacts, runtime descriptors, or platform
adapters, not to `fabric.mem`.

### Spatial entry shape

One entry per port, total `load_group_size + store_group_size`. Entry
indices `[0, load_group_size)` correspond to load ports
`0..load_group_size-1`; entry indices
`[load_group_size, load_group_size + store_group_size)` correspond to
store ports `0..store_group_size-1`.

```
{base_addr = ... : iN_addr, element_log2_size = ... : i4, valid = true}
```

Spatial entries MUST NOT carry the `tag` key.

### Temporal entry shape (CAM lookup by tag)

Exactly `addr_table_size` entries:

```
{base_addr = ... : iN_addr, element_log2_size = ... : i4,
 tag = ... : iT, valid = true}
```

Among entries with `valid == true`, all `tag` values must be
distinct.

### Per-entry verifier rules

* `base_addr` is an IntegerAttr whose integer width equals the
  resolved `loom_addr_bits` (see "Global constants" below).
* `element_log2_size` is an IntegerAttr of width 4 (`i4`). It is
  treated as an unsigned 4-bit value. Its value MUST satisfy
  `element_log2_size <= log2(loom_mem_bus_width / 8)` against the
  resolved `loom_mem_bus_width`.
* `valid` is a BoolAttr.
* Temporal: `tag` is an IntegerAttr whose integer width equals
  `tag_width`. `tag` is interpreted as an unsigned `tag_width`-bit
  value.

The 4-bit `element_log2_size` field encodes values 0..15. The maximum
valid value is derived from the resolved `loom_mem_bus_width`.
Module-level overrides may further reduce this maximum (see below).

## Global constants and per-module overrides

`docs/spec-config-ssot.md` owns the resolved global values for Loom
address width and memory bus width. Compatibility inputs such as
environment variables may feed the configuration resolver, but they are
not independent hidden defaults. They must be recorded as explicit
configuration provenance when used.

`fabric.module` carries two new optional attributes that override
these resolved values for ops nested inside the module body:

```
fabric.module @top(...) attributes {
  loom_addr_bits = 32 : i32,
  loom_mem_bus_width = 1024 : i32
} { ... }
```

The verifier uses helpers

```cpp
namespace fabric {
unsigned resolveLoomAddrBits(Operation *op);       // walks to enclosing fabric.module
unsigned resolveLoomMemBusWidth(Operation *op);
}
```

to read the per-module override (if any) or fall back to the resolved
configuration values.

## Custom assembly format

Anonymous form (readable named-group syntax):

```mlir
%sub_or_void, %d0, %dn0, %d1, %dn1, %sd0 = fabric.mem [spatial] mgr(%mgr)
    load(%la0, %lc0, %la1, %lc1)
    store(%sa0, %sd0_data, %sc0)
    [{load_group_size = 2 : i32, store_group_size = 1 : i32}]
    {addr_table = [
       {base_addr = 65536  : i48, element_log2_size = 2 : i4, valid = true},
       {base_addr = 65792  : i48, element_log2_size = 2 : i4, valid = true},
       {base_addr = 131072 : i48, element_log2_size = 2 : i4, valid = true}
     ], mem_enable = true}
    : (memref<?x!fabric.bits<32>>,
       !fabric.bits<32>, !fabric.bits<0>,
       !fabric.bits<32>, !fabric.bits<0>,
       !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
    -> (memref<?x!fabric.bits<32>>,
        !fabric.bits<32>, !fabric.bits<0>,
        !fabric.bits<32>, !fabric.bits<0>,
        !fabric.bits<0>)
```

* `mgr(%mgr)` is required. The actual memref operand appears here.
* `load(...)` and `store(...)` appear in this fixed order. Both are
  optional (`load_group_size` or `store_group_size` may be `0`). The
  per-port operands are flat in a single comma-separated list inside
  each clause; each load port contributes 2 operands `(addr, ctrl)`
  and each store port contributes 3 operands `(addr, data, ctrl)`.
* `[ ... ]` carries `hw_params`. `{ ... }` carries the
  `addr_table` / `mem_enable` pair.
* The trailing functional-type signature lists the operand types
  followed by `->` and the result types. The optional `memref_sub`
  appears in the result-type list directly; the verifier detects it
  from the first result's type (memref vs bits/bits_tag).

Named template form (declaration only):

```mlir
fabric.mem @MyMem [spatial]
       (memref<?x!fabric.bits<32>>,
        !fabric.bits<32>, !fabric.bits<0>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
       -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>)
       [{load_group_size = 1 : i32, store_group_size = 1 : i32}]
       {addr_table = [
           {base_addr = 0    : i48, element_log2_size = 2 : i4, valid = true},
           {base_addr = 4096 : i48, element_log2_size = 2 : i4, valid = true}
         ], mem_enable = true}
```

The named form has zero SSA operands and zero SSA results in the
enclosing `fabric.module` body; the port signature is captured in a
`function_type` attribute. Actual usage of a named `fabric.mem` goes
through `fabric.instantiate`.

## Verifier checklist

* Schedule branch correctness:
  * Spatial: per-port operands/results use `bits<W>`; addr port uses
    `bits<index_width>`; ctrl/done use `bits<0>`; data uses
    `bits<W_mgr>`.
  * Temporal: per-port operands/results use `bits_tag<W, T>`; addr
    uses `bits_tag<index_width, T>`; ctrl/done use `bits_tag<0, T>`;
    data uses `bits_tag<W_mgr, T>`.
* `memref_mgr` element type is `!fabric.bits<W_mgr>` for some
  `W_mgr`.
* Optional `memref_sub` (first result if it is a memref) element type
  is `!fabric.bits<W_sub>`. `W_sub` is independent of `W_mgr`.
* Operand count equals `1 + 2*load_group_size + 3*store_group_size`.
* Result count equals
  `(has_sub ? 1 : 0) + 2*load_group_size + store_group_size`.
* `load_group_size + store_group_size >= 1`.
* Temporal-only attrs (`tag_width`, `addr_table_size`) are absent on
  spatial; required on temporal.
* All-or-nothing on `(addr_table, mem_enable)`.
* Programmed (spatial): `addr_table` length =
  `load_group_size + store_group_size`; entries MUST NOT carry
  `tag`.
* Programmed (temporal): `addr_table` length = `addr_table_size`;
  among `valid = true` entries, `tag` values are distinct; `tag`
  width = `T`.
* Each entry: `base_addr` width = `loom_addr_bits` (resolved);
  `element_log2_size` is 4-bit and `<= log2(loom_mem_bus_width / 8)`;
  `valid` is BoolAttr.
* Named template form: 0 SSA operands AND 0 SSA results;
  `function_type` required.
* Anonymous form: `function_type` absent.

## Negative tests

The verifier exercises the following rejections (see
`test/fabric/unit/mem/invalid.mlir`):

* Bad schedule keyword.
* `load_group_size + store_group_size` both zero.
* `tag_width` present in spatial.
* `tag_width` or `addr_table_size` missing in temporal.
* `memref_mgr` element type not `!fabric.bits<W>`.
* `memref_sub` element type not `!fabric.bits<W_sub>`.
* Per-port operand/result type does not match the schedule's expected
  `addr` / `ctrl` / `data` shape.
* Store data port width does not equal `memref_mgr` element width.
* `addr_table` length does not match the expected per-schedule
  count.
* `element_log2_size` exceeds `log2(loom_mem_bus_width / 8)`.
* Temporal duplicate `tag` value among `valid = true` entries.
* All-or-nothing violation (`addr_table` without `mem_enable` or vice
  versa).

## Cross-references

* `spec-fabric-module.md` -- SpatialCore/CGRA template container and the body
  whitelist.
* `spec-fabric-pe.md` -- schedule predicate `[spatial] | [temporal]`.
* `spec-fabric-pe-temporal.md` -- temporal-domain background and
  tag-stream handshake.
* `spec-fabric-switch.md` -- sibling leaf op with the same
  schedule-predicate / hw-params / sw-configs / function-type
  pattern.
* `spec-fabric-instantiate.md` -- instantiation of a named
  `fabric.mem` template into a `fabric.module` body.

## Maintenance

Implementation locations that must mirror this spec are:

* `Fabric_MemOp` in `include/Fabric/IR/FabricOps.td` for the IR shape;
* `MemOp::parse`, `MemOp::print`, and `MemOp::verify` in
  `lib/Fabric/IR/FabricMemOp.cpp` for parser, printer, and verifier
  logic;
* `loom::getDefaultLoomAddrBits` /
  `loom::getDefaultLoomMemBusWidth` in
  the configuration SSOT implementation for the resolved global values;
* `fabric::resolveLoomAddrBits` /
  `fabric::resolveLoomMemBusWidth` in
  `lib/Fabric/IR/FabricMemOp.cpp` for the per-module override
  walker.
