# Fabric PE Allowed Operations Specification

## Overview

This document is the **single source of truth** for operations allowed inside
`fabric.pe` bodies. Both [spec-fabric-pe.md](./spec-fabric-pe.md) and
[spec-adg-api.md](./spec-adg-api.md) reference this document.

## Allowed Dialects

`fabric.pe` bodies may include operations from the following dialects:

- `arith`
- `math`
- `dataflow` (restricted subset)
- `handshake` (restricted subset)
- `fabric.pe` (nested)
- `fabric.instance` (to instantiate named PEs)

## arith Dialect (30 operations)

| Operation | Description |
|-----------|-------------|
| `arith.addf` | Floating-point addition |
| `arith.addi` | Integer addition |
| `arith.andi` | Bitwise AND |
| `arith.cmpf` | Floating-point comparison |
| `arith.cmpi` | Integer comparison |
| `arith.divf` | Floating-point division |
| `arith.divsi` | Signed integer division |
| `arith.divui` | Unsigned integer division |
| `arith.extsi` | Signed integer extension |
| `arith.extui` | Unsigned integer extension |
| `arith.fptosi` | Floating-point to signed integer |
| `arith.fptoui` | Floating-point to unsigned integer |
| `arith.index_cast` | Index to integer cast |
| `arith.index_castui` | Index to unsigned integer cast |
| `arith.mulf` | Floating-point multiplication |
| `arith.muli` | Integer multiplication |
| `arith.negf` | Floating-point negation |
| `arith.ori` | Bitwise OR |
| `arith.remsi` | Signed integer remainder |
| `arith.remui` | Unsigned integer remainder |
| `arith.select` | Conditional select |
| `arith.shli` | Shift left |
| `arith.shrsi` | Arithmetic shift right |
| `arith.shrui` | Logical shift right |
| `arith.subf` | Floating-point subtraction |
| `arith.subi` | Integer subtraction |
| `arith.trunci` | Integer truncation |
| `arith.sitofp` | Signed integer to floating-point |
| `arith.uitofp` | Unsigned integer to floating-point |
| `arith.xori` | Bitwise XOR |

## math Dialect (7 operations)

| Operation | Description |
|-----------|-------------|
| `math.absf` | Absolute value (float) |
| `math.cos` | Cosine |
| `math.exp` | Exponential |
| `math.fma` | Fused multiply-add |
| `math.log2` | Logarithm base 2 |
| `math.sin` | Sine |
| `math.sqrt` | Square root |

## dataflow Dialect (4 operations)

| Operation | Description |
|-----------|-------------|
| `dataflow.carry` | Loop-carried dependency |
| `dataflow.invariant` | Loop-invariant value |
| `dataflow.stream` | Index stream generator |
| `dataflow.gate` | Stream alignment adapter |

**Constraint:** A `fabric.pe` body must contain exactly **one** `dataflow`
operation and the `fabric.yield` terminator. Multiple `dataflow` operations in
a single PE, mixing `dataflow` with other dialects, and nesting via
`fabric.instance` are all not allowed. Each dataflow operation maps to a
dedicated hardware state machine and cannot share a PE with other operations.

## handshake Dialect (8 operations)

| Operation | Description |
|-----------|-------------|
| `handshake.cond_br` | Conditional branch |
| `handshake.constant` | Constant value (runtime configurable) |
| `handshake.fork` | Data fork (one input to multiple outputs) |
| `handshake.join` | Data join (multiple inputs to one output) |
| `handshake.load` | Memory load |
| `handshake.mux` | Multiplexer |
| `handshake.sink` | Data sink (consumes and discards) |
| `handshake.store` | Memory store |

All other `handshake` operations are **not allowed** inside `fabric.pe`.

The `handshake` dialect is defined by CIRCT, not by Loom. For detailed
semantics of these operations, see
[externals/circt/docs/Dialects/Handshake/RationaleHandshake.md](../externals/circt/docs/Dialects/Handshake/RationaleHandshake.md).

## Exclusivity Rules

Certain operations have exclusivity constraints that restrict what else can
appear in the same PE body:

| Rule | Constraint |
|------|------------|
| **Load/Store Exclusivity** | If `handshake.load` or `handshake.store` is present, the body must contain exactly one of these and no other non-terminator operations. |
| **Constant Exclusivity** | If `handshake.constant` is present, the body must contain exactly one `handshake.constant` and no other non-terminator operations. |
| **Dataflow Exclusivity** | The body must contain exactly one `dataflow` operation and `fabric.yield`. No other operations, no multiple dataflow ops, no `fabric.instance`. |
| **Instance-Only Prohibition** | A PE body must not consist solely of a single `fabric.instance`. |

## Homogeneous Consumption Rule

A `fabric.pe` body must use operations from exactly one consumption group:

**Full-consume/full-produce group:**
- All `arith` operations
- All `math` operations
- `handshake.load`, `handshake.store`, `handshake.constant`
- `handshake.fork`, `handshake.join`, `handshake.sink`

**Partial-consume/partial-produce group:**
- `handshake.cond_br`
- `handshake.mux`

Mixing operations from different groups is not allowed.

## Prohibited Operations

The following operations are **never** allowed inside `fabric.pe`:

- `fabric.switch`
- `fabric.temporal_pe`
- `fabric.temporal_sw`
- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`
- Any `handshake` operation not in the allowlist above

## Related Documents

- [spec-fabric-pe.md](./spec-fabric-pe.md): PE body constraints and semantics
- [spec-fabric.md](./spec-fabric.md): Fabric dialect overview
- [spec-adg-api.md](./spec-adg-api.md): ADGBuilder API reference
