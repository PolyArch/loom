# Fabric Reconfigurable Ops

This document specifies the set of software op kinds that the loom fabric
dialect can wrap inside a `fabric.op`, and which of those have **runtime
sw_config axes** that the subgraph enumerator explores. The canonical source
is `opSchemas()` in `lib/Fabric/IR/FabricOps.cpp`.

## Background

`fabric.op` is the sole bridge between software computation kinds (in
`dataflow`, `arith`, `math`, ...) and physical hardware capability. Every
`fabric.op` carries:

* `op_list` -- a non-empty array of `FlatSymbolRefAttr`s. Names the
  software op kind(s) this hardware block can execute. When the array has
  more than one entry, all entries must belong to a single hardware-share
  group (see `spec-fabric-hw-share-group.md`).
* `hw_params` -- an optional length-1 array wrapping a dictionary of
  parameters that change the hardware itself. Modifying `hw_params` means
  changing the synthesized block. Examples: the `bitmask` allowed-set for a
  variadic `dataflow.sync`, the `predicate` allowed-set for a configurable
  `arith.cmpi`.
* `sw_configs` -- an optional dictionary of runtime configuration values
  that alter the materialized software function without changing the
  hardware. Examples: `op_sel` to pick a member of a share group, `bitmask`
  to pick the active subset of a variadic op, `predicate` to pick a
  comparison code.

The op_list is itself a special hardware parameter: its presence and
contents fix what the block can compute. The `op_sel` `sw_config` then
chooses among the listed members at runtime.

## Two axes that the enumerator explores

Inside a `fabric.fu`, exactly five op kinds carry **structural** sw_config
axes whose choices change the materialized software graph topology, not
just attribute values:

* `fabric.mux` -- `sel` + `discard` + `disconnect`
* `fabric.demux` -- `sel` + `discard` + `disconnect`
* `fabric.op[@dataflow.sync]` -- `bitmask` (M-bit, N <= M ones)
* `fabric.op[@dataflow.mux]` -- `bitmask` (same envelope)
* `fabric.op[@dataflow.demux]` -- `bitmask` (same envelope)

Beyond the five structural axes, the enumerator also explores **attribute
axes** when `hw_params` declares an allowed set for them. These do not
change subgraph topology; they parameterize an existing op:

* `op_sel` -- multi-member `op_list` share groups.
* `predicate` -- `arith.cmpi` / `arith.cmpf` predicates, when restricted
  by `hw_params`.
* `step_op`, `cont_cond` -- `dataflow.stream` axes, when restricted.
* `const_hex_value` -- `dataflow.constant`, when the hardware fixes the
  allowed value set via `hw_params` (otherwise this is a free runtime
  constant, not enumerated).

If `hw_params` does not declare an allowed set for one of the attribute
axes, that attribute is a free `sw_config` and the enumerator does not
fan out across it.

## Runtime sw-configurable ops (the canonical seven)

The following op kinds support runtime reconfiguration via `sw_configs`.
They are the ones a hardware designer most commonly puts inside a
`fabric.fu` to give the FU programmable behavior.

| Software op | Configurable axes | Notes |
|---|---|---|
| `dataflow.stream` | `step_op`, `cont_cond` (when restricted by `hw_params`) | A streaming source whose stride and continuation predicate can be re-fixed at config time. |
| `dataflow.sync` | `bitmask` over M ports | Variadic. Selects the active N <= M input/output port pairs. The remaining ports are pruned upstream and downstream from the materialized subgraph. |
| `dataflow.constant` | `const_hex_value` (when restricted by `hw_params`) | Compile-time-set runtime constant. Without `hw_params` restriction the value is a free sw_config, not enumerated. |
| `dataflow.mux` | `bitmask` over M data ports | Variadic. Bitmask selects N active data ports; materialized `sel` width is `i1` for N=2 and `index` for N>=3. Hardware port stays at `bits<ceil(log2(M))>`. Data-dependent gating (only consumes the selected data input). |
| `dataflow.demux` | `bitmask` over M output ports | Variadic. Mirror of mux: bitmask selects which output ports are active; only `outputs[sel]` carries a value at runtime. |
| `arith.cmpi` | `predicate` (when restricted by `hw_params`) | Integer comparison. Default is the full 10-way set; `hw_params=[{predicate=[...]}]` restricts to a hardware-allowed subset. |
| `arith.cmpf` | `predicate` (when restricted by `hw_params`) | Floating-point comparison; same enumeration shape as `cmpi`. |

These seven plus the share-group `op_sel` axis (any multi-member op_list)
cover every runtime sw_config knob the enumerator explores.

## Non-configurable ops (no sw_config axes)

The following ops are accepted in `op_list` but have **no** sw_config axes
of their own. The enumerator emits a single template per occurrence,
modulo any other configurable elements elsewhere in the FU.

* All single-output, fixed-arity arithmetic: `arith.{addi, subi, muli,
  divsi, divui, remsi, remui, shli, shrsi, shrui, andi, ori, xori, minsi,
  maxsi, minui, maxui, addf, subf, mulf, divf, remf, minimumf, maximumf}`.
* Integer-floating casts: `arith.{sitofp, uitofp, fptosi, fptoui}`.
* All `math.*` unary ops: `math.{sin, cos, tan, sinh, cosh, tanh, exp,
  exp2, expm1, log, log2, log10, log1p, floor, ceil, round, trunc,
  roundeven, sqrt, rsqrt, absf, absi, erf}`.
* Fixed-arity dataflow ops: `dataflow.{carry, invariant, gate}`.
* `arith.select` -- strict-SSA eager 2-input mux. Distinct semantics from
  `dataflow.mux` (eager evaluation, consumes both data inputs). Does not
  belong to any share group, must occupy `fabric.op` alone.

The runtime knobs an FU may carry for these come from the surrounding
`fabric.mux` / `fabric.demux` and the variadic ops, not from the op
itself.

## Excluded ops

The following are deliberately not in `opSchemas()` and `fabric.op` will
reject them:

* `llvm.alloca`, `ub.poison`, `arith.constant` -- these are compile-time
  or pseudo ops that have no fabric realization.
* `dataflow.{load, store}` -- memory ops that the partitioner handles
  separately (they live at graph level, not inside `fabric.fu`).
* `dataflow.{graph, yield}`, `dataflow.subgraph` -- region ops, not
  computations.

## How sw_configs become enumerator axes

The enumerator (`SubgraphEnumerator::enumerateFuSubgraphs`) walks each
`fabric.fu` body and builds a list of `ChoiceAxis` records. One axis is
created when:

1. The op is a `fabric.mux` or `fabric.demux` -- yields a `_mode` axis
   covering all `(sel, discard, disconnect)` combinations.
2. The op is `fabric.op[@dataflow.sync]`, `fabric.op[@dataflow.mux]`, or
   `fabric.op[@dataflow.demux]` -- yields a `bitmask` axis. The allowed
   bitmasks are taken from `hw_params=[{bitmask=[...]}]` if present;
   otherwise the full 2^M - 1 enumeration is used (capped at M = 8).
3. The `fabric.op`'s `op_list` has more than one entry -- yields an
   `op_sel` axis with one value per member.
4. The `fabric.op`'s `hw_params` declares an array allowed-set for a
   non-`bitmask` key (`predicate`, `step_op`, `cont_cond`,
   `const_hex_value`, ...) -- yields one axis per such key.

The Cartesian product of all axes is the raw configuration space. After
per-config materialization, the enumerator runs `subgraphsIsomorphic`
between every pair of surviving templates and keeps the
lexicographically-smallest configuration per isomorphism class.

## Design principle

Distinct sw_configs are intended to map to distinct software functions. If
a `fabric.fu` produces many software-isomorphic templates under different
sw_configs, that is a smell in the FU design itself, not in the
enumerator. The enumerator's dedup discards the redundant configurations
deliberately.

## Maintenance

The canonical source of truth for the runtime sw-configurable set is
`opSchemas()` in `lib/Fabric/IR/FabricOps.cpp`. The `Variadic*` kinds
(`OpSchema::VariadicSync`, `VariadicMux`, `VariadicDemux`) flag which
schema entries get a bitmask axis; non-variadic schemas have an axis only
when `hw_params` restricts them. To extend either set, edit
`opSchemas()` and add a corresponding lit test under
`test/fabric/unit/fu_enum/` that pins the new axis behavior, then mirror
the change here.
