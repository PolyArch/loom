# LOOM Function Unit Allowed Operations and Timing Specification

## Overview

This document is the normative specification for what may appear inside one
`fabric.function_unit` body.

It defines:

- the operation allowlist for FU bodies
- the structural and typing rules for FU bodies
- the body-level behavior classes that determine whether `latency` and
  `interval` are meaningful
- the FU-local completion and `temporal_pe` output-drain model

This document is the single source of truth for the future
`fabric.function_unit` body validator.

Related documents:

- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-dataflow.md](./spec-dataflow.md)

## Design Intent

`fabric.function_unit` is LOOM's hardware abstraction for one software-visible
operation or one software-visible subgraph.

Compared with the legacy design's older `fabric.pe` body rules, LOOM keeps an explicit body
allowlist but intentionally relaxes several blanket exclusivity rules:

- `handshake.load` and `handshake.store` are not forced to be singleton or
  body-exclusive
- `handshake.constant` is not forced to be singleton or body-exclusive
- LOOM does not use the legacy design's homogeneous-consumption grouping rule

LOOM instead judges legality by three layers of rules:

- explicit body operation allowlist
- explicit structural and typing constraints
- explicit body-level timing class

This means that a compound FU body is legal when:

- every operation is individually allowed
- the body obeys the structural and typing rules below
- the whole body still has one coherent externally visible firing behavior

## FU Body Model

One `fabric.function_unit` body is a single-block SSA graph over native
software semantic values.

Normative consequences:

- FU block arguments are the FU inputs
- internal SSA values are FU-internal software values
- `fabric.yield` defines the FU outputs
- PE-side `!fabric.bits<...>` or `!fabric.tagged<...>` transport types must
  not appear inside the FU body
- width adaptation and tag handling belong to the enclosing PE and switch
  network, not to the FU body itself

The body is not a place to build routing structure, memory hierarchy
structure, or nested PE hierarchy. It is only a place to describe the
software behavior implemented by one hardware computation resource.

## Allowed Operation Families

Only the operations listed in this section may appear as non-terminator
operations inside one `fabric.function_unit` body.

### Fabric Operations

Allowed:

- `fabric.mux`
- `fabric.yield` as the terminator

Rules:

- `fabric.mux` is LOOM-specific and may appear only inside
  `fabric.function_unit`
- `fabric.yield` must be the only terminator of the body
- no other `fabric.*` operation is allowed inside an FU body

### `arith` Dialect

Allowed operations:

- `arith.addf`
- `arith.addi`
- `arith.andi`
- `arith.cmpf`
- `arith.cmpi`
- `arith.divf`
- `arith.divsi`
- `arith.divui`
- `arith.extsi`
- `arith.extui`
- `arith.fptosi`
- `arith.fptoui`
- `arith.index_cast`
- `arith.index_castui`
- `arith.mulf`
- `arith.muli`
- `arith.negf`
- `arith.ori`
- `arith.remsi`
- `arith.remui`
- `arith.select`
- `arith.shli`
- `arith.shrsi`
- `arith.shrui`
- `arith.sitofp`
- `arith.subf`
- `arith.subi`
- `arith.trunci`
- `arith.uitofp`
- `arith.xori`

Notes:

- `arith.cmpi` and `arith.cmpf` have runtime-configurable predicates as
  described in [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- `arith.constant` is not currently in the FU-body allowlist; constant
  injection inside one FU body is modeled by `handshake.constant`

### `math` Dialect

Allowed operations:

- `math.absf`
- `math.cos`
- `math.exp`
- `math.floor`
- `math.fma`
- `math.log2`
- `math.sin`
- `math.sqrt`

### `llvm` Dialect

Allowed operations:

- `llvm.intr.bitreverse`

No other `llvm.*` operation is currently allowed inside one FU body.

### `dataflow` Dialect

Allowed operations:

- `dataflow.carry`
- `dataflow.gate`
- `dataflow.invariant`
- `dataflow.stream`

Notes:

- all four current `dataflow` operations are dedicated fixed-behavior
  state-machine FUs in LOOM
- therefore each of them must occupy one exclusive `fabric.function_unit`
- a `function_unit` that contains any `dataflow.*` operation must use
  `latency = -1` and `interval = -1`

### `handshake` Dialect

Allowed operations:

- `handshake.cond_br`
- `handshake.constant`
- `handshake.join`
- `handshake.load`
- `handshake.mux`
- `handshake.store`

Notes:

- LOOM does not currently allow `handshake.sink` inside one FU body
- unused FU outputs are handled by PE-side discard or disconnect behavior, not
  by inserting `handshake.sink`
- LOOM does not require `handshake.load`, `handshake.store`, or
  `handshake.constant` to be body-exclusive

### Everything Else

All non-terminator operations outside the allowlist above are illegal inside
one `fabric.function_unit` body unless this document is amended.

## Runtime-Configurable Body Operations

Some allowed FU-body operations carry runtime-configurable fields.

Current configurable FU-body operations are:

- `fabric.mux`
- `handshake.constant`
- `handshake.join`
- `arith.cmpi`
- `arith.cmpf`
- `dataflow.stream`

No other FU-body operation currently contributes runtime configuration bits.

The current configurable fields are:

- `fabric.mux`: `sel`, `discard`, `disconnect`
- `handshake.constant`: output literal value
- `handshake.join`: `join_mask`
- `arith.cmpi`: integer comparison predicate
- `arith.cmpf`: floating-point comparison predicate
- `dataflow.stream`: `cont_cond`

Normative notes:

- these fields belong to the FU-internal runtime configuration payload, not to
  the enclosing PE mux or demux state
- body textual attributes provide the structural default or initial value, but
  the final mapped configuration may overwrite them
- configurable-field serialization order is defined in
  [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)

## Typing Rules

### FU Boundary Types

The FU signature must use native semantic types.

Currently allowed FU input and output types are:

- signless integers such as `i1`, `i8`, `i16`, `i32`, `i64`
- floating-point types such as `f16`, `f32`, `f64`
- `index`
- `none`

The following are not allowed on FU ports:

- `!fabric.bits<...>`
- `!fabric.tagged<...>`
- `memref<...>`
- PE, switch, or storage container types

### FU-Internal Value Types

Internal SSA values inside one FU body follow the same rule:

- use native semantic value types only
- do not use Fabric transport types
- do not use memory-reference types

Practical consequence:

- tag stripping or tag insertion must happen outside the FU body
- address transport into memory-capable structures happens through PE and
  switch wiring, then appears inside the FU body as native `index`
- `handshake.load` and `handshake.store` operate on native typed values in the
  FU body view

## Structural Rules

### Region and Block Structure

A `fabric.function_unit` body must contain exactly one block and must end in
`fabric.yield`.

The body must not contain nested control-flow regions or nested symbol
definitions.

### Yield Contract

Normative rules:

- `fabric.yield` must be the terminator
- the number of `fabric.yield` operands must equal the declared result count
- yield operand types must match the declared result types
- a yield operand must not be a direct block argument of the same FU

The last rule forbids trivial passthrough FUs. If a design wants a pure
forwarding path, it must be modeled outside the FU body by PE or switch
routing structure.

### Argument Consumption

Every FU block argument must be consumed by at least one non-terminator body
operation.

LOOM treats unused FU inputs as illegal dead interface.

### Minimum Body Content

An FU body must contain at least one non-terminator operation.

An empty body or a body that only forwards block arguments to `fabric.yield`
is illegal.

### Single-Block SSA Discipline

The body must be representable as one single-block SSA graph.

Normative consequences:

- no `func.*`, `cf.*`, `scf.*`, or `affine.*` control-flow structure is
  allowed inside an FU body
- no nested `fabric.function_unit`, `fabric.spatial_pe`, `fabric.temporal_pe`,
  `fabric.spatial_sw`, or `fabric.temporal_sw` is allowed
- stateful behavior must be expressed through allowed body operations such as
  `dataflow.carry`, `dataflow.invariant`, or `dataflow.stream`, not through
  region control flow

### Container and Hierarchy Prohibition

The following operations are never allowed inside one FU body:

- `fabric.module`
- `fabric.instance`
- `fabric.spatial_pe`
- `fabric.temporal_pe`
- `fabric.spatial_sw`
- `fabric.temporal_sw`
- `fabric.memory`
- `fabric.extmemory`
- `fabric.fifo`
- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`

These operations belong to ADG hierarchy, routing, memory topology, or tag
boundary modeling. They are not FU-body operations.

## Compound FU Bodies

LOOM allows one `fabric.function_unit` body to contain multiple internal
operations and to represent a compound software subgraph rather than one
single software op.

Typical legal patterns include:

- arithmetic pipelines such as multiply-then-add
- predicate formation followed by `handshake.cond_br` or `handshake.mux`
- memory address generation plus one or more `handshake.load` or
  `handshake.store`
- internal configurable alternatives expressed by one or more `fabric.mux`

LOOM does not require a compound body to be homogeneous by operation family.
For example, the following mixtures are legal in principle:

- `arith.*` together with `handshake.*`
- `handshake.load` or `handshake.store` together with arithmetic helpers
- `handshake.constant` together with other allowed operations

The `dataflow` family is the current exception:

- `dataflow.stream`
- `dataflow.gate`
- `dataflow.carry`
- `dataflow.invariant`

Each of these must occupy an exclusive FU body and must not be mixed with
other non-terminator operations.

The only requirement is that the whole body still has one coherent externally
visible FU behavior as described by the timing classes below.

## Body-Level Timing Classes

### Single-Fire Single-Result-Set Behavior

An FU body belongs to this class when one firing produces exactly one
externally visible result tuple.

The tuple may contain:

- zero results
- one result
- multiple results

The key property is that one firing has one FU-local completion event and one
result tuple.

For this class:

- `latency` is meaningful
- `interval` is meaningful
- `latency` must be `>= 0`
- `interval` must be `>= 1`

Current operations that belong to this class when used as standalone FU
behaviors:

- all allowed `arith.*`
- all allowed `math.*`
- `llvm.intr.bitreverse`
- `handshake.cond_br`
- `handshake.constant`
- `handshake.join`
- `handshake.load`
- `handshake.mux`
- `handshake.store`

The `dataflow` family is not in this class in LOOM's current hardware model.

### Dedicated Dataflow State-Machine Behavior

LOOM treats all four current `dataflow` operations as dedicated fixed state
machines rather than ordinary scalar-latency datapath FUs.

For this class:

- `latency` is not modeled as one scalar fire-to-completion delay
- `interval` is not modeled as one scalar refire spacing
- `latency` must be `-1`
- `interval` must be `-1`
- the body must contain exactly one non-terminator `dataflow` operation

Current members of this class:

- `dataflow.carry`
- `dataflow.gate`
- `dataflow.invariant`
- `dataflow.stream`

### Compound-Body Classification Rule

For one compound FU body, the timing class is a property of the whole body,
not of any single internal operation.

Normative rule:

- if the body contains any `dataflow.*` operation, it enters the dedicated
  dataflow state-machine class and must contain no other non-terminator
  operation
- if the whole body behaves as one firing to one result tuple, then it is
  single-fire single-result-set

Current conservative classifier for LOOM:

- any body containing `dataflow.stream`, `dataflow.gate`, `dataflow.carry`, or
  `dataflow.invariant` is treated as a dedicated dataflow state-machine FU
- any other currently allowed body is expected to satisfy the single-fire
  single-result-set contract

This is the intended first validator contract. A future LOOM revision may add
more refined whole-body behavior inference.

## Additional Per-Operation Rules

### `fabric.mux`

`fabric.mux` is the only FU-internal structural routing primitive.

Normative rules:

- it may appear only inside one `fabric.function_unit`
- it may be used to select among alternative internal producers or consumers
- it does not change the body-level timing class by itself
- its runtime-config fields are part of the FU internal configuration payload

See [spec-fabric-function_unit.md](./spec-fabric-function_unit.md) for the
full `sel`, `discard`, and `disconnect` semantics.

### `fabric.mux` vs `handshake.mux` vs `arith.select`

These three operations all express some form of selection, but they belong to
three different semantic layers and must not be conflated.

#### `fabric.mux`

`fabric.mux` is a configuration-time structural selector.

Normative interpretation:

- its `sel` comes from FU-internal runtime configuration, not from one
  runtime token operand
- once configured, it behaves as one fixed hard path inside the selected FU
  shape
- it is best understood as one FU-internal static routing primitive, similar
  in spirit to a tiny `spatial_sw` inside the `function_unit`
- tech-mapping and config generation treat it as part of the hardware shape
  choice of the FU

In other words, `fabric.mux` chooses which hardware subgraph is active, not
which runtime input token is selected on one dynamic firing.

#### `handshake.mux`

`handshake.mux` is a runtime dataflow operator.

Normative interpretation:

- its selector is a runtime operand
- one firing consumes the selector and the selected data input
- non-selected data inputs are not consumed by that firing
- therefore non-selected inputs remain blocked with respect to that firing

`handshake.mux` is not a hard configured route. It is an executing software
operator with firing-time consume behavior.

#### `arith.select`

`arith.select` is a runtime datapath operator.

Normative interpretation:

- its selector is a runtime operand
- one firing consumes the selector and all data operands
- the result value is whichever data operand the selector chooses
- even the non-chosen data operand is still consumed as part of the firing

`arith.select` is therefore different from `handshake.mux` even though both
are runtime-controlled selectors.

#### Summary

The three-way distinction is:

- `fabric.mux`: configuration-selected hard route inside the FU body
- `handshake.mux`: runtime-controlled handshake selection that consumes only
  the selected input and blocks the others
- `arith.select`: runtime-controlled datapath selection that consumes all
  inputs and only chooses which consumed value becomes the output

### `handshake.constant`

Normative rules:

- it is allowed to coexist with other allowed operations
- it contributes runtime-config bits for its literal value
- it is not body-exclusive in LOOM

### `handshake.join`

LOOM treats one `handshake.join` inside an FU body as a fixed maximum hardware
fan-in synchronizer with runtime-selectable participating inputs.

Normative rules:

- textual operand count defines hardware fan-in
- current supported hardware fan-in range is `1..64`
- the join contributes one runtime-config bit per textual input
- multiple joins may appear inside one FU body

### `handshake.load` and `handshake.store`

Normative rules:

- they may coexist with other allowed operations
- they are not singleton-only and not body-exclusive in LOOM
- multiple loads and stores may appear in one compound body if the whole body
  still represents one coherent FU behavior

This is a deliberate LOOM relaxation compared with the legacy design's earlier PE-body
rules.

### `dataflow.carry`, `dataflow.gate`, and `dataflow.invariant`

Normative rules:

- each must occupy an exclusive FU body
- the containing FU must use `latency = -1` and `interval = -1`
- they are not mixed with arithmetic, handshake, or `fabric.mux` operations in
  the current LOOM hardware model

### `dataflow.stream`

Normative rules:

- it must occupy an exclusive FU body
- therefore the containing FU must use `latency = -1` and `interval = -1`
- its continuation condition field contributes runtime configuration bits

## Prohibited Patterns

The following patterns are illegal even if each individual operation were
otherwise allowed:

- direct passthrough from FU input block argument to `fabric.yield`
- unused FU block arguments
- nested container or routing operations inside the body
- tag-boundary manipulation inside the body
- control-flow regions inside the body
- memory hierarchy construction inside the body
- any body that mixes one `dataflow.*` operation with any other non-terminator
  operation
- body-local operation sets that cannot be classified into one coherent
  body-level timing class

In particular, the following old legacy-style assumptions are not part of LOOM:

- no load/store exclusivity rule
- no constant exclusivity rule
- no homogeneous-consumption grouping rule

## Validator Checklist

The future `fabric.function_unit` body validator should enforce at least the
following rules:

1. every non-terminator body operation is in the allowlist above
2. the body has exactly one block and ends in `fabric.yield`
3. `fabric.yield` arity and types match the FU signature
4. no yield operand is a direct block argument
5. every block argument is consumed
6. the body contains at least one non-terminator operation
7. no prohibited Fabric hierarchy, routing, memory, or tag operation appears
   inside the body
8. no control-flow or region-bearing program structure appears inside the body
9. `handshake.join` operand count lies in the supported `1..64` range
10. body timing class and `latency` or `interval` settings are consistent
11. any body containing one `dataflow.*` operation is checked under the
    dedicated dataflow state-machine rules

This document defines the intended validator contract even if the current code
does not yet enforce all of it.

## `temporal_pe` Completion and Output Arbitration

This section defines how FU-local completion interacts with
`fabric.temporal_pe`.

### Fire

A `function_unit` fires when it consumes one complete input tuple for its
currently selected behavior.

For a direct single-op FU, this means the underlying operation has accepted
all operands required by that operation.

For a compound FU body that implements one software subgraph, this means the
FU has accepted one externally visible input tuple for that subgraph.

### Completion

A `function_unit` completes when it has produced one externally visible result
tuple for one firing.

The completion event is FU-local. In a `temporal_pe`, FU-local completion is
not identical to immediate visibility on the PE egress, because the result may
wait in one FU-local output register before arbitration grants an egress slot.

### `latency`

`latency` is the FU-local delay from firing to FU-local completion.

Normative rules:

- valid range is `0..`
- `0` means combinational completion in the same cycle as the firing
- `-1` means not applicable
- `latency` is defined only for single-fire single-result-set behavior

`latency` does not include extra waiting introduced by temporal-PE output
arbitration after the FU has already completed.

### `interval`

`interval` is the minimum number of cycles from one firing to the next firing
of the same FU, assuming no additional blocking condition exists.

Normative rules:

- valid range is `1..`
- `1` means fully pipelined
- `-1` means not applicable
- `interval` is defined only for single-fire single-result-set behavior

In a `temporal_pe`, the refire condition is the conjunction of:

- the FU's intrinsic `interval` constraint
- the temporal scheduler selecting that FU
- all required operands being ready
- the FU not being busy because one or more FU-local output registers still
  hold undrained results

### Issue Model

Normative rule:

- each `temporal_pe` may fire at most one FU per cycle

The selected FU is determined by the active instruction slot and the temporal
PE scheduler.

### FU-Local Output Registers

Every FU output port in one `temporal_pe` has a dedicated FU-local output
register.

Normative rules:

- every FU completion writes its produced result values into these FU-local
  output registers
- there is no direct bypass from FU completion to one temporal-PE egress port
- the arbitration stage always observes FU-local output registers, never raw
  FU combinational outputs

This rule applies even when static analysis suggests that no conflict is
possible in one particular program.

### Busy Definition

An FU is busy if any of its FU-local output registers still contains one
undrained valid result.

Normative rules:

- a busy FU must not fire again
- this prohibition applies even if an instruction selects that FU and all
  operand inputs are ready
- the FU becomes idle only after all valid output-register contents produced by
  the previous firing have been drained to their intended PE egresses

Therefore the temporal-PE refire rule is stricter than raw `interval` alone.

### Output Conflict Scenario

Two different FUs may complete in the same cycle even though only one FU fires
per cycle, because their latencies may differ.

Example:

- `fuA` fires at cycle `0`
- `fuB` fires at cycle `1`
- `fuA.latency = 4`
- `fuB.latency = 3`

Both FU-local completions occur at cycle `4`.

### Arbitration Policy

When multiple valid FU-local output registers request the same temporal-PE
egress opportunity, the PE uses round-robin arbitration.

Normative rules:

- arbitration order is by FU definition order inside the `temporal_pe`
- FU definition order is the same order used for opcode numbering
- after reset, the initial highest priority is the lowest opcode
- after a successful grant, the round-robin pointer advances so later grants
  continue from the next FU in cyclic order

If one or more requesting FUs are not granted:

- their results remain stored in their own FU-local output registers
- those FUs remain busy
- they become eligible for output again in later arbitration cycles

### Observation About `latency`

In a `temporal_pe`, the observable egress timing of one FU result is:

- FU fire time
- plus FU-local `latency`
- plus any additional waiting time in the FU-local output register until the
  arbiter grants egress

Therefore:

- `latency` models FU-local compute delay
- arbitration delay is a separate temporal-PE effect

## LOOM Relaxation Summary Compared with the legacy design

LOOM deliberately differs from the legacy design's older PE-body rules in the following
ways:

- `fabric.mux` is an allowed LOOM-specific FU-body operation
- `handshake.sink` is not an allowed FU-body operation
- `handshake.load` and `handshake.store` are not body-exclusive
- `handshake.constant` is not body-exclusive
- current `dataflow` operations are exclusive dedicated FUs with
  `latency = -1` and `interval = -1`
- no homogeneous-consumption grouping rule is used
- FU bodies are structurally stricter about hierarchy: no nested
  `fabric.instance` and no nested PE definitions inside the body

## Related Documents

- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-dataflow.md](./spec-dataflow.md)
