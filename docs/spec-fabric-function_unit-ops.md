# FCC Function Unit Operation and Timing Specification

## Overview

This document specifies:

- what classes of software behavior a `fabric.function_unit` may implement
- how `latency` and `interval` are interpreted
- when those two hardware parameters are meaningful
- how a `fabric.temporal_pe` observes FU completion and drains FU outputs

This document is the normative timing companion of:

- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)

## Fundamental Execution Terms

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
wait in an FU-local output register before arbitration grants an egress slot.

### Single-Fire Single-Result-Set Behavior

An FU belongs to this class if one firing produces exactly one result tuple.

The tuple may contain:

- zero results
- one result
- multiple results

The key property is not the result count. The key property is that one firing
has one completion event and one result tuple.

For this class, `latency` and `interval` are meaningful.

### Multi-Beat or Multi-Result-Set Behavior

An FU belongs to this class if one firing may produce multiple externally
visible result sets over time.

For this class, `latency` and `interval` are not meaningful as fixed FU-level
hardware parameters, because there is no single completion point and no single
refire rule that captures the full behavior.

For this class:

- `latency` must be `-1`
- `interval` must be `-1`

The current normative example is `dataflow.stream`.

## Hardware Parameters

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

## Supported Operation Classes

The allowlist for what may appear inside a `fabric.function_unit` body is
defined by [spec-fabric-function_unit.md](./spec-fabric-function_unit.md).
This document classifies those allowed bodies by externally visible timing
behavior.

## Pure Datapath Operations

Representative operations:

- `arith.*`
- `math.*`
- `llvm.intr.bitreverse`

Behavior:

- one firing consumes one full operand tuple
- one completion produces one result tuple
- no FU-internal externally visible phase splitting exists

Timing model:

- `latency` is meaningful
- `interval` is meaningful

This class includes both:

- one-op FUs
- compound FU bodies whose whole externally visible behavior still remains one
  firing to one result tuple

## Predicate and Conditional Routing Operations

Representative operations:

- `arith.select`
- `handshake.mux`
- `handshake.cond_br`

Behavior:

- one firing consumes one full operand tuple
- one completion produces one result tuple, although only a subset of result
  ports may be active depending on runtime predicate or select state

Timing model:

- `latency` is meaningful
- `interval` is meaningful

The fact that a conditional operator may selectively activate outputs does not
turn it into a multi-result-set FU. It still has one FU-local completion event
per firing.

## Handshake Utility Operations

Representative operations:

- `handshake.constant`
- `handshake.join`
- `handshake.sink`

Behavior:

- one firing consumes one full operand tuple according to the operation's own
  handshake contract
- one completion produces one result tuple, possibly an empty tuple or a tuple
  containing `none`

Timing model:

- `latency` is meaningful
- `interval` is meaningful

## Memory Request and Response Operations

Representative operations:

- `handshake.load`
- `handshake.store`

Behavior:

- one firing consumes one full memory-operation input tuple
- one completion produces one result tuple for that memory operation

Timing model:

- `latency` is meaningful
- `interval` is meaningful

This remains true even when the result tuple contains multiple values, such as
data plus address or control bookkeeping. It is still one completion event for
one firing.

## Stateful Single-Result-Set Dataflow Operations

Representative operations:

- `dataflow.gate`
- `dataflow.carry`
- `dataflow.invariant`

Behavior:

- the FU may maintain state across firings
- each firing still has one FU-local completion event
- each firing still produces one result tuple

Timing model:

- `latency` is meaningful
- `interval` is meaningful

Statefulness alone does not invalidate `latency` or `interval`. The deciding
criterion is still whether one firing maps to one completion event and one
result tuple.

## Multi-Beat Generator Operations

Representative operations:

- `dataflow.stream`

Behavior:

- one start event may cause a sequence of externally visible outputs over time
- the operation does not have one stable FU-local completion point that
  captures the whole behavior

Timing model:

- `latency = -1`
- `interval = -1`

This rule also applies to any future FU behavior with the same shape:

- one firing may emit multiple result sets
- one firing may keep generating outputs across later cycles
- no single completion event fully characterizes the behavior

## Compound FU Bodies

A `fabric.function_unit` body may contain multiple internal operations and may
implement a software subgraph instead of a single software op.

Normative rule:

- `latency` and `interval` are properties of the externally visible FU
  behavior, not of any single internal body operation

Therefore:

- if the whole FU behaves as one firing to one result tuple, `latency` and
  `interval` are meaningful and describe the aggregate FU
- if the whole FU behaves as one firing to multiple result sets over time,
  `latency` and `interval` must both be `-1`

## `temporal_pe` Completion and Output Arbitration

This section defines how FU-local completion interacts with
`fabric.temporal_pe`.

### Issue Model

Normative rule:

- each `temporal_pe` may fire at most one FU per cycle

The selected FU is determined by the active instruction slot and the temporal
PE scheduler.

### FU-Local Output Registers

Every FU output port in a `temporal_pe` has a dedicated FU-local output
register.

Normative rules:

- every FU completion writes its produced result values into these FU-local
  output registers
- there is no direct bypass from FU completion to a temporal-PE egress port
- the arbitration stage always observes FU-local output registers, never raw
  FU combinational outputs

This rule applies even when static analysis suggests that no conflict is
possible in one particular program.

### Busy Definition

An FU is busy if any of its FU-local output registers still contains an
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

## Summary Rules

- `latency` models fire-to-FU-local-completion delay
- `interval` models the minimum intrinsic fire-to-next-fire spacing
- both are meaningful only for single-fire single-result-set behavior
- `dataflow.stream` is the current normative example where both must be `-1`
- every `temporal_pe` FU output is captured by FU-local output registers before
  PE egress
- no FU in `temporal_pe` may bypass that output-register and arbitration stage
- a temporal-PE FU remains busy while any of its output registers still holds
  an undrained valid result

## Related Documents

- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-dataflow.md](./spec-dataflow.md)
