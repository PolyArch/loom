# Loom Dataflow Part 1: Streaming And Channel Ops

This document specifies the precise token timing semantics of the
stream-shaping ops in the `dataflow` dialect:
`dataflow.stream`, `dataflow.carry`, `dataflow.invariant`, and
`dataflow.gate`.

It also owns the thread-level ordered-channel type and endpoint semantics for
`dataflow.channel.create`, `dataflow.channel.send`, and
`dataflow.channel.receive`. Graph-local SSA streams and thread-level channels
are related at `dataflow.graph.launch`, but they are not the same IR object.

Vector stream grouping and scalar/packed stream adaptation are specified
separately in `docs/spec-dataflow-vectorization.md`. This document owns
the scalar stream-shaping primitives that those vectorization ops build
on.

This document owns the semantic contract for these operations.
`include/Dataflow/IR/DataflowOps.td` and
`lib/Dataflow/IR/DataflowOps.cpp` plus
`lib/Dataflow/IR/DataflowChannelOps.cpp` are implementation projections and
must conform to it. Compiler specs that use these operations, especially
`docs/spec-compiler-part-3-dfg.md`, reference this contract rather than
redefining it.

## 1. Token Stream Model

A dataflow SSA value denotes an ordered token stream. A token may carry
ordinary data, such as an integer, floating point value, vector, or
`none`, or it may represent a control event. Multiple SSA uses of the
same stream are token broadcast: each use observes the same ordered
token sequence.

The four ops in this document shape streams. They do not define memory
regions. A `memref<...>` value in the compiler frontend represents a
memory-region binding. It must not be duplicated, delayed, or
phase-normalized by `dataflow.carry`, `dataflow.invariant`, or
`dataflow.gate`. Dynamic control chooses address, data, operation, and
`none` ordering streams; it does not choose the frontend memref binding
itself. Memory accesses use the memref as the static binding operand of
`dataflow.load` / `dataflow.store`; ordering is carried by explicit
`none` tokens.

## 2. Phase Streams

A `phase : i1` stream is loop control, not a validity bit. For one
normally closed activation:

```
K = number of true decisions and body executions
M = K + 1
phase = true^K, false
```

The final false token is an ordinary close transition. It resets the
stateful actors driven by that phase stream, but it does not imply an IV,
body value, feedback value, or body execution. Repeated activations are
represented by concatenated phase segments; each false token closes one
segment and returns the actors to their initial states.

## 3. `dataflow.stream`

`dataflow.stream` produces a valid IV stream and a loop-level phase stream
from scalar integer recurrence operands:

```
%iv, %phase = dataflow.stream %init, %limit, %step
  step add while slt : iN
```

The operation starts in Idle. Idle consumes one `%init`, `%limit`, `%step`
triple and establishes a current value initialized to `%init`.

* If the `while` predicate holds for `(current, limit)`, it emits `current`
  on `%iv`, emits `true` on `%phase`, advances the current value with the
  selected `step` kind, and remains active.
* If the `while` predicate does not hold, it emits only `false` on `%phase`
  and returns to Idle.

The step kind is a `dataflow::StreamStepKind` value: `add`, `sub`, `mul`,
`sdiv`, `udiv`, `shl`, `ashr`, or `lshr`. The continuation predicate is the
upstream `mlir::arith::CmpIPredicate` enum. These generated enums are the
shared implementation representation of the closed choices specified here.
The parser, verifier, simulator, and hardware-configuration projection must
consume that representation rather than maintain string-based copies.

The canonical operation requires `%init`, `%limit`, `%step`, and `%iv` to
share a scalar signless integer type. `%phase` is always `i1`.

For `init = 0`, `limit = 5`, `step = 1`, step kind `add`, and predicate
`slt`:

| Result | Tokens |
|--------|--------|
| `%iv` | `[0, 1, 2, 3, 4]` |
| `%phase` | `[T, T, T, T, T, F]` |

For a zero-trip activation with `init = limit = 5`:

| Result | Tokens |
|--------|--------|
| `%iv` | `[]` |
| `%phase` | `[F]` |

Because `%iv` already has exactly K tokens, body address computation and
memory effects consume it directly. Pairing `%phase` and `%iv` through a
`dataflow.gate` is incorrect: the two streams have different cardinalities.

## 4. `dataflow.carry`

`dataflow.carry` is a two-state token element for loop-carried values or
hidden loop-carried `none` state:

```
%output = dataflow.carry %phase, %init, %next : T
```

The operation starts in `init` state.

* In `init` state, it waits for one `%init` token, forwards it to
  `%output`, and transitions to `carry` state.
* In `carry` state, it first inspects the head `%phase : i1` token.
* If `%phase` is true, it requires and consumes one `%next` token, consumes
  the phase token, forwards `%next` to `%output`, and stays in `carry` state.
* If `%phase` is false, it consumes only the phase token, emits no output,
  and returns to `init` state.

`%init`, `%next`, and `%output` have the same type. `%phase` is `i1`.

For `%phase = [T, T, F]`, carry consumes two next values and produces three
outputs:

| Event | `%phase` consumed | `%next` consumed | `%output` |
|-------|-------------------|------------------|-----------|
| init | none | none | `init` |
| first transition | true | `next0` | `next0` |
| second transition | true | `next1` | `next1` |
| close | false | none | none |

For one closed activation, carry consumes one init, K next values, and M
phase tokens, and emits M outputs: `[init, next0, ..., next(K-1)]`.

## 5. `dataflow.invariant`

`dataflow.invariant` latches one initial value and replays it while a
condition stream remains true:

```
%output = dataflow.invariant %cond, %init : T
```

The operation starts in `init` state.

* In `init` state, it waits for one `%init` token, records the value,
  forwards it to `%output`, and transitions to running state.
* In running state, it waits for one `%cond : i1` token. It does not
  consume another `%init` token in this state.
* If `%cond` is true, it re-emits the recorded value and stays in
  running state.
* If `%cond` is false, it emits no output, clears the recorded value,
  and returns to `init` state.

`%init` and `%output` have the same type. `%cond` is `i1`.

For a loop-level condition stream `[true, true, false]`, the output is
three copies of the initial value: one immediate copy, one for the
first true condition, and one for the second true condition. The false
condition resets the op and emits nothing.

For a zero-trip loop-level condition stream `[false]`, the output is
one copy of the initial value. That copy belongs to loop phase, not
body phase; a selector such as `dataflow.demux` can route it to the
loop-exit path, leaving the body path empty.

`dataflow.invariant` is appropriate for scalar values, index-like
values, vector values, and `none` control tokens. It is not appropriate
for frontend `memref<...>` bindings.

## 6. `dataflow.gate`

`dataflow.gate` converts a `(cond, value)` stream into a region-local
phase:

```
%after_cond, %after_value = dataflow.gate %before_cond, %before_value : T
```

The operation starts in `init` state. It always consumes
`%before_cond` and `%before_value` together when it fires.

* In `init` state, `(false, X)` emits nothing and stays in `init`.
* In `init` state, `(true, X)` emits `X` on `%after_value`, emits no
  `%after_cond` token, and transitions to `continue`.
* In `continue` state, `(true, X)` emits `true` on `%after_cond`,
  emits `X` on `%after_value`, and stays in `continue`.
* In `continue` state, `(false, X)` emits `false` on `%after_cond`,
  emits no `%after_value` token, and returns to `init`.

`%before_value` and `%after_value` have the same type. The condition
operands and results are `i1`.

For a parent input with K true tokens and one trailing false close,
`%after_value` has exactly K tokens. `%after_cond` is empty when K is zero;
otherwise it has K tokens and is phase-shifted:

| Input `%before_cond` | `%after_value` | `%after_cond` |
|----------------------|----------------|---------------|
| `[false]` | `[]` | `[]` |
| `[true, false]` | `[v0]` | `[false]` |
| `[true, true, false]` | `[v0, v1]` | `[true, false]` |

The `%after_cond` result is not a validity bit for `%after_value`.
Instead, it is the local close stream for the region whose values have
already been gated:

* `%after_cond = true` means this region execution is not the last one.
* `%after_cond = false` means this region execution is the last one.

The parent false-lane input value is intentionally dropped by `gate`. If a
lowering needs that value, as with the false `scf.condition` operands
that become `scf.while` loop results, it must preserve the value with a
separate projection such as `dataflow.demux`.

## 7. Compiler Usage Rules

Lowering passes that use these ops must preserve the following rules:

* Loop-level phase streams keep the final false close token.
* `dataflow.stream` IVs already have body cardinality and are consumed
  directly; they are not paired with phase through `dataflow.gate`.
* Parent-domain carry and invariant outputs are projected through
  `dataflow.gate` before body arithmetic or memory use.
* Loop results and exit frontiers are projected from the false lane of
  `dataflow.demux %phase, %parent_value`.
* Carry feedback contains exactly K real next values. The false close
  transition never consumes a dummy feedback token.
* Body-local state whose values have already been gated may be driven by
  the body-local close stream from `dataflow.gate`.
* False-lane values are not recovered from `dataflow.gate`; they must
  be routed separately if they are semantically needed.
* Phase fanout is independent per use. One blocked consumer does not prevent
  another phase consumer from firing when its own operands are ready.
* Frontend `memref<...>` values are bindings, not stream values shaped
  by these ops. Dynamic control shapes the address, data, operation,
  and `none` order streams instead.

## 8. Thread-Level Ordered Channels

The sole logical channel type is:

```text
!dataflow.channel<T>
```

`T` is an ordinary typed message payload and may be scalar, vector, tile,
descriptor, coordinate/value pair, or another finite value type. It must not
contain a channel, `!dataflow.thread_token`, or a memory capability. A channel
handle is connectivity identity, not a FIFO pointer or payload; it cannot be
loaded, stored, nested in another channel, or sent as a message.

Each dynamic execution of `dataflow.channel.create` creates a fresh logical
channel instance. Creation occurs outside `loom.spatial_region` and
`dataflow.graph`; it cannot be CSE'd as a pure value. The initial profile has no
channel-level session, epoch, open, close, reset, or built-in EOS operation.
Known logical domains or an explicit payload protocol own termination.
`DynamicWorkDomain` quiescence is owned by
`docs/spec-compiler-part-4-partitioned-data.md` and may retire its launch token;
it does not close a channel or manufacture an EOS message. A concurrently
receiving consumer that cannot otherwise know when to stop remains unsupported
unless its payload protocol expresses termination explicitly.

`dataflow.channel.send` and `dataflow.channel.receive` are InstructionCore
stored-program operations inside a `dataflow.thread` body and are forbidden in
a canonical `dataflow.graph`. Send blocks as required to submit one message in
program order. Receive blocks until it can consume and return the oldest
message. Logical capacity, occupancy, physical latency, and physical buffering
are unobservable; physical stalls may change performance but never content or
order. The initial profile has no try-send, try-receive, size, empty/full, or
select-any-ready operation.

A logical channel has at most one producer/output binding and may have several
consumer/input bindings. Each consumer binding owns a total deterministic
`source_map` from its logical consumer domain to the producer domain. Several
consumers may select one producer, yielding multicast in which every branch
observes the same ordered sequence. Many-to-one competitive receive requires
an explicit merge, router, reduction actor, or memory-backed work queue; it is
not implicit channel arbitration.

At a graph launch, channel input bindings derive graph-local stream inputs and
channel output bindings consume graph-local stream outputs. Channel handles do
not enter the graph body. A graph-local stream close is not a channel message,
consumer EOS, channel close, or thread completion. Channel message delivery is
an ordinary causal edge; conflicting shared-memory visibility must respect
that causality without adding a channel-specific memory-order mode.

Mapping selects a physical path or network that preserves FIFO behavior. It
may use NoC links, switches, buffers, virtual channels, `fabric.fifo`, or a
memory-backed ordered queue when Fabric declares the capability. The logical
channel does not prescribe one physical FIFO, route, capacity, or protocol.

Stable verification anchors cover payload-type rejection, fresh creation,
send/receive placement and type agreement, one-producer/multi-consumer source
mapping, ordered blocking behavior, and the absence of hidden EOS or
capacity-visible behavior. Tests do not enumerate physical transports,
message types, or report formatting.

## 9. References

* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering templates
  that use these streaming semantics.
* `docs/spec-dataflow-part-2-control.md` -- firing semantics for
  `dataflow.mux`, `dataflow.demux`, `dataflow.sync`, and
  `dataflow.constant`.
* `include/Dataflow/IR/DataflowOps.td` -- operation declarations that
  implement this contract.
* `lib/Dataflow/IR/DataflowOps.cpp` and
  `lib/Dataflow/IR/DataflowChannelOps.cpp` -- verifier implementations.
* `test/dataflow/unit/stream/`, `test/dataflow/unit/carry/`,
  `test/dataflow/unit/invariant/`, `test/dataflow/unit/gate/`, and
  `test/dataflow/unit/channel/` -- unit-level syntax and verifier tests.
