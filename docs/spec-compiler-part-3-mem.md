# Loom Compiler Part 3 Memory Frontier Lowering

This document is the memory-order source of truth for graph-local SCF to
Dataflow lowering. The concrete owner is `loom-lower-graph-memory`; it
normalizes supported memory leaves and recursively lowers structured graph
regions in one traversal.

The Dataflow operation contracts remain owned by the Dataflow specifications.
This document defines only the compiler analysis state and the ordinary SSA
event network produced from it.

## 1. Scope

The implemented slice covers:

* `dataflow.load`, `dataflow.store`, rank-one `memref.load`, and rank-one
  `memref.store` leaves;
* sequential composition;
* arbitrary nesting of `scf.if`, source-sequential `scf.for`, and
  `scf.while`;
* basic graph-local alias-root partitions;
* conservative unknown accesses;
* value, execution, write-frontier, and read-frontier projection through the
  same structured selectors;
* pre-mutation rejection of residual `scf.parallel` and `scf.forall` that
  reach a graph without an already materialized schedule boundary.

The lowering does not select parallel width, ownership, serialization,
unrolling, reduction order, or any other schedule policy. Those decisions
must be made before graph-region lowering and normalized into supported
structured input.

## 2. One Recursive Owner

The compiler-local contract is:

```text
lower_region(E_in, values_in, {W_in[p], R_in[p]})
  -> (E_out, values_out, {W_out[p], R_out[p]})
```

`E` is execution permission and structural completion. `W` and `R` are
memory-order frontiers for alias partition `p`. They share the ordinary
`none` SSA type but remain semantically distinct throughout lowering.

The contract is an implementation function, not an IR object. Canonical IR
does not contain partition ids, dependence snapshots, compound-region
objects, chain-scope attributes, memory tokens, or memory-specific join
operations.

The recursive owner replaces the former split among reduction, invariant,
control, and sync passes. No later pass reconstructs structured memory order
or graph completion from a hidden effect scan.

## 3. Basic Alias Partitions

Partition identity is local to one `dataflow.graph.func` lowering run.

A known root is found by peeling supported view-like values until reaching a
storage or boundary root. The baseline recognizes:

* graph and function block arguments, conservatively grouped at the graph
  boundary unless an argument carries explicit no-alias evidence;
* `memref.alloc`, `memref.alloca`, and `memref.get_global` results;
* `ViewLikeOpInterface` producers, including the standard memref view family;
* single-input `builtin.unrealized_conversion_cast` bridges;
* the existing view-like `dataflow.partition_layout` and
  `dataflow.map_info` boundaries.

Distinct allocation roots are independent partitions. Repeated
`memref.get_global` operations are keyed by global symbol. Distinct graph
arguments are not independent merely because they occupy different ABI
positions: launch verification permits one actual capability in several
positions. The analysis does not use address ranges, affine disjointness,
bank identity, physical ports, or element-type compatibility to split a root.

If the root walk reaches an unrecognized producer, the access is unknown.
The graph first discovers every known partition, then assigns each unknown
access to every known partition and to one graph-local unknown bucket. This
has two consequences:

* an unknown access orders against every known access;
* unknown accesses still order against each other when no known root exists.

Access-to-partition membership is kept in a transient operation map before
SCF operands are projected. Selector demuxing must not change alias identity.
The map is discarded after explicit event edges are emitted.

## 4. Canonical Frontier State

Each partition has exactly two analysis values:

```text
(write_frontier[p], read_frontier[p])
```

`write_frontier[p]` covers the maximal write tail not superseded by a later
causal write. `read_frontier[p]` covers that write frontier plus maximal reads
not superseded by a later write. The invariant is:

```text
write_frontier[p] <= read_frontier[p]
```

At graph entry, both components equal the leading graph start value for every
partition. A region that does not touch `p` forwards both components without
creating projection or recurrence actors.

Compiler `join` means an all-of causal frontier. It is materialized with
ordinary `dataflow.sync` after deduplication and conservative transitive
reduction. Mutually exclusive alternatives use `dataflow.mux`, never
`dataflow.sync`.

## 5. Leaf Transfers

For a read covering partitions `P(access)`:

```text
ctrl = join(E, W[p] for p in P(access))
done = read.done
W[p] remains unchanged
R[p] = join(R[p], done)
```

For a write covering partitions `P(access)`:

```text
ctrl = join(E, R[p] for p in P(access))
done = write.done
W[p] = done
R[p] = done
```

These equations are the complete hazard authority:

* RAW: a read waits for the current write frontier;
* WAR: a write waits for all outstanding reads;
* WAW: a write waits for the read frontier, which covers the prior write;
* RAR: a read does not wait for prior reads.

For `R0; R1; W2; R3` on one partition, `R0` and `R1` both receive the incoming
write frontier. `W2` receives an all-of of both read completions. `R3` receives
`W2.done`. A different root keeps its own incoming frontier and receives no
cross-partition edge.

Memory completion does not become execution permission. A straight-line leaf
does not replace `E`; its effects are represented only in `W/R`. This avoids
reintroducing RAR order through a structural token. Structured children do
produce a new `E_out`, and a parent continuation waits for that structural
exit.

## 6. Selection

For `scf.if`, the condition drives all projections:

* demux `E_in` into false and true execution lanes;
* demux every captured non-memory value into matching lanes;
* demux `W_in[p]` and `R_in[p]` for each partition touched by either branch;
* recursively lower each branch;
* mux results, execution, `W`, and `R` componentwise with the same condition.

Memref bindings are static capabilities and are not demuxed. Address, data,
selector, and event values are projected as ordinary streams.

A missing else region is an identity false path. Its execution lane and every
touched incoming frontier component flow directly to the corresponding mux.
No fake load, fake store, safe address, dummy done, or eager `arith.select`
may stand in for an unexecuted memory access.

The same captured value may feed several uses within one branch; normal SSA
multi-use provides token broadcast. A branch-local zero-operand constant is
converted to `dataflow.constant` using that branch's execution permission.

## 7. Source-Sequential `scf.for`

`dataflow.stream` produces `K` valid induction values and a `T^K F` loop
selector. Index bounds are cast to the configured integer index width before
the stream and the induction value is cast back for source index uses.

The loop owns independent recurrence rings for:

* execution permission;
* every source iter_arg;
* `W[p]` for each touched partition;
* `R[p]` for each touched partition.

Each ring uses `dataflow.carry` under the loop selector. A matching
`dataflow.demux` sends true-lane values into the body and the false-lane value
to loop exit. Captured non-memory values are replayed with
`dataflow.invariant` and projected into body phase with `dataflow.gate`.
Memref capabilities are not replayed.

The recursively lowered body supplies all recurrence feedback values. The
execution feedback is the body's structural exit; memory feedback is the
body's resulting frontier pair. The rings are independent even when a write
assigns the same `done` to both memory components.

For zero trip count, the stream emits only `F`. No body address or access
fires. Every carry exposes its init value on the false lane, so source values,
execution, `W`, and `R` transfer through the loop unchanged.

No dependence is removed because a loop appears parallelizable. Source
iteration order remains authoritative until an earlier transformation has
materialized a different schedule with provenance.

## 8. `scf.while`

For a while loop whose after region executes `K` times:

* before executes `K + 1` times;
* after executes `K` times;
* the before condition stream is `T^K F`.

Execution, source inits, and touched `W/R` components use condition-driven
carry rings. Their outputs enter before directly, because before includes the
final false condition check.

After recursively lowering before:

* false-lane execution is `E_out`;
* `dataflow.gate` projects before execution into after phase;
* false-lane condition arguments become while results;
* true-lane condition arguments become after block values;
* false-lane `W/R` is the loop exit state;
* true-lane `W/R` enters after.

After results feed the next before activation. A false condition consumes no
dummy feedback.

The final false before effects are therefore visible at loop exit. A read in
that final before activation updates `R_out`; a following write outside the
loop must wait for it. For `K = 0`, before still executes once, after does not
execute, and the first before effects remain in the outgoing frontier.

If the condition never becomes false, no execution exit, frontier exit, or
while result is produced.

## 9. Nested Composition

Nesting uses only function composition of `lower_region`:

* an inner while exit becomes the outer for body result and recurrence
  feedback;
* an inner for is lowered entirely within the selected execution and frontier
  lanes of an outer if;
* deeper combinations repeat the same rules without dedicated pairwise
  lowering paths.

The parent consumes only the child's execution, yielded values, and frontier
pair. It does not reach into child leaves to reconstruct a tail.

## 10. Parallel Failure Boundary

Residual `scf.parallel` or `scf.forall` is checked across every graph before
the pass mutates any graph. Unscheduled input fails with a diagnostic that a
selected schedule and provenance are required. Scheduled but still residual
parallel SCF also fails and must be normalized by its owning transformation.

The graph-region owner does not:

* infer a width or ownership domain;
* serialize the region;
* unroll it;
* choose reduction order;
* use traversal order as a hidden schedule.

## 11. Graph Boundary

This slice deliberately does not define `dataflow.graph.return` completion.
The pass emits internal structural and memory frontiers but cannot claim that
the existing graph ABI has a complete retirement frontier.

In particular:

* memory `done` is not substituted into the return's leading `none` operand;
* the removed graph-sync pass is not a hidden completion authority;
* no effect scan retroactively defines graph retirement;
* graph launch retirement closure and verifier coverage remain separate work.

The current legacy terminator operand is preserved mechanically, but neither
`%start` nor any partial execution or memory frontier is valid completion
evidence. Integrating this lowering requires a separately confirmed graph
retirement rule that also covers stateful close and reset.

## 12. Supported Failure Modes

The owner rejects before mutation when:

* raw parallel SCF reaches a graph;
* an effectful or unmodeled nested operation reaches a graph;
* a memref leaf is not rank one;
* structured control carries a memref result or memref loop state;
* the graph entry lacks the leading `none` execution value.

Unsupported top-level LLVM accesses may remain unchanged under the existing
address-normalization contract. Supported LLVM load, store, memcpy, and
memset forms are normalized before recursive region lowering, after which the
same frontier rules apply. An unsupported effectful operation inside a
structured region must fail closed instead of being hoisted.

## 13. Non-Goals

This lowering does not define:

* graph ABI redesign or retirement verification;
* Dataflow ODS changes;
* vector, masked, gather, or scatter memory ports;
* range-sensitive or polyhedral alias partitioning;
* cross-graph partition identity;
* Fabric memory banks, ports, services, or contention;
* DFG simulator, TechMapping, or PnR behavior;
* parallel schedule selection.

Those concerns must consume the explicit canonical event network or be owned
by an earlier transformation. They must not rebuild memory order from source
text order, simulator traversal, or physical placement.
