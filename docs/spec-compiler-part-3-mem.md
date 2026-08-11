# Loom Compiler Part 3 Memory Frontier Lowering

This document is the memory-order source of truth for graph-local SCF to
Dataflow lowering. The concrete owner is `loom-lower-graph-memory`; it
normalizes supported memory leaves and recursively lowers structured graph
regions in one traversal.

The Dataflow operation contracts remain owned by the Dataflow specifications.
This document defines only the compiler analysis state and the ordinary SSA
event network produced from it.

The resulting canonical memory actors and their explicit `ctrl` and `done`
network are canonical software semantics. Their operation contracts are owned
by `docs/spec-dataflow-memory-consistency.md` and
`docs/spec-dataflow-vectorization.md`. TechMapping, SpatialMapping, and
SystemMapping may realize that network on Fabric resources, but they must not
reconstruct missing memory order from source order, graph text order,
traversal, or physical placement. The downstream realization boundary is
specified by `docs/spec-mapping-memory.md`.

## 1. Scope

The lowering contract covers:

* scalar and fixed-ranked vector forms of canonical `dataflow.load` and
  `dataflow.store`, including the masked contiguous and gather/scatter forms
  defined by `docs/spec-dataflow-vectorization.md`;
* canonical atomic load/store, `dataflow.atomic_rmw`,
  `dataflow.cmpxchg`, `dataflow.fence`, and volatile access contracts defined
  by `docs/spec-dataflow-memory-consistency.md`;
* normalized scalar `memref.load` and `memref.store` leaves over a canonical
  linear memory space;
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
lower_region(E_in, values_in, {W_in[p], R_in[p]}, SB_in)
  -> (E_out, values_out, {W_out[p], R_out[p]}, SB_out)
```

`E` is execution permission and structural completion. `W` and `R` are
memory-order frontiers for alias partition `p`. They share the ordinary
`none` SSA type but remain semantically distinct throughout lowering.
`SB` is the path-sensitive analysis relation containing only
sequenced-before obligations that remain observable after the selected
Structured Program Candidate's legal transformations. It covers atomic/fence,
volatile, release, and acquire requirements across alias partitions. It is not
one serialized token or an IR object.

The contract is an implementation function, not an IR object. Canonical IR
does not contain partition ids, dependence snapshots, compound-region
objects, chain-scope attributes, memory tokens, sequenced-before records, or
memory-specific join operations.

The recursive owner replaces the former split among reduction, invariant,
control, and sync passes. No later pass reconstructs structured memory order
or graph completion from a hidden effect scan.

## 3. Basic Alias Partitions

Partition identity is local to one `dataflow.graph` lowering run.

A canonical root is found by peeling an accepted side-effect-free memref view
until reaching an explicit storage or boundary root. The finalized surface
recognizes:

* a graph memory input, whose root identity comes from its launch binding;
* a fresh `memref.alloc` result, whose root is unique for each invocation;
* a verified side-effect-free view that preserves the source root. The initial
  accepted set contains `memref.cast`; adding another view form requires one
  matching root, region, and simulator contract before admission.

Graph launch memory bindings require exact memref capability types. An LLVM
pointer cannot bind a graph memref through a conversion, inferred base, or
special address-space-zero rule. SCF optimization may first prove and
materialize a rooted memref capability plus integer offset, or it may retain
the pointer as a value consumed by a `PointerAddressed` memory actor together
with an independently bound service capability. Neither path materializes a
graph-body bridge. `builtin.unrealized_conversion_cast` is never a canonical
root, view, actor, or boundary bridge.

The Canonical Dataflow finalizer assigns one `LogicalMemoryRootRef` to each
static imported-memory formal role and canonical fresh-allocation definition.
An imported graph memory argument does not create a competing root: its exact
`dataflow.graph.launch` binding resolves through root-preserving views to the
upstream static role. A fresh allocation result is the root-defining value.
View operations remain typed structural relations and receive no root ID of
their own.

Persistent consumers use the closed forms owned by
`docs/spec-compiler-part-3-dfg.md`: `LogicalMemoryViewRef`,
`LogicalMemoryRootOrViewRef`, and `MemoryExposureRef`. This document does not
redeclare their wire variants.

The root-local inventory resolves every admitted static view to its unique
root-preserving relation. Reusing one graph under different roots creates
separate structural view references in those root inventories rather than a
view entity. A memory exposure identifies one launch-contextual graph memory
result. It describes a provided capability boundary, not a token producer or
an addressed memory operation.

This persistent reference identifies a static software role. Runtime object
identity is derived separately: an import is bound through the exact launch
and runtime memory registry, while a fresh allocation combines its static root
reference with the graph invocation occurrence. Two imported roles may resolve
to one runtime object through explicit alias topology without merging their
static IDs. Partition identity below remains local analysis state and is not
the persistent root catalog.

A memory input binds an established external memref capability through an
exact graph-launch type match. An LLVM pointer never satisfies a graph memory
port. A first-class pointer value used by a `PointerAddressed` actor resolves
through the runtime object registry to one object and byte offset independently
of the service-capability binding.

Distinct graph memory inputs are conservatively may-alias unless explicit
no-alias evidence distinguishes them. Distinct fresh allocations are
independent roots. The analysis does not use address ranges, affine
disjointness, bank identity, physical ports, or element-type compatibility to
split a root.

`memref.get_global`, `memref.alloca`, globals, static pointer bases, and
unrecognized capability producers are not canonical roots. A pre-final
analysis may conservatively group an unresolved access while building an event
network, but finalization rejects any such residual producer rather than
granting it an external-memory authority.

A source-origin `llvm.alloca` accepted by the Structured
`PromoteOrderedBufferToChannel` decision is not an exception to this rule. That
decision must remove the complete proved allocation closure before D0; a
residual allocation or pointer use remains non-canonical and is rejected.

Access-to-partition membership is kept in a transient operation map before
SCF operands are projected. Selector demuxing must not change alias identity.
The map is discarded after explicit event edges are emitted.

## 4. Canonical Frontier State

Each partition has exactly two alias-hazard analysis values:

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

Cross-partition sequenced-before requirements do not add another component to
each alias partition. The implementation may use disposable all-effect,
atomic/fence, volatile, and acquire frontier caches to compress `SB`, but the
required relation is the authority. Cache shape, traversal order, and
intermediate joins are not observable and are discarded after event edges are
published.

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

Atomic load uses the read equation and atomic store uses the write equation.
Atomic RMW and compare-exchange conservatively use the write equation because
each firing may both read and write; a failed compare-exchange may retain the
resulting causal edge without inventing a write. Fence has no alias-partition
read or write effect.

In addition to alias hazards, lowering materializes the sequenced-before rules
from the actor contracts:

* atomic actors and fences in one logical source strand preserve their
  selected order;
* volatile actors in one logical source strand preserve their relative order;
* release actors and fences wait for prior memory-effect tails whose
  visibility they publish;
* acquire actors and fences precede later constrained memory effects;
* `acq_rel` and `seq_cst` apply both directions; and
* atomic-volatile actors participate in both strand relations.

These ordinary event edges preserve local ordering only. Reads-from,
modification order, synchronizes-with, and the global sequentially-consistent
order remain dynamic consistency-domain state. Different dynamic thread
instances are not joined by a compiler-created global frontier.

One vector addressed memory actor is one canonical firing. Its active lanes do
not create independent frontier records or an implicit lane order.
`P(access)` is the conservative union of alias partitions that any active lane
may access. A dynamic mask or address vector cannot weaken that set merely
because one observed execution disables a lane. A statically proven all-zero
mask may be simplified by an ordinary semantics-preserving Dataflow rewrite;
otherwise the firing retains its explicit `ctrl` and `done` obligations.

The vector operation's owning semantic contract determines lane activity,
inactive-load fill, duplicate-gather behavior, and rejection or explicit
ordering of duplicate scatter addresses. This lowering only projects the
whole firing through the same `(W,R)` equations as a scalar access. It does
not define a second vector-memory ordering model.

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
* project the required `SB_in` tails through the same branch selector;
* recursively lower each branch;
* mux results, execution, `W`, `R`, and path-sensitive `SB` tails with the same
  condition.

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

Required sequenced-before tails use the same condition-driven recurrence
mechanics when they cross an iteration. This does not serialize unrelated
plain accesses or create a persistent loop-order object.

Each ring uses `dataflow.carry` under the loop selector. A matching
`dataflow.demux` sends true-lane values into the body and the false-lane value
to loop exit. Captured non-memory values are replayed with
`dataflow.invariant` and projected into body phase with `dataflow.gate`.
Memref capabilities are not replayed.

The recursively lowered body supplies all recurrence feedback values. The
execution feedback is the body's structural exit; memory feedback is the
body's resulting frontier pair and any path-live sequenced-before tails. The
rings are independent even when a write assigns the same `done` to both memory
components.

For zero trip count, the stream emits only `F`. No body address or access
fires. Every carry exposes its init value on the false lane, so source values,
execution, `W`, `R`, and `SB` transfer through the loop unchanged.

No dependence is removed because a loop appears parallelizable. Source
iteration order remains authoritative until an earlier transformation has
materialized a different schedule with provenance.

## 8. `scf.while`

For a while loop whose after region executes `K` times:

* before executes `K + 1` times;
* after executes `K` times;
* the before condition stream is `T^K F`.

Execution, source inits, touched `W/R` components, and path-live `SB` tails use
condition-driven carry rings. Their outputs enter before directly, because
before includes the final false condition check.

After recursively lowering before:

* false-lane execution is `E_out`;
* `dataflow.gate` projects before execution into after phase;
* false-lane condition arguments become while results;
* true-lane condition arguments become after block values;
* false-lane `W/R` is the loop exit state;
* true-lane `W/R` enters after;
* false-lane `SB` tails leave the loop; and
* true-lane `SB` tails enter after.

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

The parent consumes only the child's execution, yielded values, frontier pair,
and path-live sequenced-before tails. It does not reach into child leaves to
reconstruct a tail.

## 10. Parallel Transfer Boundary

Residual `scf.parallel` or `scf.forall` is checked across every graph before
the pass mutates any graph. Raw or unowned parallel input fails. A fixed finite
parallel region is accepted only when its Structured Program Candidate owns a
typed, verifier-proven `P[]` schedule and the recursive transfer can derive one
complete frontier relation for that exact domain. The lowering must not trust
the mere presence of string-named attributes as proof.

Until the typed producer and verifier establish this provenance, the boundary
fails closed. Forged, malformed, foreign-owner, or domain-mismatched
provenance is invalid even when the residual SCF shape is otherwise supported.
Part 3 consumes the selected schedule; it does not choose parallel width,
serialization, ownership, or reduction order.

The graph-region owner does not:

* infer a width or ownership domain;
* serialize the region;
* unroll it;
* choose reduction order;
* use traversal order as a hidden schedule.

## 11. Graph Boundary

`dataflow.graph.return` is a structural graph-boundary declaration, not an
implicit runtime return. Its operand segments are:

```text
values(...) streams(...) memories(...) complete(...)
```

`complete` is mandatory, non-empty, variadic, unordered all-of, and contains
only `none` values. The launch-facing done event is exactly:

```text
launch.done = all_of(graph.return.complete)
```

There is no hidden effect scan, graph-quiescence test, or removed sync pass
that can define completion independently.

A memory result in the `memories` segment is a `MemoryExposureRef`. Returning
the capability does not issue a memory service operation and therefore creates
no request, response, or completion leg. Mapping may bind the exposure to a
provider boundary, but the actual service legs remain owned by the addressed
memory actors that later use the capability.

After canonical publication, TechMapping may classify an explicit edge as
realization-internal or external. SpatialMapping and SystemMapping may select
the physical mechanisms that preserve it. No Mapping profile deletes, infers,
or replaces the canonical load/store `ctrl` and `done` obligations. The
Canonical Dataflow Program remains the memory-order source of truth.

After recursive lowering, this pass constructs the memory-owned retirement
frontier from:

```text
execution_out
read_frontier_out[p] for every live alias partition p
terminal observable sequenced-before tails
existing explicit non-start completion obligations
```

The candidate set is deduplicated and causally transitively reduced. A graph
with no derived work and no value publication may retain `%start`. Once a
derived execution or memory frontier exists, the provisional `%start` witness
is removed. The reduced frontier is joined to one publication base. Each
transportable scalar value is passed through a `(none, T) -> (none, T)`
`dataflow.sync` with that base; the returned values use the typed outputs and
the `complete` segment is the all-of of the `none` outputs. Pointer and memref
capability payloads remain boundary bookkeeping whose establishment is covered
by the structural and memory frontier rather than by a transport sync. With no
scalar value outputs, the reduced `none` frontier is written directly to
`complete`.

The frontier's causal closure covers final values, stream boundary close and
commit obligations, memory capability establishment and promised visibility,
all observable side effects, invocation-local state close/reset, and all
non-detached async work. This pass contributes structural execution and final
per-partition read frontiers; stream, exported-memory, and other async
producers must contribute their explicit completion witnesses through the same
segment. It never reconstructs them from operation order or effect metadata.
Memory exports preserve an imported root or view, or expose a fresh allocation
root. Every export retains a memref result payload. Exports do not copy
contents and do not add a memory token; completion only carries the promised
visibility and retirement obligation.

## 12. Supported Failure Modes

The owner rejects before mutation when:

* raw or unverifiably owned parallel SCF reaches a graph;
* an effectful or unmodeled nested operation reaches a graph;
* a residual LLVM load, store, memcpy, memmove, or memset remains after
  normalization and therefore has no explicit completion event;
* a source memory access has not been normalized to the canonical linear
  memory-space form required by its scalar or vector Dataflow actor;
* structured control carries a memref result or memref loop state;
* the graph entry lacks the leading `none` execution value.

LLVM memcpy, memmove, and memset intrinsics are expanded into their exact
structured loop semantics before ownership selection. Supported LLVM load and
store forms are then normalized before recursive region lowering, after which
the same frontier rules apply. Every residual raw LLVM memory operation fails
closed. The finalized-graph gate also rejects residual
`memref.load`/`memref.store`, `memref.get_global`, raw pointer arithmetic,
pointer-bearing operations, `builtin.unrealized_conversion_cast`, and unknown
memory-capability producers. An unsupported effectful operation inside a
structured region must likewise fail closed instead of being hoisted.

## 13. Non-Goals

This lowering does not define:

* vector lane behavior, masks, gather/scatter address semantics, or duplicate
  scatter policy, which are owned by `docs/spec-dataflow-vectorization.md`;
* range-sensitive or polyhedral alias partitioning;
* cross-graph partition identity;
* Fabric memory banks, ports, services, or contention;
* runtime selection of graph stream or memory boundary bindings;
* whole-graph causal-closure proof for arbitrary hand-authored frontiers;
* parallel schedule selection.

Physical vector ports, byte enables, coalescing, banking, and memory-service
selection belong to Fabric and Mapping. Those concerns must consume the
explicit canonical event network or be owned by an earlier transformation.
They must not rebuild memory order from source text order, simulator traversal,
or physical placement.

TechMapping and physical memory realization are specified by
`docs/spec-mapping-artifact.md` and `docs/spec-mapping-memory.md`; this compiler
spec does not duplicate their records or search rules.
