# Dataflow Memory Consistency

This document is the semantic authority for plain, atomic, volatile, and
fence actors in a Canonical Dataflow Program. It defines software-visible
access contracts, firing and retirement, vector atomic granularity, and the
boundary between static causal edges and dynamic consistency relations.

Compiler lowering owns construction of ordinary `none` event edges. Fabric
owns hardware capability. Mapping selects and proves a realization.
ConfigurationABI owns physical encoding. Simulation and runtime execute the
selected contract. None of those consumers may copy or reinterpret the
software access contract defined here.

## Access Contract

The common atomic-access fields are one closed typed value:

```text
AtomicAccessContract {
  ordering : AtomicOrdering
  sync_scope : SyncScopeRef
  vector_granularity : absent | WholePayload | PerLane
  volatile : bool
}

MemoryAccessContract =
    Plain {
      volatile : bool
    }
  | Atomic(AtomicAccessContract)

AtomicRmwContract {
  kind : AtomicRmwKind
  access : AtomicAccessContract
}

CompareExchangeContract {
  success_ordering : AtomicOrdering
  failure_ordering : AtomicOrdering
  sync_scope : SyncScopeRef
  vector_granularity : absent | WholePayload | PerLane
  weak : bool
  volatile : bool
}

FenceContract {
  ordering : AtomicOrdering
  sync_scope : SyncScopeRef
}
```

Every canonical memory actor owns exactly one applicable aggregate contract.
Its fields are orthogonal semantic facts, but the aggregate is their only
persistent owner. There are no independent ordering, scope, atomicity, RMW
kind, weakness, volatility, MMIO, or coherence attributes that can disagree
with it. Scalar atomic accesses omit `vector_granularity` because both vector
cases degenerate to one atomic object.

`AtomicOrdering` has exactly the LLVM semantic values `unordered`,
`monotonic`, `acquire`, `release`, `acq_rel`, and `seq_cst`. Source-language
`relaxed` therefore normalizes to `monotonic`; Dataflow does not add a second
ordering vocabulary. Operation verification enforces the LLVM legality
matrix. In particular, an atomic load rejects `release` and `acq_rel`, while
an atomic store rejects `acquire` and `acq_rel`.

`SyncScopeRef` is the closed sum:

```text
SyncScopeRef = System | SingleThread | TargetScopeKey
```

`TargetScopeKey` is a typed target-namespaced key, never a free unscoped
string. The compiler-target contract uniquely owns its meaning. Lowering must
resolve an LLVM target-specific sync scope to such a key; an unresolved scope
cannot enter a SpatialCore graph and remains InstructionCore code or causes
candidate rejection. A later CompilerTargetBinding selects a compatible
physical target but does not redefine the scope.

## Canonical Actors

`dataflow.load` and `dataflow.store` retain their existing scalar and vector
shapes. Their `MemoryAccessContract` distinguishes plain, atomic, volatile,
and atomic-volatile behavior without multiplying operation names.

The additional canonical actors are:

```text
dataflow.atomic_rmw
  consume: memory binding, address, update value, optional mask, ctrl
  produce: old value, done
  static: one AtomicRmwContract

dataflow.cmpxchg
  consume: memory binding, address, expected, desired, optional mask, ctrl
  produce: old value, success, done
  static: one CompareExchangeContract

dataflow.fence
  consume: ctrl
  produce: done
  static: one FenceContract
```

The RMW kind is a closed typed enumeration with exact LLVM `atomicrmw`
semantics. A generic atomic region may normalize to this actor only when its
body is proven equivalent to one enumerated operation. Otherwise an earlier
structured transform must form a legal compare-exchange loop, keep the code
on the InstructionCore, or reject the candidate. Canonical Dataflow does not
carry a generic atomic-region DSL.

`cmpxchg` preserves distinct success and failure orderings and strong versus
weak behavior. Its verifier applies LLVM's failure-order restrictions. The
old value, success result, and done event are one retirement publication. A
scalar or `WholePayload` compare-exchange produces one `i1` success result. A
`PerLane` compare-exchange produces a success vector with the exact access
shape and `i1` elements. An RMW old value and done event are likewise one
retirement publication.

## Issue, Linearization, And Retirement

An atomic memory actor has the same three semantic moments as a plain access:

```text
issue -> linearize -> retire
```

Issue consumes all dynamic operands and the unique control event. At
linearization, the selected consistency domain performs one whole-payload
atomic action or the declared set of per-lane atomic actions. Retirement
publishes all actor results and done together.

An atomic store or successful compare-exchange modifies its atomic object's
modification order. An atomic RMW reads the immediately preceding value in
that order and appends its write as one indivisible action. A failed
compare-exchange performs no write. Weak compare-exchange may fail
spuriously. Atomic load, failed compare-exchange, and RMW read-from choices
obey the exact ordering and scope contract.

Actor `done` is not a global barrier. It denotes completion under this
actor's access contract. A static causal edge does not make a plain access
atomic, and an Atomic contract does not introduce an unstated causal edge.

## Static And Dynamic Order

The final Structured Program Candidate owns the selected single-strand order
after all legal SCF-stage transformations. Mechanical lowering preserves only
the observable and memory-model constraints of that order as ordinary
`none`-typed event edges.

The semantic relations are:

```text
sequenced-before   compiler-known order in one logical source strand
synchronizes-with  dynamic relation formed by compatible release/acquire
                    operations, scopes, and reads-from choices
happens-before      transitive closure of sequenced-before and
                    synchronizes-with
```

Atomic modification order, reads-from, synchronizes-with, and the global
sequentially-consistent order are dynamic consistency-domain state. They are
not new Dataflow edges, Mapping records, or simulator traversal order.

Mechanical lowering must preserve these local requirements:

* atomic actors and fences in one logical source strand retain the selected
  Structured Program Candidate's sequenced-before order;
* volatile actors in one logical source strand retain their relative order;
* a release operation or fence waits for prior memory effects whose visibility
  it publishes;
* an acquire operation or fence causally precedes later memory effects whose
  visibility it constrains;
* `acq_rel` applies both directional rules;
* `seq_cst` applies both directional rules and participates in the dynamic
  global sequentially-consistent order; and
* an atomic-volatile actor participates in both the atomic/fence and volatile
  strand relations.

These rules do not create a global chain across dynamic `dataflow.thread`
instances. Cross-instance order comes only from compatible atomic operations,
fences, source/runtime synchronization, and the selected consistency domain.

The existing per-alias `(write_frontier, read_frontier)` pair remains the sole
canonical analysis state for RAW, WAR, and WAW hazards within one alias
partition. Cross-partition sequenced-before requirements above are an
independent semantic relation. Compiler implementations may compress that
relation with disposable all-effect, atomic/fence, volatile, and acquire
frontier caches. Cache shape is not IR or Artifact schema. Only the resulting
deduplicated and transitively reduced ordinary event edges are published.

## Vector Atomic Granularity

`WholePayload` means the complete data payload is one atomic object. The
initial contract admits it only for an `element` access with one logical
address and no dynamic mask. A vector-valued memory element may therefore be
atomically accessed as one value when its exact type, width, alignment, and
hardware capability are supported.

`PerLane` means every active lane is an independent atomic object. No lane
order is implied. Inactive lanes perform no access. The actor retires once all
active lanes have completed and assembles value and success results in
canonical row-major lane order. Repeated active addresses are legal; their
actions participate in the addressed object's modification order rather than
acquiring a hidden lane order.

A vectorized collection of independent scalar atomics uses `PerLane`. A
source whole-vector atomic uses `WholePayload`. The compiler may not switch
between these contracts merely because payload widths are equal. Physical
lanes, beats, and Physical Tags do not change the software granularity.

## Volatile And MMIO

Volatile is an observability contract, not synchronization. A volatile access
preserves its dynamic operation count and its order relative to other volatile
operations in the same logical source strand. It does not create a
cross-thread synchronizes-with relation and does not make a plain access
atomic.

Compiler, Mapping, and implementation must not speculate, eliminate,
duplicate, merge, or reinterpret a volatile actor. A Fabric use pattern may
use internal beats or protocol retries only when the selected provider
contract exposes one at-most-once logical operation with the same observable
width, type, and side effects. Provider-visible splitting or replay is
illegal.

MMIO is a property of the bound logical and physical service contract, not a
Dataflow operation name or access-ordering value. A volatile access to an MMIO
binding therefore uses the same actor surface as a volatile access to storage.
The binding and selected service must prove compatible observable behavior.
If the bound range may trap or requires an unsupported fault protocol, the
operation remains InstructionCore code rather than acquiring hidden graph
exception semantics.

Coherence is likewise not a Dataflow actor attribute. Fabric owns consistency
and coherence capability, Mapping binds the software requirement to it, and
the selected implementation or simulator executes its dynamic state.

## Derived Views And Downstream Ownership

`CanonicalMemoryAccessView` remains a nonpersistent projection of the exact
addressed actor. In addition to shape, lane, address, data, and mask facts, it
projects the actor's exact typed contract and vector atomic granularity when
present. It does not copy those fields into an independently serialized
record. Fence has no addressed-access view.

The Canonical Service Schema owns each operation kind's argument, result,
effect, completion, and parameter legality once. Concrete ordering, scope,
volatility, and granularity values remain owned by the Dataflow actor.
Fabric capabilities declare accepted domains. Mapping records only the
selected correspondence and proof witness. Runtime requests and simulator
events derive the selected values and cannot become another authority.

## Verification Anchors

Anchor-level verification covers only stable semantic boundaries:

* invalid load, store, RMW, compare-exchange, and fence ordering combinations;
* unresolved target-specific synchronization scope rejection;
* one relaxed histogram update per dynamic worker without a static global
  worker chain;
* release publication and acquire consumption across different alias roots;
* relative ordering of volatile accesses to different addresses without
  ordering unrelated nonvolatile accesses;
* whole-payload versus per-lane vector atomic behavior and repeated per-lane
  addresses;
* atomic result and done publication as one retirement event; and
* rejection of an MMIO mapping that cannot preserve one at-most-once logical
  volatile operation.

Tests do not enumerate every ordering cross-product, hardware protocol,
vector shape, or RMW kind. They do not freeze textual assembly, transient
frontier-cache shape, simulator scheduling, or physical transaction count.
