# Fabric Resource Contract

This document defines the shared typed atoms used when a Fabric resource has
state, capacity, atomic acquisition, release, or arbitration. It removes
duplicated resource-state conventions without introducing a generic
`fabric.resource` operation or a second hardware graph.

## Ownership And Non-Artifact Scope

The atoms in this document are embedded by concrete Fabric resources. They are
not artifacts, independently addressable entities, or an extension registry.
The owning resource specification defines its state kinds, requesters, claims,
and legal parameter values.

```text
ResourceContract {
  states[]
  resource_transitions[]
  use_patterns[]
  requester_order
  grant_policy?
}
```

One resource owns one complete contract. Mapping references its typed
structural keys and selects only declared exact refinements. Simulation,
runtime, and RTL implement the same contract rather than reconstructing local
schedulers.

## ResourceState

```text
ResourceState {
  state_key
  initial_value
  capacity_dimensions[]
}
```

`state_key` is owner-defined and closed. The initial value is canonical and
must describe the all-free or otherwise explicitly declared reset state.
Capacity dimensions use typed integer units owned by the resource; free-form
names and property maps are forbidden.

Dynamic occupancy is execution state and is not persisted. A stateful resource
must return to its canonical reusable state under its declared close/reset
contract before a conflicting invocation can reuse it.

## Resource Commit Transition

```text
ResourceTransition {
  transition_key
}
```

A resource transition is one closed, owner-defined atomic relation over the
resource's typed dynamic state and the accepted request. The concrete resource
specification owns its exact pre-state, request, result, and post-state
relation. The shared contract stores only the typed key and never admits a
predicate DSL, callback, property bag, or implementation-private mutation.

The relation may atomically update several states and capacity dimensions. It
is not a capacity claim: state produced by one committed use may remain until a
later use commits its own transition. No later use releases or inherits an
earlier use's claim.

`transition_key` is scoped to the concrete resource owner's closed transition
inventory and is embedded by its UsePattern. It is not an independently
addressable Fabric entity or persistent reference. Consumers resolve it only
through the exact `FabricUsePatternRef` and owning Fabric artifact.

## Atomic UsePattern

```text
UsePattern {
  pattern_key
  eligibility
  claims[]
  acquire_event
  release_event
  commit? {
    event
    resource_transition
  }
  timing_and_progress
}
```

A pattern is one atomic resource use. Its claims cannot be split by Mapping or
runtime. All claims are acquired together at `acquire_event` and the complete
claim envelope returns together at `release_event`. Claims therefore represent
temporary reservations only; durable occupancy, queue contents, cursors, and
logical resource state are changed only by the optional commit transition.

When a commit is present, its one owner-defined transition is applied atomically
at its exact event. The commit event may equal the acquire event. Version 1.0
admits no cancellation after acquisition: the declared timing and progress
contract must lead an accepted use through its commit, when present, and claim
release. A resource that needs cancellation or rollback requires a future
closed contract rather than a private convention.

The timing contract must order `acquire_event <= commit.event <=
release_event`, where equality denotes one atomic event. A pattern without a
commit must still order acquisition no later than release. An owner declaration
that cannot establish this order is invalid.

The concrete resource validator must also prove that every eligible transition
and every explicitly admitted concurrent commit set preserves the owner's state
invariants and capacity bounds. This proof is resource-specific; the shared
contract does not infer queue, cursor, payload, or state-machine behavior from
numeric capacity alone.

Eligibility, transition semantics, and claim parameters are typed by the
owning resource. The contract states acquisition, commit, release, and every
Mapping-visible timing, capacity, backpressure, and progress guarantee.

Internal implementation transactions may refine one accepted use only when
they preserve the declared external firing, retirement, ordering, and progress
semantics. They cannot acquire another claim envelope, apply another resource
transition, or become additional software actors or Mapping uses.

## Requester Ordering And GrantPolicy

Requester identity is a closed typed structural reference owned by the
resource. The resource stores one exact requester sequence only when its order
is semantically observable by arbitration.

The first profile permits:

```text
GrantPolicy =
    FixedPriority {
      requester_order
    }
  | RoundRobin {
      requester_cycle
      reset_cursor
      advance = OnSuccessfulGrant
    }
```

`FixedPriority` grants the first eligible requester in the exact permutation.
`RoundRobin` scans the exact cycle from the current cursor and advances only
after a successful grant. Reset establishes `reset_cursor`.

The policy may be absent only when the verifier proves that no two requesters
can be simultaneously eligible for the same capacity. A default priority,
authoring order, map iteration order, or simulator arrival race is forbidden.
Additional policies require a real hardware and execution contract; they are
not admitted through a predicate DSL.

## Resource-Specific Composition

Concrete resources embed only the atoms they need:

* a stateless boundary uses one atomic transfer pattern and no state,
  transition, or grant policy;
* a spatial PE may have statically disjoint use patterns and no grant policy;
* a temporal PE uses instruction-context requesters and declared state banks;
* a switch uses transfer-pattern requesters and a declared arbitration policy;
* a memory engine uses operation-row and service requesters; and
* a system transport resource uses transfer or service-leg requesters.

Memory operation rows, FU configured graphs, temporal-PE register-file
dependencies, and other internal realization witnesses may absorb software
edges. The owning resource records that explicit relation. The shared resource
contract does not infer or manufacture absorption.

## Verification

Verification rejects:

* duplicate or unknown state, pattern, requester, or claim keys;
* duplicate or unknown resource-transition keys;
* noncanonical initial state or overflowing capacity;
* a use pattern with an undeclared claim or ambiguous release;
* a commit that names an undeclared transition or event;
* a timing contract that does not order acquire, optional commit, and release;
* an owner transition that can violate state or capacity invariants;
* contention without an exact grant policy;
* a grant policy that omits, duplicates, or foreign-references a requester;
* a Mapping attempt to split an atomic use; and
* an implementation whose timing, release, or grant behavior differs from the
  declared contract.

Anchor tests cover one disjoint resource without arbitration, fixed-priority
contention, round-robin reset and successful-grant advancement, one stateful
use whose short-lived claim and durable commit transition are distinct, and
one resource-specific internal transaction decomposition. They do not create
a cross-product fixture for every concrete resource.
