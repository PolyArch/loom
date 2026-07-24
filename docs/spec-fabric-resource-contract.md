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

## Atomic UsePattern

```text
UsePattern {
  pattern_key
  eligibility
  claims[]
  acquire_event
  release_event
  timing_and_progress
}
```

A pattern is one atomic resource use. Its claims cannot be split by Mapping or
runtime. Eligibility and claim parameters are typed by the owning resource.
The contract states when all claims are acquired, when they release, and every
Mapping-visible timing, capacity, backpressure, and progress guarantee.

Internal implementation transactions may refine one accepted use only when
they preserve the declared external firing, retirement, ordering, and progress
semantics. They do not become additional software actors or Mapping uses.

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
* noncanonical initial state or overflowing capacity;
* a use pattern with an undeclared claim or ambiguous release;
* contention without an exact grant policy;
* a grant policy that omits, duplicates, or foreign-references a requester;
* a Mapping attempt to split an atomic use; and
* an implementation whose timing, release, or grant behavior differs from the
  declared contract.

Anchor tests cover one disjoint resource without arbitration, fixed-priority
contention, round-robin reset and successful-grant advancement, and one
resource-specific internal transaction decomposition. They do not create a
cross-product fixture for every concrete resource.
