# Fabric Memory

Shared `ResourceState`, atomic `UsePattern`, requester ordering, and
`GrantPolicy` meanings are owned by `docs/spec-fabric-resource-contract.md`.
This document owns their memory-specific state keys, claims, requesters, and
timing.

## Purpose

`fabric.mem` is the Fabric realization for canonical addressed memory actors
and fences. It describes physical memory-operation capability and
memory-service capability. It does not describe a software address space,
choose a workload binding, or own configured Mapping state.

A fully elaborated occurrence is one physical memory-capability occurrence.
Its orthogonal parts are:

```text
fabric.mem occurrence
  optional Operation Engine Capability
  optional Local Memory Service
  zero or more manager/requester endpoints
  zero or more subordinate/provider endpoints
  configurable service dispatch capability
```

The Operation Engine and Local Memory Service are independently present. Loom
does not introduce `fabric.storage`: local storage remains a typed subresource
of `fabric.mem`, while a pure forwarding or routing resource belongs to the
explicit Fabric transport or service-transform model.

The Local Memory Service, when present, is one physical service and contention
domain. Manager endpoints may reach other services; their capacity and
contention remain owned by those selected service contracts.

## Memory Spaces And Interfaces

A software memory space and a hardware memory interface are different
objects:

* a logical memory root or view owns software identity, interval, layout,
  aliasing, and lifetime;
* a physical memory service or region owns storage or proxy identity,
  physical range, ordering, visibility, and coherence guarantees;
* a manager/requester endpoint can initiate typed memory operations;
* a subordinate/provider endpoint can accept and serve typed memory
  operations.

An endpoint is a path to a service, not the service or address space itself.
One endpoint may carry several mapped logical memories when its declared
range, context, or translation capability distinguishes them. One logical
memory may use several endpoints. The explicit Mapping records in
`spec-mapping-memory.md` own those sparse many-to-many relations.

Manager and subordinate roles are endpoint-relative. A subordinate result may
feed a manager input to compose services. Neither role implies ownership,
allocation, mutability, or a distinct memory identity.

## Legal Occurrence Forms

The capability model admits these useful forms:

* operation engine without local storage, backed by at least one reachable
  manager service;
* operation engine with a Local Memory Service, with manager and subordinate
  endpoints independently optional;
* storage-only occurrence, with a Local Memory Service and at least one
  subordinate endpoint through which the service is reachable.

An occurrence with neither an Operation Engine nor a Local Memory Service is
not `fabric.mem`. A subordinate endpoint alone is not storage backing. A
manager endpoint alone is useful only when an Operation Engine or explicit
service transform can issue requests through it.

Independently bindable banks are separate `fabric.mem` occurrences. Banking
that is selected only inside the implementation may remain within one Local
Memory Service, but its observable conflicts and guarantees must satisfy that
service's architecture contract.

## Operation Engine

Every operation row declares whether its address input consumes the Dataflow-
owned `RootRelative` or `PointerAddressed` projection. A RootRelative row owns
element-index scaling. A PointerAddressed row consumes the complete pointer
representation and performs no implicit base addition or element scaling; its
manager or local-service binding owns translation from that pointer address
space to the selected service region. One row cannot accept both forms through
a raw integer ordinal.

Memory data ports may carry pointer payloads only when their typed capability
domain admits the exact DataLayout-owned pointer format and full
representation width. A wide bits port alone does not establish pointer
provenance, address-space, or exceptional-value semantics.

An Operation Engine owns one canonical ordered inventory of physical memory
operation ports. Each inventory entry contains exactly:

```text
MemoryOperationPort {
  endpoint_inventory
  non-empty canonical capability_alternatives
  resource_contract
  operation_pattern_semantics
}

MemoryOperationCapability {
  role_to_endpoint_relation
  accepted_actor_contract_domain
  parameterized_access_domain, absent for fence
  non-empty admissible_use_pattern_refs
}
```

The occurrence `function_type` is the sole owner of physical endpoint kinds
and widths. A port's `endpoint_inventory` is an ordered set of those endpoint
ordinals; it does not repeat their types. The embedded `resource_contract` is
the sole owner of that port's ResourceState and UsePattern inventories. The
entry in `operation_pattern_semantics` at each UsePattern ordinal supplies its
memory-specific meaning without repeating claims, timing, or arbitration.
Each capability alternative defines its maximal role-bound endpoint subset; the
selected actor schema and access class mechanically derive which of those
roles are active for one firing. Inactive endpoints consume nothing, produce
nothing, and exert no backpressure. Read, write, RMW, compare-exchange, and fence counts,
physical payload capacities, optional mask capacity, and ResourceState and
UsePattern reference domains are derived from the inventory, alternatives,
function type, and resource contract. Different operation ports in one engine
may have different fixed physical capacities.

One port may expose several capability alternatives only when the same
physical endpoint inventory and ResourceState genuinely implement those
alternatives. This is the memory-operation analogue of hardware sharing: a
shared superset port can admit read and write alternatives, while separate
read and write ports remain separate physical capacity. Within one
alternative, the Cartesian product of accepted actor-contract clauses and
access classes must be a real physical capability for every valid Canonical
Dataflow actor in that product. A physical relation such as plain-vector plus
atomic-scalar, but not atomic-vector, therefore uses two alternatives rather
than an over-admitting product or a predicate. Alternatives with the same
observable actor relation and use pattern are invalid duplicates. A
single-operation port is the degenerate one-alternative case.

The maximal load bundle is:

```text
(address_payload, optional lane_mask, ctrl) -> (data_payload, done)
```

The maximal store bundle is:

```text
(address_payload, data_payload, optional lane_mask, ctrl) -> done
```

The maximal RMW, compare-exchange, and fence bundles are:

```text
(address_payload, update_payload, optional lane_mask, ctrl)
    -> (old_payload, done)

(address_payload, expected_payload, desired_payload, optional lane_mask, ctrl)
    -> (old_payload, success_payload, done)

(ctrl) -> (done)
```

Store does not produce a data result. The mask endpoint is physically present
only when the port can consume a dynamic lane mask. An unmasked actor may use a
mask-capable port with that endpoint inactive; inactive means no consume and no
backpressure and must be proven by the capability relation. A masked actor
cannot use a port without a mask endpoint or an exact declared internal mask
source.

The exact software requirement for an addressed actor is the nonpersistent
`CanonicalMemoryAccessView` derived from that actor. Fabric supports it through
one relation:

```text
SupportsMemoryAccess(port, alternative, access, correspondence) =
  alternative belongs to port
  and the actor's registered OperationSchemaId and exact contract are accepted
  and access form is accepted
  and element-size and flattened-lane domains are accepted
  and mask form is accepted
  and ordered actor roles correspond to physical roles
  and address, data, and mask payloads fit their endpoints
  and alignment and subword-write requirements are accepted
  and at least one declared use pattern realizes the access semantics
```

Fence has no `CanonicalMemoryAccessView`. Its selected capability alternative
instead accepts the exact `FenceContract`, role correspondence, consistency
target, and use pattern.

The access forms are `element`, `contiguous`, and `indexed`. The exact ranked
software type remains in Dataflow. Fabric uses its derived element width,
row-major flattened lane count, address geometry, mask form, and payload
capacities. Equal total data width is not sufficient compatibility:
`vector<4xf32>` and `vector<2xf64>` require different element, lane, address,
and mask geometry even though both carry 128 data bits.

Supported access sizes, alignment, inactive-lane suppression, masked-load zero
fill, all-zero-mask completion, and subword-write behavior are explicit
capabilities. In particular, a narrow store requires byte-enable,
write-strobe, or equivalent declared semantics; zero-extension into a wider
port is not sufficient. A capability cannot claim masked load support by
reading inactive addresses and masking only the returned value.

For human inspection, one addressed port has a mechanically derived geometry
class:

```text
ElementAccessOnly
VectorAccessOnly
ElementAndVectorAccess
```

`ElementAccessOnly` admits only `element` views. `VectorAccessOnly` admits only
views with an explicit contiguous or indexed lane shape.
`ElementAndVectorAccess` admits both. These names are neither persisted fields
nor cost classes. In particular, one `element` may itself have a vector-typed
memref element, so `scalar` is only an informal UI alias for
`ElementAccessOnly`. An occurrence with separate element and vector ports may
derive aggregate support for both, but it is not the same hardware as one
shared hybrid port. Exact port inventory, ResourceState, and UsePatterns remain
the authority, and central Evaluation owns all area, power, frequency, and
performance comparisons.

### Persistent Capability Schema

`loom.fabric 1.0` persists one closed typed relation. It does not persist an
exact actor geometry per alternative and does not replace domain records with
counts or generic integer properties.

The only reusable domain atoms are:

```text
ClosedEnumDomain<E> {
  non-empty sorted unique values from the exact owner enum E
}

UnsignedDomain {
  non-empty sorted disjoint inclusive intervals UnsignedInterval
}

UnsignedInterval {
  lower : uint64
  upper : uint64
}

AlignmentDomain {
  accepted_log2_byte_alignments : UnsignedDomain
}
```

An `UnsignedDomain` merges overlapping or adjacent intervals. A finite set is
represented by singleton intervals, so there is no separate finite-domain
variant. Width and lane domains contain only positive values. Alignment
exponents are in `[0, 63]`; the typed C++ view accepts a byte alignment only
when it is a nonzero power of two and its exact base-two exponent is in the
domain. Empty, unsorted, overlapping, adjacent-but-unmerged, overflowing, or
out-of-range domains are noncanonical and invalid.

The parameterized access relation is:

```text
ParameterizedMemoryAccessDomain {
  non-empty canonical access_classes
}

MemoryAccessClass {
  access_form             : MemoryAccessForm
  element_width_bits      : UnsignedDomain
  flattened_lane_count    : UnsignedDomain
  mask_inactive_pairs     : non-empty sorted unique array<
    (MemoryMaskForm, InactiveLaneSemantics)>
  source_alignment        : AlignmentDomain
  read_subword            : ClosedEnumDomain<ReadSubwordSemantics>
  write_subword           : ClosedEnumDomain<WriteSubwordSemantics>
}
```

`MemoryAccessForm` and `MemoryMaskForm` are the Dataflow-owned enums used by
`CanonicalMemoryAccessView`; Fabric does not declare aliases for them.
`ReadSubwordSemantics`, `WriteSubwordSemantics`, and
`InactiveLaneSemantics` are Fabric-owned closed enums because they state
physical guarantees. Their version 1.0 semantic values and stable Fabric wire
tags are:

```text
ReadSubwordSemantics  = NotApplicable(0) | Exact(1) | ZeroExtend(2)
WriteSubwordSemantics = NotApplicable(0) | Exact(1) | ByteEnable(2)
InactiveLaneSemantics = NotApplicable(0) | Suppress(1)
                      | SuppressAndZeroFill(2)
```

These tags are `loom.fabric 1.0` schema values. They do not inherit C++ enum
ordinals, source declaration order, or printer spelling. A codec must reject
an unknown tag rather than preserve it as an opaque future value.

`mask_inactive_pairs` preserves the semantic correlation between a software
mask form and the physical inactive-lane guarantee. Its mask-form and
inactive-lane domains are derived projections, never independently editable
sets. In particular, admitting both `(Absent, NotApplicable)` and
`(Dynamic, SuppressAndZeroFill)` does not also admit the invalid Cartesian
cross-pairs.

One access class stores one access form because access-form correlation is
structural. A capability that admits several forms stores several classes in
the same alternative; it does not duplicate the alternative, endpoint
relation, actor-contract domain, or use-pattern selection.

The access relation has one fixed field order:

```text
access_form
element_width_bits
flattened_lane_count
mask_inactive_pair
source_alignment_log2_bytes
read_subword
write_subword
```

Its unique normal form is a reduced ordered relation. At each field, the
normalizer partitions accepted values into maximal canonical domains whose
recursively normalized suffix relation is byte-identical. It then emits one
root-to-leaf product for each partition, in lexicographic field-byte order.
`access_form` remains a singleton partition; unsigned fields use maximal
interval unions; enum and pair fields use sorted unique closed sets. A strict
importer recomputes this normalization and rejects a different decomposition,
even when that decomposition denotes the same tuple set.

The persisted result remains the flat `access_classes` array; no decision tree
or predicate language is stored. This fixed reduction gives every finite or
range relation one encoding, preserves cross-field correlations, and avoids
enumerating exact geometries. For example, `{32} x {1,2}` plus `{64} x {1}` is
canonical when the suffix relations differ, while splitting `{32} x {1,2}`
into two singleton-lane classes is rejected. Structural Dataflow invariants
still apply: an `element` view has lane count one; contiguous and indexed
views carry their exact flattened lane counts; payload and mask widths must
fit the selected physical endpoints.

The shared reduced-relation wire is a `u64be` row count followed by framed
rows. A row is a `u64be` field count followed by framed domains. A domain
starts with the stable `u32be` tag `Finite(0)` or `Unsigned(1)`. A finite
domain then stores a `u64be` atom count and framed semantic-owner atom bytes;
an unsigned domain stores a `u64be` interval count followed by `u64be` lower
and upper bounds. All frame lengths are `u64be`. Rows and finite atoms are
strictly increasing by their complete encoded bytes; unsigned intervals use
the canonical order and merging rules above. Unknown domain tags, empty
domains, inconsistent field counts or domain kinds, trailing bytes, and any
decode-and-reencode difference are invalid. The relation codec owns only this
framing; it never interprets or assigns a second encoding to finite atoms.

Dynamic-mask admission has one invariant meaning: inactive addresses are not
evaluated, inactive operations are suppressed, inactive load-like result
lanes are zero-filled, and an all-zero mask completes without a service
request. There is no independent `all_zero_mask_completes` field: a dynamic
mask class that cannot provide this invariant must not be declared.

The exact actor-contract relation is:

```text
MemoryActorContractDomain {
  actor_schema : OperationSchemaId
  non-empty canonical contract_clauses selected by actor_schema
}

LoadStoreContractClause =
    Plain {
      volatile_values
    }
  | Atomic {
      orderings
      sync_scopes
      vector_granularity_values
      volatile_values
    }

AtomicRmwContractClause {
  rmw_kinds
  orderings
  sync_scopes
  vector_granularity_values
  volatile_values
}

CompareExchangeContractClause {
  success_failure_ordering_pairs
  sync_scopes
  vector_granularity_values
  weak_values
  volatile_values
}

FenceContractClause {
  orderings
  sync_scopes
}
```

Enum and boolean fields use non-empty closed finite domains.
`vector_granularity_values` is a sorted unique finite set of the closed sum
`Absent | WholePayload | PerLane`; `Absent` preserves the exact scalar case
and is not a wildcard. Compare-exchange ordering is a sorted unique set of
exact `(success, failure)` pairs rather than the Cartesian product of two
sets. `sync_scopes` is a sorted unique finite set of exact Dataflow
`SyncScopeRef` values; there is no wildcard string or consumer-defined scope
predicate. Every clause variant must match the exact memory-contract variant
owned by its registered actor schema. The actor schema also derives load,
store, RMW, compare-exchange, or fence operation kind, so Fabric persists no
second `MemoryOperationKind` enum.

Actor contracts use the same reduced ordered relation, with clause variant
first and the fields of that clause in their declaration order. Clause variant
is a singleton partition. Compare-exchange success/failure ordering remains
one atomic pair field. The normalizer groups maximal domains only when their
normalized suffix relation is identical, emits clauses in canonical byte
order, and rejects overlapping, duplicated, or differently decomposed
equivalent relations. Authoring order is never semantic.

`source_alignment_bytes` remains part of the exact OperationSchema-owned actor
projection, but its accepted physical domain occurs only in
`MemoryAccessClass::source_alignment`. The actor-contract clause does not
repeat it. Plain access derives alignment one for this compatibility query;
atomic actors use their exact declared source alignment. Thus actor semantics
and physical alignment capability each have one owner.

The Fabric-owned clause tags are stable version 1.0 wire values:

```text
LoadStorePlain(0)
LoadStoreAtomic(1)
AtomicRmw(2)
CompareExchange(3)
Fence(4)
```

The actor-domain record is one framed OperationSchema owner encoding, followed
by a `u64be` clause count and that many framed clause records. A clause record
is its `u32be` Fabric tag, its exact `u64be` field count, and one framed finite
domain per field in the declaration order above. A finite domain is a `u64be`
atom count followed by that many framed bytes from the atom's semantic owner.
A compare-exchange ordering-pair atom is the framed success-ordering owner
encoding followed by the framed failure-ordering owner encoding. All framing
lengths are `u64be`. Unknown tags, wrong field counts, trailing bytes, empty
domains, noncanonical owner atoms, and any byte sequence that differs from a
strict decode-and-reencode are invalid. Fabric assigns no local encoding to a
Dataflow-owned atom.

One persistent port record is exactly:

```text
MemoryOperationPortRecord {
  endpoint_inventory : sorted unique physical endpoint ordinals
  resource_contract  : ResourceContractRecord
  operation_pattern_semantics : array<MemoryOperationPatternRecord>
  capability_alternatives : non-empty canonical array<
    MemoryCapabilityAlternativeRecord>
}

MemoryCapabilityAlternativeRecord {
  actor_contract_domain : MemoryActorContractDomain
  role_to_endpoint : sorted unique array<
    (ServiceValueRole, physical endpoint ordinal)>
  parameterized_access_domain : absent | ParameterizedMemoryAccessDomain
  admissible_use_patterns : non-empty sorted unique UsePatternKey values
}

MemoryOperationPatternRecord {
  transaction_projection : MemoryPortTransactionProjection
}

MemoryPortTransactionProjection =
    Direct(0)
  | ActiveLanesRowMajor(1)
```

The projection tags above are stable `loom.fabric 1.0` wire values. They do
not inherit a C++ enum ordinal or printer spelling, and an unknown tag is
invalid.

The persistent port wire uses the field order above. It begins with a
`u64be` endpoint count and that many `u64be` endpoint ordinals, followed by a
framed `ResourceContractRecord`, a `u64be` operation-pattern count and that
many `u32be` transaction-projection tags, then a `u64be` capability count and
that many framed capability-alternative records. One capability alternative
contains, in order, a framed actor-contract domain, a `u64be` role-binding
count, each framed owner-encoded `ServiceValueRole` followed by its `u64be`
endpoint ordinal, a `u32be` access-domain presence tag (`Absent(0)` or
`Present(1)`) and the framed access domain when present, then a `u64be`
use-pattern count and that many `u32be` owner-local `UsePatternKey` ordinals.
All lengths and counts not otherwise stated are `u64be`. Strict import checks
every frame, owner codec, count, stable tag, ordering, uniqueness, and trailing
byte before decode-and-reencode equality; no host enum layout or raw MLIR
attribute encoding enters this wire.

`ServiceValueRole` is the Dataflow Canonical Service Schema role enum. Fabric
does not declare a second memory endpoint-role enum. The access domain is
absent exactly for the registered fence schema and present for every addressed
schema. Use-pattern keys resolve through the containing port's one resource
contract. Canonical alternative order is lexicographic by complete record
bytes; array ordinal is the
`FabricMemoryCapabilityAlternativeRef` payload. A duplicate, subsumed, empty,
wrong-schema, wrong-role, wrong-endpoint, or use-pattern-free alternative is
invalid.

The complete capability array is one reduced ordered relation over this fixed
field order:

```text
actor schema and exact contract value
addressed access tuple or the fence NoAccess value
canonical role-to-endpoint binding
selected UsePatternKey
```

The finalizer symbolically normalizes the union denoted by all authoring
alternatives, using the actor- and access-domain reducers above and treating
one complete role binding as a singleton value. The persisted alternatives
are the maximal root-to-leaf products of that one relation; the final
UsePattern field is stored as the sorted unique
`admissible_use_patterns` domain. A strict importer recomputes the complete
relation normalization and rejects equivalent decompositions such as splitting
one actor/access domain across several alternatives or factoring one
UsePattern set differently. This reduction operates on interval and finite
domains rather than enumerating every concrete geometry.

`role_to_endpoint` is a role-keyed total function over the maximal role set for
that alternative, not a claim that every listed endpoint fires for every
admitted actor. Every maximal role appears exactly once, and every endpoint is
in the containing port inventory with compatible direction, kind, and width.
Roles
required by the selected OperationSchema are active. `ServiceValueRole::Mask`
is present in the record exactly when at least one access class admits
`MemoryMaskForm::Dynamic`, and is active exactly for a selected dynamic-mask
access; it is inactive for an absent-mask access and then consumes nothing and
exerts no backpressure. Version 1.0 has no other conditionally active role.
The active relation is therefore derived mechanically from the actor schema,
selected access class, and this one binding record; no role predicate or
second configured-role table is persisted.

For a Spatial engine, active untagged role bindings are injective. For a
Temporal engine, active input roles may name the same tagged ingress. This is
not a property invented by one UsePattern: the Temporal operation-row
architecture structurally owns one independent `(endpoint, PhysicalTag)`
matcher and one ordered operand queue for every externally supplied input
role. SpatialMapping must assign nonconflicting Physical Tags to every
may-overlap incompatible interpretation in that ingress's local match domain.
Active output roles within one capability alternative remain injective in
version 1.0. Reusing one tagged egress across different resident rows remains
legal under the existing Temporal grant, tag, and capacity contracts, but
serializing several result roles of one firing onto one egress would require
a future closed retirement-serialization capability. No UsePattern comment or
consumer convention may widen these structural rules.

`operation_pattern_semantics` has exactly the same length and ordinal order as
`resource_contract.use_patterns`. Array position is the one `UsePatternKey`;
there is no repeated key or second pattern inventory. Every declared use
pattern must be referenced by at least one capability alternative; an
unreachable pattern is invalid configuration space rather than dormant
capacity. Its closed projection has these meanings:

* `Direct` preserves one typed parent Canonical Service request and projects
  exactly one port-local child transaction. The child preserves the selected
  actor contract, complete access view, dynamic mask, and logical retirement
  identity. A fence has no access view and always uses this projection. Its
  matching generic UsePattern has exactly one internal-transaction slot. An
  all-zero dynamic mask leaves that slot inactive and completes without
  submitting the parent request.
* `ActiveLanesRowMajor` is legal only for contiguous or indexed access. It
  preserves the same one parent Canonical Service request and derives one
  port-local scalar-element child transaction for each possible lane in
  canonical row-major order. Each child derives its address and payload from
  the exact parent access view and carries no child mask. A child is active
  exactly when its parent lane is active. Inactive children reach no service;
  load-like inactive lanes are zero-filled during the one final row-major
  assembly. An all-zero mask activates no child transaction but still
  completes the original actor.

For `ActiveLanesRowMajor`, internal-transaction ordinal `i` is the port-local
slot for flattened lane `i`. The generic UsePattern must contain the maximum
admitted lane count for every capability tuple that selects it, and must
contain no permanently unreachable higher slot. Equivalently, its
internal-transaction count is exactly that maximum; slots above a selected
narrower access's lane count remain inactive for that firing. All slots remain
inside the UsePattern's one atomic claim envelope. `WholePayload`
atomic granularity requires `Direct`, `PerLane` requires
`ActiveLanesRowMajor`, and scalar `Absent` granularity requires `Direct`.
Plain vector operations may use either projection when the declared actor,
access, endpoint, and resource relation admits it.

The projection never creates a second Canonical Service request or Runtime
ABI request identity. It derives a typed child-transaction plan nested under
the one parent request. Physical beat width, beat decomposition, service-local
queues, and service-local claims are owned by the selected Local or external
memory-service contract. Mapping may select a compatible port pattern and
service pattern, but cannot infer lane or beat splitting from a width ratio or
create an undeclared projection.

The embedded `ResourceContractRecord` is the exact persistent projection
defined by `docs/spec-fabric-resource-contract.md`. Its state and use-pattern
array positions are the only owner-local ordinals. Therefore:

```text
FabricResourceStateRef =
  (MemoryOperationPort(FabricMemoryOperationPortRef), state ordinal)

FabricUsePatternRef =
  (MemoryOperationPort(FabricMemoryOperationPortRef), use-pattern ordinal)
```

Counts may be derived for allocation and validation but are never accepted as
a substitute for the state or pattern records.

Subword and inactive-lane values have exact compatibility meaning:

* `ReadSubwordSemantics::Exact` requires the complete logical result payload
  width to equal the selected data-result endpoint width.
* `ReadSubwordSemantics::ZeroExtend` admits a narrower result, places it in the
  low bits, and guarantees zero in every unused high physical bit.
* `WriteSubwordSemantics::Exact` requires the complete logical write payload
  width to equal the selected data-input endpoint width.
* `WriteSubwordSemantics::ByteEnable` admits a narrower whole-byte payload and
  guarantees that bytes outside the selected logical write remain unchanged.
  Non-byte-multiple writes require a future explicit physical guarantee and
  are not inferred from this value.
* `InactiveLaneSemantics::Suppress` issues no request for an inactive lane.
  It is the dynamic-mask value for a pure store-like actor.
* `InactiveLaneSemantics::SuppressAndZeroFill` additionally produces zero for
  every inactive load-like result lane. It is required for load, RMW, and
  compare-exchange actors with a dynamic mask.

The irrelevant read or write subword field is exactly `NotApplicable` for an
actor schema that lacks that direction. An absent mask pairs exactly with
`NotApplicable`. These schema rules are validated before domain membership is
queried and prevent enum products from admitting nonsensical combinations.

### Typed C++ Projection

Strict import exposes immutable typed views, never raw `ArrayAttr`, integer
domain tags, or unvalidated ordinals:

```text
MemoryOperationPortView::endpoints()
  -> canonical range<FabricTransportEndpointRef>
MemoryOperationPortView::resourceContract()
  -> const ResourceContract &
MemoryOperationPortView::resourceStates()
  -> canonical range<FabricResourceStateRef>
MemoryOperationPortView::usePatterns()
  -> canonical range<FabricUsePatternRef>
MemoryOperationPortView::operationPattern(
    FabricUsePatternRef)
  -> const MemoryOperationPatternView &
MemoryOperationPortView::capabilityAlternatives()
  -> canonical range<FabricMemoryCapabilityAlternativeRef>

MemoryCapabilityAlternativeView::actorContractDomain()
  -> const MemoryActorContractDomainView &
MemoryCapabilityAlternativeView::roleBindings()
  -> canonical range<(ServiceValueRole,
                      FabricTransportEndpointRef)>
MemoryCapabilityAlternativeView::activeRoleBindings(
    CanonicalActorSchemaProjection actor,
    optional<CanonicalMemoryAccessView> access)
  -> canonical range<(ServiceValueRole,
                      FabricTransportEndpointRef)>
MemoryCapabilityAlternativeView::accessDomain()
  -> optional<const ParameterizedMemoryAccessDomainView &>
MemoryCapabilityAlternativeView::admissibleUsePatterns()
  -> canonical range<FabricUsePatternRef>

MemoryOperationPatternView::transactionProjection()
  -> MemoryPortTransactionProjection

deriveMemoryPortTransactionPlan(
    MemoryOperationPatternView pattern,
    CanonicalActorSchemaProjection actor,
    CanonicalService parent_service,
    optional<CanonicalMemoryAccessView> access)
  -> Expected<MemoryPortTransactionPlanView>

MemoryPortTransactionPlanView::parentService()
  -> const CanonicalService &
MemoryPortTransactionPlanView::transactions()
  -> canonical range<MemoryPortChildTransactionView>
MemoryPortTransactionPlanView::assembly()
  -> MemoryPortAssemblyView

MemoryPortChildTransactionView::ordinal()
  -> uint64
MemoryPortChildTransactionView::activation()
  -> Always | ParentMaskAny | ParentMaskLane(uint64)
MemoryPortChildTransactionView::projection()
  -> ParentRequest | ElementLane(uint64)

MemoryPortAssemblyView::resultRoles()
  -> canonical range<ServiceValueRole>
MemoryPortAssemblyView::resultStrategy(ServiceValueRole)
  -> PassThroughParent
   | ParentResponseOrZeroOnEmptyMask
   | RowMajorLaneValues {
       lane_count : uint64
       inactive_value : NotApplicable | ZeroBits
     }
MemoryPortAssemblyView::retirement()
  -> SingleParentRetirement
```

Domain views provide typed `contains` queries and canonical ranges. The one
`SupportsMemoryAccess` implementation consumes an exact registered
`CanonicalActorSchemaProjection`, its derived `CanonicalMemoryAccessView`, an exact
role correspondence, and one selected typed use-pattern reference. No caller
reconstructs domains from counts, names, MLIR dictionaries, or endpoint
widths.

`deriveMemoryPortTransactionPlan` is the sole projection owner for `Direct`
and `ActiveLanesRowMajor`. It verifies that the actor projection, parent
service, and optional access view describe the same exact actor; rejects an
access view for a fence or a missing access view for an addressed actor; and
derives every child activation, address and payload projection, port-local
transaction ordinal, assembly obligation, and final retirement relation.
`Direct` produces one `ParentRequest` child whose activation is `Always` for a
fence or unmasked access and `ParentMaskAny` for a masked access.
`ActiveLanesRowMajor` produces one `ElementLane(i)` child for each possible
flattened lane; its activation is `Always` for an unmasked access and
`ParentMaskLane(i)` for a masked access. These child views are nonpersistent
and have no Canonical Service, Runtime ABI, actor, memory-action, or retirement
identity of their own. Finalization, Mapping, CGRA-sim, and RTL lowering must
consume this one derived plan rather than reimplementing lane address, mask,
contract, or assembly projection.

The assembly view contains every non-completion result role of the parent
service in canonical role order. Unmasked `Direct` uses
`PassThroughParent`. Dynamically masked `Direct` uses
`ParentResponseOrZeroOnEmptyMask`: it passes through the parent response when
`ParentMaskAny` is true, and locally produces all-zero bits for every result
role when it is false and no parent request was submitted.
`ActiveLanesRowMajor` uses `RowMajorLaneValues` for each result role, inserts
all-zero bits for every inactive lane of a dynamically masked access, uses
`NotApplicable` for an unmasked access, and joins all active children into one
`SingleParentRetirement`. Stores and fences have an empty result-role range and
the same single parent retirement. No consumer may turn a child into a scalar
actor, synthesize a scalar Canonical Service, or assign it an independent
consistency or volatile-observation identity; per-lane atomic actions remain
the existing derived children of the parent actor contract.

The access-domain projection is exactly:

```text
UnsignedDomainView::intervals()
  -> canonical range<UnsignedInterval>
UnsignedDomainView::contains(uint64 value)
  -> bool
AlignmentDomainView::containsBytes(uint64 bytes)
  -> bool

ParameterizedMemoryAccessDomainView::accessClasses()
  -> canonical range<MemoryAccessClassView>
ParameterizedMemoryAccessDomainView::matchingClass(
    CanonicalMemoryAccessView access)
  -> optional<MemoryAccessClassView>
ParameterizedMemoryAccessDomainView::contains(
    CanonicalMemoryAccessView access)
  -> bool

MemoryAccessClassView::accessForm()
  -> MemoryAccessForm
MemoryAccessClassView::elementWidths()
  -> const UnsignedDomainView &
MemoryAccessClassView::flattenedLaneCounts()
  -> const UnsignedDomainView &
MemoryAccessClassView::maskInactivePairs()
  -> canonical range<(MemoryMaskForm, InactiveLaneSemantics)>
MemoryAccessClassView::sourceAlignments()
  -> const AlignmentDomainView &
MemoryAccessClassView::readSubwordSemantics()
  -> canonical range<ReadSubwordSemantics>
MemoryAccessClassView::writeSubwordSemantics()
  -> canonical range<WriteSubwordSemantics>

MemoryActorContractDomainView::actorSchema()
  -> OperationSchemaId
MemoryActorContractDomainView::clauses()
  -> canonical range<MemoryActorContractClauseView>
MemoryActorContractDomainView::contains(
    CanonicalActorSchemaProjection actor)
  -> bool
```

`MemoryActorContractClauseView` is the closed typed variant matching the
schema-specific clause records above. It exposes typed enum, boolean, exact
`SyncScopeRef`, and compare-exchange ordering-pair ranges; it is not a map of
field names to values. Every external typed value persists through canonical
bytes produced and validated by its semantic owner. C++ enum values, TableGen
case numbers, and declaration order are implementation details and cannot be
used as a persistent codec. Pair arrays sort lexicographically by owner
canonical bytes.

### OperationSchema Dependency And Fail-Closed Boundary

The persistent capability record depends on the Dataflow owners for exact
canonical codecs and validators for `OperationSchemaId`,
`ServiceValueRole`, `MemoryAccessForm`, `MemoryMaskForm`, `AtomicOrdering`,
`AtomicRmwKind`, `VectorAtomicGranularity`, `SyncScopeRef`, and the typed
boolean and optional-value forms used by the actor-contract clauses. Fabric
embeds those owner-produced canonical bytes inside its own framed fields; it
does not assign local aliases or trust in-memory enum values. The
OperationSchema projection of every memory actor must include its complete
aggregate contract, including `source_alignment_bytes` for atomic access,
RMW, and compare-exchange. Omitting that field is semantic loss and cannot be
repaired by Fabric.

Consequently, the typed memory-capability implementation must not merge before
the OperationSchema owner has merged that complete projection and every
referenced Dataflow owner exposes its stable codec and validator. Independent
domain, ResourceContract, and transaction-projection helpers may be developed
before that merge, but no public `MemoryOperationPortRecord`, capability
alternative, importer, or
Mapping admission path may substitute a Fabric-local operation enum, actor
role enum, exact-geometry record, raw ordinal, wildcard contract, or empty
domain while waiting.

Before that dependency is available, legacy `hw_params` remains authoring
shorthand only. It cannot produce a canonical operation-port inventory,
`FabricMemoryOperationPortRef`, or finalized Fabric root. Finalization fails
closed with `Invalid(missing-memory-capability-contract)` rather than
publishing a partial capability; an unavailable software implementation does
not weaken the schema or become a permissive fallback.

Legacy homogeneous `load_group_size`, `store_group_size`, and `data_width`
fields may exist only as authoring shorthand mechanically expanded into this
inventory before Fabric finalization. They are not canonical hardware
authorities and cannot coexist with an independently editable expanded
inventory.

`operation_schedule` belongs only to the Operation Engine:

```text
Operation Engine present: operation_schedule = spatial | temporal
Operation Engine absent:  operation_schedule absent
```

The surface forms `fabric.mem [spatial]` and `fabric.mem [temporal]` may remain
shorthand when an engine is present. A storage-only occurrence has no schedule.
Keeping an ignored schedule would create duplicate hardware descriptions and
is invalid.

### Spatial Ports

Spatial operation endpoints are fixed, untagged ports:

```text
address payload : !fabric.bits<A>
data payload    : !fabric.bits<D>
lane mask       : !fabric.bits<M>, when present
ctrl/done       : !fabric.bits<0>
```

`A`, `D`, and `M` are per-endpoint physical capacities. A contiguous or
element access uses one scalar address payload. An indexed access uses one
complete flattened address-vector payload. In every case the selected endpoint
requires `A >= address_bits`, `D >= data_bits`, and, for a dynamic mask,
`M >= mask_bits` from the canonical access view. Width adaptation is
low-bit aligned, but no selected endpoint may narrow below the complete
software payload.

Define `P` as the operation-port inventory size. A Spatial configured
operation table has exactly `P` rows in inventory order. Row position fixes
the physical port; the selected actor and its access view derive the active
access configuration. Each active physical port hosts at most one Spatial
software operation in a configuration.

### Temporal Ports And Resident Capacity

Temporal operation endpoints are the tagged counterparts:

```text
address payload : !fabric.bits_tag<A, T>
data payload    : !fabric.bits_tag<D, T>
lane mask       : !fabric.bits_tag<M, T>, when present
ctrl/done       : !fabric.bits_tag<0, T>
```

Temporal hardware declares both a tag width `T` and a bounded resident
operation capacity `K`:

```text
P = operation-port inventory size
K = operation_table_size
```

`K` is independent of `P` and of `2^T`. It is not a tag-indexed address space,
and `K = P` is only one possible hardware choice. The hardware description
must bound `K` to a physically realizable value compatible with its declared
ingress match domains.

The Operation Engine owns this fact through one typed resident-context
contract:

```text
MemoryEngineContract =
    Spatial
  | Temporal { resident_context_count: positive uint64 }
```

A Spatial engine has no resident-context field. For a Temporal engine every
operation port owns the structural context-reference range `[0, K)`, so
`FabricMemoryOperationContextRef(port, row)` is valid exactly when `row < K`.
Equal row ordinals under different ports are alternative views of the same
physical operation-table row; one Mapping cannot activate more than one port
selection for that row. This projection does not copy queue capacity. The
selected port's existing `ResourceContractRecord` remains the sole owner of
operand/result holding, issue, retirement, and arbitration capacity.

Each Active Temporal row selects one physical port, one capability alternative,
and one correspondence per active actor role. An external input role has:

```text
input_match = (physical_ingress_endpoint, PhysicalTag)
```

An external result role has:

```text
output_write = (physical_egress_endpoint, PhysicalTag)
```

Input content matching is local to one physical ingress endpoint. A matching
token is deposited into the ordered operand queue for exactly one resident row
and role. Every active `(physical_ingress_endpoint, PhysicalTag)` interpretation
must therefore be unique within that local match domain. Tags may be reused
across disjoint ingress domains. The selected operation kind and capability
alternative are configured row state, not runtime match fields.

Different operands or results of one actor may use different Physical Tags.
For example, one vector-read row may match address, mask, and control on tags
`1`, `7`, and `3`, then write data and done with tags `5` and `6`. The row fires
only after all required role queues satisfy its selected use pattern. A role
fed by an exact selected internal source bypasses external tag matching for
that role.

At most `K` rows participate, each comparison is `T` bits wide, and each row
has only the role comparators required by its selected physical capability.
The design must not imply a `2^T`-deep table or a global CAM. Mapping assigns
Physical Tags at real tagged writers or ingresses only where may-overlap
incompatible local interpretations require distinction. The memory table
consumes those assignments and does not create another tag authority, firing
identity, iteration identity, logical-memory identity, or vector-lane
identity. A vector payload is one tagged token; lane-level service
transactions remain internal to the selected use pattern.

## Operation Rows And Internal Dependencies

The canonical Fabric description owns operation-table shape, field capability
and semantic domains, queue capacity, and match circuitry. SpatialMapping owns
the selected operation placements, memory bindings, source choices, service
targets, tags, and active rows. Finalization projects those choices into a
configured `memory_operation_table`; `ConfigurationABI` alone defines its
physical bit/address layout and programming contract.

The configured occurrence is one closed sum:

```text
MemoryConfiguration =
    Disabled
  | Active {
      operation_rows
      provider_decode
      physical_refinements
    }
```

`Disabled` carries no row, selector, tag, provider-decode, or refinement
values. The physical inactive encoding belongs only to ConfigurationABI. An
Active projection with no active request source canonicalizes to `Disabled`.

Each physical row is `Unused` or an `Active` variant. `Unused` carries no
fields. The minimum Active Spatial load fields are:

```text
capability_alternative, base_addr, derived_access_projection,
addr_source_sel, optional_mask_source_sel, ctrl_source_sel,
service_target_sel,
expose_data, expose_done
```

The minimum Active Spatial store fields are:

```text
capability_alternative, base_addr, derived_access_projection,
addr_source_sel, data_source_sel, optional_mask_source_sel, ctrl_source_sel,
service_target_sel,
expose_done
```

RMW and compare-exchange rows use the same schema rule with their exact
update, expected, desired, old-value, success, mask, control, and completion
roles. A fence row has only control, completion, and consistency-target roles.
The load and store listings are representative role sets, not a second closed
operation catalog.

`derived_access_projection` is the configured projection of the exact
Dataflow actor into the selected port's parameterized access domain. It
contains no independent software type or shape authority. An absent software
mask selects the one canonical all-active state and carries no source
selector. A dynamic mask selects exactly one compatible external operand or
declared internal source.

A Temporal Active row adds `physical_port_sel`, one `input_match` for each
externally supplied input role, and one `output_write` for each externally
exposed result role. `capability_alternative` already determines the operation
kind and active role set; there is no second operation-kind selector. An
internally supplied or unexposed role has no corresponding external match or
write field. The Temporal table has exactly `K` physical rows. Unused rows have
one canonical semantic state whose physical value is defined by
`ConfigurationABI`. Equivalent rows do not create a Mapping choice;
finalization assigns them deterministically.

Each row has independent ordered operand state. A load has address, optional
mask, and control queues; a store has address, data, optional mask, and control
queues. An indexed address vector and a vector data or mask operand each occupy
one ordered token position, not one queue entry per lane. Result and completion
holding state must implement the declared backpressure and fanout contract.
Queue depths and multicast holding capacity are Fabric facts, never simulator
or Mapping defaults.

Each memory-operation port and the optional Local Memory Service own distinct,
complete `ResourceContractRecord` values. A port pattern claims only
port-owned contexts, operand/result holding state, issue state, and retirement
state. A Local Memory Service pattern claims only service-owned ports, banks,
queues, and outstanding-response capacity. A `ClaimRecord` never crosses an
owner boundary and an ordinal from one contract is invalid in the other.

Dispatch connects the two contracts through the existing ready/valid service
boundary. A port retains an accepted request in its declared holding state
until the selected service accepts it; service admission then executes one
service-owned atomic UsePattern. A zero-extra-buffer implementation may expose
downstream readiness directly, but it still does not copy service state into
the port contract. Mapping selects compatible port and service patterns and
emits their ordinary typed `ResourceUse` records; it cannot synthesize a
cross-owner pattern or generic arbiter graph. Queue contents, occupancy,
outstanding transactions, and grant cursors are nonpersistent execution state.

One accepted actor firing selects exactly one declared memory-operation port
use pattern. Its exact transaction projection produces either one Direct child
or row-major active-lane children under the same parent request. It owns the
port-local child issue order, holding state, result assembly, completion join,
and port-local claims.
The one parent request and its active port-local child transactions are
admitted by compatible patterns owned by the selected Local Memory Service or
reachable external service; those service patterns own physical beat handling
and service-local claims. Inactive lanes issue no child transaction, active
load lanes are assembled in canonical row-major order, and the actor exposes
exactly one load `data + done` packet or one store `done` event. Mapping may
select and parameterize declared patterns but cannot invent a decomposition,
create another request identity, or move claims between owners.

The port use-pattern domain owns the port-local static claim envelope for every
possible dynamic mask it accepts. Each service use-pattern domain separately
owns its service-local envelope. Mapping proves both without assuming runtime
lane values. Concrete execution may omit inactive-lane transactions and
consume fewer dynamic service grants, but it cannot require a resource outside
either declared envelope.

Fabric declares a typed internal source-to-sink eligibility relation. Mapping
may select only declared connections, including examples such as:

```text
store.data       <- load.data
load/store.ctrl  <- load/store.done
load/store.addr  <- load.data
load/store.mask  <- compatible internal value
```

The address form is legal only when explicitly supported. A selected internal
edge uses source and destination row identity, so it does not repeat Temporal
tag lookup. A transfer may enter a destination row with a different external
tag; that is a local configured context transition, not a general transport
retag operation.

A load retires its `data` and `done` as one ordered completion packet. Internal
forwarding and external exposure of either component form one atomic multicast
obligation: no observer or destination may receive `data` without the matching
`done`, receive `done` without the matching `data`, or observe a different
order. The source waits until every selected destination can accept both, or
first enters explicit holding state that preserves the complete packet and
multicast. A store retirement remains one ordered `done` event. An unselected
canonical edge remains an external point-to-point obligation and must be routed
through explicit Fabric resources.

## Local Memory Service

Local Memory Service reuses the Canonical typed memory-service contract. It
adds only the local physical storage capacity and an optional implementation
refinement:

```text
Local Memory Service
  capacity_bytes
  memory_service_contract
  optional implementation_refinement
```

The canonical `memory_service_contract` is the sole owner of the service's
ResourceState, UsePattern, capability, timing, progress, and grant records.
They are not repeated as sibling fields of `LocalMemoryService`. The same
typed `MemoryServiceContractRecord` is embedded by a local `fabric.mem` service
and by a System `memory_service`; neither owner may replace it with a behavior
enum or endpoint-local capability copy.

The exact record is:

```text
MemoryServiceContractRecord {
  regions           : canonical non-empty sequence<MemoryServiceRegionRecord>
  resource_contract : ResourceContractRecord
  capabilities      : canonical non-empty sequence<MemoryServiceCapabilityRecord>
}

MemoryServiceRegionRecord {
  address_base_bytes : uint64
  size_bytes         : positive uint64, base + size representable
  behavior           : Storage
                     | Mmio { accepted_access_domain }
}

MemoryServiceCapabilityRecord {
  actor_contract_domain      : MemoryActorContractDomain
  access_domain              : ParameterizedMemoryAccessDomain | NoAccess
  service_region_ordinals    : canonical sorted unique uint64 sequence
  service_beat_width_bits    : uint64
  admissible_use_patterns    : canonical non-empty set<UsePatternKey>
  consistency_binding        : None
                             | LocalProvider {
                                 release_visibility_point
                                 progress
                               }
                             | SystemDomain(MemoryConsistencyDomainRef)
}

LocalProviderProgress =
    BoundedCompletionCycles { max_issue_to_retire_cycles: positive uint64 }
  | FairEventual
```

The actor schema derives the Canonical Service kind. `NoAccess` is required
exactly for a fence; every addressed kind requires one access domain and a
non-empty service-region set. A fence has an empty region set and
`service_beat_width_bits = 0`; every addressed capability has a positive beat
width. Region ordinals index the containing record and every selected region
must accept the complete access domain. Region address intervals are sorted,
disjoint, and non-adjacent unless their complete behavior records differ;
equivalent adjacent records are merged. A local region is an offset interval
inside `capacity_bytes`; a System region uses the same field as an absolute
service address.

`None` is legal only when every admitted actor clause is plain and needs no
MemoryConsistency provider. An atomic, RMW, compare-exchange, or fence clause
requires one consistency binding. `LocalProvider` means that the exact local
service is the sole participant; its progress cycles are measured by the
memory occurrence's effective Fabric clock. `SystemDomain` is legal only for a
System memory service and references the exact separate domain contract.
Local Memory Service rejects `SystemDomain`, and System memory service rejects
`LocalProvider`. The service's one ResourceContract remains the only owner of
its state, capacity, timing, requester order, and grant behavior in all three
cases.

Capabilities use the same actor-domain and access-domain codecs as operation
ports. Authoring alternatives with equal actor/access/region/beat/consistency
fields are merged by checked set union of `admissible_use_patterns`; the
result is sorted by complete canonical bytes. Overlapping alternatives that
assign different physical facts to one concrete actor/access point are
invalid. This is the one service-admission relation; endpoints and Mapping
reference it rather than reconstructing it.

The persistent wire follows the field order above. Counts, region ordinals,
address fields, sizes, beat widths, bounded progress, and frame lengths are
`u64be`; closed variants and `UsePatternKey` values are `u32be`. Actor, access,
ResourceContract, and Fabric-reference payloads are framed exact production
bytes from their semantic owners. Strict import validates every nested owner,
reconstructs the normalized relation, re-encodes through the production codec,
and requires byte equality. `MemoryServiceContractAttr.record` is exactly this
canonical production wire carried as bytes by MLIR. Generic MLIR attribute
encodings, property bags, behavior-only records, and trailing data are invalid.

The Operation Engine endpoint payload width and the selected memory service's
transaction or beat width are independent Fabric facts. Operation endpoints
accept complete logical address, data, and mask tokens. For a local target the
Local Memory Service contract owns the beat; for a manager target its endpoint
and reachable service contracts own the service payload. The selected
operation use pattern first derives one Direct child or row-major active-lane
children under the actor's one parent service request. The selected service
contract may then realize the active children with one or several physical
beats. The reverse path assembles one logical result before actor retirement.
Neither Mapping nor Runtime may infer either projection from a width ratio.

The architecture contract owns supported read, write, RMW, compare-exchange,
and fence operations; access size and alignment; subword-write semantics;
payload and beat width; latency and initiation interval or service rate;
maximum outstanding requests; ordering, completion, visibility, and coherence
guarantees; and typed resource capacities and use patterns. The canonical
Dataflow actor contracts are defined by
`docs/spec-dataflow-memory-consistency.md`; Canonical Service kinds and
`MemoryConsistencyDomain` are defined by
`docs/spec-fabric-system-adg.md`. A Fabric capability declares the exact
contract domain it accepts. Ordinary read or write support does not imply
atomic, volatile, MMIO, coherence, RMW, compare-exchange, or fence support.

When a capability references a `MemoryConsistencyDomain`, compatibility uses
that domain's exact Fabric-owned release visibility point, fixed
linearization/retirement invariants, progress variant, ResourceStates, and
atomic UsePatterns. An operation engine, provider, Mapping record, or backend
cannot fill an omitted guarantee with a default or weaken it to best effort.

A Local Memory Service region uses the closed `Storage | Mmio` behavior from
`docs/spec-fabric-system-adg.md`. A capability that admits volatile actors must
preserve one at-most-once provider-observable logical operation. A mapped MMIO
region must be non-trapping for its selected access domain because the current
Canonical Dataflow Program has no graph fault protocol. Neither operation rows
nor Mapping may infer these properties from an address range or endpoint name.

Port structures are composed from resource capacities and operation use
vectors. The contract does not need a parallel enumeration such as
`single_port`, `1r1w`, or `true_dual_port`. Bank function, queue structure,
SRAM organization, ECC, and clock or power details are implementation
refinements unless they change observable behavior. Request eligibility,
capacity, cycle-visible grant and state-update policy, latency, completion
order, and backpressure visibility must be exact Fabric facts or
Mapping-selected exact refinements declared by Fabric. An implementation,
runtime, or simulator may not invent them.

Initial contents, logical allocation, object lifetime, and invocation-specific
addresses are not hardware capability. They belong to Deployment and runtime
inputs. Reset, reconfiguration, or power behavior enters the service contract
only when it changes observable content retention.

## Dynamic Consistency Execution

An issued read, write, RMW, compare-exchange, or fence derives the
nonpersistent `MemoryAction` defined by
`docs/spec-dataflow-memory-consistency.md`. The selected operation capability,
use pattern, dispatch target, and `MemoryConsistencyDomain` determine which
exact execution provider admits and linearizes it. `fabric.mem` does not own a
parallel memory-model state or copy the Dataflow access contract.

One actor issue remains one logical memory-service request and one retirement
publication. A declared use pattern may perform several lane operations,
physical beats, local forwarding actions, or protocol retries, but those are
children of the same request. They do not create extra actor firings,
retirements, atomic objects, or provider-visible volatile operations.

A Local Memory Service whose complete domain closure is inside the
SpatialCore may execute through the exact Fabric-local provider. A manager
target delegates its external obligations through the Runtime ABI-owned
Spatial Service boundary. The external provider owns its modification order,
reads-from, cache, coherence, and system ordering. The Operation Engine,
Bridge, and local simulator must not shadow that state.

## Configurable Service Dispatch

Operation contexts and subordinate requests are request sources. Addressed and
fence targets remain distinct typed alternatives:

```text
AddressedRequestSource = read | write | RMW | compare-exchange context
                         | subordinate endpoint plus decoded binding/context
AddressedServiceTarget = local service | manager endpoint

FenceRequestSource = fence context
FenceServiceTarget = local MemoryConsistency domain | manager endpoint
```

Fabric owns the fixed eligibility relation `H_dispatch`. It states which
physical selectors and cross-connect paths exist. Every addressed Mapping
MemoryOperationEntry and ExposureEntry owns exactly one closed typed
`LocalMemoryServiceRef | ManagerEndpointRef` target. Every FenceOperation owns
one `MemoryConsistencyDomainRef | ManagerEndpointRef` target. Those fields
collectively are normalized `C_dispatch`; there is no parallel persistent
relation record.
The verifier checks `C_dispatch` is a subset of `H_dispatch`. It is a partial
function that selects one target for each enabled concrete request domain. A request that accesses
several targets requires an explicit sharding, replication, mirroring, or
coherence service; it is not a special dispatch mode.

One occurrence carries exactly one canonical
`MemoryConnectivityContractRecord`. Its closed wire is:

```text
MemoryConnectivityContractRecord {
  operation_ports : array<OperationPortDispatchRecord>
  subordinate_endpoints : array<SubordinateDispatchRecord>
  internal_connections : sorted unique array<InternalConnectionRecord>
}

OperationPortDispatchRecord {
  capability_target_domains : array<non-empty sorted unique
      MemoryDispatchTarget>
}

MemoryDispatchTarget =
    LocalMemoryService
  | ManagerEndpoint { manager_endpoint_ordinal: uint64 }

SubordinateDispatchRecord {
  max_exposed_bindings : positive uint64
  match_fields : sorted unique set<Range | Prefix | AddressSpace | Context>
  address_transform : None | ConstantBaseOffset
  target_domain : non-empty sorted unique set<MemoryDispatchTarget>
}

InternalConnectionRecord {
  source_endpoint_ordinal : uint64
  sink_endpoint_ordinal : uint64
}
```

The operation-port array follows the canonical port inventory. Its nested
array follows that port's canonical capability-alternative inventory, so
source identity is structural and is never repeated as an ID. The subordinate
array follows the subordinate endpoint inventory. A single-binding provider
may have an empty match-field set because its selector is constant; a provider
with more than one configured binding must declare at least one bounded match
field. `ConstantBaseOffset` admits only the simple base translation derived
from one Memory Binding. It is not a generic transform.

Manager ordinals index the occurrence's manager endpoint inventory, not the
function signature and not a PnR dense index. `LocalMemoryService` is legal
only when the occurrence declares that optional resource. Every target domain
is non-empty. These rules make unreachable request sources invalid hardware
rather than dormant configuration space.

Internal-connection ordinals index the occurrence's token endpoint inventory,
with inputs first and outputs second. The source must be an output, the sink
must be an input, and the source payload must be at least as wide as the sink.
The selected actor roles and capability alternative provide the semantic type
check. A Temporal internal edge addresses the destination row directly and
therefore does not transport or compare the external Physical Tag. The
relation is eligibility only; Mapping owns the selected sink-to-source edges.

The persistent wire uses `u64be` counts and ordinals and `u32be` closed-union
tags. Array positions own operation-port, capability-alternative, and
subordinate source identity. Strict import decodes, validates against the
exact occurrence, re-encodes, and requires byte equality; reordered target
sets or internal connections are noncanonical.

An operation row carries its derived `service_target_sel`. A subordinate
endpoint that exposes several logical memories uses bounded provider decode
derived from Exposure Entries and Memory Bindings. The endpoint capability
must bound entry count and allowed range, prefix, address-space, or context
match fields. Arbitrary predicates, an address-width-sized direct table, and
an unbounded CAM are invalid architecture models. Complex hashing,
translation, cache, or coherence behavior requires an explicit typed service
transform.

Responses return to the recorded transaction origin and context. There is no
independent response-route configuration. Fabric must provide sufficient
response tracking and backpressure state for every declared outstanding
request guarantee.

Selecting a Local Memory Service or manager endpoint does not select a
different software/runtime request schema. Both use the Runtime ABI-owned
`SpatialServiceRequest` and `SpatialServiceResponse`; adapters may translate
that one typed boundary to the selected service mechanism but may not
reinterpret the Mapping binding or create a parallel protocol.

The active operation and service connectivity is runtime-reconfigurable, but
the runtime only installs configuration derived from an immutable Mapping.
It never chooses a new target or repairs a mapping. A Spatial configuration
changes only at a quiescent boundary with no in-flight transaction and empty
affected queues. A Temporal context may be changed when that context is
quiescent, unless the Fabric contract explicitly supplies versioned context
state.

## Ownership And Derived Configuration

The ownership boundary is:

* Fabric owns operation ports and contexts, Local Memory Service, endpoint
  capability, service and queue contracts, internal-connectivity eligibility,
  `H_dispatch`, matching capacity, and semantic configuration fields and
  domains.
* TechMapping selects a Memory Realization and exact internal-edge witnesses.
* SpatialMapping owns Memory Engine Bindings, Memory Bindings,
  MemoryOperationEntries, Exposure Entries, their exact dispatch-target
  fields, selected internal connections, and event-relative `ResourceUse`
  including Physical Tags.
* SystemMapping extends a Spatial boundary proxy to the selected system
  provider through `ServiceRealization` and system `ResourceUse`.
* `ConfigurationABI` alone owns physical bit/address encoding and the
  programming contract. `HardwareConfigurationImage` artifacts carry its
  immutable encoding of the selected Mapping for exact Programming Units.
* Deployment carries those images and memory images; runtime applies them and
  supplies invocation-specific allocations and authorization.

`addr_table` and `mem_enable` are retired names. Configured rows and
provider-decode entries are deterministic semantic projections of the owners
above. Raw physical fields exist only in the exact `ConfigurationABI` and its
`HardwareConfigurationImage`; they are not another Fabric or Mapping record.

## Representation And Implementation Ownership

Three interfaces remain distinct:

* Fabric MLIR owns the typed operation endpoints, manager and subordinate
  `memref` capabilities, capability alternatives, ResourceState, UsePatterns,
  timing, and service behavior.
* The internal RTL leaf-channel ABI is mechanically derived from Fabric:
  `bits<W>` becomes data plus valid/ready, `bits_tag<W,T>` becomes data plus
  tag plus valid/ready, `bits<0>` becomes valid/ready only, and
  `bits_tag<0,T>` becomes tag plus valid/ready only. Zero-width payloads never
  become zero-width HDL vectors.
* A selected HardwareImplementation refines manager and subordinate memory
  capabilities to AXI, TileLink, CXL, or a custom protocol. Fabric does not
  freeze one of those protocol pinouts.

Stateful memory logic uses its exact Fabric clock and reset domains;
`Spatial | Temporal` does not decide whether the leaf is stateful. Semantic
configuration fields are deterministic projections of Mapping and Fabric.
Only `ConfigurationABI` owns their physical wires, addresses, codebooks, and
inactive encodings; backend-local `cfg_*` names or structs cannot become a
public configuration authority.

The operation-port and memory-service capability schemas above are closed.
An occurrence that selects a Local Memory Service must carry one strict
`MemoryServiceContractRecord`; a missing, malformed, noncanonical, or
behavior-only record is invalid.
Concrete SRAM, controller, cache, queue, comparator, and provider-decode
structure is implementation-owned only where a closed Fabric contract leaves
it unobservable. An implementation refinement must preserve the capability
contract and must not add Mapping semantics.

## Schedule And Access-Geometry Examples

The following six examples use two independent physical operation ports, one
read and one write, plus one manager memory-service capability whose physical
service beat is 64 bits. Let `B<W>` denote `!fabric.bits<W>` and `BT<W,4>`
denote `!fabric.bits_tag<W,4>`. The manager `memref` endpoint remains a memory
capability and is not converted into a tagged Dataflow channel.

The Spatial element-only interface is:

```text
manager<M64>
read  (address:B<32>, control:B<0>)
    -> (data:B<32>, completion:B<0>)
write (address:B<32>, data:B<32>, control:B<0>)
    -> (completion:B<0>)

accepted addressed views: element f32
derived class: ElementAccessOnly
```

The Spatial vector-only interface is:

```text
manager<M64>
read  (address:B<32>, mask:B<4>, control:B<0>)
    -> (data:B<128>, completion:B<0>)
write (address:B<32>, data:B<128>, mask:B<4>, control:B<0>)
    -> (completion:B<0>)

accepted addressed views: contiguous vector<4xf32>, mask absent or dynamic
derived class: VectorAccessOnly
```

The Spatial hybrid interface has the same maximal physical endpoint shape as
the vector-only interface, but its capability alternatives admit both the
element `f32` view and the contiguous `vector<4xf32>` view:

```text
accepted addressed views:
  element f32
  contiguous vector<4xf32>, mask absent or dynamic
derived class: ElementAndVectorAccess
```

One read port represents that hybrid support without one alternative per
geometry. The following rendering separates its port-owned pattern table from
its single capability alternative:

```text
MemoryOperationPort:
  endpoint_inventory = {address, mask, control, data, completion}
  resource_contract.use_patterns = {
    0: <one complete pattern with one internal-transaction slot>
  }
  operation_pattern_semantics = {
    0: { transaction_projection = direct }
  }
  capability_alternatives = [
    {
      actor_contract_domain:
        actor_schema = OperationSchemaId::DataflowLoad
        contract_clauses = [Plain { volatile_values = {false} }]

      role_to_endpoint:
        address -> address
        mask -> mask
        control -> control
        data -> data
        completion -> completion

      parameterized_access_domain:
        access_classes = [
          {
            access_form = element,
            element_width_bits = {8, 16, 32},
            flattened_lane_count = {1},
            mask_inactive_pairs = {(absent, not_applicable)},
            source_alignment.accepted_log2_byte_alignments = {[0, 63]},
            read_subword = {zero_extend},
            write_subword = {not_applicable}
          },
          {
            access_form = contiguous,
            element_width_bits = {32},
            flattened_lane_count = {4},
            mask_inactive_pairs = {
              (absent, not_applicable),
              (dynamic, suppress_and_zero_fill)
            },
            source_alignment.accepted_log2_byte_alignments = {[0, 63]},
            read_subword = {exact},
            write_subword = {not_applicable}
          }
        ]

      admissible_use_patterns = {0}
    }
  ]
```

The set notation above is the human rendering of canonical singleton/range
domains. Alignment values are exponents: `[0, 63]` admits byte alignments
`2^0` through `2^63`. Both classes remain inside one physical
capability alternative, so endpoint bindings, actor-contract admission, and
use-pattern references are not duplicated. Contiguous `vector<2xf64>` is
rejected because element width 64 is absent even though its total payload also
has 128 bits.

For an element firing on that hybrid port, the 8-, 16-, or 32-bit value
occupies the low data bits, unused high result bits are zero, and the mask
endpoint is inactive. A narrow store is legal only because the declared
subword-write use pattern provides the required byte enables or equivalent
behavior. Width adaptation alone would not prove that store.

The three Temporal interfaces replace every operation-channel `B<W>` above
with `BT<W,4>` and retain the same accepted-view domains. They therefore form
Temporal element-only, vector-only, and hybrid cases. The manager service
endpoint remains untagged. A representative resident vector-read context is:

```text
context[2] selects contiguous-vector-read on physical read port 0
  address <- (read_address_endpoint, tag 1)
  mask    <- (read_mask_endpoint,    tag 7)
  control <- (read_control_endpoint, tag 3)
  data    -> (read_data_endpoint,    tag 5)
  done    -> (read_done_endpoint,    tag 6)
```

The tags identify local physical role interpretations, not vector lanes. The
row fires after its address, mask, and control queues are ready. A selected
internal source would replace the corresponding input match rather than
performing another tag lookup.

For the vector cases, one masked `vector<4xf32>` load with mask `1011`
consumes one base-address token, one four-bit mask token, and one control
token. `Direct` retains one parent-access child; `ActiveLanesRowMajor` derives
four possible children and activates lanes 0, 1, and 3. A compatible service
pattern may realize either plan with several 64-bit beats. The actor still
owns one parent request, assembles one 128-bit result in canonical row-major
lane order, zero-fills the inactive lane, and publishes one `data + done`
retirement packet. Store similarly omits inactive lane children and publishes
one `done` event.

The external address, data, and mask are never split into lane routes. The
same vector capability rejects contiguous `vector<2xf64>` despite equal total
width because its element-width and lane-count projections are outside the
domain. It also does not imply an `element` access to one
`vector<4xf32>`-typed memref element; that view has one 128-bit element and no
lane shape and must be admitted explicitly.

## Validation Anchors

Anchor-level tests should cover:

* canonical finite/range domain normalization, rejection of empty or
  overlapping access classes, rejection of an equivalent noncanonical
  decomposition of the complete capability relation, and strict typed
  actor-contract decoding;
* one hybrid 128-bit port whose single load alternative accepts element `f32`
  and contiguous `vector<4xf32>` but rejects contiguous `vector<2xf64>`;
* complete ResourceState and UsePattern record round-trip through the port,
  with count-only, raw-ordinal, and wrong-owner substitutes rejected;
* `Direct` and `ActiveLanesRowMajor` typed child-transaction projection,
  including fence Direct handling, dynamic-mask suppression, row-major
  assembly, Direct all-zero local result production without a parent response,
  exact internal-transaction capacity, one parent request identity, and
  atomic-granularity rejection;
* independent `P`, `K`, and `T` capacities with content matching local to a
  physical ingress;
* same-ingress Temporal input-role matching with distinct may-overlap tags,
  tag reuse across disjoint match domains, injective output roles within one
  firing, and rejection of duplicate local ingress interpretations;
* one shared hybrid port versus separate element and vector ports, with exact
  capacity differences and no persistent derived geometry class;
* distinct element, contiguous, and indexed access compatibility, including
  rejection of equal-width but incompatible element or lane geometry;
* dynamic-mask routing, inactive-lane suppression, zero-fill for masked loads,
  all-zero-mask completion without a service request, and absence of a mask
  selector for an unmasked actor;
* a wide operation endpoint backed by narrower declared service beats, with
  row-major assembly and exactly one logical retirement event;
* one Physical Tag per routed Temporal vector token role, never one Tag per
  lane or one forced common tag for the entire operation row;
* selected `load.data -> store.data` and `done -> ctrl` internal dependencies,
  including joint load `data + done` retirement and store `done` retirement;
* operation-engine-only, engine-plus-local-service, and storage-only forms;
* one Local Memory Service and one System memory service importing the same
  canonical service capability record, with owner-incompatible consistency
  bindings rejected;
* many logical memories sharing one physical service through distinct
  bindings;
* fixed `H_dispatch` versus Mapping-selected `C_dispatch`;
* bounded subordinate provider decode and mechanical response return; and
* deterministic semantic operation-table projection followed by encoding
  through the exact `ConfigurationABI`, including zero-payload control and
  tagged-control RTL channels without zero-width HDL vectors.

Tests should not freeze printer layout, internal comparator topology, queue
container shape, or a particular protocol implementation. They must not build
a geometry-by-contract Cartesian fixture matrix.
