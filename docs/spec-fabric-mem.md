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

An Operation Engine owns one canonical ordered inventory of physical memory
operation ports. Each inventory entry contains exactly:

```text
MemoryOperationPort {
  endpoint_inventory
  non-empty canonical capability_alternatives
  resource_state_domain
  use_pattern_domain
}

MemoryOperationCapability {
  canonical_operation_kind
  active_physical_role_to_endpoint_relation
  accepted_actor_contract_domain
  parameterized_access_domain, absent for fence
  non-empty admissible_use_pattern_refs
}
```

The occurrence `function_type` is the sole owner of physical endpoint kinds
and widths. A port's `endpoint_inventory` is an ordered set of those endpoint
ordinals; it does not repeat their types. Each capability alternative selects
the active subset and gives those endpoints exact actor roles. Inactive
endpoints consume nothing, produce nothing, and exert no backpressure. Read,
write, RMW, compare-exchange, and fence counts, physical payload capacities,
and optional mask capacity are derived from the inventory, alternatives, and
function type. Different operation ports in one engine may have different
fixed physical capacities.

One port may expose several capability alternatives only when the same
physical endpoint inventory and ResourceState genuinely implement those
alternatives. This is the memory-operation analogue of hardware sharing: a
shared superset port can admit read and write alternatives, while separate
read and write ports remain separate physical capacity. Alternatives with the
same observable actor relation and use pattern are invalid duplicates. A
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
  and operation kind is accepted
  and exact actor contract is in the accepted domain
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

The Operation Engine and optional Local Memory Service each own their closed
typed `ResourceState` values, canonical initial state, capacity dimensions,
atomic UsePatterns, stable typed requester order, and exact GrantPolicy or
exact refinement domain. One memory-operation pattern may atomically claim an
operation context, operand/result holding state, service port, bank, and
outstanding-response capacity. Mapping cannot split that pattern into
independent reservations or construct a generic arbiter graph. Queue contents,
occupancy, outstanding transactions, and grant cursors are nonpersistent
execution state.

One accepted actor firing selects exactly one declared memory-operation use
pattern. That pattern may issue one or several internal service transactions
or physical beats. Fabric owns their lane and beat decomposition, resource
claims, issue order, result assembly, and completion join. Inactive lanes issue
no transaction, active load lanes are assembled in canonical row-major order,
and the actor exposes exactly one load `data + done` packet or one store `done`
event. Mapping may select and parameterize a declared pattern but cannot invent
or edit this decomposition.

The use-pattern domain also owns the static claim envelope for every possible
dynamic mask accepted by the port. Mapping proves that envelope without
assuming runtime lane values. Concrete execution may omit inactive-lane
transactions and consume fewer dynamic service grants, but it cannot require a
resource outside the declared envelope.

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

The Operation Engine endpoint payload width and the selected memory service's
transaction or beat width are independent Fabric facts. Operation endpoints
accept complete logical address, data, and mask tokens. For a local target the
Local Memory Service contract owns the beat; for a manager target its endpoint
and reachable service contracts own the service payload. The selected
operation use pattern may lower one typed request to several service beats when
those contracts permit it. The reverse path assembles one logical result before
actor retirement. Neither Mapping nor Runtime may infer decomposition from a
width ratio.

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

The semantic model above is closed. Concrete SRAM, controller, cache, queue,
comparator, and provider-decode structure is implementation-owned only where
Fabric leaves it unobservable. An implementation refinement must preserve the
capability contract and must not add Mapping semantics.

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

For an element firing on that hybrid port, the 32-bit value occupies the low
data bits, unused high result bits are zero, and the mask endpoint is inactive.
A narrow store is legal only because the declared subword-write use pattern
provides the required byte enables or equivalent behavior. Width adaptation
alone would not prove that store.

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
token. A declared use pattern may lower it to several 64-bit service beats,
but only active lanes 0, 1, and 3 access memory. It assembles one 128-bit
result in canonical row-major lane order, zero-fills the inactive lane, and
publishes one `data + done` retirement packet. Store similarly omits the
inactive lane and publishes one `done` event.

The external address, data, and mask are never split into lane routes. The
same vector capability rejects contiguous `vector<2xf64>` despite equal total
width because its element-width and lane-count projections are outside the
domain. It also does not imply an `element` access to one
`vector<4xf32>`-typed memref element; that view has one 128-bit element and no
lane shape and must be admitted explicitly.

## Validation Anchors

Anchor-level tests should cover:

* independent `P`, `K`, and `T` capacities with content matching local to a
  physical ingress;
* per-role Temporal input and output tags, tag reuse across disjoint match
  domains, and rejection of duplicate local ingress interpretations;
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
* many logical memories sharing one physical service through distinct
  bindings;
* fixed `H_dispatch` versus Mapping-selected `C_dispatch`;
* bounded subordinate provider decode and mechanical response return; and
* deterministic semantic operation-table projection followed by encoding
  through the exact `ConfigurationABI`, including zero-payload control and
  tagged-control RTL channels without zero-width HDL vectors.

Tests should not freeze printer layout, internal comparator topology, queue
container shape, or a particular protocol implementation.
