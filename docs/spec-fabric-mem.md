# Fabric Memory

## Purpose

`fabric.mem` is the Fabric realization for canonical `dataflow.load` and
`dataflow.store` operations. It describes physical memory-operation capability
and memory-service capability. It does not describe a software address space,
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
MemoryOperationPortCapability {
  operation_kind
  physical_role_to_endpoint_relation
  parameterized_access_domain
  use_pattern_domain
}
```

The occurrence `function_type` is the sole owner of physical endpoint kinds
and widths. The inventory assigns semantic roles to those endpoint ordinals
and owns the hardware access domain; it does not repeat endpoint types. Load
and store counts, physical data capacity, and optional mask capacity are
derived from the inventory and function type. Different operation ports in one
engine may have different fixed physical capacities.

The maximal load bundle is:

```text
(address_payload, optional lane_mask, ctrl) -> (data_payload, done)
```

The maximal store bundle is:

```text
(address_payload, data_payload, optional lane_mask, ctrl) -> done
```

Store does not produce a data result. The mask endpoint is physically present
only when the port can consume a dynamic lane mask. An unmasked actor may use a
mask-capable port with that endpoint inactive; inactive means no consume and no
backpressure and must be proven by the capability relation. A masked actor
cannot use a port without a mask endpoint or an exact declared internal mask
source.

The exact software requirement is the nonpersistent
`CanonicalMemoryAccessView` derived from `dataflow.load` or `dataflow.store`.
Fabric supports it through one relation:

```text
SupportsMemoryAccess(port, access, correspondence) =
  operation kind is accepted
  and access form is accepted
  and element-size and flattened-lane domains are accepted
  and mask form is accepted
  and ordered actor roles correspond to physical roles
  and address, data, and mask payloads fit their endpoints
  and alignment and subword-write requirements are accepted
  and at least one declared use pattern realizes the access semantics
```

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

An Active Temporal row is selected by exact content match on:

```text
(operation_kind, physical_port_sel, tag_match)
```

Matching is local to the selected operation kind and physical ingress. At
most the resident rows assigned to that domain participate, with depth bounded
by `K` and compare width `T`. The design must not imply a `2^T`-deep table or a
global wide CAM. A tag value may be reused in different physical match domains;
Active rows within one domain must have unique tag matches.

All external operands for one dynamic load or store firing match the same row.
External results use that row's configured tag. Mapping assigns Physical Tags
at real tagged writers or ingresses only where may-overlap incompatible local
interpretations require distinction. The memory table consumes that local
assignment and does not create another tag authority, firing identity,
iteration identity, logical-memory identity, or vector-lane identity. A vector
payload is one tagged token; lane-level service transactions remain internal
to the selected use pattern.

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
base_addr, derived_access_projection,
addr_source_sel, optional_mask_source_sel, ctrl_source_sel,
service_target_sel,
expose_data, expose_done
```

The minimum Active Spatial store fields are:

```text
base_addr, derived_access_projection,
addr_source_sel, data_source_sel, optional_mask_source_sel, ctrl_source_sel,
service_target_sel,
expose_done
```

`derived_access_projection` is the configured projection of the exact
Dataflow actor into the selected port's parameterized access domain. It
contains no independent software type or shape authority. An absent software
mask selects the one canonical all-active state and carries no source
selector. A dynamic mask selects exactly one compatible external operand or
declared internal source.

A Temporal Active row adds `operation_kind`, `physical_port_sel`, and
`tag_match` to the corresponding load or store fields. The Temporal table has
exactly `K` physical rows. Unused rows have one canonical semantic state whose
physical value is defined by `ConfigurationABI`. Equivalent rows do not create
a Mapping choice; finalization assigns them deterministically.

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

The architecture contract owns supported read and write operations; access
size and alignment; subword-write semantics; payload and beat width; latency
and initiation interval or service rate; maximum outstanding requests;
ordering, completion, visibility, and coherence guarantees; and typed resource
capacities and use patterns. The canonical Dataflow atomic, RMW,
compare-exchange, fence, and volatile operation contracts are defined by
`docs/spec-dataflow-memory-consistency.md`. Their Fabric service capability
and consistency-domain realization are not yet defined and must not be
inferred from ordinary load/store support.

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

## Configurable Service Dispatch

Operation contexts and subordinate requests are request sources. Local storage
and manager endpoints are service targets:

```text
RequestSource = load context | store context |
                subordinate endpoint plus decoded binding/context
ServiceTarget = local service | manager endpoint
```

Fabric owns the fixed eligibility relation `H_dispatch`. It states which
physical selectors and cross-connect paths exist. Every Mapping AccessEntry
and ExposureEntry owns exactly one closed typed
`LocalMemoryServiceRef | ManagerEndpointRef` target. Those fields collectively
are normalized `C_dispatch`; there is no parallel persistent relation record.
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
* SpatialMapping owns Memory Engine Bindings, Memory Bindings, Access Entries,
  Exposure Entries, their exact dispatch-target fields, selected internal connections, and
  event-relative `ResourceUse` including Physical Tags.
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

The semantic model above is closed. This document does not define Fabric
assembly spelling or typed attribute layout for a Local Memory Service,
service refinement, or provider-decode capability; only a dedicated operation
specification may do so. Local memory implementation refinement may choose
concrete SRAM, controller, or cache mechanisms. Interconnect Implementation may
choose AXI, TileLink, CXL, or custom protocol mechanisms. Both must refine this
capability contract and must not add Mapping semantics.

## Vector Access Example

Consider separate load and store operation ports with these physical roles:

```text
address : bits<32>
data    : bits<128>
mask    : bits<4>
ctrl/done : bits<0>

parameterized access domain:
  access_form = contiguous
  element_bits = 32
  lane_count = 4
  mask_form = absent | dynamic

selected use pattern:
  one logical firing -> up to four ordered 32-bit service transactions
```

The backing service beat is 32 bits. A masked `vector<4xf32>` load with active
mask `1011` consumes one base-address token, one four-bit mask token, and one
control token; lane zero is the least-significant mask bit. It issues service
transactions only for active lanes 0, 1, and 3, assembles
lane results into one 128-bit token with lane zero in the least-significant
slice, fills the inactive lane with zero, and publishes one `data + done`
packet. The analogous store issues no write for the inactive lane and publishes
one `done` event.

The external address, data, and mask are never split into lane routes. The same
port rejects a contiguous `vector<2xf64>` access despite its equal total width,
because its element-width and lane-count projections are outside the domain.
It also does not imply support for an `element` access to one
`vector<4xf32>`-typed memory element; that access has `element_bits = 128`, no
lane shape, and `lane_count = 1` and requires a separately admitted capability
domain.

## Validation Anchors

Anchor-level tests should cover:

* independent `P`, `K`, and `T` capacities with content matching local to a
  physical ingress;
* tag reuse across disjoint match domains and rejection within one domain;
* distinct element, contiguous, and indexed access compatibility, including
  rejection of equal-width but incompatible element or lane geometry;
* dynamic-mask routing, inactive-lane suppression, zero-fill for masked loads,
  all-zero-mask completion without a service request, and absence of a mask
  selector for an unmasked actor;
* a wide operation endpoint backed by narrower declared service beats, with
  row-major assembly and exactly one logical retirement event;
* one Physical Tag for a Temporal vector token, never one Tag per lane;
* selected `load.data -> store.data` and `done -> ctrl` internal dependencies,
  including joint load `data + done` retirement and store `done` retirement;
* operation-engine-only, engine-plus-local-service, and storage-only forms;
* many logical memories sharing one physical service through distinct
  bindings;
* fixed `H_dispatch` versus Mapping-selected `C_dispatch`;
* bounded subordinate provider decode and mechanical response return; and
* deterministic semantic operation-table projection followed by encoding
  through the exact `ConfigurationABI`.

Tests should not freeze printer layout, internal comparator topology, queue
container shape, or a particular protocol implementation.
