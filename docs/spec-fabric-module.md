# Fabric Module

This document specifies `fabric.module`, the SpatialCore or CGRA
template container of the fabric dialect. `fabric.system` is the
SoC/system container.

`fabric.module` is the SpatialCore or CGRA-level ADG container. It is
not a system-level SoC container and it does not use `fabric.link` for
internal connectivity.

The module boundary has two orthogonal planes. `bits` and `bits_tag` are
handshake-bearing token transports used to realize graph value, stream,
control, and completion obligations. `memref` is a memory-service capability,
not a token stream or a physical storage identity.

## Identity

* Mnemonic: `module`.
* The op is a `Symbol` with a required `sym_name`.
* The body is a single block (a `SizedRegion<1>` with one block).
* The body region is `Graph`-kind, so SSA dominance is not enforced and
  back-references between body ops are permitted.
* The op is `IsolatedFromAbove`: every value used inside the body must
  come from the body's block arguments or be defined inside the body.
  No external SSA value may leak into the module body.
* The body is closed by a `fabric.yield` terminator.

`sym_name` is an authoring and intra-MLIR reference aid, not persistent
hardware identity. `docs/spec-fabric-artifact.md` owns root finalization,
canonical semantic bytes, dependency framing, and ArtifactIdentity;
`docs/spec-fabric-identity.md` owns the persistent Fabric local-reference
catalog. Each consuming schema separately declares which of those references
are visible to Mapping.

## Clock And Reset Slots

A reusable Module owns symbolic Clock and Reset roles, never concrete System
domain contracts. `docs/spec-fabric-identity.md` uniquely defines the closed
`FabricModuleDomainSlotRef`, `FabricModuleDomainMemberRef`, and
`ModuleDomainAssignment` wires. The `fabric.module` root carries them as two
required typed properties:

```text
domain_slots : canonical sorted-unique array<FabricModuleDomainSlotRef>
domain_assignments : canonical sorted-unique array<ModuleDomainAssignment>
```

These properties are part of the root operation rather than Module-body child
operations, so the body whitelist does not gain a second domain schema. Slot
references are dense within `Clock` and `Reset`. A slot name is authoring
metadata and does not enter identity. A Module slot owns no period, phase,
polarity, synchronization, initial state, or release latency. A Module cannot
declare or reference a concrete `HardwareDomainRef`.

An authoring Module that omits the complete domain relation is canonical
shorthand for exactly one Clock slot and one Reset slot, with every boundary
face and physical owner assigned to ordinal zero of the corresponding kind.
Finalization materializes that relation before canonical bytes are written.
The shorthand applies only when no slot, assignment, or instance binding was
authored. Once any row is authored, the complete explicit relation is required
and missing rows fail closed. Canonical and imported Modules therefore never
carry an implicit or partial relation. Every finalized Module has at least one
Clock slot and one Reset slot; an explicitly empty inventory is not a second
canonical case.

Every Module boundary face is assigned to exactly one Clock slot and exactly
one Reset slot. Every `FabricModulePhysicalOwnerRef` is also assigned to exactly
one Clock slot and exactly one Reset slot. These total assignments express
topological domain association, including for a combinational owner; they do
not imply that every owner has Clock or Reset signal ports.

An internal owner with a nonempty canonical `ResourceState` inventory is
stateful, consumes the Clock and Reset signals of its assigned slots, and must
define every state's canonical reset value. An owner with no state inventory is
combinational and consumes neither signal. Hidden registered state, a clocked
owner without Fabric-owned state, and state without Reset are invalid. Version
4.0 introduced this rule. A later resetless stateful contract requires an
explicit closed resource-contract variant and a Fabric major revision. Hierarchy,
containment, insertion order, and a parent owner's assignment never imply an
assignment for a child owner.

When one Module instantiates another Module, only the instance edge's explicit
`domain_slot_bindings` relation from `docs/spec-fabric-instantiate.md` relates
the callee slots to this Module's slots. It is a total child-slot-to-parent-slot
function, not inheritance. Elaboration composes the callee assignments through
that function and publishes only the resulting flat assignments; the child
slot inventory, Module boundary, and instance binding do not survive in the
finalized parent Module.

The source and destination of every ordinary Module-local physical connection
must resolve to equal symbolic Clock and Reset slots. A cross-slot relation is
legal only through an explicit typed crossing resource whose contract owns
both faces. `loom.fabric 5.0` defines no Module-local Clock or Reset crossing
carrier, so a Module containing such a relation fails closed. A backend cannot
repair it by inserting a synchronizer, FIFO, reset bridge, or timing exception.

## Inputs (entry-block arguments)

`fabric.module` carries zero SSA operands of its own. Module inputs are
declared as the entry block's arguments, mirroring `func.func`. The
syntax is:

```mlir
fabric.module @top(%a : !fabric.bits<32>,
                   %b : memref<8xi32>,
                   %c : !fabric.bits_tag<8, 2>,
                   %d : !fabric.bits_tag<0, 3>) -> (...) {
  ...
}
```

Allowed input types:

| Type                      | Allowed | Rationale                                     |
|---------------------------|---------|-----------------------------------------------|
| `!fabric.bits<W>`         | yes     | Native handshake-bearing fabric port.         |
| `!fabric.bits_tag<W, T>`  | yes     | Native fabric port; `W = 0` is the tag-only form. |
| `memref<...>`             | yes     | Manager/requester memory capability import.   |
| Any other MLIR type       | no      | Rejected by the verifier.                     |

`i32`, `f32`, `vector`, `tensor`, `index`, etc. are not valid module
input types and are rejected with a clear diagnostic.

Each input port has its own type; widths and shapes are independent
across ports.

## Outputs

`fabric.module` declares Variadic output port types on the op
signature:

```mlir
fabric.module @top(...) -> (!fabric.bits<32>, memref<8xi32>) {
  ...
}
```

The same allowed-type table applies to outputs. A `memref` output is a
subordinate/target memory capability export. Outputs are produced by the
`fabric.yield` terminator inside the body. The yield value count must equal
the module's declared output count, and yield values must conform to the
physical connection compatibility and memory-role rules below.

A module may have zero outputs (`-> ()`) or zero inputs
(`fabric.module @top()`).

## Body whitelist

`fabric.module` body may contain only:

* `fabric.pe` (both `[spatial]` and `[temporal]`)
* `fabric.switch` (both `[spatial]` and `[temporal]`; see
  `docs/spec-fabric-switch.md`)
* `fabric.mem` (an optional `[spatial]` or `[temporal]` Operation Engine,
  an optional Local Memory Service, or both; see `docs/spec-fabric-mem.md`)
* `fabric.fifo` (see `docs/spec-fabric-fifo.md`)
* nested named `fabric.module` template declarations; sibling top-level
  declarations remain in their enclosing symbol table rather than the body
* `fabric.instantiate` (binds a previously-defined fabric symbol
  into this scope; see `docs/spec-fabric-instantiate.md`)
* `fabric.boundary` (single op covering all three boundary directions
  -- `[s2t]`, `[t2t]`, `[t2s]` -- between the spatial `bits` domain
  and the temporal `bits_tag` domain; see
  `docs/spec-fabric-boundary.md`)
* `fabric.yield` (terminator)

`builtin.unrealized_conversion_cast` is **not** in the whitelist. All
fabric module values must come from a real fabric producer (a sub-
module result) or from the module's entry-block arguments.

The core SpatialCore tile matrix is:

| Tile kind | Spatial schedule | Temporal schedule |
|-----------|------------------|-------------------|
| `fabric.pe` | `fabric.pe [spatial]` | `fabric.pe [temporal]` |
| `fabric.switch` | `fabric.switch [spatial]` | `fabric.switch [temporal]` |
| `fabric.mem` Operation Engine | `fabric.mem [spatial]` | `fabric.mem [temporal]` |

`fabric.fu` is not a module-level tile kind. A FU is a functional-unit
container owned by a PE. Named FU templates may be visible through symbol
tables where the FU spec permits them, but that symbol placement does
not make FU a peer of PE, switch, or memory at the SpatialCore tile
level.

`fabric.fifo`, `fabric.boundary`, and `fabric.instantiate` are required
support constructs for buffering, spatial/temporal domain conversion,
and template reuse. They do not replace the core tile matrix.

The schedule predicate belongs to a `fabric.mem` Operation Engine. A
storage-only memory occurrence has no schedule and is not a third schedule
variant. Loom does not add a parallel `fabric.storage` op.

## Memory Capability Roles

Memory boundary direction determines protocol role without adding role
attributes or a second memory type family:

* every `fabric.module` `memref` input is a manager/requester capability;
* every `fabric.module` `memref` result is a subordinate/target capability;
* all `memref` operands of anonymous `fabric.mem`, in signature order, are
  manager-side imports, and all `memref` results, in signature order, are
  subordinate-side exports;
* `fabric.instantiate` preserves the target signature roles mechanically:
  target `memref` inputs become manager-side operands and target `memref`
  results become subordinate-side results.

Manager and subordinate are endpoint-relative roles, not permanent roles
attached to an SSA memref value. Legal connections include:

* forwarding a module manager input to one or more internal manager operands;
* connecting a subordinate provider result to a manager/requester operand;
* forwarding a subordinate provider result to a module subordinate output;
* using the same imported or provided capability at multiple endpoints.

The provider-to-requester case is ordinary memory-service composition. Any
`fabric.mem` subordinate result may feed any `fabric.mem` manager operand, and
an equivalent `fabric.instantiate` result may feed a manager operand.

The module export invariant is narrower: each yielded module `memref` result
must originate from any signature-derived subordinate result of an anonymous
`fabric.mem` or from a `memref` result of `fabric.instantiate`. Export
provenance is not restricted to the first subordinate result. Directly
yielding a module manager input is invalid because no subordinate provider was
introduced. The verifier does not impose token linearity or infer service
capacity from SSA use count.

A memory endpoint is a path to a physical service, not the service or a
software address space itself. One endpoint may carry several logical-memory
bindings when declared range or context capability distinguishes them. One
logical memory may be accessed or exposed through several endpoints. Explicit
SpatialMapping or SystemMapping records own those sparse many-to-many
relations; module SSA use count does not create or constrain them.

Operation Engine ports, an optional Local Memory Service, manager endpoints,
and subordinate endpoints are orthogonal `fabric.mem` capabilities. Their
active request-source-to-service-target relation is Mapping-selected runtime
configuration constrained by Fabric eligibility. Port presence must not imply
that every operation uses one fixed manager service. Runtime may install only
the configuration derived from the selected immutable Mapping; it does not
choose another relation.

## Dataflow And Fabric Boundary Symmetry

Canonical `dataflow.graph` inputs and outputs use the same three boundary
classes: `value`, `stream`, and `memory`. Value and stream are token-plane
contracts with different cardinality and publication rules. Memory is a stable
object or service capability whose transactions are carried by separate token
flows.

The corresponding memory directions are symmetric across the software and
hardware boundaries:

| Boundary | Memory input/import | Memory output/export |
|----------|---------------------|----------------------|
| `dataflow.graph` | the graph uses an externally supplied memory object or service | the graph provides a memory object or service capability |
| `fabric.module` | the module requests an external service through a manager endpoint | the module provides a service through a subordinate endpoint |

Input and output describe capability crossing the boundary. They do not
describe load versus store, ownership transfer, allocation, mutability,
coherence, or object lifetime. System transport channel direction is a
different coordinate system; memory protocol role must be stated as
manager/requester or subordinate/provider rather than inferred from a bare
`input-side` or `output-side` label.

SpatialMapping composes graph memory imports and exports with one or more
reachable module manager or subordinate endpoints through explicit Memory
Bindings, Access or Exposure entries, and service paths. This is not a
positional one-to-one graph-port-to-module-port rule.

## Physical Connection Type Compatibility

This section is the single source of truth for ordinary directed
physical connections inside `fabric.module`. Operation-specific specs
may constrain the declared ports of a resource, but they must not
redefine the semantics of a connection between two such ports.

The rule applies at all four module-level connection classes:

1. a module input endpoint connected to a resource input endpoint;
2. a resource output endpoint connected to another resource input
   endpoint;
3. a resource output endpoint connected to a module output endpoint; and
4. a module input endpoint connected directly to a module output endpoint.

The source and destination must have the same port kind:

* `bits` may connect only to `bits`;
* `bits_tag` may connect only to `bits_tag`;
* `memref` may connect only to an exactly matching `memref` type.

Memory endpoint roles determine whether a connection is boundary forwarding
or complementary provider-to-requester service composition. They do not add
another type-compatibility rule to the SSA value.

The fourth class is a token-plane boundary passthrough. It is legal only for
`bits` or `bits_tag`, obeys the same same-kind low-bit alignment rule as every
other ordinary connection, forwards valid with the payload, and propagates
ready in the reverse direction. It creates no resource endpoint, traversal,
capacity, buffering, or handshake owner. A direct module-input-to-output
`memref` passthrough remains illegal under the module export provenance rule.

An ordinary connection must not convert `bits` to `bits_tag` or
`bits_tag` to `bits`. Such a spatial/temporal domain transition requires
an explicit `fabric.boundary` resource. Protocol conversion, tag-value
remapping, buffering, arbitration, and clock, reset, or power-domain
crossing likewise require the corresponding explicit Fabric resource.

### Point-to-Point Transport Values

Direct module-body `!fabric.bits` and `!fabric.bits_tag` SSA values are
point-to-point transports. Each module entry-block argument and each
result of an operation directly in the module body may have at most one
consuming operand use owned by an operation in that body.
`fabric.yield` is a consumer under this rule. A transport may be unused.

Broadcast and fan-in require explicit routing resources. A switch may
route one input to multiple output ports, but those ports are distinct
SSA results and each result remains a point-to-point transport.

The rule does not apply to `memref` values, which represent memory
capabilities rather than token transports. It also does not inspect
values or uses inside nested `fabric.pe` or `fabric.fu` regions; those
regions follow their own verifier contracts.

### Same-Kind Width Semantics

Equal widths are not required for `bits` and `bits_tag` connections.
The only legal width-mismatch semantics is LSB alignment:

* `bits<Ws>` to `bits<Wd>` drops the high `Ws - Wd` source bits when
  `Ws > Wd` and zero-fills the high `Wd - Ws` destination bits when
  `Ws < Wd`;
* `bits_tag<Ws, Ts>` to `bits_tag<Wd, Td>` applies the same rule
  independently to the data field and the tag field;
* the tag-only form `bits_tag<0, T>` applies the rule to the tag field
  and carries no data bits.

Width normalization does not change the valid/ready transfer event.
It is intrinsic to the physical connection and is not an adapter,
buffer, configurable resource, or additional route hop. PnR and other
consumers must derive it from the connected endpoint types and must not
invent an adapter record for a pure same-kind width change.

The endpoint types are the canonical width owner. `fabric.module` has no
module-local address-width, memory-bus-width, or similar override. A resource
whose width is not mechanically determined by its typed interface must declare
that hardware fact in the resource's canonical typed capability.

For `memref<...>` no width relaxation is allowed. Source and destination
must have the same element type, shape, layout, and memory space. Exact type
equality remains independent from module export provenance: it does not make
a module manager input a subordinate provider.

### IR Expression

Both endpoint types must remain explicit in IR. A consumer signature or
an operation-specific `to <destination-type>` clause records the
destination endpoint type when it differs from the producer SSA type.
The `to` clause is connection typing, not a hardware resource.

Examples include:

```mlir
fabric.pe [spatial](%pa = %src : !fabric.bits<32>
                              to !fabric.bits<16>) -> ...

%0 = fabric.fifo %src [max_depth = 4, bypassable = false]
                : !fabric.bits<32> to !fabric.bits<8>

fabric.yield %v0 : !fabric.bits<32> to !fabric.bits<16>,
             %v1 : !fabric.bits<8>
```

The first two examples declare the consumer-side physical port width;
the final example declares the module output endpoint width. A `to`
clause is illegal for `memref` because `memref` connections require an
exact type match. Anonymous `fabric.switch` and `fabric.boundary`
incoming type lists use the same `source-type to destination-port-type`
form. Their result types remain the resource output-port types.

Resource-internal constraints remain owned by the corresponding
resource spec. For example, a switch may require all of its declared
ports to be uniform even though neighboring resources connected to
those ports may use different widths under this module-level rule.

### Handshake Dependency Projection

Fabric owns the ready/valid behavior of every transport endpoint, direct point
connection, resource-local traversal, configured mode, and physical
refinement. The persistent boundary signal vocabulary is:

```text
HandshakeSignalRef = (FabricTransportEndpointRef, Valid | Ready)
HandshakeDependencyArc = (source HandshakeSignalRef,
                          destination HandshakeSignalRef)
```

A resource owner may require internal conjunction or distribution nodes to
represent its equations without materializing their boundary transitive
closure. Such an `OwnerLocalHandshakeJunction` is a sealed-view node. It has no
`EntityId`, persistent reference, route capacity, endpoint meaning, or backend
identity and cannot escape the owning `HandshakeOwnerModel`.

The compiler factorizes the sealed semantic model into an immutable structural
template and one or more physical bindings:

```text
HandshakeStructuralTemplate {
  ordered owner-local junctions
  unique potential dependency arcs
  structural activation fragments
}

HandshakeOwnerModel {
  exact physical owner
  shared HandshakeStructuralTemplate instances
  ordered boundary signal and traversal bindings
  typed occurrence, row, and configuration activation bindings
}
```

The factorization is derived only through Fabric-owned definition relations.
FU occurrences use their exact `FabricFuTemplateRef`; Memory Operation Engine
occurrences use their exact `FabricMemoryEngineTemplateRef`. A switch input
uses its canonical occurrence-owned crosspoint order as one row shape. No
consumer may infer structural equivalence from names, paths, object addresses,
private hashes, or independently reconstructed property sets.

Sharing structural storage never shares a physical activation. Every concrete
occurrence binding retains its endpoint and traversal references. Every
Temporal switch `(occurrence, resident row, input)` retains distinct local
junction identity, Physical Tag, backpressure state, arbitration state, and
runtime activation even when all rows reference one immutable row shape.
Expanding the factorized model must produce exactly the same owner-local graph
and selected boundary reachability as compiling each occurrence and row
independently.

An activation fragment is an owner-local set of potential arcs selected by one
exact typed Fabric choice, such as one physical traversal, one FU-occurrence
capability row, one memory operation plan, or one transfer pattern together
with its declared physical refinement. A definition-level template alone is
not an activation because it does not identify a physical occurrence. A use
pattern alone is not an activation because it does not identify the selected
actor, role, mask, or configuration context.

The compiled owner graph must preserve the exact directed dependency relation
between boundary signals for every legal selection. It need not preserve an
internal circuit shape, and it must not materialize a boundary transitive
closure when a linear-size owner-local dependency graph represents the same
relation. Canonical owner-local node and arc order is a derived view contract,
not persistent Fabric identity. Structural templates and their instance
bindings are rebuildable in-memory views and are never serialized as a second
Fabric schema.

A directed point connection contributes producer-valid to consumer-valid and
consumer-ready to producer-ready arcs. Each resource owner resolves the exact
activation fragments from an exact configured view. These arcs are projections
of the resource's normative handshake equations; they are not separately
persisted fields, caller-supplied summaries, or backend netlist guesses.

The hardware-only Fabric root has no workload-selected route table, FIFO mode,
tag row, or refinement assignment. Fabric structural verification therefore
must not union mutually exclusive or disabled traversal alternatives and call
that union an active graph. It validates every alternative locally and rejects
only a root-complete cycle composed entirely of arcs that are unconditional in
every legal configured view. Configuration-dependent global closure belongs
to Mapping.

An owner-model boundary dependency is unconditional exactly when the owning
Fabric behavior projects that dependency for every value in its
already-declared legal configuration and refinement domains. This is a derived
universal property of existing typed domains, not a new predicate language or
persisted guard. The Fabric finalizer derives the root-complete unconditional
boundary relation once from the same owner models. It may stream or bit-pack
that one-time relation, but it cannot enumerate a global Cartesian product of
independent owner configurations. If a dependency is absent from even one
legal local alternative, it is configuration-dependent and is checked only
after Mapping selects a concrete view.

The SpatialMapping and SystemMapping verifiers derive the complete selected
combinational handshake graph from the exact root-complete Fabric and the
exact Mapping-selected routes, configured functions, service plans, and
physical refinements. They resolve each affected owner model, activate exactly
the selected fragments, and reject every directed cycle in that graph.
Unselected alternatives contribute no arc. Runtime tags, token values, traffic
assumptions, simulator delta iteration, HDL loop-breaking, or
implementation-specific signal defaults cannot excuse a selected cycle. A
compact owner-local graph and its fully expanded boundary relation must produce
the same cycle verdict. The exact gate and failure ownership are specified by
`docs/spec-mapping-verification.md`.

A cycle in the Fabric-owned unconditional graph fails Fabric finalization as
`Invalid(UnconditionalCombinationalHandshakeCycle)`. This diagnostic cannot be
used for a cycle that exists only after one concrete Mapping selection.

## Stateful Resource Lifecycle

Every stateful Fabric resource declares a canonical initial state as part of
its hardware behavior. A legal activation closes through the resource's normal
protocol transitions and returns its invocation-local state to that initial
state before the same state context is handed off or reconfigured. A normal
graph invocation therefore does not carry a second reset operand, token, or
operation.

Successful handoff additionally requires all accepted work to have retired and
all resource-owned queues and in-flight transactions for that state context to
be empty. Nontermination, deadlock, cancellation, and abnormal termination do
not satisfy this contract and must not manufacture completion. Resource specs
may refine the close and quiescence conditions, but they cannot replace this
lifecycle with a competing invocation-reset protocol. Mapping may overlap uses
of independently provisioned state contexts, but it cannot weaken a resource's
declared state-isolation or handoff requirements.

## Authoring-Only Visualization Metadata

An authoring-stage `fabric.module` may carry optional visualization hints in
its attribute dictionary. Regular topology helpers may emit attributes such
as `visual_layout` and `coordinates_semantic = false` so GUI and report tools
can draw arrays, meshes, rings, or pipelines in an expected shape. These hints
must not define connectivity, placement legality, routing cost, simulation
behavior, RTL lowering, or hardware cost.

Fabric finalization removes these attributes before canonical semantic
serialization and identity generation. A retained hint belongs to a removable
visualization projection that references the exact finalized Fabric identity;
it is not stored in the canonical Fabric artifact. Therefore adding, deleting,
or changing a hint cannot create a second canonical payload for the same
Fabric identity. See `docs/spec-mapping-visualization.md`.

The minimal module-local `visual_layout` form is an array of records:

| Field | Required | Meaning |
|-------|----------|---------|
| `node` | yes | Human-readable visual subject label. |
| `x` | yes | Display x coordinate, integer. |
| `y` | yes | Display y coordinate, integer. |

The `node` labels are visualization subjects only; they do not create SSA
values, ports, edges, or route endpoints. Duplicate labels or coordinates may
be rejected when the hint is converted into a visualization projection, but
must not change authoring-stage Fabric verification. If
`coordinates_semantic` is present, it must be `false`. A finalizer rejects a
claim that any such coordinate is semantic instead of silently changing the
Fabric identity contract.

## Verifier rules

* The body whitelist accepts only `fabric.pe` (both schedules),
  `fabric.switch` (both schedules), `fabric.mem` (scheduled when an Operation
  Engine is present and unscheduled when storage-only),
  `fabric.fifo`, `fabric.module`, `fabric.instantiate`,
  `fabric.boundary` (covering all three directions `[s2t]` /
  `[t2t]` / `[t2s]`), and the `fabric.yield` terminator. Any other
  op is rejected with a diagnostic that lists the allowed names.
* Each block-argument type must be one of the allowed module port
  types (`!fabric.bits<W>`, `!fabric.bits_tag<W,T>`, `memref<...>`).
* Each declared result type must be one of the same allowed types.
* The block-argument count and types must match the declared input
  types.
* The region kind is `Graph`.
* The op is `IsolatedFromAbove`: external SSA values cannot leak in;
  entry-block arguments are the only inputs.
* Every ordinary physical connection preserves port kind. Same-kind
  `bits` and `bits_tag` width differences are accepted with the
  canonical LSB-aligned semantics; `bits`/`bits_tag` transitions require
  an explicit `fabric.boundary`.
* A pure same-kind width difference does not require or imply an
  adapter resource.
* Every direct module-body `bits` or `bits_tag` transport source has at
  most one direct module-body consuming use. `fabric.yield` counts as a
  consumer; `memref` values and nested PE/FU region values are excluded.
* Memory roles are endpoint-relative. Module inputs and all `fabric.mem`
  memref operands are requester endpoints; all `fabric.mem` memref results,
  qualifying `fabric.instantiate` results, and module results are provider
  endpoints.
* A subordinate provider result may connect to a manager operand and may also
  be forwarded to a module output.
* Each yielded module `memref` result must originate from any subordinate
  result of an anonymous `fabric.mem` or a memref result of
  `fabric.instantiate`. Direct module-input passthrough is rejected.
* Imported and provided memory capabilities may have multiple uses; no token
  linearity check is applied to `memref` values.
* Existing operation and `fabric.yield` verifiers own operand/result shape,
  exact type matching, and `to`-clause legality.
* Clock and Reset slot inventories are canonical and each assignment targets a
  same-kind slot owned by this Module.
* Every boundary face and every `FabricModulePhysicalOwnerRef` has exactly one
  Clock and one Reset assignment; no extra assignment targets a foreign owner.
  ResourceState presence, not assignment presence, determines signal use.
* Every ordinary physical connection remains within equal symbolic Clock and
  Reset slots. A cross-slot connection is rejected while no explicit
  Module-local crossing resource exists.
* Every Module-target `fabric.instantiate` binds the callee's complete Clock
  and Reset slot inventory exactly once to existing same-kind slots of this
  Module. The composed effective slots of its boundary faces must match the
  adjacent parent-side assignments.
* `fabric.yield` inside `fabric.module` must have exactly as many
  operands as the module's declared result count, and each yield value
  must satisfy the physical connection compatibility rule against the
  corresponding module result type.

## Target Universe

The `fabric.module` target universe includes:

* all legal module input and output type combinations;
* the full `fabric.{pe,switch} [spatial|temporal]` tile matrix and scheduled or
  storage-only `fabric.mem` capability;
* FIFO resources;
* spatial-to-temporal, temporal-to-temporal, and temporal-to-spatial
  boundary ops;
* named and anonymous forms for supported module-body constructs;
* template instantiation rules for module, PE, switch, memory, and FU symbols;
* explicit Module-instance Clock/Reset slot correspondence with no inherited
  or inferred binding;
* point-to-point Graph-region SSA connectivity and same-kind
  width-normalization points;
* explicit symbolic Clock and Reset slots with complete boundary and physical-
  owner association, while ResourceState presence alone determines which
  owners consume Clock and Reset signals;
* endpoint-relative manager/subordinate memory roles, complementary
  provider-to-requester connections, sparse Mapping-owned endpoint bindings,
  and module export provenance.

The target universe does not include module-internal `fabric.link`.
System-level topology belongs to the typed Transport Architecture resources,
endpoints, and directed connectivity owned by `fabric.system`.

## Validation Anchors

Anchor-level validation covers one legal mixed token/memory module, rejection
of an unlisted body op, point-to-point fanout rejection, same-kind LSB width
normalization, a required explicit tagged-domain boundary, complete symbolic
Clock/Reset assignment, rejection of a missing or duplicate assignment,
rejection of a hidden cross-slot connection, exact nested-Module slot
composition, identity equivalence between an omitted Module relation and its
explicit single-domain rows, rejection of an incomplete or wrong-kind instance
binding, and
rejection of a manager import exported as a subordinate capability. Downstream
consumers resolve the exact finalized Fabric artifact and typed module
reference.

Tests do not freeze diagnostic wording, parser formatting, every port-width
combination, every whitelist member, or downstream cache layout.

## Unsupported Scope Policy

Unsupported module constructs must produce verifier diagnostics or
structured unsupported-scope records in downstream tools. A downstream
tool must not invent a missing module resource or replace module
connectivity with an implicit mesh, coordinate rule, or module-internal
link model.

## Relationships To Other Contracts

An exact `fabric.module` template is referenced by typed SpatialCore
occurrences and attachments owned by `fabric.system` AccCores, and by Mapping
artifacts where required. The system architecture owns one exact, structural,
one-to-one module-endpoint-to-AccCore-endpoint attachment for each fully
elaborated occurrence; the attachment is identity correspondence, not a route
or adapter. Value, stream, control, and completion endpoints retain their
typed transport contracts across it. Memory endpoints retain typed service
capability and are never recast as an untyped data plane. A module may be
produced by the ADG Builder, a builtin template, or an exact Fabric importer;
after finalization those source paths are semantically identical. Mapping,
CGRA models, hardware generation, Evaluation, and removable projections
consume the exact Fabric Artifact. System-level connectivity belongs to
`docs/spec-fabric-system-adg.md`; software-to-hardware binding belongs to
`docs/spec-mapping-artifact.md`.

## Cross-references

* `spec-fabric-pe.md` -- inner PE container (spatial and temporal
  schedules), including PE-side width and FU-boundary details.
* `spec-fabric-instantiate.md` -- the `fabric.instantiate` op that
  binds a previously-defined module, PE, switch, memory, or FU symbol
  into the current scope as a fresh hardware instance.
* `spec-fabric-reconfigurable-op.md` -- parameterized operation capability and
  the configured projections derived from that capability.
* `spec-fabric-hw-share-group.md` -- legal hardware-share groups for
  `fabric.op` `op_list` members.
* `spec-fabric-artifact.md` -- Fabric root variants, canonicalization,
  finalization, dependency closure, and persistent identity.
* `spec-fabric-resource-contract.md` -- shared typed state, use-pattern, and
  arbitration atoms embedded by concrete resources.
* `spec-fabric-fifo.md` -- finite buffering capability, Mapping-selected
  buffered or bypass traversal, and exact cycle behavior.
