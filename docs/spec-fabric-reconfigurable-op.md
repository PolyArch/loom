# Fabric Reconfigurable Operations

This document specifies the parameterized capability contract for
`fabric.op`, FU-local configurable topology, exact TechMapping realization,
and derived hardware configuration.

## Semantic Ownership

Each fact has one semantic owner:

* A registered software operation schema owns exact actor semantics. Its
  `OperationSchemaId` and closed `CanonicalDataflowActorOpInterface`
  projection own operation identity, types, arity, semantic attributes,
  instance validity, transition descriptor identity, and the interpretation
  of configurable parameters.
* A typed Hardware Sharing Group (HSG) owns the global legality of
  implementing specified software operation families with one real physical
  implementation family.
* A concrete `fabric.op` owns one indivisible physical datapath and scheduling
  resource. Its implementation family, `op_list` projection, `hw_params`,
  physical ports, and typed constraints jointly define its parameterized
  capability.
* `fabric.fu` topology owns the physical `fabric.op`, `fabric.mux`, and
  `fabric.demux` resources, their SSA wiring, and the canonical finite
  inventory of `FabricFuCapabilityTemplateRef` records selecting those
  resources and edges.
* Canonical Dataflow owns exact actor instances and their schema projections.
  TechMapping owns one exact realization by selecting a capability template
  and binding those exact actors, and therefore their exact types and closed
  semantic projections, through ordered actor-to-operation port and
  software-to-FU-boundary correspondence.
* SpatialMapping owns only Fabric-declared physical or QoR refinements that
  preserve the TechMapping realization's software semantics.
* The finalizer derives `sw_configs` from those authorities and the
  typed Fabric configuration-field domains. `ConfigurationABI` alone owns the
  physical encoding and programming representation of those fields.

For a concrete `fabric.op` resource `R`, the conceptual relation is:

```text
Capability(R) =
  Interpret(registered operation schemas,
            HSG(R's implementation family),
            R.op_list,
            R.hw_params,
            R.physical ports,
            R.typed constraints)

Supports(R, A, P) =
  Capability(R) accepts exact actor semantics A
  under ordered software-to-physical port correspondence P
```

This notation describes ownership and interpretation; it does not introduce a
new IR field. `A` includes the exact registered `OperationSchemaId`, function
type, and closed semantic projection. `P` preserves operand and result
ordinals. A Configured Function or adapter may cache that projection but may
not copy an arbitrary operation attribute dictionary or infer an alias from an
operation name.

The hardware parameters in `Capability(R)` and an exact selected software
configuration jointly determine the configured software function. HSG
membership or `op_list` syntax alone never authorizes a type, attribute,
arity, operation family, or port relation that the complete typed relation
does not accept.

The former persisted Cartesian exact-mode model is retired; none of its
representations remain normative. The derived semantic-field relation below
classifies only the behavior choices that require physical configuration. It
does not restore an actor-mode catalog.

## Implementation Families And `op_list`

One software operation family may be legal in more than one HSG implementation
family, but each concrete `fabric.op` binds exactly one implementation family.
The HSG authorizes physical sharing; it does not grant every family member to
every concrete resource and does not prove resource-time exclusivity.

The binding is an explicit typed `ImplementationFamilyId` attribute on the
operation. It is not inferred from `op_list`, port widths, an FU helper name,
or backend classification. Consequently, two resources may expose the same
operation schema through different implementation families, while one
resource can never become an implicit union of several families.

`op_list` is the readable projection of the software operation-family subset
enabled by the concrete resource. `hw_params` restricts that subset to the
typed parameter domains and correlations implemented by the resource. They
are two structured projections of one capability relation, not independent
authorities.

In particular, `op_list` is not the operation currently programmed for one
workload. TechMapping selects one exact admitted actor operation and parameter
point. Finalization derives that selection as typed `sw_configs`, and
`ConfigurationABI` alone encodes it as physical bits. Canonical unconfigured
Fabric therefore contains the complete enabled subset but no workload-selected
member. A singleton `op_list` requires no operation selector; its configured
field set may contain only another necessary parameter or may be empty.

An additive circuit feature is represented by the additional operation
schemas it actually enables, not by a parallel feature flag. For example, an
ordinary floating-to-integer converter may list only `arith.fptosi` and
`arith.fptoui`, while a converter with real clamp and NaN handling may also
list `llvm.fptosi.sat` and `llvm.fptoui.sat`. Both bind the same
`ScalarFloatToInteger` implementation family and use the same typed
integer/float format relation. `op_list` is the sole enabled-member authority;
the parameter record must not repeat saturation support.

Verification must reject:

* an `op_list` member outside the selected implementation family;
* an HSG member used by matching but not enabled by this concrete resource;
* a parameter domain that no listed operation schema interprets;
* duplicate or orphan declarations; and
* an incomplete relation between enabled families, parameters, physical
  ports, and constraints.

Operations that do not share a real implementation family require separate
`fabric.op` resources connected by explicit FU topology.

## `hw_params` And Physical Ports

Pointer-capable scalar integer parameters extend the ordinary integer-width
domain with a finite relation of exact pointer formats:

```text
PointerFormat = {
  address_space        : u32
  representation_bits : u32
  address_bits         : u32
  kind                 : StableIntegral
}
```

The relation is empty for an ordinary integer-only resource. It is nonempty
only when the concrete resource's `op_list` enables a pointer schema. Endpoint
capacity must cover `representation_bits`; GEP index operands must be admitted
by their own integer widths. `representation_bits`, `address_bits`, and the
software candidate's selected `index` width are independently validated and
must never be inferred from one another.

`hw_params` stores hardware facts: fixed implementation parameters, supported
typed semantic parameter domains, configurable arity and port-selection
constraints, and legal correlations among configuration fields. It describes
a compact relation. It does not enumerate every exact actor, constant,
predicate, arity, or configuration bit pattern.

For specification purposes, `FamilyCapabilityParams(F)` denotes the one closed
typed `hw_params` record schema selected by implementation family `F`. This is
notation, not another IR object, Artifact, registry, or generic container. The
family descriptor binds `F` to that schema exactly once. Family-specific
records may compose reusable typed atoms such as:

```text
IntegerWidthSet
FloatFormatSet
FloatBehaviorProfile
CastRelation
PredicateSet
```

The records contain only fields interpreted by their family and enabled
operation schemas. There are no unknown keys, optional field bags, generic
predicates, or independently editable configuration-field tables. For
example:

```text
ScalarIntegerAddSubParams {
  integer_widths
}

ScalarIntegerCompareMinMaxParams {
  operand_widths
  comparison_predicates
}

ScalarIntegerCastParams {
  width_pairs
  resolved_index_widths
}

ScalarSpecialMathParams {
  formats
  behavior
  accuracy_guarantee : SpecialMathAccuracyTier
}
```

The cast relation is a typed rule over the two finite domains rather than a
Cartesian enumeration. `op_list` remains the only concrete enabled-member
projection: these parameter records do not repeat add/sub, compare/min/max, or
cast operation membership.

`resolved_index_widths` is the normalized finite subset of `{32, 64}` that the
concrete cast resource admits when exactly one actor endpoint has MLIR `index`
type. `width_pairs` remains the directed integer representation relation. An
exact Structured/Dataflow candidate owns one selected canonical index width;
admission requires that width to occur in `resolved_index_widths` and requires
the corresponding directed pair in `width_pairs`. The Fabric relation does not
select a candidate's index width, and physical payload width remains transport
capacity rather than an implicit index-width choice.

For the initial scalar compute families, the family-level rule admits scalar
shapes while the concrete relation owns supported integer widths, floating
formats, compare and min/max policies, cast source/destination domains, and
their correlations. Family IDs do not encode widths or selected predicates.
The exact actor type and attributes select one point in the relation.

Integer overflow and exact flags that constrain legal software inputs do not
create hardware configuration fields when ordinary modular hardware behavior
already satisfies every defined result. Floating-point rounding, NaN,
subnormal, and fast-math admission are observable capability facts and must be
explicit in the concrete relation. A strict implementation may satisfy a
relaxed actor only when the registered operation schema proves that refinement;
the backend cannot infer it.

The `ScalarMath*` families use `ScalarSpecialMathParams` rather than the
ordinary scalar-float record. `accuracy_guarantee` is one value from the
`SpecialMathAccuracyTier` domain owned by the Structured compiler contract.
Fabric owns this guarantee as a property of the concrete circuit; it does not
copy the actor's selected accepted maximum. Admission requires:

```text
hardware accuracy guarantee <= actor accepted maximum
```

under that domain's stronger-to-weaker order. A correctly-rounded circuit can
therefore implement an actor accepting up to two ULP, while a two-ULP circuit
cannot implement a correctly-rounded or one-ULP actor. A non-correctly-rounded
hardware guarantee also requires `afn` in the capability's required fast-math
mask. Fast-math proves permission to approximate; it never selects or implies
an accuracy guarantee.

`FloatBehaviorProfile.fastmath` is the permission mask required by the
physical implementation, not a list of actor spellings it recognizes. The
registered floating admission relation requires
`hardware_required_fastmath` to be a subset of the actor's fast-math mask. A
strict implementation therefore has an empty requirement and refines an actor
that permits `nnan`, `ninf`, `nsz`, reassociation, contraction, reciprocal, or
approximate functions. An implementation that relies on one of those
permissions rejects an actor that does not grant it. This subset proof is
owned by the registered typed admission provider and is never inferred by a
backend.

Physical ports own transport kind and payload capacity. `!fabric.bits<N>`
does not identify a software type: the same width may carry multiple vector,
scalar, integer, or floating-point representations. Exact software type comes
from the registered actor schema and TechMapping binding.

Capability legality requires compatible port kinds. `bits` and `bits_tag` are
distinct and cannot be exchanged implicitly. For an untagged `bits<W>` path
carrying an exact software value that needs `N` bits, every selected segment
must satisfy `W >= N`. Low-bit-aligned widening zero-fills high bits and legal
narrowing truncates high bits without crossing below the exact semantic width.

`sw_configs` is not hardware capability. It is one typed configured-field set
derived after TechMapping and SpatialMapping have selected all authoritative
facts. A semantic field exists only when at least two admitted points require
different configured behavior in this physical resource. A width that merely
limits admission, for example, need not become a configuration field when the
same modular datapath realizes every admitted width without selecting it.
Neither `hw_params` nor canonical Fabric stores a workload's selected value,
mask, predicate, topology route, or raw configuration bits.

### Derived Semantic Field Relation

For every concrete `fabric.op`, the Fabric finalizer derives exactly one sealed
joint relation from the exact registered operation schemas, enabled `op_list`,
typed `hw_params`, physical ports, and constraints:

```text
FabricOpSemanticFieldRelation =
    None
  | Finite {
      canonical_behavior_keys[]
      admitted_actor_projection_to_key
      canonical_key_codec
    }
  | Direct {
      encoded_bit_count
      canonical_bit_domain
      admitted_actor_projection_to_bits
      canonical_bit_codec
    }
```

This is one concrete-resource relation, not another IR operation, persistent
record, HSG descriptor field, backend registry, or workload selection. It is
sealed into `ResolvedFabricOpCapabilityView`; the corresponding canonical
`FabricSemanticConfigFieldRef` inventory contains exactly one composite field
when the relation is non-`None`, and no field when it is `None`. Multiple actor
properties that jointly select physical behavior are components of one
canonical behavior key or one canonical direct-bit carrier. They are never
independent relations whose domains may be combined as a Cartesian product.

`None` means every admitted actor point has one equal physical behavior and
therefore creates no semantic configuration field.
`Finite` owns the exact behavior equivalence relation, the canonical finite
key domain, the total admitted-actor-to-key projection, and the canonical key
codec. `Direct` owns one fixed-width canonical bit carrier, its exact
schema-derived validity domain, a total admitted-actor-to-bits projection, and
the canonical bit codec without enumerating the domain. The domain may be the
entire `2^encoded_bit_count` carrier or a proper schema-derived subset. The
projector's semantic image equals that admitted domain; Fabric exposes the
canonical domain validator and ConfigurationABI cannot define another one. A
different relation kind, missing projection, noncanonical key, invalid direct
value, or ambiguous projection is invalid Fabric rather than a backend choice.

Behavior equivalence is physical behavior required by exact actor semantics,
not spelling equality or approximate QoR similarity. Non-defined result
refinements such as poison or undef do not create keys, modes, or RTL
sidebands. `Disabled` is not an additional Fabric behavior key. An ABI
`inactive_value` is any encodable member of the relation domain; the disabled
resource/topology contract, rather than that member's active semantics, proves
that the encoded value is unobservable.

For scalar integer division and remainder, each finite behavior key is exactly:

```text
(role, active integer width)

role = Quotient | Remainder
```

The concrete implementation family fixes signedness; `role` distinguishes its
quotient and remainder behaviors. The active width is the canonical singleton
semantic width selected by that actor point. A provider cannot collapse roles,
omit width, include a full operation-schema identity, or derive a second mode
number from `op_list` order. For a Direct vector index or selector, the fixed
carrier width and validity domain are resolved by the registered operation
schema before this relation is sealed; native host width and backend defaults
never enter the relation.

Mapping owns the authoritative actor and refinement selections. The relation's
projector mechanically derives one transient selected semantic value from each
admitted active selection, and Mapping's `ConfiguredHardwareProjection` carries
that derived value without becoming another semantic-selection authority. If
several selected actors or uses target the same physical field, equal projected
values collapse to that one field value; unequal projected values make the
complete Mapping invalid. ConfigurationABI persists only the resulting physical
encoding.

The relation result is fixed by the `loom.fabric 2.0` schema and the exact
canonical Fabric identity. A registry implementation identity may invalidate a
cached elaboration, but it cannot change the relation result for the same Fabric
identity. An incompatible relation change requires a Fabric major-version
change.

The configured operation projection is a closed sum:

```text
OperationConfiguration =
    Disabled
  | Active { semantic_configuration, physical_refinements }
```

`Disabled` carries no operation selection, semantic parameter, mask, or
refinement. ConfigurationABI alone owns physical inactive bits. The concrete
`fabric.op` schema uniquely owns its typed pipeline and holding
`ResourceState`s, canonical initial state, capacity dimensions, atomic
UsePatterns, stable typed requester order, and exact GrantPolicy or exact
refinement domain. One actor transition may atomically claim multiple operand,
pipeline, and result-holding states. Mapping may select a declared refinement
and bind typed workload values but cannot split the pattern or define another
scheduler.

## Stateless And Stateful Execution Contracts

There is no universal `fabric.op` pipeline or a parallel state-machine
framework. Every concrete operation resource uses the existing Fabric-owned
`ResourceState`, capacity, `UsePattern`, timing, progress, and grant
abstractions. The registered software operation schema remains the sole owner
of mathematical or logical actor transitions.

For the initial `CoreAluFu` and arithmetic portions of `MacFu`, each
semantically stateless scalar operation is implemented by a compute resource
with one registered elastic `ResourceState`, which is also its sole result
holding slot. The resource is therefore physically stateful and consumes its
assigned Clock and Reset even though the software operation has no logical
state. Acceptance consumes all required operands atomically only when that
state has capacity. A firing accepted in local cycle `t` publishes its result
in cycle `t + 1`. The latency is one cycle and the initiation interval is one
under downstream progress. A stalled result remains stable. Consumption of one
held result and acceptance of its replacement may occur in the same cycle.
There is no hidden input queue or drain.

This baseline does not apply to operation schemas with logical state, such as
`dataflow.stream`, `dataflow.carry`, `dataflow.invariant`, or
`dataflow.gate`. Their operation schemas uniquely own condition-dependent
operand consumption, result production, and logical state transitions. A
concrete Fabric resource owns only the physical state capacity, holding
resources, atomic transition use patterns, exact transition timing, and
backpressure behavior needed to implement that schema.

For a stateful transition, blocked result capacity cannot cause early operand
consumption or a state update. Only results produced by the selected
transition create output obligations; an inactive result never backpressures
that transition. Physical state and already published results remain stable
while blocked. Whether a transition with no result, a result-producing
transition, or a following transition can advance in a given cycle is stated
by that operation's exact use patterns rather than inferred from the
semantically stateless scalar baseline.

An FU containing resources for logically stateful and stateless operations
does not become one macro firing. Each configured Canonical Dataflow actor transition executes
independently through its selected operation resource. `MacFu` imports the
canonical `LoopCarry` capability for recurrence templates. The HSG registry
owns that family identity, and this document owns its Fabric resource
contract; the helper duplicates neither.

### Loop Control Resource Contracts

The loop-control implementation families are `LoopStream`, `LoopCarry`,
`LoopInvariant`, and `LoopGate`. `LoopControlFu` is only a Builder composition
helper. It neither creates a common implementation family nor owns a second
state-machine definition.

A concrete `LoopStream` resource has a closed typed capability containing:

* a non-empty set of scalar signless integer widths;
* exactly one fixed `dataflow::StreamStepKind`;
* a non-empty set of supported `mlir::arith::CmpIPredicate` values;
* exact physical operand and result roles; and
* its resource-state, use-pattern, holding, timing, and progress contract.

The selected predicate is a semantic `sw_configs` field. A resource that
supports several integer widths also selects the exact actor width so its
comparison, recurrence update, and modular-width behavior remain unambiguous.
The step kind is fixed hardware capability and is never repeated as a
software configuration field.

`LoopCarry`, `LoopInvariant`, and `LoopGate` are bit-preserving token-plane
resources. Their concrete port capacities may admit exact scalar integer,
floating-point, fixed-ranked vector, and scalar `!llvm.ptr<AS>` actor types
whose semantic payload fits the selected same-kind physical path, plus `none`
under Fabric's zero-payload control-token convention. A pointer payload also
requires the exact module-derived stable-integral `PointerLayout(AS)` and port
capacity for all `representation_bits`; the token resource does not acquire
pointer arithmetic or dereference capability merely by transporting those
bits. Equal payload width does not identify the semantic type; TechMapping
still proves exact operation type and ordered port correspondence. These
resources do not interpret payload bits and therefore do not enumerate the
Cartesian product of such types. Frontend `memref` bindings are not
token-plane payloads and are never admitted by these families.

The operation schema owns the logical transition cases specified in
`docs/spec-dataflow-part-1-streaming.md`. The concrete Fabric resource maps
each case to one exact atomic `UsePattern` over:

* the applicable context-state entry;
* required operand heads;
* execution resources;
* active result capacity; and
* any declared result-holding or in-flight state.

Only outputs active in the selected transition case claim capacity or create
backpressure. A blocked use pattern cannot consume an operand, update logical
or physical state, or publish a partial result. Fabric does not independently
decode the condition into another transition table.

The minimum logical-state storage implemented by the four families is:

| Family | Per-context state |
| ------ | ----------------- |
| `LoopStream` | `Idle` or `Running(current, limit, step)` |
| `LoopCarry` | initial/running mode bit; no carried payload storage |
| `LoopInvariant` | initial/running mode bit and one payload latch |
| `LoopGate` | initial/continuing mode bit |

Physical busy, in-flight, pipeline, and holding states required by the exact
timing contract are additional Fabric-owned `ResourceState`s, not additional
logical actor states. A temporal PE instantiates the declared state for each
resident `InstructionContextRef`; a spatial PE uses its sole context. The
`fabric.op` defines state shape and capacity but never creates a parallel
context identity or context-selection mechanism.

Timing is exact per concrete resource rather than inherited from a universal
stateful shell. The closed family-specific contract identifies result
publication offsets for active results, next-state availability, resource
initiation interval, and any holding or in-flight capacity needed to realize
them.

The initial `LoopCarry`, `LoopInvariant`, and `LoopGate` resources are
elastic-transparent:

* forwarding adds no registered cycle;
* initiation interval is one under downstream progress;
* there is no hidden output queue; and
* a result-producing transition commits only when all of its active outputs
  can accept the result.

Their inputs and state remain stable while stalled. A registered add followed
by this canonical carry therefore retains a one-cycle recurrence path and can
accept one recurrence transition per cycle under progress; the carry does not
insert another register stage.

`LoopStream` separately declares result-publication and next-state timing.
An add, subtract, or shift update may make the next state available each
cycle, while a multiply or divide update may be multi-cycle. Acceptance
atomically reserves all resources required by the selected transition. The
same context cannot perform its next transition until its next state is
available. Other contexts may interleave only when the concrete Fabric
capacity, initiation interval, and grant policy permit it.

## Resolved Capability View

Consumers may mechanically elaborate each concrete resource into an immutable
non-persistent C++ value:

```text
ResolvedFabricOpCapabilityView {
  occurrence
  implementation_family
  enabled_operation_schemas
  parameterized_capability
  physical_ports
  semantic_field_relation
  resource_state_and_timing_contract
  physical_refinement_domains
}
```

This view is derived solely from registered operation schemas, the normative
implementation-family registry, and the exact canonical `fabric.op`. It is a
cold elaboration result and may be cached as a compact hot-path structure for
verification, TechMapping, and RTL emission.

The view is not an IR operation, Artifact, persistent schema, configured
function, or semantic owner. It must not split the one joint relation into
independent dimensions, enumerate a Cartesian product of exact modes, or
preserve a backend-local support table. The exact Fabric identity owns the
semantic result. Registry implementation identity may participate only in
cache invalidation so stale derived values are recomputed and checked against
that result; serialized Fabric remains the authority.

## Generic Operation-Schema Mechanism

All configurable operations use the same capability, matching, and
finalization mechanism. Operation schemas provide the operation-specific
interpretation; Mapping does not add parallel schemas for special cases.
Every rule below consumes the same registered `OperationSchemaId` and closed
semantic projection used by graph admission and simulation.

* A configurable `dataflow.sync` capability describes its legal input/output
  lane capacity and active-set constraints. TechMapping's ordered operand and
  result correspondence selects the exact all-of software lanes. The active
  physical-lane set is derived from the image of that correspondence and the
  capability relation; it is not a persistent Mapping record. Its bit encoding
  belongs to `ConfigurationABI`.
* A `dataflow.mux` actor owns its runtime selector operand. TechMapping maps the
  selector and every software input-choice ordinal to ordered physical ports.
  All mapped choices remain runtime route obligations; no selector value is an
  `sw_configs` choice.
* A `dataflow.demux` actor symmetrically owns its runtime selector and data
  input. TechMapping maps every software output-choice ordinal. No one runtime
  output may be frozen as the actor's programmed configuration.
* A `dataflow.constant` actor owns its exact type and value. The capability
  relation describes the encodable representation and value domain, and the
  finalizer derives the actor's typed configuration value without enumerating
  that domain. `ConfigurationABI` encodes the value physically.
* A `dataflow.stream` capability has one typed `step_kind` as a fixed hardware
  parameter and a typed domain of supported predicates. The exact predicate
  remains actor semantics and is finalized through the generic relation.
  Different `step_kind` values require distinct physical operation resources.
* `dataflow.pack` and `dataflow.unpack` own exact fixed-vector and packed
  integer types with equal total bit width. Equal width does not authorize a
  different element type or shape. They may bind one shared implementation
  family only when the typed HSG registry and backend realize one genuine
  reinterpretation datapath.
* `dataflow.parallelize` and `dataflow.serialize` own their exact element type,
  lane count, mask, phase, ordered cardinality, and state transition
  semantics. Co-location in one FU does not imply physical sharing. A common
  HSG is legal only when one backend-supported stateful lane-buffer
  implementation realizes both operation families.
* Comparisons, fixed or configurable arity, and other semantic attributes are
  interpreted by their registered operation schemas and matched as exact
  actor semantics.

The fixed-vector structural families use this same mechanism. For
`vector.extract` and `vector.insert`, the family-owned typed projector derives
the row-major slice width, static bit offset, and compile-time stride of every
dynamic position from the exact actor types and registered position payload.
Dynamic positions remain runtime operands. For `vector.shuffle`, the projector
derives one ordered selector per result leading block from the registered mask;
each selector is exactly `Poison` or one source-block ordinal.

Only a choice implemented by programmable hardware becomes an `sw_configs`
field. A hardwired type, position, or mask produces no duplicate field. A
reconfigurable occurrence may expose an extract/insert mode, static offset,
shape mode, or shuffle selectors through its one Fabric configuration-field
schema. The projector mechanically re-encodes those fields from the actor;
the Fabric record never copies the actor's vector type, position array, or
mask as another semantic authority.

`ConfigurationABI` encodes each resulting typed field with `FiniteCodebook`
when the physical domain is small and finite or `DirectBits` when the field is
a fixed-width direct selector or offset. It does not enumerate all vector
shapes, positions, or shuffle masks. An unsupported projection is a typed
capability mismatch, not permission for a backend-private mode table.

The software `dataflow.mux` and `dataflow.demux` actors are runtime operations.
FU-local `fabric.mux` and `fabric.demux` are static configurable physical
routing resources. They share the generic typed finalization framework but not
selector semantics.

An inactive physical port creates no token, consumption, or backpressure
obligation only when the operation schema and capability relation explicitly
guarantee all three properties. A matcher must not infer inactivity from a
missing binding or compensate with a hidden drain.

## FU Templates And Explicit Topology

An FU exposes a finite, normalized domain of condition-relevant structural and
capability templates. The canonical owner and record shape are
`FabricFuTemplateRef` and `FabricFuCapabilityTemplateRecord` in
`docs/spec-fabric-identity.md`. The domain covers choices of physical
resources and FU-local routes. Exact software-to-FU boundary correspondence is
selected by TechMapping and is not copied into the Fabric record. The domain
does not enumerate large or symbolic software parameter domains.

Fabric SSA multi-use inside an FU is real token broadcast. Every consumer
participates in delivery and backpressure. When mutually exclusive physical
datapaths share FU inputs, each shared input must pass through an explicit
`fabric.demux` or equivalent declared selector, and shared results must pass
through a matching `fabric.mux`:

```text
input a -> demux -> add.a / mul.a
input b -> demux -> add.b / mul.b
                    add / mul -> mux -> FU result
```

The input demuxes and result mux must select one coherent branch. Connecting
both operations directly to each input describes broadcast to both datapaths,
not mutual exclusion. An inactive operation, an unselected mux input, or an
unbound configurable lane cannot act as an implicit token sink.

Conditional relevance is normalized in the template domain. Invalid
assignments are absent, irrelevant fields are removed or canonicalized, and
equivalent raw bit patterns do not become search choices. Distinct templates
or actor-to-resource correspondences remain distinct TechMapping candidates
even when their software projections are isomorphic, because they retain
different physical domains.

Within one selected template and one exact actor/op/port correspondence, the
normalized semantic assignment is injective with respect to the complete
typed and attributed software graph. Two valid assignments must not
materialize isomorphic configured functions; such a duplicate is a Fabric
schema or verifier error, not an enumerator deduplication opportunity.
Different physical candidates that materialize the same function remain
physical candidates, but they do not create additional semantic configuration
variants or persisted configured-function entities.

## Exact Realization Projection

The configured software function is instantiated from all of the following:

```text
Materialize(FU, template, actors, correspondence) =
  InstantiateCapability(FU physical topology,
                        selected structural/capability template,
                        exact Canonical Dataflow actors,
                        ordered actor/op/port correspondence,
                        ordered FU-boundary correspondence)
```

The projection contains exact operation identities, types, semantic
attributes, ordered operand and result edges, real fanout, and exact FU
boundary correspondence. Consequently, topology alone does not establish
function equality. Different predicates, constants, vector shapes, result
ordinals, or boundary maps may denote different functions on the same
physical graph.

TechMapping persists the exact Fabric-owned
`FabricFuCapabilityTemplateRef` and only the
correspondences that cannot be derived. It references exact Dataflow actors
rather than copying their semantic parameters. It does not persist a
configured-function copy, active masks, raw `sw_configs`, legality booleans,
candidate scores, or solver state.

The capability template owns no parallel state or timing descriptor. Its
active node set mechanically selects the concrete Fabric-owned
`ResourceState`, `UsePattern`, transition timing, progress, and physical
refinement closure of those nodes. An RTL provider consumes those exact
contracts through resolved Fabric views; provider availability cannot add
members, change the selected graph, or replace its state and timing semantics.

SpatialMapping cannot change this projection. It may select only closed,
Fabric-declared physical refinements such as a semantic-preserving pipeline,
bypass, latency, power, or QoR choice. A refinement that changes the software
graph, exact operation semantics, selected physical operation, FU-local
topology, or boundary correspondence belongs to TechMapping instead.

The configured FU is a physical graph and capability boundary, not a macro
firing boundary. Its active operations execute the corresponding Canonical
Dataflow actor transitions independently, subject to ordinary readiness,
commit, publication, and backpressure rules. `InstructionContextRef` names
only the resident configuration/runtime-state namespace in the parent PE; it
does not own this projection or replace actor transitions.

## Edge Realization Boundary

A Canonical Dataflow edge becomes realization-internal only when an explicit
configured-FU relation, configured-memory relation, or temporal-PE
register-file realization proves the connection. The configured-FU case must
follow the selected FU topology and exact actor/op/port correspondence. Mere
co-location in one FU, PE, instruction context, or physical resource never
absorbs an edge, and selectors or available local storage do not constitute an
implicit witness.

## Configuration Handoff

Mapping derives one temporary semantic projection by a cold, deterministic
operation:

```text
ConfiguredHardwareProjection =
  DeriveFields(CanonicalDataflow,
               TechMapping,
               Fabric,
               complete SpatialMapping)
```

This derivation performs no search. It must reject a configuration field that
cannot be derived from the exact Dataflow actor, TechMapping capability and
ordered correspondence, SpatialMapping occurrence and instruction context,
and the Fabric-owned typed field projector. Fabric defines the typed field
meaning and legal domain. A topology-sensitive family projector consumes the
exact TechMapping-owned ordered operand/result port correspondence; it does not
infer an active-port mask from actor arity. Values for one Mapping-selected
physical configuration slot must be unique; equal repeated derivations
collapse, while unequal values make the Mapping invalid. The complete Mapping
verifier owns this cold derivation and may retain its result only as a removable
sealed-view cache.

No generic refinement value type exists. A concrete Fabric resource that
exposes a non-singleton refinement domain must own its exact typed value codec,
legal set, and semantic-preservation proof. Until such an owner is implemented,
strict Mapping import rejects every nonempty physical-refinement assignment;
the configured-hardware projection does not receive a refinement row and must
not treat opaque bytes as a value.

CGRA admission requires the validated semantic projection as a cold proof and
does not copy its values into a simulator-owned runtime schema or decode
physical programming bits. The same temporary projection is handed to the
unique finalization chain in
[Configuration and Deployment](spec-configuration-deployment.md), where the
exact `ConfigurationABI` defines one canonical physical encoding. Fabric,
Mapping, a simulator, and a backend do not emit an alternate image, exact-mode
index, or independent decoder encoding.

## Validation Anchors

Anchor tests should pin only the stable semantic boundaries:

* one registered operation schema projection is consumed unchanged by
  Canonical Dataflow admission, Configured Function matching, and Fabric
  capability interpretation;
* an HSG member remains unavailable until the concrete capability and exact
  software configuration accept it;
* one registered operation schema may belong to two implementation families,
  while each concrete resource accepts only members of its explicit family;
* a multi-member `op_list` describes hardware capability while one exact
  selected member is derived in `sw_configs`, and a singleton capability has
  no redundant operation-selector field;
* one joint `None`, finite, or direct semantic-field relation derives field
  need, joint domain, projection, and codec without a backend mode table,
  including exact finite div/rem keys `(role, active width)`;
* mutually exclusive branches require explicit FU demux/mux topology, and
  co-location does not absorb an external edge; and
* static and dynamic vector slice actors plus one poison-containing shuffle
  derive the exact typed physical fields without a mask table, shape table, or
  redundant field for a hardwired fact;
* one semantically stateless scalar firing uses its registered elastic
  `ResourceState` and exact one-cycle contract, while one logically stateful
  transition is governed by its operation-specific state and use patterns; and
* equal repeated semantic assignments to one physical configuration slot
  collapse to one value, conflicting assignments are rejected, and a declared
  semantic-preserving physical refinement leaves the software function
  unchanged.

Tests must not require exhaustive parameter enumeration, field Cartesian
products, printer layout, raw bit-pattern multiplicity, or a special Mapping
schema for one operation family.
