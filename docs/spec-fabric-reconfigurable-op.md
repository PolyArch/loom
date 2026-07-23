# Fabric Reconfigurable Operations

This document specifies the parameterized capability contract for
`fabric.op`, FU-local configurable topology, exact TechMapping realization,
and derived hardware configuration.

## Semantic Ownership

Each fact has one semantic owner:

* A registered software operation schema owns exact actor semantics,
  including operation identity, types, arity, semantic attributes, and the
  interpretation of configurable parameters.
* A typed Hardware Sharing Group (HSG) owns the global legality of
  implementing specified software operation families with one real physical
  implementation family.
* A concrete `fabric.op` owns one indivisible physical datapath and scheduling
  resource. Its implementation family, `op_list` projection, `hw_params`,
  physical ports, and typed constraints jointly define its parameterized
  capability.
* `fabric.fu` topology owns the physical `fabric.op`, `fabric.mux`, and
  `fabric.demux` resources and their SSA wiring.
* Canonical Dataflow owns exact actor instances, types, and semantic
  attributes. TechMapping owns one exact realization by selecting a capability
  template and binding those exact actors, and therefore their exact types and
  attributes, through ordered actor-to-operation port and
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
new IR field. `A` includes the exact registered operation, function type, and
all semantic attributes. `P` preserves operand and result ordinals.

The hardware parameters in `Capability(R)` and an exact selected software
configuration jointly determine the configured software function. HSG
membership or `op_list` syntax alone never authorizes a type, attribute,
arity, operation family, or port relation that the complete typed relation
does not accept.

The former finite exact-mode model is retired; none of its representations
remain normative.

## Implementation Families And `op_list`

One software operation family may be legal in more than one HSG implementation
family, but each concrete `fabric.op` binds exactly one implementation family.
The HSG authorizes physical sharing; it does not grant every family member to
every concrete resource and does not prove resource-time exclusivity.

`op_list` is the readable projection of the software operation-family subset
enabled by the concrete resource. `hw_params` restricts that subset to the
typed parameter domains and correlations implemented by the resource. They
are two structured projections of one capability relation, not independent
authorities. Verification must reject:

* an `op_list` member outside the selected implementation family;
* an HSG member used by matching but not enabled by this concrete resource;
* a parameter domain that no listed operation schema interprets;
* duplicate or orphan declarations; and
* an incomplete relation between enabled families, parameters, physical
  ports, and constraints.

Operations that do not share a real implementation family require separate
`fabric.op` resources connected by explicit FU topology.

## `hw_params` And Physical Ports

`hw_params` stores hardware facts: fixed implementation parameters, supported
typed semantic parameter domains, configurable arity and port-selection
constraints, and legal correlations among configuration fields. It describes
a compact relation. It does not enumerate every exact actor, constant,
predicate, arity, or configuration bit pattern.

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
derived after TechMapping and SpatialMapping have selected all authoritative facts.
Neither `hw_params` nor canonical Fabric stores a workload's selected value,
mask, predicate, topology route, or raw configuration bits.

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

## Generic Operation-Schema Mechanism

All configurable operations use the same capability, matching, and
finalization mechanism. Operation schemas provide the operation-specific
interpretation; Mapping does not add parallel schemas for special cases.

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
capability templates. The domain covers choices of physical resources,
FU-local routes, and boundary correspondence. It does not enumerate large or
symbolic software parameter domains.

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

TechMapping persists the selected capability template and only the
correspondences that cannot be derived. It references exact Dataflow actors
rather than copying their semantic parameters. It does not persist a
configured-function copy, active masks, raw `sw_configs`, legality booleans,
candidate scores, or solver state.

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
  DeriveFields(Fabric,
               TechRealization(CanonicalDataflow, TechMapping, Fabric),
               PhysicalRefinement(complete Mapping, Fabric))
```

This derivation performs no search. It must reject a configuration field that
cannot be classified as a TechMapping-derived semantic/topology choice, a
SpatialMapping-selected semantic-preserving physical refinement, or a value
mechanically derived from those facts. Fabric defines the typed field meaning
and legal domain. The temporary projection is handed to the unique finalization
chain in `docs/spec-configuration-deployment.md`, where the exact
`ConfigurationABI` defines one canonical physical encoding. Fabric, Mapping,
and a backend do not emit an alternate image or invent an encoding.

## Validation Anchors

Anchor tests should pin only the stable semantic boundaries:

* an HSG member remains unavailable until the concrete capability and exact
  software configuration accept it;
* mutually exclusive branches require explicit FU demux/mux topology, and
  co-location does not absorb an external edge; and
* duplicate normalized semantic assignments are rejected while a declared
  semantic-preserving physical refinement leaves the software function
  unchanged.

Tests must not require exhaustive parameter enumeration, field Cartesian
products, printer layout, raw bit-pattern multiplicity, or a special Mapping
schema for one operation family.
