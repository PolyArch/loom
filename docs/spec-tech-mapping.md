# TechMapping Generation

This document is the normative owner of the production TechMapping generator:
its invocation scope, derived candidate-row domain, deterministic search,
semantic work accounting, and generator outcomes. It does not own persistent
Mapping spelling, identity, or verification. Those contracts remain owned by
[Mapping Artifact](spec-mapping-artifact.md),
[Mapping Identity](spec-mapping-identity.md),
[Mapping Memory](spec-mapping-memory.md), and
[Mapping Verification](spec-mapping-verification.md).

TechMapping selects semantic realizations only. Placement, occurrence and
context selection, endpoint attachment, routing, resource allocation, tags,
physical refinements, and QoR belong to Spatial or System PnR and Evaluation.

## Invocation

One generation invocation consumes:

```text
TechMappingGenerationInputs {
  canonical_dataflow_program D
  graph_cover_scope covers
  fully_finalized_fabric F
  resolved_tech_mapping_config_view C
}
```

Production generation uses the common invocation-only diagnostic channel owned
by [Mapping Invocation Diagnostics](spec-pnr.md#mapping-invocation-diagnostics).
It defines no TechMapping-specific environment parser, logger, statistics wire,
or persistent trace.

`covers` is a canonical non-empty set of `GraphRef` values owned by exact
`D`. It is an invocation scope, not another artifact, identity catalog, or
configuration authority. Every reference must resolve to a reachable graph in
`D`, and the selected TechMapping root uses exactly the same set. Independent
scopes require independent invocations. A graph-free program has no legal
TechMapping generation invocation and remains outside this generator's domain.

`F` is one fully finalized Fabric root. The generator consumes only its sealed
typed capability views. It cannot use builder handles, symbols, operation
names, physical occurrence counts, coordinates, or backend-provider tables as
semantic capability.

`C` is the versioned component view mechanically derived from the complete
ResolvedConfig. It contains exactly the generator policy and its three
positive limits:

```text
ResolvedTechMappingConfigView {
  match_row_attempt_limit: uint64
  partial_cover_expansion_limit: uint64
  candidate_publication_limit: uint64
}
```

The view descriptor identity is `loom.tech_mapping.config`, version 1.0. Its
exact descriptor bytes are the ASCII bytes
`loom.tech_mapping.config.1.0`, without a trailing zero byte. Its canonical
view bytes are the three `u64be` values in the field order above, with no field
names or optional payload. `WorkUnitDescriptorRef` uses that same owner policy
descriptor and the zero-based ordinals zero, one, and two in the same order.
The central config library owns the single projector from exact ResolvedConfig
and uses the Common component-view digest contract.

The complete ResolvedConfig identity and the descriptor and digest of `C` are
recorded by the ordinary invocation manifest. `C` is not an artifact and does
not enter TechMapping identity.

### Root-Complete Central Adapter

The central DSE registry provides one deterministic root-complete adapter for
finite candidate sets. Its typed inputs are a canonical finite set of exact
Canonical Dataflow Artifact references and exactly one finalized Fabric
Artifact reference. For each Dataflow Artifact in canonical reference order,
the adapter strictly imports `D`, derives `covers` as the complete canonical
`GraphRef` catalog of that exact `D`, and invokes the production generator
above with the same exact `F` and resolved config view.

A graph-free `D` contributes no TechMapping candidate. `ProvenInfeasible` for
one `D` likewise contributes no candidate and does not suppress candidates
from another `D`. A generated finite prefix is complete for that owner
invocation even when its termination is `SemanticLimitReached`. An
`Incomplete` result stops the canonical Dataflow-candidate traversal and
retains only the already published candidate prefix. `Invalid` and internal
owner failures remain adapter errors and cannot be converted to an empty set.

Every output is an ordinary immutable TechMapping Artifact and receives one
mechanical lineage edge. The enclosing generator invocation already owns the
exact `D/F` inputs, so the mechanical edge has no parent or owner payload.
The adapter does not create a graph-cover Artifact, persist `GraphRef` values
in resolved config, scan the Artifact Store, or replace the explicit
independent-scope invocation. A caller that selects a proper subset of graphs
must continue to invoke the production owner with that exact non-empty scope.

## Typed Match Rows

The generator mechanically derives a finite domain of two closed row kinds:

```text
TechMatchRow = ComputeMatchRow | MemoryMatchRow
```

Row construction is driven by a closed `MatchRowSeed` union. A compute seed is
the prospective graph-local actor-to-template-node injection plus every
ordered operation-port and FU-boundary correspondence. A memory seed is the
prospective graph-local actor-to-Memory-Operation-Engine-template relation plus
every ordered template-relative operation-port, graph-boundary,
capability-alternative, and internal-edge witness. All seed fields are typed
references or owner-defined finite-domain values from exact `D/F`; a seed has
no identity or serialized form.

The prospective compute-seed domain begins with two owner-defined membership
relations. The exact concrete `fabric.op` must enable the actor's registered
`OperationSchemaId`, and the generated HSG family must admit the exact ordered
software-to-physical port correspondence. A schema-nonmember or a port tuple
outside that family relation is not a `MatchRowSeed` and consumes no
match-row attempt. Type, format, width, parameterized capability, pointer
layout, resource, complete FU topology, and boundary compatibility are row
admission checks after that membership projection; each failure there remains
one charged seed rejection. Implementations must obtain both membership
relations from the sealed OperationSchema/HSG/Fabric projections and cannot
reproduce them with operation names or private port rules.

For fixed-vector arithmetic and structural actors, that exact correspondence
uses the actor's standard vector types and registered semantic payload. Slice
extract maps source, ordered dynamic-position operands, and result roles;
slice insert maps inserted value, destination, ordered dynamic positions, and
result roles; shuffle maps its two operands and result. The family projector
validates element domain, resolved index width, block geometry, complete token
widths, and concrete resource capacity. TechMapping does not flatten a vector
into lane identities, infer a shuffle from equal widths, or scalarize an actor
that lacks a compatible row.

The prospective memory-seed domain begins with the exact relation returned by
the selected Fabric memory operation port's canonical capability matcher. A
port or capability alternative whose actor-contract domain, access domain,
payload and endpoint widths, role map, or operation-pattern relation does not
admit the exact Dataflow memory actor is not a `MatchRowSeed` and consumes no
match-row attempt. Once an exact alternative is selected, joint spatial-port
or temporal-residency capacity, temporal ingress distinguishability,
graph-boundary correspondence, and selected
internal-edge closure remain row-admission checks; each failure there is one
charged seed rejection.

Temporal ingress distinguishability is the row-local consequence of the
Fabric-owned row architecture: a Temporal engine selects one resident row by
matching the tag that arrives on that row's ingress endpoint, so two rows on
one operation port must present different tags there. A tag belongs to the
producing software edge, so two actors whose same operation-port endpoint is
driven by one producer always arrive together and no Mapping decision can
separate them. Such actors are not admitted into one row. A producer that is
itself a selected actor of the row never reaches the ingress and is excluded
from the test. This reads the Dataflow edge relation the generator already
consumes; it reads no tag assignment, route, or occurrence count. The generator and independent verifier must consume
the same sealed memory-operation-port relation rather than reconstructing any
part of it in Mapping.

Seed keys are the corresponding prospective persistent payload keys before
validation and Mapping-local identity assignment. The generator enumerates the
finite compute and memory seed domains lazily in lexicographic seed-key order.
This order is known before a seed is admitted as a row and is the sole order
for `match_row_attempt_limit`. A successful attempt yields one row; a failed
attempt yields one typed rejection and no partial row.

A `ComputeMatchRow` is one complete candidate Compute Realization over actors
from one graph in `covers`. It contains exactly the selected Fabric-owned FU
capability-template reference and every non-derived actor, operation-port,
boundary-port, and absorbed-edge correspondence required to materialize the
persistent record.

The selected FU topology must equal the canonical software-edge relation plus
the exact physical dispositions required by dead actor results. A result with
no canonical consumers creates no `mapping.compute_boundary` and no transport
obligation. If its selected operation capability does not prove that physical
production is suppressed, the selected template must carry that mapped result
to one FU output boundary. Exact `D`, the actor result-port map, and that
template edge then derive a mandatory PE output `Discard` for SpatialMapping.
A disconnected output is not a discard. No other unmatched template edge,
implicit sink, or invented software edge is legal.

A `MemoryMatchRow` is one complete candidate Memory Realization over canonical
memory actors from one graph in `covers`. It contains exactly the selected
`FabricMemoryEngineTemplateRef`, actor-to-template-operation-port and
capability-alternative relations, token/value/control graph-boundary endpoint
correspondences, and selected template-relative internal-edge witnesses
required to materialize the persistent record. It contains no concrete memory
occurrence, service, dispatch target, context, route, or configured-mode
encoding.

The persistent record owners define the field meanings. A row is an ephemeral
typed value with no `EntityId`, artifact identity, generic property map, raw
ordinal escape, or alternate serialization. Row construction must call the
same OperationSchema, HSG, Fabric capability, memory-access, width, vector,
floating-behavior, resource, and boundary relations used by independent
TechMapping verification. The generator cannot reproduce those rules in a
name table or weaker compatibility predicate.

Every row is internally complete and independently valid against exact `D`
and `F`. Rows that cross graph boundaries, mix compute and memory realization
kinds, omit a required terminal, invent an internal edge, or rely on a
provider fallback are not members of the candidate domain.

Rows have one canonical semantic key: the canonical persistent realization
payload before Mapping-local `EntityId` assignment. Exact duplicate rows
collapse to one domain member. The row domain is ordered by that key.

## Exact-Cover Search

A candidate is a canonical set of pairwise actor-disjoint rows that covers
every actor in every graph in `covers` exactly once. The selected rows must
also provide complete, nonconflicting internal-edge and exposed-boundary
classification. Materialization assigns Mapping-local identities only after a
complete cover has been selected.

The production search is deterministic and lazy:

1. visit MatchRowSeeds in canonical key order and charge one match-row attempt
   before validating each seed;
2. propagate every actor with exactly one remaining compatible row to a fixed
   point;
3. choose the uncovered actor with the fewest remaining compatible rows,
   breaking ties by canonical `ActorRef`;
4. visit that actor's remaining rows in canonical row order;
5. factor independent actor-row incidence components and search each component
   independently, expanding each component's partial covers by ascending
   demand estimate, then ascending operation-port imbalance of the selected
   rows, then canonical row order, where the demand estimate is
   the selected rows unioned with one remaining compatible row for every
   uncovered actor; and
6. enumerate the canonical lazy product of component covers without
   materializing the Cartesian product.

Row derivation first consumes its canonical seed prefix up to exhaustion or
`match_row_attempt_limit`. Exact-cover search then operates on exactly that
derived row prefix. Reaching the row limit does not make a row incomplete; it
only means later seed keys are outside this invocation's finite domain. Cover
search stops at `partial_cover_expansion_limit`, complete row-domain
exhaustion, or `candidate_publication_limit`, whichever applies first.

Selecting or rejecting one row during search consumes one partial-cover
expansion. A complete cover is independently materialized, verified,
finalized, and published before it enters the output set. A publication slot
is consumed before Artifact-identity deduplication, so cache state cannot
change the formal prefix.

Candidates are ordered by ascending realization demand, then by ascending
operation-port imbalance, then by the
lexicographic sequence of selected canonical row keys after component-product
normalization. Realization demand is the selected row count: one FU capability
instance per compute row and one Memory Operation Engine per memory row.
Operation-port imbalance is the largest number of memory rows that select one
exact (engine template, operation port) pair. One engine holds one configured
operation per physical operation port, so rows that select one port compete for
occurrences only with each other; concentrating a cover on the first canonical
port collapses the placeable supply to the occurrence count while the other
ports of the same template stay idle. Both terms are properties of the cover
alone. The generator still cannot read occurrence
counts, coordinates, or any other physical inventory fact, and demand order
neither admits nor rejects a row.

Row-key order alone is not a neutral tie-break here. Single-actor rows and the
first canonical operation port of the first canonical template hold the
smallest keys, so a key-ordered prefix systematically leads with the cover that
binds one actor per realization on one template. That cover maximizes the
occurrence demand every downstream SpatialMapping must satisfy, and a Fabric
that cannot supply one occurrence per actor rejects it before placement even
when a grouped cover of the same actors fits. Demand order leads with the cover
that asks for the fewest engines instead. Both orders are total and derived
only from the derived row prefix, so the same exact inputs, component view, and
limits produce the same candidate prefix independent of worker count,
completion order, container layout, cache population, or host timing.

The generator does not promise exhaustive enumeration. Even independent
binary row choices grow exponentially, so complete enumeration is not a
production semantic requirement. The configured limits define a deterministic
finite invocation domain; central DSE may evaluate and promote its completed
published candidates.

## Work And Outcomes

The three limits are distinct owner-local semantic work units. They are not
wall-time, memory, thread, or solver limits and cannot be inferred from one
another. Execution limits may interrupt the invocation but cannot change its
formal prefix.

The generator result is exactly one of:

```text
Generated {
  canonical non-empty candidate ArtifactRootReference set
  termination: SearchExhausted | SemanticLimitReached
}
ProvenInfeasible
Incomplete(typed reason)
Invalid(typed reason)
InternalError(typed reason)
```

`Generated` contains only complete verifier-clean `mapping.tech` artifacts.
`SemanticLimitReached` records that alternatives may remain outside this
invocation's finite prefix; it does not weaken any published candidate.
Search-state and termination facts remain in the invocation record and never
enter TechMapping bytes.

`ProvenInfeasible` is legal only when row derivation completed, every exact-
cover component was exhaustively searched, and no cover exists. If a semantic
or execution limit is reached before that proof and no candidate was
published, the result is `Incomplete(proof_not_established)`. A limit, timeout,
cancellation, or local heuristic failure can never prove infeasibility.

The TechMapping generator does not invoke CP-SAT as its primary search and
does not construct a placement, route, occurrence-capacity, or QoR model.

## Validation Anchors

Anchor-level tests cover:

- one forced compute row and one forced memory row;
- competing multi-actor rows with exact one-time actor coverage;
- stable finite prefix and work accounting under each semantic limit;
- exhaustive no-cover versus limit-before-proof outcome separation;
- connected-component lazy product without Cartesian materialization;
- exact ordered terminal and boundary correspondence;
- one dead control result that derives an explicit PE output discard without
  creating a software boundary or route;
- Fabric-finalizer rejection when an active operation output has no complete
  physical path, and TechMapping rejection when a dead result instead enters
  another mapped operation or any non-discard topology;
- rejection of a wrong-kind, foreign, incomplete, or duplicate row; and
- independent verifier agreement for every published candidate.

Tests must not build operation, type, row-count, or graph-shape matrices. They
must not preserve a search container, recursion layout, diagnostic wording,
or a particular parallel implementation.
