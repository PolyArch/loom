# Software Function To FU Synthesis

This document specifies reverse synthesis from exact software function graphs
to Fabric FUs governed by the same parameterized capability relation used for
forward materialization and TechMapping.

## Canonical Inputs And Outputs

The semantic input is a non-empty set `S` of canonical, typed, attributed
software function graphs. Every actor is interpreted by its registered
operation schema.

The output `F` is canonical Fabric capability consisting of:

* explicit FU topology and fixed FU boundary ports;
* concrete `fabric.op`, `fabric.mux`, and `fabric.demux` resources;
* one HSG-legal implementation family for each `fabric.op`;
* each operation resource's `op_list` projection, `hw_params`, physical ports,
  and typed constraints; and
* a finite normalized domain of Fabric-owned
  `FabricFuCapabilityTemplateRecord` values.

The output does not enumerate exact software parameter values, complete
software functions, or raw configuration words. Materialized members of the
supported function set are derived views, not persisted variant entities.

## Synthesize And Materialize Contract

Forward materialization and reverse synthesis share one contract:

```text
F  = Synthesize(S)
S' = { Materialize(F, legal_binding) }
require S subset-of S'
```

`legal_binding` selects a capability-template record and supplies exact
actors plus ordered actor-to-operation port and software-to-FU-boundary
correspondence. `Materialize` interprets that binding through the registered
operation schemas and Fabric capability relation. A successful synthesis must
prove that every member of `S` has at least one complete legal binding.

Both finite anchor cases are valid:

* `S = S'`, when the synthesized hardware implements exactly the inputs; and
* `S` is a strict subset of `S'`, when an HSG-legal parameterized template
  admits additional functions.

Every additional function must follow from the declared implementation
family, capability domains, typed constraints, physical topology, and a legal
binding. It must not arise from a Cartesian product of unrelated fields,
unspecified values, hidden drains, or post-hoc graph-isomorphism repair.

`S'` may be large or symbolic. Acceptance never requires complete domain
enumeration. `encodingCount` and `extraCapabilityCount = |S' - S|` are
applicable only when the relevant function set is finite and can be counted
exactly. They are optional Hardware DSE or Evaluation metrics, not universal
result fields, legality conditions, or implicit ranking criteria. Neither
`S'` nor its individual materialized variants are persisted or enumerated as
an artifact family.

## Coverage Witness

Synthesis acceptance needs a complete witness for each input function. The
witness contains the selected structural/capability template and the exact,
ordered actor/op/input/result/FU-boundary correspondence needed to
materialize that input. A hard-coded covered-function statistic is not proof.

The witness is an acceptance result for constructing `F`; it is not a
persistent Mapping Artifact and does not place an FU occurrence. TechMapping
later constructs its own exact realization for a concrete Canonical Dataflow
Program and the finalized Fabric artifact.

Before Fabric finalization, a synthesis witness may use a draft-local handle
to the normalized record. Successful finalization resolves that handle to the
canonical `FabricFuCapabilityTemplateRef`; the handle, witness, and synthesis
order do not enter persistent identity.

Coverage failure rejects the synthesized FU. Two independently produced
witnesses with the same normalized binding are duplicates. Distinct templates
or actor-to-resource correspondence remain distinct physical realizations
even when their software graphs are isomorphic; they do not create duplicate
semantic function variants.

## Capability Construction

Synthesis derives each operation resource from the registered operation
schemas and typed HSG registry:

* Operations may share one `fabric.op` only when one real implementation
  family admits all required software operation families.
* Each concrete resource binds that one family and enables only the needed
  subset through `op_list` and `hw_params`.
* `hw_params` records compact typed domains and correlations implemented by
  the hardware. It does not copy one exact tuple for every input actor.
* Fixed implementation parameters, variable semantic domains, arity limits,
  physical ports, and constraints must form one closed relation.
* Exact constants, predicates, types, and other actor attributes remain owned
  by the input software graphs and are supplied by legal bindings.

For each physical port position, synthesis may choose a payload capacity that
covers the supported exact software values. A wider physical `bits` path does
not change a function's exact software type. Port-kind compatibility and every
selected path segment's capacity remain mandatory.

Synthesis constructs only condition-relevant structural/capability templates.
Invalid assignments are excluded; irrelevant fields and equivalent raw
encodings are removed or canonicalized. Large constant, predicate, arity, and
similar semantic domains remain parameterized.

For one selected template and exact actor/op/port correspondence, normalized
semantic assignments must map injectively to complete typed and attributed
software graphs. If two valid assignments materialize isomorphic functions,
the synthesized capability is invalid; synthesis must repair the relation
rather than retain both variants for later deduplication.

## Explicit Mutually Exclusive Datapaths

Separate physical datapaths require separate `fabric.op` resources. If they
are mutually exclusive and share a software input, synthesis inserts an
explicit `fabric.demux` or equivalent selector for that input. If their
results share an FU output, synthesis inserts a matching `fabric.mux`.

All input selectors and result selectors for one realization must choose a
coherent branch. Direct FU SSA multi-use is real broadcast to every consumer;
it cannot mean that only one branch is active. Synthesis must not rely on an
inactive operation or unselected mux input to drain a broadcast token.

The finite template domain may correlate operation selection, routing, and
boundary correspondence. It must not expand independent local fields into a
configuration Cartesian product and then discard invalid projections.

## Parameterized Operations

Operation-specific behavior is expressed through registered schemas rather
than synthesis-only cases:

* sync uses ordered all-of input/result correspondence, from which the active
  physical-lane set is derived;
* software mux and demux actors preserve their runtime selector operands and
  ordered choice correspondence;
* FU-local `fabric.mux` and `fabric.demux` express static selected topology;
* constants bind exact typed values without enumerating their encodable
  domain; and
* predicates, fixed or configurable arity, streams, and other attributes are
  matched according to their exact operation schemas.

An omitted physical port is legal only when the operation schema and
capability relation guarantee that it neither consumes nor produces a token
and creates no backpressure obligation.

## Fabric Acceptance

Before returning a candidate, synthesis verifies at least:

* FU topology, SSA coherence, and explicit routing for mutually exclusive
  branches;
* unique implementation-family binding and typed HSG legality for every
  `fabric.op`;
* agreement among `op_list`, `hw_params`, physical ports, and constraints;
* validity of every exact actor binding under its registered operation schema;
* complete ordered input, result, and FU-boundary correspondence;
* physical port-kind and payload-capacity compatibility;
* normalization of structural/capability templates and configuration fields;
* absence of selected `sw_configs` in canonical Fabric; and
* a valid coverage witness for every member of `S`.

Acceptance checks the relation directly. It does not require every relation
point to have an index, every declared parameter value to appear in `S`, or
all of `S'` to be materialized.

## Edge Realization Boundary

Synthesis may make a Canonical Dataflow edge internal only by placing it in an
explicit configured-FU relation supported by the synthesized topology and
exact correspondence. Downstream Mapping may also use an explicit
configured-memory relation or temporal-PE register-file realization. Without
one of those typed relations, the edge remains an external transfer
obligation; physical co-location never absorbs it.

## Mapping And Finalization Boundary

Synthesis creates hardware capability, not a workload configuration.
TechMapping for exact `D + F` selects the exact finalized
`FabricFuCapabilityTemplateRef` and binds exact actors, attributes, ordered
operation ports, and FU boundary ports. SpatialMapping selects the exact
occurrence and instruction context for that realization. Complete Mapping
verification then derives the temporary `ConfiguredHardwareProjection`
through the sole definition and derivation operation in
`docs/spec-fabric-reconfigurable-op.md`.

Physical refinements are not an input to this projection. The current exact
Mapping contract has no
generic physical-refinement value codec, so strict Mapping import rejects every
nonempty refinement assignment before deriving configured hardware. A concrete
Fabric owner must first publish the domain's closed typed value codec and
admissibility relation; opaque bytes cannot substitute for that owner.

Neither synthesis nor TechMapping writes raw `sw_configs` back into canonical
Fabric. Fabric owns typed configuration-field meanings and domains;
`docs/spec-configuration-deployment.md` owns the only physical-image
finalization path, and `ConfigurationABI` alone owns physical encoding.
Hardware DSE that synthesizes a different FU must finalize a new Fabric
artifact before TechMapping that new `F`.

For the admitted scalar i32 add/sub followed by terminal sync graph-set, the
production reverse-synthesis workflow derives a one-AccCore System shell,
one normalized Module timing profile, and one packed System ConfigurationABI
from the finalized Module. It emits both one exact TechMapping per graph and
one exact whole-domain TechMapping. Separate root-complete Spatial PnR
invocations preserve the per-graph evidence while assigning the whole-domain
realizations to distinct resident instruction contexts of the shared FU. Only
that joint SpatialMapping enters the existing System PnR generator; the
portable SpatialCore RTL generator follows through ordinary DSE Plan edges.
Every graph must be reachable from a root thread; rootless or partially
unreachable inputs fail with a typed reverse-synthesis rejection before System
Mapping. A completed projection independently imports each TechMapping,
SpatialMapping, SystemMapping, and RTL HardwareImplementation, verifies exact
graph, Module, System, ABI, and SpatialCore ownership, reconstructs the unique
normalized timing and default packed ABI references, checks the canonical
portable operation-leaf specialization without publishing on the verification
path, and derives the physical configured-hardware projection as a Deployment
precondition. Deployment
remains its existing owner and consumes a selected SystemMapping plus explicit
executable and runtime-platform leaves; reverse synthesis does not invent
those selections.

`loom-dse --fu-reverse-synthesis-dataflow` is the public file-to-artifact
caller for this workflow. It publishes the finalized canonical Dataflow,
authors the ordinary resolved five-node plan, and emits the derived
`loom.fu_reverse_synthesis.workflow_evidence` JSON projection. Reusing the
same immutable stores, configuration, producer identity, input, and journal
root must retain every projected Fabric, Mapping, ConfigurationABI, and
portable RTL identity while reporting zero newly dispatched Generate owners.
The imported InvocationManifest outcome solely selects
`completed_selection`, `completed_no_feasible_candidate`, or `incomplete`;
only the first disposition carries a complete candidate root projection.
Unsupported operations and incomplete implementation constraints are rejected
by `verifyScalarIntegerAddSubFuSynthesisDomain` before plan publication with a
closed `FuReverseSynthesisFailure` value.

## Determinism And Diagnostics

Input ordering, resource ordering, template ordering, witness ordering, and
final Fabric output are deterministic. Concurrent workers may construct local
candidates, but final merge uses the complete normalized semantic key and does
not mutate a caller-owned IR context concurrently.

Success reports exact input coverage. Finite capability counts are reported
only when exact counting applies; otherwise they are explicitly inapplicable.
Failure uses a closed reason and does not retain the rejected candidate as
canonical Fabric. Cost and search diagnostics are Evaluation evidence and do
not define capability or Mapping semantics.

## Validation Anchors

Anchor tests should cover only:

* one finite case with `S = S'`;
* one finite case where `S` is a strict subset of `S'`;
* one symbolic typed constant or value-domain case that proves neither mode
  enumeration nor persisted function variants are required; and
* rejection of one incomplete binding whose edge would otherwise disappear
  through co-location; and
* the bounded rooted add/sub-plus-sync workflow through SpatialMapping,
  SystemMapping, portable RTL, Deployment, and independent journal/artifact
  replay, paired with a rootless typed rejection.

Tests must not pin printer whitespace, internal container layout, exhaustive
parameter products, universal capability counters, or implementation-specific
matcher traversal.
