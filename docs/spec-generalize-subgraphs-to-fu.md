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

## Mined Composite FU Templates

The sections above define synthesis for an explicitly chosen input set. This
section defines how Hardware DSE chooses that set from software it already
compiles, so that the hardware subgraph is the common subgraph of the software
subgraphs and one Fabric realization covers several actors.

### Mining Input Set

The mining input `S` is the set of canonical Dataflow graphs of the selected
candidates of one application portfolio, or of the candidates of one
application. Graphs enter by canonical entity order. A foreign graph reference,
a duplicate graph, and an empty set are typed rejections. Mining reads only the
canonical token-plane producer/consumer relation and each actor's registered
operation-schema projection; it never reads Fabric, Mapping, or evaluation
state.

### Common Subgraph Relation

A mined shape is a connected induced subgraph of one graph's token-plane actor
relation.

* A node is one actor typed by its registered operation schema and its exact
  ordered operand and result types. Exact attribute payloads, including
  constants, predicates, and overflow flags, are not part of the node type;
  they remain owned by the software graphs and are supplied by legal bindings.
* An occurrence is a set of actors of one graph. The relation induced on that
  set is the shape's internal edge set: every token edge whose producer and
  consumer both lie in the set is internal, and a shape never omits one. An
  omitted internal edge would be an external transfer between two actors of one
  FU, which the Edge Realization Boundary does not admit without a second typed
  relation.
* An operand whose producer lies outside the occurrence is one FU input port. A
  result with at least one consumer outside the occurrence, including a graph
  egress terminal, is one FU output port. A result consumed both inside and
  outside keeps its output port, because direct SSA multi-use is real broadcast.
* An actor carrying a memory-capability operand or result is outside the token
  relation and is not an admitted node.
* Parameterized closure: a node's `op_list` is the observed schema set at that
  position, which this relation makes a singleton because node identity already
  includes the schema. The miner does not choose implementation families. The
  canonical capability derivation is the single owner of which family implements
  a set of actors, and a miner that guessed one would become a second owner of
  Fabric admission. Merging two schemas of one family onto one resource is
  therefore a property of synthesis over the actor set it is given, exactly as
  the Capability Construction rules already state, and never a mining decision.
  No mined candidate needs an FU-local selector or carries a mutually exclusive
  datapath, so the rank below needs no selector term.

The boundary profile is part of candidate identity, so every occurrence of one
candidate presents the same ordered FU boundary.

### Candidate Identity And Determinism

A candidate's identity is the canonical code of its node types, internal edges,
and ordered boundary ports. The code is the least one over the labelings that
place a connected prefix, so when several nodes are interchangeable at a
labeling step the search takes the least completion, and identity never depends
on actor identity, on graph order, or on the order in which growth reached the
shape. Two occurrences in different graphs therefore reach the same candidate
exactly when their induced shapes and boundaries agree. Interchangeable nodes
may bind different actors of different occurrences, which is a relabeling of one
shape and changes no port, no edge, and no capability.

Graphs are visited in canonical entity order, a candidate's occurrences are
reported in graph and actor order with each occurrence's actors in node order,
and candidates are reported in rank order with the canonical code as the final
tie-break. Enumeration is bounded in both the shapes and the embeddings it
retains; exhausting either bound is a typed mining failure and never a silently
truncated result.

### Size Bound

A mined candidate has at least two and at most `maximumActorCount` nodes, at
most `maximumBoundaryPortCount` boundary ports, and occurrences in at least
`minimumGraphSupport` members of `S`. The bounds are properties of the mining
request, not of the Fabric relation: they keep enumeration finite and keep a
template's FU boundary within what a PE can present.

Graph support is the prune that makes level-wise growth exact. Removing a
non-cut node from every occurrence of a shape yields a smaller shape present in
at least the same graphs, so support never increases with size and a shape whose
support is below the bound can have no admissible extension. The boundary-port
bound is not monotone, because adding a node can internalize an edge; a
candidate over that bound is therefore still extended and only withheld from the
ranked result.

### Rank

```text
coveredActorCount = actors of a greedy disjoint packing of the occurrences,
                    taken in canonical occurrence order
graphCount        = number of graphs of S holding at least one occurrence
score = coveredActorCount * graphCount
        - boundaryPortCost * boundaryPortCount
```

Coverage counts a packing rather than the union of the embeddings because two
embeddings that share an actor cannot both be realized: the union would let a
narrow shape with many overlapping embeddings outrank the wider shape that
actually absorbs the same actors. The packing is greedy and canonical, and it
is a ranking statistic over mined embeddings only; the Fabric coverage witness
remains its own owner and is never derived from it.

`coveredActorCount * graphCount` is the coverage of actors across `S` weighted
by how many members of `S` the template serves, so a template shared by the
portfolio outranks one that is hot in a single graph. The boundary-port term
prices the FU boundary, which is what a PE must present and route. The order is
score descending, then covered actors descending, then node count descending,
then boundary ports ascending, then canonical code ascending; it is total.

### Entering The Fabric Capability Domain

For one candidate, `Synthesize` receives the exact actor sets of its
occurrences and applies the rules above unchanged. Each node becomes one
`fabric.op` whose implementation family is the least registered family that
owns the node's schemas and whose canonical capability derivation admits every
occurrence's actor projection at that node, whose `hw_params` is the least
envelope that derivation returns for that actor set, and whose `op_list` is the
node's observed schema set. The node resources are
wired in the induced topology inside one `fabric.fu`, and the FU exposes exactly
the candidate's ordered boundary ports. Synthesis publishes one
`FabricFuCapabilityTemplateRecord` whose active nodes are all node resources and
whose active edges are the internal edges together with the boundary
correspondence. TechMapping's cover search consumes that record like any other
record; because it is composite, one selected realization binds `nodeCount`
actors instead of one.

Mining is family-agnostic, while synthesis is bounded by the canonical
capability derivation. A candidate whose node family has no inverse policy is
rejected with the existing typed capability-derivation reason. This keeps one
owner for admissible hardware and does not weaken the mined relation. A mined
shape whose internal relation contains a cycle is a loop recurrence; its FU
needs an explicit backedge, and the current synthesis profile rejects it with
the existing typed topology reason rather than authoring one implicitly.

Coverage uses the existing witness owner without change: one witness per
occurrence, carrying the selected capability template and the exact ordered
actor, operation-port, and FU-boundary correspondence, checked by the same
realization-closure verifier TechMapping uses. Mining introduces no second cover
algorithm and publishes no Mapping artifact.

Canonical finalization relabels FU graph nodes, so an authored node ordinal is
not a canonical one. The finalizer already publishes the authored-to-canonical
relation for the capability rows an author exposed; it publishes the FU graph
nodes of those same FUs in the same transaction, and the witness names each
node's operation resource through that relation. Synthesis never reconstructs
the relabeling from its own authoring order, and it does not obtain the witness
by running a Mapping search: the witness is an acceptance result for
constructing `F`, so deriving it from TechMapping would invert the order the
Synthesize and Materialize contract fixes and would make Mapping a prerequisite
of the hardware it is supposed to consume.

### Composite Supply For A Compute-Context Hall Deficit

A compute-context Hall deficit names demand groups and the capability templates
that admit them. When the templates admitting the deficient groups include a
composite record, and one admissible Spatial PE can gain an occurrence of it,
that occurrence is the supply the deficit prefers. One added Temporal context
lets the cover admit one more single-actor realization, so a relation whose
demand grows with its supply never closes; one composite occurrence instead
removes `nodeCount` actors from the demand per realization it covers. The
decision remains the existing FU-inventory change against the exact parent
Module, and the closure remains atomic: a single decision must make the complete
observed relation admissible. `docs/spec-dse-feedback.md` owns the direction
policy, its bound, and its diagnostics.

A mined template reaches a Module by being authored into it, not by a mutation.
Every hardware mutation resolves its prototype against the exact parent Module,
so the mutation vocabulary redistributes authored variety and never invents an
FU kind; a decision payload that carried a whole FU structure would make the
rewrite-config codec a second owner of Fabric capability. The one ADG Builder
materialization of a mined template therefore places its FU while a Module is
built, exactly as the builtin FU catalog places its own composite units, and the
ordinary FU-inventory decision then redistributes that occurrence to the PEs a
Hall deficit names.

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
authors the ordinary resolved five-node plan, and emits a removable JSON
projection marked by `projection_kind = fu_reverse_synthesis_workflow`. The
projection is not an Artifact schema or a second workflow owner. A successful
or incomplete execution projects its imported InvocationManifest receipt and
typed outcome; a preflight rejection instead projects the exact Dataflow and
ResolvedConfig roots plus a closed `FuReverseSynthesisFailure`, without
inventing an InvocationManifest.

Reusing the same immutable stores, configuration, producer identity, input,
and journal root must retain every projected Fabric, Mapping,
ConfigurationABI, and portable RTL identity while reporting zero newly
dispatched Generate owners. Recomputing under a distinct journal root must
dispatch all five Generate owners and reproduce the same required output
roots. The imported InvocationManifest outcome solely selects
`completed_selection`, `completed_no_feasible_candidate`, or `incomplete`.
The workflow projection separately names each required output and any empty
slot; it does not reinterpret an empty slot as proof of infeasibility.
Incomplete outcomes retain their exact reason, unsatisfied obligations, usable
Artifact roots, and Evidence roots from the InvocationManifest owner.

The current scalar add/sub profile does not admit software selector actors.
Selector-bearing graphs and unsupported operations are rejected before plan
publication with a closed `FuReverseSynthesisFailure` and durable preflight
projection. This bounded profile therefore does not claim the general
selector-preservation closure described above.

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
* one mining case over two graphs sharing one multiply-accumulate shape, which
  pins the ranked candidate's covered actor set and its ordered FU boundary, and
  the synthesis of that mined candidate back to its exact input actors; and
* the bounded rooted add/sub-plus-sync workflow through SpatialMapping,
  SystemMapping, portable RTL, Deployment, and independent journal/artifact
  replay, paired with a rootless typed rejection.

Tests must not pin printer whitespace, internal container layout, exhaustive
parameter products, universal capability counters, or implementation-specific
matcher traversal.
