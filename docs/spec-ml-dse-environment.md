# ML DSE Environment

This document defines the interaction boundary through which a learned search
policy explores Loom's design space. The environment presents one episode as a
sequence of typed design decisions over one subject, either a single
`fabric.module` or a complete multi-core `fabric.system`, optionally together
with the software expression of the workloads that subject must run. It scores
each transition against a resolved objective and retains the decision sequence
for later replay.

Which decisions an episode may take is configuration rather than schema. The
environment binds a set of exploration domains, each of which is one existing
candidate generator, and a domain the configuration omits contributes nothing.
A parameter family over an existing parent kind, existing reference kinds, and
an existing completion form is one catalog member and one obligation-table row.
A family that needs a new reference kind, a new value form, or a new completion
form extends the observation projection and the completion algebra as well;
what appending never requires is widening a decision union or adding a
permission flag.

## Ownership

Every fact the environment exposes resolves to one exact owner:

- [Evaluation and DSE](spec-dse-feedback.md#candidate-generators) owns
  candidate-generator kinds, their closed decision unions, decision domains,
  canonical decision order, and lineage contributions;
- [Evaluation and DSE](spec-dse-feedback.md#objectives-and-quality-gates) owns
  `ObjectiveDimension`, `ExactAffineQuantization`, `ObjectiveVector`,
  `WeightedLevel`, `TotalOrdering`, `SearchEnergyRef`, and quality gates;
- [ADG Builder](spec-adg-builder.md#failure-atomic-finalization) owns draft
  construction and authoring-boundary checks;
- [Fabric Artifact](spec-fabric-artifact.md#finalization-and-publication) owns
  the finalization pipeline, canonical bytes, root identity, the verified
  pre-publication closure, and publication;
- [Fabric Identity](spec-fabric-identity.md#owner-local-reference-kind-catalog)
  owns the local-reference kind catalog and, at
  [Mapping-Visible Entity Catalog](spec-fabric-identity.md#mapping-visible-entity-catalog),
  `EntityId`;
- the `fabric.*` specifications own every typed hardware parameter domain;
- [SCF To DFG](spec-compiler-part-3-dfg.md#canonical-dataflow-rewrite-catalog)
  owns the closed Canonical Dataflow rewrite catalog, its decision wire, its
  canonical decision order, and the equivalence obligations every rewrite
  preserves;
- [Place And Route](spec-pnr.md#objective-projection) owns the Mapping
  violation catalog `V` and measure catalog `G`;
  [Invocation Contracts](spec-pnr.md#invocation-contracts) owns invocation
  outcomes;
  [Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism)
  owns search policy and the seeded PRNG protocol; and
  [Evaluation Transaction](spec-pnr.md#evaluation-transaction) owns the
  ephemeral probe adapter and its `rebuild`, `probe`, `commit`, and `discard`
  operations;
- [TechMapping Generation](spec-tech-mapping.md) owns TechMapping construction
  and its exact `D` and `F` binding;
- [Evaluation Metrics](spec-evaluation-metrics.md) owns metric kinds, units,
  observation forms, and `ExactRatio`;
- [Resolved Configuration](spec-config-ssot.md#component-views) owns
  component-view framing, canonical view bytes, and `component_view_digest`,
  and at [Cache Dependencies](spec-config-ssot.md#cache-dependencies) owns the
  cache-family contract;
- [ML Environment Core](spec-ml-core-environment.md) owns the observation
  container and its combined node-and-link space, the action surface and
  masking algebra, the step accounting identity, the
  termination-versus-truncation rule, the reward boundary, the PRNG preimage
  shape, the `loomml` package layering and its interaction contract, and the
  benchmarking obligations every harness satisfies; and
- [Full-Stack Architecture](spec-loom-stack.md) owns external dependency
  pinning and the fast-evaluation cascade.

The environment owns only the episode protocol, the exploration-domain catalog,
the enumeration rule, its action index contract, and the schedule-derived
reference rule, the DSE observation
catalogs and their curriculum-neutral partition, the workload-feasibility
invariant, the step and rejection outcome algebra, the visited set and the
per-state rejection mask, the trajectory retention record, and its own
benchmarking stages.

## Nonsemantic Boundary

The environment is a nonsemantic search harness, on the terms
[Nonsemantic Boundary](spec-ml-core-environment.md#nonsemantic-boundary) states
for every ML environment. It defines no hardware action language, no mutable
candidate intermediate representation, and no second design space. Every
decision it applies is an existing owner-typed candidate-generator decision,
every legality answer comes from the ordinary Builder, Fabric finalizer, and
Dataflow rewrite owners, and every mappability answer comes from the ordinary
Place and Route owner.

No episode subject is published while the episode holds it. The single
exception is the component a `Reattach` completion publishes so the subject can
depend on it, defined under Step. That component is published by its own
generator through the ordinary path, with the `CandidateDecision` lineage
contribution that generator's descriptor owns, so it is an ordinary
intermediate candidate rather than a publication mode this document invents;
what the environment withholds is selection, not lineage.

The environment adds nothing to the subjects deliberately absent from the first
version by
[Full-Stack Architecture](spec-loom-stack.md#deliberate-first-version-boundaries).

An environment run is therefore never an authority for candidate selection. A
design the environment discovers becomes a real candidate only by replaying its
trajectory through ordinary `Generate` and `Promote` plan nodes, as defined by
Trajectory Retention And Replay below.

## Exploration Domains

An exploration domain is one existing candidate generator whose decisions the
environment may enumerate. Domains are the unit of permission and the unit of
extension: a domain is permitted exactly when the resolved configuration binds
it, and a new parameter family is added by appending one member to the closed
catalog below and one row to the obligation table, to the extent the intro
states, and never by widening a decision union or adding a parallel switch.

```text
ExplorationDomainKind =
    SpatialTopology           // 0
  | SpatialMicroarchitecture  // 1
  | SystemComposition         // 2
  | DataflowRewrite           // 3
```

Ordinals are stable. A new kind appends; reordering, deleting, or repurposing
one is an incompatible change to this document's schema version. Each kind
binds its generator's own resolved configuration and defines no decision, match
rule, or value domain of its own.

One table is the complete per-domain contract. Every rule elsewhere in this
document quantifies over its rows rather than naming a domain, so appending a
domain is appending a row:

| Domain | Generator | Operates on | Required subject | Completion |
| --- | --- | --- | --- | --- |
| `SpatialTopology` | kind 13 `spatial_topology_rewrite` | one `fabric.module` | either | `Reattach` via `SystemComposition` under `System` |
| `SpatialMicroarchitecture` | kind 14 `spatial_microarchitecture_rewrite` | one `fabric.module` | either | `Reattach` via `SystemComposition` under `System` |
| `SystemComposition` | kind 15 `system_composition_rewrite` | one `fabric.system` | `System` | none |
| `DataflowRewrite` | the [Dataflow rewrite generator](spec-dse-feedback.md#deterministic-work-candidate-sets-and-cache) | one Canonical Dataflow Program | either | `ReplaceParent` |

"Operates on" is both the exact parent a decision derives from and the kind of
child it produces; every registered generator preserves its parent's kind.
"Completion" is the mechanical follow-up defined under Step, and a `Reattach`
names the domain that performs it.

Every episode's probe requires the Spatial PnR and TechMapping views
unconditionally, and a `System` episode's probe requires the System PnR view
its subject arm carries. None of those is per-domain, so no row names a view
and no rule derives one from the table.

A domain is admissible only when its row states every column and its decisions
are representable, meaning every owner-local reference they can carry resolves
to exactly one node under the target-closure rule of Combined Node And Link
Space. Representability is a property of the domain and the projection
together, so a future domain is admitted by extending both.

Adoption is then one rule quantified over the table rather than a list of
per-domain checks: for every bound row, the subject it requires matches
`episode_subject`, and the domain any `Reattach` completion names is itself
bound; and no two bindings share a domain kind. Binding a Spatial row under a
`System` subject without also binding `SystemComposition` is therefore invalid
at adoption, because its completion could not run.

Each row's decision union, its typed value domains, its canonical decision
order, and the rules that make one of its domain members well formed belong
entirely to that generator. The hardware unions of kinds 13, 14, and 15 are
defined by [Candidate Generators](spec-dse-feedback.md#candidate-generators);
the closed Canonical Dataflow rewrite catalog and its member ordinals are
defined by
[Canonical Dataflow Rewrite Catalog](spec-compiler-part-3-dfg.md#canonical-dataflow-rewrite-catalog),
which states that no other document may add, remove, rename, or reorder a
rewrite kind. This document reproduces none of them, so a catalog revision does
not strand a second copy here.

What the environment adds is only the enumeration rule: for a bound row, its
generator's decisions in that generator's own canonical order, over the exact
parent the row's "Operates on" column names. Every owner-local enumeration rule
survives unchanged, including the ones that make a decision's availability
depend on the current Fabric, and every rewrite's equivalence obligation is its
catalog owner's. A software decision therefore changes how a workload is
expressed and never what it computes.

`DataflowRewrite` explores the complete catalog over the workloads the episode
already holds; there is no environment-local catalog-kind selector, because a
subset selector would make the enumeration a filtered view of the generator's
canonical order rather than that order itself. Its binding carries the
generator's own immutable component view,
`loom.dataflow_rewrite_generator.config.1.1`, whose `scope_expansion_limit` and
semantic value domain are owned by
[Resolved Configuration](spec-config-ssot.md#schema-ownership). The other three
rows bind `SpatialTopologyRewriteConfig`,
`SpatialMicroarchitectureRewriteConfig`, and `SystemCompositionRewriteConfig`
respectively, the exact descriptor-owned records defined by
[Candidate Generators](spec-dse-feedback.md#candidate-generators). No row
introduces an environment-authored configuration record.

Two families remain outside this contract rather than represented by inert
domains. `FabricTemplateConfig` of kind 12 is absent because a template
expansion selects an episode seed rather than advancing an episode.
`ImplementationFlowConfig` of kind 16 is absent because it produces a
`HardwareImplementation` rather than a design the feasibility gate can close
against a workload. Either is reopened by appending a domain kind that can
state all five obligations above.

## Resolved Environment Configuration

The environment consumes one immutable component view with schema descriptor
bytes `loom.ml_dse_environment.config.1.0`, following the framing, canonical
byte representation, and digest contract owned by
[Resolved Configuration](spec-config-ssot.md#component-views):

```text
ResolvedMlDseEnvironmentConfigView {
  episode_subject: EpisodeSubject
  enabled_domains: canonical nonempty set<ExplorationDomainBinding>
  episode_policy: EpisodePolicy
  probe_policy: ProbePolicy
  selected_objective_closure: SelectedObjectiveClosure
  observation_policy: ObservationPolicy
  determinism_policy: EnvironmentDeterminismPolicy
}

EpisodeSubject =
    SpatialModule
  | System {
      admissible_modules: canonical nonempty set<ArtifactRootReference>
      system_pnr_config_view: ResolvedPnrConfigView
    }

EpisodePolicy {
  seed_roots: canonical nonempty set<ArtifactRootReference>
  workload_pool: canonical nonempty set<WorkloadBinding>
  workloads_per_episode: positive uint64
  start_retry_bound: positive uint64
  step_bound: positive uint64
  consecutive_rejection_bound: positive uint64
  visited_state_retention: uint64
  rejection_reward_code: uint64
  stop_reward_code: uint64
}

WorkloadBinding {
  dataflow: ArtifactRootReference
  tech_mapping: ArtifactRootReference
}

ProbePolicy {
  spatial_pnr_config_view: ResolvedPnrConfigView
  tech_mapping_config_view: ResolvedTechMappingConfigView
  warm_start_from_parent: bool
}

ObservationPolicy {
  enumeration_bound: positive uint32
  include_placement_edges: bool
  include_resource_states: bool
}
```

`EpisodeSubject` carries what its arm needs rather than leaving subject-keyed
fields elsewhere to be required-or-forbidden by prose. The `System` arm holds
the finalized Module candidates kind 15 selects from and the System PnR view
that arm's probe requires, so neither can be present without the subject that
uses them or absent under the subject that needs them.

`EpisodePolicy` owns every remaining input `reset` consumes, so an episode
start is a function of this view and the copy coordinates alone. Every
`seed_roots` member's root kind must match the subject arm, and
`workloads_per_episode` must not exceed the pool size.

`tech_mapping_config_view` is unconditional: a TechMapping binds exact `D` and
`F`, so every advancing step regenerates at least one, whatever the bound
domains are.

Membership in `enabled_domains` is the permission. A domain the set binds is
explored; a domain the set omits contributes nothing to the enumeration, no
masked entries, and no observation blocks. There is no separate permission flag
beside the binding, because a permitted domain with no configuration and a
configured domain that is forbidden are both states this contract has no use
for.

`SelectedObjectiveClosure` and `ResolvedPnrConfigView` are the exact records
owned by
[Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism),
materialized as the selected transitive closure with view-local ordinals under
the projection rule owned by
[Component Views](spec-config-ssot.md#component-views).

This view therefore contains two objective closures, and their roles are
disjoint. The closure inside `probe_policy.spatial_pnr_config_view` governs
PnR's internal search only: its total ordering, search energy, and focused
closure steer the probe toward a closed Mapping and never leave it. The view's
own `selected_objective_closure` is the sole reward authority; its search
energy is the value the reward differences, and it may carry dimensions the
probe has no notion of, such as a parameter-backed predicted metric. Neither
closure derives, overrides, or defaults the other.

Adoption requires them to be consistent where they overlap. Every
`MappingViolationSource` and `MappingMeasureSource` dimension in the reward
closure must reference a descriptor that the probe view's temporary-violation
policy and measure catalog admit, so a reward dimension cannot read a fact the
probe never produces. A dimension referencing an obligation tagged `HeldOut` is
invalid here for the reason it is invalid everywhere. A reward closure whose
Evaluation-sourced dimensions no registered parameter contract satisfies is
invalid at adoption. A closure whose dimensions are unsatisfied by any selected
closure is invalid at adoption rather than at the first step that needs the
missing value. The set of parameter contracts an episode invokes needs no field
of its own: it is exactly the set the closure's Evaluation-sourced dimensions
reference, and a field beside it could only disagree.

The canonical view encoder writes fields in the schema order above, under the
encoding and adoption rules
[Resolved Configuration](spec-config-ssot.md#component-views) owns. This view
adds only that its optionals use a `u32be` absent or present discriminant
followed by the payload when present.

The PRNG preimage shape, the meaning of `effective_seed` and the copy
coordinates, the `reset` seed override, and the exclusion of policy sampling
from every environment stream are owned by
[Determinism And Copy Coordinates](spec-ml-core-environment.md#determinism-and-copy-coordinates).
This document adds only its own domain separator, which is

```text
ASCII("loom.ml_dse_environment.prng.sha256_seeded_xoshiro256starstar.1.0")
```

and its stream purpose ordinals, which are `SeedSelection = 0` and
`WorkloadSelection = 1`.

### Curriculum-Neutral Fields

[Curriculum](spec-ml-core-training.md#curriculum) requires each environment
document to partition its own view by what a field determines. Here the
shape-fixing side is `episode_subject`, `enabled_domains`,
`selected_objective_closure`, `probe_policy`, `observation_policy`, and
`determinism_policy`: the first two fix the action space's extent, the closure
and the reward codes fix what a return means, the observation policy fixes the
column catalog the embedding tables are sized for, and the determinism policy
fixes the stream layout.

Everything inside `EpisodePolicy` except the two reward codes is neutral, which
is what a curriculum over this environment actually wants to move: the seed
roots, the workload pool, the workloads per episode, the step and rejection
bounds, and the visited-set retention. Moving from small architectures to large
ones is a change of `seed_roots`, and lengthening episodes is a change of
`step_bound`. The Mapping cache survives such an advance, because its key names
one seed root and one workload rather than the sets this view declares:
extending `seed_roots` or `workload_pool` leaves every entry the earlier stages
warmed valid, which is what makes growing a pool the affordable curriculum it
is meant to be.

## Workload Feasibility Invariant

Every state the environment occupies is workload-feasible. For each Canonical
Dataflow Program in the state's current workload set, the probe must produce a
Mapping that could be finalized: full owner recomputation succeeds and all five
Mapping violation magnitudes are zero. This is the same closure condition
[Place And Route](spec-pnr.md#final-closure-and-verification) requires of a
selected candidate, evaluated over the ephemeral candidate rather than a
published Artifact.

The closure level follows the episode subject. A `SpatialModule` episode
requires one closed Spatial Mapping per workload against the current Module. A
`System` episode requires one closed System Mapping per workload against the
current System, together with the Spatial Mappings that System Mapping imports;
the environment does not accept a System whose constituent Modules close
individually while the System does not, because service continuity, transport
capacity, and progress closure are System-level facts.

The invariant is a gate on state occupancy, not a term in the energy. Removing
the last function unit that supplies an operation the workload needs, deleting
the last route between two required endpoints, shrinking a memory or FIFO below
what the workload's claims demand, narrowing a boundary inventory the workload
binds, removing the AccCore a workload's launch requires, or cutting the last
transport connection carrying a service leg all end the step. The agent never
occupies such a state and never receives a graded score for approaching one.

When `DataflowRewrite` is bound the workload set is itself part of the state. A
software decision replaces one workload with its rewritten child, and the
invariant is evaluated over the resulting set: a rewrite that produces a
program the current hardware cannot map is rejected exactly as a hardware
decision that removes a resource the workload needs. The two directions are
symmetric, which is the point of exploring them jointly. Rewrite legality
itself is never at issue here, because every catalog rewrite is externally
equivalence-preserving by its owner's contract; what a rejection reports is
that this hardware cannot map that expression of the workload.

Three consequences follow.

The enumeration mask is advisory and feasibility is authoritative. Decision
domains constrain which decisions exist, and the Builder rejects a draft that
violates an authoring-boundary rule, but neither knows in advance which legal
decision removes the resource some workload happens to require. An unmasked
action is therefore not a guarantee that the step advances. The environment
treats a legal-but-infeasible decision as an ordinary first-class outcome; it
is not an assertion failure, not an exception, and not a silent no-op that
leaves the agent unable to distinguish action from inaction.

Proof and budget are different answers and are never merged. A PnR
`ProvenInfeasible` result is a sound proof that no Mapping exists for that
workload on that candidate. Exhausting the probe's configured search work with
residual violations is inconclusive: it establishes nothing about the design.
Only exact admission or a sound bound may prove infeasibility, so an exhausted
budget is never reported as a proof. Both reject the step, and both are
recorded under distinct reasons so that a rising inconclusive rate is legible
as a probe-budget problem rather than as a shrinking design space.

Failure is attributed per workload. The step record names the exact Canonical
Dataflow that failed, its reason, and its residual violation magnitudes by
descriptor ordinal. A step that fails several workloads records the first
failure in canonical probe order; the environment does not continue probing
after the invariant is already broken.

## Action Space

The action space is the concatenation of each enabled domain's canonical
decision order. For the current state the environment enumerates every bound
domain in ascending `ExplorationDomainKind` ordinal, and within a domain
enumerates that generator's decisions in the generator's own exact canonical
order, truncated by its `max_children_per_parent` or `scope_expansion_limit`.
Those are the generators' own semantic work policies and are the only
truncations applied. `enumeration_bound` is an observation capacity, not a
further truncation: a state whose enumeration exceeds it is refused as
described under Observation, never silently shortened. Ascending domain ordinal
is the only cross-domain rule, so enabling or disabling a domain shifts a
suffix of the enumeration but never permutes another domain's internal order.
An action is one ordinal in the concatenation:

```text
EnumeratedDecision {
  domain: ExplorationDomainKind
  decision_kind: u32be ordinal in that domain's closed decision union
  parent: ParentSelector
  targets: ordered nonempty sequence<DecisionTarget>
  value: ReplacementPrototype(prototype ordinal)
       | BoundedIntegerDelta(delta)
       | NoValue
}

DecisionTarget {
  reference: owner-local reference
  role: Primary | GroupMember
}

ParentSelector =
    EpisodeSubject
  | SubjectComponent(FabricLocalReference)
  | CollectionOrdinal(uint32)

ActionIndex = uint32
```

`parent` qualifies the whole decision because every reference in one decision's
normalized payload is local to the same root; a decision does not straddle two
parents. `targets` carries every reference that payload names, in its canonical
order, with at most one `Primary`. A decision naming one entity has a single
target, and one naming an entity plus a reference set has a `Primary` followed
by its `GroupMember` members, which is the shape the observation's target arcs
and the policy's anchor summary both consume.

An owner-local reference is the reference its owner already defines: a
`FabricLocalReference` from
[Fabric Identity](spec-fabric-identity.md#owner-local-reference-kind-catalog),
or the `ActorRef`, `GraphRef`, or `StaticGraphLaunchRef` the rewrite catalog's
normalized decision carries. The environment transports them unchanged and
assigns no parallel identifier.

Every such reference is local to one root, so a target is meaningless without
naming which root, and `ParentSelector` is that qualifier in one form for every
domain. `EpisodeSubject` selects the subject itself. `SubjectComponent` selects
a component of the subject by an ordinary Fabric reference, which is how a
decision whose row operates on a `fabric.module` addresses one Module of a
`fabric.system`. `CollectionOrdinal` selects a member of an ordinal-indexed
collection the state holds, which is how a decision addresses one workload. The
legal selector for a decision is derived from its row's "Operates on" column
against the episode subject rather than written out per domain: a row that
operates on the subject's own kind uses `EpisodeSubject`, and any other
selector for it is invalid.

`NoValue` covers a decision fully determined by its target.

Collection ordinals are assigned at `reset` in the canonical order of the
selected collection and are stable for the whole episode. A `ReplaceParent`
completion substitutes the child at exactly its parent's ordinal and leaves
every other ordinal untouched, so a retained target, a retained mask bit, and a
node block never silently repoint at a different member because an identity
sorted differently after a decision.

The interaction surface, the `enumeration_bound` capacity, the masking algebra,
the elective stop, and the refusal of an over-capacity enumeration are owned by
[Action Surface And Masking](spec-ml-core-environment.md#action-surface-and-masking).
This document adds only which mask bits an environment clears: an index whose
decision was already rejected at this state, under the per-state rejection mask
Backtracking defines.

Every enumerated decision's references resolve into the combined node-and-link
observation space, so a decision that names an occurrence, a connection, a
transport link, an actor, or a graph is scored the same way: by attending over
the graph nodes it targets. Per-node and per-link actions are therefore the
observation's shape rather than a second indexing scheme, and the action index
stays a single ordinal even for a decision that carries several references.

Indexing the canonical decision order is what removes the per-type stride a
flat product space over node slots and decision types would need, and it
inherits determinism directly from the generator descriptor.

### Schedule-Preserving Transformations

Some decisions are worth taking only in a form the current schedule determines;
why that is worth the coupling is
[Why Some Design-Space Actions Read The Schedule](rationales/ml.md#why-some-design-space-actions-read-the-schedule).
A schedule-preserving transformation is one whose `GroupMember` references are
chosen from the state's own Mapping rather than from the Fabric's structure
alone: the decision kind, its `Primary` target, and the legality of the child
are exactly what they would be otherwise, and what the Mapping supplies is
which further references the decision names.

`RemoveOccurrence` is the case that motivates the class. Deleting an occurrence
the Mapping currently routes through breaks every net whose route traverses it,
so a useful removal carries reconnection: links joining the upstream and
downstream endpoints of the traversals that occurrence was carrying. Which
links those are is not a question the topology can answer. The topology admits
an enormous number of reconnections, nearly all of them useless, and the
complete set would replace one occurrence with a connectivity explosion. The
Mapping names the few that the workload actually uses, because they are the
ones its routes already run through.

That is what makes the class schedule-*preserving*. A child reconnected from
the parent's routes leaves the parent's placement carryable, so warm-starting
that child's probe begins from a mapping that still closes rather than one the
edit invalidated. A structurally reconnected child usually has to be re-mapped
from nothing, which costs the probe the option exists to save and discards the
evidence that the parent's placement was good.

The reference set is Mapping-derived; the enumeration's extent is not. A state
offers one entry per removable occurrence whether or not a Mapping exists, so
the enumeration's length remains a function of the state and the resolved
configuration alone. This is load-bearing rather than incidental: it is what
keeps step 7 ahead of step 8, keeps the reset capacity test cheap enough to
precede workload feasibility, and keeps that test's retained per-seed verdict
valid. Only the references inside an entry wait for the state's probe, and they
are filled when the observation that reports them is built.

The environment authors no reconnection semantics. Every reference it names is
an ordinary owner-local reference, every decision it emits is an existing
generator decision, and stage 3 runs the owner's ordinary acceptance path over
the result — so a reconnection the Mapping suggested and the owner refuses is
an ordinary rejection rather than a special case. The Mapping is being read as
evidence about which references are worth naming, never as authority for
whether the child is legal.

Path dependence follows the probe. With `warm_start_from_parent` clear, a
state's Mapping is a pure function of that state, so the reference set is too
and the core's enumeration-purity rule holds unchanged. With it set, the
Mapping is path-dependent by construction, so a schedule-preserving decision's
references are path-dependent with it. That is the same cost the option already
carries for the reward, extended to the content of an action, and it is
recorded here rather than discovered when two runs over one state disagree
about what they were offered.

Replay needs no Mapping. A `Trajectory` records each step's full
`EnumeratedDecision`, references included, and replay runs the recorded
decision rather than re-deriving it, so a replayed step reproduces the
reconnection the episode actually took without reconstructing the probe that
suggested it.

## Observation

### Combined Node And Link Space

The combined node-and-link space, its `GraphNodeRole` catalog, the
connection-as-node rule, the two enumeration encodings, and the target-closure
rule are owned by
[Combined Node And Link Space](spec-ml-core-environment.md#combined-node-and-link-space)
and [Enumeration Encoding](spec-ml-core-environment.md#enumeration-encoding).
This document states only what that space contains for a DSE episode.

This environment uses the `DecisionNodes` encoding, because its decisions have
variable target arity: one names a single entity, and another names a
distinguished subject plus a reference set whose size the generator decides.
Only out-degree expresses that, so the `Decision` block is present and
`decisions` is empty.

Every decision kind this environment enumerates depends on the closure.
Removing an occurrence names an occurrence, replacing a point connection names
a connection, changing a transport connection names a System link, and
refactoring a graph definition names graphs and launches, so an exploration
domain whose decisions can carry a reference the space cannot represent is not
admissible. That admissibility test is the reason the Exploration Domains table
requires representability of every row.

`DataflowOperation` and `DataflowValue` blocks are present exactly when a bound
domain operates on a Canonical Dataflow Program, which needs no configuration
flag because the binding already decides it. `AdjustParallelConnectionCount`
has exactly one target because a parallel connection is one node carrying its
count as a feature.

When `include_placement_edges` is set, the probe's current Mapping contributes
one `Placement` arc from each Dataflow node to the Fabric node that realizes it
and one `Route` arc from each Dataflow value node to each Fabric node its route
traverses, which is how the policy sees the present mapping. The two roles are
distinct because the two relations are: realization is one-to-one and is what a
capacity decision moves, while traversal is one-to-many and is what a
connectivity decision moves, and an encoder given one role for both would have
to separate them from context it does not have. Those arcs carry no Mapping
semantics of their own and are derived from the probe result of the state being
observed.

### The Graph Instance

The `Observation` and `GraphInstance` container shapes, the no-padding rule,
the negative-one absent sentinel, the buffer lifetime, and the obligations
every column catalog satisfies are owned by
[The Graph Instance](spec-ml-core-environment.md#the-graph-instance). The
closed column catalogs this environment owns are:

```text
DseNodeFeatureColumn =
    Role                    // 0
  | EntityKind              // 1
  | CapabilityCount         // 2
  | CapacityMagnitude       // 3
  | BufferDepth             // 4
  | ResidualViolationCount  // 5
  | PlacedDegree            // 6
  | DecisionDomain          // 7
  | DecisionKind            // 8
  | DecisionValueForm       // 9
  | DecisionValueOrdinal    // 10
  | DecisionDeltaMagnitude  // 11
  | DecisionDeltaSign       // 12

DseArcRole =
    Structural              // 0
  | Placement               // 1
  | Route                   // 2
  | DecisionTargetPrimary   // 3
  | DecisionTargetMember    // 4

DseScalarFeatureColumn =
    StepOrdinal             // 0
  | StepBound               // 1
  | WorkloadCount           // 2
  | ConsecutiveRejections   // 3
```

The six decision columns span the `Decision` block and the entity columns span
the entity blocks, on the core's role-block span rule.

There is no node-ordinal column: a node's ordinal is its position in the array,
and the core's own rule forbids consuming an ordinal as a projected scalar.

`EntityKind` is the owner-local reference kind ordinal from
[Fabric Identity](spec-fabric-identity.md) for a Fabric node and the Dataflow
owner's node or value kind ordinal for a Dataflow node. Every System entity
therefore appears as an ordinary `FabricOccurrence` node carrying its own
Fabric Identity kind ordinal, and transport and service connections appear as
ordinary `FabricConnection` nodes.

There is no `LiveDecisionCount` scalar. The live count is the `Decision`
block's size, which the instance already carries, and a scalar beside it would
be a second source for one extent.

A decision node carries at most one `Primary` target arc, its distinguished
subject when its owner defines one, and any number of `GroupMember` target
arcs, which are the members of a reference set or tuple the decision carries. A
decision whose owner defines no distinguished subject has only `GroupMember`
arcs. That is the shape a decision naming one entity plus a complete adapter
set requires; a projection that kept only the first target would silently
discard what the decision actually rewrites.

When a bound domain operates on a Canonical Dataflow Program, the
`DataflowOperation` and `DataflowValue` blocks span every workload in the
state's current workload set rather than one workload, in collection-ordinal
order, so a rewrite decision's target node is addressable exactly as a Fabric
decision's target is. Advancing on a software decision changes how many nodes
those blocks contribute, which changes the instance's extents; that is ordinary
for this space and needs no accommodation.

`CapabilityCount`, `CapacityMagnitude`, and `BufferDepth` are mechanical
projections of the exact Fabric owner facts for the referenced entity.
`ResidualViolationCount` and `PlacedDegree` are projections of the probe
result. The typed atoms of the `FabricResourceStateRef` catalog the core
appends under `include_resource_states` are owned by
[Fabric Resource Contract](spec-fabric-resource-contract.md).

The scalars a rejected step updates in place, under the core's buffer-lifetime
rule, are the cleared mask bit and the step and rejection counters.

A state whose enumeration exceeds `enumeration_bound` is rejected with
`EnumerationBoundExceeded`, which is the core's refusal rule reported under
this environment's own reason name; truncation would permanently hide the
decisions of whichever domains sort last.

## Episode Start

An episode is created from an exact seed set and an exact workload set. Each
seed is a finalized Fabric root whose kind matches `episode_subject`: a
`fabric.module` for a `SpatialModule` episode and a `fabric.system` for a
`System` episode. A seed is a builtin target, a user-supplied Fabric, or one
output of the `fabric_template` generator; exploration never begins from an
empty mutable graph, and a seed of the wrong kind is invalid rather than
adapted. Each workload is an exact Canonical Dataflow Program Artifact together
with the TechMapping the probe requires; both are ordinary static inputs and
acquire no environment-local identity.

The `System` subject arm carries the admissible finalized Module candidate set
that kind 15 consumes, since `AddAccCore` and `ReplaceSpatialAttachment` select
from an explicit finite set rather than synthesizing a Module. That set is an
ordinary generator input the configuration supplies; the environment does not
author or extend it. A Module a Spatial domain derives mid-episode is reachable
by a later completion because it is the child of the decision being completed,
not because the environment added it to a candidate set it does not own.

`reset` performs this ordered protocol:

1. derive the episode's PRNG streams from the effective seed and the copy
   coordinates;
2. select one seed root from `seed_roots` through `SeedSelection`, or adopt the
   one the start override supplies;
3. select the episode's initial workload set through `WorkloadSelection`, or
   adopt the one the start override supplies;
4. enumerate the seed and reject it when the enumeration is empty or exceeds
   `enumeration_bound`;
5. insert the initial state's identity into the visited set; and
6. establish the workload-feasibility invariant by obtaining one closed Mapping
   at the episode subject's closure level for every selected workload, then
   build the first observation.

Step 5 is what makes the seed state visited from the episode's first step.
Omitting it would make the one state every episode is guaranteed to occupy the
one state a decision may return to for free, and a pair of inverse decisions
would cycle through it indefinitely without ever being rejected as
`AlreadyVisited` — precisely the circling the visited set exists to catch.

Step 4 precedes step 6 for the reason the step protocol orders its own tests:
the enumeration is a pure function of the seed and the configuration, while a
closure-level Mapping is the dominant start cost. Enumerating first means a
seed that could never offer a legal action is rejected for a fraction of one
Mapping, rather than after `workloads_per_episode` of them, and the retry
budget multiplies that saving by every redraw.

Being a pure function of the seed and the enumeration-relevant fields of the
view, step 4's verdict is retained per seed root against those fields, and a
redraw that lands on a seed already judged consults that verdict instead of
re-enumerating. Keying it on the enumeration-relevant fields rather than on the
whole view is what lets a stage advance that only grows the workload pool keep
every verdict, on the same terms the Mapping cache survives one. A run of
`10^5` episodes over a few hundred seed roots would otherwise recompute a few
hundred fixed answers thousands of times each, and an unusable seed would be
re-enumerated and re-rejected on every episode that happened to draw it. This
is an ordinary cache: a hit and a miss produce the same formal result, and the
retry budget still bounds the redraws.

The per-workload work in step 5 has no cross-item dependence and may run
concurrently, on the same terms stage 8 of a step states.

The start override is the Gymnasium `options` argument, carrying an exact
episode start rather than a drawn one:

```text
EpisodeStartOverride {
  seed_root: ArtifactRootReference
  workloads: ordered nonempty sequence<WorkloadBinding>
}
```

The core's override rules bind it: its members must appear in `seed_roots` and
`workload_pool`, and when present it replaces steps 2 and 3 entirely, so
neither `SeedSelection` nor `WorkloadSelection` is consulted.

Step 5 is a precondition, not a best effort. A seed that cannot map every
selected workload is not a valid episode start:

```text
EpisodeStartOutcome =
    Started
  | SeedInfeasible { seed_root, workload }
  | SeedMappingNotClosed { seed_root, workload }
  | SeedEnumerationEmpty { seed_root }
  | SeedEnumerationBoundExceeded { seed_root, bound, required }
  | RetryBudgetExhausted
  | Invalid { violated precondition }
```

`SeedInfeasible` and `SeedMappingNotClosed` carry the proof-versus-budget
distinction the feasibility invariant defines. Either causes a retry with the
next draw from the same deterministic streams, bounded by `start_retry_bound`;
exhausting it is `RetryBudgetExhausted` and never a silently degraded episode
over a subset of the workloads.

Under an episode start override there is no next draw, so `start_retry_bound`
is what the core's no-retry rule leaves unconsulted, and the outcome is
returned rather than converted to `RetryBudgetExhausted`. `Invalid` covers a
seed whose root kind does
not match the subject arm and a workload whose TechMapping does not bind the
selected seed.

The two enumeration failures are both start failures rather than valid
episodes, and both are retried. A drawn seed whose enumeration is empty is
reported as `SeedEnumerationEmpty` because `Started` must guarantee at least
one legal action index; a drawn seed whose enumeration exceeds
`enumeration_bound` is reported as `SeedEnumerationBoundExceeded`, carrying the
capacity and the required length. Neither is `Invalid`, since another seed in
the same set may enumerate within capacity, and `reset` step 4 already tests
them together. Only a configuration whose every seed fails one of the two is an
invalid configuration, and that is reported when the retry budget is exhausted
over it.

Mapping and TechMapping results may come from an ordinary cache family under
[Cache Dependencies](spec-config-ssot.md#cache-dependencies), which already
owns the versioned canonical dependency key, the removability rule, and the
requirement that a hit and a miss produce the same formal result. This document
adds only the granularity: the cached unit is one workload, not the episode's
workload set. A Mapping depends on the seed root, that one workload with its
TechMapping, and every probe view that produced it, and on nothing about which
other workloads the episode happened to draw. Every probe view means both the
Spatial PnR view and, for a `System` episode, the System PnR view its subject
arm carries: a System-episode entry produced under one System PnR contract must
not be returned to an episode running another, which is exactly the case where
a hit would return a Mapping a miss would not have produced. Keying on the set
would instead make the key space the combinations of the pool rather than the
pool itself, so changing one workload would discard the mappings of every other
workload in the draw even though they are byte-identical and reusable.
Per-workload keying makes the same cache reach a useful hit rate after a few
dozen episodes instead of effectively never.

The same grant covers stage 8, and it matters more there than at `reset`. A
TechMapping is a pure function of the child identity, one workload, and the
TechMapping view, unconditionally. A probe result is a pure function of the
child identity, one workload with its TechMapping, and the probe views whenever
`warm_start_from_parent` is false, which is exactly the configuration whose
energy is path-independent. Both recur constantly once a policy converges on a
decision kind, and they recur across every environment copy in the run, so a
step-time cache reaches a useful hit rate far sooner than the start-time one it
extends. With warm starting enabled the probe is path-dependent by construction
and is not cacheable; that is a further cost of the option, and it is measured
rather than assumed.

## Step

One step applies exactly one enumerated decision. The protocol is ordered, and
every stage may reject:

1. validate the action against the live mask and the retained per-state
   rejection mask;
2. derive a fresh draft from the exact parent the decision's domain names and
   apply the one owner-typed decision to it;
3. run the domain owner's ordinary acceptance path for that child, stopping
   before durable publication;
4. publish the accepted child through its own generator when and only when the
   decision's row completes by `Reattach`, skipping a component whose identity
   the store already resolves;
5. derive the completion its domain's row names, so the state is again a
   root-complete subject of the episode's kind;
6. reject when the resulting state identity is already in the visited set;
7. reject when the resulting state's enumeration exceeds `enumeration_bound`;
8. for each workload in canonical order, regenerate its TechMapping if the
   decision invalidated it and then probe it against the candidate,
   warm-started from the parent state's mapping when `warm_start_from_parent`
   is set, requiring a closed Mapping with all five violation magnitudes zero;
9. read the `G` measures from the probe and every predicted metric from the
   parameter-bundle inference kernels the selected closure's dimensions
   reference; and
10. quantize each objective source, form the new `ObjectiveVector`, and
    evaluate the selected search-energy `WeightedLevel`.

The order is cheapest-rejecting-first as far as the completion form allows, and
where it holds it is load-bearing rather than incidental. Stages 6 and 7 both
answer from the completed state alone: a state identity and the enumeration
length are functions of that state and the resolved configuration, with no
Mapping, TechMapping, or objective value involved. Placing them ahead of stage
8 means a revisit or an over-capacity candidate never costs a probe, and both
are expected outcomes rather than corner cases. This is the property
[Schedule-Preserving Transformations](#schedule-preserving-transformations)
preserves by keeping a Mapping-derived reference set inside an entry whose
existence the Mapping does not decide.

For a `None` or `ReplaceParent` completion the ordering is unqualified, because
stage 4 publishes nothing and those two rejections therefore also precede every
durable write. A `Reattach` is the exception, and it is the reason stage 4 sits
where it does. An Artifact owner requires every direct dependency of a root to
be durably published before that root's closure is derived, and a subject that
names a component depends on it, so a component held only as an unpublished
verified closure cannot be reattached at all. The completion therefore cannot
be derived before its component is published, and the state identity stages 6
and 7 test does not exist until the completion has run. Deriving the completion
first and publishing afterwards is not a cheaper ordering but an invalid one:
it would name a subject closure over a dependency the store does not hold.

A `Reattach` step consequently pays its component's store insertion before the
visited and capacity tests, which is the case where the memo below is worth the
most.

Stage 4 deduplicates. The component's canonical identity is already exact from
stage 3, and the store is content-addressed, so a component whose identity the
store already resolves is not re-serialized and not re-inserted. A converging
policy revisits the same component constantly, and without this rule every
revisit would rewrite bytes the store already holds. An implementation may
answer stage 6 earlier still, from a memo of
`(parent state identity, EnumeratedDecision)` to child state identity, and skip
stages 2 through 5 entirely on a hit; the child identity is a deterministic
function of that pair, so the memo is an optimization and not a second identity
authority. On a `Reattach` row that memo is what keeps a revisit from paying a
publication for a state the episode is about to reject.

Stage 8 interleaves regeneration with probing per workload rather than
regenerating every TechMapping first. The two orders are observationally
identical, because probe order is already the order in which a failure is
reported, but the interleaved one stops at the first failing workload instead
of paying regeneration for workloads it never probes. That matters because a
Fabric decision invalidates every workload's TechMapping while the probe aborts
on the first failure, so the barrier form would waste all but one regeneration
on precisely the rejections early abort exists to make cheap.

Independent work within a stage may run concurrently. Per-workload regeneration
and probing in stage 8 and per-contract inference in stage 9 have no cross-item
dependence, and a concurrent implementation satisfies this contract exactly
when it reports the failure lowest in the configured probe order among those it
observed. Cancelling the remaining probes once a failure is known is permitted
and expected; what is fixed is the reported result, not the execution order.

Stage 3 is what makes the candidate real without making it persistent. For a
Fabric domain it produces the `VerifiedFabricClosure` owned by
[Finalization And Publication](spec-fabric-artifact.md#finalization-and-publication)
and does not publish it, so the child has passed the same finalizer and
verifier as a published Fabric and carries the same canonical identity while
remaining un-referenceable by any other owner. For `DataflowRewrite` the child
passes the same rewrite match, normalization, and equivalence obligations its
catalog owner requires. In every case the candidate is fully verified rather
than provisionally accepted, and its identity is exact enough to serve as a
visited-set key.

What stage 4 publishes is the component alone. It is published by its own
generator as an ordinary intermediate candidate, which is the status its owner
already gives it, while the subject closure that consumes it remains
unpublished. Deferred publication is a property of the episode's subject, not
of every value a step touches.

This is a real cost and it is stated rather than hidden: a `Reattach` step pays
canonical publication and store insertion for its component, so a `System`
episode with a Spatial domain bound is more expensive per step than a
`SpatialModule` episode over the same decisions, and it accumulates
content-addressed objects for components no selected design may ever reference.
An episode that does not want that cost does not bind a Spatial domain under a
`System` subject; the alternating batches of the ordinary joint search reach
the same designs without it.

Stage 5 exists because a generator's child is not always the episode's subject,
and it is driven by the "Completion" column rather than by named domain pairs:

```text
Completion =
    None
  | Reattach { via: ExplorationDomainKind }
  | ReplaceParent
```

`None` means the child is already the subject. `Reattach` means the child is a
component of the subject, so stage 4 publishes that child through its own
generator's ordinary path and the named domain then applies the one decision
that rebinds exactly the parent selector the original decision targeted,
leaving every other relation untouched. The reattach invocation receives the
published child as its admissible component set, which is an ordinary generator
input the caller supplies per invocation, not a configuration field the
environment extends. `ReplaceParent` means the child substitutes for its parent
in an ordinal-indexed collection the state holds, at exactly that parent's
ordinal, and publishes nothing.

A row's completion is derived rather than chosen: `None` when the row's
"Operates on" kind is the episode subject's kind, `ReplaceParent` when it
operates on a member of a collection the state holds, and otherwise `Reattach`
via the unique bound row whose "Operates on" kind is the subject's. That is the
same derivation the `ParentSelector` rule already uses, so the table's
Completion column records the outcome rather than introducing a free parameter.

No completion invents a decision the agent did not take. Both forms are
functions of the decision's own parent selector, which is why they need no
choice of their own; a domain whose completion would require a second free
choice cannot state its row and is therefore not admissible. A step may expand
to more than one generator invocation on replay, which Trajectory Retention And
Replay records exactly.

Stage 8's regeneration runs on every advancing step, because a TechMapping is
bound to exact `D` and `F` and every decision changes one of them. What differs
is how much it invalidates: a software decision invalidates the TechMapping of
exactly the rewritten workload, while a Fabric decision invalidates every
workload's, so the stage is most expensive for the domain that changes the
fewest programs. Regeneration uses the
[root-complete TechMapping path](spec-tech-mapping.md#root-complete-central-adapter)
with `probe_policy.tech_mapping_config_view`. A workload whose TechMapping
cannot be regenerated has no probe input at all, which is a distinct rejection
from a workload that maps poorly.

Stage 8 is also the feasibility gate defined above. Stage 9 obtains predicted
metrics by calling a registered `ModelParameterContract` 's `project_features`
and `infer` in process; this is the parameter-backed tier of the
fast-evaluation cascade owned by
[Full-Stack Architecture](spec-loom-stack.md#joint-design-exploration). A
predicted value may rank and may reward, but it can never reject a candidate as
impossible, and `infer` returning `Unsupported` outside its exact training
support region is never replaced by an extrapolation, a bound, a midpoint, or a
default.

Only the contracts the selected closure's dimensions reference are invoked, and
that set is the definition rather than a configured list. Inference is the
expensive tier of the cascade, so a contract whose prediction no objective
dimension reads is never run at all.

The outcome algebra is closed:

```text
StepResult {
  transition: optional<StepTransition>
  episode_end: optional<TerminalReason>
}

StepTransition =
    Advanced { energy_delta_code, energy_delta_sign }
  | Rejected { reason: RejectionReason }

RejectionReason =
    DraftRejected { domain, authoring diagnostic }
  | FinalizationRejected { domain, finalizer diagnostic }
  | ClosureCompletionRejected { domain, completion diagnostic }
  | TechMappingUnavailable { workload }
  | AlreadyVisited { identity }
  | WorkloadProvenInfeasible { workload, violation magnitudes }
  | WorkloadMappingNotClosed { workload, residual violation magnitudes }
  | ObjectiveUnavailable { dimension }
  | EnumerationBoundExceeded { bound, required }

TerminalReason =
    ElectiveStop
  | StepBoundReached
  | ConsecutiveRejectionBoundReached
  | EnumerationEmpty
```

`WorkloadProvenInfeasible` and `WorkloadMappingNotClosed` stay separate for the
reason the feasibility invariant states: one is a proof and the other is a
budget artifact. `TechMappingUnavailable` is separate from both because the
workload never reached the probe, so nothing was established about mappability
at all. `ClosureCompletionRejected` reports that a child was accepted by its
own owner but could not be reattached as a root-complete subject, which is a
System-level or workload-set-level failure rather than a defect in the child.
`ObjectiveUnavailable` is this environment's name for the core's
unavailable-objective outcome, and `EnumerationBoundExceeded` is its name for
the core's over-capacity refusal. Every reason names the domain, workload, or
bound it belongs to, so a rejection profile attributes cost and failure to the
domain that produced it.

A transition and an episode ending are separate facts because they co-occur, on
the terms
[Step Accounting And Episode Endings](spec-ml-core-environment.md#step-accounting-and-episode-endings)
states. Here the rejection that reaches `consecutive_rejection_bound` both
rejects and ends the episode; the advance that lands in a state with no
admissible decisions both advances and ends it; and an elective stop ends the
episode with no transition at all.

An absent transition means the policy ended the episode instead of acting,
which happens exactly for `ElectiveStop`; a third union member for that case
would let a consumer write two checks that can disagree. `EnumerationEmpty` is
reported on the step that advances into the empty state, not on a later call,
so the consumer is never handed a state with no legal action index.

`Advanced` commits the candidate as the new current state, inserts its identity
into the visited set, and clears the per-state rejection mask for the new
state. Every `Rejected` transition restores the parent state exactly and
advances no episode position other than the step counter. The state identity
used for the visited set is the episode subject's identity together with the
ordinal-indexed identity sequence of the current workload set, so a software
decision that returns the design to a previously visited pair is caught exactly
as a hardware decision that does.

`episode_end` distinguishes termination from truncation on the terms
[Step Accounting And Episode Endings](spec-ml-core-environment.md#step-accounting-and-episode-endings)
states. `ElectiveStop` and `EnumerationEmpty` are terminations;
`StepBoundReached` and `ConsecutiveRejectionBoundReached` are truncations.

## Backtracking

A step never mutates the parent. Stage 2 derives a fresh draft from the exact
parent, so a rejection has nothing to roll back: it discards the child and the
parent is the value it already was. The parent's canonical identity after a
rejection therefore equals its identity before the attempt as a consequence of
the protocol, not as a promise about anyone's restore path. The only
transactional state in a step is internal to the probe, where
[Evaluation Transaction](spec-pnr.md#evaluation-transaction) owns `commit` and
`discard`.

The environment then clears the rejected decision's bit in a per-state
rejection mask so the same decision is not re-proposed from the same state. The
mask belongs to the state, not to the episode: advancing to a new state starts
a new empty mask, and returning to a previously visited identity is already
rejected by the visited set. The visited set retains up to
`visited_state_retention` identities in insertion order, which is what prevents
an episode from spending its step budget alternating between two states.

The step accounting identity
[Step Accounting And Episode Endings](spec-ml-core-environment.md#step-accounting-and-episode-endings)
states holds over an episode of this environment with no unaccounted remainder.
Consecutive rejections are counted, exposed through `scalar_features`, and end
the episode at the configured bound.

## Reward

Reward is the signed difference of the selected search energy across the
transition, on the terms
[Reward Contract](spec-ml-core-environment.md#reward-contract) states. This
document says only which two states the difference is taken between and what
this environment's non-transition outcomes cost.

The parent's energy the core retains is the one produced when the parent was
entered, by `reset` or by the stage 10 that advanced into it.

A `Rejected` transition yields `rejection_reward_code` with a negative sign,
independent of the rejection reason.

`stop_reward_code` applies with a negative sign to `ElectiveStop` alone.
`StepBoundReached` and `ConsecutiveRejectionBoundReached` yield zero because
they are truncations, which the core prices at zero. `EnumerationEmpty` also
yields zero, since the step that reports it already carries its own `Advanced`
reward. Only the ending the policy elects is priced.

Path dependence is a consequence of `warm_start_from_parent` and is stated
rather than hidden. Warm starting seeds each probe from the parent's mappings,
so the mapping a probe returns, and therefore the `G` measures and the energy,
can depend on how a state was reached. State identity does not depend on the
path, so two paths to the same design may report different energies and a cycle
of deltas need not sum to zero. A configuration that requires path-independent
energy, such as one whose analysis assumes telescoping deltas, sets
`warm_start_from_parent` false and pays the cold-probe cost; a configuration
optimizing throughput sets it true and accepts a reward that is exact per
transition but not a potential function.

## Trajectory Retention And Replay

An episode retains one transient record:

```text
Trajectory {
  config_view_digest: ComponentViewDigest
  episode_subject: SpatialModule | System
  seed_root: ArtifactRootReference
  initial_workloads: ordered sequence<WorkloadBinding>
  effective_seed: u64
  env_runner_index: uint32
  vector_index: uint32
  local_episode_index: uint64
  advanced_steps: ordered sequence<TrajectoryStep>
}

TrajectoryStep {
  decision: EnumeratedDecision
  completion: optional<EnumeratedDecision>
  child_identity: ArtifactIdentity
}
```

A step records the decision the agent chose, the mechanical completion stage 5
derived, and the exact identity of the child that step produced, which stage 3
computed whether or not it published.

Replay is a finite sequence of resolved plans, one per recorded step, each
resolved after its predecessor completed. A step's plan binds its parent as an
exact static artifact set, which is available because the preceding step's plan
published it, runs the `Generate` node for the recorded decision, and selects
the output whose identity equals `child_identity`. Selection by recorded
identity is an ordinary property of the retained trajectory, not a new plan
mechanism: the next step's plan binds that exact artifact as a static input.
There is no `Promote`, no best-child rule, and no Artifact Store scan.

How narrowly a step's node can be aimed depends on what its generator's
configuration exposes, and the two cases differ. A hardware rewrite generator
owns a decision-domain set whose member fixes one exact target selector plus a
finite value set or bounded range, so replay may bind the singleton domain
admitting exactly the recorded decision; the node then produces that one child
and publishes no siblings. The Dataflow rewrite generator exposes no such
narrowing. Its whole resolved configuration is one expansion limit, and it
enumerates the catalog over a frontier in canonical order, so the smallest node
that reaches the recorded decision also publishes every decision preceding it.
Those siblings are ordinary valid Artifacts that the plan does not select, and
accepting them is the cost of replaying a software step. Claiming a singleton
domain for that generator would be claiming a configuration field it does not
have.

Replay is therefore one `Generate` node per recorded decision, including a
completion when the step carries one, with no special arithmetic per domain: a
`Reattach` completion is itself an enumerated decision of the domain its row
names, and a `ReplaceParent` completion records none because the generator's
own child is already the new collection member. Replay publishes real Artifacts
at every step, with the ordinary `CandidateDecision` lineage contributions
those descriptors own, which is exactly the cost the episode deferred. A replay
that selects any output other than the recorded identity is a different plan
that explores a neighborhood; it is not this trajectory.

A `Trajectory` is a compact way to name a sequence of generator decisions so
that an authoritative run can reconstruct them; a candidate it names becomes
real only when a real `Generate` node publishes it and a real `Promote` node
acquires Evidence and applies quality gates.

## Python Boundary

The `loomml` package layering, the RLlib conformance target and its
obligations, the zero-copy array contract, the outcome-versus-exception rule,
and the threading and `fork` rules are owned by
[Python Boundary](spec-ml-core-environment.md#python-boundary). This
environment occupies `loomml.env.dse` and `loomml.rllib.dse` and adds no rule
of its own.

Two of the core's obligations are worth naming against this environment's own
values, because they are what the obligations are for here. The decision nodes
and target arcs the parametric-action convention carries are the ones The Graph
Instance defines, and the `reset` seed the caller supplies replaces
`determinism_policy.master_seed` as the episode's effective seed and is
recorded in the `Trajectory`.

## Benchmarking

The harness obligations, the required breakdowns, the instrumentation rule, and
the ratio-based regression budget are owned by
[Benchmarking Harness Contract](spec-ml-core-environment.md#benchmarking-harness-contract).
This environment ships `loom-dse-env-bench`, whose rejection-reason breakdown
is keyed by `RejectionReason` and whose decision partition is
`ExplorationDomainKind`, because the domains do not cost the same.

The harness decomposes one step into these stages and reports each separately:

```text
enumeration          build the canonical decision order and mask
draft_apply          derive the child draft and apply one decision
accept               the domain owner's acceptance path for the child
identity             canonical bytes and root identity
completion           the completion its domain's row names
techmap              regenerate an invalidated TechMapping
probe                per workload, closed Mapping or typed failure
inference            project_features and infer per referenced contract
observation          build the observation arrays
marshal              expose buffers across the Python boundary
```

and one reset into `seed_load`, `cache_lookup`, `cold_mapping` per workload,
and `first_observation`.

Two further breakdowns are required beyond the core's two. The probe is
reported per workload, together with the mean workloads probed and TechMappings
regenerated per rejected step, which is what shows whether the interleaving of
stage 8 is earning its place. And `warm_start_from_parent` is measured the same
way, with and without the parent's mappings, split by whether the decision
changed the Fabric or a workload.

## Anchor Verification

Stable tests cover adoption rejecting a reward closure whose Mapping-sourced
dimension the probe view does not admit, whose Evaluation-sourced dimension no
configured parameter contract satisfies, or which references a held-out
obligation; the set of parameter contracts an episode invokes being exactly the
set the reward closure's Evaluation-sourced dimensions reference, with no
configuration field beside it; adoption rejecting a duplicate domain binding, a
bound row whose required subject does not match the subject arm, and a
`Reattach` completion whose named domain is not itself bound; an omitted domain
contributing no enumerated entry, no mask bit, and no observation block; an
exploration domain whose decisions carry a reference the combined space cannot
represent being refused at adoption; a decision whose row operates on the
subject's own kind rejecting any selector but `EpisodeSubject`, and a decision
whose row operates on a component requiring a `SubjectComponent` or
`CollectionOrdinal` selector; a `ReplaceParent` completion leaving every other
collection ordinal unchanged; a state whose enumeration exceeds
`enumeration_bound` reporting `EnumerationBoundExceeded` rather than being
shortened; a revisited or over-capacity candidate being rejected without any
probe or TechMapping regeneration running; a schedule-preserving decision's
`GroupMember` references being those the state's Mapping routes through its
`Primary` target, the enumeration's length being unchanged by whether a Mapping
exists, a reconnection the domain owner refuses being an ordinary rejection,
and a replayed step reproducing the recorded references without reconstructing
a probe; a decision returning the design to
the seed state being rejected as `AlreadyVisited` on the episode's first
opportunity; a start failure under an `EpisodeStartOverride` being returned as
it stands rather than retried or converted to `RetryBudgetExhausted`; a
rejected step regenerating no
TechMapping for a workload the probe never reached; a rejection that reaches
the consecutive-rejection bound reporting both its `RejectionReason` and its
`TerminalReason`; an advance into an empty enumeration reporting both on the
same step; enabling a domain shifting only the enumeration suffix at and after
its ordinal; enumeration and mask agreement, including a cleared bit after a
rejection at that state; a draft, acceptance, or completion rejection producing
no child, no identity, and no lineage; a decision that removes a capability,
route, capacity, AccCore, or transport connection a workload requires being
rejected with the correct per-workload reason and leaving the parent
byte-identical; a rewrite whose child the current hardware cannot map being
rejected with the same per-workload reasons as a hardware decision; a workload
whose TechMapping cannot be regenerated reporting `TechMappingUnavailable`
rather than a mapping outcome; a probe budget too small to close a reachable
Mapping reporting `WorkloadMappingNotClosed` and never
`WorkloadProvenInfeasible`; a concurrent probe implementation reporting the
same failure as a sequential one; every state reachable within an episode
satisfying the feasibility invariant at the episode subject's closure level; a
`Reattach` completion publishing its component and leaving a root-complete
subject whose only changed relation is the targeted one; `reset` refusing a
seed whose root kind does not match the subject arm and refusing a seed that
cannot map every selected workload; a drawn seed with an empty enumeration
reporting `SeedEnumerationEmpty` and one whose enumeration exceeds
`enumeration_bound` reporting `SeedEnumerationBoundExceeded`, both retrying
rather than starting and neither being `Invalid`; a `Reattach` step publishing
its component before the visited and capacity tests while a `None` or
`ReplaceParent` step publishes nothing before them; a realization arc carrying
`Placement` and a route-traversal arc carrying `Route`; an
`EpisodeStartOverride` consulting neither selection stream and producing the
same episode on every copy, and one naming inventory outside the configured
sets being rejected; a per-workload cache entry surviving a change to another
workload in the draw, and a `System`-episode entry not being returned under a
different System PnR view; `ObjectiveUnavailable` never producing a numeric
reward; a truncation yielding zero reward while an elective stop yields the
configured stop code; equal configuration, coordinates, seed, and action
sequence reproducing an identical trajectory; and trajectory replay selecting
the output whose identity equals `child_identity` at every step and reproducing
identical finalized identities in order, including the node a `Reattach`
completion contributes, with a hardware step narrowed to a singleton domain
publishing no sibling and a software step publishing the catalog prefix it
must.

Tests do not pin the live node, arc, or decision counts of any particular
state, the bound values a profile selects, which reconnection a particular
Mapping implies, wall-time numbers, per-domain cost ratios, probe-ordering
heuristics, corpus contents, diagnostic text, or Python formatting.
