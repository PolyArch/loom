# Configuration And Artifact Identity Rationale

Normative contracts are owned by
[Resolved Configuration](../spec-config-ssot.md),
[Full-Stack Traceability](../spec-full-stack-traceability.md),
[Intermediate Reports](../spec-intermediate-artifacts.md), and
[Visualization](../spec-mapping-visualization.md).

## Why Configuration Is Resolved Once

Loom has many components, but a component-local option parser or hidden default
would make the same invocation mean different things at different stages. A
single EDA-style acceleration profile is therefore resolved before execution
into one flattened typed configuration. Builtin presets and a custom path are
two authoring routes into that same schema, not two configuration systems.

At most one builtin parent is allowed. General inheritance graphs and merge
orders were rejected because the emitted resolved configuration must be the
complete, replayable semantic input. Components receive versioned projections
of that object and cannot reopen inheritance, inspect the source profile, or
apply another `auto` rule.

Semantic configuration and invocation bindings are intentionally separate.
Seeds, deterministic work units, model selection, and other choices that can
change a formal result are semantic. Paths, output locations, host parallelism,
license endpoints, and wall-clock limits affect whether an attempt completes,
not which result is correct. Mixing them would make replay host-dependent and
could let a timeout select a different candidate.

Schema versions use `X.Y` because consumers need an explicit distinction
between incompatible changes and compatible schema growth. Unknown fields are
not an extension mechanism; an owner must introduce and validate a new schema
version.

## Why Local Tool Binding Is Separate

Executable locations, module initialization, container entrypoints, license
environment names, and scratch roots differ by host without changing the
question Loom is compiling or evaluating. Putting them in ResolvedConfig would
make semantic identity depend on installation layout. Reading a conventional
file implicitly would instead make the same command depend on ambient
repository state.

Loom therefore accepts machine-local bindings only through an explicit
`--loom-local-config=<path>` input. Missing tool entries fall through to the
current environment and finally module discovery; invalid explicit entries
fail closed. Resolution is performed once and frozen into a nonsemantic
invocation bundle, so generated scripts do not become another configuration
system.

The separation is about ownership, not whether a value looks like a command
line option. Tool version, effort, corner, constraints, libraries, or any
option that can change a result still belongs to the semantic model or
generator binding. Only provider-declared operational launch values may remain
local.

ResolvedConfig 3.3 therefore adds one typed Voltus rail provider binding rather
than a generic EDA property map. The stable provider build and exact PGV member
fingerprints change the model result and belong in its component view; the
matching local directory does not. Making the binding optional preserves
software-only profiles while making absence explicit and preventing ambient
tree selection.

## Why ResolvedConfig 2.0 Removes The Provisional Flat Knobs

The first implementation carried a display `config_id`, three global hardware
widths, a string ranking policy, and floating-point objective weights. None was
a sound semantic owner. Display identity belongs to invocation provenance;
address and bus widths belong to the exact finalized Fabric; and DSE already
owns typed objective sources, exact quantization, levels, and orderings.
Keeping the provisional fields would make later consumers choose between two
authorities.

They were therefore removed in one incompatible schema transition instead of
being retained as deprecated aliases. A 2.0 component either consumes a typed
field from its declared view or recovers hardware facts from its exact Artifact
input. It never translates an old string or floating value into the new typed
model, because that translation would itself become an undocumented policy.

## Why ResolvedConfig 3.0 Shrinks The Mapping Violation Catalog

The provisional Mapping objective registry named resource-time overbooking,
buffer overuse, and hard service shortfall independently from generic capacity
overuse. Fabric's typed `ResourceContract` subsequently established that time
determines the overlap query, buffers are owner-defined durable capacity state,
and service slots are ordinary capacity dimensions. The three names therefore
did not own independent facts.

Keeping them as aliases would make one Fabric counterexample selectable several
times in an objective and would preserve obsolete ordinals indefinitely.
ResolvedConfig 3.0 instead selects the five independent Mapping violations and
projects the corresponding version-2 PnR views. Old spellings are rejected;
typed owner witnesses preserve detailed diagnostics without becoming another
configuration authority.

## Why Selected Component Closures Are Materialized

An ordinal into the complete DSE view is compact, but it is not self-contained:
the same ordinal can denote another record after unrelated catalog insertion,
and a consumer must retain a hidden dependency on the whole DSE view. Hashing
the complete DSE view repairs correctness but invalidates a PnR freeze when an
unselected model or objective changes.

The PnR projector instead materializes the selected typed transitive closure,
canonicalizes it by DSE-owned semantic keys, and assigns view-local references.
This is not a second objective authority. It is the same kind of removable,
validated component projection as any other ResolvedConfig view. The approach
has the minimum dependency surface: every consumed fact is present, every
unconsumed fact is absent, and the exact view digest is sufficient for cache
lineage.

## Why Artifacts Are Narrow Semantic Owners

An Artifact is an immutable, finalized semantic object in Loom's derivation
graph. It is not every file written by a tool. Structured and Dataflow programs,
Fabric, Mapping, requests, evidence, configuration, implementation, and
deployment can be Artifacts because each has a closed owner. Logs, paths,
waveforms, reports, scratch checkpoints, and visualization layouts do not
become Artifacts merely because a manifest names them.

Common owns only schema framing, digest computation, exact reference framing,
and immutable storage. Each family owns its canonical semantic bytes and local
reference catalog. A generic Artifact framework with property bags or a global
entity taxonomy was rejected because it would move family semantics into a
second interpreter.

Typed cross-artifact references combine an exact Artifact identity with an
owner-defined local target. Symbols, source locations, printer positions,
construction handles, and PnR dense indices cannot replace either component.
Independent objects receive artifact-local entity identity only when they have
an independent semantic or physical role. Owner-relative leaves such as ports,
resource states, and actor terminals remain typed structural references rather
than receiving redundant IDs.

## Why Canonicalization Is Family-Owned

Canonical bytes must preserve semantic differences while erasing authoring
accidents. The family finalizer therefore owns canonical labeling, ordering,
normalization, and validation; Common hashes only the resulting bytes with one
fixed preimage. This avoids both insertion-order identity and a Common layer
that must understand every dialect.

Canonicalization is not permission to rewrite upstream semantics. For example,
LLVM payload identity preserves exact validated module-owned DataLayout
spelling even when two spellings are structurally compatible at link time.
Identity and compatibility answer different questions and must not be
collapsed into one canonical string.

## Why Publication Is One Object At A Time

Artifact dependencies are already immutable, content-addressed objects that
may be shared by many roots. A multi-object transaction would need rollback,
ownership, crash recovery, and a publication manifest without adding semantic
correctness. The store therefore publishes one complete object per operation.
An owner resolves and validates all dependencies first; publishing the root is
that root's only commit point.

This model admits deduplication and honest crash recovery. A failed durability
acknowledgement cannot promise that a complete final object is absent, so retry
uses the same deterministic put/get operation. Readers observe absent or
complete validated objects, never a partial semantic artifact.

An external PDK, library, macro, rule, or IP file does not become an Artifact
merely because a provider consumes it. Its provider-owned semantic binding
freezes the exact expected digest, and the local invocation bundle maps that
identity to a machine-local path and validates the bytes before use. Keeping
such files outside the Artifact Store preserves the one-object publication
rule and avoids inventing a platform-content Artifact family solely to mirror
licensed or private storage.

## Why Reports And Visualization Are Projections

`--loom-viz-export` is a formal driver capability because humans need source,
IR, hardware, Mapping, and execution views. It is still a projection: it can be
deleted and regenerated from exact artifacts without changing compilation,
Mapping, Evaluation, or Deployment.

Visualization does not implicitly request acceleration, Mapping, or
simulation. This preserves ordinary compiler behavior and prevents a viewer
from becoming a hidden pipeline authority. Raw reports and trace payloads have
the same rule until an exact owner, schema, importer, and lineage contract is
defined.

Fabric authoring uses the same boundary. Hardware architects need to inspect
both a heterogeneous AccCore system and each reusable SpatialCore definition
before software Mapping exists. Rendering from the finalized root and its exact
dependency closure prevents a Builder draft or UI payload from becoming a
second hardware authority.

Layout is computed offline because topology is immutable and authoring review
does not require interactive node movement. Shipping precomputed coordinates
and routes makes the standalone HTML deterministic, removes a large browser
layout dependency, and avoids different machines presenting different graph
geometry. Pan, zoom, search, filtering, and detail navigation remain ordinary
viewer state and do not justify moving semantic or layout derivation into the
browser.

## Why Human-Readable Target Names Are Projections

During early design, a SpatialCore target was described with a readable form
containing PE, memory, and switch counts followed by a canonical circuit
digest. The counts remain useful in diagnostics and user interfaces, but they
cannot identify hardware: equal counts say nothing about topology,
capabilities, timing, state, or protocol.

The finalized Fabric Artifact identity is therefore the only machine target
identity. A display string may mechanically combine inventory summaries with
that full identity, and a UI may abbreviate it, but caches, Mapping,
Deployment, compatibility checks, and deduplication use the exact Artifact
reference and canonical bytes. Functionally equivalent but structurally
different hardware remains distinct; equal canonical hardware content
deduplicates through the Common store without a second SpatialCore identity
registry.
