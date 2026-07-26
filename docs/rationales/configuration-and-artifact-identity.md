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
