# Resolved Configuration

This document owns Loom's profile resolution, flattened semantic configuration,
and mechanically derived component views. It does not own component semantics,
Mapping constraints, runtime invocation bindings, or artifact schemas outside
the ResolvedConfig family.

## Public Selection

The only public semantic profile selector is:

```text
--loom-accel-profile=<builtin-preset-or-config-path>
```

The initial EDA-style builtin presets are:

```text
report_only
quick_explore
balanced_explore
performance_explore
implementation
strict_implementation
```

Names describe flow intent and exploration effort, not a pass switch. Omitting
the option selects one designated builtin default. A custom profile may inherit
at most one builtin preset and apply schema-declared semantic overrides. Loom
does not support arbitrary profile chains, sibling merge order, import graphs,
or format-specific inheritance semantics.

Human-authored JSON or YAML may represent the same typed profile schema. TOML
compatibility and component-local parsers are not part of the contract. The
central resolver is the only parser and validation authority.

## Resolution

Every semantic configuration follows one derivation:

```text
builtin preset or profile file
  + allowed semantic overrides
  -> validate one selected profile schema
  -> resolve defaults, inheritance, and auto values
  -> flattened canonical resolved-config.json
  -> finalized ResolvedConfig Artifact
  -> mechanically derived immutable component views
```

The flattened result contains one explicit typed value for every active field.
No component may reinterpret `auto`, apply a hidden default, merge another
profile, or read an ambient environment variable after resolution.

Unknown fields and enum values are rejected unless the selected schema version
declares them. The first contract has no generic extension namespace. A future
field is added by the owning schema under the `X.Y` versioning rule rather than
by silently accepting opaque keys.

`resolved-config.json` is the canonical persistent serialization and can be
used again as profile input. It stores flattened semantic values only. Source
paths, selected preset name, include location, override spelling, comments,
environment, and resolution diagnostics are InvocationManifest provenance and
do not enter ResolvedConfig semantic bytes.

ResolvedConfig uses the common ArtifactIdentity SHA-256 v1 contract. Changing
authoring syntax, whitespace, field order, or source location without changing
the flattened typed values does not change identity.

## Schema Ownership

All schema versions use `X.Y`: `X` is an incompatible change and `Y` is a
compatible extension. The ResolvedConfig schema owns the canonical composition
of component domains. Each domain owner defines its fields, types, units,
defaults, validation rules, and semantic effect exactly once.

Closed enums use typed enum definitions and convert to strings only at parser,
printer, canonical JSON, or diagnostic boundaries. Components do not repeat
string tables or accept unknown names with an implicit fallback.

Configuration owns values that can change a formal compiler, DSE, Mapping,
simulation-model, backend, or artifact-generation result. It does not own:

* Dataflow, Fabric, Mapping, Evaluation, or runtime semantics;
* intrinsic hardware legality or capability, which Fabric owns;
* invocation-specific Mapping restrictions, which MappingConstraintSet owns;
* the native `PnrIndex` width, which is a build ABI;
* paths, output locations, visualization destinations, or package paths;
* wall time, host parallelism, license concurrency, storage quota, process
  retries, or other physical execution limits; or
* runtime handles, allocations, timestamps, logs, and diagnostics.

Address, bus, or similar architecture widths may appear as ADG Builder or
builtin-template inputs, but finalized hardware semantics are owned by the
resulting exact Fabric identity. They are not free runtime Mapping knobs.

Deterministic semantic work limits that bound the formal search are owner-local
configuration. Physical execution limits may stop an attempt and produce an
incomplete outcome, but they cannot select a best-so-far candidate or change
the resolved semantic plan.

There is no global fidelity profile or ladder. Evaluation policy authorizes
exact models, obligations, budgets, and promotion behavior through its own
ResolvedConfig domain.

## Component Views

A component consumes one immutable typed view derived by the central config
library. Examples include `ResolvedPnrConfigView`, a DFG-simulation model view,
and a backend view. A view is not an Artifact, semantic identity, registry
entry, or independently authorable configuration.

Each view has:

* one schema identity and `X.Y` version;
* one deterministic projector from exact ResolvedConfig;
* one canonical byte representation; and
* one mechanically derived `component_view_digest`.

The digest contract is fixed:

```text
SHA-256(
  bytes("loom.component.view.digest.v1\0")
  || u32be(length(schema_descriptor_bytes))
  || schema_descriptor_bytes
  || u64be(length(canonical_view_bytes))
  || canonical_view_bytes
)
```

It is a compact integrity and dependency value, not ArtifactIdentity and not a
replacement for `ResolvedConfig.id`. It cannot be separately authored or
modified. The projector, descriptor, canonicalizer, and digest framing each
have one owner.

A component API receives only its typed view. It must not also inspect hidden
fields in the full ResolvedConfig, because doing so makes its declared
dependency closure false. Changing an unconsumed config field leaves the view
bytes and digest unchanged; changing a consumed field or view schema changes
them.

## Cache Dependencies

Only a real expensive derived result may define a cache family. That family
owns one versioned canonical dependency key containing every dependency it
actually consumes, such as:

* exact input Artifact identities;
* component-view schema descriptor and digest;
* consumer, importer, model, or build semantic identity; and
* cache schema version.

The entire typed closure is framed and hashed once. Common provides framing and
SHA-256 utilities but does not define a generic cache object or understand
family fields.

For example, a PnR freeze key contains exact Dataflow, TechMapping, and Fabric
identities, the PnR view descriptor and digest, and importer/cache schema
identity. Two different full ResolvedConfig artifacts may reuse that cache only
when their PnR views and every other cache dependency are identical.

A cache key is removable execution metadata. It does not enter the Artifact
DAG, result ArtifactIdentity, or semantic configuration. A cache hit and miss
must produce the same formal result.

## Invocation Separation

The InvocationManifest records the exact ResolvedConfig identity and only the
component-view descriptors and digests actually consumed by the invocation.
It also owns profile-source provenance, input and output bindings, source and
hardware paths, output directories, `--loom-viz-export`,
`--loom-deploy-output`, tool/runtime provenance, and retained execution
records.

Changing an output or artifact-store location does not change ResolvedConfig,
any semantic component view, Mapping, simulation semantics, or a cache key that
does not consume that location. Configuration fields are not copied into every
downstream Artifact; a family carries a ResolvedConfig reference only when its
own schema explicitly makes that reference semantic.

## Determinism

The replay boundary is:

```text
codebase semantic/build identity
+ exact ResolvedConfig ArtifactIdentity
+ input ArtifactIdentities
+ pinned toolchain/backend identities
+ exact model parameter bundle identities
+ exact immutable external Evidence when consumed
-> deterministic semantic artifacts and projections
```

Randomized components use versioned PRNG protocols, explicit seeds, canonical
ordering, and deterministic semantic work budgets from their typed views.
Worker completion order, host concurrency, wall time, cache state, paths, and
licenses may determine whether an attempt completes but not its formal result.

## Validation Anchors

Stable tests cover:

* a builtin or one-parent profile resolves to one flattened canonical object;
* equivalent JSON/YAML authoring yields identical ResolvedConfig identity;
* unknown fields, unknown enums, inheritance chains, and unresolved `auto`
  values fail before finalization;
* the emitted resolved JSON can be consumed again without semantic change;
* changing an unconsumed field leaves a component view and view-only cache key
  unchanged;
* changing a consumed field, input Artifact, view schema, or cache schema
  changes the corresponding derived key;
* path and execution-limit changes do not alter semantic identity or formal
  selection; and
* identical dependency closures produce identical view digests and cache keys.

Tests do not create one fixture per field, preserve YAML formatting, snapshot
diagnostic text, define a generic component registry, or prebuild cache schemas
for nonexistent consumers.
