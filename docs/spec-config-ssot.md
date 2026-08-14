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

The EDA-style acceleration profile is orthogonal to hardware-target selection.
It may contain one typed `hardware.target` domain, but `quick_explore`,
`implementation`, and the other flow-intent names are not Fabric target names.
The builtin hardware authoring selection uses the exact
`BuiltinTargetPreset` enum owned by `docs/spec-adg-builder.md`.

Resolution replaces that enum with the selected ADG template identity,
template schema version, and complete typed parameter values. The enum spelling
is recorded only as invocation provenance and is excluded from canonical
ResolvedConfig bytes. Omitting hardware selection resolves `Default`; it does
not produce a target-less configuration.

An external `--loom-hardware=<fabric.mlir>` binding is mutually exclusive with
an explicitly selected builtin target. Import and Fabric finalization produce
the exact Fabric Artifact input. The source path remains an invocation binding.
After either source path, compiler components consume the exact Fabric
reference and cannot branch on whether the hardware originated from a builtin
or user C++ Builder.

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

The current schema is `loom.config.resolved 5.0`. Version 2.0 was an
incompatible replacement for the earlier provisional schema: it removed the
authoring-only `config_id`, the free global `addr_bits`, `index_width`, and
`mem_bus_width` knobs, the string `ranking_policy`, and the floating-point
`ResolvedDseObjective` list. Hardware widths remain inputs to the exact
hardware target and are recovered from the finalized Fabric; objective facts
use the typed records owned by [DSE Feedback](spec-dse-feedback.md#objectives-and-quality-gates).

Version 3.0 replaces the provisional eight-entry Mapping violation catalog
with the five independent facts owned by
[Place And Route](spec-pnr.md#objective-projection). Event-relative resource
occupancy, buffer occupancy, and service capacity are projections of the one
Fabric-owned capacity relation; service compatibility and quality remain typed
admission or Evaluation facts. Removing the three competing enum values and
rewriting objective ordinals is incompatible. A 3.0 parser rejects old fields
and retired violation spellings rather than adopting, translating, or aliasing
them.

Version 3.1 is a compatible extension that adds the Dataflow rewrite
generator's positive semantic expansion limit. It changes neither an existing
field nor any Mapping, PnR, or objective meaning.

Version 3.2 is a compatible extension that adds the Structured
MemoryCommunication generator's positive semantic expansion limit. It changes
neither an existing field nor physical memory ownership.

Version 3.3 compatibly adds an optional typed Evaluation provider-binding
domain. An absent binding leaves the corresponding model unavailable; it never
selects an ambient provider build or external input.

The current schema composes these active policy domains:

```text
ResolvedConfig {
  hardware_target
  dse {
    structured_ownership
    schedule
    memory_communication
    dataflow_rewrite
    tech_mapping
    spatial_pnr
    system_pnr
    evaluation_and_objective_catalogs
  }
  evaluation {
    cadence_voltus_static_rail: optional {
      stable_provider_build_identity
      power_grid_library_members[] {
        relative_path
        sha256
      }
      power_grid_library_entrypoints[]
    }
  }
}
```

`evaluation.cadence_voltus_static_rail` is the sole resolved source for the
result-affecting Voltus build and complete PGV tree used by model kind 12. The
member table is nonempty, sorted by canonical relative path, duplicate-free,
and contains the exact SHA-256 fingerprint of every member. The nonempty,
duplicate-free entrypoint array gives the exact provider load order; every
entry is a canonical relative path present in the member table, and the first
entry is the technology PGV required by the provider. Entrypoints do not repeat
fingerprints or introduce files outside the tree. The binding contains no local
directory key or absolute path. The model's descriptor-owned projector consumes
only this typed binding; local configuration resolves a matching tree without
changing the model binding.

Version 4.0 adds the required ordered PGV entrypoint array. The 3.3 member table
proved tree content but could not determine which members were load roots or
the provider-required technology-first order. Inferring either from filenames
would make an adapter convention a second semantic authority, so the required
field is an incompatible replacement rather than a compatible default.

Version 5.0 adds the required positive `hardware_target.parameters.mesh_dimension`
field and requires it to exceed one. The field is the sole semantic owner of
the square Spatial and Temporal mesh extent generated by the selected builtin
template. Earlier configuration serialized resource multiplicities but left
topology extent hidden behind preset-specific expansion. No other scale value
can recover that choice, so adding the required field is incompatible rather
than assigning a compatibility default.

`dse.evaluation_and_objective_catalogs` materializes exactly the owner tables
of the [Resolved Configuration View](spec-dse-feedback.md#resolved-configuration-view):
model authorizations, Evidence obligation templates, objective dimensions,
weighted levels, total orderings, quality gates, and resolved plan nodes. The
canonical JSON uses structured fields for model authorizations and objective
records. Records whose schema is owned by DSE or Evaluation use lowercase
hexadecimal text of that owner's canonical bytes; parsing must immediately
invoke the exact typed adopter and require canonical re-encoding. These byte
fields are not generic extension payloads, and an unknown, malformed, stale,
noncanonical, or unregistered owner record is invalid. The
`ResolvedDseConfigView` projector consumes these typed records from the complete
`ResolvedConfig`; callers cannot construct a parallel view directly.

`hardware_target` resolves the authoring selection described above. It is not
part of either PnR component view: Spatial and System PnR consume the exact
finalized Fabric identity as a separate input. The two PnR policy domains use
the same closed field types and codecs but remain separate values, so they may
select different search policies and objective closures without introducing a
second schema. Fabric template expansion consumes the resolved template
identity, schema version, and complete typed parameter record. Preset defaults
are authoring provenance and cannot replace those resolved parameters.
Compiler and execution tools that consume a ResolvedConfig have no second
builtin-preset selector. When they expose a Fabric root reference, it is a
projection of the target built from that exact resolved record.

The Structured ownership generator policy owns:

```text
dse.structured_ownership.scope_expansion_limit: positive uint32 = 64
```

The Structured Schedule generator policy owns:

```text
dse.schedule.scope_expansion_limit: positive uint32 = 64
```

This is the number of `scf.for` scopes admitted from canonical Structured
operation order into one invocation's finite domain. Every admitted scope
retains its complete owner-derived decision domain. The value is a semantic
work limit, not a worker count or wall-time budget.

The current Structured ExecutionShape generator has no configurable semantic
field. Its exact Fused/Split domain is owned by the
[Structured ExecutionShape Generator](spec-compiler-part-2-scf.md#structured-executionshape-generator)
contract. An empty canonical component view and its digest therefore select the
complete current policy; process-local worker or cache settings cannot alter
it.

The Structured MemoryCommunication generator policy owns:

```text
dse.memory_communication.scope_expansion_limit: positive uint32 = 64
```

This is the number of memory-relevant structural scopes admitted from
canonical Structured entity order into one invocation's finite domain. Every
admitted scope retains its complete applicable decision and parameter domain.
It is a semantic work limit, not a physical memory capacity, worker count, or
wall-time budget. The immutable component view descriptor is
`loom.structured_memory_communication_generator.config.2.0`; its canonical
bytes are one unsigned 64-bit big-endian projection of this resolved value.
Import rejects zero and every value outside the positive `uint32` owner domain;
the wider wire encoding does not enlarge the semantic value set. Version 2.0
replaces the constant-staging-only scope domain of 1.0; adopters reject the 1.0
descriptor rather than reinterpreting its limit.

The Dataflow rewrite generator policy owns:

```text
dse.dataflow_rewrite.scope_expansion_limit: positive uint32 = 256
```

Its exact accounting and incomplete-search behavior are owned by
[Deterministic Work, Candidate Sets, and Cache](spec-dse-feedback.md#deterministic-work-candidate-sets-and-cache).
The default is a usable baseline for the complete normalized rewrite catalog,
not a completeness promise for every program. An invocation whose next
canonical decision does not fit still returns the exact typed incomplete
outcome owned by that contract.
The immutable component view descriptor is
`loom.dataflow_rewrite_generator.config.1.1`; its canonical bytes are exactly
one unsigned 64-bit big-endian projection of this resolved value. The adopter
requires an exact descriptor, positive value, digest match, no trailing bytes,
and byte-exact re-encoding.

The TechMapping generator policy owns:

```text
dse.tech_mapping.match_row_attempt_limit: positive uint64 = 65536
dse.tech_mapping.partial_cover_expansion_limit: positive uint64 = 262144
dse.tech_mapping.candidate_publication_limit: positive uint64 = 16
```

These values define the deterministic finite search domain described by
[TechMapping Generation](spec-tech-mapping.md). They are not wall-time,
memory, worker-count, or solver limits.

The two PnR policy domains each own one complete authoring record whose field
semantics and validation are defined by
[Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism):

```text
PnrPolicyAuthoringRecord {
  search_policy
  determinism_policy
  temporary_violation_policy
  selected_total_ordering
  selected_search_energy
  focused_closure_dimensions
  evaluation_interaction_bindings
}
```

References in this authoring record address the DSE catalogs in the same exact
ResolvedConfig. They never leave the central resolver. Projection computes the
selected transitive closure, canonicalizes it independently of unrelated
catalog entries, assigns view-local references, and emits the self-contained
PnR component view. A component cannot observe the original catalog ordinals
or the digest of the complete DSE view.

The designated default profile is `balanced_explore`. Its initial PnR policy
uses `PathFinder` with the `Multiplicative` price kernel, admits every closed
Mapping violation kind only as a temporary search state, gives every admitted
violation a positive SearchEnergy term, orders the complete violation vector
before `StaticSchedulePressure` and then `TotalSelectedTraversalClaim`, selects
no focused-closure metric or route-guidance binding, and enables bounded
`CpSat` repair. All numeric values,
including seeds, proposal weights, semantic work limits, cooling parameters,
PathFinder pressure parameters, and repair bounds, are emitted explicitly by
the 3.0 resolver. No PnR kernel supplies a missing value or chooses a profile
default.

The limit counts complete ownership-scope expansions. Expanding one scope
enumerates its entire finite typed decision domain; it does not truncate that
domain based on worker count, wall time, cache state, or completion order. This
field is semantic because changing it changes the finite Generate domain.
Physical candidate-worker count remains an invocation Execution Limit and is
not a ResolvedConfig field.

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
library. Examples include `ResolvedFrontendConfigView`,
`ResolvedPnrConfigView`, a DFG-simulation model view, and a backend view. The
consuming component's specification owns its field schema;
`ResolvedFrontendConfigView` is owned by
`docs/spec-compiler-part-1-source.md`. A view is not an Artifact, semantic
identity, registry entry, or independently authorable configuration.

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

When a component consumes selected records from another configuration domain,
its projector materializes only the selected transitive closure. It copies the
owner-typed records, canonicalizes them by their complete semantic keys,
assigns references local to the component view, and mechanically rewrites all
internal references. The copied records remain mechanical projections of
their original owner schemas; the component may not reinterpret or extend
them. This rule lets a cache ignore unrelated catalog entries while preventing
an ordinal into a larger view from becoming a dangling or shadow authority.

When the consumer is selected by an exact static descriptor, such as an
`EvaluationModelDescriptor`, that descriptor owns the component-view schema
and registered typed projector/adopter. A persistent binding may omit the
schema bytes only when its exact descriptor reference recovers them uniquely;
it then stores the canonical view bytes and mechanically derived digest. Import
must resolve the descriptor, recompute the digest, adopt an owner-typed value,
and require decode/re-encode equality. A raw byte vector plus a validation
callback is not an immutable typed component view, and copying the schema into
the binding would create a competing authority.

A component view may have a deliberately empty field set. Such a view has
empty canonical view bytes and states that the component consumes no semantic
ResolvedConfig field under that view version. This is a closed dependency set,
not permission for the component to inspect the full ResolvedConfig. Adding
the first consumed field changes the owning view schema and its deterministic
projector; the component specification decides whether that requires a major
or minor schema-version change.

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

For example, a Spatial PnR freeze key contains exact Dataflow, TechMapping,
Fabric, and MappingConstraintSet identities, the Spatial PnR view descriptor
and digest, and importer/cache schema identity. A System PnR freeze key uses
its exact system inputs and the distinct System PnR view descriptor and
digest. Neither key contains the complete ResolvedConfig identity. Two
different full ResolvedConfig artifacts may reuse a cache only when their PnR
views and every other cache dependency are identical.

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

Machine-local tool, runtime, and external-file configuration is
supplied only through the explicit `--loom-local-config=<path>` option defined by
[External Tool Invocation](spec-external-tool-invocation.md). It is not a
profile parent, ResolvedConfig field, component view, Artifact, or implicit
repository default. The exact expected fingerprint or tool-bundled resource
identity belongs to the consuming provider binding; the local file map only
makes matching bytes accessible. The local resolver freezes its selected
executable, external-input, environment, module, and runtime bindings into an
owner-specific invocation bundle. The central InvocationManifest references
that retained record instead of copying or reinterpreting its fields.

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
* reordering an unselected DSE catalog or changing an unselected record leaves
  a selected PnR closure unchanged, while changing any selected transitive
  record changes its bytes and digest;
* a selected PnR closure round-trips without consulting the complete DSE view,
  and foreign, stale, or noncanonical local references are rejected;
* path and execution-limit changes do not alter semantic identity or formal
  selection; and
* identical dependency closures produce identical view digests and cache keys.

Tests do not create one fixture per field, preserve YAML formatting, snapshot
diagnostic text, define a generic component registry, or prebuild cache schemas
for nonexistent consumers.
