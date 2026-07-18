# Loom Configuration SSOT

This document specifies Loom's target configuration source-of-truth
contract. It owns how Loom loads optimization constants, hardware model
parameters, simulator model parameters, DSE weights, tool profiles, and
run policies without allowing component-local constants to drift apart.

## Purpose

Loom configuration answers this question:

```text
For this run, which parameter values, profiles, policies, and model
constants are authoritative, and can every downstream artifact prove it
used the same resolved configuration?
```

Configuration is a first-class input artifact. It is not a hidden side
channel, a loose set of command-line defaults, or a collection of
component-local structs that happen to share field names.

## Boundary

The configuration SSOT owns:

* optimization weights and objective profiles;
* compiler pipeline profiles and search-policy names;
* DSE objective records, ranking policies, and stochastic seeds;
* tech-mapping and FU-synthesis parameters;
* ADG Builder recipe defaults and architecture-template parameters;
* PnR search policies, resource-compatibility policies, and mapping
  cost-model parameters;
* DFG-sim and CGRA-sim model parameters, cost tables, latency tables,
  fidelity profiles, and simulator limits;
* RTL, EDA, FPA, and reporting profile selection;
* global machine constants when a run needs to vary them, including
  address width, index width, and memory bus width.

The configuration SSOT does not own language or dialect semantics.
Verifier rules for dataflow token behavior, Fabric SSA connectivity,
Fabric op legality, artifact schema shape, status vocabularies, and
mapping-route contiguity remain in their owning specs and code
verifiers. Test fixture data also remains fixture-owned.

## Formats

Human-authored configuration should prefer YAML. Machine interfaces
should prefer JSON. YAML and JSON are semantically equivalent and must
resolve to the same canonical configuration model. Parsing and emitted
resolved configuration default to JSON when a stream, stdin input, or
extension-less path does not declare a format.

TOML may be accepted for backward compatibility, but it is not a
separate semantic model. A TOML input must resolve through the same
schema, provenance, validation, canonical JSON emission, and
artifact finalization rules as YAML and JSON.

The canonical program-to-program output is resolved JSON. The
ResolvedConfig ArtifactIdentity is finalized from its typed schema descriptor
and canonical resolved JSON bytes. It does not depend on the original YAML,
JSON, TOML, file order, comments, or whitespace.

## Resolution Model

A Loom run starts from one top-level configuration. The top-level
configuration may import or include other files, but the resolved output
is a single canonical tree with one path for every parameter.

Configuration resolution applies sources in this order:

* centralized built-in default configuration;
* explicitly named profile imports;
* the top-level run configuration body;
* explicit command-line or local activation overrides.

An override is legal only when its source, target key, and precedence
are recorded in provenance. Two unordered sibling imports that set the
same canonical key are ambiguous and must fail resolution.

Every resolved key has:

* canonical path;
* value;
* declared type and unit when applicable;
* owner component or shared owner;
* default/provided/override provenance;
* validation rule identity;
* optional profile identity.

Component tools receive typed configuration views derived from the
canonical tree. A typed view may rename fields for local API
convenience, but it must not own independent defaults or an independent
`ArtifactIdentity`. Current artifacts record the exact `ResolvedConfig`
`ArtifactIdentity`; they do not acquire generic component-view descriptor
or canonical-byte fields.

A future cache family may include a closed typed-view descriptor and its
deterministic canonical bytes after that view and cache contract are
defined. Such a cache key is neither an `ArtifactIdentity` nor
configuration authority.

## Early-Fail Rules

Configuration resolution must fail before compilation, mapping,
simulation, DSE, RTL, or reporting when any of these conditions hold:

* unknown key outside an explicitly allowed extension namespace;
* duplicate key in one mapping;
* type mismatch;
* unit mismatch;
* enum value outside the owning schema;
* numeric value outside the owning validation range;
* unresolved import or cyclic import;
* two sources set the same canonical key without explicit precedence;
* a lower-level component config repeats a literal value instead of
  referring to the canonical resolved key;
* a command-line option changes a config-owned value without recording
  an explicit override;
* a component artifact names a ResolvedConfig ArtifactIdentity that differs
  from one of its required input artifacts;
* a DSE candidate combines evidence produced under incompatible
  ResolvedConfig identities.

Repeating the same literal value in multiple layers is still a
source-of-truth violation unless one occurrence is a declared default
and the other is an explicit override with provenance. Equal duplicated
literals are not allowed to pass silently.

Unknown objectives, unknown policies, and unknown fidelity profiles must
fail with structured diagnostics. They must not fall back to runtime,
analytic, default, or scaffold behavior.

## Defaults

Code may hard-code default parameter values only in the centralized
default configuration implementation. Component-local structs may expose
typed accessors or compatibility adapters, but their defaults must be
copied from the centralized default configuration.

The default configuration is itself part of the public configuration
schema. It must be printable as canonical resolved JSON and must pass
the same validation path as user-supplied configuration.

Semantic constants are not configuration defaults. Examples include
Fabric verifier arity rules, Dataflow firing semantics, artifact
required-key sets, and mapping-route endpoint-contiguity checks.

## Artifact Propagation

Every machine-consumed artifact produced under a resolved configuration
must carry:

* the ResolvedConfig ArtifactIdentity;
* profile identities used by the producer;
* override provenance when overrides affected the producer;
* diagnostics for absent optional configuration evidence.

CSV summaries may project those fields. JSON artifacts and MLIR
attributes or manifests are the preferred SSOT carriers.

When a tool consumes multiple configured artifacts, it must compare
their ResolvedConfig ArtifactIdentities. A mismatch is a structured
failure or blocked condition unless the tool has an explicit migration
rule that records both configurations and proves the difference is
irrelevant to the consumed evidence. A derived component view does not add
a second configuration identity.

## Configuration Domains

The target configuration tree includes these domains:

* `global`: machine-wide constants and deterministic run settings;
* `compiler`: pipeline profiles, placement policies, operation and
  type capability profiles;
* `dse`: objective records, continuous weights, preset weight profiles,
  ranking policies, Pareto policy, seeds, and fidelity requirements;
* `adg_builder`: recipe defaults and architecture-template parameters;
* `pnr`: placement, routing, resource-compatibility, and mapping search
  policies;
* `sim`: DFG-sim and CGRA-sim model parameters, limits, and fidelity
  profiles;
* `rtl`: RTL lowering profiles and structural options;
* `eda`: tool and library profile selection without public private
  paths;
* `fpa`: estimation model, activity-source policy, and report fidelity
  requirements;
* `reporting`: export profiles and projection policy.

These domains are organizational views over one resolved configuration.
They are not independent files with independent defaults.

FU synthesis settings remain on the dedicated consumed `SynthConfig` surface.
An unconsumed `fabric_techmap` domain must not appear in `ResolvedConfig`,
canonical resolved JSON, or the ResolvedConfig ArtifactIdentity.

## DSE Weights

DSE must expose continuous objective weights and named preset profiles.
Both are configuration values. Presets are aliases or profiles that
resolve into ordinary objective records and weights before selection.

The compiler must not expose placement choices such as thread-parallel
versus SpatialCore loop placement as direct force switches. Those
choices are DSE candidates selected by configured objectives, weights,
constraints, and feedback fidelity.

## Diagnostics

Required diagnostic classes include:

* `config_parse_failed`;
* `config_unknown_key`;
* `config_duplicate_key`;
* `config_type_mismatch`;
* `config_unit_mismatch`;
* `config_range_violation`;
* `config_conflicting_sources`;
* `config_unrecorded_override`;
* `config_identity_mismatch`;
* `config_missing_required_profile`;
* `config_unknown_policy`;
* `config_unknown_objective`.

Diagnostics must identify the canonical key, source provenance, owning
schema, and affected component when applicable.

## Acceptance Criteria

The configuration SSOT target is complete when:

* YAML and JSON inputs can describe equivalent runs and emit identical
  canonical resolved JSON;
* the built-in default configuration can be emitted, validated, and
  finalized through the same path as user configuration;
* component tools consume typed views from one resolved configuration;
* component-local defaults are removed or become compatibility adapters
  over the centralized default configuration;
* unknown keys, duplicate keys, conflicting sources, unknown objectives,
  and unknown policies fail before downstream work starts;
* configured artifacts carry the ResolvedConfig ArtifactIdentity;
* cross-artifact consumers reject incompatible ResolvedConfig identities;
* tests distinguish movable configuration defaults from semantic
  verifier/schema constants and fixture data.
