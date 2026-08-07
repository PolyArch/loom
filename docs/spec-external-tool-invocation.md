# External Tool Invocation

This specification owns Loom's nonsemantic boundary for binding local external
tools, locating explicitly named machine-local external files, and
materializing independently executable invocation bundles. It does not own
tool, target, library, or IP semantics, process supervision, resource
isolation, scheduling, or Evaluation result schemas.

## Ownership

A local external-tool provider owns one static typed local descriptor, its
driver-script projection, and its version probe. The descriptor declares the
logical tool key, executable names, provider-recognized environment roots,
module candidates, runtime compatibility, and the exact local binding fields
it expects from its semantic owner.

The exact `CandidateGeneratorDescriptor` or `EvaluationModelDescriptor` owns
the typed semantic `prepare/import` boundary and its result contract. A local
tool descriptor may supply provider-specific driver and parser components, but
it cannot choose semantic inputs, output slots, Evidence meaning, or an
importer independently. Any importer identity retained by a bundle is a
mechanical verification projection of the exact semantic descriptor and its
provider form, not a caller-authored callback name.

The version probe is structured data: an argument token array, a nonempty set
of accepted process exit codes, an optional required output marker, and an
optional stable-line selector. A selected candidate is valid only when its
exit code is accepted, its required marker is present, and its stable-line
selector matches exactly one nonempty output line. The normalized selected
line, or the normalized complete output when no selector exists, is the frozen
version identity. Provider descriptors, rather than generic runner policy,
own tool-specific conventions such as a successful version command returning
a nonzero exit code.

The local binding resolver owns only this derivation:

```text
explicit local configuration
  or current process environment
  or module-system discovery
  + exact provider descriptor
  + exact semantic model or generator binding
  -> frozen local tool and runtime binding

explicit local external-file map
  + exact semantic input fingerprints
  -> frozen local external-input projections
```

The resolver does not select a model, change a semantic option, acquire a
license, prepare a container image, or interpret a report. A local binding is
nonsemantic provenance. It is valid only when its version probe and provider
checks prove that it realizes the exact semantic tool identity already
selected by the model or generator binding.

An `ExternalToolInvocationBundle` is an owner-specific nonsemantic attempt
record. It has no `ArtifactIdentity` and is not the central
`InvocationManifest`. The central manifest may retain a reference to the
bundle; it does not copy the bundle's fields. The bundle references exact
semantic inputs and materializes them for execution without becoming another
semantic owner.

## Configuration Planes

Semantic configuration continues to use the sole public selector owned by
[Resolved Configuration](spec-config-ssot.md#public-selection). External-tool
invocation does not repeat or alias its spelling.

Machine-local binding configuration uses the separate explicit option:

```text
--loom-local-config=<path>
```

Loom never searches for or implicitly loads a repository-local configuration
file. The repository ships `loom-local-config.example.json` with placeholders
and ignores `/loom-local-config.json`; users may choose any local-config path.
Generated bundles containing host paths also remain outside tracked source.
Omitting `--loom-local-config` supplies an empty local configuration and does
not disable environment or module discovery.

The local configuration is strict versioned JSON. Its initial authoring shape
is:

```text
LocalToolConfigV1 {
  schema = "loom.local_tool_config"
  version = "1.0"
  module? { init: absolute_path }
  external_files?: {
    local_file_key: absolute_file_path
  }
  runtime? {
    policy: "auto" | "host" | "polyarch_container"
    polyarch_container? {
      binding?: ExecutableBinding | ModuleBinding
      os?: nonempty_string
      inherit_environment?: [environment_name]
      provider_options?: provider_owned_object
    }
  }
  tools?: {
    logical_tool_key: {
      binding?: ExecutableBinding | ModuleBinding
      inherit_environment?: [environment_name]
      provider_options?: provider_owned_object
    }
  }
}

ExecutableBinding { executable: absolute_path }
ModuleBinding { modules: [nonempty_module_name, ...] }
```

The module list is an ordered explicit activation sequence. An omitted
`binding` permits normal environment and module discovery while retaining the
other local operational settings. `provider_options` is accepted only through
the exact provider descriptor's closed typed local schema. The initial common
schema has no arbitrary argument list or environment-value map.

`external_files` maps opaque machine-local keys to explicitly named absolute
ordinary files. The resolver rejects duplicate canonical paths, symlinks,
special files, and observed mutation while reading. It hashes every configured
file, indexes the exact bytes by SHA-256, requires the fingerprint already
frozen by the semantic generator or model binding, and produces a nonsemantic
local projection for the bundle. When several configured files have identical
bytes, the resolver freezes the lexicographically first canonical path; the
choice cannot change semantics.

The key, path spelling, and selected projection mode are not target,
HardwareImplementation, Request, or Evidence identity. The map cannot declare
technology identity, target, corner membership, provider compatibility,
library role, expected fingerprint, directory filter, or glob. Those facts
remain owned by the exact provider descriptor and resolved semantic binding.
Listing a file never authorizes recursive scanning, PDK import, tool
installation hashing, or implicit file discovery.

The initial common contract admits explicitly named ordinary files only. A
provider that consumes a logical directory must declare the complete ordinary
file set and deterministic projected layout. It cannot promote ambient
directory membership into semantic input or reuse the removed platform
directory-manifest mechanism.

Omitted `runtime.policy` resolves to `auto`; omitted environment lists and
provider-option objects are empty. In `auto`, each provider supplies one
ordered compatible PolyArch/container OS preference, and the resolver freezes
the first composition that passes preflight. There is no hidden executable,
module, or container path default: those bindings still follow the three-tier
resolution rule.

Unknown fields, duplicate keys, relative executable, initialization, or
external-file paths, empty tool or local-file keys, invalid environment names,
and simultaneous executable and module bindings for one tool are rejected. A
present but invalid explicit binding is an error and never falls through to
another source. An omitted tool entry may continue through environment and
module discovery.

Secret values are not stored in local configuration, bundle manifests,
generated scripts, diagnostics, or central invocation records. Configuration
may name a required environment variable; the generated script checks and
inherits its value only when it runs.

Executable paths, module initialization, license-variable names, scratch
locations, and equivalent host bindings are nonsemantic. A tool option, PVT
choice, effort, constraint, library identity, runtime component, or container
property that can change the formal result belongs to the exact model or
generator binding. Labeling such a value as a local provider option is
invalid.

## Local Output Placement

An invocation bundle and everything written beneath it are machine-local
attempt material, not tracked source. The EDA-specific disclosure class is
owned by [EDA Tooling](spec-eda-tooling.md); this document owns only placement.
An EDA bundle root must be outside the source worktree or beneath a
repository-owned ignored output root. Loom repository commands, examples, and
tests use `build/eda-runs/`, which is derived from the existing top-level
ignored `/build/` root. They do not introduce a second EDA-only ignore root.

The same placement rule applies to a local Artifact Store or Blob Store that
contains direct EDA-generated implementations, Evidence, invocation records,
or their payloads. Selecting a local output path does not make the path, its
contents, or Git ignore state semantic input. Loom's compiler libraries do not
invoke Git, edit ignore rules, or reinterpret repository tracking as Artifact
identity. Repository automation separately verifies that the canonical
repository-local root is ignored.

Synthetic inputs authored specifically to test a stable parser or driver
contract are source fixtures rather than captured attempts. They remain
eligible for tracking only under the disclosure boundary in EDA Tooling.

## Binding Resolution

Every tool uses exactly this precedence:

1. an explicit entry in the supplied local configuration;
2. the current process environment; and
3. module-system discovery only when neither earlier source yields a valid
   binding.

The environment source searches `PATH` by descriptor executable order, then
provider-declared tool-root variables by declared order. An already loaded
module that changed `PATH` is therefore an environment binding, not a new
resolution tier. Candidate probes may skip an environment candidate that does
not realize the exact selected semantic tool identity and continue within the
same tier. They do not continue to a lower tier after a valid candidate is
selected.

Module discovery is capability-based rather than distribution-name-based.
Generated Bash tries, in order:

1. the explicit local initialization script;
2. an already defined `module` function when no explicit script was supplied;
3. `$MODULESHOME/init/bash` when present;
4. `/etc/profile.d/modules.sh`;
5. installed Lmod profile scripts;
6. `/usr/share/Modules/init/bash`; and
7. `/usr/share/modules/init/bash`.

This covers common EL and Debian/Ubuntu Environment Modules layouts and common
Lmod installations without branching on `/etc/os-release`. Discovery uses the
portable terse module interface. Lmod `spider` may supplement hierarchical
module discovery after Lmod is identified, but it is not the portable
baseline.

Provider descriptors supply a deterministic ordered list of module aliases.
An adapter places explicitly validated current aliases before generic suite or
site aliases. Catalog maintenance, rather than runtime date arithmetic, keeps
those validated defaults within the provider's supported release generation,
normally the current two-year tool generation. For an unversioned alias, the
module system owns site-default selection. Loom does not parse version strings,
rank arbitrary installation-directory names, or implement a competing
"latest" policy. A successful probe records both the requested activation and
the exact ordered loaded-module closure. Final execution loads the frozen exact
closure and verifies the resolved executable and version instead of repeating
discovery.

The same compatibility probe applies to an executable selected by explicit
configuration or the current environment. Explicit configuration has highest
binding precedence, but it cannot force an incompatible executable to realize
the already selected semantic provider. A deliberately selected older release
is usable only when that provider binding declares it compatible; otherwise
resolution fails closed. No resolver searches vendor installation trees to
find an allegedly newer release.

Module output is not parsed from presentation-oriented `module avail` text.
Discovery and verification use terse output, the loaded-module state, the
resolved executable, and the provider version probe. Module activation always
occurs in generated Bash; Loom does not attempt to reproduce module environment
mutations in C++.

After module activation, a provider may resolve its executable from a declared
tool-root variable plus a normalized relative launcher path before consulting
the resulting `PATH`. This remains part of the module-discovery tier. It lets a
suite module select its supported launcher explicitly, including a 64-bit
launcher when a presentation default is unusable, without embedding a host
installation path. The resolved executable is canonicalized and frozen just
like a `PATH` result.

## Provider-Owned Architecture Selection

When a supported 64-bit launcher or mode exists, the provider descriptor must
prefer it over a 32-bit launcher. The descriptor owns the launcher-relative
path, required suite environment mode, and mandatory architecture argument
tokens. These values are projected into the structured command and generated
script; they are not caller-supplied generic arguments and cannot be removed
through local provider options.

The VCS provider includes `-full64` in every compile, elaboration, and link
command that creates the simulator, not only in its version probe. The Xcelium
provider resolves its declared 64-bit launcher. Cadence DDI providers require
the suite's 64-bit execution mode when the suite exposes one. Vivado and
Quartus Prime providers select their registered 64-bit launchers. A candidate
that provides only an incompatible 32-bit executable is unavailable; Loom does
not install compatibility libraries or silently change architecture mode.

The selected executable, architecture mode, mandatory tokens, and probe result
are frozen in the local binding and bundle manifest. Host word size is
normally invocation provenance rather than HardwareImplementation identity.
Any argument that can change the formal generated result remains owned by the
exact semantic model or generator binding regardless of where the provider
places it on the command line.

## Runtime Binding

Tool binding and PolyArch/container binding are resolved independently with the
same explicit-configuration, environment, then module precedence. `host`
requires direct execution. `polyarch_container` requires a verified
PolyArch/container binding. `auto` prefers the container only when the complete
tool/runtime composition passes preflight; otherwise it freezes a host binding
and records the rejected composition reason.

Orthogonal selection does not imply that every pair composes. Preflight proves:

- the exact executable and result-affecting dependencies are available in the
  final runtime;
- the provider version probe matches the selected semantic binding;
- bundle inputs, work directory, declared outputs, and required mounts are
  accessible;
- required environment-variable names are present at execution time; and
- a provider-owned or site-owned wrapper does not conflict with an outer
  PolyArch/container runtime.

An explicitly requested runtime that fails preflight is an error. Runtime
selection never changes after a bundle is frozen, and execution failure never
causes a silent host/container or tool fallback.

PolyArch/container owns its image build, mount, network, namespace, and runtime
behavior. Module systems own their activation mechanics. Schedulers, shells,
containers, and site policy own resource limits. Loom records the exact
resolved runtime identity needed for provenance but does not recreate any of
those services.

## Bundle Contract

The manifest schema owned by this section is:

```text
loom.external_tool_invocation 2.0

SemanticInvocationClosure =
    CandidateGenerator {
      exact typed input bindings
      exact ResolvedCandidateGeneratorBinding canonical bytes
      derived CandidateGeneratorBindingIdentity
    }
  | Evaluation {
      exact EvaluationRequest ArtifactRootReference
    }

ExternalToolPreparationContext {
  strict adopted LocalToolConfig
  bundle_destination
}

PreparedExternalToolInvocation {
  bundle_root
  manifest_sha256
}
```

The context and prepared handle are ephemeral nonsemantic C++ values. Their
paths never enter an Artifact, Request, Evidence, or generator binding. The
prepared handle does not own or recover the semantic closure; every import
receives the full typed closure again and recomputes its expected manifest.
`manifest_sha256` is only an integrity and lookup key.

The 2.0 manifest uses stable closure tags `CandidateGenerator = 0` and
`Evaluation = 1`. Canonical JSON spells them `candidate_generator` and
`evaluation`. Candidate resolved-binding canonical bytes and all descriptor-
derived identity bytes are lowercase hexadecimal with fixed digest length
where applicable. The binding's own DSE codec and adopter remain authoritative;
the bundle JSON parser cannot reinterpret those bytes. This is a major change
from manifest 1.0, whose free semantic-binding field cannot be imported as a
typed 2.0 closure.

Bundle finalization is failure-atomic. A complete bundle contains:

```text
tool-invocation.json
run.sh
drivers/...
inputs/...
outputs/...
```

`tool-invocation.json` is the sole bundle manifest. It records:

- the bundle schema and version;
- one exact `SemanticInvocationClosure`;
- materialization rows that reference the closure's typed input slots or exact
  Request-owned Artifacts and bind them to relative paths;
- the SHA-256 digest of every materialized driver and input byte sequence;
- every provider-declared external input slot, its semantic fingerprint, and
  either its materialized relative path or frozen absolute local path;
- the mechanically derived provider semantic identity and provider-form tag;
- frozen tool and runtime bindings, their resolution sources, and version
  probe results;
- the structured version-probe arguments, accepted exit codes, required
  marker, and stable-line selector used to reproduce that result;
- the module initialization path, requested activation, and exact loaded
  module closure when used;
- commands as token arrays, not shell fragments;
- required inherited environment-variable names, never their values;
- declared driver, input, output, raw-report, and completion-record paths; and
- the exact semantic-descriptor-derived result importer identity.

For a Candidate Generator, the full resolved binding remains present in the
closure and the stored `CandidateGeneratorBindingIdentity` must equal a fresh
derivation from it. For Evaluation, the exact Request recovers the full resolved
model binding. A compact digest is never sufficient to adopt configuration,
select a descriptor, or invoke an importer, and the bundle alone is never a
binding authority.

The result-importer identity is the verification digest:

```text
SHA-256(
  bytes("loom.external_tool_importer.v1\0")
  || u64be(length(exact semantic descriptor-reference bytes))
  || exact semantic descriptor-reference bytes
  || u32be(ProviderForm::ExternalPrepareImport))
```

The manifest stores this digest as 64 lowercase hexadecimal characters and the
importer recomputes it from the full typed closure. It is not a callback name,
dynamic-library symbol, or importer selection authority.

All bundle-owned paths are relative to the bundle root. Frozen host
executables, module initialization paths, and directly referenced external
input files may be absolute. Paths reject lexical parent traversal and escaping
symlinks. Commands and local-config strings are escaped as data when projected
into Bash; no value becomes an unevaluated shell fragment.

Provider Tcl, Python, and equivalent driver scripts are deterministic
projections of exact semantic inputs and the exact provider binding. `run.sh`
is the executable projection of the bundle manifest and frozen local binding.
Neither is an independently editable authority. The script performs no tool,
module, or runtime search. It initializes the frozen module closure when
needed, validates every materialized content digest, rehashes every directly
referenced external file, validates the frozen provider version with the
descriptor's exact exit-code and stable-line rules, invokes the provider
driver, retains raw stdout/stderr and reports in declared locations, and
atomically publishes one completion record.

Inputs and expected workload observations are materialized from exact owner
Artifacts. Explicit PDK, library, macro, or IP files may be materialized beneath
the ignored bundle or referenced at their frozen local paths according to the
provider projection contract; equal verified bytes have identical semantic
input identity. A bundle never derives its expected functional result from the
same RTL provider that it is testing.

When an input is a `HardwareImplementation`, preparation first strict-imports
its exact typed `representation_root`, reads every required logical byte
sequence only through the referenced `BlobDigest` and `BlobStore`, rehashes the
bytes, and derives the top object from that root. A caller cannot substitute
raw RTL, a top name, a previous work directory, or the most recent vendor
database. Every `GenerationConstraint`, external binding, and memory binding
is consumed exactly, mechanically preserved, or rejected before bundle
materialization; the local tool or driver cannot supply a second constraint
body or hidden default.

## Execution And Collection

Bundle preparation is the default operation. A Loom command may optionally
execute the generated top-level script and wait for its exit, but that launcher
is a thin script invocation. It does not implement a second environment model,
process-tree supervisor, cgroup manager, memory controller, container runtime,
retry engine, scheduler, or license manager.

Each concurrent work unit receives an independent finalized bundle and work
directory. Make, Ninja, Slurm, a shell, a container orchestrator, or another
caller may execute bundles in parallel. No bundle depends on mutable process
environment left by another bundle.

The completion record is nonsemantic attempt state and is written atomically.
It distinguishes launch or activation failure, tool exit, missing declared
output, and successful driver completion. A signal, externally enforced
timeout, scheduler cancellation, resource limit, or interrupted host may leave
no valid completion record; that remains an incomplete attempt.

A prepared bundle with no valid completion record is merely incomplete. Loom
does not infer whether an external process is still running, acquire an
execution claim, retry the script, or create a replacement attempt. The caller
or its external execution owner decides whether to wait, cancel, rerun, or
prepare another owner attempt and is responsible for preventing concurrent
writes. A new attempt may retain the same semantic WorkUnitKey but receives an
independent bundle. None of these execution choices changes semantic identity
or introduces a Loom Job state machine.

The shared strict-import helper reads only the exact manifest, valid completion
record, and declared outputs. It verifies attempt integrity and returns one
ephemeral immutable output snapshot to the descriptor-owned importer; it never
scans a scratch directory or infers the nearest report. This helper is a
library operation, not a third semantic importer or persistent output owner.

A Candidate Generator importer finalizes only its descriptor-owned Artifact
outputs and returns dense descriptor output bindings plus typed lineage
contributions; the central `InvocationManifest` alone validates and records
those contributions. A flow that derives hardware first finalizes the new
`HardwareImplementation`; it cannot publish Evidence from the generator
import. An Evaluation importer consumes an exact finalized
`EvaluationRequest`, first finalizes any descriptor-owned output Artifacts such
as `SimulationExecution`, and returns their dense descriptor output bindings
plus one normalized `EvaluationEvidenceOutcome` to the EvaluationEvidence
finalizer. Neither finalizer scans ArtifactStore to discover output membership.
Generation reports remain attempt
material in the baseline contract. An Evaluation over the finalized
implementation prepares its own bundle, so one tool execution is never
silently adopted by two semantic descriptors. Missing, malformed, partial, or
incompatible output publishes neither a partial implementation nor partial
Evidence.

Raw logs, reports, waveforms, tool databases, and the completion record remain
owner-attempt material until an exact raw-bundle Artifact owner is defined.
Normalized metrics and findings remain owned only by Evaluation.

## Failure Contract

Preparation and import distinguish at least:

- invalid local configuration;
- unavailable or incompatible explicit binding;
- no environment or module candidate;
- module initialization or activation failure;
- tool/runtime composition failure;
- missing required runtime environment;
- provider version mismatch;
- bundle finalization failure;
- tool execution failure;
- missing declared output; and
- result-import or normalization failure.

No failure authorizes a different semantic model, unbounded retry, alternate
tool, changed runtime, estimate substitution, or best-so-far result. Execution
limits may make an attempt incomplete but cannot change formal selection.

## Validation Anchors

Stable tests cover:

- strict local JSON and absence of implicit local-config loading;
- explicit external-file entries resolving only named ordinary files, with
  symlink, mutation, digest mismatch, glob, and recursive scan rejected;
- equal external-file bytes at different local paths producing the same
  semantic binding while preserving distinct nonsemantic path provenance;
- explicit binding precedence and fail-closed invalid explicit paths;
- environment selection before any module probe;
- Environment Modules and Lmod initialization across common EL and
  Debian/Ubuntu layouts;
- exact validated module aliases preceding generic site aliases without
  runtime version-string ranking;
- site-default module selection followed by exact loaded-module freezing;
- module-root launcher selection without changing resolution-tier precedence;
- preferred 64-bit launcher selection, rejection of an incompatible 32-bit
  candidate, and mandatory VCS `-full64` projection into compile and
  elaboration commands;
- nonzero successful version statuses and stable-line normalization;
- independent tool/runtime selection plus rejected incompatible composition;
- shell-safe projection of adversarial paths, arguments, and module names;
- deterministic byte-identical manifests and scripts from identical inputs;
- exact HardwareImplementation bytes and top derived from its representation
  root, with raw-source, top-name, and prior-workdir substitution rejected;
- independent parallel bundles with no shared mutable environment;
- atomic completion publication and rejection of missing or partial results;
- descriptor-owned prepare and import as separate calls, with an incomplete
  bundle remaining nonsemantic while any retry decision stays caller-owned;
- generator import publishing no Evidence and evaluator import requiring an
  exact finalized Request;
- derivation-before-Evaluation and exact Request/implementation coupling; and
- absence of secrets, cgroup policy, memory supervision, and raw report fields
  from semantic Artifacts and Evidence.

Tests do not pin distribution names, human `module avail` formatting, one
container engine, vendor report wording, process identifiers, or scheduler
behavior.

## Exclusions

External tool invocation does not:

- define Fabric, RTL, implementation, Evaluation, or deployment semantics;
- own result-affecting tool options or hidden defaults;
- acquire or expose credentials and license values;
- build container images or manage mounts, networks, namespaces, or cgroups;
- enforce memory, CPU, wall-time, process-tree, or host concurrency limits;
- schedule workflows, retry tools, or select fallbacks;
- normalize reports without the exact provider importer; or
- promote raw execution material into semantic Artifacts.
