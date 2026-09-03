# External Tool Invocation

This specification owns Loom's nonsemantic boundary for binding local external
tools, locating explicitly named machine-local external files, and
materializing independently executable invocation bundles. It does not own
tool, target, library, or IP semantics, process supervision, resource
isolation, scheduling, or Evaluation result schemas.

## Ownership

`loom.external_tool.backend_catalog 1.0` is the sole owner of every backend
tool's logical key, official product name, static typed local descriptor, and
validated release profiles. The descriptor declares executable names,
provider-recognized environment roots, ordered module candidates, runtime
compatibility, the version probe, and the exact local binding fields expected
from its semantic owner. The catalog is closed, deterministic, iterable, and
rejects duplicate keys, official names, conformance features, and provider
descriptor instances.

```text
BackendToolCatalogEntry {
  logical_tool_key
  official_product_name
  provider_descriptor
  validated_releases[] {
    conformance_feature
    module_alias_ref: optional
    exact_version_probe
  }
}
```

A validated release profile records a repository conformance baseline. It may
enable tests only after its exact structured probe succeeds for the executable
selected from the current test environment. Executable presence alone is not a
release match. A profile does not select a semantic model, authorize a tool
build for a Request, or replace the exact result-affecting provider build in a
model or generator binding.

Adapters reference their catalog entry and cannot repeat a logical key,
official name, module release, or version-probe convention. Machine-local
configuration may contain an unused key, but only a registered catalog entry
can consume it or create a provider capability. Lit and other conformance
harnesses consume a machine-readable projection of this catalog; they cannot
maintain another executable/version/feature table.

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

The mapped-RTL mode of `loom-system-run` is intentionally stricter. It requires
both `--mapped-rtl-local-tool-config=<path>` and
`--mapped-rtl-provider-build=<exact normalized probe line>`. The selected HDL
simulator's `tools` entry must contain an explicit executable or module
binding. An omitted binding, including a stanza that contains only inherited
environment names or provider options, cannot authorize ambient environment,
`PATH`, provider-default, or module-alias discovery for this mode. The mapped
RTL provider resolves and probes that explicit binding through the ordinary
typed provider contract, requires exact equality with the declared build, and
accepts the release only through the backend catalog's validated-release
relation. This stricter invocation boundary does not change the general
three-tier resolution rule for other consumers.
The driver's mapped-RTL job and model-thread options remain their sole owner;
the selected tool stanza cannot repeat them as provider options.

The local configuration is strict versioned JSON. Its initial authoring shape
is:

```text
LocalToolConfigV1_1 {
  schema = "loom.local_tool_config"
  version = "1.1"
  experiment_root?: absolute_path
  module? { init: absolute_path }
  external_files?: {
    local_file_key: absolute_file_path
  }
  external_file_trees?: {
    local_file_tree_key: absolute_directory_path
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

`inherit_environment` admits only execution-availability state such as a
provider license or credential handle. Values are intentionally absent from
the manifest, logs, and reusable-result identity. A seed, effort, PVT value,
library selector, feature switch, or any other result-affecting option cannot
be inherited through this plane; its semantic owner must project it through a
typed provider option, generated driver, structured command, semantic input,
or external content fingerprint.

`external_files` maps opaque machine-local keys to explicitly named absolute
ordinary files. `external_file_trees`, added compatibly in local-config 1.1,
maps separate opaque keys to absolute directory roots. Local-config 1.0 remains
valid but cannot spell a file tree. The resolver rejects duplicate canonical
paths, symlinks, special files, and observed mutation while reading. It hashes
every configured ordinary file by SHA-256 and produces a nonsemantic local
projection for the bundle. When several configured files or trees have
identical required contents, the resolver freezes the lexicographically first
canonical path; the choice cannot change semantics.

The keys and path spellings are not target,
HardwareImplementation, Request, or Evidence identity. The map cannot declare
technology identity, target, corner membership, provider compatibility,
library role, expected fingerprint, tree membership, directory filter, or glob. Those facts
remain owned by the exact provider descriptor and resolved semantic binding.
Listing a file never authorizes recursive scanning, PDK import, tool
installation hashing, or implicit file discovery.
It also never authorizes the configured path as a command executable. A
provider may resolve one of its closed auxiliary-tool roles from this local
map, but only the resulting typed `auxiliary_tool_executables` record grants
that exact path `argv[0]` authority in a finalized 2.4 bundle.

An `ExternalFileTreeRequirement` is one provider-owned input slot plus a
nonempty canonical sorted-unique list of `(relative_path, SHA-256)` ordinary
file members. Relative paths are canonical, nonempty, and cannot traverse
parents. Resolution recursively inspects a configured tree only to prove exact
equality with that already frozen list: a missing, extra, changed, symlink, or
special member is invalid. The requirement, not the scan, owns semantic
membership and layout. Empty directories have no semantic role. This is the
only logical-directory contract; it cannot promote ambient membership into a
binding or reuse the removed platform directory-manifest mechanism.

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
An EDA bundle root must be outside the source worktree or in a Git-ignored
repository-local directory. An explicit `experiment_root` in the supplied
local configuration has highest priority. Without it, repository automation
selects the first usable location in this order: an existing Git-ignored
top-level `build/` in the current worktree; `/scratch/loom-<uid>` when
`/scratch` has more than 100 GiB available; `~/.cache/loom`; and
`/tmp/loom-<uid>`. The resolver creates the selected external or user-local
directory. Every attempt receives an independent child beneath that root.

Reusable external-tool results use the same resolver rather than another path
policy. An explicit absolute `LOOM_EXTERNAL_TOOL_CACHE_ROOT` environment value
overrides only the cache location. Otherwise the cache is the
`external-tool-cache` child of the resolved experiment root, so an ordinary
build uses `build/external-tool-cache` while an explicitly selected experiment
root, large scratch root, user cache, or temporary fallback retains the same
precedence. The cache root is private to its owning user and carries a
Loom-owned format marker. Repository
`distclean` may remove that exact marked namespace, including an explicit
external cache root, but cannot recursively clean an unmarked parent or infer
other cache locations.

The same placement rule applies to a local Artifact Store or Blob Store that
contains direct EDA-generated implementations, Evidence, invocation records,
or their payloads. Selecting a local output path does not make the path, its
contents, or Git ignore state semantic input. Loom's compiler libraries do not
invoke Git, edit ignore rules, or reinterpret repository tracking as Artifact
identity. Repository automation separately verifies that a repository-local
root is ignored. A pre-commit staged-path check rejects top-level `build/` and
`loom-local-config.json`; it is publication hygiene rather than a semantic
validator and does not inspect commit history or tool-output content.

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

The versioned aliases and their conformance probes are catalog-owned release
profiles. Generic site aliases remain descriptor-owned fallbacks and never
imply that a particular repository baseline was verified.

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
loom.external_tool_invocation 2.4

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

ExternalToolSemanticContract {
  provider_identity
  semantic_closure: SemanticInvocationClosure
  result_importer_identity
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

`ExternalToolInvocationBundleSpec` and
`ExternalToolInvocationImportExpectation` each consume one complete
`ExternalToolSemanticContract`. CandidateGenerator and Evaluation are the only
owners that derive this value. An adapter may pass the value through but may
not author any of its three fields, expose the low-level owner codecs as an
adapter protocol, or assemble a contract from display names and private
bytes.

The 2.x manifest uses stable closure tags `CandidateGenerator = 0` and
`Evaluation = 1`. Canonical JSON spells them `candidate_generator` and
`evaluation`. Candidate resolved-binding canonical bytes and all descriptor-
derived identity bytes are lowercase hexadecimal with fixed digest length
where applicable. The binding's own DSE codec and adopter remain authoritative;
the bundle JSON parser cannot reinterpret those bytes. This is a major change
from manifest 1.0, whose free semantic-binding field cannot be imported as a
typed 2.0 closure.

Manifest 2.1 compatibly adds `external_file_trees`; the 2.0 form remains
importable and denotes an empty tree-input list. A 2.0 manifest cannot contain
the new field. Manifest 2.2 compatibly adds `tool_produced_executables`; the
2.0 and 2.1 forms remain importable and denote an empty produced-executable
list. Manifest 2.3 compatibly adds optional `parallel_command_groups`; its
absence and all older forms denote fully ordered command execution. Manifest
2.4 compatibly adds the required canonical `auxiliary_tool_executables` array.
Each record has a provider input slot, machine-local key, canonical absolute
path, and exact content digest. This typed domain is the only owner for
auxiliary compiler, linker, archiver, build-tool, and provider-built
build-time launcher commands, including an executable that a tool-generated
makefile names through a make variable of a frozen command; ordinary
`external_files` are data and never acquire `argv[0]` authority. An empty 2.4
array and every older manifest denote no auxiliary command owners. Such tools
are not inherited from the execution environment after bundle finalization.
Their roles and digests participate in tool provenance and cache identity, and
the launcher revalidates both the bytes and executable permission before each
attempt. Older manifests permit only the primary frozen tool and listed
tool-produced executables as command owners. An older manifest cannot contain
a field or command form introduced by a newer form. Bundle
finalization is failure-atomic. A complete bundle contains:

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
- every provider-declared external ordinary-file input slot, its semantic
  fingerprint, and either its materialized relative path or frozen absolute
  local path;
- every typed auxiliary-tool slot, machine-local key, canonical executable
  path, and exact content fingerprint;
- every provider-declared external file-tree input slot, its canonical member
  paths and fingerprints, and its frozen absolute local root;
- the mechanically derived provider semantic identity and provider-form tag;
- frozen tool and runtime bindings, their resolution sources, and version
  probe results;
- the structured version-probe arguments, accepted exit codes, required
  marker, and stable-line selector used to reproduce that result;
- the module initialization path, requested activation, and exact loaded
  module closure when used;
- commands as token arrays, not shell fragments, whose executable is the
  frozen primary tool, an exact `auxiliary_tool_executables` record, or one
  exact listed tool-produced executable; a generated controller command may
  additionally name other listed produced executables as exact argument
  tokens;
- canonical sorted, nonoverlapping parallel command groups, each an
  end-exclusive range of adjacent frozen-tool commands plus a bounded worker
  limit;
- canonical `work/`-relative tool-produced executable paths, when a compiler
  must generate a program that a later command executes;
- required inherited environment-variable names, never their values;
- declared driver, input, output, raw-report, and completion-record paths; and
- the exact semantic-descriptor-derived result importer identity.

For a Candidate Generator, the full resolved binding remains present in the
closure and the stored `CandidateGeneratorBindingIdentity` must equal a fresh
derivation from it. For Evaluation, the exact Request recovers the full resolved
model binding. A compact digest is never sufficient to adopt configuration,
select a descriptor, or invoke an importer, and the bundle alone is never a
binding authority.

The provider identity is exactly the
`implementation_semantic_identity` recovered from the closure's exact
CandidateGenerator or Evaluation model descriptor. Its provider form must be
`ExternalPrepareImport`. The CandidateGenerator owner encodes typed input
bindings as `u64be(binding_count)` followed by each dense `u32be(slot)` and
`u64be(artifact_count)` plus Common canonical root-reference bytes. It encodes
the resolved binding as the canonical descriptor-reference bytes, one
length-framed canonical config view, and the exact 32-byte config digest. The
Evaluation owner encodes a model descriptor reference as
`u32be(schema_major) || u32be(schema_minor) || u32be(model_kind)` and places
the exact `EvaluationRequest` ArtifactRootReference in the closure. These are
owner codecs, not ExternalTool or adapter codecs.

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

The shared ExternalTool layer owns only the domain-separated digest framing.
The CandidateGenerator and Evaluation derivation APIs supply their exact
descriptor-reference bytes and return the complete contract. Bundle
finalization and strict import therefore compare one owner-derived value; five
EDA adapters cannot become five independent implementations of identity or
closure encoding.

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
referenced external ordinary file and every member and membership count of a
referenced external file tree, validates the frozen provider version with the
descriptor's exact exit-code and stable-line rules, invokes the provider
driver, executes only the manifest-frozen command schedule, retains raw
stdout/stderr and reports in declared locations, and atomically publishes one
completion record.

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
retry engine, dynamic scheduler, or license manager. Its only command-level
concurrency is the exact bounded fork-join schedule frozen in the manifest.

A parallel group contains at least two commands, has at least two and no more
workers than commands, and may contain only commands whose executable is the
frozen tool. It cannot consume a tool-produced executable. Group ranges are
canonical, sorted, and nonoverlapping. Commands outside a group execute in
ordinal order; every group boundary and every worker-sized chunk within a
group is a barrier. The launcher starts no more than the frozen worker limit,
waits for the complete chunk, and then collects stdout and stderr in command
ordinal order. If several commands fail, the lowest failing command ordinal
selects the tool exit code. A launcher infrastructure failure retains its
typed launcher exit code and cannot be hidden by a tool failure.

The launcher opens every chunk's ordinary command streams before starting any
tool process and collects only through those retained descriptors. A tool
cannot replace a scratch path with a link, pipe, device, or growing file to
redirect or block collection. Status is bounded to four bytes, timing output
to 64 KiB, and each command stdout or stderr stream to 1 GiB; exceeding a
bound is a launcher infrastructure failure rather than a tool exit.

Each started command produces an attempt-bound operational observation with
its command ordinal, wall duration, and exit code. The observation file is
published atomically and bound to both the manifest digest and fresh attempt
token. It is not semantic input, declared output, cache content, or evidence;
a cache hit therefore has no command-execution observations. These timings
identify provider compile and controller costs without relabeling them as
simulation time.

Provider-local build options may freeze one total inner build-job budget and
one outer build-worker limit. The provider deterministically divides the total
budget across each simultaneously active chunk, writes the resulting inner
job counts into generated drivers, and freezes the outer limit in the manifest
group. It cannot multiply the total job budget by the number of commands.
Machine policy selects the worker limit from available memory; the generic
launcher neither infers RSS capacity nor expands it to the host thread count.

Each concurrent work unit receives an independent finalized bundle and work
directory. Make, Ninja, Slurm, a shell, a container orchestrator, or another
caller may execute bundles in parallel. No bundle depends on mutable process
environment left by another bundle.

Executions of the same finalized bundle-root path are instead serialized by
one exclusive fence at the stable adjacent path
`<bundle-root>.loom-execution.lock`. The fence file is operational, remains in
place across generations, and is never replaced or removed by execution. A
direct `run.sh` acquires it before clearing prior operational state and
atomically publishing a fresh attempt token. The observed library executor
performs the same transition while holding the fence, then passes inherited
fence and bundle-root descriptors to every generated launcher. It retains its
descriptor through launcher termination, completion validation, command-
observation collection, and receipt sealing. Every conforming launcher
descendant retains the inherited fence until it has relinquished all authority
to write the bundle, so a detached descendant still delays the next
generation. Attempt creation is part of the fenced execution entry point
rather than a public token-publication operation. The fence and token are
nonsemantic: neither changes the prepared manifest, invocation identity,
persistent result-cache key, or owning WorkUnitKey.

After admission, the opened bundle-root descriptor is the live execution
identity. Completion, command-observation, cache, cleanup, and receipt-sealing
operations use that descriptor and cannot be redirected to a different inode
later bound to the logical bundle-root path. Recovery import begins after the
live descriptor lifetime and therefore resolves the current durable generation
through the logical path.

Observed execution samples its caller-owned execution control while waiting
for fence admission. A stop before ownership returns the typed
`ExternalToolExecutionAdmissionStoppedError` and leaves the prior token,
completion, observations, and declared outputs untouched. Once the executor
publishes a fresh token, interruption belongs to that new generation and uses
the sealed stopped or incomplete attempt semantics below.

Every command before a tool-produced executable uses either the exact frozen
primary tool binding or an exact ordinary executable frozen by path and digest
in `auxiliary_tool_executables`. An `external_files` data record can never own
a command. A listed produced path is canonical, relative, strictly below
`work/`, and absent from materialized inputs and declared outputs. The shared
launcher removes every listed path before entering the tool. Immediately
before a later command may execute one directly, or before a listed generated
controller may receive another listed executable as an exact argument token,
every referenced path must be a newly created ordinary executable file and not
a symbolic link. The manifest freezes every path and the complete controller
argument vector; provider logic must reject any controller-child set that
differs from its semantic input. The controller remains provider logic for one
cooperative simulator closure, not a second bundle launcher or a general
process scheduler. An arbitrary host executable, bundled shell fragment,
manifest-hidden child path, previously built simulator, or output-directory
program is never admitted by this form. This form exists for compile-then-run
tools such as Verilator, including one modeled hierarchy implemented by
cooperating generated processes; a tool that can complete its work directly
uses only frozen-tool commands.

The completion record is nonsemantic attempt state and is written atomically.
It distinguishes launch or activation failure, tool exit, missing declared
output, and successful driver completion. A signal, externally enforced
timeout, scheduler cancellation, resource limit, or interrupted host may leave
no completion record; that remains an incomplete attempt.

A prepared bundle with no completion record is merely incomplete. A present
malformed, noncanonical, or manifest-unbound completion is instead an
integrity failure. Loom does not infer whether an external process is still
running, retry the script, or create a replacement attempt. The caller or its
external execution owner decides whether to wait, cancel, rerun the fenced
prepared root, or prepare another owner attempt. A new generation may retain
the same semantic WorkUnitKey. None of these execution choices changes
semantic identity or introduces a Loom Job state machine.

The shared expectation-bound attempt importer validates the prepared-manifest
handle, exact provider identity, semantic closure, importer identity, semantic
and external inputs, and declared-output set before exposing any result. A
present completion must bind that validated manifest before the importer
returns one closed nonsemantic outcome:

```text
ExternalToolInvocationAttemptOutcome =
    Incomplete
  | Failed { InvocationCompletionStatus, exit_code }
  | Imported { immutable declared-output snapshot }
```

Only `Success` completion may produce `Imported`; only that path opens,
verifies, and snapshots declared output bytes. `Incomplete` and `Failed`
contain no output snapshot. A success-only compatibility wrapper projects the
first two alternatives back to import errors. Neither API scans a scratch
directory or infers the nearest report. They are library operations, not a
third semantic importer or persistent output owner. The exact generator or
evaluator descriptor owns any later derivation into its semantic outcome;
External Tool Invocation does not define a universal status mapping.

There are exactly two import authorities. Recovery-only import has no live
executor witness and therefore accepts only the valid completion bound to the
currently published durable generation. A current-process execute-to-import
path retains the sealed execution observation and must pass it to the
receipt-bound importer; it cannot fall back to recovery import. That importer
proves that the executor issued the observation, requires its sealed manifest,
token, exit disposition, cache disposition, and command observations to match,
and rechecks the same current generation and completion after snapshotting all
declared outputs. The receipt owns no output bytes; only the returned imported
bundle owns the immutable snapshot.

The observed executor releases its fence descriptor when execution and receipt
sealing return, not after a later semantic import. An inherited descendant
remains a fence co-owner until it relinquishes bundle-write authority, and the
next generation cannot begin before every such holder closes. A later
generation may therefore supersede an unimported receipt only after the prior
execution lifetime ends. In that case the older receipt must reject, the later
generation may import, and an older conforming launcher cannot overwrite the
later generation's operational state or outputs.

A launcher descendant that explicitly closes its last inherited fence
reference, calls `LOCK_UN` on the shared open-file description, or otherwise
relinquishes the fence while retaining authority to write the bundle violates
this execution contract. The generic launcher cannot prove quiescence for such
an arbitrary detached process. A provider that cannot establish the
conforming-descendant contract for its complete tool closure must use an
external lifecycle owner that proves quiescence before releasing the fence,
allocate an independent attempt root, or return its owning domain's typed
`Unsupported` outcome before execution. The generic External Tool layer does
not claim containment of a process that has relinquished its fence authority.

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

## Persistent Result Reuse

External Tool Invocation owns one optional content-addressed result cache for
successful prepared invocations. Adapters, Candidate Generators, Evaluators,
and tests cannot define provider-private cache keys or cache directories. A
cache entry contains only the exact declared output bytes from one successful
invocation. It contains no raw attempt log, semantic Artifact, Evidence,
completion authority, license value, host identity, or mutable job state.

The cache address is the ordered triple below. Each component uses a distinct
domain-separated SHA-256 codec; the three digests remain visible independently
rather than being replaced by one caller-authored string.

```text
ExternalToolResultCacheKey {
  input_material_sha256
  execution_configuration_sha256
  tool_version_sha256
}
```

`input_material_sha256` covers every materialized semantic input by canonical
relative path, exact source Artifact reference, and content digest, plus every
external-file slot and every external-file-tree member by logical slot,
relative member path, and expected fingerprint. `execution_configuration_sha256`
covers the exact provider and semantic closure, importer identity, structured
commands, inherited environment names, normalized generated-file bytes,
declared outputs, and tool-produced executable closure, including the exact
parallel command groups and worker limits. Generated-file and command
normalization replaces only manifest-known bundle, executable,
external-file, and external-tree paths with typed logical tokens; it does not
apply textual timestamp heuristics or reinterpret provider languages.
`tool_version_sha256` covers the logical tool key, normalized exact version
identity, exact version probe, and exact resolved launcher content digest. It
also covers the runtime kind and exact container key, version, version probe,
launcher content digest, and operating system when a container is selected.

Host-side cache diagnostics consume the invocation-local `LOOM_VERBOSE_LEVEL`
binding owned by
[Loom Full-Stack Architecture](spec-loom-stack.md#invocation-diagnostics).
Level one reports cache availability, hit, miss, discard, and publication
failures; level two additionally reports successful publication. The cache
does not parse a second environment binding, and the verbosity level never
enters prepared invocation semantics or the cache key. Diagnostics explicitly
projected into an external command, including an RTL simulation plusarg, are
mechanically derived from the Common-owned value and use the same spelling.
The external-tool cache normalizer removes exactly this closed presentation
argument before hashing command configuration. A caller or provider cannot
independently author it, and a prepared external invocation never inherits the
host binding implicitly.

Absolute bundle and executable paths, binding-source choice, module
initialization paths, requested or loaded module aliases, local external-file
keys and paths, cache location, process identity, file metadata and times,
completion times, scheduler state, license values, and diagnostic verbosity do
not enter the key. A result-affecting seed, effort, PVT condition, library
content, runtime component, or provider option must already appear in semantic
inputs, generated files, structured commands, external content, or the exact
tool/runtime identity. It cannot be omitted merely by labeling it local.

Only a canonical `Success` completion whose `outputs/` tree contains exactly
the lifecycle files, declared outputs, and their necessary parent directories
may publish an entry. Publication verifies and snapshots all declared outputs,
writes one private staging entry, and atomically renames it while holding the
exact-key writer lock. An extra file, directory, symbolic link, or special
entry makes the attempt non-cacheable because omitting it on restoration could
change a strict provider import. Failed, incomplete, cancelled, timed-out, or
partially published attempts are never cached. Concurrent readers of one key
observe either no entry or one complete entry; different keys remain
independently executable.

After a real tool reports success, the launcher revalidates the same manifest,
materialized inputs, external files and trees, required environment and module
closure, executable, and exact version identity without removing or rewriting
completed outputs. A change makes that attempt non-cacheable while preserving
its real completion and ordinary strict import. Thus a long-running tool cannot
publish its result under a key derived from inputs that changed during the run.

A hit is accepted only after the current prepared bundle, materialized inputs,
external inputs, environment requirements, executable binding, and exact
version probe have passed the same pre-execution validation used by a real
run. The cache entry schema, key triple, output membership, and every payload
digest are then checked. Corrupt or incomplete entries are discarded as
misses. Valid bytes are restored atomically beneath the current bundle, then a
cache-hit diagnostic log and new completion record are published for the
current attempt. The completion is written last and binds the current manifest
digest.
The ordinary expectation-bound importer still validates that completion and
all restored outputs; cache lookup never calls or replaces the semantic
importer.

Real-tool hardware conformance commands that predate provider bundle adoption
use the same cache root and ordered key domain through one ExternalTool-owned
command adapter. Its input component is the exact pre-execution working tree
plus content-addressed external path arguments; its configuration component is
the normalized argument vector and result-affecting compiler flag environment;
its tool component is the exact product version and resolved launcher digest.
A successful command publishes a copy-on-write snapshot of its complete
post-execution working tree and captured output streams. A hit validates that
snapshot, restores it beneath the current test work directory, and replays the
streams so subsequent independent binaries and output oracles still execute.
Version queries, nonzero exits, special filesystem entries, and results that
cannot be snapshotted without copying are not reused. This adapter is test
infrastructure for direct conformance commands, not another provider or a
semantic Artifact owner.

Cache presence, absence, corruption, worker order, and cache path cannot change
the semantic result or work identity. An unavailable entry causes real tool
execution. Failure to publish a cache entry after a successful real execution
is a nonsemantic diagnostic and cannot turn that execution into failure.
Runtime-configurable diagnostics may report keys and hit, miss, discard, and
publication events while normal execution remains quiet. Level one reports
cache lifecycle events; level two additionally reports the three independent
content digests. Neither level exposes inherited environment values.

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
- the mapped-RTL system driver requiring a declared build and an explicit
  selected-simulator binding before it imports a Deployment package;
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
- catalog uniqueness, adapter-to-catalog identity, and machine-readable
  conformance feature derivation from a successful exact release probe;
- rejection of presence-only or wrong-version feature discovery;
- independent tool/runtime selection plus rejected incompatible composition;
- shell-safe projection of adversarial paths, arguments, and module names;
- deterministic byte-identical manifests and scripts from identical inputs;
- compile-then-run bundles accepting only fresh manifest-listed `work/`
  executables produced after a frozen-tool command, with stale, absolute,
  escaping, materialized, declared-output, non-executable, and symlink paths
  rejected;
- exact HardwareImplementation bytes and top derived from its representation
  root, with raw-source, top-name, and prior-workdir substitution rejected;
- independent parallel bundles with no shared mutable environment;
- atomic completion publication, exact failed status preservation, and no
  output snapshot for absent or partial completion;
- completion-to-manifest binding before exposing failed or successful outcomes;
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
