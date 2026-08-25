# EDA Tooling

This document defines the portable boundary between Loom and external RTL,
FPGA, ASIC, formal, and physical-design tools. Local tool binding, script
materialization, and optional script execution are owned by
[External Tool Invocation](spec-external-tool-invocation.md); Evaluation
schemas are owned by [DSE and Evaluation](spec-dse-feedback.md).
Persistent implementation state is owned by
[Hardware Implementation](spec-hardware-implementation.md). The shared ASIC or
FPGA target manifest and technology-corner catalog are owned by
[Implementation Platform](spec-implementation-platform.md); exact library,
macro, IP, rule, and tool-bundled resource inputs are owned by the descriptor
that consumes them.

## Tool Descriptors And Bindings

An EDA `EvaluationModelDescriptor` declares an immutable observation
capability:

* required Artifact schemas;
* modeled phenomena and supported metrics/findings;
* required target, technology-corner, explicit-file, and tool-bundled resource
  input slots;
* full-execution method and determinism contract;
* parser/adapter semantic identity.

One resolved model binding supplies exact result-affecting inputs such as the
stable provider build, selected standard-cell or macro file fingerprints,
tool-bundled FPGA resource keys, corner mapping, constraint-translation
choices, parser version, and semantic effort. It never supplies an independent
constraint body when the subject HardwareImplementation already owns an exact
`GenerationConstraint` payload. Executable paths, module activation, license
servers, host selection, scratch roots, and parallel-job limits are invocation bindings.
They do not enter candidate semantics unless the model descriptor explicitly
declares a result-affecting value.

Loom selects by descriptor capability and exact model binding, not by a global
fidelity level or hard-coded workstation tool name.

Local tool and runtime resolution follows the shared explicit-configuration,
current-environment, then module-discovery precedence. An adapter does not
inspect `PATH`, source module initialization, choose a container, or read a
machine-local configuration itself. It supplies its provider descriptor and
consumes the frozen binding produced by the shared resolver.

The provider descriptor is referenced from the single
`loom.external_tool.backend_catalog 1.0` entry owned by
[External Tool Invocation](spec-external-tool-invocation.md#ownership). An
adapter cannot copy its logical key, official product name, validated module
release, or version-probe rule. Repository conformance features are derived
from the same catalog by real probes; they are not a second provider or
semantic-build registry.

An adapter consumes the exact `GenerationConstraint` payload of its
HardwareImplementation. It may translate that payload into vendor syntax, but
it cannot infer a hidden clock, reset exception, CDC waiver, false path,
multicycle path, floorplan rule, or timing target from report text or local
defaults.

A flow that creates a new `HardwareImplementation` is instead selected through
the central `CandidateGeneratorDescriptor` and
`ResolvedCandidateGeneratorBinding`. One physical tool invocation may derive
an implementation and retain raw reports as attempt material, but it
does not make EvaluationModelDescriptor an implementation generator.

## Capability Obligations And Provider Catalog

Hardware-backend completeness is stated against an explicit conformance
scope, not against a closed product matrix. A scope declares the following
capability obligations that it needs. Each obligation is realized by an exact
registered descriptor and binding:

| Capability obligation | Descriptor role | Accepted implementation | Published result |
| --- | --- | --- | --- |
| RTL or gate functional observation | evaluator | `Rtl` or `GateNetlist` | `EvaluationEvidence` and, for a real workload, `SimulationExecution` |
| ASIC logic synthesis | generator, then evaluator | `Rtl`, ASIC target, and declared library inputs | `GateNetlist`, then synthesis Evidence |
| ASIC physical implementation | generator, then evaluator | `Rtl` or `GateNetlist`, ASIC target, and declared physical inputs | `AsicPhysical`, then physical Evidence |
| parasitic extraction | generator | routed `AsicPhysical`, ASIC target, and declared RC inputs | extracted `AsicPhysical` |
| timing, area, or power observation | evaluator | one exact implementation, target, provider inputs, and conditions | normalized Evidence |
| physical verification | evaluator | one exact physical implementation and declared rule inputs | normalized findings |
| static FPGA implementation | generator, then evaluator | `Rtl`, exact FPGA ordering code, and provider resource binding | `FpgaPhysical`, `FpgaImage`, and implementation Evidence |
| implementation-space search | generator | exact typed flow inputs and decisions | immutable implementation candidates |

One tool process may realize more than one registered descriptor, but an
implementation is finalized before any evaluator observes it. Product names
do not create Artifact, Evidence, flow, or ecosystem identities. A run selects
exact generator and evaluator bindings; it never selects a global ecosystem
mode.

The built-in provider catalog initially covers these baseline routes and may
grow without changing the capability model:

| Ecosystem | Baseline providers | Additional provider roles |
| --- | --- | --- |
| open-source ASIC | Verilator, Yosys, OpenROAD with its registered timing and extraction engines | independent formal or physical-verification adapters when available |
| Synopsys ASIC | VCS, Design Compiler, Fusion Compiler, PrimeTime, and PrimePower | StarRC, IC Validator, Formality, and other compatible recent tools |
| Cadence ASIC | Xcelium, DDI Genus and Innovus, Joules, Tempus, and Voltus | Quantus, Pegasus, Cerebrus, and other compatible recent tools |
| AMD/Xilinx FPGA | Vivado | Vitis when an exact HLS, software-platform, or device-execution contract requires it |
| Intel/Altera FPGA | Quartus Prime Pro | additional device-programming or execution adapters with exact contracts |

An installed executable or suite directory is not catalog support by itself.
A provider is supported only after its descriptor, deterministic driver,
declared outputs, strict importer, failure mapping, and capability anchors are
registered. Built-in providers are maintained against explicitly validated
recent releases, normally from the current two-year tool generation. This is
a catalog maintenance rule, not a runtime subtraction from the wall clock and
not permission to scan installation trees or guess version order.

Catalog breadth does not require the Cartesian product of every provider,
platform, recipe, and operation. Every adapter has license-independent driver
and importer anchors; each baseline route has a real representative smoke
platform; broader platform and recipe coverage is scheduled as explicit
conformance work. A result claims only the descriptors, implementation state,
target and provider-input closure, and checks that actually completed.

OpenROAD is the required open-source physical provider, not a proxy for ASIC
signoff. Its exact implementation input is the Yosys-derived `GateNetlist`. A
coherent placed or routed database may publish the matching closed
`AsicPhysical` variant. The `Extracted` variant is legal only when the complete
parasitic payload closure is materialized. A layout stream is a payload of the
corresponding coherent physical implementation, not another representation.

Vivado and Quartus Prime target the exact vendor ordering code of an
`ImplementationPlatform`. Their verified provider build and exact device key
own access to the bundled primitive and timing database; Loom does not import
or hash that database. A successful implementation publishes an immutable
routed FPGA state before a separate full-device image implementation may be
derived. The initial contract is static full-device implementation; partial
reconfiguration and measured-device execution remain outside the baseline
capability obligations.

VCS and Xcelium initially provide functional RTL and gate-netlist simulation.
Timing-annotated simulation is not implied by a gate-netlist input. It requires
an explicit timing-annotation payload owner and model dependency before an
adapter may claim it.

Every provider uses the same shared tool and runtime resolver, invocation
bundle, completion contract, HardwareImplementation finalization, and Evidence
registries. An unavailable executable, license, target, external input,
primitive, or provider capability is typed `Unavailable` or `Unsupported`;
it never selects another provider or a lower-fidelity model implicitly.

The provider matrix names products for capability communication only. Exact
supported local keys and validated release profiles come from the backend tool
catalog; exact result-affecting builds still come from each resolved model or
generator binding.

DesignWare, ChipWare, AMD/Xilinx primitives, and Intel/Altera primitives are
RTL implementation providers rather than environment-discovery or Evaluation
providers. Their exact occurrence-scoped selection and dependency ownership is
defined by [Fabric To RTL](spec-rtl-lowering.md). Tool installation paths and
module aliases remain local invocation bindings.

Cerebrus is a candidate generator over exact typed implementation decisions,
not an evaluator, environment manager, or mutable-current-best authority.
Stratus and Vitis HLS may be candidate generators only after their descriptor
has an exact high-level body and protocol input contract. The baseline
Fabric-to-RTL path already produces RTL and does not route RTL back through an
HLS tool. Virtuoso may originate exact custom-cell or macro views for a
provider-owned external input binding; it is not a default digital RTL
implementation stage and does not turn those bytes into Platform fields.

## Invocation Bundles

An exact generator descriptor's successful `prepare`, or the prepared branch
of an evaluator descriptor's `prepare`, emits one finalized
`ExternalToolInvocationBundle` containing
exact materialized Artifact inputs, frozen references to declared external
files, generated constraints, workload inputs and expected observations where
applicable, provider Tcl/Python or equivalent drivers, a top-level Bash script,
declared output locations, and an exact importer identity. Command lines are
structured tokens before script projection. Machine-local paths, module
activation, inherited environment names, and PolyArch/container binding are
frozen into the nonsemantic bundle manifest. The script validates every frozen
external-file fingerprint before invoking the tool.

Generator preparation consumes exact typed input slots and one
`ResolvedCandidateGeneratorBinding`; evaluator preparation consumes one exact
`EvaluationRequest`. Central plan admission proves input readiness,
Artifact-family importers prove each exact Artifact and Blob closure, semantic
descriptor callbacks prove flow-specific compatibility, and the external layer
proves local tool/runtime availability. These owners do not restate one total
admission predicate.

For a downstream flow whose input is an existing HardwareImplementation, its
semantic callback accounts for the exact target, corner, external and memory
bindings, representation root, top object, and every `GenerationConstraint`
before a bundle is materialized. For the first implementation-producing flow,
root, top, and constraint are output facts validated by its importer instead.
A missing or unsupported required owner fails before the first point at which
that owner could be consumed. Raw RTL, a free top name, a caller-authored
semantic-binding string, or a backend-default constraint is never an alternate
input.

An evaluator whose exact valid Request is outside its stable provider
capability returns typed `Unsupported(RuntimeCapabilityUnavailable)` directly
from preparation. Evaluation finalizes that outcome against the exact Request;
the adapter does not generate a no-op script or fabricated completion record.
Candidate generation has no corresponding terminal semantic result and retains
the single prepared-bundle form.

The top-level script performs no discovery and does not contain a second copy
of result-affecting model or generator configuration. Tool options that can
change implementation or Evaluation output come only from the exact semantic
binding. The bundle may translate those typed values into vendor syntax.

Loom prepares bundles by default. Optional execution invokes the generated
script; resource isolation, limits, scheduling, container lifecycle, and
license services remain external. Independent bundles may be executed in
parallel without sharing mutable process environment.

After execution, the shared expectation-bound attempt importer verifies the
prepared handle, exact semantic expectation, attempt integrity, and
completion-to-manifest binding. It returns an incomplete attempt, the exact
failed completion status and exit code, or an ephemeral immutable declared-
output snapshot. Only successful completion opens declared outputs. The same
semantic descriptor's `import` operation interprets a successful snapshot or
derives its own typed non-success outcome from a validated failed attempt; the
external layer does not infer semantic failure kind from a process exit code.
A generator import finalizes a complete implementation and returns dense
descriptor output bindings plus lineage contributions but no Evidence; an
evaluator import finalizes any descriptor output Artifacts and returns their
dense descriptor output bindings plus one normalized
`EvaluationEvidenceOutcome` to the EvaluationEvidence finalizer. Neither path
scans ArtifactStore for result membership. An evaluator cannot mutate or
replace the subject. Generation reports are not reused by Evaluation in the
baseline two-call contract; the evaluator prepares a new exact bundle over the
finalized implementation.

## HardwareImplementation Generation

The initial RTL implementation is a `MechanicalDerivation` from exact Fabric,
one exact SpatialCore occurrence subject, exact `ConfigurationABI`, and the
resolved generator binding. It publishes that subject's closed `Rtl`
representation root as `loom.hardware_implementation 4.1`. A later flow that
consumes existing hardware state and preserves new state creates another
immutable `HardwareImplementation`. `InvocationManifest`, not the output
Artifact, owns both derivation records. Representative later derivations are:

* RTL elaboration or synthesis to a gate-level implementation;
* FPGA synthesis and implementation;
* physical placement, routing, extraction, or stream-out; and
* explicit insertion or transformation of implementation buffers/state.

ASIC and FPGA use the same immutable implementation family. The first FPGA
contract produces a static full-device implementation and image. Partial
reconfiguration, DFT, ATPG, multi-power-state intent, retention, fault
injection, and silicon bringup are explicitly deferred rather than represented
by empty fields or generic tool options.

Each output is complete under its own exact dependencies and payload closure;
it records no parent implementation or generator binding. The manifest edge
records the exact typed inputs and resolved binding. A purely mechanical
transformation uses `MechanicalDerivation`; a search choice uses
`CandidateDecision`. Multiple paths to identical canonical output state
converge on one HardwareImplementation identity. Logs and QoR observations do
not become fields of `HardwareImplementation`.

A generator importer processes one valid declared-output snapshot in this
order: interpret provider outputs into canonical logical payload bytes; publish
those bytes through BlobStore; construct the exact typed representation root,
format ref, top, interfaces, activity, macro, and external-binding closure;
finalize the HardwareImplementation; strict-reimport it through the same owner
and representation-format validators; and only then return its descriptor
output binding plus lineage contribution. A failure after BlobStore insertion
may leave deduplicated unreferenced blobs, but no partial root or output binding
is published.

A downstream generator materializes production bytes only from the exact input
HardwareImplementation representation root and its BlobStore-verified
BlobDigests. The representation root supplies the exact top object. Bundle
paths, tool database directories, filenames, and reports cannot replace either
authority.

For a Fabric-to-RTL generator, an implementation-only recipe selection is a
typed occurrence-scoped entry in the resolved generator configuration:

```text
FabricPhysicalOccurrenceOwnerRef -> BackendRecipeKey
```

`BackendRecipeKey` is a closed typed value owned by that candidate-generator
descriptor's resolved configuration schema, not a global string registry or a
Fabric attribute. The referenced RTL provider declares recipe availability and
external dependencies without redefining the Fabric capability.

The recipe may change gate structure or another implementation detail only
when it preserves the exact Fabric-observable semantics, timing, capacity,
progress, and `ConfigurationABI`. The selected map belongs to resolved
generator configuration and the manifest derivation edge. Its materialized
RTL, black-box contracts, platform reference, and external bindings contribute
to HardwareImplementation identity, but the map itself does not become another
lineage field. Numeric policy, supported actor domains, latency, initiation
interval, buffering, or other Fabric-visible choices must already be
represented by Fabric or by a Fabric-declared Mapping refinement; the
generator cannot reclassify them as recipes.

## Evaluation

Lint, formal checks, RTL workload execution, timing, area, power, DRC, and other
observations are ordinary Evaluations over exact immutable subjects. They
produce `EvaluationEvidence`. Workload-running RTL simulation also produces
`SimulationExecution`. Raw products remain owner-attempt or scratch material
and have no current Artifact schema.

Mapped RTL and gate-netlist execution use the ordinary Spatial
`SimulationExecution` authoring and finalization API. The exact Evaluation
model descriptor selects the HDL engine and the exact Request selects the
workload; an adapter does not add an external-HDL execution kind.

The initial mapped RTL provider is Evaluation model kind 21,
`mapped_rtl_simulator`, over case kind 12. Its exact subjects are one
HardwareImplementation and one Deployment whose Spatial Launch relation
selects that implementation, one SpatialMapping context, and the complete
configuration-image set for the exact Spatial workload. Its resolved config
view owns the stable HDL simulator build identity; executable paths, module
activation, scratch location, and wall-time limits remain invocation bindings.
The provider accepts an `Rtl` representation root and publishes the ordinary
Spatial `SimulationExecution` plus exact integral CycleCount. Gate-netlist
execution requires another exact model descriptor rather than an implicit
fidelity switch or fallback under kind 21.

Rail analysis reports the provider-neutral whole-case
`MaximumVoltageDrop` MetricKind in volts. Voltus and any other static or
dynamic rail provider normalize their native node observations to that same
metric. The exact model descriptor owns analysis method, activity basis,
network coverage, and uncertainty, and consumes or requires the applicable
HardwareImplementation-anchored `SupplyVoltage` base conditions. Tool names,
report severity classes, selected nodes, and private voltage-drop fields do
not become MetricKinds.

The initial Voltus descriptor is the exact static, explicit-assumption model
registered as Evaluation model kind 12. Evaluation projects one
`CompleteRailAnalysisConfiguration` from its descriptor-owned config view and
the already validated Request. The projection contains the exact process
corner, global applied supply, global temperature, global required clock
period, explicit activity assumption, static method, complete-network coverage,
and `ExactWithinModel` uncertainty. Temperature is a distinct typed condition;
it cannot be inferred from the technology corner, PGV contents, or a tool
default. The period targets the explicit assumption's sole global clock; its
absolute frequency cannot be inferred from SDC bytes, PGV contents, or a tool
default. Voltus consumes that projection together with the exact routed
HardwareImplementation, ImplementationPlatform binding, and complete PGV file
tree. It does not accept caller-authored Tcl values for any projected fact.
The `loom.eda.cadence.voltus.rail@1` implementation semantic identity fixes
Voltus high-definition rail accuracy and the adapter emits that mode
explicitly. Accuracy is neither an ambient tool default nor a second mutable
model input; changing the fixed provider algorithm requires another
implementation semantic identity.

The initial provider accepts only one routed `indexed_def_physical` root. That
root supplies exactly one self-contained DEF, its retained structural gate
netlist, and its generation constraints. Preparation parses the DEF rather
than a filename or producer record and requires exactly one routed special net
with `USE POWER`, exactly one routed special net with `USE GROUND`, and at
least one connected top-level `PIN` of the matching use for each net. These
facts mechanically select the sole rail domain and the voltage-source sites.
The global applied supply targets the power net relative to the ground net;
the ground voltage is exactly zero. A second power or ground net, an absent or
unrouted special net, an absent connected supply pin, multiple DEF payloads,
an opaque physical database, or an incomplete retained logical closure is
typed `Unsupported`. The adapter does not guess conventional net names, omit a
network fragment, or synthesize source locations.

Static power preparation reads the retained netlist and constraints, applies
the exact Request period, transition density, and static probability as the
single global vectorless activity assumption, and produces the current data
consumed by static rail analysis. DEF supplies physical connectivity; the
exact PGV tree supplies provider cell and technology models. The normalized
result is the maximum delivered-voltage deficit over both nets in the sole
complete domain, not the first report row or a provider severity threshold.

The model's resolved config view additionally carries the exact stable Voltus
provider build identity, canonical PGV member path/fingerprint table, and
ordered PGV entrypoint paths from the typed ResolvedConfig 6.0 binding. These
facts enter Request identity. Every entrypoint references the same member table;
the first is the provider-required technology PGV and the remaining order is
consumed unchanged. Preparation requires the resolved executable build to match
that identity and uses `resolveExternalFileTrees` to select one local tree with
exactly those members. A local path, directory key, filename-based root or
technology inference, first-match scan, or nearby PGV tree is not a semantic
substitute.

This initial model supports only one global applied supply, one global
temperature, and one global activity clock with one exact required period. A
provider must return typed `Unsupported` for a physical implementation that
cannot be represented by that exact contract. It cannot choose one domain,
substitute a Fabric nominal voltage or temperature, omit uncovered power nodes,
or infer activity from an ambient report. A later multi-domain or dynamic
provider uses another exact model descriptor while retaining the same
provider-neutral MetricKind.

FPGA prototype or measured-hardware execution uses the same
EvaluationRequest, SimulationWorkload, SimulationRuntimeInput, and Evidence
owners when a concrete evaluator is available. Loom does not add a measured
result family, signoff boolean, benchmark report authority, or implicit
promotion based on a tool's exit status.

An invocation that derives and evaluates hardware first finalizes the new
implementation, then issues Evaluation against that exact identity. It must not
attach observations to the input design or mutate an existing implementation.

## Repository Disclosure Boundary

Artifact persistence and repository publication are different decisions. A
valid semantic Artifact may remain confined to a machine-local store. Every
direct EDA attempt and every result derived directly from that attempt is
local-only and must not be committed to this public repository, even when the
result has been normalized into a valid semantic Artifact.

The local-only training-corpus class includes:

* concrete invocation bundles, materialized inputs, generated per-run scripts,
  manifests, completion records, and attempt metadata;
* tool stdout and stderr, logs, reports, databases, checkpoints, waveforms,
  netlists, parasitics, layouts, images, and other declared outputs;
* EDA-derived `HardwareImplementation`, `SimulationExecution`, and
  `EvaluationEvidence` roots and their reachable payloads; and
* training, validation, or held-out collections, invocation manifests, sample
  rows, and calibration results whose source is direct EDA Evidence.

This prohibition applies to open-source and commercial tool attempts alike, so
the repository has one disclosure rule rather than a vendor-dependent matrix.
Provider drivers, parsers, schemas, and deterministic generators remain normal
source. Small fixtures may be tracked only when they are authored synthetic
data, contain no captured tool output or proprietary payload, and assert a
stable semantic contract instead of vendor report wording.

No EDA-derived data product is eligible for repository publication. This
includes predictive or analytical `ModelParameterBundle` roots and payloads
trained, calibrated, or otherwise derived from direct EDA Evidence. Selection
records, bindings, and provenance produced by an EDA attempt remain local even
when they select independently authored parameters. The derived bundles remain
valid machine-local semantic Artifacts, but repository disclosure does not
create a sanitized Evidence format, public dataset Artifact, public weight
projection, or second model-weight representation.

All repository-local EDA material follows the ignored-root contract in
[External Tool Invocation](spec-external-tool-invocation.md). A path outside
the repository may be selected instead. Neither choice changes implementation,
Evidence, or model identity.

## External Artifacts

Backend-native products such as reports, logs, waveforms, netlists, extracted
parasitics, databases, and bitfiles are stored in owner-attempt or scratch
storage, or in the owning `HardwareImplementation` when they represent semantic
hardware state. Every other product remains raw attempt or scratch material;
no current raw-bundle Artifact is implied. `EvaluationEvidence` 1.0 does not
refer to such material. Normalized metrics and findings remain owned only by
Evidence.

High-cost products use the local output placement contract in External Tool
Invocation. A caller-selected path outside the worktree or the resolved user
data location is valid; repository-local work uses the one canonical ignored
root. Public specs and portable manifests never require private installation
paths, credentials, license data, user names, or host names.

## Target And External Inputs

ImplementationPlatform identifies the shared ASIC technology release or exact
FPGA ordering code. It does not contain standard-cell, SRAM, IO, RC, rule,
timing, power, primitive, pin, or user-IP files. Every generator or evaluator
descriptor declares the exact typed external input slots it consumes.

An explicit ordinary file contributes its exact SHA-256 fingerprint to the
resolved semantic binding. A resource distributed with a tool contributes the
stable provider build identity and exact resource key instead. Provider slot
compatibility and role are descriptor-owned; there is no global library-role
property bag, filename inference, implicit directory scan, or fallback to a
nearby input. A flow that needs several files declares several slots.

The same platform may intentionally feed different provider representations of
one corner, such as Liberty text for one flow and a vendor database for another.
The platform-owned `TechnologyCornerRef` is the common semantic corner; each
resolved provider binding owns its exact mapping to consumed models. Evidence
therefore records one target/corner question and one exact model binding rather
than pretending that a particular file format defines the corner.

A process corner condition uses an exact ImplementationPlatform-owned
`TechnologyCornerRef` into `loom.implementation_platform 1.0`; it is not a free
corner string. The ImplementationPlatform codec and validator resolve that
typed local reference before an adapter runs. Evaluation stores only the
owner-framed local-reference bytes and exact platform identity, never an
Evaluation-owned corner ordinal or property map. Voltage, temperature,
required clock period, and relative-clock schedule use the canonical typed
payloads in `docs/spec-evaluation-metrics.md`. An EDA adapter maps those facts
to tool syntax through its exact model binding without redefining them.

Local resolution from a logical reference to files is an execution concern and
must not create a second target, corner, or library-profile authority. Paths
and local-file keys occur only in explicit local configuration and the ignored
bundle. The bundle verifies that the selected local file still realizes the
fingerprint frozen by the semantic binding. Tool-bundled resources are verified
by provider-owned build and resource probes rather than whole-tree hashing.

## Failure And Completion

Execution distinguishes at least:

* descriptor or capability mismatch;
* missing semantic model input;
* unavailable executable or activation failure;
* license or permission unavailability;
* timeout or cancellation;
* tool execution failure;
* missing declared output;
* parser/normalizer failure; and
* completed Evaluation with adverse typed findings.

The bundle-owned completion record begins only when the finalized bundle's
script is attempted. It distinguishes launch or activation failure, tool
execution, declared-output failure, and successful driver completion. Bundle
preparation and descriptor-owned import return their own typed API errors and
do not create or mutate the completion record. An interrupted script without a
valid atomic completion record is an incomplete attempt, not an
`ExecutionFailed` Evidence value.

Infrastructure failures and execution limits do not select a different formal
candidate. Timeout or cancellation maps to `CancelledOrTimeout`; tool or
adapter failure maps to `ExecutionFailed`. Resource unavailability may leave
attempt and controller state incomplete, but cannot create partial Evidence.
No tool adapter silently falls back to another model.

## Calibration And Online Learning

High-fidelity Evidence may be admitted as training or calibration input for a
faster model through the central DSE plan. An ordinary typed `Generate` node
creates immutable `loom.model_parameter_bundle 1.0` candidates from exact
Training Evidence, trainer configuration, and seed; Validation and HeldOut
Evidence are additional typed admission inputs used only to prove pairwise
sample-group isolation before fitting. It does not create a separate
training-request Artifact. Each bundle references one registry-owned parameter
contract shared by the predictor and calibration validator. Ordinary
validation and held-out promotion bind the candidate as the bundle subject of
the `fpa_model_parameter_calibration` case and bind the exact ground-truth
Evidence collection from the corresponding typed Promote input. Validation may
rank; HeldOut may only feed the terminal model-release gate. A selected bundle
enters a new resolved predictor binding only after its contract matches the
predictor slot. Training provenance remains in `InvocationManifest`, and
online updates never mutate a bundle or model binding used by an in-flight
deterministic invocation.

The initial FPA contract admits only exact Evaluation model kind 20,
`openroad_routed_static_fpa`, as its ground-truth target. Its target key fixes
the OpenROAD provider build, normalization contract, and fidelity. Routed-flow,
library/platform, Fabric, and operating-condition facts remain typed model
features. Evidence from another EDA provider or implementation fidelity remains
valid for its own descriptor but cannot enter that parameter bundle. Supporting
it requires another exact ground-truth model and parameter contract, or an
explicit multi-source contract whose features own source identity.

Ground-truth collection uses the central finite DSE plan and Journal. Every
model-data sample's complete newly required dependency slice is limited to ten
active minutes. The FPA entry point rejects a campaign policy above four
active hours and propagates the remaining budget as the same absolute dispatch
deadline used by the plan executor. Uncached Mapping, RTL, implementation, and
EDA prerequisites remain visible and charged in that plan. A timeout,
incomplete attempt, or typed Unsupported outcome produces no training sample.

## Anchor Verification

Stable tests cover semantic versus invocation binding, exact manifest
derivation inputs, occurrence-scoped recipe selection, owner-specific
admission before each fact is consumed, exact representation-root and
BlobStore input materialization, separate prepare/import,
derivation-before-evaluation, implementation-state convergence,
capability-obligation resolution without an ecosystem mode, output collection,
typed failure classification, canonical parameter payload validation before
Blob Store publication, producer/consumer parameter-contract matching, typed
validation/held-out subject binding, pairwise sample-group isolation before
training, exact provider/fidelity target-key matching, campaign execution-limit
admission, held-out exclusion from ranking, and repository-local output ignore
coverage. An executable without its driver/importer contract is not admitted
as provider support. Vendor command lines, local module names, licenses, and
report text are adapter behavior, not captured repository fixtures or global
provider-by-platform-by-recipe fixture matrices.
