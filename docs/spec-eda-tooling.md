# EDA Tooling

This document defines the portable boundary between Loom and external RTL,
FPGA, ASIC, formal, and physical-design tools. Process execution is owned by
[Evaluation ToolRunner](spec-evaluation-tool-runner.md); Evaluation schemas are
owned by [DSE and Evaluation](spec-dse-feedback.md).
Persistent implementation state is owned by
[Hardware Implementation](spec-hardware-implementation.md), and immutable ASIC
or FPGA technology inputs are owned by
[Implementation Platform](spec-implementation-platform.md).

## Tool Descriptors And Bindings

An EDA `EvaluationModelDescriptor` declares an immutable observation
capability:

* required Artifact schemas;
* modeled phenomena and supported metrics/findings;
* required technology or platform input slots;
* full-execution method and determinism contract;
* parser/adapter semantic identity.

One resolved model binding supplies exact result-affecting inputs such as tool
version, standard-cell or FPGA platform data, corner, constraints, parser
version, and semantic effort. Executable paths, module activation, license
servers, host selection, scratch roots, and parallel-job limits are invocation
bindings. They do not enter candidate semantics unless the model descriptor
explicitly declares a result-affecting value.

Loom selects by descriptor capability and exact model binding, not by a global
fidelity level or hard-coded workstation tool name.

An adapter consumes the exact `GenerationConstraint` payload of its
HardwareImplementation. It may translate that payload into vendor syntax, but
it cannot infer a hidden clock, reset exception, CDC waiver, false path,
multicycle path, floorplan rule, or timing target from report text or local
defaults.

A flow that creates a new `HardwareImplementation` is instead selected through
the central `CandidateGeneratorDescriptor` and
`ResolvedCandidateGeneratorBinding`. One physical tool invocation may derive
an implementation and retain raw reports for a following Evaluation, but it
does not make EvaluationModelDescriptor an implementation generator.

## HardwareImplementation Derivation

The initial RTL implementation is a `MechanicalDerivation` from exact Fabric,
exact `ConfigurationABI`, and the resolved generator binding. It has no parent
implementation. A later flow that consumes existing hardware state and
preserves new state creates another immutable `HardwareImplementation`.
Representative later derivations are:

* RTL elaboration or synthesis to a gate-level implementation;
* FPGA synthesis and implementation;
* physical placement, routing, extraction, or stream-out; and
* explicit insertion or transformation of implementation buffers/state.

ASIC and FPGA use the same immutable lineage. The first FPGA contract produces
a static full-device implementation and image. Partial reconfiguration, DFT,
ATPG, multi-power-state intent, retention, fault injection, and silicon bringup
are explicitly deferred rather than represented by empty fields or generic
tool options.

Each later output records the exact parent implementation and
implementation-defining generator binding. A purely mechanical transformation
uses `MechanicalDerivation`; a search choice uses `CandidateDecision`. Logs and
QoR observations do not become fields of `HardwareImplementation`.

For a Fabric-to-RTL generator, an implementation-only recipe selection is a
typed occurrence-scoped entry in the resolved generator configuration:

```text
FabricEntityRef -> BackendRecipeKey
```

`BackendRecipeKey` is a closed typed value owned by that candidate-generator
descriptor's resolved configuration schema, not a global string registry or a
Fabric attribute. The referenced RTL provider declares recipe availability and
external dependencies without redefining the Fabric capability.

The recipe may change gate structure or another implementation detail only
when it preserves the exact Fabric-observable semantics, timing, capacity,
progress, and `ConfigurationABI`. The selected map therefore contributes to
`HardwareImplementation` lineage and identity but not to Fabric identity.
Numeric policy, supported actor domains, latency, initiation interval,
buffering, or other Fabric-visible choices must already be represented by
Fabric or by a Fabric-declared Mapping refinement; the generator cannot
reclassify them as recipes.

## Evaluation

Lint, formal checks, RTL workload execution, timing, area, power, DRC, and other
observations are ordinary Evaluations over exact immutable subjects. They
produce `EvaluationEvidence` and raw detailed bundles. Workload-running RTL
simulation also produces `SimulationExecution`.

FPGA prototype or measured-hardware execution uses the same
EvaluationRequest, SimulationWorkload, SimulationRuntimeInput, and Evidence
owners when a concrete evaluator is available. Loom does not add a measured
result family, signoff boolean, benchmark report authority, or implicit
promotion based on a tool's exit status.

An invocation that derives and evaluates hardware first finalizes the new
implementation, then issues Evaluation against that exact identity. It must not
attach observations to the parent design or mutate an existing implementation.

## External Artifacts

Backend-native products such as reports, logs, waveforms, netlists, extracted
parasitics, databases, and bitfiles are stored in content-addressed raw bundles
or in the owning `HardwareImplementation`, according to whether they represent
semantic hardware state. A bundle manifest contains payload digests, tool
products, and the exact `EvaluationRequest` reference. It never refers to an
Invocation or `EvaluationEvidence`; owner-specific attempt records retain
invocation provenance and may reference the bundle. Evidence may refer to the
bundle, giving an acyclic direction from raw execution material to normalized
observations. Normalized metrics and findings remain owned only by Evidence.

High-cost products use a caller-selected artifact root, with a resolved default
under Loom's user data area when no path is supplied. Public specs and portable
manifests never require private installation paths, credentials, license data,
user names, or host names.

## Library And Platform Inputs

Technology data is referenced through typed model-input slots. It may represent
standard cells, SRAMs, IO, RC data, FPGA devices, timing/power models, or other
platform facts. The exact immutable content or release identity is part of the
semantic model binding when it can affect results. The requested PVT corner,
required clock, and other ground-truth evaluation scenario facts are typed
`EvaluationCondition` values rather than model inputs.

A process corner condition uses an exact technology-family-owned
`TechnologyCornerRef`; it is not a free corner string. Voltage, temperature,
required clock period, and relative-clock schedule use the canonical typed
payloads in `docs/spec-evaluation-metrics.md`. An EDA adapter maps those facts
to tool syntax through its exact model binding without redefining them.

Local resolution from a logical reference to files is an execution concern and
must not create a second library-profile authority.

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

Infrastructure failures and execution limits do not select a different formal
candidate. Timeout or cancellation maps to `CancelledOrTimeout`; tool or
adapter failure maps to `ExecutionFailed`. Resource unavailability may leave
attempt and controller state incomplete, but cannot create partial Evidence.
No tool adapter silently falls back to another model.

## Calibration And Online Learning

High-fidelity Evidence may be admitted as training or calibration input for a
faster model through the central DSE plan. Training creates a new immutable
parameter bundle and resolved model binding. Online updates never mutate a
model used by an in-flight deterministic invocation.

## Anchor Verification

Stable tests cover semantic versus invocation binding, exact implementation
parentage, occurrence-scoped recipe identity, derivation-before-evaluation,
output collection, and typed failure classification. Vendor command lines,
local module names, licenses, and report text are adapter tests rather than
global fixture matrices.
