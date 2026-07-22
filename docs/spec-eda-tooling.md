# EDA Tooling

This document defines the portable boundary between Loom and external RTL,
FPGA, ASIC, formal, and physical-design tools. Process execution is owned by
[Evaluation ToolRunner](spec-evaluation-tool-runner.md); Evaluation schemas are
owned by [DSE and Evaluation](spec-dse-feedback.md).

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

Each later output records the exact parent implementation and
implementation-defining generator binding. A purely mechanical transformation
uses `MechanicalDerivation`; a search choice uses `CandidateDecision`. Logs and
QoR observations do not become fields of `HardwareImplementation`.

## Evaluation

Lint, formal checks, RTL workload execution, timing, area, power, DRC, and other
observations are ordinary Evaluations over exact immutable subjects. They
produce `EvaluationEvidence` and raw detailed bundles. Workload-running RTL
simulation also produces `SimulationExecution`.

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
parentage, derivation-before-evaluation, output collection, and typed failure
classification. Vendor command lines, local module names, licenses, and report
text are adapter tests rather than global fixture matrices.
