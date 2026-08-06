# Real Application Portfolio

This document owns Loom's repository conformance portfolio of complete,
multi-operation, multi-stage applications. It does not redefine source
language semantics, compiler Artifacts, Simulation workloads, Evaluation
Evidence, or external project revisions.

## Purpose And Authority

The real-application portfolio validates composition across the complete Loom
stack after operator-level contracts have been established. Its sole membership
and execution-selection authority is:

```text
test/applications/manifest.json
```

Directory enumeration, CI job lists, generated reports, dashboards, and
hard-coded application lists in runners are not alternate inventories. The
manifest is a thin repository conformance input, not an Artifact and not a
product program format. Loom's public source boundary remains C and C++ through
LLVM IR; the portfolio does not add ONNX, TFLite, framework graphs, or another
frontend product boundary.

The initial portfolio has exactly these stable application identities:

```text
mlperf-tiny-streaming-wakeword
mlperf-tiny-visual-wake-words
llama2-c-tinystories-260k
gapbs-pagerank
loom-multisensor-attention
```

The first four consume pinned upstream source packages. Their Gitlink entries
own the exact upstream revisions; the application manifest references the
source roots but never copies a commit hash or version alias. The multisensor
attention application is Loom-owned and reifies the complete
`project`/`attention`/`stats` workload used by the heterogeneous system
conformance anchor.

## Manifest Contract

Each manifest row owns only the repository-level selection needed to run one
application:

* its stable application identity and source-package root;
* the exact build entry, language mode, source selection, and compiler and link
  options;
* named workload and runtime-input selections;
* the independent oracle or typed invariant bound to each selection; and
* membership in the `smoke`, `validation`, and `scale_eda` execution
  selections.

The referenced source package owns program sources and build semantics. Existing
Loom owners produce the linked LLVM module, Structured Program Candidate,
Canonical Dataflow Program, Mapping, Deployment, HardwareImplementation,
SimulationWorkload, SimulationRuntimeInput, EvaluationRequest, and
EvaluationEvidence. The manifest does not copy any of those payloads or define
an `ApplicationArtifact`.

One application identity may have several named input selections. Those
selections change exact workload or runtime-input identity, not application
membership. The three execution selections are scheduling and conformance
policy over the same inventory:

* `smoke` is the bounded, deterministic developer gate;
* `validation` exercises representative functional and quality behavior; and
* `scale_eda` selects long-running scale, RTL, physical implementation, and
  EDA work where its required providers are available.

These names do not define training, validation, or held-out data roles.
`CalibrationPartitionRole` remains owned only by the model-training contract.
A runner may select any canonical subset explicitly, but it cannot publish a
different membership inventory or weaken the selected row's oracle.

## Inputs And Static Data

Fixed program data, including model weights compiled into the linked program,
is lowered through the existing executable-closure contract and becomes exact
`StaticMemoryImageLeaf` content when deployment requires it. Runtime samples,
graph inputs, sensor streams, and other per-run values become exact
`SimulationRuntimeInput` content. A manifest path is never the semantic identity
of either form.

Large weights and datasets may live in an ignored or user-owned cache. A
manifest row binds the expected digest and logical selection; import verifies
the bytes before constructing the existing owning object. Missing or mismatched
content is an explicit unavailable or invalid input, never a substitute dataset
or a skipped pass. Proprietary input, direct EDA output, and other restricted
material must not enter Git.

## Correctness And Numerical Accuracy

Every selected execution has one independent correctness authority. A fully
deterministic observation may use exact expected values. An application whose
contract admits nondeterminism or bounded numerical approximation uses a
descriptor-owned typed invariant or oracle instead. A free-form tolerance in a
runner or report is not a correctness contract.

When selected special-math actors admit a non-correctly-rounded result, each
execution engine must independently satisfy the same application oracle or
invariant. Pairwise bit equality between DFG, CGRA, RTL, gem5-backed, or native
execution is required only when the exact observable contract proves a unique
deterministic value. Agreement between two implementations does not replace an
independent oracle.

Correctness gates precede performance and quality comparison. Every selected
case uses the ordinary typed completion, unsupported, incomplete, or failure
outcome owned by its producer. Aggregate reports preserve those disjoint
outcomes and cannot hide a failure behind a mean, pass rate, or best case.

## Evidence And Improvement Loop

Application execution produces only existing semantic records:

* `EvaluationEvidence` owns normalized correctness, performance, energy, and
  physical-quality observations;
* `InvocationManifest` owns derivation lineage, deterministic work accounting,
  and nonsemantic operational observations; and
* `ModelParameterBundle` owns immutable derived model parameters.

Human-readable summaries and dashboards are removable projections. There is no
mutable latest-best record. Every comparison or promotion names an exact
baseline and exact candidate Evidence. Model training consumes explicit
Training, Validation, and HeldOut Evidence sets through the central DSE
contract; a held-out release gate must pass before an updated parameter bundle
is promoted.

Raw longitudinal measurements, direct EDA Evidence, reports, databases,
waveforms, bitfiles, and training corpora remain in ignored or user-owned
storage. A publishable `ModelParameterBundle` may enter Git only under its
existing disclosure contract and never carries source samples or attempt
material.

## Portfolio Admission

A later application may enter the portfolio only when it:

1. is a complete linked C or C++ program with meaningful multi-operation or
   multi-stage behavior;
2. has deterministic source/build selection and exact input identities;
3. supplies an independent exact oracle or typed invariant for every selected
   execution;
4. exercises a stack behavior not already represented adequately by the
   existing portfolio; and
5. uses ordinary Loom Artifacts, Evidence, configuration, and failure outcomes
   without application-specific compiler or backend semantics.

Changing membership or an oracle is a reviewed semantic change to the manifest
and this portfolio contract. Adding another input selection or changing a gate
selection is a reviewed conformance-policy change. Neither is inferred from
which files happen to exist.

## Anchor Verification

Stable tests validate manifest schema and uniqueness, source-root and Gitlink
resolution, digest verification for cached inputs, deterministic inventory and
selection derivation, exact binding to existing workload/input/oracle owners,
and rejection of duplicated revisions or alternate runner inventories. At
least one anchor traverses every cell of the Spatial-only/System-with-gem5 by
DFG/CGRA/RTL execution matrix. Tests do not snapshot reports, require private
EDA material, or duplicate application semantics in the harness.
