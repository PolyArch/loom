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

The current portfolio has exactly these stable application identities:

```text
gapbs-pagerank
llama2c-kernels
loom-multisensor-attention
mlperf-tiny-anomaly-detection
mlperf-tiny-keyword-spotting
mlperf-tiny-visual-wake-words
```

The five Gitlink-backed rows consume pinned upstream source packages. Their
Gitlink entries own the exact upstream revisions; the application manifest
references the source roots but never copies a commit hash or version alias.
The multisensor attention application is Loom-owned and reifies the complete
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

The tracked JSON contract is schema `loom.application_portfolio` version
`1.0`. Its exact structural shape is:

```text
{
  "schema": "loom.application_portfolio",
  "version": "1.0",
  "applications": [{
    "identity": <stable logical name>,
    "source": {"kind": "gitlink" | "repository", "root": <repo path>},
    "build": {
      "entry": <selected C/C++ translation unit>,
      "language": "c" | "c++",
      "sources": [<source-relative translation units>],
      "compiler_options": [<argument>],
      "link_options": [<argument>]
    },
    "cached_inputs": [
      {"logical_name": <name>, "path": <cache path>, "sha256": <digest>}
    ],
    "inputs": [{
      "name": <name>,
      "workload": <logical workload selection>,
      "runtime_input": <logical runtime-input selection>,
      "cached_inputs": [<cached logical name>],
      "oracle": {"kind": "exact" | "typed_invariant", "entry": <repo path>}
    }],
    "selections": ["smoke" | "validation" | "scale_eda"]
  }]
}
```

Applications, source selections, cached inputs, named inputs, cache
references, and execution selections are strictly ordered and unique.
Compiler and link option order remains semantic and is preserved. All paths
are normalized visible-ASCII relative paths; stable logical names use
lowercase ASCII letters, digits, `.`, `_`, or `-`. Execution-selection order
is `smoke`, `validation`, then `scale_eda`. The build entry is one member of
the exact source selection. Every cached declaration is referenced by a named
input.
Unknown fields are invalid, so a Gitlink row cannot copy a revision, version
alias, tolerance, or untyped property into the manifest. Workload and
runtime-input names are repository selections for their existing owners, not
new Artifact identities.

Source admission resolves a Gitlink only from its mode `160000` repository
index entry, requires the checkout `HEAD` to equal that entry, and verifies
that selected translation units are tracked and unchanged at that commit.
Repository sources, selected translation units, and oracle entries must exist
without escaping their admitted roots. An oracle entry cannot be a selected
program translation unit. Cache bytes must match their declared SHA-256.
Missing Gitlink checkout or cache content is typed unavailable; a wrong mode,
revision mismatch, modified selected source, path escape, or digest mismatch
is invalid. Admission never initializes a submodule or substitutes content.

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

* `EvaluationEvidence` owns normalized registered correctness, performance,
  and physical-quality observations;
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

## Optimization Scope

An optimization scope is the exact canonical set of selected manifest rows,
named workload inputs, and runtime inputs bound as ordinary DSE plan inputs. It
is not an Artifact, mode enum, target class, or mutable benchmark suite. A set
containing one application naturally supports an application-specific design;
a set selected from one coherent application domain supports a domain-specific
design; the declared complete supported portfolio spanning application domains
supports a general design. Those descriptions are human projections of the
exact selected roots and never change candidate, Mapping, or Evidence identity.

Every selected member independently passes its source-backed correctness
oracle before contributing performance or physical quality. A release policy
also owns an explicit per-member acceleration or typed-support gate; an
aggregate mean, Pareto point, or favorable member cannot hide an unmapped,
unsupported, incorrect, or regressed selected workload. Aggregate objectives
rank only candidates that have already satisfied those member-local gates.

Hardware optimization produces a complete `fabric.system`, not an unrelated
set of SpatialCore Modules. For every released System, each selected AccCore
occurrence must be the physical target of at least one accepted portfolio
SystemMapping. This proves that occurrence inventory participates in the
selected workload set instead of rewarding unused hardware. Several
occurrences may share one Module identity, but their occurrence-qualified use,
resources, and cost multiplicity remain distinct.

Training, Validation, and HeldOut partitions for parameter calibration remain
orthogonal to optimization scope and to the `smoke`, `validation`, and
`scale_eda` execution selections. A workload may participate in a release
scope without entering model fitting, and a calibration sample does not become
an application conformance result merely because it used the same source.

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
and rejection of duplicated revisions or alternate runner inventories. They
also prove member-local correctness and acceleration gates, reject aggregate
masking, and require every released AccCore occurrence to have at least one
selected SystemMapping user. At least one anchor traverses every cell of the
Spatial-only/System-with-gem5 by DFG/CGRA/RTL execution matrix. Reproducible
release anchors cover one exact single-application set, one exact domain set,
and the exact declared complete supported cross-domain set without introducing
scope identities. Tests do not snapshot reports, require private EDA material,
or duplicate application semantics in the harness.
