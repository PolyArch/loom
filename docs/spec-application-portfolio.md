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
vecadd-memory
```

The two Gitlink-backed rows consume pinned upstream source packages. Their
Gitlink entries own the exact upstream revisions; the application manifest
references the source roots but never copies a commit hash or version alias.
The multisensor attention application is Loom-owned and reifies the complete
`project`/`attention`/`stats` workload used by the heterogeneous system
conformance anchor. The repository-owned `vecadd-memory` row is the regular
contiguous-memory witness paired with the irregular PageRank row.

The repository-owned `mlperf-tiny-anomaly-detection` runner consumes the exact
int8 anomaly-detection model and DCASE feature dataset from the pinned
`mlperf-tiny` Gitlink. It validates the model's ten-layer fully connected
TFLite topology, executes every quantized layer, and exposes one warm-up plus
four measured samples under a ten-second deadline. Its exact oracle records
all 2,560 measured output bytes independently reproduced with both the
`ai-edge-litert` 2.2.0 `BUILTIN_REF` kernel and its default XNNPACK delegate,
using one thread. The host runner uses real-valued requantization compatible
with those reference semantics; it does not claim bit equivalence to optimized
fixed-point builtin or TFLite Micro kernels, or complete MLPerf anomaly MSE
reporting. This is bounded host inference with admitted source, model, dataset,
profile, and oracle provenance. It is not a TFLite product frontend and does
not establish canonical Simulation or Evaluation Evidence.

## Manifest Contract

Each manifest row owns only the repository-level selection needed to run one
application:

* its stable application identity and source-package root;
* the exact build entry, language mode, source selection, and compiler and link
  options;
* named workload and runtime-input selections;
* the independent oracle or typed invariant bound to each selection;
* the bounded warm-up, measured-sample, oracle-coverage, and execution-deadline
  profile bound to each input; and
* membership in the `smoke`, `validation`, and `scale_eda` execution
  selections.

The tracked JSON contract is schema `loom.application_portfolio` version
`2.0`. Its exact structural shape is:

```text
{
  "schema": "loom.application_portfolio",
  "version": "2.0",
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
      "oracle": {"kind": "exact" | "typed_invariant", "entry": <repo path>},
      "profile": {
        "warmup_samples": <unsigned integer>,
        "measured_samples": <positive unsigned integer>,
        "oracle_coverage": "all_measured_samples",
        "deadline_milliseconds": <positive unsigned integer>
      }
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

The profile owns no duplicated total or oracle sample count. Its exact input
budget is derived as `warmup_samples + measured_samples`; the sum must fit in
an unsigned 64-bit integer. Warm-up samples establish execution state but do
not contribute correctness or performance observations. `measured_samples`
and `deadline_milliseconds` must both be nonzero. The only admitted
`oracle_coverage` is `all_measured_samples`, so every measured sample is gated
by the selected oracle while no warm-up sample is misreported as evidence.

Source admission resolves a Gitlink only from its mode `160000` repository
index entry, requires the checkout `HEAD` to equal that entry, and verifies
that selected translation units are tracked and unchanged at that commit.
Repository sources, selected translation units, and oracle entries must exist
without escaping their admitted roots. An oracle entry cannot be a selected
program translation unit. Cache bytes must match their declared SHA-256.
Missing Gitlink checkout or cache content is typed unavailable; a wrong mode,
revision mismatch, modified selected source, path escape, or digest mismatch
is invalid. Admission never initializes a submodule or substitutes content.

The selected-input admission entry point validates only the named input's
oracle and cached-input references. The multi-application admission entry
point retains its all-input behavior for corpus-level validation.

The public product compiler accepts one exact portfolio input through the
co-required manifest, repository-root, application-identity, and input-name
selectors. A cache root is optional unless the selected input references
cached content. In this mode the admitted manifest row is the sole owner of
the source list and compiler/link options; additional user compiler inputs are
invalid. The driver resolves and admits the row before compilation, derives
absolute selected source paths, and retains the same resolved selection in its
in-process compile-to-Deployment invocation. The standalone final-link replay
helper cannot attach a portfolio selection because it cannot prove that the
input link was produced from that selection.

The current product source binding admits exactly zero warm-up samples and one
measured sample. A profile with any other sample count is typed unsupported
until an application runner executes the declared counts; the product path
does not silently reinterpret or ignore a larger profile. Consequently, the
TinyML member's one-warm-up/four-measured profile is directly executable by
its bounded host runner but returns `loom_portfolio_profile_unsupported` from
the current product source-binding path.

The pair decision projects the resolved application identity, input name,
source/build selection, declared workload and runtime-input names, declared
oracle and bounded profile, and referenced cache digests. Until a portfolio
runner binds those declarations to canonical Simulation inputs and oracle
results, the projection publishes `execution_binding_established: false`; it
must not be interpreted as sample or correctness Evidence. The manifest and
repository paths are operational inputs and never enter pair, candidate,
Mapping, workload, or runtime-input identity. The canonical source program,
workload, runtime input, Fabric, Mapping, and execution Evidence remain owned
by their existing Artifacts; the portfolio projection is repository
provenance, not a second copy of those payloads.

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

Stable tests currently validate manifest schema and uniqueness, source-root
and Gitlink resolution, selected-input cache and oracle admission, bounded
profile parsing, deterministic smoke inventory, native host output for the
five admitted rows, byte-exact bounded TinyML inference under its declared
deadline, product-driver argument projection, and rejection of partial,
injected, replayed, target-conflicting, or unsupported-profile selections.

Portfolio closure additionally requires a runner that binds every selected
profile to canonical Simulation workload/runtime-input roots, enforces its
warm-up, measured-sample, oracle-coverage, and execution-deadline contract,
and publishes the resulting correctness and performance Evidence. Release
anchors must then prove member-local acceleration gates, reject aggregate
masking, cover the supported execution matrix, require every released AccCore
occurrence to have a selected SystemMapping user, and cover the declared
single-application, domain, and complete cross-domain sets. Those gates do not
exist until their owning runtime and evidence paths publish the required
typed outcomes.
