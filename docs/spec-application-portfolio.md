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
four measured samples for `smoke`, plus two warm-up and two measured samples
for `validation`, under a ten-second deadline. The exact smoke oracle records
all 2,560 measured output bytes independently reproduced with both the
`ai-edge-litert` 2.2.0 `BUILTIN_REF` kernel and its default XNNPACK delegate,
using one thread. The validation oracle covers the next two measured samples
after its longer warm-up. The host runner uses real-valued requantization
compatible with those reference semantics; it does not claim bit equivalence
to optimized fixed-point builtin or TFLite Micro kernels, or complete MLPerf
anomaly MSE reporting. This is bounded host inference with admitted source,
model, dataset, profile, and oracle provenance. It is not a TFLite product
frontend and does not establish canonical Simulation or Evaluation Evidence.

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
* the exact input rows selected by the `smoke`, `validation`, and `scale_eda`
  execution policies.

The tracked JSON contract is schema `loom.application_portfolio` version
`3.0`. Its exact structural shape is:

```text
{
  "schema": "loom.application_portfolio",
  "version": "3.0",
  "applications": [{
    "identity": <stable logical name>,
    "source": {"kind": "gitlink" | "repository", "root": <repo path>},
    "build": {
      "entry": <selected C/C++ translation unit>,
      "language": "c" | "c++",
      "sources": [<source-relative translation units>],
      "compiler_options": [<argument>],
      "link_options": [<argument>],
      "operator_protocol_symbols": [<linked symbol>]
    },
    "cached_inputs": [
      {"logical_name": <name>, "path": <cache path>, "sha256": <digest>}
    ],
    "inputs": [{
      "name": <name>,
      "workload": <logical workload selection>,
      "runtime_input": <logical runtime-input selection>,
      "cached_inputs": [<cached logical name>],
      "compiler_options": [<input-specific argument>],
      "oracle": {"kind": "exact" | "typed_invariant", "entry": <repo path>},
      "profile": {
        "warmup_samples": <unsigned integer>,
        "measured_samples": <positive unsigned integer>,
        "oracle_coverage": "all_measured_samples",
        "deadline_milliseconds": <positive unsigned integer>
      }
    }],
    "selection_inputs": {
      "smoke": [<input name>],
      "validation": [<input name>],
      "scale_eda": [<input name>]
    }
  }]
}
```

Applications, source selections, cached inputs, named inputs, cache
references, and input names within each execution selection are strictly
ordered and unique.
Compiler and link option order remains semantic. The exact selected compiler
sequence is derived by appending the input-specific options to the build
options. Host and product consumers use that one derived sequence; the
inventory additionally retains the input-specific subsequence so a tier's
compile-time input provenance is inspectable. A consumer either preserves the
option order or consumes a documented option through an existing semantic
owner. Operator-protocol symbols are ordered, unique linked entry symbols owned
by the build selection; their order retains the candidate preference previously
expressed by the product invocation. A portfolio invocation cannot supply a
competing CLI symbol list. All paths are normalized visible-ASCII relative
paths; stable logical names use lowercase ASCII letters, digits, `.`, `_`, or
`-`.
The optional execution-selection fields are interpreted in the fixed order
`smoke`, `validation`, then `scale_eda`. Every present field has a nonempty
exact input-name set, every name resolves within the same application, and
every input belongs to at least one execution selection. The build entry is
one member of the exact source selection. Every cached declaration is
referenced by a named input.
Unknown fields are invalid, so a Gitlink row cannot copy a revision, version
alias, tolerance, or untyped property into the manifest. Workload and
runtime-input names are repository selections for their existing owners, not
new Artifact identities.

The host projection passes the ordered link options to the native compiler.
The product projection has no host sysroot: the exact `-lm` option is consumed
as a dependency on Loom's existing freestanding math runtime and is not sent
to the initial RISC-V LLD invocation. Other `-l` options are invalid until the
product runtime owns their implementation. This interpretation preserves the
manifest as the build dependency owner without inventing an empty target
library or importing a host library into the product image.

The current manifest binds all five applications to real bounded `smoke` and
`validation` rows. `gapbs-pagerank`, `loom-multisensor-attention`, and
`vecadd-memory` also own exact `scale_eda` rows, covering irregular memory,
Attention, and regular contiguous memory respectively. Every declared tier
selects its own actual bounded input row, runtime-input name, and exact oracle.
For compiled fixtures, input-specific constants make the selected values and
memory footprint part of the derived source build. `selection_inputs` is the
only tier-to-row relation; a runner never infers it from an input name. Adding
a tier requires another real bounded input and oracle under the same contract.

The profile owns no duplicated total or oracle sample count. Its exact input
budget is derived as `warmup_samples + measured_samples`; the sum must fit in
an unsigned 64-bit integer. Warm-up samples establish execution state but do
not contribute correctness or performance observations. `measured_samples`
and `deadline_milliseconds` must both be nonzero. The only admitted
`oracle_coverage` is `all_measured_samples`, so every measured sample is gated
by the selected oracle while no warm-up sample is misreported as evidence.

Source admission resolves a Gitlink only from its mode `160000` repository
index entry, requires the checkout `HEAD` to equal that entry, and verifies
that selected translation units are tracked and unchanged at that commit. A
checkout is a source root carrying its own `.git` entry. A linked worktree
owns no Gitlink checkouts; admission resolves its Gitlinks under the primary
worktree that Git reports for the repository and validates that checkout
against the linked worktree's own index entry.
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

The pair decision and product build derive operator-protocol symbols from the
selected build. They do not accept an independent symbol list, so candidate
discovery and the portfolio report cannot silently name different kernels.

The current product source binding admits exactly zero warm-up samples and one
measured sample. A profile with any other sample count is typed unsupported
until an application runner executes the declared counts; the product path
does not silently reinterpret or ignore a larger profile. Consequently, the
TinyML member's smoke and validation profiles are directly executable by its
bounded host runner but each returns `loom_portfolio_profile_unsupported` from
the current product source-binding path.

The versioned `loom.application_pair_decision` projection records the resolved
application identity, input name, source/build selection, input-specific
compiler options, declared workload and runtime-input names, declared oracle
and bounded profile, and referenced cache digests. Its typed execution binding
is `declared_only`, `canonical_simulation`, or
`canonical_simulation_and_oracle`. The last state is reached only after the
existing Mapping runtime owner completes source-backed DFG and CGRA
Simulation and a native `SimulationComparison` reports no finding. Its exact
Evidence references are carried by the selected Mapping observation, and the
derived `execution_binding_established` compatibility field is true only in
that state. A pre-admission or unsupported profile decision remains
`declared_only` and cannot be interpreted as correctness Evidence.

A complete bounded front-end that retains no candidate because the exact
Fabric refused every representable ownership coordinate publishes
`exact_hardware_incompatible`, not a proof-not-established or generic
unsupported decision. Its detail names the refused actor and, for a memory
actor, the Dataflow memory contract class (`volatile`, `atomic_access`,
`atomic_rmw`, `compare_exchange`, or `fence`) that no Fabric capability admits.
The program keeps its verified host path and the host-only baseline stays
complete; no owner lowers such a contract to a plain access to obtain a
candidate. The product build reports the same refusal as the typed error
`loom_pre_mapping_exact_hardware_incompatible`. A rejection that comes from
the exact CGRA execution provider after Mapping carries the same contract
class through the selected Mapping's runtime disposition.

The manifest exact-output host report is an additional conformance gate, not
a substitute for that canonical Simulation binding. Manifest and repository
paths are operational inputs and never enter pair, candidate, Mapping,
workload, or runtime-input identity. The canonical source program, workload,
runtime input, Fabric, Mapping, and execution Evidence remain owned by their
existing Artifacts; the portfolio projection is repository provenance, not a
second copy of those payloads.

A successful decision is published inside
`loom.application_pair_evidence` version `1.0`. That envelope is the canonical
join of candidate and analytic-gate inventories, actual Tech/Spatial/System
work, selected Mapping checkpoint, failure-cone counters, work ledgers,
Mapping outcome inventory, and the pair decision. A pre-admission or causal
failure uses `loom.application_pair_disposition` version `1.0` and retains the
same typed decision without fabricating Mapping work. Area, power, and energy
remain explicit unsupported observations with null values when no owning
Evidence exists.

The selected candidate may retain several Mapping observations. The envelope's
selected plan ordinal and Mapping root must identify exactly one of them; that
same observation owns the selected System, completed runtime and comparison
Evidence references, and measured DFG, CGRA, and resource-cost values projected
into the final objective. Candidate-level convenience fields cannot select a
different observation or combine facts from several plans.

The referenced source package owns program sources and build semantics. Existing
Loom owners produce the linked LLVM module, Structured Program Candidate,
Canonical Dataflow Program, Mapping, Deployment, HardwareImplementation,
SimulationWorkload, SimulationRuntimeInput, EvaluationRequest, and
EvaluationEvidence. The manifest does not copy any of those payloads or define
an `ApplicationArtifact`.

One application identity may have several named input selections. Those
inputs change exact workload or runtime-input identity, not application
membership. `selection_inputs` is the sole mapping from each scheduling tier
to exact rows; no runner may substitute every application input or infer a row
named after the tier. The three execution selections are scheduling and
conformance policy over the same inventory:

* `smoke` is the bounded, deterministic developer gate;
* `validation` exercises representative functional and quality behavior; and
* `scale_eda` selects long-running scale, RTL, physical implementation, and
  EDA work where its required providers are available.

These names do not define training, validation, or held-out data roles.
`CalibrationPartitionRole` remains owned only by the model-training contract.
A runner may select any canonical subset explicitly, but it cannot publish a
different membership inventory or weaken the selected row's oracle.

## Bounded Host Runner

The bounded host runner is an operational conformance path for one exact
application/input selection or one explicit manifest tier. It consumes
`ApplicationManifest` and `SourceAdmission`; it does not parse a second
manifest shape, repeat source or cache admission, or infer a source set. A tier
run resolves exact rows through `selectApplicationInputs` and invokes the same
single-row runner for each member. It selects
`clang` for C and `clang++` for C++ from `PATH` unless the invocation names an
explicit compiler executable. Compilation runs with the repository root as
the compiler working directory, preserves manifest compiler and link option
order, and compiles the canonical source paths returned by admission in their
admitted order. The runner likewise consumes admission-owned oracle and cache
paths instead of resolving manifest paths again. Compiler outputs and captures
live in one unique invocation directory below the repository's ignored `temp`
directory and are removed when the invocation returns.

Host compilation defines `LOOM_APPLICATION_HOST_EXECUTION=1` after the selected
manifest options. Fixtures use that host-only boundary to emit their independent
observable oracle values; product compilation does not define it and retains
the ordinary source-to-Deployment behavior.

The host executable ABI is derived only for selections that reference cached
inputs. Such an executable receives the admitted absolute cache paths in the
selected manifest order, followed by the decimal `warmup_samples` and
`measured_samples` values. A selection without cached inputs receives none of
those derived arguments and is host-runnable only with zero warm-up samples
and one measured sample. Any other no-cache profile is typed
`unsupported_profile` rather than silently executing the wrong count. This
conditional ABI is owned here and is not a generic Simulation, Deployment, or
product runtime ABI. A future application whose host entry cannot consume this
shape requires an explicit portfolio contract change rather than
application-name dispatch in the runner.

This is a Linux host path. Execution inherits the invoking environment with
`LC_ALL=C`, disconnects stdin, and captures stdout and stderr separately. The
profile deadline covers only host executable wall time, measured with a
monotonic clock; it does not include compilation. Completion must be observed
before the deadline. Expiration terminates the detached host process group and
produces a typed timeout with no exit status. A leader that exits while another
member remains in its process group produces `execution_failure`; the group is
terminated before captures are read. A zero host exit is compared byte-for-byte
with the selected exact oracle. Typed invariant oracles remain typed
unsupported until their owning checker is registered; the runner never
reinterprets one as an exact oracle.

The runner preserves disjoint `source_unavailable`, `compile_failure`,
`execution_failure`, `timeout`, `oracle_mismatch`, `unsupported_oracle`,
`unsupported_profile`, and `succeeded` outcomes. Its deterministically ordered
JSON projection is schema `loom.application_host_run` version `1.0`. The
projection records the exact application/input and source/build selection,
workload and runtime-input names, cached-input declarations and digests,
oracle selection, complete profile, source-admission status, selected compiler
and compile exit status, host exit status and wall nanoseconds, oracle status,
and the typed outcome.
An explicit tier run wraps its unchanged member reports in
`loom.application_host_selection_run` version `1.0` and records the exact
execution-selection name. The wrapper has no aggregate performance metric and
cannot turn a failed member into a successful tier.
Signal and timeout sentinels are not exit statuses. Human compiler and runtime
diagnostics are preserved on the report across successful and failed stages but
remain outside that JSON projection.

This report is not an Artifact, `InvocationManifest`, or
`EvaluationEvidence`. Exact host stdout conformance does not join the manifest
workload/runtime-input names to canonical Simulation roots, establish a
Simulation execution result, or publish correctness or performance Evidence.
Those semantic bindings remain the responsibility of their existing runtime
and Evaluation owners.

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

`loom-application-manifest-inspect` emits the deterministic
`loom.application_portfolio_inventory` projection only after the canonical C++
manifest parser accepts the source document. The evidence generator consumes
that projection and refuses raw manifest JSON, so it cannot become a second
manifest parser or normalize a document rejected by the semantic owner.

The derived evidence manifest joins each exact inventory row independently to
its bounded host report and pair decision. It publishes separate host
conformance, typed pair-disposition, and canonical application-QoR gates. An
explicit unsupported, timeout, or proof-not-established pair closes only the
typed disposition gate. Every non-TinyML selected row also requires canonical
QoR; the current TinyML product-profile refusal is the sole explicit
host-plus-typed-residual exception. The per-member evaluation records every
contributing report and pair count so an untyped duplicate cannot be hidden by
a valid row. Unsupported objective dimensions retain a null value.

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

Stable tests validate manifest schema and uniqueness, exact tier-to-input
selection, source-root and Gitlink resolution, selected-input cache and oracle
admission, bounded profile parsing, native host output for the five admitted
rows, byte-exact bounded TinyML inference under its declared deadline,
product-driver argument projection, and rejection of partial, injected,
replayed, target-conflicting, or unsupported-profile selections.

Canonical release closure remains pair-local. The Attention, Llama kernel,
regular `vecadd-memory`, and irregular PageRank anchors require exact manifest
selection, completed source-backed Simulation and oracle Evidence, one
selected Mapping candidate, host baseline, and complete application QoR. The
bounded TinyML row independently proves its real one-warm-up/four-measured
host profile and exact oracle, while its current product profile limit is a
typed unsupported pair rather than fabricated Mapping or QoR. The derived
evidence manifest verifies each host and pair projection against the exact
manifest row, reports every tier independently, and retains null unsupported
QoR dimensions as typed residuals. Additional runtime profile shapes require
production Evidence from their existing runtime, Mapping, and Evaluation
owners before they can be selected.

## Implementation Responsibility Review

`lib/Application/BuildDiagnostics.cpp` is the sole presentation owner of the
three versioned pair envelopes: the pair decision, the pair evidence join,
and the pre-admission pair disposition. The semantic derivation of the pair
decision record and its disposition vocabulary live in `PairDecision.cpp`,
which contains no serialization; the build transaction in `Build.cpp`
decides when a decision is terminal and this owner decides how it is
rendered; the visualization bundle embeds the same pair-decision projection
verbatim rather than re-encoding it, and two independent verification
harnesses pin that byte identity. The envelopes share one leaf encoder set
and one quality-provenance encoding, so an envelope-line split would either
duplicate the schema authority or add a pass-through encoding layer; the
file therefore stays the single envelope owner. The recorded consolidation
obligation runs in the other direction: the local candidate, materialized,
work-counter, and planning-record serializers duplicate the DSE-owned
serializers in `PreMappingEvidence` and have drifted on the temporal-witness
shape; the duplicate copies must be retired in favor of the DSE owner after
a deliberate decision on the canonical witness field shape.
