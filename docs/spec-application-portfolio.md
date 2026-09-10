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
after its longer warm-up. The shared host/product implementation uses
real-valued requantization compatible with those reference semantics; it does
not claim bit equivalence to optimized fixed-point builtin or TFLite Micro
kernels, or complete MLPerf anomaly MSE reporting. The product entry consumes
the admitted model and
dataset bytes, executes the manifest counts, and writes every measured output
byte through the ordinary Deployment and System memory ABI. A separate
Evaluation model compares that observation with the independently admitted
oracle. This remains a bounded C++ application, not a TFLite product frontend.

## Manifest Contract

Each manifest row owns only the repository-level selection needed to run one
application:

* its stable application identity and source-package root;
* the exact build entry, language mode, source selection, and compiler and link
  options;
* an explicit product entry symbol and measured output extent when the row has
  a product execution ABI;
* named workload and runtime-input selections;
* the independent oracle or typed invariant bound to each selection;
* the bounded warm-up, measured-sample, oracle-coverage, and execution-deadline
  profile bound to each input; and
* the exact input rows selected by the `smoke`, `validation`, and `scale_eda`
  execution policies.

The tracked JSON contract is schema `loom.application_portfolio` version
`4.0`. Version 4.0 incompatibly requires a product-execution selection and a
pinned oracle digest and encoding. Its exact structural shape is:

```text
{
  "schema": "loom.application_portfolio",
  "version": "4.0",
  "applications": [{
    "identity": <stable logical name>,
    "source": {"kind": "gitlink" | "repository", "root": <repo path>},
    "build": {
      "entry": <selected C/C++ translation unit>,
      "language": "c" | "c++",
      "sources": [<source-relative translation units>],
      "compiler_options": [<argument>],
      "link_options": [<argument>],
      "operator_protocol_symbols": [<linked symbol>],
      "product_execution": null | {
        "entry_symbol": <external C symbol>,
        "measured_output_bytes_per_sample": <positive unsigned integer>
      }
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
      "oracle": {
        "kind": "exact" | "typed_invariant",
        "entry": <repo path>,
        "sha256": <digest>,
        "encoding": "utf8" | "hex_sample_lines"
      },
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
competing CLI symbol list. `product_execution` is required and null for an
ordinary host-shaped row. A non-null selection requires at least one operator
protocol symbol, an exact `hex_sample_lines` oracle, and a positive per-sample
output extent. Cached-input order, the selected profile, and this small
application-specific selection mechanically derive the complete product ABI;
the manifest does not duplicate its argument list. All paths are normalized
visible-ASCII relative paths; stable logical names use lowercase ASCII letters,
digits, `.`, `_`, or `-`.
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
program translation unit. Cache and oracle bytes must match their declared
SHA-256.
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

For a product row with `N` cached inputs, the selected C entry has `N`
`(pointer, byte_count)` pairs followed by `warmup_samples`,
`measured_samples`, `output_pointer`, and `output_byte_count`, and returns an
`i32` status. All scalar arguments are unsigned 64-bit values. Cached bytes,
counts, and independent zeroed output storage are owned by the Structured
Program workload/runtime input and projected without reinterpretation into
the Deployment Host entry and System activation. Cached interfaces are typed
Input and the final output interface is typed Output. This
`cached_inputs_profile_output_v1` ABI executes every declared TinyML warm-up
and measured sample; no count is replaced by a product-local default. Oracle
bytes never enter the runtime input, guest arguments, or candidate identity;
only the independent Evaluation comparison receives them.

The `loom.application_pair_decision` version `3.0` projection records the
resolved application identity, input name, source/build selection,
input-specific compiler options, declared workload and runtime-input names,
declared oracle and bounded profile, and referenced cache digests. Its typed
execution binding is `declared_only`, `canonical_simulation`, or
`canonical_simulation_and_oracle`. The last state is reached only after the
existing Mapping runtime owner completes source-backed DFG and CGRA
Simulation and a native `SimulationComparison` reports no finding. Its exact
Evidence references are carried by the selected Mapping observation, and the
derived `execution_binding_established` compatibility field is true only in
that state. The decision names the selected System, the selected Mapping, and
the resource-time schedule finalist (`selected_schedule_hint_digest`) whose
verified observation selected them. Equivalent schedule hints share one
Mapping plan and are each verified against the same Mapping, so the plan and
Mapping alone do not identify one observation; the finalist digest does. The
decision also derives the attempted and verified adjacent resource-time
Mapping-repair counts and the highest-priority typed incomplete reason from
the retained repair observations. This status remains separate from the
selected parent's application disposition and from the later Deployment-level
transition finalizer. Each observation retains its exact spectrum endpoint,
generated and eligible Mapping frontiers, per-side plan-incomplete reasons,
runtime disposition, runtime Evidence roots, and oracle Evidence roots. A side
is verified only when its exact Mapping and requested class occur in one
verified spectrum scenario, its provider plan completed, and its completed
runtime Evidence is joined to the declared oracle Evidence. When a
hardware-alternative observation comes from an
executed mutation repair, it
also names that exact durable record. General hardware-frontier exploration
has no parent Mapping repair and therefore names no such record. A
pre-admission or unsupported profile decision remains `declared_only` and
cannot be interpreted as correctness Evidence.

The manifest profile deadline bounds host executable wall time under the
bounded host runner contract. Mapping runtime validation uses the invocation's
existing Mapping execution deadline. It admits the complete source-backed
replay sequence and performs DFG execution, CGRA execution, and independent
comparison for each replay, including durable Request, Execution, and Evidence
publication within that Mapping budget. Independent replays may execute
concurrently within the existing execution site's in-process CPU claim.
Workers share one bounded invocation import cache through retained
attachments; they do not multiply its capacity. Immutable preparations may be
shared, while inputs, queues, memory, reservations, and execution counters
belong to each fresh session. Results are joined in the original replay
ordinal order. A failed run retains the prefix through the earliest failed
ordinal, independent of worker completion order, and no partial replay set
establishes the application oracle.

The derived resource-time Mapping-repair counters and typed reason remain
presentation-only projections of the retained application observations;
they do not enter pair identity, Mapping selection, or transition legality.
All pair projections use the same current decision vocabulary and fixed units.

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
second copy of those payloads. The published application runtime manifest
binds the pair identity, its source program, Fabric, workload, runtime input,
selected System and Mapping, the entry Deployment, and the activation
workload and runtime input that Deployment executes; the build projects it
once into its diagnostics (`application_runtime_manifest` statistics), so an
execution manifest naming the same Deployment, activation workload, and
activation runtime input is verifiably bound to the pair decision's exact
identities without decoding the package. A product manifest also binds the
derived ABI, entry symbol, exact profile, per-sample output extent, output
interface, and a Blob containing the decoded expected bytes. The source oracle
file remains the manifest-owned authority; that Blob is derived only after
digest and exact line/order/extent validation.

A successful decision is published inside
`loom.application_pair_evidence` version `3.0`. That envelope is the canonical
join of candidate and analytic-gate inventories, actual Tech/Spatial/System
work, selected Mapping checkpoint, failure-cone counters, work ledgers,
Mapping outcome inventory, exact promoted-parent hardware attempt lineage,
and the pair decision. A pre-admission or causal
failure uses `loom.application_pair_disposition` version `2.0` and retains the
same typed decision without fabricating Mapping work.
Each Mapping observation names `runtime_mapping` only when its runtime fields
were measured for that exact SystemMapping. The surrounding generated Mapping
frontier cannot stand in for this identity.
The successful envelope's repair-record inventory is derived from the same
`loom.application.activation_decision` 4.0 owner projected by runtime manifest
9.0. The strict activation decision is also the sole application replay and
Evidence join owner. Its immutable object retains derived cycle totals,
Request dependency roots, execution output roots, and the aggregate CGRA
retirement predicate; these facts are never separately serialized or assigned
an identity. Runtime manifest construction strictly imports that decision in
its current ArtifactStore, checks exact source and selection equality, and
uses those facts for its own Deployment closure and activation constraints.
Cold import continues to recompute the full proof at the activation owner.

The join strictly resolves every declared source replay input through the DFG
model owner in the current ArtifactStore, then reuses that exact resolution
for its DFG Evidence. Source input availability, canonical runtime bytes and
workload lineage are dependency obligations. Value construction and standalone
joins report an invalid declared source input as `DependencyMismatch`. Cold
decision decoding first imports its typed invocation reference; failures at
that boundary retain `InvocationMismatch`. CGRA
Mapping and Fabric ownership remain validated by the CGRA model owner.
Per-case runtime buffers are released after resolution; the input resolutions
remain local to that join invocation.

A coherent CGRA closed-wait result remains distinct from retirement, and the
runtime manifest rejects it as `RuntimeEvidenceMismatch`. A CGRA claim with an incompatible model or incoherent
terminal/finding pair fails earlier at the activation Evidence owner as
`EvidenceMismatch`; a manifest importing that decision reports
`ActivationDecisionMismatch`. This reflects the strict proof owner's
responsibility rather than preserving duplicate validation for historical
error ordering.

When the selected
`hardware_dse_alternative` observation names a mutation repair, the runtime
manifest names the same unique record. Omitting a unique record that selects
the activation SystemMapping is invalid; a general hardware-frontier selection
and every non-hardware disposition name no selected repair record.

Promotion lineage is invocation-scoped. Each `quality_invocations` entry owns
one InvocationManifest run key, local plan-ordinal base, promotion count,
promotion observations, and the attempts they caused. Consumers must complete
the parent-to-child join inside that entry; aggregate Mapping attempts are not a
substitute for an invocation identity. The invocation-local promotion
count and attempt inventory remain mandatory in version 2.0.

Version 2.0 replaces the ambiguous host-work observation with
`host_only_runtime_picoseconds` and adds `candidate_runtime_picoseconds`.
The former is the exact source program's host-only Runtime estimate; the
latter is the complete Structured candidate's Runtime estimate. Both use the
same `StructuredFabricLowConfidence` model binding, exact Fabric, workload,
and runtime input, and both carry the analytic evidence grade. A block
activation count, a dynamic leaf count, a native wall time, or a candidate
estimate cannot populate the source baseline. Missing or inapplicable model
results remain unsupported with null values. The host-only vector contains
no selected candidate's DFG or CGRA observation.

Every objective dimension is a non-negative integer in the fixed unit owned
by `ApplicationObjectiveDimension`. DFG and CGRA cycles, resource-core cost,
and Mapping work are runtime-measured. Host residual work is the complete
selected Structured candidate's executable host leaf count weighted by its
exact observed block activations. It is derived by the Structured analytic
owner over all modeled executable blocks, including work outside the
pre-Mapping protocol-root domain, and carries the analytic evidence grade.
The block observation or exact activity-preserving lineage is the source of
truth; overlapping scope sums and unselected protocol roots are not a
whole-program residual. External calls remain executable Structured leaves;
the count is not a target instruction count or measured host cycles.
Cut transfer work remains bytes crossing the host/accelerator cut, and
launch/synchronization work remains the planning projection's structural
count. They carry analytic grade and are never added to cycle or leaf counts.
The Structured analytic model owns all conversions used in its complete
Runtime estimate; consumers cannot add those modeled costs a second time.
Area in square micrometers, power (dynamic plus leakage) in microwatts, and the energy of one measured CGRA execution at the
predicted limiting clock in picojoules are joined only from the selected
Mapping's completed calibrated FPA observation under the invocation's frozen
`EdaPredictionModelWeight`; they carry the calibrated evidence grade. Without
that observation they remain explicit unsupported observations with null
values; the exact decimal metrics stay owned by the FPA Evidence root.

The completed same-hardware disposition is `verified_feasible`.
It proves Mapping and functional execution feasibility, not acceleration.
The decision's `benefit_status` is a mechanically derived closed value:
`unknown`, `predicted_beneficial`, or `predicted_not_beneficial`. Only matched
analytic source/candidate runtimes can produce a prediction, with strict
less-than defining predicted benefit. A selected hardware alternative or an
absent exact candidate/runtime join yields unknown. The prediction neither
overrides the central DSE ordering policy nor becomes runtime-measured
Evidence. This immutable planning boundary has no final-application QoR
completion field. Build success enables execution evidence acquisition.

The product driver accepts `--loom-gem5-readiness=<path>` to pin the native
System evaluator for bounded-quality Mapping selection. Each verified Mapping
produces its executable Deployment without an activation decision; this image
is executable before choosing a winner. The source HostOnly image and candidate
run through the ordinary gem5 CGRA Evaluation provider under the enclosing
Mapping deadline and the portfolio's unchanged simulated-work limit. Exact
external-tool reuse remains owned by the invocation-bundle contract.
Within one bounded-quality invocation, completed System Evidence is retained
by its exact Evaluation Request identity. Repeated requests reuse that observed
result and still pass the ordinary strict System Evidence join. This includes
the common HostOnly request across different mappings of the same source and
System. Only Evidence with a `Completed` outcome is retained; the ordinary
quality join still rejects functional disagreement or an absent computation
interval. The retained observations expire with the bounded invocation; they
introduce no persistent lookup index or competing Request key.

Native System quality retains the Spatial DFG, CGRA, and independent source
comparison Evidence, and adds the complete host/candidate System Evidence pair.
The System pair must use the same exact machine and observation conditions,
execute activation inputs re-derived from the source invocation, and produce
identical complete functional observations. The host-only image uses the same
System and HostCore compiler target as the candidate. Failed, unsupported,
unmeasured, or interrupted System execution produces typed incomplete quality;
it cannot retain a positive Objective. Product-oracle qualification remains a
separate publication obligation.

`ApplicationSystemRuntime` quality appends `system_computation_ticks` to the
Spatial runtime measures and ranks this measure first, followed by CGRA cycles,
DFG cycles, and physical AccCore cost. Optional calibrated FPA measures follow
the complete runtime prefix. `ApplicationRuntime` retains its Spatial-only
measurement domain. The choice is explicit in the invocation, never inferred
from ambient tool availability. Activation decision 4.0 admits the native
Evidence pair and independently re-derives its candidate computation time;
this derived value is not another serialized measurement authority.

`qualifyApplicationSystemQor` owns the post-execution relation. It strictly joins
the exact runtime manifest, host-only and mapped System executions, their
completed Runtime Evidence, and each required passing product-oracle Evidence.
Both requests use identical gem5 binding, model, model inputs, configuration,
conditions, and replicate index. Their complete functional observations agree.
The exact Runtime values must equal the complete program tick windows. The
result is derived from immutable owners, not stored as another Artifact or
written back into the earlier pair decision.

Performance qualification measures the useful computation explicitly delimited
by the source's `loom_computation_begin()` and `loom_computation_end()` calls.
The Runtime ABI owns these observations. Both the host-only and candidate
images use the same source boundary, independent of which subregions DSE
selects for acceleration. Input generation, warmup, and result verification
remain outside it. All residual host computation, dispatch, communication,
synchronization, and output publication between the boundaries remain charged.
A workload has at most one computation interval; a batch encloses all its
measured samples in that interval rather than summing favorable subintervals.

The interval begins with the input data visible in shared external memory and
ends after the computation's output data is visible there. This is coherent
System memory visibility, not a claim that dirty cache lines have reached
physical DRAM. The same cache hierarchy and visibility contract apply to both
members. No initializer or oracle is offloaded merely to improve a benchmark
ratio. Full-program Runtime remains an exact separate diagnostic; it cannot
substitute for an absent computation interval.

Speedup is the host computation interval divided by the candidate computation
interval. Shared-memory utilization is the difference of the native cumulative
service samples at the computation boundaries divided by that same interval.
There is one service observer. Idle cycles and necessary host work do not
vanish from the denominator.

Compute occupancy uses the retired Compute-kind actor firings of the standalone
CGRA replay for each invocation that completed inside the candidate computation
interval. DFG replays remain correctness oracles. The boundary device refuses
unfinished accelerator invocations at either marker, so this membership selects
whole invocations and excludes warmup. The denominator is the distinct mapped
compute PE count times the distinct launched AccCore count times the interval's
reference cycles. The SpatialCore clock-domain contract supplies the reference
period; no frequency is assumed. An interval without accelerator launches has
zero compute occupancy and cannot qualify as acceleration.

The `loom.application.system_qor_projection` version `3.0` reports exact roots,
full-program durations and memory activity, each member's optional computation
interval, measured speedup, and the candidate's compute occupancy. Qualification
requires strict computation speedup and strictly more than 90 percent shared
memory service utilization or compute occupancy over that computation interval.
Neither the threshold nor the machine capacity changes with the measurement
boundary. Missing intervals produce null speedup and `unmeasured` status and
bottleneck; portfolio qualification rejects them. A boundary present on only
one pair member is an invalid comparison.

Bottleneck classification is explanatory, not a second gate. It selects
`memory_bandwidth_bound` for saturated memory service, then `compute_bound`
for saturated compute occupancy. Otherwise it selects `host_bound` when the
candidate's first-root-Start to last-root-Completion span, intersected with the
computation interval, covers less than one tenth of the useful computation;
otherwise it selects `latency_bound`. Initialization and verification outside
the computation interval do not contribute to this classification. System
outcomes are feedback inputs for DSE; successful Mapping alone establishes no
measured performance benefit.

The System driver always retains valid measured results, including regressions.
The real-application verifier and portfolio qualification consume this
post-execution result and enforce its performance target. Ordinary semantic
fixtures require a complete valid pair without asserting a useful acceleration
for an intentionally tiny program. Neither flow asks the pre-execution decision
to contain future measurements.
`host_only_baseline_complete` means that the exact source analytic baseline
is available; it does not make target runtime or the full QoR gate complete.
An honest unknown benefit, a feasible artifact, or a rejected incidental
initialization graph does not close the real-application optimization gate.
The compiler must retain the incomplete result while continuing the bounded
candidate workflow toward a useful application kernel. Product qualification
must preserve the declared kernel launch, full input, independent oracle,
performance budget, and required resource-use objective.

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
`loom.application_portfolio_inventory` version `2.0` projection only after the
canonical C++ manifest parser accepts the source document. The evidence
generator consumes that projection and refuses raw manifest JSON, so it cannot
become a second manifest parser or normalize a document rejected by the
semantic owner.

The derived evidence manifest joins each exact inventory row independently to
its bounded host report and pair decision. It publishes separate host
conformance, typed pair-disposition, canonical application-QoR, and declared
product-execution gates. An explicit unsupported, timeout, or
proof-not-established pair closes only the typed disposition gate. Every
selected row requires canonical QoR. Each TinyML row additionally joins the
exact pair to its Application runtime manifest and requires DFG and CGRA
System executions whose separate product-oracle Evidence reports no mismatch.
The per-member evaluation records every
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
selected Mapping candidate, host baseline, and complete application QoR. Both
bounded TinyML rows also execute their full one-plus-four and two-plus-two
profiles through the product entry, observe the complete measured-output
memory, and publish independent oracle Evidence for both System engines. The
derived evidence manifest verifies each host and pair projection against the
exact manifest row, reports every tier independently, and retains null
unsupported QoR dimensions as typed residuals. Additional runtime profile
shapes require
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
