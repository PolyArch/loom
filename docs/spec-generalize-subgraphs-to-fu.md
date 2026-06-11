# Subgraph-to-FU Generalization

This document specifies the target contract for
`loom-generalize-subgraphs-to-fu`, the Fabric technology pass that
generalizes one or more software `dataflow.subgraph` partition units
into a reusable PE-local `fabric.fu` template.

Given a synth group of `dataflow.subgraph` instances from
`dataflow.graph` definitions, the pass emits a legal Fabric template
whose materializations under `loom-enumerate-fu-subgraphs` include every
input subgraph in that group. The emitted template may enumerate
additional legal configurations when the merged hardware has a larger
configuration space than the observed software set.

This is a target-state specification, not an implementation plan. It
defines semantic boundaries, observable pass behavior, failure
semantics, strategy obligations, and verification evidence. Concrete
C++ class layouts, private helper APIs, parallel-execution mechanics,
file splits, and test directory organization are non-normative.

## Owning Specifications

This spec is subordinate to these owning specs:

* `docs/spec-core-dialect-boundary.md` for the ownership boundary
  between software dataflow, hardware Fabric, mapping artifacts,
  runtime, and reports.
* `docs/spec-fabric-module.md`, `docs/spec-fabric-pe.md`, and
  `docs/spec-fabric-fu.md` for SpatialCore, PE, and FU semantics.
* `docs/spec-fabric-reconfigurable-op.md` for `fabric.op`
  configuration axes and enumerator semantics.
* `docs/spec-fabric-hw-share-group.md` for legal operation sharing.
* `docs/spec-mapping-artifact.md`, `docs/spec-mapping-placement.md`,
  and `docs/spec-mapping-search.md` for downstream software-to-hardware
  binding and evidence.

If this document conflicts with an owning spec, the owning spec wins and
this document must be updated.

## Semantic Boundary

`dataflow.subgraph` remains a software partition unit. It does not carry
PE identity, FU identity, tile identity, routes, schedule slots,
temporal tags, resource-sharing decisions, or placement decisions. The
partitioning compiler owns which regions become subgraphs.

`fabric.fu` remains PE-local. It is not a module-level tile kind, not a
top-level hardware target, and not a hardware identity wrapped in
`func.func`. The SpatialCore tile kinds remain `fabric.pe`,
`fabric.switch`, and `fabric.mem`; each tile may be `spatial` or
`temporal` as specified by the Fabric specs. FUs belong inside PEs.

`fabric.module` remains the SpatialCore template container. Its internal
connectivity is represented by graph-region SSA values and the legal
Fabric operations for that container, not by a private link syntax
invented by this pass. Mapping artifacts, PnR, simulation, and RTL
flows decide later whether and how software subgraphs bind to concrete
resources.

The pass follows the Loom RISC/Occam rule: it does not invent a meta
hardware instruction. It composes the existing Fabric configuration
primitives, especially `fabric.op`, `fabric.mux`, `fabric.demux`, and
`fabric.yield`, inside a legal PE-local FU body.

## System Contract

The intended flow is:

```text
dataflow.graph
  -> graph partitioning
  -> dataflow.subgraph groups
  -> loom-generalize-subgraphs-to-fu
  -> fabric.module / fabric.pe / fabric.fu templates
  -> loom-enumerate-fu-subgraphs
  -> materialized dataflow.subgraph candidates
  -> subgraph matching and mapping artifact construction
  -> software-to-hardware binding evidence
```

Architect-authored FUs and synthesized FUs share the same Fabric
semantics. A synthesized FU is reusable hardware template evidence; it
is not a placement result. Its stable identity is the owning
`fabric.module` symbol, the owning `fabric.pe` symbol, and the PE-local
`fabric.fu` symbol.

The central correctness invariant is coverage: re-running
`loom-enumerate-fu-subgraphs` on the synthesized FU must produce a set
of materialized software subgraph candidates that contains an isomorphic
match for every input subgraph in the synth group. This invariant is
verified at synthesis time and is the end-to-end correctness gate.

## Goals And Non-Goals

Goals:

1. Coverage correctness: every input subgraph in a successful synth
   group is covered by at least one enumerated FU materialization.
2. Cost preference: among legal covered templates explored by the
   selected strategy, prefer the candidate with the lowest configured
   hardware-cost estimate.
3. Tiered target scope:
   * tier A covers identical DAG topology with varying operation
     identity at aligned positions;
   * tier B covers a common skeleton with localized branch differences,
     extra or missing edges, and fanout-shape variation;
   * tier C covers heterogeneous topology with graph-region feedback
     patterns represented through legal Fabric operations.
4. Strategy interchangeability: `anchor`, `mcs`, `incremental`, and
   `incremental_random` expose one common semantic result contract.
5. Determinism: deterministic configurations produce stable IR,
   diagnostics, coverage reports, and cost choices independent of
   parallel scheduling.
6. Self-verification: synthesized output is checked by the existing FU
   enumerator and subgraph matcher rather than a separate hand-written
   coverage proof.

Non-goals:

* The pass does not decide graph partitioning.
* The pass does not rewrite input subgraphs except for failure
  annotations on rejected groups.
* The pass does not emit `fabric.fifo`; buffering is a downstream
  scheduling concern.
* The pass does not synthesize multiple FUs for one synth group. One
  group produces at most one FU template.
* The pass does not perform placement, routing, scheduling, physical
  floorplanning, EDA execution, or RTL/FPA estimation.
* The pass does not introduce a private hardware-share-group table or a
  strategy-local sharing override.

## Glossary

* **input subgraph**: a `dataflow.subgraph` operation contained in a
  `dataflow.graph` definition body.
* **synth group**: a set of input subgraphs intended to be covered by
  one synthesized FU. A string-valued `loom.synth_group` attribute names
  the group. Subgraphs without the attribute belong to the implicit
  `"default"` group.
* **synthesized FU**: the PE-local `fabric.fu` template produced for a
  synth group.
* **materialization**: one software `dataflow.subgraph` candidate
  enumerated from a configured `fabric.fu`.
* **coverage report**: structured evidence that maps every input
  subgraph to at least one enumerated materialization, or records the
  miss that caused synthesis failure.
* **alignment**: the semantic relation that decides which operation or
  value positions across input subgraphs may be represented by the same
  legal FU-body operation position or FU boundary port.
* **share group**: a multi-member hardware-share group defined by
  `docs/spec-fabric-hw-share-group.md`. Operations may share one
  `fabric.op` only when the owning share-group and data-path width rules
  permit it.

## Pass Interface

```text
Pass:     loom-generalize-subgraphs-to-fu
Scope:    ModuleOp
Inputs:   dataflow.subgraph ops inside dataflow.graph definitions,
          optionally annotated with loom.synth_group
Output:   the same module plus a legal Fabric template container for
          each successful group; failed groups are annotated
Options:  config=<path>
          fail-as-error=<bool>
          dump-stats=<bool>
```

The output shape for each successful group is an owning
`fabric.module` containing a legal `fabric.pe` that owns one PE-local
`fabric.fu` template for the group. The pass may use named or anonymous
Fabric forms only where the owning Fabric specs permit them. Generated
module/PE/FU identities use a deterministic sanitized group-name scheme
and are checked for collision at the full module/PE/FU symbol path.
PE external-port routing is represented by the PE configuration and
mapping evidence specified by `docs/spec-fabric-pe.md`, not by a
private wiring convention invented by this pass.

The pass is part of the Fabric technology pass set exposed through the
standard `loom` driver.

`dump-stats=true` emits deterministic per-group diagnostics containing
the selected strategy, cost, coverage count, and failure reason when
applicable. These diagnostics are evidence, not a replacement for the
coverage report.

## Configuration Surface

The concrete serialized config schema is owned by the pass config
verifier. Its accepted semantic axes are:

* strategy selection from the public strategy set and an explicit
  fallback order;
* timeout, candidate, resource, and deterministic parallelism bounds;
* hardware-cost weighting for legal Fabric structures;
* sharing and mux/demux decomposition policy;
* input-order, restart, and graph-search policies that preserve the
  common coverage contract;
* tier-C feedback policy for either legal alignment or conservative
  separate-state fallback.

Every accepted config axis must have focused evidence for its observable
effect. A malformed config reports `config_parse_failed` and does not
mutate user IR.

## Input And Output Requirements

Input requirements:

1. Each input `dataflow.subgraph` must satisfy the subgraph verifier
   contract, including explicit boundaries, supported body operations,
   legal graph-region structure, and memory exclusions where required by
   the dataflow specs.
2. Invalid subgraphs are skipped, annotated with
   `loom.synth_failed = "invalid_input"`, and reported with a
   diagnostic.
3. Groups are processed in lexical order of group name for deterministic
   output.

Output requirements:

1. Successful groups append a legal Fabric template and do not mutate
   the input software subgraphs.
2. Failed groups do not append partial Fabric IR and annotate every
   offending input subgraph with exactly one closed failure reason.
3. Failure in one synth group does not prevent other independent groups
   from emitting legal successful templates.
4. Rerunning the pass on a previously synthesized module/PE/FU symbol
   path for the same group is idempotent and emits a remark.
5. Name collisions with non-synthesized Fabric symbols are reported as
   `symbol_conflict` failures.
6. The emitted `fabric.module`, `fabric.pe`, `fabric.fu`,
   `fabric.instantiate`, and nested FU body operations pass the MLIR
   verifier before the output is accepted.
7. State-bearing dataflow operations inside a synthesized FU are
   represented as legal `fabric.op` configurations, for example
   `fabric.op [@dataflow.carry]`; bare dataflow operations are not
   emitted into the FU body.

Observable acceptance criteria:

1. Empty input produces an unchanged module and a deterministic remark.
2. A single valid input subgraph produces a template that enumerates and
   matches that subgraph.
3. Distinct `loom.synth_group` values produce independent FUs in
   lexical group-name order.
4. Every successful group has a coverage report with
   `covered=<m>/<m>`.
5. Failure to synthesize one group does not prevent successful groups
   from emitting legal Fabric templates.
6. Rerunning on already synthesized output is a no-op for that group.
7. Any verifier failure drops the candidate and reports
   `verifier_failed`.
8. Any invalid input subgraph is annotated with `invalid_input` and is
   not enqueued for synthesis.

## Minimal IR Shape

This illustrative shape shows the target ownership hierarchy. Exact
assembly syntax is owned by the dialect printers.

```mlir
fabric.module @loom_synth_fus(%a : !fabric.bits<32>,
                              %b : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  %pe_out = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                              %pb = %b : !fabric.bits<32>)
                             -> !fabric.bits<32> {
    fabric.fu @fu_alu_int_32
        (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32> {
    ^bb0(%aa : !fabric.bits<32>, %bb : !fabric.bits<32>):
      %r = fabric.op [@arith.addi, @arith.subi] (%aa, %bb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %r : !fabric.bits<32>
    }
  }
  fabric.yield %pe_out : !fabric.bits<32>
}
```

The PE result in this sketch is the PE external output port. Which PE
inputs drive which FU inputs, and which FU outputs drive which PE
outputs, is PE configuration evidence owned by
`docs/spec-fabric-pe.md`. It is not expressed by PE-body SSA wiring in
the subgraph-generalization contract.

Failed groups are reported on the software subgraphs:

```mlir
dataflow.subgraph (...) -> (...) {
  ...
} {loom.synth_group = "loose",
   loom.synth_failed = "cross_share_group"}
```

## Failure Reasons

Failure reasons are a closed set:

* `cross_share_group`: aligned operations would require illegal sharing
  across hardware-share groups and no legal mux/demux decomposition was
  accepted.
* `topology_mismatch`: the selected strategy could not express the
  structural difference using legal Fabric operations.
* `feedback_align_conflict`: tier-C feedback signatures could not be
  aligned or conservatively separated into a covered candidate.
* `timeout`: the selected strategy exceeded its configured time budget.
* `resource_exhausted`: the selected strategy exceeded its configured
  candidate or memory budget.
* `unsupported_op`: an input subgraph contains an operation that cannot
  be represented by `fabric.op` according to the reconfigurable-op spec.
* `invalid_input`: an input `dataflow.subgraph` violates the dataflow
  verifier contract.
* `verifier_failed`: a synthesized candidate failed MLIR verification.
* `symbol_conflict`: the generated Fabric symbol path conflicts with a
  non-synthesized symbol.
* `config_parse_failed`: the pass configuration failed to load or
  failed schema validation.

The exact string is stored in `loom.synth_failed` on the affected input
subgraph operations and appears in diagnostics. `fail-as-error=true`
upgrades failure diagnostics to errors without changing the closed set.

## Strategy Obligations

All strategies share one result contract:

* input: one synth group, the group's dataflow subgraphs, the active
  config/profile identity, and canonical Fabric/dataflow semantics;
* success: one legal generated PE containing one PE-local FU, stable
  module/PE/FU identity, coverage evidence, cost evidence, and
  diagnostics;
* failure: one closed failure reason and diagnostics, with no partial
  Fabric IR appended for that group.

Strategy-specific target obligations:

* `anchor` is the tier-A baseline for isomorphic topology. It may cover
  restricted tier-B cases only when the difference is representable by
  legal local mux/demux structure.
* `mcs` is the graph-native common-skeleton strategy across all tiers.
  It chooses the lowest-cost candidate that passes coverage
  verification under explicit time and candidate caps. It must not hide
  an incremental fallback inside the strategy; fallback is controlled by
  the outer strategy chain.
* `incremental` is the coverage-preserving extension strategy across
  all tiers. Every accepted intermediate candidate and the final result
  must preserve coverage for the relevant input prefix.
* `incremental_random` is the deterministic seeded restart variant of
  `incremental`. It returns the lowest-cost successful covered
  candidate and identical seed plus config produce identical results.

The strategy table is:

| Strategy | Tier A | Tier B | Tier C | Primary evidence |
| --- | --- | --- | --- | --- |
| `anchor` | yes | restricted | no | isomorphic coverage and legal sharing |
| `mcs` | yes | yes | yes | verified common-skeleton candidates |
| `incremental` | yes | yes | yes | monotonic coverage-preserving extension |
| `incremental_random` | yes | yes | yes | seeded deterministic restart selection |

## Shared Semantic Components

### Alignment

Alignment must agree with the subgraph matcher on value and operation
identity. It covers body operation results, subgraph boundary values,
commutative operand normalization, multi-result operations, multi-yield
outputs, and graph-region back-edges. A strategy may use different
search mechanics, but accepted alignments must be compatible with the
same enumerator/matcher roundtrip used for final coverage.
Accepted alignments materialize as legal `fabric.op` configurations,
`fabric.mux` or `fabric.demux` arms, or PE-local `fabric.fu` boundary
ports. They do not create a separate Fabric node-kind namespace.

### Coverage Verification

Coverage verification uses the canonical FU enumeration and subgraph
matching semantics as the oracle. For each input subgraph, the coverage
report records a matching materialized candidate or a miss. A candidate
is accepted only when every input in the synth group is covered.

Parallel matching may be used as an optimization, but single-worker and
multi-worker coverage reports must be equivalent for deterministic
configurations.

### Feedback And Tier C

Tier-C feedback support is expressed through graph-region semantics and
legal Fabric operations. `dataflow.carry`, `dataflow.gate`, and
`dataflow.invariant` are represented as `fabric.op` configurations.
They do not own `step_op` or `cont_cond` attributes. Those loop-shape
parameters are observed through upstream `dataflow.stream` operations as
specified by `docs/spec-dataflow-part-1-streaming.md`. Feedback
compatibility is determined by observable dataflow structure, carried
value types, condition sources, and upstream stream configuration.

If feedback paths cannot be aligned, a conservative strategy may keep
incompatible state paths separate. Such a candidate is accepted only if
the enumerator and matcher prove it covers every input.

### Cost Model

The cost model is an analytic ranking function for legal FU candidates.
It accounts for Fabric operations, operation share groups, bit-width,
mux/demux structure, and state-like Fabric op configurations. The same
function ranks candidates within a strategy, candidates produced by
fallback strategy runs, and restart results. It is pure: the same
candidate and config produce the same score across runs and threads.

The compiled default weights are implementation details. The normative
requirements are:

1. Operation base cost is linear in bit-width for a fixed share group;
   an i64 operation has twice the base cost of an i32 operation in the
   same group.
2. Mux and demux costs are positive and proportional to width and port
   count.
3. Adding a state-like Fabric op configuration strictly increases cost
   under positive state penalty.
4. Cost evaluation is pure across runs and threads.
5. Ties are broken by deterministic structural identity.

## Determinism And Parallelism

Parallelism is an optimization, not a semantic axis. The pass may
parallelize independent synth groups, coverage matching, graph search,
and random restarts, but accepted IR and diagnostics must match the
single-worker result for deterministic configurations.

Successful groups are emitted in lexical group-name order. Parallel
execution must not change accepted IR, diagnostic order, coverage
report order, or failure annotations.

Emission must canonicalize:

* `fabric.op` operation lists and configuration dictionaries;
* FU operand and result port order;
* mux and demux arm order;
* generated Fabric symbols;
* candidate ranking ties.

Any implementation strategy must provide deterministic ordering before
emitting externally visible IR, diagnostics, or reports.

## Policy Choices

`hw_params` are emitted as observed-value unions for every configurable
axis required by the enumerator. The required axes are owned by
`docs/spec-fabric-reconfigurable-op.md`. Omitting a needed axis is a
coverage bug because the enumerator cannot materialize candidates for
values it cannot see.

`fabric.mux` and `fabric.demux` use the mode semantics defined by the
Fabric specs and reconfigurable-op spec. Their use must preserve
software dataflow meaning and may not be used to smuggle placement,
routing, or scheduling decisions into `dataflow.subgraph`.

`dataflow.load` and `dataflow.store` are out of scope for this pass
until memory-port and memory-effect synthesis are specified. Inputs that
contain unsupported memory operations fail with `unsupported_op`.

Config options are part of the pass contract only after they are
accepted by the pass config verifier and covered by tests. A malformed
config does not mutate user Fabric IR.

## Objective Verification

The objective verification surface is:

1. Every closed failure reason has negative evidence that checks both
   diagnostics and `loom.synth_failed`.
2. Every accepted config option has focused evidence for its observable
   behavior.
3. Every strategy has positive coverage evidence and at least one
   relevant negative case.
4. Cross-strategy evidence compares coverage and semantic properties,
   not byte-identical FU text.
5. The synth -> enumerate -> match roundtrip is the end-to-end gate and
   must not be replaced by fake, stub, or hand-written coverage claims.
6. Deterministic configurations are checked by repeated runs with the
   same seed, parallelism settings, and input set.
7. Timeout and resource-exhaustion cases produce structured diagnostics
   and are not recorded as pass.
8. Generated Fabric IR is verified before it is accepted.
9. Downstream mapping, simulation, runtime, RTL, FPA, report, and DSE
   consumers observe synthesized FUs through their owning artifacts.
   Selected software-to-hardware binding remains a mapping artifact
   responsibility; unsupported scope is carried forward as structured
   evidence where implementation is incomplete.

## Related Specifications

* `docs/spec-core-dialect-boundary.md`
* `docs/spec-fabric-module.md`
* `docs/spec-fabric-pe.md`
* `docs/spec-fabric-fu.md`
* `docs/spec-fabric-reconfigurable-op.md`
* `docs/spec-fabric-hw-share-group.md`
* `docs/spec-dataflow-part-1-streaming.md`
* `docs/spec-compiler-part-3-dfg.md`
* `docs/spec-compiler-part-3-placement-framework.md`
* `docs/spec-mapping-artifact.md`
* `docs/spec-mapping-placement.md`
* `docs/spec-mapping-search.md`
