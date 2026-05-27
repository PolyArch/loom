# Loom Compiler Part 3 Placement Framework

This document defines the common placement-partition framework used by
the Loom compiler front-end. It exists because the same compiler design
problem appears at three hardware boundaries:

| Name | Input region | Placement unit | Boundary chosen |
|------|--------------|----------------|-----------------|
| L1 accelerator placement | LLVM / SCF-shaped host code | `loom.acc_region` | HostCore vs. AccCore |
| L2 graph placement | `dataflow.thread` definition's body | `dataflow.graph` definition (paired with `dataflow.graph.launch` at cut site) | ScalarCore vs. SpatialCore |
| L3 fabric placement | `dataflow.graph` definition's body | `dataflow.subgraph` | SpatialCore graph vs. `fabric.fu` granularity |

The common problem is: given a structured input region, legality
constraints, an exploration policy, and a cost model, choose a partition
into placement units. The exact policy may start simple, but the
interface must stay open to later hardware-aware, profile-guided, or
search-based policies.

Part 3 instantiates this framework immediately for L2 graph placement.
Part 2 owns the L1 instance for `loom.acc_region` selection. Fabric tech
mapping owns the L3 instance for `dataflow.subgraph` partitioning.

## 1. Core Model

A placement pass is described by five ingredients:

* **Placement problem.** The input region, the target placement unit,
  and the hardware boundary being chosen.
* **Admission constraints.** Correctness rules that decide whether an
  operation, nested region, value, effect, or boundary crossing may be
  placed in a candidate unit.
* **Candidate partition.** A full assignment of the input region's
  placeable contents into ordered placement units plus residual parent
  code.
* **Cost model.** A deterministic function that ranks legal candidate
  partitions. The baseline interface is numeric: lower score is
  preferred. Implementations may internally compute structured costs,
  but the externally visible ordering must be deterministic.
* **Exploration policy.** The algorithm that generates legal candidate
  partitions and asks the cost model to choose among them. Examples
  include source-order greedy selection, beam search, ILP-backed
  selection, and profile-guided search.

Admission constraints and the cost model are deliberately separate.
Admission decides what is legal. The cost model decides which legal
answer is preferable. A cost model must never make an illegal partition
legal, and an admission rule must not encode a performance preference
unless violating it would break correctness or the target IR contract.

## 2. Baseline Policy Requirements

Every placement instance must provide one deterministic baseline policy
that needs no profiling data and no target-specific tuning file. This
baseline is used by lit tests and by bring-up pipelines.

The baseline policy must:

* Produce the same partition for the same input IR and pass options.
* Preserve operation order and explicit memory / async dependencies.
* Emit only IR that satisfies the verifier rules of the target placement
  unit.
* Expose diagnostics for rejected required placement rather than silently
  widening a boundary.
* Record enough implementation-local information to explain the chosen
  partition in tests or debug output.

The baseline policy does not need to be performance-optimal. It is the
stable reference instance of the framework.

## 3. Candidate Partition Contract

A candidate partition is legal only when all of the following hold:

* Every placement unit satisfies the target op's verifier contract.
  For example, L2 graph placement must produce `dataflow.graph`
  definitions that are `IsolatedFromAbove` and whose bodies satisfy
  the graph whitelist, plus matching `dataflow.graph.launch` ops
  at each cut site whose ctrl/done plumbing and operand types
  resolve against the def's `function_type`.
* Every cut materializes explicit boundary operands and results
  required by the target IR. No placed region may directly use an
  SSA value from its parent scope unless the target op explicitly
  permits that use. In this milestone the placement-unit defs --
  `loom.acc_region` (L1, the temporary Part 2 to Part 3 marker),
  `dataflow.graph` def (L2), and `dataflow.subgraph` (L3) -- are
  all `IsolatedFromAbove`. `dataflow.thread`, the L1 final-form
  callable produced by Part 3 from `loom.acc_region`, is also
  `IsolatedFromAbove` and is the kernel the L1 placement instance
  hands off to (paired with `dataflow.thread.launch` ops at host
  scope or inside parent thread definitions). The L2 final form
  similarly pairs each `dataflow.graph` def with one or more
  `dataflow.graph.launch` ops inside the enclosing thread
  definition's body. The "explicitly permits" escape is therefore
  reserved for future extensions.
* Effect visibility is preserved. Any op whose execution affects program
  order, memory state, or async completion must continue to declare
  effects accurately enough for generic MLIR optimizers to preserve the
  observable behavior.
* Program order that remains semantically relevant is represented by
  SSA dependencies, memory effects, explicit control tokens, async
  tokens, or conservative barriers.
* Residual parent code is still legal for the parent execution tier.

For optional placement, unplaceable code remains in the parent tier when
that is legal. For required placement, unplaceable code is a diagnostic.

## 4. Cost Model Shape

The common cost-model interface is intentionally small:

```
score = evaluate(problem, candidate_partition)
```

The result is an ordered score value. A plain numeric score is sufficient
for the baseline implementation; future implementations may compute a
structured record such as launch count, estimated reconfiguration count,
boundary traffic, fabric resource pressure, and expected reuse. Such a
record must still define a deterministic total order for candidates that
reach a policy decision point.

Tie-breaking is part of the policy contract. If two candidates have
equal score, the policy must use a stable structural key such as source
order, placement-unit count, or a deterministic candidate id. Tests must
not depend on container iteration order.

Cost models are advisory. They may choose among legal partitions only.

## 5. Exploration Policy Shape

An exploration policy owns candidate generation and final choice:

```
candidates = enumerate(problem, admission_constraints)
legal      = filter(candidates, admission_constraints)
chosen     = argmin(legal, cost_model.evaluate, tie_breaker)
```

The pseudocode is descriptive, not a required implementation structure.
An implementation may construct and score candidates incrementally as
long as the observable chosen partition is deterministic for a fixed
input and option set.

Policies may use conservative pruning. They may not prune the only legal
candidate when placement is required. When optional placement has no
legal unit for some code, that code remains residual parent-tier code.

## 6. L1 Accelerator Placement

L1 decides which structured host-code regions become `loom.acc_region`.
Part 2 owns this instance.

The admission constraints include source intent, structured-control
recoverability, call legality, boundary operand legality, memory-region
metadata availability, and user options that require or forbid
acceleration. The cost model may later consider launch overhead,
HostCore-to-AccCore transfer volume, reuse, working-set size, and
parallel structure.

This framework does not change the Part 2 hand-off rule:
`loom.acc_region` is still the only committed accelerator boundary
consumed by Part 3, and `func.func` is not an implicit accelerator
boundary.

## 7. L2 Graph Placement

L2 decides which code inside a `dataflow.thread` definition's body
becomes a `dataflow.graph` definition + a `dataflow.graph.launch`
at the cut site. Part 3 owns this instance.

The admission constraints are the Part 3 graph verifier contract,
ScalarCore / SpatialCore boundary rules, effect visibility rules,
and the `IsolatedFromAbove` boundary materialization rule. In the
baseline implementation, L2 graph placement is source-order
greedy: it opens a graph run for a contiguous legal sequence,
closes it at a required cut, materializes the run as a
(def + launch) pair, and continues searching for the next legal
run. A richer policy may choose larger or smaller graph units
based on reconfiguration cost, graph launch frequency, fabric
pressure, graph-result traffic, or expected reuse.

`dataflow.thread.fence`, ScalarCore-only calls, illegal graph-body
ops, and parent terminators are required cuts in the baseline L2
policy. `dataflow.thread.launch` ops deserve a stronger rule:
per `docs/spec-compiler-part-3-dfg.md` §3 Constitutional Rule 2,
a thread definition's body must not directly contain both a
`dataflow.graph.launch` and a `dataflow.thread.launch` at the
same nesting level. Therefore, when L2 graph placement encounters
a thread definition's body whose direct children include any
`dataflow.thread.launch` op, the baseline policy does not open a
`dataflow.graph.launch` at that level at all; graph placement
runs only inside thread definitions whose direct children are
graph-admissible code without sibling thread launches. Equivalent
phrasings are "L2 graph placement runs on innermost thread
definitions" or "presence of a direct `dataflow.thread.launch`
at the level under consideration suppresses graph emission at
that level". The details are specified in
`docs/spec-compiler-part-3-impl.md`.

## 8. L3 Fabric Placement

L3 decides how a `dataflow.graph` definition's body is
partitioned into `dataflow.subgraph` units for fabric tech
mapping. Fabric tech mapping owns this instance.

The admission constraints include the `dataflow.subgraph` verifier,
the fabric-op support matrix, memory-op exclusion from `fabric.fu`,
acyclic or feedback legality as required by the selected fabric
strategy, and the target template library. The cost model may consider
FU utilization, mux / demux cost, reconfiguration pressure, template
coverage, and time-multiplexing opportunities.

The existing `loom-partition-graph-into-subgraphs` pass already has
multiple policy implementations and a cost model. Future work may align
its public terminology with this framework without changing the
correctness contract of existing fabric tests.

## 9. Diagnostics and Tests

Placement tests should distinguish correctness from policy preference:

* Verifier and negative tests cover admission failures.
* Baseline policy tests pin the deterministic baseline output.
* Cost-model or search-policy tests use their own fixtures and must not
  rewrite baseline expectations.
* Diagnostics should name the rejected placement instance and the
  admission rule that failed.

When a future policy intentionally changes placement quality, it should
be introduced under an explicit pass option or configuration path so
baseline tests remain stable.

## 10. References

* `docs/spec-compiler-part-2-scf.md` -- L1 `loom.acc_region` selection.
* `docs/spec-compiler-part-3-dfg.md` -- Part 3 IR contracts and L2
  graph-placement verifier constraints.
* `docs/spec-compiler-part-3-impl.md` -- Part 3 baseline L2 policy.
* `docs/spec-fabric-reconfigurable-op.md` and
  `docs/spec-generalize-subgraphs-to-fu.md` -- fabric-side L3 context.
