# Engineering Methodology Rationale

Normative product and verification boundaries are owned by
[Loom Full-Stack Architecture](../spec-loom-stack.md) and each subsystem's
conformance-anchor section.

## Distilled Complexity

Loom is intrinsically complex because it spans compiler IR, configurable
hardware, PnR, simulation, runtime, and EDA. Occam's Razor is applied to the
conceptual surface rather than line count: complete behavior should emerge
from a small set of orthogonal owners and composable invariants.

The practical test for a new type, state, record, operation, or mode is whether
it represents a fact that cannot be derived from an existing owner. Generic
paths, property bags, duplicate status fields, one-use wrappers, and
compatibility aliases usually preserve accidental implementation shape rather
than essential distinctions. Strengthening an invariant or deleting a second
authority is preferred to adding reconciliation logic.

## Anchor-Level TDD

Tests are executable contracts for stable semantic boundaries. The most
valuable anchors protect canonical identity, exact cross-artifact coupling,
state transitions, ordering, failure classification, and real end-to-end
behavior. They should fail when a future change reintroduces a semantic bug.

The project deliberately avoids fixture matrices, printer snapshots, wrapper
tests, broad mocks, and duplicated assertions for every enum combination.
Those tests make a rapidly changing implementation look complete while making
valid refactoring expensive. A stable invariant deserves TDD; an unstable
container layout does not.

## Single Source Of Truth

SSOT is enforced per fact, not by putting all code in one module. Dataflow owns
software execution, Fabric owns hardware structure and capability, Mapping
owns selected software-to-hardware relations, ResolvedConfig owns semantic
configuration, EvaluationEvidence owns observations, and projections own no
new facts.

Derived native caches, configured views, reports, bitstreams, and generated
backend files are useful only when their derivation and validation against the
owner are explicit. When two representations disagree, the remedy is to remove
or regenerate the secondary representation, not to add a tie-breaking rule.

## Honest Incompleteness

Missing providers, unsupported semantics, invalid input, execution failure,
budget exhaustion, and adverse completed evidence are different outcomes.
Scaffolding, an empty artifact, an X-filled RTL stub, a wrapper exit code, or a
test status file cannot claim work that was not performed.

This rule is particularly important across fidelity levels. A fast estimate is
useful only when labeled as an estimate. It cannot silently stand in for CGRA
execution, RTL simulation, physical implementation, or measured evidence.

## Determinism And Performance

Semantic determinism is based on exact code/build identity, resolved semantic
configuration, immutable input artifacts, model identities, and explicit
deterministic work. Host parallelism, wall time, licenses, paths, and cache
state may prevent completion but cannot choose a different formal result.

Performance is still a correctness concern for search-heavy components. Dense
indices, cache-friendly derived views, pruning, incremental recomputation, and
parallelism are useful implementation techniques when the owning descriptor
and deterministic protocol admit them; they are not universal evaluator or
schema requirements. A full-only evaluator remains valid when it truthfully
declares that interaction surface. A functional microbenchmark is not enough
when its per-action cost compounds across a large search. Timeouts are
diagnostics for profiling and algorithm review, not defaults to increase
mechanically.

## Continuous Slop Removal

Implementation should be simplified after behavior is proven. Stale dual
paths, repeated semantic strings, test-only artifact builders, wrapper-owned
pipelines, hidden defaults, defensive states made impossible by validation,
and obsolete fixtures are removed rather than documented indefinitely.

This does not mean deleting missing capabilities from the roadmap. Slop is
accidental machinery; an unimplemented but architecturally required compiler,
mapper, simulator, or backend capability remains real work.
