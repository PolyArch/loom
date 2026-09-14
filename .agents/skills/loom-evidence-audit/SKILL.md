---
name: loom-evidence-audit
description: Audit Loom capability, correctness, performance, completion, artifact-integrity, corpus, Mapping/PnR, simulation, runtime, RTL, EDA, cache, and end-to-end claims using exact live evidence. Use for read-only capability reviews, implementation acceptance, failing or timed-out workflows, unsupported features, conflicting test totals, or any request to prove what Loom can currently do. Do not use to change product semantics; route semantic defects through loom-spec-driven-change.
---

# Loom Evidence Audit

Match every claim to evidence of the same scope and fidelity. Prefer exact
artifacts and realistic execution over summaries, wrappers, and test counts.

## Confirm The Authorized Mode

- **Inspection-only**: read existing source, docs, artifacts, logs, and live
  state. Do not execute workloads, populate caches, invoke external tools, or
  write reports.
- **Evidence execution**: run the bounded commands and tools authorized by the
  user. State expected cost, restricted dependencies, and output locations
  before expensive or commercial-tool work.
- **Implementation acceptance**: inspect the change and run risk-proportionate
  evidence, but do not repair findings unless the user also requested fixes.

A request for review, audit, status, or diagnosis defaults to inspection-only
unless it explicitly requests execution. Never transition from a finding to a
source or documentation edit without separate authorization.

## Define The Claim

1. State the capability or completion claim narrowly enough to falsify.
2. Resolve the current Git revision, semantic configuration, input artifact
   identities, tool identities, and requested fidelity.
3. Locate the normative conformance boundary through `docs/README.md`.
4. List the producers, independent verifiers or oracles, and terminal outcomes
   required by that boundary.
5. Read [the domain evidence checks](references/evidence-checks.md) for the
   affected workflow.

Do not mutate source during a requested audit or diagnosis. If the user asks
for implementation, finish the audit classification before editing.

## Build The Evidence Matrix

Use one row per independently falsifiable claim:

| Claim | Required evidence | Observed evidence | Identity | Terminal outcome | Verdict | Limits |
|---|---|---|---|---|---|---|

Use only these claim verdicts:

- **Proven**: evidence directly establishes the exact claim.
- **Contradicted**: a valid counterexample refutes the claim.
- **Unresolved**: available evidence establishes neither result.

Record the exact typed terminal outcome owned by the current workflow. Examples
include success, adverse completed evidence, unsupported semantics, invalid
input, proven infeasibility, budget exhaustion, execution failure,
unavailability, interruption, and not attempted. These examples are not a new
product enum; use the canonical owner's current terms. Do not collapse terminal
outcomes into a claim verdict.

## Inspect Or Collect Evidence

In inspection-only mode, evaluate only evidence that already exists. Record
unattempted execution as a limit and stop when new execution would be required
to resolve the claim.

In evidence-execution or implementation-acceptance mode:

1. Validate artifact closure, exact references, resolved configuration, and
   tool readiness before expensive execution.
2. Run the smallest realistic production path that can prove or refute the
   claim. Inspect its artifacts and trace rather than only its exit status.
3. Compare against an independent reference or exact verifier. A producer
   cannot be its own oracle.
4. Use a negative control or near-neighbor to prove that the gate distinguishes
   real behavior from unconditional success.
5. Expand from a representative case to a bounded cohort only after the stage
   and failure taxonomy are established.
6. Record exact commands, semantic environment, identities, time and resource
   budgets, terminal state, and raw evidence under an ignored experiment
   directory.
7. Re-run risk-proportionate evidence at the final combined revision.
8. Preserve evidence required for review or reproduction through its canonical
   artifact owner or an approved durable report or evidence store. A `temp/`
   path alone is never acceptance evidence.

For implementation acceptance, inspect the affected dependency cone and run
focused anchors before broader suites. Run commercial or expensive EDA only
when the claim requires that fidelity, the affected owner changed, or the user
explicitly requests it.

## Guard Evidence Integrity

- Do not count empty graphs, skipped stages, dummy traces, X-filled shells,
  universal resources, stale artifacts, feature-gated tests, or mocks as the
  real capability.
- Do not widen a narrow fixture or minimal workload into a corpus, application,
  or full-stack completion claim.
- Do not treat a timeout as infeasibility or an interrupted test as a pass.
- Do not raise timeouts before checking capacity, identity, stage accounting,
  isolated execution, resource contention, and algorithmic progress.
- Require exact integer, address, control, ordering, and non-floating memory
  behavior. Allow floating differences only when their semantic provenance is
  established by the owning contract.
- Verify cold and warm cache behavior separately when cache correctness or
  performance is claimed. A warm result must prove the expensive tool was not
  invoked.
- Rebuild stale readiness or derived outputs through their owner rather than
  editing reports.

## Report The Result

Lead with contradictions and missing evidence. Include the evidence matrix,
commands or machine-readable reports, artifact roots, identities, resource
limits, and residual unsupported boundaries.

Recommend `$loom-spec-driven-change` when evidence demonstrates that the
current normative contract is defective. Invoke it in design mode unless the
user separately authorized documentation or implementation changes. Use
`$loom-context-recovery` before trusting claims recovered from a long or
interrupted session.
