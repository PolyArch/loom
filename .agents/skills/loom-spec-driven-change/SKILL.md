---
name: loom-spec-driven-change
description: Define and deliver Loom changes from canonical WHAT and WHY before implementing HOW. Use for architecture, IR, schema, artifact, identity, Mapping, simulation, runtime, hardware, Evaluation, or other cross-component semantic changes; for suspected spec/code conflicts; or when a review exposes an undefined or duplicated semantic owner. Do not use for a behavior-preserving local edit whose existing contract is already clear.
---

# Loom Spec-Driven Change

Use tracked documentation to close the semantic contract before changing its
implementation. Treat the current specification as implementation authority,
but let reproducible evidence challenge and improve it at its owner.

## Confirm The Authorized Mode

Choose the mode from the user's request before changing state:

- **Design or review**: inspect, classify, and report. Do not edit or commit.
- **Documentation change**: update and commit WHAT/WHY only. Stop before HOW.
- **Full change**: update and commit WHAT/WHY when needed, then implement HOW in
  later commits.

A review that discovers a defect does not authorize a fix. A request to
implement a semantic change authorizes the full workflow unless the user
narrows it.

## Establish The Authority

1. Read the repository `AGENTS.md` and `docs/README.md`.
2. Follow `docs/spec-loom-stack.md` and `docs/rationales/README.md` to the
   narrowest normative owner and its rationale.
3. Inspect the live implementation, conformance anchors, and affected
   producers and consumers. Treat plans, transcripts, Issues, and `temp/` as
   context rather than product authority.
4. Read [the authority checklist](references/authority-checklist.md) when the
   change crosses an artifact, identity, schema, or component boundary.

## Classify The Change

First require an observable distinction, intended semantic owner, and affected
consumer. If the proposal does not define them, report it as underspecified and
stop before a broad implementation audit.

Choose exactly one primary classification for the proposed change:

- **Implementation lag**: the current specification is closed and correct,
  but HOW does not conform.
- **Specification gap**: an essential behavior has no complete normative
  owner.
- **Specification defect**: a reproducible counterexample or architectural
  contradiction invalidates the current contract.
- **Non-semantic implementation change**: behavior and public contracts remain
  unchanged. Route a clearly local implementation task to the normal
  repository workflow rather than continuing this skill.
- **No semantic change justified**: the current contract already represents
  the requested distinction, or the proposal would add a duplicate owner.
  Reject the proposed change and preserve the existing contract.

Do not edit documentation merely to narrate implementation. Do not call an
unimplemented but coherent contract a defect.
Classify additional inconsistencies discovered during inspection separately;
they do not change the primary classification or authorize a repair.

## Close WHAT And WHY

For a gap or defect in documentation-change or full-change mode:

1. State the problem with one concrete positive example and one counterexample.
2. Trace ownership, typed inputs and outputs, identity, ordering, validation,
   failure behavior, version boundaries, and downstream consumers only where
   the change affects them.
3. Compare the smallest viable designs. Prefer an existing owner or a derived
   view over a new entity.
4. Update one current contract in the owning `spec-*.md` and explain the reason
   in the corresponding rationale. Remove superseded alternatives from the
   normative surface.
5. Verify links, terminology, internal consistency, and implementability.
6. Commit WHAT and WHY without product HOW changes.

In design or review mode, report the same analysis without editing. If a
high-impact choice remains genuinely unresolved, present the evidence and
alternatives and stop before implementation.

## Make HOW Conform

After the documentation commit exists, or when the existing contract was
already sufficient:

1. Record the exact documentation revision and public Issue that govern the
   work.
2. Implement the complete affected owner slice in a later commit.
3. Migrate all consumers and delete any superseded parser, codec, cache,
   fallback, alias, fixture, or test that no longer has an owner.
4. Exercise the production path with realistic input.
5. Retain only tests that protect fragile semantic joints.
6. Use `$loom-evidence-audit` before making a capability or completion claim.

## Required Output

Report the applicable subset of:

- the normative and rationale owners;
- the change classification;
- the positive example and counterexample for a gap or defect;
- the selected contract and rejected shadow authorities;
- the WHAT/WHY commit and later HOW commits when changes were authorized;
- exact evidence and remaining unsupported boundaries.

Never report the change as complete while WHAT, WHY, HOW, or required evidence
still disagree.
