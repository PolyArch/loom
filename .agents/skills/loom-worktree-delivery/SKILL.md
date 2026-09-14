---
name: loom-worktree-delivery
description: Create, use, synchronize, review, integrate, publish, or safely clean an isolated Loom linked worktree for a bounded implementation task. Use after WHAT and WHY are closed, when starting work from an Issue, coordinating non-overlapping parallel owners, recovering worktree state, or preparing a branch and pull request. Do not use to invent unresolved architecture or to manipulate another active worker's state.
---

# Loom Worktree Delivery

Deliver one coherent semantic owner slice through one branch and one linked
worktree. Keep topology and state explicit; use existing repository entry
points for build and synchronization behavior.

Read [the delivery checklist](references/delivery-checklist.md) before creating,
integrating, or cleaning a worktree.

## Confirm The Authorized Mode

- **Preparation**: inspect and propose topology, ownership, evidence, and stop
  conditions. Do not create or mutate worktrees, refs, or remote state.
- **Local delivery**: an explicitly approved task permits local worktree,
  branch, implementation, verification, and commits. It does not permit a
  remote write. A public Issue may remain pending while its exact payload is
  awaiting approval.
- **Publication**: require the accepted public Issue and the exact remote-write
  approval required by `$loom-github-research-workflow` before push or pull
  request creation.

## Establish The Delivery Contract

Require:

- a public Issue for publication, or an explicitly approved task for local
  delivery;
- the exact governing specification and, when applicable, rationale revision;
- a bounded semantic owner and affected dependency cone;
- non-overlapping worktree ownership;
- acceptance evidence and known expensive tools;
- the base reference, integration owner, and publication target.

If product semantics are unresolved, use `$loom-spec-driven-change` and stop
implementation. Do not ask a builder to choose architecture implicitly.

## Create Or Adopt A Worktree

1. Inspect live worktree topology and resolve the repository common directory,
   base reference, branch, and explicit target path.
2. Require the exact target path from the user's request or accepted task. If
   it is absent, stop before topology mutation rather than assuming a
   maintainer-specific home directory or organization-internal layout.
3. Validate that the target path is absent and narrow, the base is current, and
   the branch is not owned by another worktree.
4. Create one branch for one task. Do not share a writable worktree among
   agents.
5. Run `make doctor` in the linked worktree before relying on build paths or
   shared externals.

Use normal `git worktree` operations for topology. Use the repository Makefile
and scripts for Loom-specific build identity, shared external tools, and linked
branch synchronization. Do not reimplement those mechanisms in this skill.
Process listings are evidence of activity, not ownership authority. Require an
explicit owner acknowledgment when live topology or activity is ambiguous.

## Implement And Verify

1. Re-read the Issue and governing documentation from the adopted revision.
2. Inspect the current implementation before assuming a historical gap still
   exists.
3. Implement the complete owner slice and remove superseded paths in the same
   dependency cone.
4. Exercise the realistic use path, then retain only qualifying tests.
5. Run `$loom-evidence-audit` at the scope needed by the claim.
6. Review the final diff for unrelated changes, hidden compatibility, private
   paths, generated artifacts, and documentation drift.
7. Commit coherent changes in English. Product WHAT/WHY commits must precede
   product HOW commits.

## Synchronize And Integrate

- Recheck cleanliness, ancestry, active owners, and the actual commit delta
  immediately before integration.
- In local-delivery mode, use `make sync-worktree` only from a linked worktree
  and only when synchronization is within the approved task. Inspect its
  documented preflight result before accepting any update. Do not run it in
  preparation mode.
- Resolve conflicts by semantic owner. Never use whole-file conflict choices
  where both sides contain independent valid behavior.
- Rebuild and re-run risk-proportionate evidence on the combined revision.
- Prove the delivered commits are reachable from the intended target before
  considering cleanup.

## Publish And Clean

Before push, Issue edits, pull request creation, merge, or Project updates, use
`$loom-github-research-workflow` and obtain approval for the exact remote
payload.

Cleanup is a separate destructive action. Remove a linked worktree or branch
only when the user requests it and live checks prove that it is inactive,
clean, fully integrated, and not the sole owner of any commit or artifact.
Prefer a recoverable operation when practical.

## Required Output

Report the applicable Issue or local authorization, docs revision, worktree
path, branch, semantic owner, commit delta, verification evidence, integration
target, publication URLs, and any state intentionally left for another owner.
Mark artifacts that were not created or published as such instead of inventing
placeholder state.
