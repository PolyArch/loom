---
name: loom-context-recovery
description: Recover the current Loom task from a Codex rollout, fork, compaction, handoff, tmux session, interrupted process, or long-running worktree. Use when asked to continue or resume work, audit a large transcript, distinguish inherited history from native user intent, reconcile completion claims with live Git and GitHub state, or prepare a trustworthy handoff. Recovery is read-only until ownership and mutation authority are re-established.
---

# Loom Context Recovery

Recover provenance first, then reconcile every historical claim with live
authorities. A transcript is evidence about past actions, never current project
state by itself.

## Extract Rollout Provenance

For a Codex JSONL rollout, run:

```bash
python3 .agents/skills/loom-context-recovery/scripts/extract_rollout_context.py \
  path/to/rollout.jsonl --format markdown
```

If the rollout metadata contains `forked_from_id`, locate the parent JSONL and
pass it explicitly:

```bash
python3 .agents/skills/loom-context-recovery/scripts/extract_rollout_context.py \
  path/to/fork.jsonl --parent path/to/parent.jsonl --format markdown
```

The script validates the parent identity and separates imported history through
a normalized record prefix. If the parent continued after the fork, automatic
recovery also requires a matched terminal turn followed by distinct parent and
child turn identities. It fails closed instead of inferring a boundary from
timestamps. Pass `--native-start-line` only after independently verifying the
boundary. Use `--format json` for machine-readable output and `--full-messages`
only when the extra sensitive context is necessary.

Read [the recovery contract](references/recovery-contract.md) before
interpreting mixed provenance or publishing a handoff.

## Recover The Current Contract

1. Identify the latest native human instruction. Keep internal goal injection,
   compaction replacement history, subagent relay, and imported parent history
   separate.
   Treat unmarked user-role relays as ambiguous; the extractor cannot infer a
   sender from prose alone.
2. State the active objective, scope, owner, worktree, mutation authority, and
   return condition for any nested task.
3. Inspect the live repository revision, uncommitted changes, worktree owners,
   running processes, canonical docs, public Issue, and private Project item as
   applicable.
4. Verify historical commit, artifact, test, and completion claims against that
   live state. Discard stale counts, paths, tool versions, and plans.
5. Classify each requested outcome as committed, verified, in flight, missing,
   contradicted, or no longer requested.
6. Select the next coherent semantic boundary. Do not resume an inherited or
   superseded task merely because it appears in a summary.

## Preserve Ownership

- Remain read-only until the current worktree owner and requested mutation are
  clear.
- Do not message, pause, merge, clean, or delete another worker's state without
  authorization.
- Do not restart an apparently stalled process until its live process, log,
  output root, and progress have been inspected.
- Do not launch a duplicate build or tool invocation against the same output
  directory.
- Treat `temp/` handoffs and transcripts as recovery clues, not WHAT, WHY,
  research planning, or completion authority.

## Produce A Recovery Card

Report:

- active session and parent chain;
- latest effective native instruction;
- current Issue or Project work item;
- live revision, worktree, owner, and dirty-state summary;
- verified completed artifacts and evidence;
- active processes and incomplete gates;
- stale or rejected inherited claims;
- the next bounded action and any required approval.

Do not claim success from a compacted summary, an old commit hash, a worker
report, an interrupted test, or a local slice that does not satisfy the active
objective.
