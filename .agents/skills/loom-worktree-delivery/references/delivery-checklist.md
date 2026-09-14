# Delivery Checklist

Use this checklist at topology-changing and publication boundaries. Do not run
Git inspection repeatedly when state has not changed.

## Preflight

- Resolve the repository root and common Git directory.
- Inspect `git worktree list --porcelain` for branch and path ownership.
- Verify the exact base reference and intended branch ancestry.
- Check the target worktree and integration worktree for local changes.
- Check for an active process or agent that owns either worktree.
- Treat process discovery as evidence only; obtain explicit acknowledgment
  when ownership is not already established by the task.
- Validate the explicit target path; do not use broad directories, home
  shortcuts, unresolved variables, or globs for destructive commands.
- Run `make doctor` after the linked worktree exists.

## Ownership Contract

- One writable worktree has one active owner.
- Parallel slices must have non-overlapping semantic owners or an explicit
  integration dependency.
- Leaf worktrees deliver to their declared integration owner rather than
  bypassing it.
- Submodules and shared external build roots remain under their repository
  owner.
- Stashes are short-lived recovery tools, not cross-worktree storage or
  handoff records.

## Implementation Review

- Confirm the historical gap still exists on the current base.
- Confirm source changes implement the governing specification revision.
- Inspect direct and dynamic consumers before deleting old behavior.
- Check build registration, link ownership, command entry points, schemas,
  external tools, and generated artifacts.
- Keep site-local paths, commercial data, logs, and ignored outputs untracked.
- Run focused evidence before broad or expensive gates.

## Integration Review

- Recompute ancestry and commit deltas after the final local commit.
- Identify semantic overlaps before applying commits.
- Preserve the newer shared contract and both branches' independent valid
  capabilities.
- Run evidence at the combined revision, not only at an isolated source tip.
- Verify the remote result after an approved push.

## Cleanup Review

- Require explicit cleanup authorization.
- Confirm no active owner or process remains.
- Confirm the worktree is clean and its commits are reachable from the retained
  target.
- Confirm no untracked artifact is the only copy of required evidence.
- Remove only the explicit linked worktree and branch; never use a repository
  root or broad parent directory as a recursive target.
