---
name: loom-github-research-workflow
description: Manage Loom research through private GitHub Projects and publish approved implementation work through public Issues and pull requests. Use to inspect or update a research Project README or fields, promote a private draft item, create or triage an Issue, link a branch or pull request, record evidence or gate decisions, or reconcile Project status after review or merge. Require an exact preview and explicit approval before every GitHub remote write.
---

# Loom GitHub Research Workflow

Keep private research planning complete while exposing only approved,
actionable implementation work in the public repository. Discover live GitHub
objects and schemas on every use; never encode a Project number or field ID as
repository truth.

Read [the privacy and promotion contract](references/privacy-and-promotion.md)
before drafting any public payload or changing a Project item.

## Resolve Live State

1. Confirm the authenticated GitHub identity and required scopes.
2. Resolve the repository, owner, default branch, Issue, pull request, and
   linked Projects from the live API or an explicit URL.
3. Verify the selected Project is private, open, linked to the intended Loom
   repository, and is the exact standalone research project requested.
4. Query the Project README, items, field definitions, option names, and
   current workflows. Resolve IDs only for the pending operation.
5. Read the public Issue or pull request and current Git revision before
   trusting a private draft's implementation status.

Prefer `gh` and GitHub GraphQL for semantic operations. Use a signed-in browser
only when the required operation has no suitable API or CLI surface.

## Separate The Authorities

- Keep research thesis, hypotheses, alternatives, roadmap, dependencies,
  unpublished evidence, gate decisions, and full progress in the private
  Project.
- Keep the publishable implementation problem, governing Loom specs, bounded
  work, acceptance evidence, and public discussion in the Issue.
- Keep HOW, review, and exact verification results in commits and the pull
  request.
- Update product WHAT and WHY in `docs/` before implementation when the
  research item changes Loom semantics.
- Keep each research Project self-contained. Do not point one Project at
  another Project's roadmap or work item.
- Keep shared repository infrastructure outside all research Projects unless a
  user explicitly assigns it to one.

## Preview Every Remote Write

Before any push, Issue, pull request, comment, label, assignment, close, merge,
Project, workflow, or field mutation, show:

- authenticated identity and exact repository or Project URL;
- operation type and target IDs or URLs;
- complete public title and body;
- labels, assignees, milestone, base, head, and linkage changes;
- Project fields and before/after values;
- private source material deliberately omitted;
- the exact command or GraphQL mutation class to be used.

Pause for explicit approval. Approval applies only to that payload and target.
If live state or the payload changes, preview again. After execution, read the
object back and report its URL and actual fields.

## Promote A Private Draft

1. Resolve the draft item and live `Public Promotion` field. Stop unless its
   value is exactly `Approved`.
2. Draft a self-contained public Issue using only approved information and the
   current Loom documentation owners.
3. Compare the exact draft title and body with the approved public payload.
4. If they are identical and safe to disclose, preview the in-place
   `convertProjectV2DraftIssueItemToIssue` operation. This preserves the
   Project item and its field values.
5. If private material must be omitted, preview a safe replacement transaction:
   create the public Issue, add it to the same Project, copy approved field
   values, verify the new item, then archive the original draft. Never delete
   the original automatically.
6. Set or preserve a stable work-item key when the live Project provides one.
   Do not copy private blocker graphs, comments, dates, or evidence links into
   the public body.
7. Read back the Issue and Project item before reporting promotion success.

If a required gate or field is absent or ambiguous, fail closed and request a
Project-owner decision rather than inventing a fallback.

## Deliver Through Issue And Pull Request

- Use the repository Issue forms when creating a public bug or change item.
- Link the Issue to the governing specification and rationale by stable path
  and heading, not exact line ranges.
- Use `$loom-worktree-delivery` after the public work contract is accepted.
- Preview the remote branch push and complete Draft PR payload together.
- Link the pull request to the Issue it will actually complete. Do not use a
  closing keyword for partial work.
- Keep the pull request body public, English-only, and free of private Project
  content or automated-tool attribution.
- After review, merge, or new evidence, preview the exact Project status,
  evidence, link, and gate-decision updates before applying them.
- Do not infer research completion from a merged implementation pull request.

## Required Output

Report the private Project URL to authorized collaborators, public Issue and
pull request URLs, governing docs revision, Project item key, executed field
changes, omitted private categories, and any remaining review or evidence gate.
