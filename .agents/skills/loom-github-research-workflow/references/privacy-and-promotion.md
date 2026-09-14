# Privacy And Promotion Contract

Apply this contract to every Loom research Project. Project visibility and
field definitions are live GitHub state and must be queried, not assumed.

## Private Planning Content

Keep these categories private unless the owner explicitly approves their exact
public wording:

- complete research roadmaps and Project README content;
- unpublished hypotheses, alternatives, negative results, and novelty claims;
- cross-item blocker graphs and internal scheduling;
- private comments, collaborators, dates, effort estimates, and priorities;
- evidence roots, dashboards, or attachments with non-public paths or data;
- gate discussions and decisions not yet selected for publication;
- information copied from another standalone research Project.

The existence of a public Issue does not authorize copying its surrounding
private plan.

## Public Issue Content

A promoted Issue should contain only:

- a concise public problem statement;
- the affected Loom component and current observable gap;
- stable links to owning specifications and rationales;
- the bounded implementation contract and explicit non-goals;
- acceptance evidence that can be collected publicly;
- public reproduction details and sanitized artifact identities;
- known public limitations needed for honest review.

Write the Issue so a contributor can act without access to the private
Project. Do not expose enough neighboring work to reconstruct the hidden
roadmap.

## Approval Snapshot

The preview must show the full rendered public payload and every mutation in
one approval snapshot. Approval is invalid when any of these change:

- authenticated account;
- repository, Project, item, base branch, or head branch;
- title, body, comment, labels, assignees, or milestone;
- Project field names, option values, or work-item key;
- conversion, replacement, archive, close, merge, or push behavior.

Read back the live object after mutation. A successful CLI exit alone is not
enough.

## Promotion Paths

### In-Place Conversion

Use only when the private draft title and body are already the exact approved
public payload. Query the live repository ID and item ID, preview the conversion
mutation, obtain approval, convert, then verify the Issue URL and preserved
Project fields.

### Redacted Replacement

Use when the draft contains private detail:

1. Preview the sanitized Issue and complete field-copy/archive transaction.
2. Create the Issue after approval.
3. Add it to the same private Project.
4. Copy only approved field values from the draft.
5. Verify the Issue item, stable work-item key, and linkage.
6. Archive the old draft so it remains recoverable but is no longer active.

If any operation fails, report the partial state exactly. Do not delete or hide
the evidence and do not retry with a broader mutation.

## Pull Requests And Evidence

- The Issue is the public work contract; the pull request implements it.
- A Draft PR may link the Issue, but a closing keyword is appropriate only when
  merging the complete PR should close that Issue.
- Keep exact test commands, public artifact references, and limitations in the
  pull request.
- Keep unpublished research interpretation and aggregate roadmap status in the
  private Project.
- Project automation may derive status from Issue or PR events, but verify the
  resulting live fields before relying on them.
