# Authority Checklist

Read this checklist for changes that cross semantic owners or durable artifact
boundaries. Resolve exact product facts from the current tracked
specifications; do not copy them into this reference.

## Locate The Owner

- Start with `docs/README.md` and the `Authority Map` in
  `docs/rationales/README.md`.
- Identify the fact being changed, not merely the file or API being edited.
- Find its authoring owner, immutable artifact owner, derived views, and every
  consumer that can observe the change.
- Determine whether an existing owner can derive the requested fact.
- Reject generic bags, name-keyed tables, caller-maintained lists, and cached
  reports that would become competing authorities.

## Test The Contract

- Give one ordinary case that must work.
- Give one near-neighbor that must be rejected or behave differently.
- State which owner distinguishes the two cases.
- Check ordering, completion, memory effects, liveness, identity, and failure
  behavior only where observable or contractually required.
- Distinguish semantic determinism from host timing, paths, licenses, and cache
  state.

## Durable Artifact Changes

When an artifact or protocol changes, inspect the complete path:

```text
authoring form
  -> canonical payload
  -> semantic identity
  -> strict import and validation
  -> publication
  -> exact consumer reference
```

- Ensure presentation metadata, source order, host paths, and private names do
  not alter identity unless the specification says they are semantic.
- Update schema or protocol identity when the same serialized value would
  otherwise acquire new meaning.
- Reject foreign, wrong-kind, stale, and out-of-range references through the
  owning typed failure model.
- Keep caches and reports derived. They must not define a second wire format or
  terminal state.

## Commit Boundary

The documentation commit may contain specifications, rationales, and
documentation-only conformance definitions. It must not contain product source
or test changes that implement the new behavior.

Later implementation commits must cite the governing documentation revision.
If implementation evidence reveals another contract defect, stop, amend the
owner in a new documentation commit, and then resume HOW.

## Simplification Review

After consumers migrate, inspect the dependency cone for:

- old codecs, parsers, aliases, flags, and fallback paths;
- duplicated validation or status fields;
- test-only production hooks and fixtures;
- reports or caches treated as final authorities;
- defensive states made impossible by the new invariant.

Remove obsolete machinery only after checking direct and dynamic consumers,
registries, build and link ownership, command entry points, serialized
contracts, and external integrations.
