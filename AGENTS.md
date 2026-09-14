# Loom Agent Development

This file defines repository development policy for coding agents. It does not
define Loom product semantics. Product contracts remain in `docs/`.

## Authority

- Start at [`docs/README.md`](docs/README.md). Tracked `docs/spec-*.md` files
  own normative WHAT. [`docs/spec-loom-stack.md`](docs/spec-loom-stack.md) is
  the full-stack entry point.
- [`docs/rationales/README.md`](docs/rationales/README.md) maps specifications
  to non-normative WHY documents. A rationale explains a choice but never owns
  schemas, behavior, defaults, or validation rules.
- Source code owns HOW and must conform to the current specifications.
- A coherent specification may intentionally lead its implementation. Do not
  delete or weaken it merely because HOW is incomplete.
- Real evidence may reveal a defective specification. In that case, repair the
  owning specification and rationale first; never create a code-only alternate
  contract.
- Private GitHub Projects own research roadmaps, hypotheses, and progress.
  Public Issues own publishable implementation work, and pull requests own the
  corresponding review and delivery record. None of these override product
  semantics in `docs/`.
- `temp/` is ignored scratch space. Nothing required to understand, implement,
  review, or reproduce Loom may depend on it.

## Architecture Orientation

Loom is a full-stack compiler and hardware framework for heterogeneous,
multicore spatial acceleration. Its stable conceptual flow is:

```text
source programs
  -> structured and Dataflow artifacts
  -> Fabric and ADG hardware artifacts
  -> Tech, Spatial, and System Mapping
  -> simulation, runtime, RTL, and physical-tool evidence
  -> Evaluation and design-space exploration
```

The compiler frontend, hardware frontend, Mapping, simulation backends,
hardware backend, and Evaluation/DSE are the six semantic components described
under `Full-Stack Components` in `docs/spec-loom-stack.md`. Follow links from
the documentation entry points to learn exact interfaces. Do not copy current
schemas, algorithms, presets, version numbers, or capability counts into this
file or an agent skill.

## Spec-Driven Changes

Architecture, IR, schema, artifact, cross-component, and externally visible
semantic changes require one closed normative owner and its rationale. When a
decision changes, commit the selected WHAT and WHY without product HOW, then
implement HOW in later commits. If existing WHAT and WHY are already
sufficient, reference them and avoid decorative documentation churn. Stop
implementation while an essential semantic owner remains unresolved.

Repository procedures for this work live under `.agents/skills/`. Use the
matching skill instead of recreating a parallel checklist in a plan or Issue.

## Engineering Rules

### Distilled Foundations

- Introduce a concept only when it represents an essential distinction that
  cannot be derived from an existing owner.
- Give every fact, rule, schema, configuration value, identity, and state
  transition one semantic owner. Derived views and caches must identify their
  source and validation rule.
- Prefer composition and stronger invariants over special cases, fallback
  ladders, compatibility aliases, generic property bags, and caller-maintained
  shadow state.
- Simplify the affected dependency cone after behavior is proven. Remove
  obsolete paths completely, but first check dynamic consumers, registries,
  build and link ownership, serialized contracts, entry points, and external
  integrations.
- Do not classify a missing required capability, typed failure distinction,
  independent oracle, or durable diagnostic as slop.

### Types And Diagnostics

- Represent closed internal domains with typed enums or tagged types. In C++,
  use `enum class`. Convert strings only at parsing, serialization, logging,
  and display boundaries through one canonical mapping.
- Give repeated domain-significant values semantic names under their owner.
- Keep reusable diagnostics quiet by default and runtime-configurable. Remove
  case-specific probes after the investigation.
- Treat schema, protocol, cache-key, and artifact-identity changes as one
  semantic change. Never reuse an old identity for new meaning.

### Tests And Evidence

- Tests are evidence, not the default development process. Understand the
  contract and implement a coherent workflow before deciding which tests are
  worth retaining.
- Follow any conformance-anchor ordering required by the owning specification.
  This does not imply universal per-function or fixture-first development.
- Commit a test only when it is necessary, reusable, likely to catch a
  plausible regression, and non-trivial.
- Prefer realistic inputs, production entry points, exact artifacts, strict
  import, typed failures, and independent oracles over mocks and wrappers.
- A green narrow test, an empty artifact, a generated shell, a dummy trace, a
  feature-gated test, or `Unsupported` cannot prove a broader capability.
- Distinguish `Unsupported`, invalid input, proven infeasibility, budget
  exhaustion, execution failure, adverse evidence, and success.
- Give expensive searches and tools explicit CPU, memory, concurrency, and
  timeout budgets. Diagnose timeouts before increasing them.

### Files And Modules

- Keep tracked files English-only and free of Emoji. Scratch files under
  `temp/` are exempt.
- Use semantic anchors such as symbol and heading names in documentation and
  plans. Do not cite fragile exact line ranges.
- Avoid large production-code blocks in specifications and plans. Use
  pseudocode and concise interface signatures.
- Review a code file's responsibility at 2,000 lines. Files between 2,000 and
  4,000 lines require a behavior-preserving modularization review after the
  coherent implementation works. No code file may exceed 4,000 lines.
- Split modules by cohesive responsibility, not with arbitrary fragments,
  `.inc` files, or pass-through wrappers used only to reduce line count.

## Workspace And Tools

- Use `rg` and `rg --files` for search. Prefer structured parsers for
  structured data.
- Every command must reduce uncertainty, produce evidence, or perform an
  authorized action. Do not run filler probes or repeatedly inspect unchanged
  Git state.
- Use the current worktree for the assigned owner only. Do not modify, pause,
  merge, or clean another active worker's worktree without authorization.
- Require an explicit linked-worktree target path from the current user's
  request or accepted task. If it is absent, inspect live topology but do not
  invent a default parent. Never encode a maintainer-specific home directory
  or organization-internal layout in repository policy. Validate explicit
  paths before any destructive or recursive operation.
- Prefer `temp/` for experiments and transient logs. Avoid `/tmp` for large
  artifacts because the root filesystem may be small.
- When a development or EDA tool is absent, use an existing `module` command
  to inspect `module avail`. If `module` is unavailable but
  `/etc/profile.d/modules.sh` exists, source that script first. Load a selected
  module explicitly before declaring the tool unavailable. This is interactive
  operator discovery only; product tool resolution must follow
  [`docs/spec-external-tool-invocation.md`](docs/spec-external-tool-invocation.md)
  and must not parse presentation-oriented `module avail` output.
- Discover required libraries from repository documentation, build
  configuration, the active environment, or user input. Never encode
  organization-internal storage or site-local library paths in tracked policy.
- Use `make doctor` for repository build-path preflight. Use the Makefile entry
  points rather than duplicating worktree build logic.
- For long-running tmux work, send text without Enter, inspect it with
  `capture-pane`, wait one second, and send Enter separately.

## Git And GitHub

- Preserve unrelated user changes. Never use destructive Git operations to
  recover a convenient local state.
- Keep commit messages and pull request bodies in English, without CJK or
  Emoji. Do not add automated-tool attribution or names to commits, pull
  requests, or code comments.
- Do not encode development bookkeeping such as numbered progress stages or
  completion labels in code, comments, commit messages, or pull request text.
- Develop publishable work through a public Issue, an isolated branch and
  worktree, and a pull request. Link a pull request to the Issue it actually
  completes.
- Keep each research Project standalone. A work item may use shared Loom
  components but must not depend on another research Project's roadmap.
- Keep research Projects private. Public Issues, pull requests, and code must
  contain only material approved for disclosure.
- Before any GitHub remote write, show the exact target, public payload, and
  field changes. Execute only after explicit approval, then read the live state
  back. This includes Issue, pull request, comment, label, Project, merge, and
  push operations.
