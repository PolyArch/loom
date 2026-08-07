# Loom Design Documentation

Every tracked `spec-*.md` file is normative. The top-level entry point is
[Loom Full-Stack Architecture](spec-loom-stack.md), which defines the stack
and authority split; subsystem specifications own their respective contracts
even when they are reached through another specification rather than linked
directly from the entry point.

Design reasoning is navigated through
[Architecture Rationales](rationales/README.md). That index is a navigation
map from normative owners to their WHY documents, not a second specification
catalog. Rationales explain why the normative contracts were chosen,
including rejected alternatives, but do not define schemas or behavior.

Implementation is the executable realization of those contracts. When code,
a rationale, and a specification disagree, the specification is the only
implementation authority; the inconsistency must be resolved rather than
interpreted as an alternate contract.
