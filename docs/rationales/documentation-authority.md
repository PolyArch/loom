# Documentation Authority Rationale

## Decision

Loom separates design knowledge into three layers:

* tracked `spec-*.md` files own normative WHAT;
* tracked rationale files own non-normative WHY; and
* source code owns HOW and must conform to the specifications.

[Loom Full-Stack Architecture](../spec-loom-stack.md) is the normative owner of
this split. [The rationale index](README.md) is only navigation.

## Why The Earlier Ledger Model Was Retired

The architecture was developed through a long-running local decision ledger.
That was useful while decisions were changing rapidly: it retained questions,
alternatives, corrections, and reasoning without forcing every intermediate
idea into a product specification. It was a poor permanent authority for two
reasons.

First, an ignored local file is not durable project history. A design required
to interpret tracked code must not depend on one workstation or backup habit.
Second, keeping current decisions in both a meeting ledger and tracked specs
creates an authority ordering problem. Even when one is nominally higher, a
reader must reconcile two complete descriptions and can accidentally revive a
superseded contract.

The tracked split preserves the useful distinction without preserving the
duplication. Specifications contain only implementable current contracts.
Rationales retain motivations and rejected alternatives, but refer to the
specification for every exact contract.

## Why WHAT And WHY Are Separate

A verifier or implementation needs closed types, ownership, ordering,
canonical bytes, failure classification, and conformance anchors. Mixing a
chronological argument into those rules makes it difficult to tell whether an
old alternative is still legal. Conversely, a specification that records only
the selected shape loses the constraints that made the choice necessary and
invites a later redesign to repeat rejected mistakes.

The split therefore follows semantic responsibility rather than document
chronology:

* a specification can be consumed mechanically and reviewed for completeness;
* a rationale can compare alternatives and preserve design evolution; and
* neither needs to duplicate the other.

## Revision Rule

When a decision changes, the owning specification is changed to one current
contract. The corresponding rationale explains what requirement invalidated
the old choice. It does not retain the old fields, state machine, or examples
as an alternate implementable design.

Open work belongs in the owning specification as an explicit unsupported or
deferred boundary only when that boundary affects current behavior. Temporary
discussion queues, implementation status, and worker coordination do not
belong in either layer.

External literature can motivate a change or provide an algorithmic component,
but it cannot silently override Loom's owners. A proposed change must first be
reconciled with the existing ownership, identity, Mapping, simulation, runtime,
and backend contracts, then update the normative owner and its rationale
together.
