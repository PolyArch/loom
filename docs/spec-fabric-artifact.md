# Fabric Artifact

This document defines the persistent artifact boundary for finalized Fabric
hardware descriptions. Fabric dialect specifications own hardware semantics;
this document owns only root variants, dependency framing, canonical bytes,
identity, finalization, and publication.

## Artifact Family And Root Variants

The first persistent family is:

```text
loom.fabric 1.0

FabricRoot =
    Module
  | System
  | InterconnectImplementation
```

The three variants share one artifact family because they use the same Fabric
semantic model, reference framing, canonicalization rules, and finalization
boundary. They retain distinct typed root payloads and verifiers. The variant
tag cannot be inferred from a symbol name, path, or MLIR operation order.

`Module` is one SpatialCore hardware template. `System` is one architecture-
only multi-core Fabric system. `InterconnectImplementation` is the exact
protocol and implementation sibling for one `System`; it refines, but never
redefines, that system's Transport Architecture.

There is no persistent finalized-design wrapper, separate family per variant,
or generic hardware manifest.

## Direct Dependencies And Derived Closure

Each root persists only exact direct `ArtifactReference<T>` dependencies. A
module instantiation references its exact module dependency. A system
references its exact module roots. An interconnect implementation references
its exact system root and any exact external implementation dependencies
required by its typed contract.

The transitive dependency closure is derived mechanically. It is never stored
as another list. Missing, foreign, wrong-kind, duplicate, cyclic, or unused
direct dependencies make finalization fail.

Builder handles, helper names, preset names, source locations, visualization
metadata, file paths, and printer positions are provenance or projections and
do not enter dependency identity.

## Canonical Semantic Relation And Bytes

Artifact identity is computed from one canonical semantic relation, not from
authoring order or raw source text. Canonicalization:

1. expands every `fabric.instantiate` needed by the root;
2. resolves typed direct references;
3. strips nonsemantic names, locations, and visualization metadata;
4. verifies all dialect, resource, capability, and domain contracts;
5. constructs the root's identifier-free typed semantic graph;
6. selects the lexicographically least canonical serialization among semantic
   graph isomorphisms;
7. assigns root-local entity identifiers and structural ordinals from that
   canonical form; and
8. writes one deterministic MLIR bytecode payload in canonical entity order.

The canonical artifact bytes are a domain-separated envelope containing the
schema descriptor, root variant, direct exact references in typed canonical
order, and canonical MLIR bytecode payload. The Common Artifact SHA-256 v1
contract computes the identity over that envelope.

The specification fixes the canonical result, not a canonical-labeling
algorithm. Individualization-refinement, orbit pruning, or another exact
algorithm is an implementation choice. A resource bound may produce typed
`Incomplete`; it may not publish a noncanonical identity.

For one supported Loom codebase and resolved configuration, semantically
equal hardware has identical canonical bytes and identity. Different canonical
bytes under one digest are an internal error. A semantic hardware difference
must change canonical bytes and identity.

## Finalization And Publication

Finalization is failure-atomic:

```text
authoring draft
  -> close scopes and helpers
  -> resolve direct typed references
  -> expand instantiations
  -> verify semantic contracts
  -> canonicalize and assign local identities
  -> write canonical bytes
  -> import and independently reverify
  -> compute ArtifactIdentity
  -> publish the root and its exact direct dependencies atomically
```

No root, local identifier, canonical byte stream, or digest is externally
visible before the entire sequence succeeds. Retrying an unchanged valid draft
must produce the same result.

Malformed hardware is `Invalid`. A well-formed custom Fabric whose selected
backend lacks a provider remains a valid Fabric artifact; the backend reports
typed `Unsupported`. A published builtin target must have provider closure for
every capability it advertises and therefore fails publication when such a
provider is absent.

## Ownership Boundaries

This artifact family does not own:

* software operation semantics, owned by the canonical operation schema;
* physical sharing legality, owned by the HSG registry;
* concrete `fabric.op` capability, owned by Fabric operation contracts;
* backend availability or recipes, owned by Fabric-to-RTL providers;
* software-selected configuration, owned by Mapping and ConfigurationABI;
* implementation state, owned by HardwareImplementation; or
* timing, power, area, and other observations, owned by EvaluationEvidence.

These owners may reference a Fabric artifact. They may not copy its topology,
capability, identity catalog, or canonicalization rules.

## Anchor Verification

Anchor tests cover:

* equivalent regular and irregular Fabrics with different source names and
  construction order producing identical bytes and identity;
* one semantic topology, capability, state, timing, or domain change producing
  a different identity;
* wrong-kind, foreign, duplicate, cyclic, and missing direct references;
* failure-atomic publication and deterministic retry;
* independent import and re-verification of canonical bytes;
* a valid custom Fabric with a missing backend provider reporting
  `Unsupported`; and
* a builtin target refusing publication when provider closure is incomplete.

Tests do not freeze one canonical-labeling implementation, MLIR printer
whitespace, Builder handle order, filesystem layout, or a large topology
fixture matrix.
