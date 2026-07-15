# Subgraph-to-FU Generalization

This document is deprecated as target-state authority.

The current implementation still generalizes and enumerates
`dataflow.subgraph` operations, and source comments still cite historical
sections of this document. That implementation surface is legacy behavior,
not the canonical software model.

The confirmed semantic destination is:

* the Canonical Dataflow Program has no canonical `dataflow.subgraph`
  partition or configured-graph authority;
* an FU configuration determines a canonical typed and attributed
  configured-function projection of explicit Fabric topology;
* Fabric exposes normalized valid semantic encodings, with distinct valid
  encodings denoting distinct configured functions; and
* TechMapping records the selected actor group, FU realization, encoding,
  and typed correspondences as one Compute Realization.

`docs/spec-mapping-artifact.md` owns the Compute Realization and Mapping
ownership boundary. The exact normalized-encoding representation and
Mapping dialect syntax remain open and are not defined here.

The historical headings below are retained only so existing source
references remain resolvable while the legacy implementation is migrated.
They do not preserve the old specification as a competing authority, and
they must not be extended with new target-state requirements.

## End-to-end interface

Deprecated implementation anchor. The old subgraph-based pass interface is
not a target-state contract.

### Failure reasons (closed enumeration)

Deprecated implementation anchor. Existing diagnostics describe the legacy
pass only.

## Configuration Surface

Deprecated implementation anchor. The legacy strategy factory recognizes,
in stable order, `anchor`, `mcs`, `incremental`, and `incremental_random`.
Their configuration does not define the target Mapping schema.

## Module architecture

Deprecated implementation anchor. Current implementation structure does
not define future Mapping or configured-function architecture.

### SynthConfig schema

Deprecated implementation anchor. Configuration remains governed by its
own SSOT; this document does not define a new schema.

## Strategies

Deprecated implementation anchor. Existing strategy APIs operate on the
legacy subgraph representation.

### Strategy: anchor (tier A by default)

Deprecated implementation anchor.

#### Acceptance criteria (anchor)

Deprecated implementation anchor.

### Strategy: mcs

Deprecated implementation anchor.

#### Acceptance criteria (mcs)

Deprecated implementation anchor.

### Strategy: incremental

Deprecated implementation anchor.

#### Acceptance criteria (incremental)

Deprecated implementation anchor.

### Strategy: incremental > extend_to_cover

Deprecated implementation anchor.

### Strategy: incremental_random

Deprecated implementation anchor.

#### Acceptance criteria (incremental_random)

Deprecated implementation anchor.

## Parallelism plan

Deprecated implementation anchor. Legacy synthesis may evaluate detached
candidates concurrently, but deterministic input order is preserved.

### MLIR mutation is never parallel

Worker closures do not mutate the user's MLIR context or module. They build
detached candidates in worker-local state, and the main thread performs
ordered module mutation after workers complete.

## Sub-algorithms shared by strategies

Deprecated implementation anchor. These helpers are not persistent schema
owners.

### Alignment

Deprecated implementation anchor.

### CoverageVerifier

Deprecated implementation anchor.

### SCC handling for tier C

Deprecated implementation anchor.

### hw_params policy

Deprecated implementation anchor. The legacy helper records the observed
attribute axes required by the current enumerator. This behavior is not the
target normalized-semantic-encoding contract.

## Legacy analytic cost formula

Deprecated implementation anchor. The current synthesizer cost model uses:

```text
cost(fu) = sum baseArea(shareGroup, bw)
         + sum carry_penalty * bw
         + sum mux_penalty   * portCount * bw
         + sum demux_penalty * portCount * bw
```

The terms are implementation ranking inputs, not Mapping facts or target
hardware cost authority.

## Examples

Deprecated implementation anchor. Historical examples are not canonical
Dataflow or Mapping forms.

### Tier C example (feedback alignment)

Deprecated implementation anchor.

## Failure reasons

Deprecated implementation anchor. Failure diagnostics remain
implementation evidence and do not define the target semantic model.
