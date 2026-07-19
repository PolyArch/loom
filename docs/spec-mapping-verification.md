# Mapping Verification

This document specifies the ownership and observable behavior of Mapping
verification. A verifier proves legality and closure for one immutable Mapping
artifact profile. It does not run search, repair records, choose fallbacks,
complete a consumer-specific view, or mutate Dataflow or Fabric artifacts.

Evaluation owns observations and metrics. The central DSE controller owns
thresholds, objective composition, ranking, acceptance, and promotion. Mapping
verification must not duplicate either authority.

## Artifact Profiles

The target Mapping artifact family has three immutable, profile-complete
outputs:

* TechMapping;
* SpatialMapping; and
* SystemMapping.

There is no `PhysicalMapping` profile and no consumer-defined completeness
profile. The current neutral C++ draft and verifier implement TechMapping only.
They must not publish empty SpatialMapping or SystemMapping records before
those persistent schemas are closed.

## Failure Model

An invalid or incomplete candidate is not a Mapping artifact. Unsupported
inputs, infeasibility, budget exhaustion, and verification failures are
ordinary typed results. Human-readable details may appear in a report or in
Evaluation Evidence when an evaluation was actually requested, but they are
not artifact-owned diagnostic records.

The verifier may return more than one finding when reference resolution is
safe, but no consumer may append records to make an artifact valid. A producer
must construct a new complete artifact and run the same verifier.

## TechMapping Verification

The implemented TechMapping verifier checks at least:

* the typed TechMapping draft boundary and exact input identities;
* exact Dataflow and Fabric artifact identities;
* resolution and kind correctness of every artifact-local entity reference;
* exact graph and actor ownership of every typed Dataflow endpoint;
* uniqueness of each software edge by producer endpoint plus consumer
  endpoint;
* closed graph coverage across disjoint Compute and Memory Realizations;
* selected FU, semantic encoding, configured-function, actor, lane, and
  boundary correspondence legality;
* selected memory implementation semantics, operation correspondence,
  logical-root coverage, boundary correspondence, and exact internal-edge
  witnesses; and
* coherent service and access-capability obligations.

An artifact-qualified software-edge reference consists only of the exact
Dataflow artifact identity and the typed producer/consumer endpoint pair.
Foreign artifact references are rejected before endpoint lookup. A pair that
does not occur in the referenced Canonical Dataflow Program is unresolved.
There is no edge entity namespace, edge number, symbol, path, printer-order,
or insertion-order fallback.

## Spatial And System Verification

SpatialMapping verification will recompute physical closure from the exact
five inputs and the selected persistent records. SystemMapping verification
will recompute execution, service, and resource-use closure from its exact
predecessors. Their precise record inventories and persistent diagnostics are
not defined here because those schemas remain open.

Neither verifier may reinterpret TechMapping, rematch raw Dataflow and Fabric
inputs, reconstruct a hidden software schedule, or accept a record because a
consumer supplied the missing fact.

## Determinism

Verification results depend only on canonical semantic inputs and the
applicable profile contract. Ordering is derived from typed artifact-local
identities and structural keys. Symbols, source-vector order, serialized
record position, and builder insertion order are not authorities.

## Validation

Tests stay at semantic anchors:

* exact profile and predecessor identity coupling;
* foreign and wrong-kind entity references;
* duplicate software endpoint pairs and invalid graph or actor endpoints;
* closed Compute and Memory Realization coverage;
* exact configured-function correspondence and exact Memory internal-edge
  witnesses; and
* deterministic derived freeze results under harmless input permutations.

Tests must not establish a diagnostic-code matrix, snapshot record spelling,
or preserve retired partial-artifact and consumer-profile behavior.
