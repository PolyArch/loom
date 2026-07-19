# Compiler Ownership And Mapping Boundaries

This document specifies the ownership boundaries that replace a generic
multi-level placement framework. AccCore outlining, SpatialCore outlining,
TechMapping, and Spatial place-and-route are distinct problems with distinct
semantic outputs. They may share implementation utilities only when those
utilities do not merge their authorities.

## Structured Program Ownership

The immutable Structured Program Candidate owns selected software scheduling,
parallelization, vectorization, reduction, and execution-domain ownership.
Compiler transformations may produce another candidate, but downstream
Mapping must not rewrite those decisions.

AccCore outlining selects which structured program regions execute as
`dataflow.thread` definitions. The resulting thread is the InstructionCore
stored-program surface. The exact search policy and persistent candidate
record syntax are outside this document.

SpatialCore outlining selects regions within a thread for spatial execution.
The compiler-internal `loom.spatial_region` boundary carries explicit typed
operands, results, streams, memory capabilities, and control. Canonical
lowering removes that temporary boundary and produces one symbol-bearing
`dataflow.graph` definition plus explicit `dataflow.graph.launch` operations.

Outlining must preserve explicit effects, causal dependencies, memory order,
completion, and `IsolatedFromAbove` boundaries. It must not infer hardware
placement from coordinates, source layout, symbols, or operation order.

## TechMapping Ownership

TechMapping consumes one finalized Canonical Dataflow Program and one
finalized Fabric Hardware Description. It alone owns:

* Compute and Memory Realization actor grouping;
* selected Fabric implementations and semantic encodings;
* actor-to-operation correspondence;
* software-boundary-to-hardware-template correspondence that, with configured-
  function topology, determines Compute internality; and
* exact Memory Realization internal-edge witnesses.

TechMapping leaves the Canonical Dataflow Program unchanged. It does not
recreate `dataflow.subgraph`, persist a second software partition, or copy a
configured software graph. Consumers must not rematch raw Dataflow and Fabric
inputs, regroup actors, or select another semantic encoding.

TechMapping search legality comes from the Mapping verifier and Fabric-owned
capabilities. Its exact production search algorithm remains open. Symbol
spelling, source order, printer order, insertion order, and compatibility
aliases are not search or tie-breaking authorities.

## Spatial PnR Ownership

Spatial PnR consumes the exact `D/T/F/C/K` coupling defined by
`docs/spec-pnr.md`. It preserves the immutable TechMapping predecessor and
selects concrete physical realization facts.

The confirmed search architecture uses deterministic transactional
multi-start simulated annealing, common Action and `MoveTransaction`
machinery, endpoint-only A*, one Route Tree per logical net, and action-local
PathFinder state. Exact numeric defaults and final persistent SpatialMapping
record fields remain open.

Spatial PnR must not repeat TechMapping, reconstruct software ownership, or
create one route record per software edge. A successful result is one complete
SpatialMapping, not a partial candidate, delta profile, diagnostic artifact, or
mapping-set manifest.

## Cost And Decision Ownership

Legality, observation, and candidate acceptance have separate owners:

* Mapping verification proves static legality and closure;
* `PnRSearchCost` owns only generic distance, connectivity, capacity,
  occupancy, and congestion used by spatial search;
* Evaluation owns accelerator-aware latency, initiation interval,
  throughput, clock, memory, area, power, and energy observations; and
* the central DSE controller owns objectives, thresholds, ranking,
  acceptance, promotion, and requests for new candidates.

Compiler or Mapping records must not persist local utilization scores,
estimated QoR, routing-cost formulas, Pareto sets, selected-candidate records,
or consumer-filled diagnostics. Ordinary reports and Evaluation Evidence may
explain a failure or observation without becoming Mapping authority.

## Determinism

Determinism derives from canonical semantic identities, typed structural
keys, exact resolved configuration, and explicit algorithm rules. It does not
derive from source order, stable symbols, textual paths, printer order, or a
pinned greedy baseline.

The same semantic inputs and resolved configuration must produce equivalent
canonical results. Tests should assert semantic ownership, exact identity
coupling, closed coverage, and deterministic derived state rather than one
incidental partition or candidate enumeration.

## Open Boundaries

This document does not define:

* exact AccCore or SpatialCore outlining search policy;
* Mapping MLIR syntax or a Mapping schema version;
* exact SpatialMapping bindings, Route Tree, ResourceUse, Physical Tag,
  buffer, memory, service, or configuration records;
* the complete production `FrozenModel`; or
* TechMapping search defaults or central DSE policy.

Implementations must not fill these gaps with generic placement wrappers,
compatibility aliases, placeholder records, or textual implementation-shape
tests.

## Validation

Anchor tests cover explicit compiler boundaries, canonical graph lowering,
TechMapping closed coverage and correspondence, exact PnR input coupling,
endpoint-pair identity, deterministic freeze, logical-net fanout, and
transactional search behavior when implemented. Mapping-owned QoR formulas,
source-order partitions, stable-symbol ordering, and retired level labels are
not test contracts.

## References

* `docs/spec-compiler-part-3-dfg.md`
* `docs/spec-compiler-part-3-impl.md`
* `docs/spec-mapping-artifact.md`
* `docs/spec-mapping-search.md`
* `docs/spec-pnr.md`
