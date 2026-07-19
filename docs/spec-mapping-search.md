# Mapping Search And PnR Policy

This document specifies confirmed search ownership and architecture. Search
operates on immutable artifacts and ephemeral native state. It does not create
a second persistent legality, Evaluation, or DSE authority.

## Ownership

TechMapping search alone selects Compute and Memory Realizations, semantic
encodings, and exact software-to-implementation correspondences. Spatial PnR
consumes one immutable TechMapping predecessor and must not rematch actors,
reenumerate FU compatibility from raw Dataflow and Fabric inputs, regroup a
legacy `dataflow.subgraph`, or select another semantic encoding.

Mapping verification owns legality. `PnRSearchCost` owns only generic search
facts such as distance, connectivity, capacity, occupancy, and congestion.
Accelerator-aware latency, initiation interval, throughput, clock, memory,
area, power, and energy come from the unified Evaluation system. The central
DSE controller owns candidate sets, objective composition, ranking,
acceptance, and promotion.

## Exact Spatial PnR Inputs

Spatial PnR consumes one exact coupling of:

* Canonical Dataflow Program `D`;
* TechMapping `T` bound to exact `D` and `F`;
* fully elaborated Fabric Hardware Description `F`;
* immutable `ResolvedPnrConfigView` `C` derived from one ResolvedConfig
  artifact; and
* MappingConstraintSet `K` bound to exact `D`, `T`, and `F`.

Freeze rejects every identity mismatch before native capacity planning. `C`
is a typed projection and has no independent artifact identity.

## Freeze And Native State

Freeze validates, resolves, indexes, and precomputes. It does not choose a
placement, route, tag, buffer, schedule, memory binding, or configuration.
The production native model has exactly these ownership classes:

* immutable `FrozenModel`;
* mutable `CandidateState`;
* mutable `SearchScratch`; and
* transactional `MoveTransaction`.

The aggregate production `FrozenModel` field inventory remains open. Existing
realization and routing freezes are partial structural views, not a completed
persistent schema.

## Spatial Search Architecture

Production Spatial search uses deterministic transactional multi-start
simulated annealing. Every move is expressed through the common Action and
`MoveTransaction` machinery so legality caches and candidate state commit or
roll back atomically.

Routing uses endpoint-only A* over the fully elaborated arbitrary directed
topology. A* node identity is only a typed transport endpoint index. Time,
tags, resource-time, deadlock state, and configuration are not appended to
node identity. Per-net routing uses one rooted Route Tree with shared trunks
and multi-sink branches. PathFinder occupancy and history are action-local
search state.

The exact simulated-annealing numeric defaults, move distribution, integer
route-cost terms, equal-cost tie rules, and cache policies remain open. This
document does not invent them.

## Determinism And Identity

Determinism is derived from canonical semantic identities, typed structural
keys, the exact resolved configuration, and explicit algorithm rules. Stable
symbol order, source text order, printer order, insertion order, paths, and
compatibility aliases are forbidden tie-breaking authorities.

Dense `PnrIndex` values are rebuildable native indices. They are never written
back into Mapping artifacts or MappingConstraintSet.

## Candidate And Failure Boundary

Search may retain mutable partial candidates, queues, journals, congestion
state, and provisional cost deltas. None is a persistent Mapping artifact.
Only one complete selected Mapping candidate is finalized for a Mapping
profile.

Unsupported inputs, proven infeasibility, no legal route, and budget
exhaustion are ordinary typed results and reports. Rejected candidates and
search history are not serialized as partial, rejected, or degraded Mapping
artifacts. Candidate collections, selected-candidate records, and Pareto sets
belong to the central DSE model rather than a Mapping-owned manifest.

## Validation

Search tests cover semantic anchors rather than a policy matrix:

* exact five-input coupling and foreign reference rejection;
* deterministic freeze under harmless descriptor permutations;
* factorized Compute and Memory candidate domains;
* endpoint-only routing over explicit directed topology;
* deterministic logical-net grouping and multi-sink Route Tree behavior; and
* atomic move commit and rollback when those components are implemented.

Tests must not preserve a greedy architecture baseline, stable symbol order,
earliest-legal absolute scheduling, one-edge-one-route state, or Mapping-owned
objective formulas.
