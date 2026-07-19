# PnR

This document specifies Loom Spatial place-and-route ownership and the
implemented native boundary. Spatial PnR consumes one exact immutable
TechMapping and produces one complete SpatialMapping when the persistent
SpatialMapping schema is available. There is no `PhysicalMapping` profile.

System-level execution, service, and transport realization belongs to
SystemMapping and its own PnR problem.

## Exact Inputs

Spatial PnR consumes:

* Canonical Dataflow Program `D`;
* verifier-clean TechMapping `T` bound to exact `D` and `F`;
* fully elaborated Fabric Hardware Description `F`;
* immutable `ResolvedPnrConfigView` `C` derived from one exact ResolvedConfig
  artifact; and
* MappingConstraintSet `K` bound to exact `D`, `T`, and `F`.

The persistent authorities are `D`, `T`, `F`, the complete ResolvedConfig, and
`K`. `C` is a typed projection with no independent artifact identity.
`PnrProblemInputs` is an ordinary borrowed grouping and is not a request
artifact.

Before capacity planning or native allocation, PnR rejects `T.D != D.id`,
`T.F != F.id`, and every `K.D/T/F != D.id/T.id/F.id` mismatch. An empty
constraint set is still an exact artifact with those bindings.

## TechMapping Authority

TechMapping alone owns Compute and Memory Realizations, selected semantic
encodings, configured-function witnesses, and software boundary
correspondence. PnR must not regroup actors, recreate `dataflow.subgraph`,
reenumerate semantic compatibility from raw Dataflow and Fabric inputs, or
choose another encoding. Any such change requires a new TechMapping.

## Result Boundary

A successful result is one immutable, profile-complete SpatialMapping that
references its exact TechMapping predecessor and preserves TechMapping
authority. The precise persistent SpatialMapping fields remain open; no empty
record family or compatibility carrier may stand in for them.

Unsupported inputs, proven infeasibility, no legal route, invalid frozen
capacity, and budget exhaustion are ordinary typed results and reports. They
do not produce partial, rejected, or degraded Mapping artifacts. Search
history, candidate collections, scores, and Pareto selection belong outside
the Mapping artifact family.

## Freeze And Native State

Freeze validates, resolves, indexes, and precomputes exact semantic inputs. It
does not choose placement, route, tag, buffer, schedule, memory binding, or
configuration.

The native ownership model has four concepts:

* immutable `FrozenModel`;
* mutable `CandidateState`;
* mutable `SearchScratch`; and
* `MoveTransaction` for atomic candidate changes.

The existing realization and routing freezes are partial structural views.
They preserve typed occurrence domains, exact endpoint and local-arc facts,
logical nets, memory service obligations, and explicit directed routing
topology. The complete production `FrozenModel` field inventory remains open.

## Software Edge And Logical Net Identity

A canonical software edge is the typed producer endpoint plus typed consumer
endpoint. When an artifact-qualified reference is required, it additionally
contains the exact Dataflow artifact identity. There is no persistent edge ID,
edge number, symbol, path, printer-order, or insertion-order alias.

Freeze groups all external edges with one exact producer endpoint into one
deterministic multi-sink logical net. The producer endpoint is stored once per
net and each sink retains its exact consumer endpoint. Compute internality is
derived from configured-function topology and exact actor-to-operation and
boundary correspondences. Only Memory Realization records carry explicit
`DataflowEdgeRef` witnesses for selected memory-internal connectivity. Freeze
may assign dense `PnrIndex` values to rebuildable native arrays, but those
values are not persistent identity.

## Placement And Routing

Spatial placement selects concrete FU occurrences, correlated instruction
contexts, and concrete memory occurrences from domains derived from
TechMapping and Fabric. Compute and Memory placements remain distinct typed
relations.

Routing uses explicit directed Fabric endpoints, point-to-point arcs, and
resource traversals. PE-local, switch-local, memory-local, boundary, and FIFO
traversals are not implicit free connections. Coordinates and visualization
layout are not topology.

Each logical net is realized as one rooted Route Tree with shared trunks and
multi-sink branches. The target model is not one-edge-one-route and does not
persist symbolic paths. Route hot state is mutable and rebuildable.

Endpoint-only A* jointly selects attachment endpoints and a physical path over
the fully elaborated directed topology. Its node identity is only
`TransportEndpointIndex`. The confirmed heuristic is the static minimum
non-negative integer lower-bound cost from an endpoint to the current target
domain. Dynamic occupancy and PathFinder penalties may contribute to `g`, but
Evaluation metrics do not become a second heuristic authority.

## Resource Use, Tags, And Buffers

Mapping has no absolute cycle-slot Schedule IR. Resource-time behavior is
derived from Fabric-owned use patterns and selected event-relative
`ResourceUse`. The immutable Structured Program Candidate remains the owner of
software scheduling decisions.

Physical Tag is local to Fabric-owned interpretation domains. It is not a
global ID or per-token sequence. A selected value is stored once at an
existing writer output or tagged ingress binding and derived through route
continuity. There is no independent `TemporalTagAssignment` family.

Selected physical buffers, services, and mapping-visible configuration must be
stored only where they cannot be mechanically derived. Their exact persistent
record fields remain open.

## Search And Cost Ownership

Production Spatial search uses deterministic transactional multi-start
simulated annealing, common Action and `MoveTransaction` machinery,
endpoint-only A*, Route Tree, and action-local PathFinder state. Simulated-
annealing numeric defaults and remaining integer route-cost details are not
defined here.

`PnRSearchCost` owns only generic distance, connectivity, capacity, occupancy,
and congestion. Accelerator-aware latency, initiation interval, throughput,
clock, memory, area, power, and energy come from unified Evaluation. The
central DSE controller owns objectives, thresholds, ranking, acceptance, and
promotion. Stable symbol order, greedy architecture baselines, and earliest-
legal absolute schedules are not target policies.

## Native Index Width

Persistent Mapping entity identities and native PnR indices have separate
authorities. A freeze-local dense index addresses rebuildable arrays and uses
the build-selected `PnrIndex` type.

`LOOM_PNR_INDEX_BITS` is the sole CMake cache setting for native index width.
It accepts `32` or `64` and defaults to `32`. CMake emits the validated value
through `PnR/BuildConfig.h`; `PnR/PnrIndex.h` owns the canonical type,
capacity measures, preflight, checked conversion, and checked arithmetic.

Before allocating arrays or beginning search, freeze checks entity counts,
CSR offsets, array lengths, products, and maximum indices. Capacity failure is
deterministic and distinguishes count, index, and offset requirements. A
32-bit failure that is representable by the 64-bit contract may recommend a
64-bit rebuild; a requirement beyond the 64-bit contract must not.

Native caches and execution evidence record the actual PnR index width. Width
is not Mapping semantics: when both builds can represent a problem, they must
produce the same Mapping semantic identity. MLIR `index` transport width is a
separate authority and must not be merged with `PnrIndex`.

## CGRA Simulation Boundary

PnR selects and verifies a mapping. CGRA simulation consumes a complete
mapping and concrete runtime inputs, observes execution, and produces
Evaluation Evidence. It may reject an invalid input but must not repair or
complete the mapping. Simulation observations may inform a later DSE request;
they are not copied into Mapping records.

## Validation

Current anchor tests cover exact five-input coupling, typed occurrence and
endpoint domains, endpoint-pair uniqueness, foreign references, deterministic
logical-net grouping, memory internal-edge witnesses, explicit directed
routing topology, and native index capacity. Persistent SpatialMapping final-
verifier tests wait for the closed record schema.
