# Mapping And PnR Rationale

Normative contracts are owned by
[Mapping Artifact](../spec-mapping-artifact.md),
[Mapping Identity](../spec-mapping-identity.md),
[Mapping Memory](../spec-mapping-memory.md),
[Mapping Verification](../spec-mapping-verification.md), and
[Place And Route](../spec-pnr.md).

## Why Mapping Has Three Profiles

Software-to-hardware realization contains three cumulative questions:

* TechMapping selects a Fabric-owned FU or memory capability template and
  proves that a canonical software subgraph is supported;
* SpatialMapping selects concrete resources, contexts, endpoints, routes,
  buffers, tags, and physical refinements inside a SpatialCore; and
* SystemMapping binds thread/graph execution, channels, services, and transport
  across heterogeneous AccCores.

TechMapping has no route because it selects a logical materialization against a
template, not a physical occurrence. Spatial and System Mapping both route,
but in different resource universes. A fourth PhysicalMapping profile was
rejected; concrete physical choices already belong to the latter two profiles.

All three are immutable roots in one Mapping artifact family. A mutable
partial/complete object or a separate witness schema would force consumers to
interpret lifecycle state. Each profile is complete under its own verifier and
references its exact predecessor.

## Why Compute Realization Replaces `dataflow.subgraph`

An FU can implement a small configured software graph. The selected actor
group, capability template, actor-to-op correspondence, boundary port
correspondence, and absorbed internal edges are target-specific facts. They
belong in a Compute Realization, not Canonical Dataflow.

The group has no independent semantics without its selected capability, so a
separate SoftwareActorGroup record was rejected. Memory actors use a parallel
Memory Realization because service, consistency, ports, and internal memory
relations are not FU compute topology.

TechMapping persists only non-derived selected relations. Exact actor
semantics remain in Dataflow, capability remains in Fabric, and configured
software projections are reconstructed from both. SpatialMapping then chooses
an occurrence of the same Fabric FU definition; it cannot re-group actors or
copy a configured graph.

## Why PnR Has Exact Immutable Inputs

Spatial PnR binds an exact Canonical Dataflow program, TechMapping, fully
elaborated Fabric, resolved search-policy view, and MappingConstraintSet. The
five-input coupling prevents a witness or constraint from being rebound to
similar hardware or a later program. System PnR uses the corresponding exact
system inputs.

Fabric must be elaborated before search. PnR cannot instantiate resources or
change capability. Mapping constraints are a separate Artifact because they
restrict the legal mapping space; ResolvedConfig controls how that space is
searched; the Mapping result records the selected solution. A generic `config`
object for all three would duplicate ownership and cache dependencies.

## Why Persistent MLIR And Native Search Data Coexist

Mapping MLIR is the typed persistent wire and independent verification
surface. MLIR operations, attributes, strings, JSON objects, and recursive
references are inappropriate for millions of hot search accesses. Import
therefore validates exact references once and freezes a removable native model
with dense indices, SoA/CSR storage, precomputed endpoint domains, adjacency,
and distance/connectivity caches.

This is not a second authority. The frozen model is rebuilt only from exact
artifacts and config views, has no persistent identity, and cannot write search
decisions back into the inputs. Index width is a centralized compile-time
choice; overflow fails with a diagnostic requesting a 64-bit rebuild rather
than silently truncating.

## Why Endpoints And Traversals Are The Routing Atoms

Resource-level adjacency loses port direction, kind, width, switch traversal,
and boundary behavior. RTL bit graphs are too low level. PnR routes over a
directed graph of typed Fabric endpoints and explicit traversals.

Placement selects a compatible FU occurrence but does not preselect a complete
PE-port assignment. Each software endpoint retains a factorized domain of
compatible physical endpoints; route search jointly selects endpoint and path.
This avoids a Cartesian explosion and lets congestion inform attachment.

A residual logical net represents exactly one canonical producer and all
unabsorbed sinks. Multi-sink routing builds one route tree with explicit
replication resources. A broadcast branch that reconverges without logical
work is redundant program structure to optimize, not a reason to relax tag or
endpoint constraints.

## Why Placement And Routing Are Coupled

Greedy placement followed by one routing pass cannot escape congestion,
capacity, or temporal-context interactions. Early experiments showed that
simulated annealing with only single-unit moves and static-distance cost simply
reproduced greedy solutions even with many moves and restarts. Transactional
swaps, endpoint-aware routes, capacity proxies, negotiated congestion, and
closure actions are necessary neighborhoods and feedback.

The production model couples placement, endpoint selection, routing, tags,
buffers, memory binding, resource use, and refinements through typed Actions
and atomic MoveTransactions. A move either re-closes all affected obligations
or rolls back. Strict place-then-route remains expressible as an action policy;
it is not a second solver or Mapping schema.

## Why Arbitrary Topology Uses A* And Negotiation

Mesh coordinates and Manhattan distance cannot define route legality for an
arbitrary directed Fabric. Endpoint-only A* uses exact connectivity and an
admissible topology-derived heuristic. Multi-net contention then requires a
negotiated router such as the specified PathFinder or dual-subgradient kernels.

Negotiation state, prices, occupancy, and best iterates are native scratch,
not Mapping fields. Stable iteration ordering, checked numeric protocols,
canonical aggregation, and deterministic work units make randomized or
parallel search replayable. Budget exhaustion is inconclusive, never proof of
infeasibility.

Bounded exact repair is a final focused solver for a small unresolved closure,
not a global replacement for annealing and routing. It consumes an explicit
bounded problem and cannot publish a best-so-far invalid Mapping.

## Why Legality And QoR Are Separate

Mapping's local solver cost can measure domain-independent facts such as
unrouted nets, capacity violation, distance, congestion, and constraint
violation. Accelerator latency, energy, frequency, memory behavior, or system
performance belong to shared Evaluation models. Central policy combines these
observations; PnR must not grow a private accelerator cost model.

Some Fabric configurations preserve function but change performance, such as
FIFO bypass. Mapping selects them, but Evaluation supplies the relevant cycle,
timing, power, or bandwidth feedback. This preserves one metric and evidence
system across compiler, mapper, and hardware DSE.

## Why Tags Are Local Allocation

A Physical Tag distinguishes logical uses that may overlap on the same tagged
match domain. It is not a software event ID, channel epoch, or globally unique
label. Nets with no possible physical competition can reuse a value; competing
uses require distinct values within the declared width.

Tag coloring is therefore derived from selected routes, contexts, resource
sharing, and overlap. Tagged endpoints may carry several logical nets only
under that proof. Untagged endpoints remain single-use for overlapping
transfers.

## Why SystemMapping Is Required

Binding threads to AccCores and channels to the system transport is not runtime
policy left after SpatialMapping. It determines which heterogeneous cores run
which logical points, which SpatialMapping each rooted graph launch uses, and
how ordered cross-kernel streams and services traverse the NoC.

The same kernel's thread points may be distributed among AccCores, while
different kernels may occupy distinct subsets and pipeline through channels.
Execution binding and channel routing must be solved together because a
software schedule may or may not permit FIFO streaming. Runtime only admits
and authorizes the immutable selected resources; it cannot remap them.

Hierarchical System PnR reuses immutable SpatialMappings. A flattened global
mode remains a search option, not a different artifact authority. Both produce
the same SystemMapping schema and face the same final verifier.

## Why System Spatial-Temporal Is A Spectrum

System-level spatial-temporal allocation and resource-level temporal hardware
are independent questions. At system level, the thread binding relation maps
logical domain points to AccCore occurrences. Thread groups, resident groups,
batches, and waves are derived partitions or scheduling views of that relation,
not additional persistent records or mutually exclusive execution modes.

`MaxSpatial` and `MaxTemporal` are useful profile names for opposite search
preferences, not the only legal answers. One candidate may dedicate different
AccCore subsets to different kernels and pipeline their channels, while each
subset processes points of its own kernel in batches. Another may distribute
one kernel across most cores. Intermediate mixtures are expected and remain
expressible by the same binding and event-relative resource-use algebra.

This spectrum must not be confused with `fabric.pe`, `fabric.switch`, or
`fabric.mem` spatial/temporal capability. The latter states how one physical
resource accepts concurrent or time-multiplexed uses. SystemMapping states
which logical work uses which physical occurrences. Keeping the axes
orthogonal prevents a profile label from becoming a hidden hardware mode or a
second schedule authority.

## Why Examples Cover Regular And Irregular Hardware

The minimal five-input example isolates coupling and endpoint routing. Memory,
multicast, irregular, and torus examples extend the same contracts rather than
introducing demo-only schemas. Narrow software values may traverse wider
same-kind links under the low-bit rule, and control tokens may use explicitly
adapted shared transport. These examples demonstrate separation between
software semantics and physical representation.

Regular and irregular hardware must expose the same compatible semantic
candidate set when their capabilities match, even though their complete
resource inventories and route domains differ. This is the useful invariant;
forcing identical resource multisets would erase topology diversity.
