# End-To-End Conformance Anchors

Conformance anchors prove that the real Loom libraries and public drivers
compose across semantic boundaries. They are not product features, report
artifacts, a second workload inventory, or a large fixture framework.

## Rules

Each anchor uses repository or externally owned workload manifests, exact
artifacts, and ordinary `EvaluationRequest` inputs. It records only minimal
expected observables and stable boundary invariants.

An anchor must not preserve whole printed MLIR, actor counts, pass order, search
trajectory, report layout, or a Cartesian product of kernels, transforms, and
Fabric targets. Unsupported or incomplete scope is a typed outcome, never a
synthetic pass.

## Frontend Source-To-Dataflow Anchors

The initial source set is:

1. `vecadd`: dynamic loop, two loads, add, store, memory frontier, and target-
   dependent scalar/vector candidate selection;
2. `vector_pack`: source-visible vector/integer reinterpretation that retains
   semantic `dataflow.pack` and `dataflow.unpack`;
3. `matmul`: real `i/j/k` nest with outer unroll-and-jam, inner vectorization,
   and reduction choices;
4. `spmm`: dynamic CSR bounds, indirect memory, sparse-dense compute, and
   reduction;
5. `gather`: indirect load, conditional/masked access, and range behavior;
6. `edge_update`: CSR scan, early exit, conditional store, and control-crossing
   memory completion;
7. `fir_filter`: DSP sliding window, nested reduction, reuse, and streaming;
8. `conv2d`: real multi-level convolution nest with tiling, interchange,
   parallelism, unroll, jam, vectorization, and reuse;
9. `stencil3d`: explicit three-dimensional domain, neighborhood accesses,
   boundary handling, permutation, tiling, reuse, and streaming; and
10. `attention`: small complete scaled dot-product attention with `QK^T`, stable
    row softmax, `P*V`, reductions, vectorization, buffering, and ordered
    channels.

Each successful anchor traverses:

```text
C/C++ -> LLVM IR -> S0 -> Sn -> D0 -> D* -> canonical verifier -> DFG-sim
```

Every anchor proves functional terminal observables and stable actor
references. Diagnostic cycle-ordered traces may be emitted to attempt or
scratch storage, but they are not persistent conformance inputs or outputs and
carry no Artifact identity or coverage claim.
Mapping is deliberately absent from this frontend gate. `attention` produces
the logical multi-thread program; heterogeneous AccCore placement and NoC
realization belong to the system anchor.

The LoomBench manifest and pinned CMSIS source trees remain their respective
membership authorities. These anchors are representative conformance cases,
not a replacement suite, and no fixed case count is duplicated here. SPEC
CPU2026 is a separate external conformance corpus.

## Hardware Anchors

Hardware anchors use the public ADG Builder interface or built-in templates to
produce exact Fabric MLIR:

* one regular SpatialCore topology with visualization coordinates only as
  removable metadata;
* one irregular arbitrary directed topology;
* one heterogeneous multi-AccCore system with distinct SpatialCores and
  InstructionCore capabilities;
* one temporal-resource design exercising tags, contexts, and explicit grant
  policy; and
* one memory/service design exercising manager/subordinate interfaces and
  configurable internal operation forwarding.

Removing visualization metadata must not change Fabric identity, legality,
Mapping, simulation, or RTL. Hardware structure, capability, and implementation
refinements are Fabric-owned; builder objects are only construction interfaces.

## Mapping And System Anchor

The canonical heterogeneous anchor launches `project(i)`, `attention(i)`, and
`stats(i)` thread domains. Two project-capable AccCores split `project(i)`;
distinct AccCores execute attention and statistics. A single logical multicast
channel uses an explicit arbitrary-topology NoC shared trunk and replication
point. Memory and external-output services are routed explicitly.

The flow proves:

```text
Dataflow + Fabric -> TechMapping -> SpatialMapping -> SystemMapping
Fabric -> ConfigurationABI
Fabric + ConfigurationABI -> HardwareImplementation
complete Mapping + ConfigurationABI -> HardwareConfigurationImage
complete dependency closure -> Deployment
```

It validates exact artifact coupling, complete realizations, arbitrary-topology
routing, local tag interference, memory/service binding, event-relative
resource use, and deterministic Mapping serialization. CGRA-sim consumes one
complete SpatialMapping without repairing it; sys-sim consumes Deployment and
Gem5SimulationBinding without remapping it.

The system anchor exercises all three Spatial engines through the same Bridge
contract. System + DFG is the first integration gate, followed by System +
CGRA and System + RTL. The first System + DFG anchor uses a built-in System
with at least two AccCores, executes nonempty graphs on more than one selected
AccCore, traverses explicit system transport and external memory, and compares
the requested terminal observables with the corresponding Spatial-only and
native references. A host-only execution or an empty Spatial launch cannot
satisfy this anchor.

## Simulation Execution Budgets

Wall-clock limits are nonsemantic execution controls. Exceeding one produces
`StoppedByLimit` and `CancelledOrTimeout`; it cannot change candidate order,
select a best-so-far prefix, prove infeasibility, or authorize fallback.

For conformance, a System execution is paired with a warmed Spatial-only
execution of the same workload, runtime input, Spatial engine fidelity, trace
capture level, and semantic limits. Active wall time includes simulator reset
and input loading, entry or launch acceptance through terminal observation,
and required observation projection. It excludes compilation, DSE, Mapping,
artifact construction, gem5 build, RTL compilation, cold process startup, and
RTL elaboration. Excluded setup is measured separately and reused across
compatible cases.

The initial reference and budget are:

```text
spatial_reference = max(median warmed Spatial-only active wall time, 100 ms)
system_budget = min(3 * spatial_reference,
                    3 * Spatial-only absolute budget)
```

The DFG Spatial-only absolute budget is the `fast` tier from the canonical
[`timeout-budgets.json`](../config/timeout-budgets.json), so its paired System
+ DFG absolute ceiling is derived by the formula above. A case below that
ceiling still receives the formula-derived budget, not the ceiling. Tiny cases
use a persistent warmed simulator and enough deterministic repetitions to make
the 100 ms floor meaningful. A ratio at or above ten is always a conformance
failure. The factor three is an initial suite-wide policy and may change only
from aggregate profiling evidence, never as a per-case exception.

CGRA Spatial-only bring-up uses the canonical `medium` tier as its bootstrap
ceiling while at least the ten representative workloads in this specification
establish warmed active-wall, reference-cycle-rate, event-count, contention,
and peak-memory evidence. Before System + CGRA conformance begins, the
conformance owner must publish one suite-wide CGRA Spatial-only absolute budget
in tracked gate configuration. Its value is selected from that aggregate
evidence and the 100 k reference-cycles-per-wall-second target. It is not an
Artifact field, semantic limit, model parameter, or per-case override. A later
change requires new aggregate evidence and one tracked gate update. The paired
System + CGRA budget consumes that exact published Spatial-only budget through
the formula above; no caller or simulator may supply a hidden second value.

Every paired result reports active wall time, the System-to-Spatial ratio,
reference cycles per wall second, engine/Bridge/host/observation CPU time,
event and activation counts, and peak resident memory. System simulation
targets at least 100 k reference cycles per wall second. Raw gem5 ticks are
not reference cycles. Corpus orchestration uses at most
`min(nproc - 4, memory-derived worker limit, 120)` outer workers and does not
hide nested oversubscription inside one case.

Host performance observations belong to an explicit fresh diagnostic attempt,
not ordinary Evaluation Evidence. The ordinary gem5 provider declares no
performance output and enables no Bridge or engine host-clock accounting. A
diagnostic preparation declares the profile outputs, requires a fresh external
attempt, validates the root-local attempt generation, and finalizes the same
functional result through the Evaluation Evidence owner before attaching an
invocation-local performance sidecar. The sidecar is neither cached nor an
input to candidate identity, ordering, legality, or objective selection.

`loom.gem5_system_performance_profile.5` owns three disjoint gem5 host
intervals: configuration through `m5.instantiate()`, `m5.simulate()` alone,
and post-simulation observation publication. Each interval reports wall time;
simulation and observation report process CPU time over the same respective
window. Exactly one typed readiness interval is present inside the
configuration interval. DFG and CGRA report managed-engine startup wall and
gem5/Python self CPU time. RTL reports external-engine socket-readiness wall
and gem5/Python self CPU time because its controller already owns the engine
process. Managed-engine process CPU covers the managed child engine's complete
lifetime and is absent for RTL. Bridge callback CPU, Bridge engine-wait wall
time, message count, invocation count, Bridge count, and clock-failure count
are separate fields. A failed clock sample makes the diagnostic profile
unavailable. A nonintegral launch-to-retirement
reference-cycle distance is reported as unavailable for that invocation rather
than changing a valid functional result. Tool failure, cancellation, and
execution limits retain their typed Evaluation outcomes and never become
infeasibility.

The execution-matrix harness reports setup through
`loom.execution_matrix_lifecycle.4.0`. The inclusive `setup` aggregate has no
parent. Its inclusive children name the stable operations
`dataflow_construction_and_publication`,
`fabric_module_construction_and_finalization`, `tech_mapping`, `spatial_pnr`,
`system_fabric_and_interconnect_construction`,
`configuration_abi_and_hardware_implementation_generation`,
`system_mapping_and_pnr`, `guest_compile_and_link`,
`runtime_binding_and_deployment_finalization`, and
`workload_and_runtime_input_publication`; every child names `setup` as its
parent. Every aggregate and child reports wall time, self CPU time, waited-child
CPU time, the self-process lifetime high-water RSS snapshot, and the maximum
waited-descendant RSS snapshot. These observations do not alter construction
order, cache policy, identity, or evidence. Inclusive rows overlap and must not
be summed; RSS snapshots are not interval deltas.

## Hardware-Implementation And Evidence Anchor

One mapped `vecadd` deployment closes the evidence chain:

```text
Fabric + ConfigurationABI -> RTL HardwareImplementation
HardwareImplementation + Deployment -> mapped RTL SimulationExecution/Evidence
RTL implementation -> physical HardwareImplementation
physical implementation -> timing/area/power Evidence
compatible cycle/timing Evidence -> registered Runtime Evidence
```

DFG, CGRA, and mapped RTL executions must satisfy the same independent terminal
`C` oracle. They must agree bit for bit when the exact workload and selected
actor contracts prove one deterministic value. If an approved special-math
accuracy tier or another typed contract admits several values, every engine is
checked independently against that same typed oracle or invariant instead;
pairwise agreement is not correctness evidence. CGRA and RTL cycles are
comparable only when their external service contract is identical.
Raw waveforms, vendor reports, databases, and logs remain owner-attempt or
scratch material until their exact Artifact owner is defined; human-readable
FPA or comparison summaries are projections.

## Negative Anchors

Keep only boundary failures that protect stable contracts:

* non-finalized Dataflow rejected before simulation;
* no legal TechMapping realization;
* no route on an explicit arbitrary topology;
* stale or incomplete Mapping rejected by a consumer;
* unsupported Fabric primitive rejected by RTL lowering;
* failed external model execution recorded without silent fallback; and
* incompatible metric subjects rejected by derived Evaluation.

## Completion

The anchor set is sufficient when the representative software and hardware
flows traverse real in-process libraries, every boundary validates exact
identities, persistent artifacts can be replayed, and failures remain typed.
Broader corpus coverage extends these contracts; it does not create parallel
schemas or test-only implementations.
