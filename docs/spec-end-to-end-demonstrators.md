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

Every anchor proves functional terminal observables. A representative subset
also proves monotonic cycle-ordered firing trace and stable actor references;
the suite does not require a duplicate trace fixture for every kernel. Mapping
is deliberately absent from this frontend gate. `attention` produces the
logical multi-thread program; heterogeneous AccCore placement and NoC
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

## Hardware-Implementation And Evidence Anchor

One mapped `vecadd` deployment closes the evidence chain:

```text
Fabric + ConfigurationABI -> RTL HardwareImplementation
HardwareImplementation + Deployment -> mapped RTL SimulationExecution/Evidence
RTL implementation -> physical HardwareImplementation
physical implementation -> timing/area/power Evidence
compatible Evidence -> derived runtime/energy Evidence
```

DFG, CGRA, and mapped RTL executions must agree on terminal `C`. CGRA and RTL
cycles are comparable only when their external service contract is identical.
Raw waveforms, vendor reports, databases, and logs remain detailed bundles;
human-readable FPA or comparison summaries are projections.

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
