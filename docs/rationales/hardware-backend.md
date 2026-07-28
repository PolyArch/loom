# Hardware Backend Rationale

Normative contracts are owned by
[RTL Lowering](../spec-rtl-lowering.md),
[Hardware Implementation](../spec-hardware-implementation.md),
[EDA Tooling](../spec-eda-tooling.md), and
[FPA Evaluation](../spec-fpa-estimation.md).

## Why RTL Is Derived From Fabric

Fabric is the architecture and capability truth. RTL lowering implements an
exact Fabric artifact under a resolved provider and ConfigurationABI; it cannot
classify operations by string, invent unsupported behavior, or change timing,
capacity, state, buffering, arbitration, and clock/reset contracts.

Provider dispatch is keyed by the Fabric-owned implementation family and
resolved capability view. A backend-local operation list, `cfg_mode` enum, or
hard-coded semantic table would compete with OperationSchema, HSG, and the
concrete resource. Missing provider support is typed Unsupported and produces
no successful RTL artifact.

## Why Fake RTL Success Is Forbidden

An X-filled module, passthrough stub, unimplemented branch, or same-provider
software model can make a build or test pass without implementing the hardware.
Such output is more harmful than an explicit failure because downstream EDA
and comparison will treat it as real capability.

Leaf providers must implement the exact typed datapath and protocol. Structural
lowering then composes boundaries, FIFOs, temporal PEs, switches, memory
engines, backpressure, and adapters from their Fabric contracts. Unknown or
unrealizable resources stop publication.

## Why HardwareImplementation Is An Artifact

Generated RTL, synthesized netlist, physical layout, or FPGA image represents
an immutable implementation state with exact lineage, payload roles,
interfaces, activity points, memory macro bindings, and platform inputs. Each
real implementation change produces another HardwareImplementation.

QoR, pass/fail status, logs, and reports do not enter that artifact. They are
Evaluation observations or attempt material. This prevents a tool result from
changing implementation identity and lets several evaluations query the same
layout under different corners or requirements.

ImplementationPlatform separately owns technology libraries, devices, and
typed corners. The backend cannot hide a host filesystem library path as
portable hardware truth.

## Why CIRCT And LLVM Are Pinned Together

CIRCT provides mature RTL and hardware IR infrastructure, so Loom should not
reimplement it. CIRCT's stable `firtool` releases are tested against an exact
LLVM revision. Loom therefore pins CIRCT to a selected stable release commit
and top-level LLVM to that revision's gitlink, builds only the top-level LLVM,
and leaves CIRCT's nested LLVM uninitialized.

This atomic pair avoids two incompatible MLIR/LLVM ABIs while retaining
unmodified upstream source. Build identity records the exact commits and
semantic options rather than following floating branches.

## Why Constraints Are Derived

Clock/reset domains, crossings, Fabric resource timing, generator bindings,
implementation interfaces, and platform facts already determine the SDC and
verification harness. A handwritten or backend-default constraint would be a
second hardware contract. Generated constraints and scripts are reproducible
payloads or attempt inputs whose derivation is recorded.

Implementation-only choices such as floorplan or tool flow are typed generator
inputs and produce a new HardwareImplementation. A choice that changes a
Fabric-visible timing or capacity fact must instead produce a new Fabric
candidate; the backend cannot hide it as an implementation option.

## Why EDA Is Evaluation

Synthesis, placement, routing, timing, power, area, and FPGA implementation are
expensive providers that create implementation artifacts and Evidence. The
tool adapter uses the common ToolRunner, exact platform and condition inputs,
and shared metric/finding registries. It does not publish a private status or
metric schema.

An unmet timing target is completed adverse Evidence, not an invalid design
artifact. Tool crash, license failure, timeout, unsupported primitive, and
structural invalidity remain distinct outcomes.

## Why Functional Oracles Must Be Independent

A generated checker from the same backend can validate protocol and ABI, but
it cannot independently prove the backend's functional implementation. Mapped
RTL execution must compare requested terminal observations with an independent
DFG or CGRA execution under compatible workload and service contracts.

Raw waveforms and reports are useful diagnostics but remain attempt or scratch
material until an exact raw-bundle owner exists. Human summaries are removable
projections of typed Artifacts and Evidence.

## Why Low- And High-Fidelity FPA Share Metrics

An analytical architecture model and an EDA-backed model differ in method,
uncertainty, cost, and accuracy, not in what frequency, area, or power means.
Both therefore publish ordinary shared MetricResults for limiting frequency,
total area, dynamic power, and leakage power. Model-owned coefficients or EDA
report parsers stay behind their descriptor; neither creates a private FPA
record. This lets calibration compare like quantities while preserving the
lower-confidence model as a fast, complete early-stage estimate.

Low confidence means inaccurate absolute values, not an incomplete question.
The early model still estimates frequency, area, dynamic power, leakage power,
and runtime with coherent relative scaling. Omitting a dimension would force
frontend DSE to use a separate ad hoc score and would make later EDA calibration
change the optimization data model instead of only improving its evidence.
