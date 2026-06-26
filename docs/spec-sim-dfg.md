# DFG-sim

This document specifies Loom DFG-sim, the pure software dataflow
semantic simulator. DFG-sim executes dataflow IR without hardware
resource limits. Its output is the software-semantic baseline for
debugging, testing, PnR cost feedback, CGRA-sim comparison, and DSE.

## Purpose

DFG-sim answers this question:

```text
What does this dataflow program do, and what is its optimistic
unconstrained execution behavior, before hardware mapping is considered?
```

DFG-sim consumes:

* dataflow IR;
* runtime input data;
* an initial memory image or memory model configuration;
* simulator configuration.

Simulator configuration is a typed view of the resolved configuration
specified in `docs/spec-config-ssot.md`. DFG-sim does not own independent
defaults for operation semantics, reciprocal throughput, event limits,
or reporting policy.

DFG-sim produces:

* functional outputs;
* final memory state or memory diffs;
* token and event traces;
* deterministic diagnostics;
* optimistic timing and activity metrics;
* a DFG-sim report usable by comparison tools.

DFG-sim does not consume Fabric ADG and does not consume a mapping
artifact. It does not choose placement, routing, schedule slots,
temporal tags, buffers, memory bindings, or hardware resources.

## Boundary With CGRA-sim

DFG-sim is the software semantic baseline. CGRA-sim is the
hardware-aware simulator specified in `docs/spec-sim-cgra.md`.

DFG-sim and CGRA-sim must agree on functional behavior for the same
software input, runtime data, and legal mapping. Performance differences
are expected because DFG-sim intentionally ignores hardware limits.

The comparison protocol is specified in
`docs/spec-sim-comparison.md`.

## Semantic Scope

The baseline semantic scope includes:

* `dataflow.stream`;
* `dataflow.carry`;
* `dataflow.invariant`;
* `dataflow.gate`;
* `dataflow.constant`;
* `dataflow.load`;
* `dataflow.store`;
* `dataflow.sync`;
* `dataflow.mux`;
* `dataflow.demux`;
* `dataflow.parallelize`;
* `dataflow.pack`;
* `dataflow.unpack`;
* `dataflow.serialize`;
* arithmetic and math operations required by target workloads;
* control, done, and memory-order token behavior required by dataflow
  graph execution.

The simulator must follow the target dataflow specs for sentinel
tokens, body-phase behavior, loop feedback, control tokens, memory
tokens, and graph completion.
Vector token-cardinality changes follow
`docs/spec-dataflow-vectorization.md`.

Unsupported operations must produce structured diagnostics. Unsupported
operations must not be silently approximated.

## Execution Model

DFG-sim uses a deterministic event model:

* every simulated SSA value has a logical token queue;
* every operation fires when its required input tokens are available and
  its semantic guards permit firing;
* memory operations observe explicit memory dependencies, effects, and
  memory-order tokens;
* graph completion is represented by dataflow completion semantics, not
  by hardware completion signals;
* tie-breaking is deterministic for simultaneously fireable events.

The event model is unconstrained by hardware resources. Multiple
operations may fire in the same optimistic step if dataflow semantics
allow it. DFG-sim does not model PE count, FU count, route capacity,
memory port count, buffer depth, clock domains, or protocol latency.

## Memory Model

DFG-sim owns a software memory model for simulated memrefs and memory
regions. It must support:

* initial input buffers;
* reads and writes through `dataflow.load` and `dataflow.store`;
* explicit memory-order tokens and sync operations;
* deterministic conflict handling based on the dataflow memory
  dependence model;
* final memory diffs for checking and reporting.

DFG-sim does not model cache hierarchy, memory bandwidth, coherence
traffic, NoC latency, or terminal memory target range conflicts. Those
are hardware concerns for Fabric ADG, PnR, and CGRA-sim.

## Optimistic Metrics

DFG-sim metrics are software and optimistic. Baseline metrics include:

* functional output values;
* final memory diffs;
* token count per value;
* operation fire count;
* event count;
* maximum logical parallelism;
* critical-path estimate over data dependencies;
* optimistic cycle or step estimate;
* diagnostics and unsupported-operation count.

The optimistic cycle or step estimate is not a hardware cycle count. It
is a lower-bound-style software estimate used for comparison and DSE.

## Determinism

Given the same input IR, runtime data, memory image, and simulator
configuration, DFG-sim must produce the same outputs, diagnostics, and
report. Stable deterministic ordering is required for tests and
comparison.

If the requested simulator configuration contains an unknown key, an
unknown model profile, a conflicting source, or an unrecorded override,
DFG-sim must fail before simulation starts.

## Report Contract

A DFG-sim report must identify:

* software IR root;
* simulator schema version;
* runtime input identity or fingerprint when available;
* simulator configuration;
* resolved configuration identity and fingerprint;
* component configuration-view identity;
* component configuration-view fingerprint;
* functional outputs and memory diffs;
* optimistic metrics;
* trace location or inline trace summary;
* diagnostics.

Reports may be consumed by PnR as cost feedback and by the simulation
comparison protocol. Reports must not contain hardware placement,
routing, schedule, or resource-sharing decisions.

## Non-Goals

DFG-sim is not PnR. It does not map software to hardware.

DFG-sim is not CGRA-sim. It does not model hardware resource limits.

DFG-sim is not RTL simulation. It does not execute generated hardware.

DFG-sim may be selected through the runtime ABI specified in
`docs/spec-runtime-abi.md`, but runtime dispatch does not change the
DFG-sim input boundary: DFG-sim consumes dataflow IR and software
runtime data, not Fabric ADG or mapping artifacts.

## Acceptance Criteria

DFG-sim is complete at the target-spec level when:

* it can execute hand-written dataflow primitive graphs;
* it can execute at least one selected application dataflow graph with
  real input data or a controlled fixture;
* functional outputs match the expected software behavior;
* unsupported operations produce structured diagnostics;
* invalid or conflicting simulator configuration fails early with
  structured diagnostics;
* reports expose functional outputs, memory diffs, token counts, event
  counts, and optimistic metrics;
* reports carry configuration identity, canonical fingerprint, and
  component-view fingerprint;
* the same workload and input can be compared against CGRA-sim through
  `docs/spec-sim-comparison.md`.
