# Simulation Artifacts

This document is the sole persistent-schema owner for workload-running
simulation inputs and typed execution observations. DFG-sim, CGRA-sim, sys-sim,
and mapped RTL simulation share these Artifact families while retaining their
own execution semantics and time authorities.

## Artifact Families

The fixed schema descriptors are:

```text
loom.simulation_workload      1.0
loom.simulation_runtime_input 1.0
loom.simulation_execution     1.0
```

Each family has one typed C++ model and one canonical serializer/parser.
Schema versions use `X.Y`: `X` denotes an incompatible change and `Y` denotes
a compatible extension. Simulator-specific request, result, trace, activity,
or report Artifact families are forbidden.

## SimulationWorkload

`SimulationWorkload` has two typed roots:

```text
SpatialSimulationWorkload {
  canonical_dataflow_ref
  graph_ref
  logical_invocation
  coordinate_and_parameter_schema
  observable_contract
}

SystemSimulationWorkload {
  deployment_ref
  program_entry
  declared_external_interactions
  observable_contract
}
```

The spatial root describes one logical SpatialCore invocation. The system root
describes one exact deployed program entry. Workloads own shapes, launch
parameters, and observable value, stream, memory, and completion contracts;
they do not own concrete input values or simulator policy.

`canonical_dataflow_ref` is an exact Common Artifact reference to one finalized
`loom.canonical_dataflow 1.0` Artifact. `graph_ref` is the Dataflow-owned typed
`GraphRef` whose artifact component must equal that exact reference. It is not
a symbol, local array index, simulator ID, or Mapping-owned graph number.
DFG-sim can therefore resolve a workload directly from the Dataflow importer;
CGRA-sim additionally validates its exact Mapping and Fabric inputs against the
same Dataflow identity.

## SimulationRuntimeInput

`SimulationRuntimeInput` also has spatial and system roots. Every root refers
to exactly one compatible `SimulationWorkload`.

The spatial root owns concrete values, ordered input streams, dynamic launch
parameters, external arrivals, and the logical-memory registry: object
identities, alias topology, runtime allocations, initial contents, and input
images. The system root owns concrete argv, stdin, external events, and memory
or device input images.

Runtime input does not contain model timing, trace-capture policy, execution
limits, Mapping repairs, physical addresses used only by a simulator, or
presentation options.

## SimulationExecution

Every model that actually runs a software workload produces one
`SimulationExecution` with a spatial or system root:

```text
SimulationExecution {
  evaluation_request_ref
  observable_contract_ref
  terminal
  terminal_observations
  output_values
  output_streams
  visible_logical_memory_state_or_diff
  completion_and_retirement_observations
  activity_summaries
  trace_manifest?
}
```

Its closed terminal algebra is:

```text
Retired
Halted { finding_kind, witness }
StoppedByLimit
```

`Retired` means the workload's completion frontier, observable obligations,
and invocation-local quiescence contract have completed. `Halted` records a
sound execution-level witness for deadlock, trap, combinational
nonconvergence, or proven nontermination. `StoppedByLimit` records that an
execution limit ended observation without such a proof.

Evaluation maps these forms exactly:

```text
Retired        -> Completed with every mandatory terminal finding Absent
Halted         -> Completed with the corresponding mandatory terminal finding
                  Present and every other mandatory terminal finding Absent
StoppedByLimit -> CancelledOrTimeout
```

Every workload-running simulator descriptor declares one typed
`SimulationExecution` output slot and the complete mandatory terminal
`FindingQuery` set. A legal EvaluationRequest must request all of those
findings. The descriptor's closed outcome-cardinality contract requires one
execution output for `Retired` and `Halted`; it governs whether an incomplete
outcome may retain zero or one stopped execution.

Capability rejection before execution is `Unsupported`; simulator, tool, or
adapter failure is `ExecutionFailed`. `SimulationExecution` never owns
normalized metrics, normalized findings, DSE decisions, or human diagnostics.
Those belong only to `EvaluationEvidence` and its registries.

Capability whose absence is discoverable only from runtime values, such as an
unordered conflicting plain-memory access or a required external consistency
behavior not implemented by the exact model, also ends with Evaluation outcome
`Unsupported`. It is not `Halted` and must not fabricate a
`SimulationExecution` result. A genuine dynamic closed wait-set is `Halted`.
A simulator, Bridge, or provider invariant violation is `ExecutionFailed`.
Execution limits and external cancellation remain `StoppedByLimit` or
`CancelledOrTimeout` according to the owning attempt and Evaluation contract.
Static invalid IR, Fabric, or Mapping is rejected before execution and never
becomes dynamic deadlock.

## Trace And Activity

An optional typed trace manifest owns the ordered list of content-addressed
trace-chunk references, time coverage, capture level, and completeness. The
referenced opaque chunk payloads and their inventory belong to an immutable
raw detailed bundle, not to `SimulationExecution` and not to a separate trace
Artifact family. They may be compressed or indexed without changing manifest
order or semantic observations.

When the capture request includes memory-consistency detail, opaque chunks may
record issue, linearization, reads-from, visibility or synchronization, and
retirement observations. These observations are a projection of the exact
execution; they are not a `ConsistencyExecution`, witness, relation, or
simulator-specific Artifact family and are never required for semantic
correctness.

Actor-bearing trace and activity records use the Dataflow-owned `ActorRef`.
Execution-local invocation occurrences and per-actor firing ordinals qualify a
dynamic event but never create persistent Dataflow entities. The complete
persistent record unions and canonical chunk wire remain owned by the
`SimulationWorkload/SimulationExecution persistent wire` design frontier; this
rule fixes only the upstream identity owner.

Trace capture is a nonsemantic invocation binding. Enabling it may change
execution cost and retained raw material, but must not change scheduling,
outputs, terminal form, cycle count, metrics, or findings. The exact capture
request belongs only to `InvocationManifest`. The execution owner's attempt
record references that request and owns attempt provenance and retained-
material references. The `SimulationExecution` trace manifest alone owns the
actual order, coverage, and completeness of captured trace data.

Activity summaries identify their observation window, coverage, scope,
attachment, and timebase. Waveforms and simulator-native activity files are
raw blobs referenced by the execution. They do not form an `ActivityProfile`
Artifact or duplicate final values and memory state.

## Ownership And Coupling

`EvaluationRequest.workload_ref` and `runtime_input_ref` contain exact Artifact
references. A `SimulationExecution` refers only to that exact Request and
recovers both inputs through it rather than copying their references.
`EvaluationEvidence` refers to the Request and binds the execution through the
descriptor-owned output slot without copying its observations. The execution
never refers to Evidence, so the content-identity graph is acyclic.

Architecture-only RTL or EDA checks do not execute a workload and therefore
must not create an empty `SimulationExecution`.

## Anchor Verification

Stable anchors cover exact workload/runtime coupling, spatial and system root
validation, the typed output slot, mandatory terminal-finding totality, the
closed terminal algebra and Evaluation mapping, logical-memory identity
preservation, runtime-dependent `Unsupported` versus `Halted` and
`ExecutionFailed`, trace observer noninterference, and rejection of normalized
results inside `SimulationExecution`. Tests do not pin trace chunk sizes,
report layouts, simulator class hierarchies, or broad workload matrices.
