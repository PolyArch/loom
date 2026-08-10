# Simulation Artifacts

This document is the sole persistent-schema owner for workload-running
simulation inputs and typed execution observations. DFG-sim, CGRA-sim, sys-sim,
and mapped RTL simulation share these Artifact families while retaining their
own execution semantics and time authorities.

## Artifact Families

The fixed schema descriptors are:

```text
loom.simulation_workload      1.1
loom.simulation_runtime_input 2.0
loom.simulation_execution     1.0
```

Each family has one typed C++ model and one canonical serializer/parser.
Schema versions use `X.Y`: `X` denotes an incompatible change and `Y` denotes
a compatible extension. Simulator-specific request, result, trace, activity,
or report Artifact families are forbidden.

## SimulationWorkload

`SimulationWorkload` has one closed root union:

```text
SimulationWorkload =
    Spatial(SpatialSimulationWorkload)
  | System(SystemSimulationWorkload)
  | StructuredProgram(StructuredProgramSimulationWorkload)
```

The root discriminants are zero, one, and two in declaration order. Version
1.1 appends `StructuredProgram`; it does not renumber either
`loom.simulation_workload 1.0` root.
The spatial root is:

```text
SpatialSimulationWorkload {
  launch_ref: RootedGraphLaunchRef
  dense_coordinates: array<uint64>
  value_input_plan:
    total table<graph value-input ordinal, SpatialValueInputSource>
  observable_contract: SpatialObservableContract
}

SpatialValueInputSource =
    Fixed(CanonicalValueSequence)
  | Runtime
```

`launch_ref` is the one Dataflow-owned structural reference that recovers the
exact Canonical Dataflow Artifact, root thread launch, static graph launch,
called graph, graph ABI, memory roots, and channel context. The workload must
not copy a `canonical_dataflow_ref`, `GraphRef`, graph symbol, or logical
invocation record.

The dense coordinate count must equal the root thread domain rank. Every
coordinate must be inside any statically known bound.
`loom.simulation_workload 1.0` admits only a
dense rooted launch. A DynamicWork workload is `Unsupported`; a producer must
not invent a string key, provisional occurrence, or channel correspondence to
serialize it.

The value-input table is exactly total over graph value-input ordinals. A
`Fixed` entry contains exactly one semantic token and makes that launch
parameter part of workload identity. A `Runtime` entry delegates only that
token to the exact `SimulationRuntimeInput`. Stream inputs are always concrete
runtime sequences, and imported memory roots are always runtime bindings.
Consequently there is no generic four-way `InputSource` union that repeats the
value, stream, and memory planes already owned by the graph ABI.

The spatial observable contract is:

```text
SpatialObservableContract {
  value_results: canonical set<graph value-result ordinal>
  stream_outputs: canonical set<graph stream-output ordinal>
  memories:
    canonical table<SpatialMemoryObservableTarget, MemoryObservationForm>
}

SpatialMemoryObservableTarget =
    LogicalMemory(LogicalMemoryRootOrViewRef)
  | Exposure(graph memory-result ordinal)

MemoryObservationForm =
    FullState
  | DiffFromRuntimeInput
```

Value and stream ordinals are owner-relative to `launch_ref`; their complete
meanings are the corresponding `GraphLaunchBoundaryTransferRef` and
`ChannelProducerRef` forms. An exposure ordinal has the complete meaning of
the corresponding `MemoryExposureRef`. Types and shapes are recovered from
the exact Dataflow owner and are never copied into the contract. Each
collection is sorted by the canonical typed target key and contains no
duplicates.

Graph completion is mandatory for every workload and derives from
`GraphLaunchBoundaryTransferRef::Done(launch_ref)`. It is not an optional
observable selector. Intermediate completion and retirement activity belongs
to execution trace or activity records rather than this contract.

The system root has the structural shape:

```text
SystemSimulationWorkload {
  program_entry_ref: DeploymentProgramEntryRef
  value_input_plan:
    total table<program value-argument ordinal, SystemValueInputSource>
  external_value_input_plan:
    total table<input or inout DeploymentExternalInterfaceRef,
                SystemValueInputSource>
  observable_contract: SystemObservableContract
}

SystemValueInputSource =
    Fixed(CanonicalValueSequence)
  | Runtime

SystemObservableContract {
  value_results: canonical set<program value-result ordinal>
  external_value_outputs:
    canonical set<output or inout DeploymentExternalInterfaceRef>
  external_stream_outputs:
    canonical set<output or inout DeploymentExternalInterfaceRef>
  memories:
    canonical table<DeploymentExternalInterfaceRef, MemoryObservationForm>
}
```

The program-entry reference already carries the exact Deployment identity, so
there is no second `deployment_ref`. The Deployment family owns both target
catalogs, ordinals, and validation. Every table and selected observable must
match the exact entry or external-interface kind, direction, and semantic type.
Input or inout stream and memory interfaces are always supplied by
SimulationRuntimeInput. Fixed or runtime value selection follows the same
single-token rule as the spatial root. This schema does not copy Deployment
entry or interface fields.

The Structured Program root supplies the source-program workload used by
pre-Mapping Evaluation and compilation DSE:

```text
StructuredProgramSimulationWorkload {
  entry_ref: StructuredEntityRef
  argument_plan:
    total table<entry ABI argument ordinal,
                StructuredProgramArgumentSource>
  observable_contract: StructuredProgramObservableContract
}

StructuredProgramArgumentSource =
    FixedValue(CanonicalValueSequence)
  | RuntimeValue
  | RuntimeMemory

StructuredProgramObservableContract {
  return_value: bool
  memories:
    canonical table<StructuredProgramMemoryTarget, MemoryObservationForm>
}

StructuredProgramMemoryTarget =
    EntryPointerArgument(entry ABI argument ordinal)
  | GlobalObject(StructuredEntityRef)
```

`entry_ref` must resolve through one exact `StructuredProgramCandidateView` to
a defined `llvm.func`. Its parent identity is the sole source-program identity;
the workload must not add a `structured_program_ref`, function symbol, module
path, or operation position. Argument order, argument and return types,
DataLayout, global definitions, and initial global state are recovered from
that exact Structured Program and are never copied into the workload.

The argument plan is exactly total in LLVM ABI order. `FixedValue` and
`RuntimeValue` are legal only for values with a canonical scalar or fixed
vector semantic shape. `FixedValue` contains exactly one token.
`RuntimeMemory` is legal only for an LLVM pointer argument and delegates its
concrete backing object and byte offset to the exact runtime input. Raw pointer
bits are never a value source. Unsupported aggregate ABI values, function
pointers, or other interfaces that lack an exact provider fail closed rather
than acquiring an opaque byte or property-map representation.

`return_value` may be selected only for a non-void return with a canonical
semantic shape. An entry-pointer target must name a `RuntimeMemory` argument.
A global target must resolve to an exact LLVM global object in the same
Structured Program. Memory targets are sorted by their typed key, contain no
duplicates, and use the shared memory observation forms. A global object uses
its exact Structured Program initializer as the runtime baseline; version 1.1
does not add a second mutable-global input authority. A workload that needs a
different initial global state must express that initialization in its exact
entry program.

Entry retirement is mandatory and mechanically derived. It is not an optional
observable field. The Structured Program root is not a test fixture: test
harnesses, command-line tools, and production invocations are merely providers
of concrete instances of this same Artifact contract.

Workloads own fixed problem shape, fixed launch values, and requested
observables. They do not own concrete runtime values, stream contents, memory
images, simulator policy, expected results, trace capture, or execution
limits.

## Canonical Semantic Values

One value representation is shared by fixed workload inputs, runtime value
inputs, runtime stream tokens, functional observations, and semantic trace
publications:

```text
CanonicalValueSequence {
  token_count: uint64
  lanes: array<SemanticLane>
}

SemanticLane =
    Defined(fixed-width semantic bits)
  | Poison
  | Undef
```

The target type is recovered from the owning workload slot, observable target,
or trace token occurrence and is not serialized again. For a `none` token,
each token has zero lanes. For a scalar, each token has one lane. For a
fixed-ranked vector, each token has the product of its dimensions in canonical
row-major lane order. The lane array length must therefore equal
`token_count * lanes_per_token`; for `none`, that product is zero even when
`token_count` is nonzero.

A defined integer, index, or floating lane stores the exact fixed-width
software-semantic bit representation. Variant tags and bits use canonical
big-endian wire order. This is serialization framing, not target memory
endianness, Fabric port layout, vector packing, or a hardware representation.
Host integer and floating types are never semantic authorities. Unknown lane
states, wrong widths, extra lanes, and unresolved target widths are invalid.

For a first-class `!llvm.ptr<AS>` value, defined semantic bits are accompanied
by the runtime-input-owned object ordinal and signed byte offset resolved from
the same object registry used by memory-root bindings. Canonical pointer values
never contain a native process address. Their exact `P(AS)`-bit projection is
validated against the module DataLayout and must round-trip through the owning
pointer codec; poison and undef carry no object target. This is a typed pointer
alternative, not an integer convention or a second alias graph.

## SimulationRuntimeInput

`SimulationRuntimeInput` uses the same root discriminant as its exact workload.
The spatial root is:

```text
SpatialSimulationRuntimeInput {
  workload_ref: exact Spatial SimulationWorkload reference
  runtime_values:
    total table<runtime graph value-input ordinal, CanonicalValueSequence>
  runtime_streams:
    total table<graph stream-input ordinal, CanonicalStreamSequence>
  memory_objects: canonical array<RuntimeMemoryObject>
  memory_root_bindings:
    total table<LogicalMemoryRootRef, RuntimeMemoryRootBinding>
}

CanonicalStreamSequence {
  values: CanonicalValueSequence
  termination: ClosedAfterLast | OpenAfterLast
}

RuntimeMemoryObject {
  byte_count: uint64
  initial_bytes: array<SemanticMemoryByte>
  pointer_values: canonical array<RuntimeMemoryPointer>
}

RuntimeMemoryPointer {
  storage_byte_offset: uint64
  address_space: uint32
  target_object_ordinal: uint64
  target_byte_offset: signed A(address_space)-bit integer
}

SemanticMemoryByte =
    Defined(uint8)
  | Poison
  | Undef

RuntimeMemoryRootBinding {
  object_ordinal: uint64
  byte_offset: uint64
}
```

`runtime_values` is exactly total over value-input ordinals whose workload
source is `Runtime`; every sequence has exactly one token. `runtime_streams`
is exactly total over graph stream-input ordinals. `ClosedAfterLast` publishes
the stream close after its final token. `OpenAfterLast` means that no close is
observed within the sequence's owning observation horizon. For runtime input,
that horizon is the complete supplied input: no later token or close exists,
and the first version rejects timed external arrivals rather than inventing a
generic environment schedule. For execution
output, the horizon ends at the execution terminal, so the same form records
an open produced prefix without asserting a counterfactual future.

Every imported logical-memory root reachable from the selected launch has
exactly one root binding, and no unrelated root may appear. A runtime memory
object is neutral byte-addressed software storage. Its initial-byte count must
equal `byte_count`; the exact Canonical Dataflow type, DataLayout, and
root/view relations alone interpret typed accesses. This avoids choosing one
aliased memref role as a privileged storage type or importing physical memory
layout into the software input.

`pointer_values` is sorted by `storage_byte_offset` and contains no duplicate
or overlapping stored representations. Every record covers exactly
`P(address_space)` defined bits in `initial_bytes`, interpreted with the exact
module DataLayout endianness, and targets an existing canonical object ordinal
with an exact signed `A(address_space)`-bit byte offset. Re-encoding the target
through the object registry must reproduce those stored representation bits.
Poison or undef storage cannot carry pointer provenance. The bytes remain the
content authority; this table supplies only the object provenance that a raw
address representation cannot recover. `loom.simulation_runtime_input 2.0`
adds this table and is therefore incompatible with that family's 1.x schema.

For a launch-derived imported linear view, the runtime input still binds only
the owning `LogicalMemoryRootRef` to an object and byte offset. The simulator
resolves the Dataflow-owned launch relation and instantiates the graph formal's
typed view over that same byte object. It does not serialize a native pointer,
duplicate a view binding, or execute a conversion actor. Several formal views
may therefore share one object ordinal while retaining their distinct exact
memref element types.

Objects have no author-selected persistent IDs. Before serialization, roots
that share one runtime object form an equivalence class. The canonical key for
that object is the sorted non-empty list of `(LogicalMemoryRootRef,
byte_offset)` bindings. Objects are sorted by this key and receive their
zero-based `object_ordinal` from that order. Binding two or more roots to one
ordinal expresses aliasing; no separate alias graph is serialized. Missing,
empty, duplicate, out-of-range, or unreferenced objects are invalid. Overlap
between two root ranges bound to the same object is legal aliasing and must
not be rejected merely because their ranges intersect.

Canonical object reordering also remaps every `target_object_ordinal` in stored
pointer records. A finalizer must not preserve draft-local ordinals or native
addresses in either the bytes or provenance table.

A fresh graph allocation is not an imported runtime object. Its execution
identity derives from its static `LogicalMemoryRootRef` and graph-invocation
occurrence, and its storage is created under the owning software semantics.
It therefore does not receive a workload-local object ID or root binding.

The system runtime root is:

```text
SystemSimulationRuntimeInput {
  workload_ref: exact System SimulationWorkload reference
  runtime_entry_values:
    total table<runtime program value-argument ordinal,
                CanonicalValueSequence>
  runtime_external_values:
    total table<runtime DeploymentExternalInterfaceRef,
                CanonicalValueSequence>
  external_stream_inputs:
    total table<input or inout DeploymentExternalInterfaceRef,
                CanonicalStreamSequence>
  memory_objects: canonical array<RuntimeMemoryObject>
  memory_interface_bindings:
    total table<DeploymentExternalInterfaceRef, RuntimeMemoryRootBinding>
}
```

The two value tables are exactly total over `Runtime` selections and each
sequence contains one token. Stream inputs are finite untimed sequences with
the same closed/open-after-last semantics as the spatial root. Memory bindings
are exactly total over every selected memory observable and every input or
inout memory interface required by the entry. Their object canonicalization,
aliasing, baseline, and range rules are the same as spatial memory objects.

A producer must not substitute argv arrays, device names, string-key maps, or
simulator-private event records for these typed interfaces.

The Structured Program runtime root is:

```text
StructuredProgramSimulationRuntimeInput {
  workload_ref: exact Structured Program SimulationWorkload reference
  runtime_values:
    total table<runtime-value entry argument ordinal,
                CanonicalValueSequence>
  memory_objects: canonical array<RuntimeMemoryObject>
  pointer_bindings:
    total table<runtime-memory entry argument ordinal,
                RuntimeMemoryRootBinding>
}
```

Every runtime value contains exactly one token of the type recovered from the
entry ABI. Every `RuntimeMemory` argument has exactly one pointer binding and
no other argument may have one. Sharing one object ordinal expresses aliasing
between entry pointers. Objects are ordered by their sorted nonempty
`(argument ordinal, byte_offset)` binding keys, exactly as spatial objects are
ordered by their logical-root binding keys. Missing, duplicate, out-of-range,
empty, unreferenced, or noncanonical objects and bindings are invalid. The
object byte count is the finite runtime extent of that pointer input; typed
accesses outside it fail execution rather than extending storage or wrapping.

The Structured Program runtime root contains no native address, argv
projection, process-global state, call-path identity, profile counters, or
candidate-specific graph input. Candidate-specific Spatial workloads and
runtime inputs are derived only after a Canonical Dataflow Program exists and
are used for exact graph replay; they do not replace this source workload in a
whole-program Evaluation case.

Runtime input does not contain model timing, trace-capture policy, execution
limits, Mapping repairs, physical addresses used only by a simulator, or
presentation options.

## Canonical Workload And Input Wire

Every displayed union assigns zero-based unsigned 32-bit discriminants in
declaration order. Unsigned counts and ordinals use 64-bit big-endian values.
Exact Artifact identities use their fixed 32-byte representation. Arrays and
tables use a 64-bit element count followed by elements in canonical order.
Nested Dataflow and Deployment references use their owning canonical
encodings; this schema does not renumber them.

Records encode fields in declaration order without names, optional property
maps, padding, or native C++ layout. Tables are sorted by their typed semantic
keys. Parsers reject duplicate or unsorted keys, unknown variants, missing or
extra fields, noncanonical bytes, unresolved references, and any cardinality
or owner mismatch. JSON, MLIR text, CLI syntax, and visualization files are
derived projections and cannot be accepted as identity bytes.

The version-1.1 Structured Program workload encodes root discriminant two,
followed by the production `StructuredEntityRef` encoding, the dense argument
plan, the return selector, and the canonical memory-observable table. Its
runtime input encodes root discriminant two, the exact workload identity,
runtime values, canonical memory objects, and pointer bindings in declaration
order. The owner codecs for `StructuredEntityRef`, semantic values, memory
bytes, and Artifact identities are reused directly; this family must not copy
their formulas.

## Spatial Vecadd Example

For a rooted graph launch whose value-input ordinal zero is `N`, whose imported
logical-memory roots are `A`, `B`, and `C`, and whose computation is
`C[i] = A[i] + B[i]`, the workload contains:

```text
launch_ref = exact rooted vecadd graph launch
dense_coordinates = the selected logical thread point
value_input_plan = {
  0 -> Fixed(one defined index token with value 1024)
}
observable_contract = {
  value_results = {}
  stream_outputs = {}
  memories = {
    LogicalMemory(Root(C)) -> DiffFromRuntimeInput
  }
}
```

The runtime input contains no runtime value or stream entries. It provides
three 4096-byte objects for `A`, `B`, and `C`, binds each root to its object at
byte offset zero, and initializes the bytes using the software DataLayout for
1024 `f32` elements. The serialized object ordinals derive from the sorted
root-binding keys rather than the source names `A`, `B`, and `C`. Binding `A`
and `B` to one object ordinal would instead state that those two imported
roles alias; no other record changes.

## SimulationExecution Root, Coupling, And Terminal

Every model that actually runs a software workload produces one
`SimulationExecution` with one untagged root:

```text
SimulationExecution {
  request_ref: exact EvaluationRequest reference
  terminal: ExecutionTerminal
  functional_observations
  progress_observations
  activity_summaries
}
```

The root does not carry a `spatial` or `system` discriminator. The exact
Request already binds the model descriptor and workload, and the workload
root owns that distinction. Repeating it in the execution would create a
second case-kind authority.

The C++ `SpatialSimulationExecution` authoring value is shared by DFG, CGRA,
mapped RTL, and mapped gate-netlist models that execute a Spatial workload.
`Spatial` names the workload-observation form, not the selected engine or
fidelity. An external-HDL evaluator constructs this same value and uses the
ordinary finalizer; it must not add an RTL-specific execution subtype, engine
tag, or parallel serializer.

The execution also has no direct observable-contract reference. It recovers
the applicable contract through:

```text
request_ref
  -> EvaluationRequest.workload_ref
  -> SimulationWorkload observable contract
```

The root field order, terminal record, Spatial and System functional and
progress observations, and activity summaries are closed below. Together they
define the complete `loom.simulation_execution 1.0` wire. The invocation-local
typed Spatial diagnostic algebra defined below has no field in that Artifact
root.

The closed terminal algebra is:

```text
ExecutionTerminal =
    Retired
  | Halted {
      finding_kind: FindingKind
      witness: registry-defined terminal-witness payload
    }
  | StoppedByLimit
```

`Retired` means the workload's completion frontier, observable obligations,
and invocation-local quiescence contract have completed. `Halted` records a
sound execution-level witness for deadlock, trap, combinational
nonconvergence, or proven nontermination. `StoppedByLimit` records that an
execution limit ended observation without such a proof.

The `FindingKind` registry is the sole owner of the witness payload schema.
The `SimulationExecution` terminal is the sole owner of the concrete witness
instance. There is no separate `terminal_observations` collection and
Evaluation Evidence must not copy the witness.

The terminal wire uses zero-based `u32be` discriminants. `Retired` is the
single word `u32be(0)`. `Halted` is `u32be(1)`, followed by
`u32be(finding_kind.ordinal)`, `u64be(witness_byte_count)`, and exactly that
many canonical witness bytes from the registered
`FindingTerminalWitnessCodec`. `StoppedByLimit` is the single word
`u32be(2)`. A `Halted` decoder resolves the exact Finding descriptor, invokes
its terminal-witness decoder and validator in the exact Request closure,
re-encodes the adopted value, and requires byte equality. An unknown kind,
missing or incomplete codec, noncanonical payload, count overflow, truncation,
or trailing payload bytes is invalid.

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
findings. The descriptor's closed outcome-cardinality contract requires
exactly one execution output for `Retired` and `Halted`. For
`StoppedByLimit`, it alone declares whether `CancelledOrTimeout` may retain
zero or one stopped execution. `Unsupported` and `ExecutionFailed` bind no
`SimulationExecution`.

For the finding corresponding to a `Halted` terminal,
`EvaluationEvidence.Present` carries one occurrence of:

```text
TerminalWitnessRef {
  execution_output_slot_ref: ModelOutputSlotRef
  execution_output_ordinal: uint64
}
```

Simulation Artifacts owns this occurrence payload codec. Its canonical bytes
are exactly `u32be(execution_output_slot_ref.ordinal)` followed by
`u64be(execution_output_ordinal)`. The terminal FindingKind descriptors
reference this codec through their `FindingOccurrenceCodec`; the codec schema
descriptor is exactly `loom.terminal_witness_ref 1.0`. They do not copy or
reinterpret the fields. The containing Evidence result's
`FindingRequestOrdinal` recovers the FindingKind, so the payload carries no kind
tag, Request reference, direct SimulationExecution reference, or witness
ordinal.

The reference resolves through the containing Evidence's `output_bindings`.
Validation requires that the slot schema is `SimulationExecution`, the
ordinal exists, the referenced execution names the same exact Request, and
its terminal is `Halted` with the same `FindingKind`. The singleton terminal
witness needs no additional witness ordinal. Every other mandatory terminal
finding is `Absent`.

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

## Spatial Functional Observations

The spatial workload's observable contract is the sole owner of selected
targets, target order, types, and memory observation forms. A spatial
execution therefore stores three positional arrays:

```text
SpatialFunctionalObservations {
  value_results: array<ValueResultObservation>
  stream_outputs: array<CanonicalStreamSequence>
  memories: array<MemoryObservationPayload>
}

ValueResultObservation =
    Published(CanonicalValueSequence)
  | NotPublished
```

The arrays align exactly with the canonical `value_results`,
`stream_outputs`, and `memories` collections recovered through the execution's
Request and workload. They do not repeat graph ordinals, memory targets,
types, or `MemoryObservationForm` tags. A published graph value contains
exactly one token. A stream may contain any token count, including an empty
open prefix.

The workload-selected memory form determines the corresponding payload
without a second persistent discriminator:

```text
FullState payload {
  byte_count: uint64
  bytes: array<SemanticMemoryByte>
}

DiffFromRuntimeInput payload {
  byte_count: uint64
  runs: array<MemoryDiffRun>
}

MemoryDiffRun {
  byte_offset: uint64
  changed_bytes: nonempty array<SemanticMemoryByte>
}
```

A full-state byte array has exactly `byte_count` elements. Diff runs are
sorted by offset, in range, nonoverlapping, nonadjacent, and maximal. Every
encoded byte differs semantically from the exact runtime baseline; adjacent
changed bytes belong to one run. An empty run array uniquely means no change.
Equality distinguishes `Defined`, `Poison`, and `Undef` states as well as
defined byte values.

`DiffFromRuntimeInput` is legal only when the exact Dataflow target and
RuntimeInput mechanically establish one unique baseline range before
simulation. Request verification rejects a target without that correspondence
or with a mismatched extent. A fresh allocation, dynamically unresolved
exposure, MMIO effect, or other target without a runtime byte baseline must
use `FullState` when the exact model supports snapshot semantics.

Each memory entry is the requested target-relative projection of one execution
state. No execution-local memory-object ID or alias graph is added. If selected
targets overlap through relations mechanically recoverable from Dataflow and
RuntimeInput, finalization verifies that their overlapping semantic bytes
agree. This check does not make either projection the storage authority.

All three arrays are exactly total over the selected targets for every
retained execution:

* `Retired` requires every value to be `Published`, every selected stream to
  be `ClosedAfterLast`, and every memory payload to describe final visible
  logical state.
* `Halted` permits a value to be `NotPublished` and a stream to contain an
  open prefix. Memory payloads describe exact logical state visible at the
  halt coordinate.
* a retained `StoppedByLimit` execution uses the same partial-observation
  forms at its stop coordinate.

There is no generic `Complete | Partial | Unavailable` wrapper. Value
publication, stream closure, and memory visibility already provide the three
planes' exact completion facts. The Request verifier rejects a model whose
descriptor statically cannot observe a selected target. Runtime-dependent
absence of required capability produces `Unsupported`; provider failure
produces `ExecutionFailed`. Neither outcome creates an execution containing
an unavailable placeholder.

## System Functional Observations

The system workload's exact Deployment-owned catalogs and observable contract
determine four positional arrays:

```text
SystemFunctionalObservations {
  value_results: array<ValueResultObservation>
  external_value_outputs: array<ValueResultObservation>
  external_stream_outputs: array<CanonicalStreamSequence>
  memories: array<MemoryObservationPayload>
}
```

The arrays align with the canonical selections in `SystemObservableContract`
and repeat no ordinals, interface references, kinds, directions, types, or
memory-form tags. The Spatial definitions of `ValueResultObservation`,
`MemoryObservationPayload`, full state, diff runs, and baseline eligibility
apply unchanged.

`Retired` requires every selected value to be published, every selected stream
to be closed, and every memory payload to describe final externally visible
state. `Halted` and a retained `StoppedByLimit` may contain unpublished values,
open stream prefixes, and memory state at the terminal coordinate. Aliased or
overlapping memory projections must agree. Provider-private process exit data,
argv, file descriptors, device names, and gem5 statistics are not functional
observations.

## Spatial Progress Observations

Every retained Spatial execution stores exactly three progress anchors:

```text
SpatialProgressObservations {
  launch_accepted: EventCoordinate
  graph_retirement_visible: optional<EventCoordinate>
  terminal_observed: EventCoordinate
}

EventCoordinate {
  reference_cycle: ExactRatio
  delta: uint64
}
```

`launch_accepted` is the coordinate at which the rooted Spatial Launch is
accepted. `graph_retirement_visible` is the coordinate at which the graph
completion frontier becomes visible at that launch boundary.
`terminal_observed` is the observation horizon at which the execution
terminal is fixed after the last fully committed atomic transition. A stop
limit cannot leave half of a transition in the retained execution.

All persistent cycle coordinates use the one canonical `ExactRatio` schema.
An integral cycle `N` has the sole encoding `N/1`; there is no integer-or-ratio
union. `delta` orders causal propagation within one reference cycle and never
adds elapsed cycles. Coordinates compare lexicographically by exact
`reference_cycle` and then `delta`, and finalization requires:

```text
launch_accepted
  <= graph_retirement_visible, when present
  <= terminal_observed
```

The progress record encodes fields in declaration order. Each coordinate
encodes the canonical `ExactRatio` numerator and denominator followed by
`delta`, all as unsigned 64-bit big-endian values. The optional retirement
anchor uses a zero-based unsigned 32-bit `Absent | Present` discriminant and,
when present, one coordinate payload. Native layout, field names, alternate
integer coordinates, and non-reduced ratios are invalid identity bytes.

`Retired` requires `graph_retirement_visible`. `Halted` and a retained
`StoppedByLimit` permit it to be absent or present. For example, an execution
may expose graph retirement and complete all functional observations, then
halt while its mapped hardware fails to restore invocation-local quiescence.
Its terminal remains `Halted`, its graph-retirement anchor remains present,
and its terminal coordinate is later.

The progress record stores no elapsed-cycle field, cycle-count metric,
wall-clock duration, event count, completion Boolean, timebase descriptor, or
clock-domain copy. DFG coordinates use the `AbstractCycle` owned by the exact
DFG timing-model descriptor. CGRA coordinates use the reference domain
mechanically derived from the exact Fabric, complete SpatialMapping, and
mapped Spatial Launch boundary. A relative clock schedule not hard-fixed by
Fabric remains an EvaluationRequest base condition.

Evaluation derives requested timing observations from these anchors rather
than copying them into the execution. A compatible retired cycle observation
uses the exact reference-cycle difference from `launch_accepted` to
`graph_retirement_visible`; `delta` is excluded. For `Halted`, a model may
derive a censored lower bound from `launch_accepted` to `terminal_observed`
only when the requested metric definition accepts that reference domain. A
retained `StoppedByLimit` preserves its coordinates, but its
`CancelledOrTimeout` Evidence has no normalized metric results.

## System Progress Observations

```text
SystemProgressObservations {
  program_entry_accepted: SystemEventCoordinate
  program_exit_visible: optional<SystemEventCoordinate>
  terminal_observed: SystemEventCoordinate
}

SystemEventCoordinate {
  gem5_tick: uint64
  delta: uint64
}
```

`gem5_tick` is the exact integer time coordinate of the gem5 EventQueue selected
by the request's Gem5SimulationBinding and model binding. `delta` gives a
deterministic commit order within one tick and does not add elapsed time. The
coordinates compare lexicographically. `Retired` requires
`program_exit_visible`; `Halted` and retained `StoppedByLimit` permit it to be
absent or present. In every case:

```text
program_entry_accepted
  <= program_exit_visible, when present
  <= terminal_observed
```

These anchors do not copy a tick frequency, wall time, exit-code policy, or
gem5 event priority. Evaluation derives metrics through the exact model. The
first System wire does not standardize a typed gem5 event trace; its
root has no trace field and raw gem5 traces remain attempt or scratch material.
DFG and CGRA diagnostic traces use the current typed Spatial event algebra
below only outside `loom.simulation_execution 1.0` identity.

## Activity Summaries

Activity summaries retain exact aggregate execution facts. They do not retain
per-occurrence events, normalized metrics, probabilities, or backend-native
toggle files. The closed record is:

```text
ActivitySummary {
  window: ActivityWindow
  coverage: ActivityCoverage
  payload:
      ActorTransitions {
        transitions:
          canonical table<ActorRef, ActorTransitionCounts>
      }
    | FabricResources {
        use_counts:
          canonical table<FabricUsePatternRef, uint64>
        resource_occupancy:
          canonical table<FabricResourceStateRef,
                          FabricResourceOccupancy>
      }
    | ImplementationSignals {
        signals:
          canonical table<HardwareImplementationActivityPointRef,
                          SignalActivity>
      }
}

ActivityWindow =
    LaunchToGraphRetirement
  | LaunchToTerminal

ActivityCoverage =
    Complete
  | Partial

ActorTransitionCounts {
  committed_firings: uint64
  retired_firings: uint64
}

FabricResourceOccupancy {
  occupied_capacity_reference_cycles: ExactRatio
  peak_occupied_capacity: uint64
}

SignalActivity {
  state_residency_reference_cycles: array<ExactRatio, 4>
  transition_counts: array<uint64, 16>
}
```

`LaunchToGraphRetirement` begins immediately after `launch_accepted` and ends
after `graph_retirement_visible`; it is legal only when that optional progress
anchor is present. `LaunchToTerminal` begins immediately after
`launch_accepted` and ends after `terminal_observed`. Point events at the
closing anchor are included. Integrated residency uses the exact
reference-cycle difference between the two anchors; `delta` orders causal
events but contributes no duration. No activity record copies coordinates,
clock domains, timebase descriptors, or elapsed-cycle metrics.

`Complete` and `Partial` describe target-inventory coverage, not temporal
sampling. Every retained value in a summary must have been observed
continuously and exactly across its complete selected window. Sampling,
gapped capture, extrapolation, and statistical estimates remain raw material
or Evaluation observations and cannot be serialized as an exact
`ActivitySummary`.

For `ActorTransitions`, the complete inventory is every canonical actor in the
rooted launch closure. A complete table contains one entry for every actor,
including explicit zero counts. A partial table contains exactly the actors
continuously observed; a missing actor is unknown, never zero. Counts aggregate
all dynamic invocation occurrences of one static actor. Per-occurrence and
per-cycle distinctions belong to trace. For every entry:

```text
committed_firings >= retired_firings
```

For a `Retired` execution over `LaunchToTerminal`, every committed firing must
also have retired. A model may retain distinct counts for a stopped or halted
execution.

For `FabricResources`, the complete inventories are all applicable,
activity-capable `FabricUsePatternRef` and `FabricResourceStateRef` values in
the exact mapped SpatialCore closure. Complete tables include explicit zero
entries. Partial tables contain exactly the continuously observed targets.
`use_counts` records selected use-pattern activations.
`occupied_capacity_reference_cycles` is the nonnegative exact integral of
occupied capacity over the selected window. `peak_occupied_capacity` cannot
exceed the capacity owned by the referenced Fabric resource state, and the
integral cannot exceed window duration multiplied by that capacity. The
execution does not store utilization, stall cost, energy, or any other derived
quantity.

For `ImplementationSignals`, the complete inventory is the exact activity-point
catalog owned by the selected `HardwareImplementation` and applicable to this
execution. `HardwareImplementationActivityPointRef` is an exact
implementation-owned reference to one observable scalar signal bit. The full
catalog, its deterministic order, and its correlation to Fabric are owned by
`HardwareImplementation`, not by simulation. Until that catalog is finalized,
an implementation-level producer must fail closed rather than invent names,
paths, or provisional IDs.

Signal state order is fixed as `T0, T1, TX, TZ`. The sixteen transition counts
use row-major source-state/destination-state order. Diagonal counts are zero
because a transition is a state change, and the four exact residency values
sum to the selected window duration. This basis permits mechanical SAIF or
toggle-table projection without making HDL hierarchy names semantic.
Waveforms, VCD, FSDB, raw SAIF, and simulator-native activity files remain raw
owner-attempt or scratch material. No current raw-bundle Artifact schema is
implied.

An activity summary must contain at least one target entry across its payload;
otherwise it is omitted. `activity_summaries` may therefore be empty. It
contains at most one summary for each `(ActivityWindow, payload kind)` pair and
sorts by those zero-based enum discriminants. This order mechanically defines
the `activity_summary_ordinal` used by Evaluation. Within each table, keys sort
by their owner-defined canonical reference bytes.

The schema-1.0 activity payloads and windows above are Spatial-only because
their duration-bearing fields use the selected Spatial reference cycle.
`SystemSimulationExecution.activity_summaries` is therefore required to be
empty. A later System activity payload must name its gem5-tick or hardware
clock-domain time basis explicitly; it cannot reinterpret these exact-ratio
cycle integrals or copy raw gem5 statistics into this root.

This specification is the semantic owner contract consumed by
`ActivityBinding.ExecutionActivity`. The `loom.simulation_execution 1.0` root,
publisher, and importer are current owners, but Evaluation consumption also
requires an activity-summary adopter, ordinal resolver, same-Request validator,
and exact source-to-target lineage adapter. Until that adapter is registered,
the Evaluation source form is owner-unavailable and every attempted authoring
or import fails closed. Parsing the outer Evaluation wire does not prove that
the adapter is available.

The summary wire encodes, in declaration order, the zero-based window,
coverage, and payload discriminants as unsigned 32-bit big-endian values,
followed by the selected payload. Tables encode an unsigned 64-bit big-endian
count followed by sorted key/value entries. Counts and capacities are unsigned
64-bit big-endian values. Exact ratios and nested references use their sole
owner-defined canonical encodings. Fixed-size state and transition arrays
carry no redundant length. Unknown fields, duplicate keys, alternate table
orders, native layout, and noncanonical ratios are invalid identity bytes.

The exact Request and typed payload determine source attachment:

* actor references must belong to the exact rooted Dataflow launch;
* Fabric references require an exact complete SpatialMapping for that
  Dataflow program and Fabric; and
* implementation references require an exact HardwareImplementation and
  Mapping/Deployment lineage that reaches the same execution.

A summary whose source basis cannot be proven from the Request closure is
invalid. An `ActivityBinding.target` is instead the destination Evaluation
target to which an evaluator projects the selected summary; it is not a second
source attachment. A capture request is a nonsemantic invocation binding.
Enabling activity capture cannot change scheduling, outputs, terminal form,
progress anchors, normalized metrics, or findings.

## Invocation-Local Spatial Diagnostic Trace

`loom.simulation_execution 1.0` contains no trace field. The current trace is
an invocation-local `SpatialDiagnosticTrace`: it has no Artifact identity,
persistent wire, manifest, chunk, coverage claim, or Evidence reference. It
may be retained only in the current attempt context or scratch material and
never enters SimulationExecution or Evaluation identity.

```text
TraceCaptureLevel = Firing | Semantic | Microarchitecture

SpatialDiagnosticTrace {
  level: TraceCaptureLevel
  frames: array<SpatialTraceFrame>
}

SpatialTraceFrame {
  coordinate: EventCoordinate
  events: nonempty canonical array<SpatialTraceEvent>
}
```

A trace may contain no frames. Each appended frame is nonempty, has a
coordinate strictly later than the preceding frame, contains only events
admitted by the selected level, and is sorted by the one canonical diagnostic
event key below. No path, opaque payload, unchecked Artifact reference, or
claim of complete or prefix coverage may be inferred from this diagnostic
value.

## Typed Diagnostic Events

Trace events use a small closed lifecycle and relation algebra over references
owned by Dataflow, Fabric, Mapping, and the exact execution. They do not use a
generic `kind + properties` record, per-operation event classes, simulator
extension maps, physical Tags as software identities, or a second persistent
entity catalog.

### Dynamic Occurrence References

The execution-local reference families are:

```text
GraphInvocationOccurrenceRef =
  invocation ordinal under the exact SpatialSimulationWorkload

ActorTransitionOccurrenceRef =
  (GraphInvocationOccurrenceRef, ActorRef, transition_ordinal)

TokenOccurrenceRef =
    GraphIngress(
      GraphInvocationOccurrenceRef,
      GraphIngressTokenRef,
      producer_sequence_ordinal)
  | ActorResult(
      ActorTransitionOccurrenceRef,
      result_ordinal,
      producer_sequence_ordinal)

MemoryActionOccurrenceRef =
  (ActorTransitionOccurrenceRef,
      ActorWide
    | Lane(row_major_ordinal))

PhysicalActionOccurrenceRef =
  (Transition(ActorTransitionOccurrenceRef)
   | Token(TokenOccurrenceRef),
   local_action_ordinal)
```

All ordinals are unsigned 64-bit semantic values. A graph-invocation ordinal
is dense in deterministic launch-acceptance order under the exact rooted
Spatial workload. `ActorRef` must belong to the graph selected by that
invocation. Transition ordinals are dense per
`(GraphInvocationOccurrenceRef, ActorRef)` and are allocated when a complete
semantic transition is first formed, before physical admission. A pending
transition can therefore be referenced while stalled. An ordinal is never
cancelled, reused, or reassigned; once its transition commits, it is also the
actor's firing ordinal.

Producer sequence ordinals are dense in the ordered Dataflow sequence emitted
by one graph-ingress or actor-result endpoint. `result_ordinal` must name a
token-plane result owned by the referenced actor. `ActorWide` names one
scalar, plain-vector, `WholePayload`, or fence memory action. `Lane` is used
only for an active `PerLane` memory action and names its canonical row-major
software lane. Inactive lanes create no lane action. An all-zero masked vector
memory transition creates no `MemoryActionOccurrenceRef` or
`MemoryLinearized` event, although its actor commit, publication, and
retirement lifecycle remains visible at the selected levels.

`local_action_ordinal` is dense from zero in the canonical physical-action
order mechanically derived for one transition or token by the exact Fabric
and complete SpatialMapping. It is not a resource ID, route index, Physical
Tag, instruction slot, or simulator container position. These occurrence
references are local to one `SimulationExecution`; none is a Dataflow entity,
Mapping entity, or independently referenceable Artifact object.

### Event Algebra

The closed event union is:

```text
SpatialTraceEvent =
    ActorCommitted(ActorTransitionOccurrenceRef)
  | ActorRetired(ActorTransitionOccurrenceRef)
  | TokenPublished(
      TokenOccurrenceRef,
      value: CanonicalValueSequence)
  | MemoryLinearized(
      MemoryActionOccurrenceRef,
      reads_from: optional<MemoryVersionRef>,
      modification_predecessor: optional<MemoryVersionRef>,
      sequentially_consistent_predecessor:
        optional<MemoryActionOccurrenceRef>)
  | PhysicalRequested(
      PhysicalActionOccurrenceRef,
      target: PhysicalActionTarget)
  | PhysicalGranted(PhysicalActionOccurrenceRef)
  | PhysicalRetired(PhysicalActionOccurrenceRef)

MemoryVersionRef =
    Initial
  | WrittenBy(MemoryActionOccurrenceRef)

PhysicalActionTarget =
    Use(FabricUsePatternRef)
  | Transfer(
      traversals: nonempty canonical set<FabricPhysicalTraversalRef>,
      use_patterns: canonical set<FabricUsePatternRef>)
```

Every `TokenPublished.value` contains exactly one semantic token. Its type is
recovered from the referenced endpoint and is not repeated. A `none` token has
`token_count = 1` and an empty lane array. Defined bits, poison, undef, fixed
vector lane order, and width validation use the sole `CanonicalValueSequence`
contract above. Semantic capture never has a `capture_values` switch.
These occurrence values and the execution's terminal functional observations
have one owner, `SimulationExecution`, and must agree wherever their horizons
overlap. Graph-ingress publications likewise agree with the exact workload and
runtime input. The trace supplies occurrence history; it does not redefine the
terminal value or stream projection.

`MemoryLinearized` records only primitive dynamic relations that cannot be
recovered from program order and the exact actor contract:

* an atomic load has `reads_from`;
* an atomic store has `modification_predecessor`;
* an atomic RMW and a successful compare-exchange have both;
* a failed compare-exchange has only `reads_from`;
* plain accesses and fences have neither object relation; and
* a `seq_cst` operation or fence names its predecessor in the exact
  sequentially-consistent order when one exists.

`Initial` is relative to the exact dynamic atomic object of the containing
action. `WrittenBy` must name an action that writes that same object. A
`PerLane` actor emits one relation record for each active lane; scalar,
plain-vector, `WholePayload`, and fence actors use `ActorWide`. The actor
contract and outcome determine which optional fields are required or
forbidden. Modification order, reads-from, and the sequentially-consistent
predecessor are diagnostic primitive relations. Synchronizes-with,
happens-before, release visibility, and acquire visibility are derived
mechanically and must not be copied into the trace.

`PhysicalActionTarget::Use` names one Fabric-owned atomic use pattern.
`Transfer` names the exact selected directed traversal set and the canonical
set of Fabric-owned use patterns acquired by that one atomic action. A direct
contention-free point transfer has an empty pattern set. A traversal with one
resource pattern has a singleton set. A broadcast carries all branch
traversals and every per-traversal pattern joined by its derived atomic
activation. A compute or memory resource action uses `Use`. The pattern set is
the execution-local projection of the exact Fabric, traversals, and complete
SpatialMapping; it is not a second grouping authority. A producer cannot omit
a selected pattern, merge unrelated actions, or invent another resource
grouping.

There is no independent stall, queue-change, occupancy, state-transition,
Tag, configuration, or blocker event. The interval from `PhysicalRequested`
to `PhysicalGranted` is the exact stall interval; equal coordinates mean no
stall. The referenced Fabric use-pattern contract and exact execution derive
resource-state and queue effects. `PhysicalRetired` closes the action. A
prefix, `Halted`, or retained `StoppedByLimit` execution may end with a request
or grant still open, but a `Retired` execution cannot retain an unretired
required physical action.

### Capture Levels

Each variant has one fixed minimum level:

```text
Firing:
  ActorCommitted
  ActorRetired

Semantic:
  every Firing event
  TokenPublished
  MemoryLinearized

Microarchitecture:
  every Semantic event
  PhysicalRequested
  PhysicalGranted
  PhysicalRetired
```

A retained level contains every covered event whose minimum level is no
greater than that level. There is no event-local level, filter DSL, or
independent flag combination. DFG-sim supports `Firing` and `Semantic`; a
request for `Microarchitecture` is `Unsupported`. CGRA-sim supports all three
when the exact `{Dataflow, Fabric, complete SpatialMapping}` closure provides
the required physical references. System and mapped-RTL diagnostic production
do not create new event variants. System diagnostic trace capture is
unsupported in the current contract. Mapped RTL may emit this invocation-local
Spatial algebra only when its exact Deployment
and HardwareImplementation activity catalog provide a total correlation to the
required Dataflow, Fabric, and Mapping references; otherwise trace capture,
not workload execution, is `Unsupported`.

Launch, graph-retirement, and terminal markers are not
`SpatialTraceEvent` variants. They are mechanically projected from
`SpatialProgressObservations`. A viewer
may render those markers together with a firing replay, but neither producer
nor viewer serializes duplicate boundary events.

### Canonical Diagnostic Ordering And Validation

The seven `SpatialTraceEvent` variants use zero-based declaration-order
discriminants for capture-level membership and canonical diagnostic keys.
There is no persistent event-record encoding. References and values remain
typed invocation-local values, and their owner-defined canonical bytes are
used only where required to construct the deterministic key.

The canonical event key is:

```text
(event discriminant, canonical primary occurrence reference)
```

The primary reference is the transition for actor lifecycle events, token for
publication, memory action for linearization, and physical action for physical
lifecycle events. Observational payload is excluded. Events in one frame sort
lexicographically by this key, and a duplicate key is invalid. Distinct
physical lifecycle phases remain distinct because their discriminants differ.
This order is serialization only; it does not define causality, arbitration,
memory order, or physical progress.

Finalization validates at least:

* every occurrence belongs to the exact Request, Spatial workload, rooted
  graph invocation, and execution horizon;
* transition and producer-sequence ordinals are dense, deterministic, and
  consistent with actor commit, retirement, and ordered-token publication;
* `ActorCommitted` precedes or equals `ActorRetired`, and every published
  result or memory action belongs to the named transition; a complete Firing
  projection of a `Retired` execution contains one retirement for every
  committed transition;
* token result ordinal, type, semantic width, lane count, and one-token
  cardinality are exact;
* memory relation fields match the actor kind, outcome, granularity, exact
  dynamic atomic object, and valid acyclic or total order required by the
  software contract;
* physical targets belong to the exact Fabric and selected complete
  SpatialMapping, transfer traversals belong to the selected route, and any
  use pattern admits the claimed atomic grouping;
* each physical action follows request, grant, retirement order without a
  missing request or duplicate lifecycle event;
* a higher capture level contains the exact lower-level projection over the
  same covered interval;
* token publications agree with input and functional-observation projections,
  and any activity-summary facts observed in the diagnostic trace agree over
  their shared horizon without treating omitted events as a coverage claim;
  and
* no string key, property map, opaque extension payload, event-local
  coordinate, simulator-private ID, or copied owner fact appears.

For example, a mapped vector add may retain:

```text
(5,0) PhysicalRequested(action7, Use(vector_add_pattern))
(7,0) PhysicalGranted(action7)
(7,1) ActorCommitted(add_transition3)
(8,0) TokenPublished(add_result9, one vector<4xf32> token)
(8,0) ActorRetired(add_transition3)
(8,0) PhysicalRetired(action7)
```

The two-cycle stall is derived from request and grant. For release/acquire,
the trace can retain:

```text
MemoryLinearized(
  store_action,
  modification_predecessor = Initial)
MemoryLinearized(
  load_action,
  reads_from = WrittenBy(store_action))
```

The exact contracts, program order, and reads-from relation derive
synchronizes-with and happens-before; the trace does not serialize them.

Diagnostic capture is a nonsemantic invocation binding. Enabling it may
change execution cost and scratch material, but must not change scheduling,
outputs, terminal form, cycle count, metrics, findings, or Artifact identity.
The current attempt context may retain the request and diagnostic value; no
persistent record may claim trace order, coverage, or completeness.

## Ownership And Coupling

`EvaluationRequest.workload_ref` and `runtime_input_ref` contain exact Artifact
references. A `SimulationExecution` refers only to that exact Request and
recovers both inputs and the workload observable contract through it rather
than copying their references. `EvaluationEvidence` refers to the Request and
binds the execution through the descriptor-owned output slot without copying
its observations or terminal witness. A terminal finding occurrence uses the
output-binding-relative `TerminalWitnessRef`. The execution never refers to
Evidence, so the content-identity graph is acyclic.

Architecture-only RTL or EDA checks do not execute a workload and therefore
must not create an empty `SimulationExecution`.

## Anchor Verification

Stable workload/input anchors cover `RootedGraphLaunchRef` ownership, dense
coordinate rank, total value-source classification, exact runtime-table
complements, value and stream cardinality, semantic lane states, mandatory
completion derivation, observable target validity, canonical object ordering,
two imported roots aliasing one object, and rejection of physical layout or
arbitrary object identity. One vecadd case covers fixed `N`, runtime `A/B/C`,
and a visible diff of `C`.

System workload/input anchors cover exact Deployment entry and interface
references, fixed versus runtime value complements, finite external streams,
memory aliasing and baselines, direction/kind/type rejection, and absence of an
argv or string-key escape. One host entry with a value, stream, and memory
interface is sufficient.

Execution anchors separately cover the single untagged root, Request-only
coupling, typed output slot and ordinal resolution, mandatory terminal-finding
totality, the closed terminal algebra and outcome cardinality, witness
ownership and reference validation, runtime-dependent `Unsupported` versus
`Halted` and `ExecutionFailed`, spatial functional-array alignment, terminal-
specific value and stream completeness, unique memory-diff runs, baseline
eligibility, alias-overlap agreement, and rejection of normalized results
inside `SimulationExecution`; canonical integral and fractional progress
coordinates; progress ordering; terminal-specific graph-retirement presence;
terminal observation only after an atomic transition; and metric derivation
that excludes `delta`.

System execution anchors cover positional functional arrays, retired versus
partial publication, exact gem5 tick ordering, mandatory program-exit presence
for `Retired`, and absence of any trace field in
`loom.simulation_execution 1.0`.

Activity anchors cover the two progress-defined windows, rejection of a
missing retirement anchor, complete versus partial target inventories,
missing-is-unknown semantics, one summary per window and payload kind,
canonical collection order, actor commit/retirement monotonicity, Fabric
capacity bounds, four-state signal residency/transition invariants, exact
Request-lineage attachment, and capture noninterference. One actor case, one
Fabric resource case, and one four-state signal case are sufficient; tests do
not build a cross-product over windows, coverage, actors, resource kinds, or
signals.

Diagnostic-trace anchors cover the three capture levels, seven event variants,
typed occurrence references, nonempty canonically ordered frames, strictly
increasing coordinates, duplicate-key rejection, and capture
noninterference. Persistent `loom.simulation_execution 1.0` import rejects any
trace, manifest, chunk, coverage, path, or opaque diagnostic field.

Tests do not pin report layouts, simulator class hierarchies, broad workload
matrices, every finding kind, every witness payload, every partial-output
combination, arbitrary clock ratios, compression formats, or chunk sizes.
