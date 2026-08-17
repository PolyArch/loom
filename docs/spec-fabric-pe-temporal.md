# Fabric PE (temporal schedule)

This document specifies the temporal-schedule branch of `fabric.pe`. The
spatial branch is documented in `spec-fabric-pe.md`. Both branches share the
same op and dispatch on the mandatory typed `schedule` enum.

## Schedule predicate dispatch

`fabric.pe [temporal]` selects the time-multiplexed branch. A single PE
holds one or more inner `fabric.fu` instances and a per-PE instruction
memory of length `num_instruction`. Each active instruction-memory entry makes
one resident configured graph eligible, with operand routing across PE input
ports and a local register-FIFO bank. Actor transitions in that graph execute
independently through their bound `fabric.op` resources, subject to each
resource's declared capacity and grant policy. The entry is a physical
configuration record, not a Mapping identity, a whole-FU firing, or a
one-actor-per-cycle quota.

Each instruction-memory row is the physical dispatch storage for exactly one
resident `InstructionContextRef`, and at most one Compute Realization may
occupy that context in one SpatialMapping. Several rows may be active at once;
their resident configured graphs may select and use several FU occurrences at
runtime. This is the defining distinction from a Spatial PE's single active FU
and sole context. Simultaneous operation firings remain subject to the exact
shared FU, operation, operand-buffer, register-FIFO, and boundary resource
contracts; multiple active rows do not manufacture duplicate hardware.

Both anonymous and named-template forms are accepted:

```mlir
%out = fabric.pe [temporal]
           (%pa = %a : !fabric.bits_tag<32, 4>)
           -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 4 : i32,
         fu_config_mode = #fabric.fu_config_mode<per_fu_config>,
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 2 : i32
       } { ... }

fabric.pe @TempPe [temporal] (!fabric.bits_tag<32, 4>)
                              -> (!fabric.bits_tag<32, 4>)
     attributes { ... } {
^bb0(%pa: !fabric.bits<32>):
  fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) { ... }
  fabric.yield
}
```

## Boundary type rule

Every PE input port and every PE output port has type
`!fabric.bits_tag<W, T>` with the same `W >= 0` and the same `T >= 1`.
The verifier extracts `(W, T)` from PE input #0 and rejects any other
port with a different shape. The `tag_width` hardware attribute must
equal `T`. The input and output counts also obey the shared overflow-safe PE
boundary selector crosspoint contract in
[Fabric PE](spec-fabric-pe.md#boundary-selector-crosspoint-contract): products
through 16 are quiet, products from 17 through 64 warn, and larger products are
invalid.

## Implicit boundary tag handling

PE ports are `!fabric.bits_tag<W, T>` externally, but the body of a
temporal PE never sees the tag bits. The boundary handles the tag
implicitly:

* **Input direction.** Each PE input port is auto-tag-stripped at the
  boundary. The entry block argument visible inside the body has type
  `!fabric.bits<W>` (the bits-data part). The PE-level
  active `OperandSelection` route decides which incoming tag is selected at
  runtime; the body simply sees the raw data bits.
* **Output direction.** Each PE result type is `!fabric.bits_tag<W, T>`.
  The active `ResultSelection` supplies both the selected inner-FU value and
  the runtime tag. A named PE's zero-operand `fabric.yield` is only its
  signature terminator and carries neither value nor tag.

This makes inner `fabric.fu` ops uniformly bits-typed regardless of PE
schedule: FU input and output ports are strict `!fabric.bits<W'>` (with
the same width relaxation rule as in the spatial branch). No bits_tag
ever appears inside the body.

### Anonymous form

The PE outer operand carries `!fabric.bits_tag<W, T>`. The default
inner block-arg type is `!fabric.bits<W>` (the implicit auto-strip).
The user may write an explicit `to <inner-type>` override of the form
`to !fabric.bits<F>` with `F <= W`, which both strips the tag and
truncates to width `F` (low-bit alignment, drop high `W - F` bits).
The implicit-default case is identical to writing `to !fabric.bits<W>`
explicitly; the printer omits the redundant `to` clause.

### Named template form

The PE op signature is `(!fabric.bits_tag<W, T>, ...) -> (...)`. The
entry block of the body is written explicitly:

```
^bb0(%pa: !fabric.bits<W>, ...):
```

The verifier requires every entry block argument type to be
`!fabric.bits<F_i>` with `F_i <= W_i` (the bits-data part of the
corresponding port type). `bits_tag` is forbidden as an entry block arg
type. The named form has no inline `to` syntax; the user writes the
desired narrower width directly in the `^bb0(...)` line.

The named-form terminator is exactly zero-operand `fabric.yield`. Result port
types come only from `function_type`, and the configured `ResultSelection`
records choose their sources. A value-bearing PE terminator is invalid.

## Hardware parameters

All eight parameters are op-level attributes. They are present only on
`fabric.pe [temporal]`; the verifier rejects any of them on a
`fabric.pe [spatial]` (spatial PEs must not carry temporal-only
attributes).

| attribute             | type             | requirement                                                     |
| --------------------- | ---------------- | --------------------------------------------------------------- |
| `tag_width`           | `I32Attr`        | required, `>= 1`, must equal boundary `T`                       |
| `num_instruction`     | `I32Attr`        | required, `>= 1`                                                |
| `num_reg_fifo`        | `I32Attr`        | optional (default 0), `>= 0`                                    |
| `reg_fifo_depth`      | `I32Attr`        | required iff `num_reg_fifo > 0`; absent or `0` otherwise        |
| `reg_fifo_ports`      | `I32Attr`        | optional (default 1); must be `1` or `2`                        |
| `fu_config_mode`      | `FuConfigModeAttr` | required, `#fabric.fu_config_mode<per_instruction_fu_config>` or `#fabric.fu_config_mode<per_fu_config>` |
| `operand_buffer_mode` | `OperandBufferModeAttr` | required, `#fabric.operand_buffer_mode<per_instruction>` / `#fabric.operand_buffer_mode<per_input_port>` / `#fabric.operand_buffer_mode<all_fu_share>` |
| `operand_buffer_size` | `I32Attr`        | required and positive for every mode; entries per mode-derived allocation unit |

These attributes describe hardware capacity and physical organization. They
do not select a workload configuration. Inner `fabric.op.hw_params` likewise
describes parameterized hardware capability; selected semantic values and
raw `sw_configs` are derived only after Mapping.

`pe_enable`, `instruction_mem`, and `per_fu_sw_configs` are not `fabric.pe`
attributes. They name fields of the configured view below, not authoring
syntax. A Fabric parser/verifier rejects them on either PE schedule rather
than accepting or canonicalizing selected workload state into a hardware
template.

The `K = numInputs()` and `L = numOutputs()` shape parameters are
read from the op signature (anonymous form) or the `function_type`
attribute (named form). Both must be `>= 1`.

The implicit shape parameters `num_fu`, `max_fu_inputs`, and
`max_fu_outputs` are derived from the `ConcreteFuOccurrenceSet` defined by the
PE and instantiate specifications. Before finalization this counts anonymous
FU occurrences plus resolved FU instantiations, while excluding named FU
declarations. After canonical elaboration it is exactly the set of anonymous
FU occurrences. The maxima use only that set.

`num_instruction` also bounds the PE-owned resident context namespace.
`docs/spec-fabric-identity.md` owns the persistent reference framing; this
specification owns the valid context range:

```text
InstructionContextRef =
  (FabricPeOccurrenceRef, ContextOrdinal)

0 <= ContextOrdinal < num_instruction
```

`InstructionContextRef` identifies only a configuration/runtime-state
namespace for one resident configured graph. It does not own or copy the
configured graph, its capability, its semantic realization, or its dynamic
actor state transitions. `instruction_mem` is physical dispatch/configuration storage
indexed by the same Fabric-owned ordinal domain. An entry is not the context
reference or state namespace and does not own context identity.
`fu_config_mode` selects a hardware configuration-storage organization, not a
different context model or encoding authority. Inner FUs and `fabric.op`
resources must not create local context IDs.

The context ordinal is also the resident dispatch capacity: one active row
holds one Compute Realization. Distinct Compute Realizations require distinct
context ordinals, while all actors already grouped inside one Compute
Realization share that realization's row. ResourceUse records refine the
runtime occupancy of resources used by a resident graph; they cannot merge two
realizations into one row or create an additional row.

## Software configuration

The configured temporal-PE view is one closed sum:

```text
TemporalPeConfiguration =
    Disabled
  | Active {
      instruction_mem
      per_fu_sw_configs?
      physical_refinements
    }
```

`Disabled` carries no instruction rows, FU configuration, selector, or tag.
The physical enable bit and inactive encoding belong only to
ConfigurationABI. `Active` must contain at least one active instruction row;
an all-unused table canonicalizes to `Disabled`.

Fabric owns their meanings, legal domains, semantic carrier width, field
order, and canonical zero and byte-padding rules. The finalizer derives their
values from the exact TechMapping realization, SpatialMapping's concrete FU
occurrence and `InstructionContextRef`, and any Fabric-declared semantic-
preserving physical refinement. These fields are not an independent capability
or Mapping authority. `ConfigurationABI` alone owns placement of that carrier
in a physical image, physical word and address packing, any transport- or
device-level padding, and the programming protocol.

The configured view is present only after finalization; canonical hardware-only
Fabric contains none of these selected values. When `fu_config_mode` is
`#fabric.fu_config_mode<per_instruction_fu_config>`, each active instruction
entry carries its FU configuration and there is no top-level
`per_fu_sw_configs` collection.

* `instruction_mem`. A fixed-length sequence of typed `InstructionEntry`
  values with length `num_instruction`.
* `per_fu_sw_configs`. Present only when `fu_config_mode` is
  `#fabric.fu_config_mode<per_fu_config>`. It maps each inner FU occurrence to
  one closed configuration record accepted by that FU's typed capability.

Absence of the configured view means no workload has been finalized; it does
not add a default software realization.

One Temporal PE occurrence owns exactly one ordinal-zero
`FabricSemanticConfigFieldRef`. It is a direct carrier for the complete bounded
instruction dispatch table, not one field per instruction, selector, FU, or
tag. The carrier begins with one active bit and then contains exactly
`num_instruction` fixed-capacity instruction rows in context-ordinal order.
Each row begins with one valid bit; an invalid row has zero payload bits.
Within a valid row, selected FU, operand selectors, result selectors, and tags
follow the `InstructionEntry` order below and finite selections use canonical
Fabric inventory order. The all-zero carrier is `Disabled`.

`fu_sw_configs` remains the semantic composition of the selected FU and
operation fields rather than a byte copy inside the PE carrier. Under
`per_instruction_fu_config`, those fields have one configuration slot per
`InstructionContextRef`. Under `per_fu_config`, they have one Static slot
shared by every active instruction row. The residency rule is owned by
`docs/spec-configuration-deployment.md`; neither form invents another context
or duplicates a behavior codec.

The Fabric-owned relation derives the exact width from the immutable PE shape
and validates the complete dispatch record, including selected-FU ownership,
selector ranges, relevant-field presence, and tag bounds. The complete
configuration validator also checks the mode-specific FU and operation slot
closure. ConfigurationABI may scatter the PE carrier but cannot split it into
independently writable row or selector authorities.

## Per-instruction format

Each `instruction_mem` entry is one member of a closed typed sum:

```text
InstructionEntry =
    Unused
  | Active {
      selected_fu,
      operand_sel[selected FU input count],
      result_sel[selected FU output count],
      fu_sw_configs?  // present only in per-instruction configuration mode
    }
```

`operand_sel` and `result_sel` use closed typed variants so an irrelevant
selector value cannot create another configuration:

```text
OperandSelection = Route(InputPortRef | RegFifoRef, tag)
                 | Discard(InputPortRef | RegFifoRef, tag)
                 | Disconnected

ResultSelection = Route(OutputPortRef | RegFifoRef, tag)
                | Discard
                | Disconnected
```

`fu_sw_configs` is interpreted by the selected FU's typed configuration
schema, not as arbitrary key/value data. Its
semantic and topology fields are derived from exact Dataflow actors and
TechMapping correspondence; semantic-preserving physical fields come from
SpatialMapping. A sync active set is derived from ordered correspondence and
the capability relation. Runtime selectors of software mux/demux actors remain
data operands and are not configuration keys.

Validation rules for each active entry are:

* `selected_fu` must identify one FU occurrence owned by this PE.
* Every `InputPortRef` is in `[0, K)` and every `OutputPortRef` is in
  `[0, L)`.
* A `RegFifoRef` requires `num_reg_fifo > 0` and is in
  `[0, num_reg_fifo)`.
* Every carried `tag` is in the PE's declared tag domain.
* Every FU input and output has exactly one selector variant.

The normalized configured view omits fields that are irrelevant under the
selected record. Fabric owns the fixed-capacity semantic table layout, field
widths, canonical zero filling, and unused high bits of its final semantic
byte. `ConfigurationABI` owns only the later physical instruction words,
addresses, placement, and physical payload padding.

## Reg FIFO semantics

When `num_reg_fifo > 0`, the PE owns a bank of `num_reg_fifo` register
FIFOs, each of depth `reg_fifo_depth` and with `reg_fifo_ports` ports
(`1` for single-ported, `2` for separate read/write). Each register
entry stores a `(data, tag)` pair, identical in shape to a single
`!fabric.bits_tag<W, T>` token. Writing to a register pushes one such
pair (the selected result-route tag is the tag value); reading pops the head.

A selector whose typed endpoint is `RegFifoRef` selects a register FIFO
instead of a PE port.

A register FIFO realizes a software edge internally only when SpatialMapping
selects an explicit temporal-PE register-file relation from the producer's
`result_sel`, through the identified FIFO and its declared ports, to the
consumer's `operand_sel`. The relation must preserve the edge's type, tag,
order, capacity, and backpressure obligations. Sharing a PE or having a free
register FIFO does not absorb an edge.

The initial local-transfer domain is deliberately narrow. It admits exactly a
single-consumer residual edge whose producer and consumer are bound to FU
occurrences under the same Temporal PE, whose payload and tag widths match the
FIFO exactly, and whose ordering contract is ordinary FIFO order. A multicast
net, a net with both local and external consumers, or a width, tag, ordering,
or lifetime mismatch remains an external residual net. For every admitted
edge, SpatialMapping selects one closed disposition:

```text
TemporalPeEdgeDisposition =
    RegisterFifo {
      write_traversal,
      register_fifo,
      read_traversal,
      resource_uses
    }
  | ExternalRoute
```

The local disposition names both exact FU-side traversals and the one FIFO;
configuration projection derives the producer `ResultSelection` and consumer
`OperandSelection` from those selected facts. A legal local disposition is the
preferred search choice. Exhausting compatible FIFO capacity removes only that
choice and leaves `ExternalRoute` available; preference is not feasibility.

Let `O` be FIFO occupancy at the start of a PE clock cycle, `R` a successful
read, and `W` a successful write. A read is eligible only when `O > 0` and its
selected operand tag equals the cycle-start head tag. A write stores the
producer result together with the configured result tag. The next occupancy is
`O - R + W`, and a token written in the cycle is never eligible for a read in
that cycle.

With two ports, read and write service are independent. A full FIFO may accept
a write exactly when a read also commits, so `write_ready = !full || R`; this
is full-queue replacement, not bypass. Read and write requesters use separate
round-robin cursors in canonical requester order, and each cursor advances only
on its own successful grant. With one port, at most one operation commits.
An eligible read has priority over writes; otherwise one eligible writer is
selected by the write cursor. A full single-port FIFO therefore cannot replace
its head in one cycle. An empty FIFO cannot satisfy a read from a same-cycle
write under either port organization.

## Operand buffer modes

All three modes expose the same logical FIFO semantics. A logical operand
queue is determined by the selected `InstructionContextRef` and FU ingress
selected by ingress routing and tag dispatch. The mode changes only the
physical storage organization:

* `per_instruction`: each logical operand queue has dedicated storage.
  Each queue is one allocation unit with `operand_buffer_size` entries.
* `per_input_port`: logical queues associated with one FU ingress bank share
  one allocation unit with `operand_buffer_size` entries.
* `all_fu_share`: all logical operand queues in the PE share one entry pool
  that is one allocation unit with `operand_buffer_size` entries.

`operand_buffer_size` is a required Fabric hardware parameter in all three
modes. It has no default. In particular, `per_instruction` depth is not an
implementation constant: depths 1 and 2 have different backpressure behavior,
canonical Fabric bytes, and Artifact identity.

The Fabric-owned potential logical-queue domain is the complete canonical set:

```text
LogicalOperandQueueKey =
  (InstructionContextRef,
   ConcreteFuOccurrenceOrdinal,
   FuInputOrdinal)
```

An Active configured view makes only its selected queues eligible; every other
key remains empty and ineligible. The allocation unit is the following total
mechanical projection and is not a Mapping record or backend choice:

```text
per_instruction -> DedicatedQueue(LogicalOperandQueueKey)
per_input_port   -> FuInput(ConcreteFuOccurrenceOrdinal, FuInputOrdinal)
all_fu_share     -> WholeTemporalPe
```

These owner-local keys are not standalone entities or persistent references.
Their canonical ordering is the lexicographic order of the typed components,
using `InstructionContextRef` order first, then concrete FU occurrence order,
then FU input ordinal.

One outer FU input owns one active logical queue key. When that input has
several FU-local SSA uses, the queue head is the physical source of the
Fabric-defined broadcast: all consumers observe the same token and participate
in one common dequeue handshake. The queue entry retires once, only when every
consumer accepts it. Mapping preserves every logical consumer obligation but
must not allocate, fill, or retire a second queue entry for another use of the
same FU boundary input.

Every shared pool maintains independent FIFO head and tail state for each
logical queue. Allocation may use a shared free-entry pool, but dequeue
eligibility is determined from the selected logical queue's head. A shared
mode must not merge different contexts, tags, or logical streams into one
global arrival-order FIFO head; that organization can introduce
implementation-induced head-of-line deadlock.

For example, global arrival-order queues `P:[tag0,tag1]` and
`Q:[tag1,tag0]` can block both contexts even after both complete operand tuples
have arrived. Independent logical heads must allow each configured actor to
consume its own matching tuple. Such a global-head stall is an invalid
implementation-induced deadlock, not an ordering requirement of Canonical
Dataflow.

Operand queues preserve order independently for each logical stream. They do
not impose arrival order across unrelated streams, and they do not replace
FU-internal pipeline, holding, or edge storage. An actor transition may commit
only after all of its required logical heads and finite output-delivery
capacity are available. Shared-pool exhaustion and route backpressure remain
real dependencies in the final progress and deadlock closure.

One incoming boundary token may match several distinct active logical operand
queues. Each queue may in turn feed several FU-local broadcast consumers. For
match vector `match[i]`, the PE performs one atomic multi-queue enqueue rather
than a sequence of independent enqueues:

```text
any_match = OR(match[i])
ready     = any_match AND AND(!match[i] OR queue_ready[i])
fire      = input_valid AND ready
enqueue[i] = fire AND match[i]
```

`queue_ready[i]` is derived as one grant-aware bundle. For every shared
allocation unit, a configured match group may contain at most one logical
queue because version 1 declares one enqueue service slot. Mapping and PnR
reject a group that selects two queues in the same unit; this is a legality
failure, not an arbitration opportunity. Matches in distinct allocation units
may fan out atomically when every selected unit has cycle-start capacity and
grants its one required enqueue service. No matching queue mutates unless the
common `fire` occurs; a token cannot be delivered to a ready subset, dropped
for a blocked subset, or enqueued again on a later retry. Each
`queue_ready[i]` observes only capacity that was free at the start of the PE
clock cycle. A dequeue in that cycle does not create a combinational
replacement path from FU readiness to ingress readiness. No match
backpressures the input and is an invalid configured routing situation, not an
implicit discard.

The temporal-PE schema uniquely owns the typed `ResourceState` values for
resident contexts, logical operand queues, register FIFOs, and shared dispatch
capacity; their canonical initial states; capacity dimensions;
owner-defined atomic resource transitions; atomic UsePatterns; stable typed
requester order; and exact GrantPolicy or exact refinement domains. One actor
transition may atomically commit all required queue-head removals while
claiming one operation pipeline, result holding capacity, and register-FIFO
ports. Mapping binds typed workload values and selected exact refinements but
cannot split that use or define another scheduler. Queue contents, occupancy,
head/tail positions, grant cursors, and in-flight transitions are nonpersistent
execution state.

The CGRA execution-plan importer consumes the same projected logical queue,
its complete consumer set, allocation-unit ordinal, and entry capacity. Its
runtime reserves every distinct unit in one ingress match group before
requesting the selected enqueue actions, commits all matching queue tails
together, and retires one physical queue head only on the common consumer
handshake. Software channel storage is only the dense per-consumer token view
of those physical queues; it cannot multiply physical occupancy, impose a
one-token default, or bypass shared-unit occupancy and Fabric grant policy.

For operand buffering, the finalizer derives one exact resource contract from
the two hardware parameters:

```text
ResourceState:
  OperandEntryPool(allocation_unit,
                   capacity = operand_buffer_size,
                   initial_occupancy = 0)
  OperandQueue(logical_queue, initial = empty)
  OperandEnqueueService(allocation_unit, capacity_per_pe_clock_cycle = 1)
  OperandDequeueService(allocation_unit, capacity_per_pe_clock_cycle = 1)

ResourceTransition:
  AppendOperand(logical_queue)
    append the accepted token at the queue tail and increment its allocation
    unit occupancy by one
  RemoveOperand(logical_queue)
    remove the cycle-start head token and decrement its allocation unit
    occupancy by one

UsePattern:
  Enqueue(logical_queue)
    acquire_event = commit_event = EnqueueCommit
    atomically claims one enqueue service slot
    release_event = NextPeClockBoundary
    commit_transition = AppendOperand(logical_queue)
  Dequeue(logical_queue)
    acquire_event = commit_event = DequeueCommit
    atomically claims one dequeue service slot
    release_event = NextPeClockBoundary
    commit_transition = RemoveOperand(logical_queue)
```

The queue tail, queue head, free entry, and occupied entry are not temporary
claims. They are parts of the Fabric-owned operand-buffer dynamic state. An
enqueue does not create a claim that a later dequeue releases, and a dequeue
does not inherit or transfer ownership of another pattern's claim. The short
service reservation and durable queue mutation occur at the same atomic commit,
so the model has one acquisition and no split transaction.

One enqueue and one dequeue may commit on the same allocation unit in one
PE clock cycle. Let `O` be occupancy at the cycle start, `D` the selected
dequeue count, and `E` the selected enqueue count for one allocation unit.
The current operand-buffer contract has `D` and `E` in `{0, 1}`. Dequeue
eligibility observes only a token present at cycle start. Enqueue eligibility
requires `O < capacity`, independently of `D`, and the next occupancy is
exactly `O - D + E`. The two selected resource transitions commit atomically
at the PE clock boundary. A non-full unit may therefore dequeue and enqueue in
one cycle, while a full unit cannot use that cycle's dequeue as ingress
capacity. The newly enqueued token cannot satisfy that cycle's dequeue.

`NextPeClockBoundary` is mechanically the next rising edge of the exact Clock
domain containing the PE. It is not an `OperandBufferLocalCycle` state object,
backend callback, or simulator-owned timing rule. Enqueue and dequeue service
slots are free after reset and remain claimed until that boundary after a
successful use.

All `Dequeue(logical_queue)` uses required by one Canonical Dataflow actor
transition are activated by that actor's existing single commit event. This is
a derived atomic activation relation, not another composite-use record. The
Mapping verifier must prove that their combined service claims fit every
allocation unit. If two required heads project to one version-1 allocation
unit in the same commit, its one-dequeue service is insufficient and that
binding is invalid. A multiported buffer, explicit operand-staging resource,
or other wider organization must be declared as a typed Fabric capability; an
implementation cannot serialize the removals privately after partially
consuming the actor inputs.

When more than one logical queue can request the same enqueue or dequeue
service slot, that service uses the shared `RoundRobin` GrantPolicy over the
canonical logical-queue order above, filtered to queues that project to that
allocation unit. Enqueue and dequeue have independent cursors, both reset to
the first canonical requester and advance only on a successful grant. A policy
is absent only when structural analysis proves that at most one requester can
be eligible. An implementation-private priority, arrival-order race, default
depth, extra port, banking scheme, reservation, or virtual channel is
forbidden. A future multiported or reserved organization requires an explicit
typed Fabric parameter or refinement.

### Context-evaluation service

Temporal instruction rows are evaluated through an explicit Fabric-owned
service. Its closed candidate domain is the context-major sequence of
`(InstructionContextRef, FU occurrence)` pairs. `all_fu_share` has one service
allocation unit for the PE; the other operand-buffer organizations have one
independently advancing unit per FU. Each candidate has one UsePattern that
claims its unit from `ContextEvaluationGrant` until the next PE clock boundary
and has no resource transition or actor commit.

Every unit filters the canonical candidate sequence to active instruction rows
that select that FU and applies the ResourceContract's `RoundRobin` policy.
The reset candidate and advance order are therefore Fabric facts. A selected
candidate advances the cursor at the clock boundary even when no actor in that
resident realization is ready. Such an idle candidate can consume only its
fair evaluation slot; it cannot retain a grant or prevent the next active
candidate from being evaluated. Inactive rows do not enter the filtered cycle.

The evaluation grant only permits the selected context to drive its FU during
that PE cycle. Canonical Dataflow actors remain independent transition units,
and their exact operation ResourceContracts still decide readiness, capacity,
arbitration, commit, and retirement. Several ready actors in one selected
realization may therefore request their true physical operations, while an
evaluation grant by itself mutates no Dataflow or queue state. Mapping derives
the active filtered service domains from compute bindings, and the cycle
simulator and RTL consume the same candidate order, allocation units, reset,
and round-robin policy.

## Handshake Dependency Projection

Every selected ingress boundary-to-FU traversal terminates at a logical
operand queue. Dequeue eligibility observes only the cycle-start queue head,
enqueue eligibility observes only cycle-start free capacity, and both
transitions commit at the PE clock boundary. The traversal therefore
contributes neither a boundary-input-valid to FU-input-valid dependency nor a
backward FU-input-ready to boundary-input-ready dependency. The queue is a
typed durable progress boundary for its exact sink
attachment because an accepted token is durably retained independently of a
later FU firing. Mapping progress analysis consumes that Fabric projection
directly; it must not require an additional external
`fabric.fifo[buffered]` after the PE ingress.

A selected FU-to-boundary result traversal is transparent. Its result-valid
and ready dependencies pass through the PE selector, while the selected
operation's Fabric-owned result-holding state supplies the registered boundary.
The PE does not own a second implicit result queue. An explicit register-FIFO
route remains a stateful break under the register-FIFO contract above.

## Mapping Ownership

TechMapping selects the FU structural/capability template and exact ordered
actor/op/port/FU-boundary correspondence. SpatialMapping selects a concrete FU
occurrence and `InstructionContextRef` with the same parent temporal PE, plus
PE ingress/egress routing and semantic-preserving physical refinements.

The Spatial physical-demand functions defined by `spec-pnr.md` mechanically
derive active logical operand queues, their atomic ingress match groups,
eligible register-FIFO local dispositions, and residual external nets from
those exact owners. PnR selects only alternatives admitted by those functions.
Strict Mapping verification and configured-hardware projection rebuild the
same demands, and simulator plan import preserves their exact result. Temporal
PE RTL implements this Fabric-owned queue and RegFIFO mechanism from configured
fields; it does not call Mapping. No layer may infer a local transfer or queue
match from PE co-location alone.

The context reference is only the resident configuration/runtime-state
namespace. `instruction_mem[i]` is a Fabric-defined configuration and dispatch
entry at ordinal `i`, not the context itself. Mapping does not persist a second
instruction-entry reference or use the entry contents as identity.
`ResourceUse` may refer to the selected context for event-relative occupancy,
but cannot select another context or copy the configuration. The finalizer
alone derives `instruction_mem` and `per_fu_sw_configs`; `ConfigurationABI`
alone encodes them as raw bits.

Neither the context nor the configured FU is a dynamic execution atom. The
Canonical Dataflow actors bound to active inner operations retain their own
readiness, state transition, commit, publication, and retirement semantics.
No whole-FU or whole-context macro firing is introduced.

A context may be handed off or reconfigured only after its actor transitions
have retired, its logical operand queues and operation result-holding states are
empty, and its stateful actors have satisfied their declared self-reset
contracts. The active result route supplies the configured Physical Tag; a
dynamic firing does not invent a separate tag identity.

## Dispatch Condition

An active instruction-memory entry at ordinal `i` makes each Canonical
Dataflow actor bound into its resident configured graph eligible. One such
actor transition may commit when:

1. The PE configuration is `Active` and entry `i` is the `Active` variant.
2. Every actor input routed from an `InputPortRef` has a head token in its
   selected logical operand queue. PE-ingress tag dispatch places tokens into
   that queue.
3. Every actor input routed from a `RegFifoRef` has a head token in the
   selected register FIFO under the semantics above.
4. The selected physical operation within the configured FU is ready to
   consume, and finite capacity is reserved for all output obligations.

The active entry's `selected_fu` selects the configured physical FU; it does
not turn every actor inside that FU into one rendezvous or impose a
one-transition-per-cycle limit on the configured graph. Other actors in the
same configured graph dispatch independently when their own transitions
become enabled. When actor transitions contend for one physical operation or
pipeline, that resource's Fabric-owned capacity and exact grant policy decide
which transitions commit.

`Discard` is an explicit temporal-PE boundary drain; `Disconnected` is inert
for that instruction-memory entry. Neither variant can repair an invalid
FU-internal broadcast, replace required FU demux/mux topology, or discharge a
logical edge required by the exact Mapping realization.

## Body whitelist

Identical to spatial: only `fabric.fu` and `fabric.instantiate`. The
named-template form additionally requires a closing zero-operand
`fabric.yield`. No
other op kind is permitted.

## Validation Anchors

Anchor tests cover `per_instruction` depths 1 and 2 producing distinct Fabric
identity and backpressure, rejection of an absent or nonpositive
`operand_buffer_size` in every mode, canonical allocation-unit derivation for
all three modes, match-group admission against one enqueue service per unit,
one enqueue's atomic short service claim plus durable append transition,
simultaneous dequeue/enqueue from non-full occupancy without same-cycle
replacement, and
deterministic round-robin contention between two logical queues. Boundary tests
also cover the shared quiet, warning, hard-limit, and overflow cases for both
anonymous and named temporal PEs. Tests do not construct a queue-count, depth,
context, FU-input, or arbitration cross product.

## Cross-reference

* Spatial branch and its configured-view semantics: `spec-fabric-pe.md`.
* Parameterized FU capability and derived configuration:
  `spec-fabric-reconfigurable-op.md`.
* Boundary ops bridging spatial and temporal domains:
  `spec-fabric-boundary.md`.
* Shared state, atomic-use, requester-order, and grant-policy atoms:
  `spec-fabric-resource-contract.md`.
