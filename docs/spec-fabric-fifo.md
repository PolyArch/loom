# Fabric FIFO

`fabric.fifo` is an explicit finite buffering resource in a
`fabric.module` transport graph. It is not the physical meaning of every
logical channel, an abstract buffer that Mapping may insert, or a second route
record. Logical FIFO behavior may be realized by a larger path containing
switches, links, virtual channels, memory-backed queues, or other explicit
Fabric resources.

## Interface And Hardware Capability

The resource has one transport input and one transport output. Both endpoints
use the same Fabric port kind: either `bits` or `bits_tag`. Same-kind width
differences use the module-owned low-bit-aligned truncation and zero-extension
rule. A FIFO does not convert between spatial and tagged domains and does not
accept `memref` endpoints.

Fabric owns three hardware capability fields:

```text
FifoCapability {
  max_depth: positive integer
  bypassable: bool
  queue_discipline: StrictFifo | PerTagVirtualChannel   // default StrictFifo
}
```

`max_depth` is the fixed total physical token-holding capacity. It is not an
effective depth that Mapping may resize, and there is no hidden skid or credit
slot outside it. A future partitionable bank, configurable depth, credit
allocation, or skid capacity must declare that separate typed capability
explicitly rather than overloading this field.

`queue_discipline` is the dequeue scheduling discipline of the physical queue.
It is an immutable hardware fact like `max_depth`, never a Mapping selection,
and it never appears in the configuration field domain. The two disciplines
are defined below. A `per_tag_virtual_channel` occurrence requires a tagged
port kind with a positive tag width, because the Physical Tag is the channel
identity it schedules on, and it owns no bypass alternative: a combinational
passthrough would route around the very queue the discipline orders, so
`bypassable = true` is rejected for it. A `strict_fifo` occurrence may use
either port kind and may be bypassable.

## Dequeue Scheduling Discipline

Both disciplines share the buffered cycle rules: one shared pool of
`max_depth` slots, at most one enqueue and at most one dequeue per local
cycle, an enqueued token visible no earlier than the following cycle, input
ready and output valid from registered cycle-start state, and no
current-cycle capacity release from a dequeue to an enqueue. They differ only
in which resident entry the output port may present.

`StrictFifo` keeps one global arrival order. The output port presents exactly
the oldest resident entry, so one token whose consumer cannot take it blocks
every later token.

`PerTagVirtualChannel` partitions the shared pool into one virtual channel
per observed Physical Tag value. Channel identity is the exact tag bit value
carried on the port, not any plan- or mapping-local ordinal; two tokens
carrying equal tag values belong to one channel no matter which route
delivered them. Within one channel, arrival order is preserved exactly and a
token never overtakes an earlier token of its own channel. The port has one
valid/ready pair, so it presents the head entry of exactly one non-empty
channel per cycle:

* The canonical channel order is the unsigned ascending order of the
  resident tag values.
* A cursor names the channel the round robin resumes at. Reset and the
  canonical empty state set the cursor to the zero tag value.
* Each cycle the port presents the head of the first non-empty channel at or
  after the cursor in canonical order, wrapping once to the lowest resident
  value. Output valid means the presented head exists; downstream ready
  applies to that head only.
* A completed dequeue removes the presented entry and moves the cursor to
  the strict successor of its tag value.
* When the presented head is not taken (`valid && !ready`), the queue commits
  one `OfferAdvance` transition at the cycle boundary and the next cycle
  presents the next non-empty channel after the refused one. A refused offer
  never removes or reorders an entry.
* With N non-empty channels, every channel is presented at least once in any
  N consecutive cycles that hold all N non-empty; this fairness bound follows
  from the cursor rule and is the whole anti-starvation contract.
* A channel that drains empty leaves the rotation; a channel that later
  receives a token re-enters at its canonical value position, including
  re-entry at a value the cursor has passed.

The queue contents, per-channel heads, cursor, and occupancy are dynamic
execution state. They are not Mapping records, Fabric capability fields, or
persistent configuration. Drain returns the queue to the canonical empty
state: zero occupancy and the cursor at the zero tag value.

The FIFO `ResourceContract` is the unique owner of these transitions. The
`BufferedQueue` state owns one shared `QueueSlot` capacity dimension of
`max_depth` plus the per-cycle `EnqueueService` and `DequeueService`
dimensions. Under `StrictFifo` the contract is exactly the pre-discipline
contract: `Enqueue`, `Dequeue`, and `SimultaneousDequeueEnqueue` patterns,
extended by `BypassTransfer` exactly when `bypassable`. Under
`PerTagVirtualChannel` the closed pattern inventory is `Enqueue`, `Dequeue`,
`SimultaneousDequeueEnqueue`, and `OfferAdvance`; the dequeue and offer
patterns carry one Physical Tag value parameter of the port's tag width that
names the channel they present, `OfferAdvance` claims no capacity and commits
the cursor transition at the cycle boundary, and a grant moves the cursor as
part of the dequeue commit. A full queue at cycle start admits no enqueue
under either discipline, including from current-cycle downstream readiness.


## Mapping-Selected Traversal

A non-bypassable occurrence exposes one buffered internal traversal. A
bypassable occurrence exposes exactly two Fabric-valid traversal alternatives:

```text
Buffered
Bypass
```

The configured projection is one closed sum:

```text
FifoConfiguration =
    Disabled
  | Active { mode: Buffered | Bypass, physical_refinements }
```

`Disabled` carries no mode or refinement. ConfigurationABI alone owns the
physical inactive encoding. The selected RouteTree owns whether an Active
occurrence uses Buffered or Bypass. Mapping does not store
a duplicate raw `bypassed` field. Configured Fabric, simulation input, and the
HardwareConfigurationImage derive the semantic mode mechanically from the
complete Mapping; `ConfigurationABI` alone owns its physical encoding. If
multiple temporal uses share one occurrence, Mapping must prove compatible
mode residency, reconfiguration, and resource-time use.

Each FIFO occurrence owns exactly one ordinal-zero
`FabricSemanticConfigFieldRef`. Its finite domain is `Disabled`, `Buffered`,
and, exactly when `bypassable = true`, `Bypass`. Canonical semantic bytes are
the single `u32be` tags 0, 1, and 2 in that order. The ABI finite codebook must
cover the exact occurrence domain and use `Disabled` as the inactive value.
Depth, bypass capability, and queue discipline are immutable hardware facts
and never appear in this field.

## Buffered Execution

Buffered mode owns a FIFO queue with capacity `max_depth`. In one local clock
cycle it accepts at most one enqueue and completes at most one dequeue.

An enqueued token is visible to dequeue no earlier than the following local
cycle. Input ready is determined only by registered free capacity at the start
of the cycle, and output valid is determined only by registered occupancy at
the start of the cycle. A dequeue from a full queue releases capacity for the
following cycle, not for an enqueue in the same cycle. When the queue is not
full at cycle start, one enqueue and one dequeue may still complete together;
the newly enqueued token cannot also dequeue in that cycle. Token order is
preserved exactly as the declared queue discipline defines it: the global
arrival order under `strict_fifo`, or the per-channel order with the
single-head offer and cursor rotation under `per_tag_virtual_channel`.

The queue contents, head, tail, and occupancy are dynamic execution state.
They are not Mapping records, Fabric capability fields, or persistent
configuration.

The FIFO schema uniquely owns its queue and bypass `ResourceState` values,
canonical empty initial state, capacity dimensions, and atomic enqueue,
dequeue, non-full simultaneous enqueue/dequeue, and bypass-transfer
UsePatterns. A pattern may claim input, output, and queue-capacity state
atomically. No buffered pattern may use current-cycle downstream readiness to
admit an enqueue into a queue that was full at cycle start. The single input
and output requester domains make grant order structural; a generic arbiter or
Mapping-defined grant rule is invalid.

## Bypass Execution

Bypass mode has no active queue state. It performs a zero-registered-cycle
combinational transfer only when the downstream obligation is ready; otherwise
backpressure propagates to the input. The `max_depth` storage cannot accept or
retain a token while bypass mode is active.

Zero registered cycles does not mean zero propagation delay. Bypass can reduce
cycle count while lengthening a combinational path. Buffered mode adds at least
one registered cycle and cuts both directions of the ready/valid combinational
path. Mapping selects only the declared alternative. Accelerator-aware timing,
frequency, initiation interval, and runtime consequences come from central
Evaluation rather than a FIFO-private cost model.

## Handshake Dependency Projection

The FIFO owner derives signal-level combinational arcs from the selected mode;
Mapping does not persist a second cycle-breaking flag.

`Bypass` contributes the transparent forward valid dependency from input to
output and the transparent backward ready dependency from output to input.
`Buffered` never contributes an input-valid to output-valid dependency because
an enqueued token is not visible until a later cycle, and never contributes an
output-ready to input-ready dependency because current-cycle dequeue cannot
create current-cycle input capacity. Buffered mode is therefore a complete
ready/valid combinational isolation point. A higher-throughput full-queue
credit or skid behavior exists only when Fabric declares that exact typed
capability or refinement and includes all additional physical capacity in its
identity.

Fabric structural verification considers only FIFO arcs unconditional across
all legal configured views. A physical topology that can form a cycle under
some switch rows or FIFO modes is legal Fabric; mutually exclusive alternatives
are not unioned into a fabricated unconditional graph. Mapping verification
uses the exact selected switch rows, FIFO modes, and refinements. A bypass
selection that closes a directed combinational handshake cycle is intrinsically
invalid Mapping. A selected buffered traversal contributes no cross-FIFO arc
to that graph.

## Lifecycle

When an activation is admitted to a FIFO state slot, buffered state is empty
and bypass state has no held token. Normal release requires all obligations of
that activation on the slot to retire and the resource to return to that
canonical state. The slot then self-restores for legal handoff; graph launch
does not carry a second reset token. Distinct invocations may overlap on other
resources when their Mapping-owned uses do not conflict. Abnormal termination
cannot fabricate completion or silently discard a token as though retirement
had occurred.

## Simulator And RTL Obligations

CGRA-sim derives the selected mode and the declared queue discipline from
exact Fabric and Mapping, then executes the cycle rules above. It must not
invent hidden queue capacity in bypass mode or buffered mode, admit an
enqueue into a cycle-start-full queue from current-cycle downstream
readiness, or provide same-cycle fall-through. Under `per_tag_virtual_channel`
it must present exactly one channel head per cycle, rotate the cursor only
through the declared grant and offer-advance transitions, schedule the next
cycle's offer after a refused offer as an explicit state transition, and never
schedule on a plan-local tag ordinal in place of the exact tag value. A
refused offer that rotates the cursor is arbitration progress, not token
progress; if every resident channel completes a full rotation without a grant
and no other event can change readiness, the execution is a closed wait, not
an infinite cursor rotation. The simulator therefore counts refused offers per
queue since the last enqueue or dequeue commit; when that count reaches the
number of resident channels, the last `OfferAdvance` retires without
re-presenting the queue and the queue sleeps until an external event (a queue
commit, released downstream capacity, or a consumer publication) restarts the
epoch. A quiescent execution reports every such sleeping queue as a typed
exhausted-rotation witness (storage, FIFO occurrence, resident channel count,
refused offers, occupancy, capacity, and the resident tag values in canonical
order) inside its closed-wait diagnostic. That witness, not a frame or wall
budget, is the terminal outcome of a virtual-channel queue with no ready
complement, and it is what makes every refused class head quoted by the
closed-wait certificate an offer the port actually made.

The virtual-channel discipline partitions dequeue order, never capacity. The
static reconvergent-capacity obligation of a `per_tag_virtual_channel`
occurrence names every resident tag class but compares one selected pool
against one proven minimum, so a pool smaller than the number of nets that can
each hold one resident token is a proven `ReconvergentCapacityShortfall`
closed wait even when the per-tag classes remove every order cycle.

Fabric-to-RTL implements the same capability and selected-mode behavior. It
compares the actual tag bits of resident entries, selects the arrival-oldest
entry of the presented channel, presents one data/tag/valid triple per cycle,
and rotates its cursor in the same cycle situations as the simulator. It may
choose any circuit structure consistent with the declared capacity, handshake,
visibility, lifecycle, and ConfigurationABI. Backend pipeline or storage
details do not become a second architectural contract.

## Verification Anchors

Anchor-level tests cover one buffered occurrence at empty, full, enqueue,
dequeue, non-full simultaneous dequeue/enqueue, and full-dequeue capacity
release boundaries; one bypassable occurrence in both legal modes with
propagated backpressure; rejection of bypass on a non-bypassable occurrence;
rejection of `per_tag_virtual_channel` on an untagged or bypassable
occurrence; derivation of configured mode from the selected RouteTree;
acceptance of a physical topology that only potentially forms a cycle; and
rejection only when selected switch rows and bypass traversals close that
cycle. Discipline anchors additionally cover global order under
`strict_fifo`, per-channel order and blocked-channel bypass under
`per_tag_virtual_channel`, cursor wraparound and channel re-entry, the shared
pool capacity boundary, the refused-offer next-cycle presentation, the
no-complement terminal witness (every resident channel refused once, no
scheduled event, a typed exhausted-rotation witness instead of a budget), the
static shared-pool capacity control, and simulator/RTL agreement on the
presented tag, validity, input ready, grants, occupancy, and cursor at every
cycle. The simulator/RTL agreement anchor walks every reachable queue state of
one small occurrence (a few slots and tag values) under every single-cycle
stimulus, with the simulator storage queue as the oracle; the walk is a
derived stimulus, not a preserved trace. Tests do not preserve internal
pointer encoding, queue implementation, raw configuration bits, or hand-written
occupancy traces.

An explicit `fabric.fifo` is a different hardware owner from a Temporal PE
operand-buffer pool. FIFO depth, bypass, and traversal feedback cannot be
reinterpreted as operand-buffer mode, depth, or admission policy. Both obey
cycle-start capacity and ordered state transitions, but their Mapping
invalidation cones and Hardware-DSE decisions remain distinct.
