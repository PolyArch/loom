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

Fabric owns two hardware capability fields:

```text
FifoCapability {
  max_depth: positive integer
  bypassable: bool
}
```

`max_depth` is the fixed total physical token-holding capacity. It is not an
effective depth that Mapping may resize, and there is no hidden skid or credit
slot outside it. A future partitionable bank, configurable depth, credit
allocation, skid capacity, or virtual-channel resource must declare that
separate typed capability explicitly rather than overloading this field.

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
preserved exactly.

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

CGRA-sim derives the selected mode from exact Fabric and Mapping, then executes
the cycle rules above. It must not invent hidden queue capacity in bypass mode
or buffered mode, admit an enqueue into a cycle-start-full queue from
current-cycle downstream readiness, or provide same-cycle fall-through.

Fabric-to-RTL implements the same capability and selected-mode behavior. It
may choose any circuit structure consistent with the declared capacity,
handshake, visibility, lifecycle, and ConfigurationABI. Backend pipeline or
storage details do not become a second architectural contract.

## Verification Anchors

Anchor-level tests cover one buffered occurrence at empty, full, enqueue,
dequeue, non-full simultaneous dequeue/enqueue, and full-dequeue capacity
release boundaries; one bypassable occurrence in both legal modes with
propagated backpressure; rejection of bypass on a non-bypassable occurrence;
derivation of configured mode from the selected RouteTree; acceptance of a
physical topology that only potentially forms a cycle; and rejection only
when selected switch rows and bypass traversals close that cycle. Tests do not
preserve internal pointer encoding, queue implementation, raw configuration
bits, or exhaustive occupancy traces.
