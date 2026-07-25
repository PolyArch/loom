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

`max_depth` is the fixed physical queue capacity. It is not an effective depth
that Mapping may resize. A future partitionable bank, configurable depth,
credit allocation, or virtual-channel resource must declare that separate
typed capability explicitly rather than overloading this field.

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
cycle. A successful dequeue may release capacity for an enqueue in the same
cycle, but the newly enqueued token cannot also dequeue in that cycle. Input
ready and output valid follow from the queue state plus this simultaneous
enqueue/dequeue rule. Token order is preserved exactly.

The queue contents, head, tail, and occupancy are dynamic execution state.
They are not Mapping records, Fabric capability fields, or persistent
configuration.

The FIFO schema uniquely owns its queue and bypass `ResourceState` values,
canonical empty initial state, capacity dimensions, and atomic enqueue,
dequeue, simultaneous enqueue/dequeue, and bypass-transfer UsePatterns. A
pattern may claim input, output, and queue-capacity state atomically. The
single input and output requester domains make grant order structural; a
generic arbiter or Mapping-defined grant rule is invalid.

## Bypass Execution

Bypass mode has no active queue state. It performs a zero-registered-cycle
combinational transfer only when the downstream obligation is ready; otherwise
backpressure propagates to the input. The `max_depth` storage cannot accept or
retain a token while bypass mode is active.

Zero registered cycles does not mean zero propagation delay. Bypass can reduce
cycle count while lengthening a combinational path. Buffered mode adds at least
one registered cycle and can cut that path. Mapping selects only the declared
alternative. Accelerator-aware timing, frequency, initiation interval, and
runtime consequences come from central Evaluation rather than a FIFO-private
cost model.

## Handshake Dependency Projection

The FIFO owner derives signal-level combinational arcs from the selected mode;
Mapping does not persist a second cycle-breaking flag.

`Bypass` contributes the transparent forward valid dependency from input to
output and the transparent backward ready dependency from output to input.
`Buffered` never contributes an input-valid to output-valid dependency because
an enqueued token is not visible until a later cycle. Its same-cycle
dequeue/enqueue rule may still contribute an output-ready to input-ready
dependency when downstream acceptance releases full capacity. Therefore
"stateful" is not a blanket assertion that every combinational dependency is
cut. A registered-ready, skid-isolated, or otherwise stronger break exists
only when Fabric declares that exact typed capability or refinement.

Fabric structural verification considers only FIFO arcs unconditional across
all legal configured views. Mapping verification uses the exact selected mode
and refinement. A bypass selection that closes a directed combinational
handshake cycle is intrinsically invalid Mapping. Selecting buffered mode is
legal only when the complete derived graph, including any remaining ready
dependency, is acyclic.

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
or same-cycle fall-through in buffered mode.

Fabric-to-RTL implements the same capability and selected-mode behavior. It
may choose any circuit structure consistent with the declared capacity,
handshake, visibility, lifecycle, and ConfigurationABI. Backend pipeline or
storage details do not become a second architectural contract.

## Verification Anchors

Anchor-level tests cover one buffered occurrence at empty, full, enqueue,
dequeue, and simultaneous dequeue/enqueue boundaries; one bypassable
occurrence in both legal modes with propagated backpressure; rejection of
bypass on a non-bypassable occurrence; derivation of configured mode from the
selected RouteTree; and a topology in which bypass closes a selected
handshake cycle while a declared registered break does not. Tests do not
preserve internal pointer encoding,
queue implementation, raw configuration bits, or exhaustive occupancy traces.
