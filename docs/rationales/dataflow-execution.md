# Dataflow Execution Rationale

Normative contracts are owned by
[SCF To DFG](../spec-compiler-part-3-dfg.md),
[Streaming And Channels](../spec-dataflow-part-1-streaming.md),
[Control Operations](../spec-dataflow-part-2-control.md), and
[Memory Consistency](../spec-dataflow-memory-consistency.md).

## Why There Is One Canonical Graph

The earlier regional graph and graph-function split represented one concept in
two incompatible forms. Loom keeps one symbol-bearing module-scope
`dataflow.graph`. Its body is a single graph region: operations are actors,
SSA use-def relations are edges, textual order has no execution meaning, and
cycles require explicit state or initialization.

Graphs forbid implicit capture because a persistent, simulatable, and mappable
program must expose every value, stream, memory capability, and symbol
dependency. Internal constants remain ordinary actors. Residual CFG or nested
SCF would reintroduce a PC into the SpatialCore graph and is rejected at
canonical publication.

The same closure rule requires total physical accounting without pretending
that every graph relation is a compute actor. Compute actors use Fabric
operation realizations, addressed actors use Fabric memory realizations,
logical roots use Mapping memory bindings, root-preserving views share their
root's storage, and launch/results/completion use boundary obligations. An
unrealized conversion has none of these owners and is therefore invalid rather
than receiving a simulator-only or Mapping-only exception.

Target-specific actor grouping is not part of the canonical graph. The design
reason for retiring `dataflow.subgraph` and placing that relation in Mapping is
owned by [Mapping And PnR](mapping-and-pnr.md).

## Why The Graph ABI Has Three Planes

Values, streams, and memories are fundamentally different but compose at one
graph boundary:

* a value is one invocation result or input;
* a stream is a zero-or-more ordered, incremental, backpressured sequence; and
* a memory is access capability to an addressable object or service.

An LLVM pointer belongs to the value or stream plane because its address is
dynamic program data. A memref belongs to the memory plane because it denotes
a logical object or service capability. Keeping them separate lets pointer
arithmetic flow through a graph without turning an address value into storage
ownership, and lets Mapping select a memory provider without inventing pointer
identity.

Input and output directions are symmetric. A memory export can expose an
internally created object just as an import consumes an external capability.
The hardware boundary has an analogous manager/subordinate distinction, but
software and hardware describe different owners and are connected only by
Mapping.

Start and retirement are invocation protocol, not application ports. Launch is
asynchronous; value results become ready on their real data dependency, while
explicit wait consumes retirement events. This permits InstructionCore work
to overlap without turning every launch into a barrier.

`dataflow.graph.return` is a structural declaration of outputs and the minimal
completion frontier, not an imperative function return reached by textual
execution. Invocation-local state is isolated and must return to its canonical
closed state before retirement; persistent state crosses invocations only
through explicit memory or messages.

## Why Thread Is Non-Recursive And Multi-Dimensional

A dynamic thread instance is the AccCore scheduling unit. Recursive threads
would create a second device-spawn runtime and obscure resource ownership.
Arbitrary nested parallelism is instead expressed with a logical launch domain,
thread splitting or fusion, and graph recurrence inside an instance.

One launch produces one collective completion token for its entire domain.
Per-instance data dependencies use channels, not a family of per-axis join
tokens. Yield is inserted at the unique outlining exit and contains the
minimal outstanding causal frontier; it is derived from the selected boundary,
not an independent DSE variable.

## Why Dynamic Work Reuses The Thread Domain

Irregular traversal needs a domain whose items can be discovered while work is
executing, but it does not need another thread, graph, or completion model.
`DynamicWork` is therefore one logical-domain variant. A controlled spawn
publishes a same-domain child only after atomically acquiring responsibility;
collective completion follows closure of the root source and an empty active
responsibility set.

The first closed profile intentionally has no channel endpoint. A work-item ID
and a channel message position answer different questions, and inventing an
implicit correspondence would reintroduce sessions, epochs, or hidden queues.
Memory and explicit atomic/coherence services carry shared irregular state.
A bounded execution-local scheduler may distribute queued items among
transient workers while one `DynamicWorkDomain` remains the responsibility and
completion owner. Deques, live assignments, cancellation requests, and the
ordered scheduling trace are Runtime state; they neither alter `WorkItemId` nor
select Mapping. Serializing those transitions with a host mutex makes queued
payloads visible to host workers without inventing program-visible memory
ordering.

The first execution adapter intentionally closes only the root singleton. Its
stable key is a Dataflow projection of the distinguished root, independent of
the Runtime dispatch occurrence. SystemMapping selects the persistent
Instruction, Spatial, and service-plan contexts from that key; the bounded
scheduler still owns only transient worker placement. The concrete CGRA entry
admits the narrower case where one scalar root payload is forwarded unchanged
to one direct graph and the thread body contains only that launch and its
yield. CGRA retirement is observed before the existing responsibility owner
completes the item. The generic synchronous executor boundary remains an
integration hook and cannot establish body execution by itself.

This does not admit a program-visible or device-side shared queue, child
publication lineage, channel correspondence, active-item migration,
remapping, launch captures, or provider image transport. Those capabilities
need their exact semantic and execution owners, not weaker ordered channel
semantics, recursive-thread ownership, or a second DynamicWork completion
rule.

## Why Channel Is A Separate Ordered Carrier

Thread completion, incremental communication, and random-access memory cannot
substitute for one another. Loom therefore has a single typed channel carrier
for thread-level ordered messages. Graphs see graph-local streams and bind them
to channel endpoints at launch; channel handles never enter a graph body.

A channel has one producer and one or more consumers. Each consumer owns a
source map from its logical domain to the producer domain, which naturally
expresses multicast without a rank-specific channel type. Multicast replication
remains a real transport obligation rather than an exception to endpoint
capacity rules.

Dynamic correspondence is between message events, not thread activations. For
each branch, the nth receive consumes the nth send in deterministic launch and
binding-local order. Coordinate equality alone cannot equate messages from
repeated launches, while adding epochs or one-activation segments would
duplicate FIFO order and prohibit legitimate rate conversion.

Channels deliberately expose no logical capacity, `try` operation, built-in
EOS, session, or release/acquire mode. Length belongs to the domain or explicit
message protocol; each delivered message is an ordinary causal edge. Physical
realization may use a NoC, switch, virtual channel, distributed buffers,
memory-backed queue, or FIFO, but must preserve the same ordered behavior.

## Why Stateful Actors Share One Transition Model

Stateless and stateful actors both consume required input heads, atomically
commit state and output obligations, and retire only after promised outputs
are accepted. This avoids an execution protocol per operation and preserves
ordered Dataflow under backpressure.

`stream`, `carry`, `invariant`, and `gate` remain four small actors because
they represent distinct recurrence roles. The phase stream has valid body
items followed by one close item; the IV stream contains only valid IVs. The
old invalid sentinel IV forced safe-address workarounds and confused
cardinality. Carry and invariant retain their intended body-value cardinality
while consuming the close phase needed to reset state.

Dynamic actor state belongs to the enclosing PE Instruction Context, not a
parallel context system on each `fabric.op`. A configured FU can contain
multiple independently firing actors; neither the FU nor a Compute Realization
is promoted to one macro firing.

## Why Completion Uses A Frontier

Collecting every historical done token grows without bound and obscures
causality. A region keeps only the terminal antichain not already covered by a
later event. A serial ordered chain can use its final done; a selection uses a
path-aware merge; a parallel construct joins all live branch frontiers.

This same algebra connects graph start, memory access control, loop recurrence,
and graph retirement. It prevents a strong global fence while proving that all
promised effects are complete.

## Why Memory Order Is An Explicit Event Network

Memory capability and memory transactions are separate planes. Capabilities
carry root, view, extent, alias, and lifetime facts; load/store actors connect
them to address, data, mask, control, result, and completion streams.

RAW, WAR, and WAW order is materialized as ordinary control/done causality
before graph publication. The simulator and mapper do not infer hazards from
text order or rebuild an alias graph. Compiler-local alias partitions carry a
write and read frontier during recursive lowering, then disappear once the
event network is explicit.

Issue, linearization, and retirement are separate semantic moments. This is
necessary for atomic, volatile, MMIO, and coherent behavior and prevents a
backend from treating a request acceptance as completed memory semantics.
Plain memory remains the simplest instance of the same consistency kernel.

Lifetime is derived from roots, views, escape, asynchronous use, and causal
last use. Owned/borrowed flags and memory-ready tokens were rejected because
they duplicate facts already present in the capability and completion planes.

## Why Vector And Reduction Semantics Stay Typed

Standard vector and mask types own lane shape. Scalar, contiguous vector,
gather, and scatter accesses use one load/store actor whose operands determine
the access form. Separate operation names would multiply the ISA without
adding semantics.

An ordered recurrence becomes a reduction only when the combiner has proven
or explicitly authorized algebraic freedom. This distinction is what permits
tree or partial reduction schedules without silently reordering a general
loop-carried state transition.

## Why Static Events Are Structural References

Graph ports, actor ports, channel sites, memory views, and transfer events
already have stable owners. Giving every leaf an EntityId would duplicate
structure and amplify identity churn; generic paths would create a second
schema interpreter. Dataflow therefore owns closed owner-relative references.

An event family is either a static produced or consumed transfer terminal or
one rooted contextual actor transition already defined by OperationSchema.
The latter is necessary when a selected System service resource is activated
by an internal memory or fence actor: no thread, graph, or channel boundary
terminal denotes that issue. Its logical coordinate and launch-parameter
projection is mechanically derived from the exact program rather than
persisted again. Runtime occurrences and Physical Tags remain transient
execution and Mapping facts.

Using a rooted actor transition avoids two competing authorities. Mapping does
not invent a service request event, and operand order does not accidentally
choose which consumption represents an atomic firing. Reserving the provider
for a whole graph launch is also rejected because an exposed capability or an
idle graph does not imply a dynamic access. Dataflow owns the firing event,
while the selected Fabric UsePattern owns service timing and completion.

The same ownership rule applies to thread definitions. A root launch already
has persistent identity and an exact callee relation, while logical coordinates
identify domain points and runtime owns concrete occurrences. Adding a
thread-definition EntityId would create a second path to the same definition
without identifying any dynamic instance. Rooted consumers therefore begin at
`RootThreadLaunchRef` and recover the definition mechanically.
### Ordered Message Correspondence

Ordered token positions constrain how multiple input roles may form one actor
tuple. Physical tags remain local interpretation keys at the selected ingress;
they do not replace Dataflow event identity or become globally unique stream
names. The compiler may expose rate and ordered-edge facts for early ranking,
but unknown correspondence remains typed incomplete until Mapping owns the
physical queue and selector projection.

Registered actor handshake cases are the owner of which input roles may be
consumed by one firing. Tech boundary projection retains only external roles;
roles fed by one common logical producer are one atomic-fanout member, while
distinct producers remain independently ordered members. Spatial search may
use this derived relation to rank ingress choices, but it cannot redefine the
actor firing or infer physical liveness from it.
