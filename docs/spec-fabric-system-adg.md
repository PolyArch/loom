# Fabric System ADG

This document specifies the target system-level Architecture
Description Graph represented in the fabric dialect. It covers whole
heterogeneous systems such as:

```text
HostCore + AccCore x M + cache hierarchy + interconnect + external memory
```

The system ADG is hardware architecture IR. It is not a software
mapping artifact, not an RTL netlist, and not a replacement for
`fabric.module`.

Loom system ADG uses physical address spaces. Virtual address
translation, MMU state, and virtual-memory page tables are outside the
Fabric ADG target. A runtime or platform adapter may translate host
pointers into accelerator-accessible physical or device addresses, but
that is not a Fabric system virtual-address feature.

## Design Principles

The system ADG follows the same minimal, RISC-style design principle as
the dataflow dialect:

* Prefer a small number of primitive node kinds with precise semantics.
* Prefer explicit graph edges over implicit topology assumptions.
* Express derived structures through builder conveniences that expand
  into primitive IR.
* Avoid generic meta-nodes when a structure can be composed from
  simpler verified nodes and links.
* Keep software selection operations and hardware routing operations
  separate in naming and semantics.

This simplicity is semantic simplicity, not reduced product scope. The
system ADG should expose first-principles hardware architecture
concepts, make them explicit and composable, and avoid opaque meta-nodes.
The resulting implementation can still be large, systematic, and
complete.

Meshes, arrays, x/y coordinates, and Manhattan routing are optional
metadata or builder conveniences. They are not system semantics.

## Relation to `fabric.module`

`fabric.module` remains the SpatialCore or CGRA fabric template. It
describes the internal fabric graph for one SpatialCore template.

`fabric.system` is the system-level ADG container. It describes physical
instances of host cores, accelerator cores, caches, memory, routers,
adapters, external ports, and links between their protocol channels.

An `acc_core` node may reference a `fabric.module` symbol as its
SpatialCore template. Every `acc_core` node is an independent physical
instance even when multiple nodes reference the same `fabric.module`
symbol.

`fabric.module` connectivity is represented by Graph-region SSA values.
`fabric.system` connectivity is represented by `fabric.link`
operations. These are different hardware levels and remain long-term
parallel contracts.

## `fabric.system`

`fabric.system` is a module-scope symbol-bearing op. It contains a
symbolic graph:

* `fabric.node` operations define physical nodes.
* `fabric.external_port` operations define complete intentional protocol
  ports at the system boundary.
* `fabric.link` operations define directed point-to-point channel
  connections.
* `fabric.clock_domain`, `fabric.reset_domain`, and
  `fabric.power_domain` operations define optional domain metadata.
* `fabric.coherence_domain` operations define cache-coherence domains.

Multiple `fabric.system` symbols may appear in one MLIR module. A
consumer must explicitly select the target system unless the module
contains exactly one system.

The body is a symbolic graph description. It does not use SSA edges or a
region terminator to encode connectivity. `fabric.link` is the
connectivity source of truth.

`fabric.system` and its child objects may carry optional visualization
metadata. Visualization metadata helps a GUI draw regular structures
such as two-dimensional meshes, three-dimensional grids, trees, or
hierarchies. It does not define topology, routing, placement, hardware
legality, or mapping legality.

## Nodes

`fabric.node` represents one physical hardware instance. The node kind
is a fixed enum, not a free-form string. The baseline node kinds are:

| Kind | Meaning |
|------|---------|
| `host_core` | Host processor core. |
| `acc_core` | Accelerator core containing ScalarCore parameters and a SpatialCore template reference. |
| `fixed_accelerator` | Fixed-function or narrowly programmable accelerator with explicit control, data, stream, or memory ports. |
| `cache` | Cache or cache-like coherent memory hierarchy node. |
| `memory` | Terminal storage service such as external memory, scratchpad, SRAM, or MMIO endpoint. |
| `dma_engine` | Explicit data-movement engine with control or descriptor ports and memory-beat ports. |
| `router` | Packet routing primitive. |
| `network_endpoint` | Interface between a non-network node and a router network. |
| `arbiter` | N-to-1 contention-resolution primitive. |
| `route_decoder` | 1-to-one-of-N deterministic routing primitive. |
| `broadcast` | 1-to-N replication primitive. |
| `width_converter` | Data-width adapter. |
| `clock_converter` | Clock-domain adapter. |
| `protocol_converter` | Protocol adapter. |
| `custom` | User-defined primitive with explicit ports, channels, and params. |

Core invariants are strongly typed through dedicated attributes. Other
node-specific data lives in an open `params` dictionary.

There is no default `io_engine` core kind. IO-facing behavior is
represented through external ports, protocol endpoints, DMA or memory
engines, or explicit system node ports.

### Node Attribute Contract

A `fabric.node` has these common fields:

| Field | Required | Meaning |
|-------|----------|---------|
| `sym_name` | yes | Unique symbol name within the enclosing system. |
| `kind` | yes | One baseline node-kind enum value. |
| `ports` | yes | Ordered array of complete `#fabric.port` attributes. |
| `clock_domain` | no | Node clock-domain override. Defaults to the system default domain. |
| `reset_domain` | no | Node reset-domain override. Defaults to the system default domain. |
| `power_domain` | no | Node power-domain override. Defaults to the system default domain. |
| `visual` | no | Visualization metadata. Never affects hardware semantics. |
| `params` | no | Node-kind-specific implementation, timing, capacity, and policy metadata. |

Baseline node kinds use the fields below as the required semantic
contract. `params` may refine a node, but it must not weaken the
required fields or required port shape for that node kind.

The interconnect rules below use input-side and output-side in the
transaction-flow sense. For manager/subordinate protocols, an input-side
port is usually a subordinate port that receives upstream requests, and
an output-side port is usually a manager port that issues downstream
requests. For stream-like protocols, the protocol definition provides
the transaction-flow classification from channel directions.

### Node Kind Contract

| Kind | Required attributes | Required ports | Required params |
|------|---------------------|----------------|-----------------|
| `host_core` | none | At least one memory-capable manager port. | none |
| `acc_core` | `spatial`, `scalar` | At least one memory-capable port. | none |
| `fixed_accelerator` | none | Non-empty explicit ports for its control, data, stream, or memory interfaces. | `function` |
| `cache` | none | At least one upstream memory-capable subordinate port and at least one downstream memory-capable manager port. | `line_bytes`, `capacity_bytes` |
| `memory` | none | At least one terminal memory target port. | none |
| `dma_engine` | none | At least one control or descriptor port and at least one memory-capable manager port. | `policy` |
| `router` | none | At least two network-facing packet ports. | `routing` |
| `network_endpoint` | none | At least one local-facing port and at least one network-facing port. | `network_protocol` |
| `arbiter` | none | At least two input-side ports and exactly one output-side port for the arbitrated transaction class. | `policy` |
| `route_decoder` | none | Exactly one input-side port and at least two output-side ports. | `decode` |
| `broadcast` | none | Exactly one input-side port and at least two output-side ports. | none |
| `width_converter` | none | Exactly one input-side port and exactly one output-side port. | `in_width`, `out_width` |
| `clock_converter` | none | Exactly one input-side port and exactly one output-side port, each with an explicit port-level `clock_domain`. | none |
| `protocol_converter` | none | Exactly one input-side port and exactly one output-side port with different endpoint protocols. | none |
| `custom` | none | Non-empty explicit ports and channels. | `kind_name` |

`clock_converter` is the only baseline node kind whose ports may carry
port-level clock-domain overrides. The two port-level clock domains must
be different. Link clock-domain legality checks use the endpoint port's
clock domain when present; otherwise they use the owning node's clock
domain.

`width_converter` ports must describe unequal widths through the
required width params. A same-width adapter is not a width conversion
and should be removed or represented as ordinary links.

`protocol_converter` is the only baseline node kind that may directly
connect two different endpoint protocols. Its two sides must name the
source and destination protocols through their port declarations.

`custom` nodes remain explicit graph nodes, not opaque meta-nodes. They
must declare a stable `kind_name`, complete ports, complete channels,
and memory capability for every `custom` protocol port.

Required node params have these baseline verifier rules:

| Param | Rule |
|-------|------|
| `line_bytes` | Positive power-of-two byte count. |
| `capacity_bytes` | Positive byte count and at least one line. |
| `routing` | One of `static_table`, `source_routed`, `adaptive`, or `custom`; `custom` requires a non-empty payload. |
| `network_protocol` | Names the packet protocol used on router-facing ports. |
| `policy` | One of `fixed_priority`, `round_robin`, `weighted_round_robin`, or `custom`; weighted and custom policies require non-empty policy payload. |
| `decode` | Non-empty deterministic decode table or custom deterministic decode payload. |
| `in_width` | Positive data width in bits. |
| `out_width` | Positive data width in bits and unequal to `in_width`. |
| `kind_name` | Non-empty stable identifier for the custom node semantics. |
| `function` | Non-empty stable identifier for the fixed-function accelerator semantics. |

### Host Core

A `host_core` node must have at least one memory-capable manager port.
It may carry a scalar-core profile in `params`. It may join a coherence
domain through its memory port.

### Accelerator Core

An `acc_core` node must define:

* `spatial = @symbol`, referencing a visible `fabric.module`.
* `scalar = #fabric.scalar_core<...>`, including at least the scalar
  core kind or ISA.
* at least one memory-capable port.

Control, configuration, debug, interrupt, and local-memory ports are
optional. Scratchpad or local SRAM is represented as a separate
`memory` node, not as hidden storage inside the `acc_core` node.

### Fixed-Function Accelerator

A `fixed_accelerator` node represents a concrete accelerator block whose
hardware behavior is not described by a `fabric.module` SpatialCore
template. It may be a fixed-function engine or a narrowly programmable
block. It remains a system node with explicit ports, channels, params,
domains, address-space participation, and optional coherence-domain
membership.

The `function` param names the block semantics. Recommended params
include supported operation family, data widths, descriptor format,
configuration registers, queue depth, latency model, throughput model,
and supported memory or stream protocols. Memory-capable ports follow
the same address-space and range rules as other system nodes.

A fixed-function accelerator must not hide DMA, memory movement,
coherence participation, interrupts, or external interfaces. Those facts
remain explicit ports, links, DMA engines, memory nodes, coherence
memberships, runtime descriptors, or mapping records.

### Cache

A `cache` node must define at least one upstream memory-capable port and
at least one downstream memory-capable port. It must define line size and
capacity. If the cache participates in a coherence domain, its line size
must match the domain line size.

Recommended cache params include associativity, set count, write
policy, allocate policy, hit latency, miss latency, and MSHR count.

### Memory

A `memory` node represents final storage services. A terminal memory
target port is any port that satisfies all of these conditions:

* the owning node kind is `memory`;
* the port role is `subordinate`;
* the port is memory-capable.

Every terminal memory target port must declare `addr_space` and a
non-empty `addr_ranges` array. External DRAM, scratchpad SRAM, local
SRAM, and MMIO endpoints are all modeled as `memory` nodes.

A `memory` node may also expose non-terminal ports, but only
memory-capable subordinate ports on a `memory` node are terminal memory
target ports.

Caches, interconnect nodes, arbiters, routers, `route_decoder` nodes,
`broadcast` nodes, `network_endpoint` nodes, `protocol_converter` nodes,
and DMA manager ports are not terminal memory targets for address-range
overlap checking.

### DMA Engine

A `dma_engine` node represents explicit data movement. It must expose
control or descriptor-facing ports and memory-capable manager ports. A
DMA engine may connect host-visible memory, accelerator-local memory,
scratchpads, external memory, or simulator memory models, but every
reachable memory target still comes from explicit ports, links, address
spaces, and mapping or runtime descriptors.

DMA descriptors, queue depth, burst support, ordering policy, and
coherence participation are represented by node params and port
metadata. A DMA engine is not an implicit IO core and must not hide
memory movement from mapping, runtime, simulation, RTL, or FPA evidence.

## Ports and Channels

System-level connectivity is expressed through protocol ports and
directed channels.

A port is a protocol interface bundle. A channel is the directed
endpoint that can be connected by `fabric.link`. A link always connects
one channel endpoint to one channel endpoint.

Port roles use `manager` and `subordinate`. The direction convention is:

```text
output -> input
manager -> subordinate
```

For protocols with mixed channel directions, direction belongs to each
channel, not to the whole port. For example, an AXI4-MM manager port has
output request channels such as `aw`, `w`, and `ar`, and input response
channels such as `b` and `r`. The corresponding subordinate port has the
opposite directions.

The target form stores port and channel declarations as structured
attributes on `fabric.node` and `fabric.external_port`. The target does
not introduce `fabric.port` or `fabric.channel` child ops.

The canonical attribute family is:

* `#fabric.port<...>` for a protocol interface bundle.
* `#fabric.channel<...>` for one directed channel inside a port.
* `#fabric.endpoint<...>` for a linkable channel endpoint.

A `#fabric.port` attribute contains:

| Field | Required | Meaning |
|-------|----------|---------|
| `name` | yes | Unique port name within the owning node. |
| `protocol` | yes | Built-in protocol enum or `custom`. |
| `role` | yes | `manager` or `subordinate`. |
| `channels` | yes | Ordered array of `#fabric.channel` attributes. |
| `addr_space` | no | Physical address-space identifier for memory-capable ports. |
| `addr_ranges` | no | Array of physical address ranges served or covered by this port. |
| `clock_domain` | no | Port-level clock-domain override. Legal only on `clock_converter` ports. |
| `optional` | no | Whether the entire port may remain unconnected. Defaults to `false`. |
| `params` | no | Protocol- or node-specific dictionary. |

Memory capability is derived from the port's protocol, role, and
protocol-specific params. `addr_space` and `addr_ranges` may appear only
on memory-capable ports. Their presence does not make a port
memory-capable by itself.

A `#fabric.channel` attribute contains:

| Field | Required | Meaning |
|-------|----------|---------|
| `name` | yes | Unique channel name within the owning port. |
| `direction` | yes | `input` or `output`, from the owning port's perspective. |
| `optional` | no | Whether this channel may remain unconnected. Defaults to `false`. |
| `default` | no | Default value for an unconnected optional input. Defaults to zero. |
| `params` | no | Protocol-specific dictionary. |

Node endpoints and external endpoints use distinct forms so that system
boundary exposure cannot be confused with an internal node channel:

```mlir
#fabric.endpoint<node = @acc0, port = "mem", channel = "aw">
#fabric.endpoint<external = @host_mem, channel = "aw">
```

The node endpoint's symbol resolves to a `fabric.node` in the enclosing
`fabric.system`. The external endpoint's symbol resolves to a
`fabric.external_port` in the same system. An external endpoint does not
carry a port name because `fabric.external_port` owns exactly one
complete protocol port.

A representative node declaration is:

```mlir
fabric.node @acc0 kind = #fabric.node_kind<acc_core>
  attributes {
    spatial = @spatial0,
    scalar = #fabric.scalar_core<kind = "rv32im">,
    ports = [
      #fabric.port<
        name = "mem",
        protocol = axi4_mm,
        role = manager,
        channels = [
          #fabric.channel<name = "aw", direction = output>,
          #fabric.channel<name = "w",  direction = output>,
          #fabric.channel<name = "b",  direction = input>,
          #fabric.channel<name = "ar", direction = output>,
          #fabric.channel<name = "r",  direction = input>
        ]>
    ]
  }
```

The `fabric.system` target does not use child-op port or channel
declarations. Nodes and external ports own protocol ports, ports own
directed channels, and links connect explicit channel endpoints.

## Protocols

The baseline protocol enum includes:

* `axi4_mm`
* `axi4_lite`
* `csr`
* `dma`
* `interrupt`
* `packet`
* `stream`
* `custom`

Built-in protocols define required channel sets, channel directions,
minimal required params, transaction-flow classification, and memory
capability for each role. They do not define a complete bus
implementation. Payload details such as AXI burst fields, protection
bits, response bits, user fields, or packet header formats are channel
payload metadata carried by the protocol params.

The verifier treats built-in protocol schemas as closed over channel
names. A built-in port may declare only the required channels and the
optional channels allowed by that protocol schema. Unknown built-in
channel names are illegal. Use `custom` for protocol extensions that
need different channel names.

### Built-in Protocol Schemas

Channel directions are from the owning port's perspective.

| Protocol | Role | Required channels | Conditionally required channels | Required params | Transaction-flow side |
|----------|------|-------------------|---------------------------------|-----------------|-----------------------|
| `axi4_mm` | `manager` | `aw:output`, `w:output`, `b:input`, `ar:output`, `r:input` | none | `addr_width`, `data_width` | output-side |
| `axi4_mm` | `subordinate` | `aw:input`, `w:input`, `b:output`, `ar:input`, `r:output` | none | `addr_width`, `data_width` | input-side |
| `axi4_lite` | `manager` | `aw:output`, `w:output`, `b:input`, `ar:output`, `r:input` | none | `addr_width`, `data_width` | output-side |
| `axi4_lite` | `subordinate` | `aw:input`, `w:input`, `b:output`, `ar:input`, `r:output` | none | `addr_width`, `data_width` | input-side |
| `csr` | `manager` | `req:output`, `resp:input` | none | `addr_width`, `data_width` | output-side |
| `csr` | `subordinate` | `req:input`, `resp:output` | none | `addr_width`, `data_width` | input-side |
| `dma` | `manager` | `cmd:output`, `status:input` | none | `addr_width`, `length_width` | output-side |
| `dma` | `subordinate` | `cmd:input`, `status:output` | none | `addr_width`, `length_width` | input-side |
| `interrupt` | `manager` | `irq:output` | none | none | output-side |
| `interrupt` | `subordinate` | `irq:input` | none | none | input-side |
| `packet` | `manager` | `flit:output` | `credit:input` when `params.flow_control = credit` | `flit_width` | output-side |
| `packet` | `subordinate` | `flit:input` | `credit:output` when `params.flow_control = credit` | `flit_width` | input-side |
| `stream` | `manager` | `data:output` | `ready:input` when `params.flow_control = ready_valid` | `data_width` | output-side |
| `stream` | `subordinate` | `data:input` | `ready:output` when `params.flow_control = ready_valid` | `data_width` | input-side |

Full-duplex traffic is expressed by two directed protocol ports or by
two directed channel-link sets. A single `stream` or `packet` port is a
single transaction-flow direction.

`dma` is a descriptor-level memory movement protocol. It is
memory-capable because its commands name physical addresses and lengths.
The actual memory-beat interfaces of a DMA engine are still represented
as separate memory-capable ports, such as `axi4_mm`, `axi4_lite`, or
custom memory ports.

Protocol params have these baseline verifier rules. Params listed as
required in the protocol schema must be present; optional params are
checked only when present, except where a default is specified.

| Param | Rule |
|-------|------|
| `addr_width` | Positive address width in bits. |
| `data_width` | Positive data width in bits and a multiple of 8. |
| `length_width` | Positive transfer-length width in bits. |
| `flit_width` | Positive packet flit width in bits. |
| `flow_control` | For `stream`, one of `none` or `ready_valid`; for `packet`, one of `none` or `credit`. |
| `protocol_name` | Non-empty stable identifier for `custom` protocol semantics. |

`flow_control` defaults to `none` when absent. If a port declares
`flow_control = ready_valid`, both sides of a connected `stream` port
must expose the `ready` channel. If a port declares
`flow_control = credit`, both sides of a connected `packet` port must
expose the `credit` channel.

`axi4_mm` may declare optional `id_width` and `user_width` params. If
present, they must be non-negative integers. `axi4_lite` must not
declare `id_width` because AXI4-Lite has no transaction IDs in the
target schema. Both AXI schemas model their five channel groups as
atomic directed channels; individual signal fields live in payload
metadata, not in separate ADG channels.

`custom` requires `params.protocol_name`, explicit channel names,
explicit channel directions, explicit transaction-flow classification,
explicit memory capability, and any custom-required params. A `custom`
protocol is a primitive extension point with a complete declared schema;
it is not an escape hatch for implicit or partially specified hardware.

Memory-capability rules:

| Protocol | Role | Memory-capable rule |
|----------|------|---------------------|
| `axi4_mm` | `manager`, `subordinate` | always memory-capable |
| `axi4_lite` | `manager`, `subordinate` | always memory-capable |
| `dma` | `manager`, `subordinate` | always memory-capable |
| `csr` | `manager`, `subordinate` | memory-capable only when `params.memory_capable = true` |
| `stream` | any | never memory-capable |
| `interrupt` | any | never memory-capable |
| `packet` | any | never memory-capable |
| `custom` | any | must declare `params.memory_capable = true` or `false` |

`csr` defaults to not memory-capable because many CSR links are control
interfaces rather than addressable memory ports. A memory-mapped CSR or
MMIO control port sets `params.memory_capable = true` and then follows
the same address-space and range rules as other memory-capable ports.

For AXI4-MM and AXI4-Lite, the canonical channel set is `aw`, `w`,
`b`, `ar`, and `r`. A complete built-in port must declare every
required channel unless the protocol schema explicitly defines a
smaller variant.

## Links

`fabric.link` is the only canonical connectivity op. It connects one
directed output channel endpoint to one directed input channel endpoint.
The canonical endpoint representation is `#fabric.endpoint<...>`.

A representative link declaration is:

```mlir
fabric.link
  src = #fabric.endpoint<node = @host0, port = "mem", channel = "aw">
  dst = #fabric.endpoint<node = @l2,    port = "up",  channel = "aw">
```

The fixed `fabric.link` fields are:

| Field | Required | Meaning |
|-------|----------|---------|
| `src` | yes | Output channel endpoint. |
| `dst` | yes | Input channel endpoint. |
| `protocol` | no | Protocol enum for this link. If absent, it is inferred from both endpoint ports. If present, it must match both endpoint ports. |
| `latency` | no | Non-negative latency metadata for simulation, RTL generation, or estimation. |
| `bandwidth` | no | Positive bandwidth metadata for simulation, RTL generation, or estimation. |
| `crossing` | conditionally | Clock-domain crossing description. Required when endpoint effective clock domains differ. |
| `visual` | no | Visualization metadata. Never affects connectivity or hardware semantics. |
| `params` | no | Open dictionary for hardware link implementation metadata. |

`src` and `dst` are the only fields required for connectivity. They
define the graph edge. All other fixed fields are metadata except
`crossing`, which becomes a legality requirement when the endpoints are
in different effective clock domains.

When `protocol` is absent, the verifier infers it from the source and
destination ports. The inferred protocols must match. If two endpoints
use different protocols, the link is illegal; the system graph must
insert a `protocol_converter` node and connect each side with its own
one-to-one links.

`latency` and `bandwidth` do not change connectivity, ownership, or
protocol legality. They are hardware facts or estimates used by
CGRA-sim, RTL generation, and FPA estimation.

`crossing` is absent when both endpoint effective clock domains are the
same. A node with no explicit clock-domain annotation belongs to the
system default clock domain, and a port with no port-level clock-domain
override inherits the owning node's clock domain. If endpoint effective
clock domains differ, `crossing` must be present and must describe the
crossing mechanism, such as `async_fifo`, `cdc_sync`,
`explicit_bridge`, or `custom`. An explicit `clock_converter` node may
be used instead of a link-level `crossing` field; in that case each link
endpoint pair is same-domain. A `crossing` field on a same-domain link
is illegal.

`params` may contain physical or implementation metadata such as wire
class, estimated length, technology hint, or buffer-depth hint. It must
not contain software mapping decisions, route paths selected by PnR,
schedule slots, temporal tags, placement decisions, or workload-specific
state. Those belong to a separate mapping artifact.

Every link is point-to-point:

* one output channel endpoint
* one input channel endpoint
* no implicit fanout
* no implicit fan-in

An output channel may appear in at most one link. An input channel may
appear in at most one link. Fanout, broadcast, arbitration, crossbar
behavior, and multi-manager sharing must be modeled through explicit
nodes with their own ports, channels, and one-to-one links.

Builder-level bundle connection syntax is allowed only as sugar. It
must expand into per-channel `fabric.link` operations, and each expanded
link must obey the point-to-point rule.

## External Ports and Dangling Channels

`fabric.external_port` declares intentional system boundary exposure. It
distinguishes a real top-level hardware port from an accidentally
unconnected internal channel.

A `fabric.external_port` is a complete protocol port, not a single
channel exposure. It owns exactly one `#fabric.port` attribute. For a
built-in protocol, that port must declare the complete required channel
set for its role. Each channel is still connected individually through a
`fabric.link` endpoint, but the externally exposed interface is the full
protocol bundle. An external port may carry optional `visual` metadata.

A representative declaration is:

```mlir
fabric.external_port @host_mem
  port = #fabric.port<
    name = "host_mem",
    protocol = axi4_mm,
    role = subordinate,
    channels = [
      #fabric.channel<name = "aw", direction = input>,
      #fabric.channel<name = "w",  direction = input>,
      #fabric.channel<name = "b",  direction = output>,
      #fabric.channel<name = "ar", direction = input>,
      #fabric.channel<name = "r",  direction = output>
    ]>
```

Non-optional internal channels must be connected or intentionally
exposed through a matching external port. Otherwise the hardware is
illegal.

Optional channels are declared at channel level with `optional = true`.
An unconnected optional input is inactive and defaults to zero unless a
channel-level `default` overrides it. An unconnected optional output is
unused and may be optimized away by downstream implementation tools if
it is otherwise unobservable.

Built-in protocol required channels are not optional unless the protocol
definition explicitly says so.

## Interconnect Primitives

The system ADG does not introduce hardware primitives named `mux` or
`demux`. Those names are reserved for software dataflow selection and
would blur the distinction between dataflow control and hardware
routing at system level. This restriction applies to `fabric.system`
node kinds. It does not rename the existing `fabric.mux` and
`fabric.demux` ops inside `fabric.fu`; those remain FU-local
configuration primitives inside `fabric.module` templates.

The baseline system interconnect primitives are:

* `route_decoder`: deterministic 1-to-one-of-N routing by address,
  route key, or protocol-defined target field. It does not replicate
  traffic.
* `arbiter`: N-to-1 selection with explicit contention policy such as
  fixed priority, round robin, weighted round robin, or custom policy.
* `broadcast`: 1-to-N replication of the same transaction or stream to
  multiple outputs.
* `router`: packet routing primitive for graph networks.
* `network_endpoint`: protocol boundary between ordinary nodes and a
  router network.
* `width_converter`, `clock_converter`, and `protocol_converter`:
  structural adapters.

A crossbar is not a primitive baseline node. The ADG Builder may expose
a crossbar convenience API, but it expands into `route_decoder`,
`arbiter`, and one-to-one links.

A NoC is not a single opaque node. It is a graph of `router`,
`network_endpoint`, adapter nodes when needed, and explicit directed
links.

## Clock, Reset, and Power Domains

Clock, reset, and power domains are explicit optional metadata. They are
used by RTL generation, FPA estimation, and legality checks that depend
on domains.

Every `fabric.system` has implicit default clock, reset, and power
domains. A node that does not declare a clock-domain, reset-domain, or
power-domain annotation belongs to the corresponding default domain.
Explicit domain annotations are required only for nodes outside the
corresponding default domain.

Ports inherit the owning node's clock domain by default. Port-level
clock-domain overrides are legal only on `clock_converter` ports. This
lets a `clock_converter` expose one transaction-flow side in the source
clock domain and the other side in the destination clock domain, while
each adjacent `fabric.link` remains same-domain.

If a link connects endpoints whose effective clock domains differ, the
crossing must be explicit. This can be represented by a link crossing
attribute or by an explicit `clock_converter` node. Silent cross-domain
connectivity is illegal. A same-domain link must not carry a link-level
`crossing` field; the verifier rejects it because no clock-domain
crossing exists on that edge.

Reset-domain and power-domain differences do not affect `fabric.link`
legality. They are metadata for RTL generation and FPA estimation. The
system ADG verifier checks that explicit reset-domain and power-domain
annotations reference declared domains, but it does not require link
crossing fields, adapters, isolation cells, level shifters, or reset
synchronizers solely because reset or power domains differ.

## Address Spaces

Memory-capable ports may declare physical address ranges. Loom does not
model MMUs or virtual memory in the Fabric system ADG. Address ranges
are physical.

`#fabric.addr_range<base, size>` denotes the half-open unsigned 64-bit
range:

```text
[base, base + size)
```

`size` must be greater than zero.

For the same physical address space, terminal memory target port ranges
must not overlap by default. Upstream cache and interconnect ports may
cover aggregate downstream ranges and are not terminal-overlap targets.
The overlap rule applies only to terminal memory target ports.
Non-terminal memory-capable manager, upstream, cache, and interconnect
ports may cover aggregate ranges and are excluded from terminal-overlap
checking.

Named memory regions are physical subranges associated with an address
space. A region may describe DRAM, scratchpad, SRAM, MMIO, or another
terminal memory service. Region identity is used by mapping, runtime,
simulation, RTL, FPA, and reports to explain which physical storage a
workload used. Region binding belongs in mapping and runtime artifacts;
`fabric.system` declares the physical address ranges and memory targets.

## Coherence and Consistency

Every `fabric.system` must declare a system memory consistency model
through a required `memory_model` field. The baseline enum is:

* `sequential`
* `tso`
* `release_acquire`
* `weak`
* `custom`

`custom` must provide a `model_name` string or a non-empty `params`
dictionary. The verifier checks that `memory_model` exists and uses a
valid enum value. Detailed ordering semantics are consumed by
simulators, lowering, and backend models; the system ADG verifier does
not implement a full memory-model proof system.

Domain-level or node-level consistency overrides may be represented as
metadata, but they do not replace the required system-level declaration.

`fabric.coherence_domain` defines cache coherence over memory-capable
ports. Membership references node port endpoints, not whole nodes and
not individual channels.

The domain is explicit and verifier-checked. It is not inferred from
topology. A domain declaration contains:

| Field | Required | Meaning |
|-------|----------|---------|
| `sym_name` | yes | Domain symbol unique within the enclosing system. |
| `addr_space` | yes | Physical address-space identifier covered by this coherence domain. |
| `line_bytes` | yes | Positive cache-line size in bytes. |
| `protocol` | yes | Coherence protocol enum or `custom`. |
| `members` | yes | Non-empty array of memory-capable port endpoints. |
| `home` | no | Optional home-agent port endpoint for directory-style protocols. |
| `params` | no | Protocol-specific dictionary. |

Membership uses port endpoints, not channel endpoints:

```mlir
#fabric.port_endpoint<node = @l2, port = "up">
#fabric.port_endpoint<external = @host_mem>
```

The node form resolves to a `fabric.node` port. The external form
resolves to the complete protocol port owned by a `fabric.external_port`.
Every member endpoint must refer to a memory-capable port whose
`addr_space` equals the domain `addr_space`.

The baseline coherence protocol enum is:

* `none`
* `snooping`
* `mesi`
* `moesi`
* `directory`
* `custom`

A representative declaration is:

```mlir
fabric.coherence_domain @coh0
  addr_space = 0
  line_bytes = 64
  protocol = #fabric.coherence_protocol<mesi>
  members = [
    #fabric.port_endpoint<node = @host0, port = "mem">,
    #fabric.port_endpoint<node = @l2,    port = "up">,
    #fabric.port_endpoint<node = @dram0, port = "mem">
  ]
```

A directory or home-agent coherence protocol may define an optional
`home` endpoint. `home` is legal for `directory` and for `custom`
protocols whose params explicitly declare home-agent semantics. It is
illegal for `none`, `snooping`, `mesi`, and `moesi`.

`protocol = none` explicitly records a non-coherent memory-sharing
group. It may contain memory-capable ports in the domain address space,
but it must not contain cache node ports. Coherent cache membership uses
one of the coherent protocols.

A memory-capable port may participate in at most one coherence domain.
This is a baseline verifier rule for the target contract.

If a cache is in a coherence domain, its line size must match the
domain line size. All cache member ports must belong to a cache node in
the same `addr_space`. Terminal memory target member port ranges remain
subject to the address-space non-overlap rule. Manager and upstream
ports may omit `addr_ranges`; if they declare ranges, those ranges must
belong to the domain `addr_space`.

## Visualization Metadata

Fabric ADG supports optional visualization metadata for human
inspection. This metadata is intended for GUI tools and reports. It
must not affect hardware semantics, verifier legality, PnR legality,
simulation, RTL lowering, or FPA estimation.

A system may define visualization layouts. A layout has:

| Field | Required | Meaning |
|-------|----------|---------|
| `name` | yes | Unique layout identifier inside the system. |
| `kind` | yes | `free_graph`, `grid2d`, `grid3d`, `hierarchy`, or `custom`. |
| `rank` | conditionally | Coordinate rank. Required for grid and custom coordinate layouts. |
| `dims` | no | Optional positive extents for grid layouts. |
| `axes` | no | Human-readable axis names. |
| `params` | no | GUI-oriented metadata. |

When represented in Fabric ADG, these layouts live in an optional
system-level `visual_layouts` attribute. Object-level `visual`
metadata may reference entries in that layout set.

Objects may attach visualization metadata that references a layout and
provides optional display hints:

| Field | Required | Meaning |
|-------|----------|---------|
| `layout` | no | Referenced visualization layout name. |
| `coord` | no | Integer coordinate vector in the referenced layout. |
| `pos` | no | Continuous drawing position for free-form layouts. |
| `group` | no | Visual grouping label. |
| `label` | no | Preferred display label. |
| `style` | no | GUI style class. |
| `params` | no | GUI-specific metadata. |

For a `grid2d` layout, `coord` rank is two. For a `grid3d` layout,
`coord` rank is three. When `dims` is present, coordinates must be
inside the declared extents. Coordinates need not be unique unless the
visualization layout declares uniqueness in `params`.

Visualization metadata may be attached to systems, nodes, external
ports, links, and domains. The metadata may describe regular views such
as a mesh accelerator array or a stacked three-dimensional topology, but
the explicit Fabric links remain the topology source of truth.

Tools that do not render visualizations must be able to ignore
visualization metadata without changing any hardware or mapping result.

## Target Universe

The `fabric.system` target universe includes:

* `host_core` nodes, `acc_core` nodes, `fixed_accelerator` nodes,
  `dma_engine` nodes, `memory` nodes, caches, routers, network
  endpoints, arbiters, route decoders, broadcasts, width converters,
  clock converters, protocol converters, external ports, and custom
  explicit nodes;
* `acc_core` nodes that carry ScalarCore metadata and reference
  `fabric.module` symbols as SpatialCore templates. ScalarCore and
  SpatialCore are not separate baseline `fabric.system` node kinds;
* physical address spaces, memory regions, memory targets, and memory
  consistency declarations;
* coherence domains and cache hierarchy metadata;
* clock, reset, and power domains;
* protocol schemas and directed channel endpoints;
* explicit one-to-one `fabric.link` connectivity;
* optional visualization metadata that never defines topology.

The first verifiable memory/coherence baseline is physical address space
plus memory regions, DMA or scratchpad resources, coherent domains, and
a simplified shared-memory coherence model. The final target includes
complete cache hierarchy and coherence protocol evidence for
heterogeneous multi-core SoCs, including L1/L2/LLC-like structures,
ordering rules, coherence events, and multi-core consistency checks.

## Required Evidence

Evidence for this spec includes verifier-positive and verifier-negative
system MLIR tests, ADG Builder examples, hardware summary artifacts, and
downstream references from mapping, simulation, RTL, FPA, and report
bundles.

Each supported node kind, port schema, link legality class,
address-space case, domain relation, and coherence-domain rule must have
positive and negative evidence.

## Objective Verification

The `fabric.system` target is objectively verifiable when:

* every target node kind and protocol class can be emitted and verified;
* every topology edge is represented by explicit endpoints and
  `fabric.link` operations;
* memory address ranges, address spaces, and coherence domains are
  checked for consistency;
* illegal protocol, domain, address-range, and coherence relationships
  are rejected;
* downstream tools can identify the selected `fabric.system` and
  referenced `fabric.module` templates without reading builder-only
  state.

## Unsupported Scope Policy

Unsupported system features must be represented by structured
diagnostics or unsupported-scope records. A tool must not silently
replace a missing cache, coherence behavior, router, DMA path, external
port, or protocol adapter with an implicit default. Visualization
coordinates must not make missing connectivity legal.

## Relationships To Other Contracts

`fabric.system` is produced by the system-level ADG Builder layer and
consumed by PnR, CGRA-sim, RTL lowering, FPA, report bundles, runtime
profiles, and DSE. SpatialCore templates referenced by `acc_core` nodes
are specified by `docs/spec-fabric-module.md`. Mapping choices belong
to `docs/spec-mapping-artifact.md`, not to this hardware description.

## Current Implementation Notes

This section is non-normative. It records current repository facts for
orientation only and is not part of target acceptance.

This document is ahead of the current implementation. The system-level
IR, verifier, and SystemBuilder API are target contracts that still need
implementation coverage. Existing SpatialCore `fabric.module` support
does not satisfy the system-level target by itself.

## Mapping Boundary

`fabric.system` does not contain software-to-hardware placement or
routing decisions. Mapping is a separate artifact specified in
`docs/spec-mapping-artifact.md`, and the PnR tool that produces it is
specified in `docs/spec-pnr.md`. The artifact references a software
dataflow graph, a selected `fabric.system`, and the hardware resources
chosen by PnR.

System nodes, links, domains, ports, and channels are hardware facts.
They remain valid whether no workload is mapped, one workload is mapped,
or multiple workloads are evaluated against the same architecture.

## ADG Builder Output Contract

The ergonomic C++ ADG Builder emits this fabric dialect target form.
Its full C++ API contract is specified in `docs/spec-adg-builder.md`.
Builder conveniences may construct common structures such as:

* heterogeneous accelerator arrays
* mesh-like router graphs
* tree interconnects
* crossbar-like fabrics
* cache hierarchies
* external memory attachments

Every convenience must lower to explicit nodes, ports, channels, and
one-to-one `fabric.link` operations. Regular-topology helpers may emit
optional visualization layouts and coordinates, but connectivity must be
emitted as links.

## Verifier Rules

The target verifier enforces:

* each `fabric.system` symbol is unique in its module;
* each `fabric.system` declares a valid `memory_model`;
* `memory_model = custom` carries a model name or non-empty params;
* node symbols are unique within the system;
* external port symbols are unique within the system;
* each external port owns exactly one complete `#fabric.port`;
* node kinds are valid enum values;
* each node satisfies the common node attribute contract;
* each baseline node kind satisfies its required attributes, required
  ports, and required params;
* port names are unique within each node or external port;
* channel names are unique within each port;
* built-in protocol and role imply the required channel set and channel
  directions;
* built-in ports declare only schema-defined required or conditional
  channels;
* required protocol params are present and satisfy the protocol schema;
* `stream` `ready` channels appear if and only if
  `params.flow_control = ready_valid`;
* `packet` `credit` channels appear if and only if
  `params.flow_control = credit`;
* paired `stream` endpoints use compatible flow-control modes;
* paired `packet` endpoints use compatible flow-control modes;
* `custom` protocol ports declare `params.protocol_name` and explicit
  transaction-flow classification;
* port memory capability is derived from protocol, role, and params;
* `custom` ports declare `params.memory_capable`;
* `addr_space` and `addr_ranges` appear only on memory-capable ports;
* port-level `clock_domain` appears only on `clock_converter` ports and
  references a declared clock domain;
* each `clock_converter` has exactly two transaction-flow sides with
  different explicit port-level clock domains;
* each `width_converter` declares unequal input and output widths;
* each `protocol_converter` has different source and destination
  protocols;
* a terminal memory target port is exactly a memory-capable subordinate
  port on a `memory` node;
* every terminal memory target port declares `addr_space` and non-empty
  `addr_ranges`;
* each link source endpoint exists and is an output channel;
* each link destination endpoint exists and is an input channel;
* absent link protocol is inferred from both endpoint ports;
* explicit link protocol matches both endpoint ports;
* links between different endpoint protocols are illegal unless an
  explicit protocol-converter node sits between them;
* each channel endpoint participates in at most one link;
* required non-optional internal channels are connected or exposed;
* external exposure declares a complete required channel set for the
  exposed protocol;
* external endpoints resolve to channels of a complete external protocol
  port;
* `acc_core` nodes reference visible `fabric.module` symbols;
* cache nodes satisfy required upstream, downstream, line-size, and
  capacity fields;
* nodes without clock-domain, reset-domain, or power-domain annotations
  belong to the corresponding system default domain;
* cross-domain links are explicit either through a link `crossing` field
  or through an explicit clock-converter node;
* same-domain links do not carry a `crossing` field;
* reset-domain and power-domain differences do not affect link legality;
* visualization layout names are unique when visualization metadata is
  present;
* visualization metadata references declared layouts and uses coordinate
  rank and extents consistent with the referenced layout;
* terminal memory target ranges in the same physical address space do
  not overlap by default;
* coherence-domain memberships reference memory-capable port endpoints;
* each coherence-domain member port uses the domain address space;
* coherence-domain `line_bytes` is positive;
* coherence-domain `members` is non-empty;
* coherence-domain `home` is legal only for directory-style protocols;
* a `protocol = none` coherence domain does not include cache node
  ports;
* a memory-capable port participates in at most one coherence domain;
* caches in a coherence domain use the domain line size;
* member port ranges, when present, belong to the domain address space.
