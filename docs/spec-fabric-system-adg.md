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

## Nodes

`fabric.node` represents one physical hardware instance. The node kind
is a fixed enum, not a free-form string. The baseline node kinds are:

| Kind | Meaning |
|------|---------|
| `host_core` | Host processor core. |
| `acc_core` | Accelerator core containing ScalarCore parameters and a SpatialCore template reference. |
| `cache` | Cache or cache-like coherent memory hierarchy node. |
| `memory` | Terminal storage service such as external memory, scratchpad, SRAM, or MMIO endpoint. |
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

### Cache

A `cache` node must define at least one upstream memory-capable port and
at least one downstream memory-capable port. It must define line size and
capacity. If the cache participates in a coherence domain, its line size
must match the domain line size.

Recommended cache params include associativity, set count, write
policy, allocate policy, hit latency, miss latency, and MSHR count.

### Memory

A `memory` node represents a terminal storage service when one of its
subordinate memory ports serves final storage. Examples include external
DRAM, scratchpad SRAM, local SRAM, and MMIO endpoints.

Caches, interconnect nodes, arbiters, routers, and DMA managers are not
terminal storage targets for address-range overlap checking.

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

The first target form stores port and channel declarations as structured
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
| `optional` | no | Whether the entire port may remain unconnected. Defaults to `false`. |
| `params` | no | Protocol- or node-specific dictionary. |

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
* `stream`
* `custom`

Built-in protocols define required channel sets and channel directions
for each role. `custom` requires an explicit protocol name, channel set,
channel directions, and any required params.

For AXI4-MM, the canonical channel set is `aw`, `w`, `b`, `ar`, and
`r`. A complete built-in port must declare every required channel unless
the protocol spec explicitly defines a smaller variant.

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

Link attributes include a small fixed set plus an open `params`
dictionary. The fixed set includes protocol identity, latency metadata,
bandwidth metadata, and explicit crossing metadata when needed.

## External Ports and Dangling Channels

`fabric.external_port` declares intentional system boundary exposure. It
distinguishes a real top-level hardware port from an accidentally
unconnected internal channel.

A `fabric.external_port` is a complete protocol port, not a single
channel exposure. It owns exactly one `#fabric.port` attribute. For a
built-in protocol, that port must declare the complete required channel
set for its role. Each channel is still connected individually through a
`fabric.link` endpoint, but the externally exposed interface is the full
protocol bundle.

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

If a link connects endpoints whose owning nodes are in different clock
domains, the crossing must be explicit. This can be represented by a
link crossing attribute or by an explicit `clock_converter` node. Silent
cross-domain connectivity is illegal.

## Address Spaces

Memory-capable ports may declare physical address ranges. Loom does not
model MMUs or virtual memory in the first system ADG target. Address
ranges are physical.

`#fabric.addr_range<base, size>` denotes the half-open unsigned 64-bit
range:

```text
[base, base + size)
```

`size` must be greater than zero.

For the same physical address space, terminal memory target port ranges
must not overlap by default. Upstream cache and interconnect ports may
cover aggregate downstream ranges and are not terminal-overlap targets.

## Coherence and Consistency

Every `fabric.system` must declare a system memory consistency model.
Domain-level or node-level overrides may be represented as metadata, but
they do not replace the required system-level declaration.

`fabric.coherence_domain` defines cache coherence over memory-capable
ports. Membership references node port endpoints, not whole nodes and
not individual channels.

The baseline coherence protocol enum is:

* `none`
* `snooping`
* `mesi`
* `moesi`
* `directory`
* `custom`

A directory or home-agent coherence protocol may define an optional
`home` endpoint. A memory-capable port may participate in at most one
coherence domain in the first target design.

If a cache is in a coherence domain, its line size must match the
domain line size.

## Mapping Boundary

`fabric.system` does not contain software-to-hardware placement or
routing decisions. Mapping is a separate artifact that references a
software dataflow graph, a selected `fabric.system`, and the hardware
resources chosen by PnR.

System nodes, links, domains, ports, and channels are hardware facts.
They remain valid whether no workload is mapped, one workload is mapped,
or multiple workloads are evaluated against the same architecture.

## ADG Builder Output Contract

The ergonomic C++ ADG Builder emits this fabric dialect target form.
Builder conveniences may construct common structures such as:

* heterogeneous accelerator arrays
* mesh-like router graphs
* tree interconnects
* crossbar-like fabrics
* cache hierarchies
* external memory attachments

Every convenience must lower to explicit nodes, ports, channels, and
one-to-one `fabric.link` operations. Coordinates may be emitted as
metadata, but connectivity must be emitted as links.

## Verifier Rules

The target verifier enforces:

* each `fabric.system` symbol is unique in its module;
* node symbols are unique within the system;
* external port symbols are unique within the system;
* each external port owns exactly one complete `#fabric.port`;
* node kinds are valid enum values;
* port names are unique within each node or external port;
* channel names are unique within each port;
* built-in protocol and role imply the required channel set and channel
  directions;
* each link source endpoint exists and is an output channel;
* each link destination endpoint exists and is an input channel;
* each channel endpoint participates in at most one link;
* required non-optional internal channels are connected or exposed;
* external exposure declares a complete required channel set for the
  exposed protocol;
* external endpoints resolve to channels of a complete external protocol
  port;
* `acc_core` nodes reference visible `fabric.module` symbols;
* cache nodes satisfy required upstream, downstream, line-size, and
  capacity fields;
* clock-domain crossings are explicit;
* terminal memory target ranges in the same physical address space do
  not overlap by default;
* coherence-domain memberships reference memory-capable ports;
* a memory-capable port participates in at most one coherence domain in
  the first target design;
* caches in a coherence domain use the domain line size.
