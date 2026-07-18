# Fabric Memory Operation Engine

## Scope

`fabric.mem` describes a Fabric-owned memory operation-engine hardware
capability. The accepted ABI contains:

* one operation engine;
* one or more manager/requester memory endpoints;
* zero or more subordinate/provider memory endpoints;
* `L` physical load operation ports;
* `S` physical store operation ports;
* an independent physical operation data width `W`;
* a Spatial or Temporal operation schedule;
* fixed request-source-to-manager dispatch eligibility.

The complete architecture permits an optional Local Memory Service, but that
subresource is not accepted by this ABI. Its canonical typed service contract
is not yet represented. Engine-plus-local and storage-only forms therefore
remain unsupported. An engine-only occurrence requires at least one manager
endpoint as its backing path.

This operation owns hardware capability only. It does not own workload memory
bindings, active dispatch, configured accesses, service target selection, or
bitstream state.

## Assembly

Anonymous form:

```text
fabric.mem [spatial|temporal]
  mgr(manager-operands)
  load(load-operands)
  store(store-operands)
  [{hardware-parameters}]
  : (input-types) -> (result-types)
```

The `load` and `store` clauses are omitted when their respective physical port
counts are zero. `mgr()` is syntactically variadic, although the implemented
engine-only form rejects an empty manager list.

Named templates place `@name` before the schedule and replace SSA operands and
results with a function type:

```text
fabric.mem @name [spatial|temporal]
  (input-types) -> (result-types)
  [{hardware-parameters}]
```

The schedule is surface shorthand for the operation engine's
`operation_schedule`. Because this ABI accepts only occurrences with an
operation engine, every accepted `fabric.mem` carries the shorthand.

## Signature

For `L = load_group_size` and `S = store_group_size`, inputs are ordered as:

1. every manager endpoint memref;
2. `L` load groups `(addr, ctrl)`;
3. `S` store groups `(addr, data, ctrl)`.

Results are ordered as:

1. every subordinate endpoint memref;
2. `L` load groups `(data, done)`;
3. `S` store `done` results.

Manager and subordinate counts are the leading signature lengths left after
subtracting the operation groups. They are not duplicated as attributes.

Every capability endpoint is a memref whose element type is
`!fabric.bits<width>`. Endpoint widths are independent of each other and of
`W`. Capability endpoints never carry a Dataflow tag.

Incoming operation operands may use
`source-type to destination-operation-port-type` when both types have the same
`bits` or `bits_tag` kind. Memref capability types must match exactly and
cannot use this normalization syntax.

## Hardware Parameters

`hw_params` is a length-one array containing a closed dictionary. Every
integer parameter is a signless `i32`.

A Spatial engine requires exactly:

```text
load_group_size     = L, where L >= 0
store_group_size    = S, where S >= 0
data_width          = W, where W > 0
dispatch_eligibility = H_dispatch
```

A Temporal engine additionally requires exactly:

```text
tag_width             = T, where T > 0
operation_table_size  = K, where K > 0
```

`L + S` must be greater than zero. Define:

```text
P = L + S
```

`P` is the number of concrete physical operation ports. `K` is the number of
resident Temporal configured rows. `K` is independent of `P` and is not
derived from `2^T`. In particular, `K != P` is legal.

A valid Temporal row is uniquely matched by
`(operation_kind, physical_port_sel, tag_match)`. A homogeneous engine has
`P * 2^T` representable physical match tuples, so it requires:

```text
K <= P * 2^T
```

The verifier checks this inequality without overflowing. It is a capacity
constraint, not an equality or a derivation of `K`. A tag may be reused on a
different physical port because those ports are distinct match domains.

`W` is an explicit hardware fact. It is never inferred from a manager endpoint
element width.

Unknown hardware keys are rejected. Spatial engines reject all Temporal-only
keys.

## Operation Ports

Spatial operation ports use:

```text
addr      : !fabric.bits<index_width>
data      : !fabric.bits<W>
ctrl/done : !fabric.bits<0>
```

Temporal operation ports replace each type with:

```text
addr      : !fabric.bits_tag<index_width, T>
data      : !fabric.bits_tag<W, T>
ctrl/done : !fabric.bits_tag<0, T>
```

Each load port consumes `(addr, ctrl)` and produces `(data, done)`. Each store
port consumes `(addr, data, ctrl)` and produces only `done`.

## Dispatch Eligibility

`dispatch_eligibility` is the manager-only projection of the Fabric-owned fixed
`H_dispatch` relation:

```text
dispatch_eligibility = {
  operation_port_requests = [
    [manager-endpoint-id, ...],  // physical operation port 0
    ...
  ],
  subordinate_requests = [
    [manager-endpoint-id, ...],  // subordinate endpoint 0
    ...
  ]
}
```

The complete architecture also permits a Local Memory Service target, but that
target is absent because this engine-only ABI does not represent the required
typed service contract. Every accepted request source therefore has a
nonempty domain of manager endpoint targets.

Request-source identities are closed and mechanical. Physical operation-port
sources are ordered:

```text
0 .. L-1       load ports
L .. L+S-1     store ports
```

Subordinate request sources use subordinate result order. The
`operation_port_requests` outer array has exactly `P` entries, and the
`subordinate_requests` outer array has exactly the signature-derived
subordinate endpoint count. Manager target identity is manager operand order
in `[0, M)`.

Every source domain is a nonempty array of signless `i32` manager identities in
strictly increasing order. Strict ordering gives one canonical encoding and
rejects duplicates. The dictionary is closed and accepts only the two source
domain arrays above.

This relation states which manager services each physical request source can
reach. It does not select a service for a workload. Active `C_dispatch`,
selected service targets, addresses, memory bindings, and configured rows
remain outside canonical Fabric hardware capability.

Spatial and Temporal engines carry the same fixed relation. Spatial operation
requests are already identified by physical port. A configured Temporal row
separately selects `operation_kind`, a compatible load/store
`physical_port_sel`, and `tag_match`; these are configured row fields, not
Fabric `H_dispatch`. The hardware ABI therefore has no per-slot
physical-port-eligibility array.

## Module Provenance

A `fabric.module` memref input is an imported manager/requester capability and
may feed one or more manager endpoints.

A module memref result must originate from a subordinate/provider result of an
anonymous `fabric.mem`, or from a memref result of `fabric.instantiate`.
Every signature-derived subordinate result of `fabric.mem` has provider
provenance. Export is not restricted to the first subordinate result.

A subordinate result may also feed a manager endpoint. Capability values are
not subject to the point-to-point transport-use restriction applied to
`bits` and `bits_tag` values.

## Rejected State

Canonical `fabric.mem` has a closed attribute set defined by its registered op
schema and this hardware ABI. It accepts no discardable attributes. Generic
operation syntax therefore cannot attach workload, local-service, dispatch
selection, or other unowned state. The custom printer preserves such invalid
attributes for diagnostics rather than silently dropping them.

The parser and verifier reject legacy workload configuration including:

* `addr_table`;
* `mem_enable`;
* `memory_operation_table`.

The ABI also has no local storage/service/refinement dictionary, configured
rows, service target selector, Memory Binding, Access Entry, persistent Memory
Realization, provider decode, response-route configuration, or bitstream
finalizer.

These exclusions preserve a single authority: Fabric describes fixed hardware
capability, while workload mapping choices remain outside the Fabric IR.

## Implementation

The operation definition is `Fabric_MemOp` in
`include/Fabric/IR/FabricOps.td`. Its custom parser, printer, and verifier are
implemented in `lib/Fabric/IR/FabricMemOp.cpp`. Module memory-export
provenance is verified in `lib/Fabric/IR/FabricModuleOp.cpp`.
