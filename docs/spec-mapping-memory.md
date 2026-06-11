# Mapping Memory Records

This document specifies mapping records for memory regions, dataflow
load/store operations, partitioned-data regions, coherence, consistency,
and physical address binding.

Loom requires cache coherence and a defined memory consistency model.
Fabric ADG and mapping memory records do not include MMU behavior or
virtual memory.

## Memory Region Binding

A memory-region-binding record maps a software memory region to a
hardware address space and memory-capable endpoint.

Required fields:

* `record_id`.
* `software_region`: software reference to a memref region,
  partitioned-data region, or named memory object.
* `address_space`: hardware reference to a physical address space or
  terminal address range declared by Fabric ADG.
* `home_endpoint`: hardware reference to a memory-capable channel
  endpoint.
* `range`: base and size as constants or symbolic expressions.

Optional fields:

* `partition`: symbolic partition descriptor for sharded or tiled data.
* `alignment`.
* `bank`.
* `coherence_domain`.
* `consistency_domain`.
* `initialization`.
* `metrics`.

The range is physical or implementation-defined within the selected
Fabric address space. The mapping artifact must not require virtual
address translation.

## Memory Operation Binding

A memory-operation-binding record maps a dataflow memory op to a
memory-capable hardware path.

Required fields:

* `record_id`.
* `operation_ref`: software reference to `dataflow.load`,
  `dataflow.store`, or another memory-effecting dataflow operation.
* `region_binding`: mapping reference to a memory-region-binding
  record.
* `request_route`: mapping reference to a route carrying the request.
* `response_route`: mapping reference to a route carrying the response,
  required for loads and optional for stores when the protocol has no
  store response.
* `ordering`: mapping reference to a memory-order record or explicit
  `unordered` marker when legal.

Optional fields:

* `access_width`.
* `burst`.
* `atomic_semantics`.
* `cache_policy`.
* `coalescing_group`.
* `metrics`.

Memory operation binding must respect the operation's dataflow control
and memory-order token dependencies. Missing ordering evidence is a
verifier error unless the software op is explicitly unordered.

## Coherence Domain

A coherence-domain record binds software memory regions and hardware
cache or memory endpoints into one coherence domain.

Required fields:

* `record_id`.
* `domain_ref`: hardware reference to a coherence domain declared by
  Fabric ADG.
* `members`: non-empty list of memory-region bindings or hardware
  cache/memory endpoints.
* `protocol`: coherence protocol name declared or accepted by the
  hardware target.

Optional fields:

* `scope`: system, cluster, AccCore group, or custom named scope.
* `home_policy`.
* `invalidation_policy`.
* `metrics`.

If a memory region is shared by multiple AccCores and is not marked
private by software semantics, it must belong to a coherence-domain
record.

## Consistency Domain

A consistency-domain record names the memory consistency model used by
the mapped workload.

Required fields:

* `record_id`.
* `model`: consistency model name, such as `sequential`,
  `release_acquire`, `relaxed_with_explicit_fences`, or `custom`.
* `scope`: software or hardware scope where the model applies.

Optional fields:

* `fence_records`: mapping references to memory-order records that
  implement explicit fences.
* `custom_payload`: required when `model` is `custom`.

Consistency is a mapping/runtime contract. Fabric ADG declares what the
hardware can support; the mapping artifact selects the model for a
workload and records the evidence needed by simulation and runtime.

## Memory Order Record

A memory-order record binds software memory-order edges to hardware
ordering mechanisms.

Required fields:

* `record_id`.
* `edge_refs`: non-empty list of software memory-order edge references.
* `mechanism`: `same_endpoint_order`, `fence`, `barrier`,
  `protocol_order`, `schedule_order`, or `custom`.

Optional fields:

* `hardware_ref`: endpoint, fence resource, barrier resource, or
  protocol object used to enforce ordering.
* `schedule_context`.
* `metrics`.

Ordering may be proven by schedule records only when all relevant
memory operations are scheduled in the same schedule context and the
hardware resource observes that order.

## Validation

The memory verifier checks:

* every software memory region has a legal address-space binding when
  it is accessed by mapped memory ops;
* every memory operation binds to a memory-capable endpoint;
* request and response routes are present and directionally legal;
* address ranges are inside terminal memory target ranges;
* shared memory belongs to a coherence domain unless software semantics
  prove it private;
* the selected consistency model is supported by the hardware and
  runtime target;
* memory-order edges are enforced by explicit mechanisms;
* no record requires MMU or virtual-memory behavior.

## Acceptance Criteria

Memory records are complete when:

* dataflow loads and stores can map to Fabric memory endpoints through
  explicit request and response routes;
* cache coherence and consistency are represented without introducing
  virtual memory;
* CGRA-sim can model memory traffic, ordering stalls, cache/coherence
  effects, and bandwidth pressure from the artifact;
* runtime lowering can allocate or bind physical address ranges using
  the same records.
