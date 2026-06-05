# Mapping Identity and References

This document specifies the identity, reference, and fingerprint model
shared by all Loom mapping artifacts. Other mapping specs define record
families that use these identifiers.

The mapping artifact is an independent relation between software
dataflow IR and hardware Fabric ADG. It must not mutate either input and
must not rely on line numbers, source offsets, printer ordering that is
not made stable, or local filesystem paths.

## Required Header

Every mapping artifact has one header record.

Required fields:

* `schema_version`: mapping artifact schema version.
* `artifact_id`: stable identifier for this candidate artifact.
* `software_root`: reference to the software module or dataflow root.
* `hardware_system`: reference to the selected `fabric.system`.
* `producer`: tool name and tool version that produced the artifact.

Optional fields:

* `software_fingerprint`: content fingerprint of the referenced
  software IR.
* `hardware_fingerprint`: content fingerprint of the referenced
  hardware IR.
* `workload_profile`: reference to workload shape, profile data, or
  concrete input class used during mapping.
* `mapping_set`: identifier of the mapping-set manifest that contains
  this artifact.
* `created_at`: timestamp for reporting only. It is never used for
  legality or deterministic ordering.

Fingerprints are optional in early flows. When present, every consumer
must validate them before using the artifact. A mismatch is a stale-input
diagnostic, not a reason to reinterpret references heuristically.

## Record Identity

Every non-header record has a required `record_id`. The `record_id` is
unique within the artifact and stable under deterministic PnR reruns with
the same inputs and options.

The canonical record-id shape is:

```
<family>/<stable-symbolic-key>
```

Examples:

* `thread_binding/main_thread/i0`
* `graph_binding/gemm_body/launch0`
* `op_binding/graph0/add_17`
* `route/graph0/add_17_to_mul_19`

Record IDs are artifact-local. External tools must not infer software or
hardware meaning from a record ID alone; they must read the referenced
objects in the record body.

## Software References

Software references are symbolic references into dataflow IR.

Required fields:

* `kind`: one of `thread_def`, `thread_launch`, `thread_instance_domain`,
  `graph_def`, `graph_launch`, `subgraph`, `operation`, `ssa_value`,
  `edge`, `memref_region`, `partitioned_region`, `control_token_edge`,
  `done_token_edge`, or `memory_order_edge`.
* `symbol`: nearest stable symbol that owns the referenced object.

Optional fields:

* `op_path`: stable operation path under the owner symbol.
* `result_index`: SSA result index.
* `operand_index`: operand index.
* `instance`: logical instance descriptor for parametric thread or graph
  instances.
* `edge_role`: value, control, done, or memory-order role.
* `fingerprint`: object-level fingerprint when available.

Line numbers and source-file offsets are forbidden. If an operation has
no stable path, the compiler must assign a stable mapping anchor before
PnR emits a reference to it.

## Hardware References

Hardware references are symbolic references into Fabric ADG or into a
referenced `fabric.module` template.

Required fields:

* `kind`: one of `system`, `node`, `external_port`, `node_port`,
  `channel_endpoint`, `link`, `module`, `module_resource`, `pe`, `fu`,
  `mem`, `switch`, `boundary`, `fifo`, `adapter`, `domain`, or
  `address_range`.
* `symbol`: nearest stable Fabric symbol that owns the referenced object.

Optional fields:

* `node`: system node symbol for node-local references.
* `port`: port symbol or ordinal under a node or resource.
* `channel`: protocol channel name for compound protocol ports.
* `endpoint_direction`: `source` or `sink` for directed channel
  endpoints.
* `resource_path`: stable resource path under a `fabric.module` symbol.
* `instance`: physical instance descriptor when a template is
  instantiated multiple times.
* `fingerprint`: object-level fingerprint when available.

Compound protocol ports are references to bundles only when the record
explicitly says it is referring to a bundle for reporting or
visualization. Legality records that drive mapping behavior reference
directed channel endpoints.

## Mapping References

Mapping references point to other records in the same artifact.

Required fields:

* `record_id`: target record ID.

Optional fields:

* `role`: producer, consumer, buffer, route_segment, schedule_context,
  memory_context, or diagnostic subject.

Mapping references are allowed only within one artifact. A mapping-set
manifest may reference many artifacts, but per-candidate records must
not point into sibling artifacts.

## Workload Shape References

When a mapping is shape-dependent, the artifact may reference a workload
shape record.

Required fields for a workload shape record:

* `shape_id`: stable shape identifier.
* `parameters`: symbolic parameter map used by PnR, such as tensor
  extents, loop trip counts, or partition sizes.

Optional fields:

* `profile_ref`: reference to profile evidence used by the cost model.
* `input_class`: human-readable label for reports.

Workload shape affects mapping choice but not software semantics. A
consumer must reject a shape-specific artifact if asked to use it for an
incompatible shape.

## Deterministic Ordering

Artifacts use deterministic ordering:

* header first;
* record families in the order listed by `docs/spec-mapping-artifact.md`;
* records within a family by `record_id`;
* dictionary keys lexically;
* arrays by semantic order when one exists, otherwise by stable
  symbolic key.

Deterministic ordering is a serialization rule. It must not be used as a
hidden source of legality.

## Validation

The identity verifier checks:

* every required header field is present;
* every `record_id` is unique;
* every software reference resolves;
* every hardware reference resolves;
* every mapping reference resolves within the same artifact;
* fingerprints match when present;
* no reference uses line numbers, byte offsets, host-local paths, or
  unresolved printer-order assumptions.

## Acceptance Criteria

The identity model is complete when:

* every detailed mapping record can use the shared reference model;
* stale software or hardware inputs are diagnosed before simulation,
  runtime, RTL lowering, or FPA estimation consumes the artifact;
* deterministic reruns produce stable record IDs and stable ordering;
* compound protocol ports can be displayed as bundles while legality
  records still reference directed channel endpoints.
