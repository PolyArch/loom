# Fabric Reconfigurable Ops

This document specifies how Fabric owns typed software capability and how an
FU exposes normalized valid semantic encodings.

## Semantic Ownership

An FU has three distinct kinds of facts:

* FU topology owns physical resources, SSA wiring, fanout, and boundary ports.
* Each `fabric.op` owns the complete typed software modes supported by that
  physical datapath.
* The FU owns the normalized set of valid cross-resource encodings.

The Hardware Sharing Group registry is the global authority for which software
operation families may share one physical datapath. `fabric.op.op_list` is the
HSG-validated operation-family envelope selected by a concrete datapath.
`fabric.op.hw_params` owns the complete typed modes within that envelope. In a
canonical FU, the distinct `op_list` entries must equal the operation
identities present in the normalized mode set.

`fabric.op.hw_params` is the canonical typed capability set implemented
by the concrete datapath. Every mode contains:

```text
{ op, function_type, input_ports, output_ports, attributes }
```

`op` is the exact software operation identity. `function_type` and
`attributes` are the complete semantic type and attribute selection for that
mode. `input_ports` and `output_ports` map ordered software operands and
results to physical `fabric.op` ports.

### Wide Sync Lane Inventory

A canonical wide `dataflow.sync` is one full-width physical capability. Its
Fabric descriptor owns one complete `paired_lanes` inventory. Each lane record
contains exactly:

```text
{ input_port, output_port, mask_bit }
```

The inventory covers the complete physical input/output signature exactly
once. Input endpoints, output endpoints, and mask positions are independently
unique and in range; input port, output port, and mask position are not assumed
to have the same index. Mask positions form a dense `0..N-1` encoding for the
`N` declared lanes. Inventory order is the complete configured software lane
order; it is capability structure, not an active subset.

The wide-sync typed mode describes the complete `N`-lane software capability.
It does not select an active subset, and valid semantic encodings or configured
function identity do not vary with a workload's lane choice. Ordered software
subset correspondence belongs only to the Mapping Artifact.

These fields are Fabric-owned capability facts. They do not compete with the
Hardware Sharing Group registry: the registry admits the operation family,
while the mode selects a concrete typed and attributed member of that family.
The mode must form a valid instance of the selected registered software
operation. The fields are stored once on the physical `fabric.op`, not
repeated in each FU encoding.

For canonical typed modes, operation-family schema metadata must not impose a
second function type or require equal physical widths on ports tied to one
software type. The complete mode is the type authority; each mapped physical
port is legal when its payload capacity is at least the corresponding software
requirement. Historical exact-width schema checks apply only to the legacy
programmed adapter form that has no typed mode tuple.

### Stream Configuration

Every normalized `@dataflow.stream` mode carries one typed `step_kind` and
one typed `predicate`. All stream modes on the same physical `fabric.op` must
share the same `step_kind`, because they describe one fixed recurrence-update
datapath. Modes may differ in `predicate`; a valid encoding selects the exact
predicate mode used by the configured function.

`step_kind` is never workload-selected `sw_configs`. The legacy programmed
adapter expresses the same hardware fact as exactly one `step_kind` plus a
non-empty set of supported predicates, and may select only one predicate.
Different stream step kinds require different physical datapaths and cannot be
unioned into one stream resource by the synthesizer.

## Valid Semantic Encodings

`fabric.fu.valid_encodings` is an array of complete legal configurations. An
encoding contains the ordered FU output ports exposed as software results and
a resource assignment list:

```text
{ outputs = [...], resources = [
    { resource = R, mode = M },
    { resource = R, select = S }
] }
```

`mode` selects one `hw_params` entry on a `fabric.op`. `select` chooses
the live arm of a routing resource. Resource assignments are ordered by
resource index.

An encoding contains exactly the assignments that affect its configured
function. It must not contain selections for dead resources, don't-care
fields, or workload-selected `sw_configs`. Cross-resource correlation is
represented by the complete encoding tuple, never by independent per-field
allowed sets.

Every declared `hw_params` mode must be selected by at least one valid
encoding. Every selected resource must be live and coherent under the FU SSA
topology. An active value may fan out to multiple active consumers, but a value
must not implicitly feed active and inactive mutually exclusive datapaths.
That case requires explicit routing resources.

## Configured Function Projection

The canonical projector has this semantic interface:

```text
projectConfiguredFunction(fu, encoding) -> ConfiguredFunction
```

The result is a typed and attributed software function graph containing:

* exact software operation identities;
* exact function types and semantic attributes;
* ordered operand edges, result indices, and fanout;
* software-operand and result order; and
* exact FU boundary input and output port correspondence.

Muxes and demuxes are not software actors. The projector follows their
selected routes. A `fabric.op` becomes one software node selected by its mode.
Configuration that is not live in the projection is invalid rather than a
don't-care.

Two distinct valid encodings must not project to isomorphic configured
functions when FU boundary identity is preserved. Such an FU is rejected by
the Fabric verifier. Consumers must not repair it with post-hoc deduplication
or first-assignment canonicalization.

## Port Kinds And Widths

Capability legality first requires the same port kind. `bits` and `bits_tag`
are distinct kinds and must not be exchanged implicitly.

For an untagged `bits<W>` physical path carrying a software value that requires
`N` payload bits, every node and boundary segment on the selected path must
satisfy `W >= N`. Exact width equality is not required. For example, an `i16`
mode may use `bits<32>` operation ports and a `bits<64>` FU boundary.

Payloads are low-bit aligned. Moving from a narrower physical segment to a
wider segment zero-fills high bits. Moving from a wider segment to a narrower
segment truncates high bits. Such a narrowing remains legal only when the
destination width is still at least the software semantic width. No selected
path segment may be narrower than the software requirement.

The software type remains exact in the `hw_params` mode and configured
function. Physical payload width does not become a second software type.

## Routing Mutually Exclusive Datapaths

When distinct physical datapaths share FU inputs, each shared input requires
an explicit demux or equivalent route selector, and their results require an
explicit mux when they share an FU output. One valid encoding selects all
matching input and output routes together with the active operation mode.

An output mux alone is insufficient because it leaves inactive datapaths
implicitly consuming broadcast inputs. That topology does not materialize the
selected software function.

## Mapping Selection And Legacy Input

Canonical `fabric.op.hw_params` is an array of complete normalized mode tuples.
A `fabric.fu` carrying `valid_encodings` must not persist selected
`sw_configs` on its internal resources or selected routing on muxes and
demuxes. Mapping selects one FU encoding. A backend may derive transient
`sw_configs = {mode = N}` and route selections from that encoding, but those
values are workload choices and must not be written back into canonical
Fabric.

A deliberately programmed normalized adapter may omit `valid_encodings` only
when every inner `fabric.op` carries normalized modes plus one explicit
`sw_configs.mode` selection and every `fabric.mux` / `fabric.demux` carries a
complete route selection. Partial or mixed programmed state is invalid. This
form is non-canonical boundary input; it does not become a capability source or
a substitute for normalized valid semantic encodings.

Raw `sw_configs.bitmask` remains accepted only on the legacy programmed
`dataflow.sync` adapter so that already-programmed Fabric can be verified. A
canonical capability or Mapping input never persists that selected mask;
later lowering derives it mechanically from the Fabric lane inventory and the
Mapping-owned `ActorToFabricOp` correspondence.

The historical length-one field-wise `hw_params` dictionary remains accepted
only for non-canonical programmed Fabric input. It is a boundary adapter and
must not enter the projector, synthesis, coverage, or canonical FU verifier.

`dataflow.subgraph` is likewise only a legacy input or display adapter. The
projector, verifier, coverage check, and synthesis acceptance gate operate on
`ConfiguredFunction` and explicit valid encodings.

## Extending Capability

Adding a software operation to Fabric requires all of the following:

* the operation is admitted by the Fabric operation schema;
* any multi-operation physical sharing is admitted by the Hardware Sharing
  Group registry;
* each physical `fabric.op` declares complete typed `hw_params` modes; and
* FU encodings select complete legal resource tuples whose projections are
  pairwise distinct under boundary-preserving identity.

Changing only `op_list`, an allowed-set field, or an enumerator table does not
add canonical capability.
