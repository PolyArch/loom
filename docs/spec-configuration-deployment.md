# Configuration And Deployment

This document is the semantic owner for Loom hardware-configuration encoding,
configuration images, the complete Deployment root, its derived runtime-image
children, and the first public package projection. Executable leaf schemas are
owned by `docs/spec-executable-closure.md`; runtime platform bindings are owned
by `docs/spec-runtime-abi.md`. This document composes exact references to those
owners without copying their facts. It does not create a new MLIR dialect or a
generic artifact-schema framework.

## Ownership

The configuration path has one owner for each fact:

```text
Fabric
  -> configurable fields, typed value domains, and field semantics

complete Mapping
  -> selected semantic configuration for exact physical resources

ConfigurationABI
  -> programming units, physical encoding, and load/activation contract

HardwareImplementation
  -> circuit and protocol implementation of one exact ConfigurationABI

HardwareConfigurationImage
  -> immutable encoded state for one exact programming unit

Deployment
  -> configuration-image closure and immutable derived runtime-image children
```

Fabric and Mapping may expose or select semantic `sw_configs`; they do not own
physical bit positions. `ConfigurationABI` owns those positions once. A
physical encoder, RTL decoder, runtime loader, or implementation backend must
consume the same ABI rather than duplicating it. Semantic DFG and CGRA
simulation does not decode this physical programming representation.

The Artifact DAG is acyclic. A `ConfigurationABI` references its exact Fabric.
A `HardwareImplementation` references the exact Fabric and ConfigurationABI it
implements. The ABI does not refer back to a final implementation.

An implementation-only backend recipe is selected by the exact hardware
candidate-generator binding per `FabricPhysicalOccurrenceOwnerRef`.
`InvocationManifest` records that selection on the derivation edge. The recipe affects
HardwareImplementation identity only through its materialized payloads,
platform reference, or implementation bindings, and is legal only when every
Fabric-observable semantic, timing, capacity, progress, and ABI fact is
unchanged. It does not create another configuration field, alter a
`HardwareConfigurationImage`, or change Fabric identity.

SystemMapping binds the architecture-only Fabric and exact Transport
Architecture. Replacing one Interconnect or hardware implementation with
another verified implementation of the same architecture and
`ConfigurationABI` does not change SystemMapping or semantic configuration
selection. The selected implementation changes Deployment closure, but the
same semantic configuration encodes to the same
`HardwareConfigurationImage` identity under that unchanged ABI.

## Artifact Families

The complete Artifact families have these fixed schema descriptors:

```text
loom.configuration_abi             3.0
loom.hardware_configuration_image  3.0
loom.deployment                    3.0
```

The three major versions change because the ABI now rejects any image that
omits a configurable Fabric owner and the exact implementation child is
`loom.hardware_implementation 3.0`. Their root shapes otherwise retain the
contracts below; an old validator cannot reinterpret the new references under
a 2.x descriptor.

The frontend relocatable accelerator payload is an input to final linking, not
a Deployment child. `CompilerTargetBinding`, `InstructionCoreBinary`, host
program, registration, and static logical-memory leaves are closed by
`docs/spec-executable-closure.md`. Deployment is the executable closure; there
is no parallel `Executable` Artifact.

Each closed owner library defines one typed C++ model and one canonical
serializer and parser. `ConfigurationABI` uses canonical JSON semantic bytes.
`HardwareConfigurationImage` uses a canonical typed header followed by its raw
payload. Deployment uses canonical JSON. Loom does not define MLIR mirrors,
Protobuf alternatives, base64
payloads, or a generic manifest registry for these families.

Schema versions use `X.Y`: `X` is an incompatible change and `Y` is a
compatible schema extension. The family declaration is the only owner of its
schema identity, version, fields, defaults, and canonicalization.

## ConfigurationABI

`ConfigurationABI` has one canonical root:

```text
ConfigurationABI {
  version
  fabric_ref
  programming_units[] {
    unit_id
    exact_fabric_resource_closure[]
    programming_model
    fields[] {
      fabric_config_slot_ref
      semantic_encoding
      destination_slices[]
      inactive_value
    }
  }
}
```

In `loom.configuration_abi 3.0`, `fabric_ref` is an exact
`loom.fabric 4.0` System root. A complete implementation cannot bind an
uninstantiated Module root. The physical references used by the two nested
fields above are closed unions:

```text
FabricPhysicalOccurrenceOwnerRef
FabricPhysicalConfigurationFieldRef
FabricPhysicalConfigurationSlotRef
```

`exact_fabric_resource_closure` contains
`FabricPhysicalOccurrenceOwnerRef`; `fabric_config_slot_ref` contains
`FabricPhysicalConfigurationSlotRef`. Its field is the exact occurrence-
qualified `FabricPhysicalConfigurationFieldRef`; its residency is the closed
`Static | InstructionContext(InstructionContextRef)` union. Their closed
variants, canonical bytes, and validators are owned by
`docs/spec-fabric-identity.md`. A bare imported Module-local owner or
configuration-field reference is invalid because it would alias two physical
SpatialCore occurrences. These unions only qualify existing Fabric facts and
do not copy topology, capability, selected values, or configuration domains.

Fabric alone derives residency. A field owned by a Spatial PE, switch, FIFO,
boundary, memory occurrence, Temporal PE instruction table, or FU/operation
under a Spatial PE is `Static`. A FU or operation field under a Temporal PE is
`InstructionContext` for every resident context exactly when that PE uses
`per_instruction_fu_config`; it is `Static` when the PE uses `per_fu_config`.
No caller may attach an arbitrary context to a static field, omit a required
resident slot, or use a context owned by another PE. This one distinction is
enough to represent both shared physical FU configuration and context-banked
configuration without a sentinel context or duplicate field identity.

A SpatialMapping's `ConfiguredHardwareProjection` contains Module-local
`FabricConfigurationSlotRef` values because SpatialMapping binds a Module
root, not a System occurrence. SystemMapping resolves each imported projection
through its exact execution binding and mechanically qualifies the complete
local slot with the selected `SpatialCoreOccurrenceRef`. Configuration-image
finalization consumes only those resulting
`FabricPhysicalConfigurationSlotRef` values. It cannot attach an occurrence to
a field without the exact imported SpatialMapping binding or renumber a local
instruction context.

The ABI describes how every Fabric-owned semantic field is represented and
installed. It does not select a field value, reference a Mapping, or constitute
a configured hardware state. A selected value exists only in the transient
`ConfiguredHardwareProjection` derived from one complete Mapping. A physical
payload exists only when `HardwareConfigurationImage` binds that projection to
this ABI. Consumers must not use the ABI as a substitute Mapping input or infer
a selected mode from `inactive_value` or codebook order.

A programming unit is the smallest physical unit for which a complete
configuration state can be independently installed and activated, such as one
SpatialCore context bank. It is not a Mapping entity, a runtime handle, or a
temporal-PE instruction slot.

The only persistent reference to a programming unit is:

```text
ProgrammingUnitRef =
  ArtifactReference<ConfigurationABI> + ABI-local unit_id
```

Its canonical bytes are the Common exact ArtifactReference encoding followed
by the `u64be` ABI-local `unit_id`. Decoding requires the referenced Artifact
to be `loom.configuration_abi 3.0` and the unit ID to exist in its canonical
programming-unit catalog.

There is no global programming-unit registry or compatibility label. Each unit
owns an exact Fabric occurrence closure. The first programming model installs
a complete image and defines when that image becomes visible and active.
AXI, JTAG, MMIO, and custom transport mechanisms belong to the implementation
that realizes the ABI; the ABI does not contain a command-script language.

### Complete-Image Programming Model

`loom.configuration_abi 3.0` has one closed programming model:

```text
CompleteImageAtomic {
  payload_bit_count
}
```

An implementation accepts all `payload_bit_count` bits into non-observable
staging state and makes the complete new state visible atomically only after
the complete image is accepted. A partial write never changes active
configuration. Transport framing, addresses, write beats, and the mechanism
that commits staging state are HardwareImplementation details derived from
this contract; they are not additional ABI fields.

One implementation transport may serve several Programming Units. Its address
layout is a removable mechanical projection, not another Artifact or an ABI
field:

```text
ConfigurationTransportLayout = derive(
  exact ConfigurationABI,
  exact SpatialCoreOccurrenceRef,
  exact implementation transport profile)
```

The projection selects exactly the Programming Units whose complete Fabric
resource closure belongs to the selected SpatialCore occurrence, rebases their
occurrence-qualified closures and fields to the imported Module definition,
orders equal definitions identically, and assigns transport-local windows. A
unit spanning more than one SpatialCore occurrence is outside a local
SpatialCore transport profile and must receive a different exact
HardwareImplementation. The projection retains the exact `ProgrammingUnitRef`
for image and runtime binding, but occurrence identity cannot perturb the
definition-local window shape. RTL generation, HardwareImplementation
interface publication, and the runtime provider must call the same derivation;
none may maintain an independent address table.

The common portable transport profile and its exact word, window, staging,
commit, status, and readback behavior are owned by
`docs/spec-rtl-lowering.md`. Another implementation may select another typed
provider transport while preserving `CompleteImageAtomic`. Such a transport
choice changes HardwareImplementation identity, not ConfigurationABI or a
HardwareConfigurationImage.

All ABI bit vectors use one fixed representation. Logical bit `i` is bit
`i % 8` of byte `i / 8`, where bit zero is the least-significant bit of the
byte. The byte vector has exactly `ceil(bit_count / 8)` bytes and unused high
bits of its last byte are zero. There is no host-endian or tool-endian variant.

Programming units are sorted by the canonical byte sequence of their exact
Fabric resource closure and receive dense `unit_id` values starting at zero.
Resource closures are nonempty, canonical sets and do not overlap. Every
configuration field belongs to exactly one unit, and its Fabric configuration
owner must occur in that unit's resource closure. Authoring order never enters
identity.

### Field Encoding

Each referenced Fabric configuration field remains one opaque canonical
semantic carrier during ABI finalization. The ABI never splits a composite
Fabric field into independently authoritative leaves. A field uses one of two
encoding atoms:

* `DirectBits`: encode one fixed-width bit vector directly.
* `FiniteCodebook`: map each allowed typed value to one unique physical code.

The closed wire shapes are:

```text
DirectBits {
  encoded_bit_count
}

FiniteCodebook {
  encoded_bit_count
  entries[] {
    semantic_value
    physical_code
  }
}

DestinationSlice {
  source_bit_offset
  destination_bit_offset
  bit_count
}
```

`semantic_value` is the exact canonical byte carrier produced by the referenced
Fabric field-domain codec. The ConfigurationABI never parses it into a local
enum, operation string, or property map. A DirectBits semantic value uses the
fixed ABI bit-vector representation. A FiniteCodebook entry's physical code
uses that same representation with `encoded_bit_count`; semantic values and
physical codes are each unique. `inactive_value` uses the same semantic carrier
and must be encodable by the selected atom.

Every field validates against one exact sealed
`FabricSemanticFieldRelation` owned by its Fabric resource:

* `None` admits no ABI field;
* `Finite` requires the set of `semantic_value` entries to equal the Fabric
  relation's canonical value domain exactly, with no missing or extra value,
  and requires every value to round-trip through that relation's codec;
* `Direct` requires `DirectBits`, exact equality of `encoded_bit_count`, the
  same canonical fixed-width carrier, and acceptance by the Fabric-owned
  direct-domain validator; and
* no other pairing is legal.

`FabricOpSemanticFieldRelation` and the Spatial PE selector schema are typed
projections of this shared relation; they do not remain alternate ABI
contracts. Operation fields retain their existing `None | Finite | Direct`
behavior relation. Spatial PE activation and selector fields are `Finite`, and
the activation field additionally requires canonical `Disabled` as
`inactive_value`.

### Complete Fabric Field Inventory

The static configuration-field inventory is derived from canonical Fabric and
is independent of a Mapping. A complete ABI contains every field in this
inventory exactly once:

* a Spatial PE owns its existing factorized activation and per-FU boundary
  selector fields;
* a Temporal PE owns one joint instruction-table field;
* each FU occurrence owns one joint configured-graph field;
* each switch, FIFO, boundary, and memory occurrence owns one joint field;
* each configurable `fabric.op` occurrence node owns its existing operation
  field, projected into the residency derived above; and
* fixed point connections and owners whose specification declares no
  configuration own no field.

Every joint field is ordinal zero for its owner. Operation fields and Spatial
PE selectors remain separate because their semantic owners already prove those
choices independent. A component's correlated route table, resident rows,
tags, mode, capability template, provider decode, or internal graph must not
be split into fields whose independent ABI values could form an invalid
Cartesian product.

The component specification owns the exact relation:

* an FU field is the finite domain `Disabled | Active(capability_template)`;
* a FIFO field is the finite domain `Disabled | Buffered`, with `Bypass` added
  exactly when that occurrence is bypassable;
* a Spatial switch uses one direct carrier containing its complete selected
  crosspoint relation; each output selects at most one physically admitted
  input and all-zero means `Disabled`;
* a Temporal switch uses one direct carrier containing all bounded resident
  entries, including entry-valid, tag, and complete route selection; unused
  rows are zero, tags of valid rows are unique, and all-zero means `Disabled`;
* a boundary uses a finite `Disabled | Active` relation when its active shape
  has no payload and otherwise one direct carrier with explicit active or
  row-valid bits so tag zero never aliases `Disabled`;
* a Temporal PE and memory occurrence each use one direct carrier for their
  complete owner-local bounded record defined by their component
  specification; nested FU and operation values remain in their own slots,
  inactive or unused rows have exactly one zero representation, and the
  relation validator enforces every owner-local cross-field invariant; and
* a Spatial PE and operation use their existing exact codecs through the same
  shared relation interface.

Direct semantic carriers use the ABI bit-vector convention, but their field
order, width, canonical inactive form, and validity predicate are Fabric facts.
The ABI may scatter or recode only as allowed by `DirectBits` or
`FiniteCodebook`; this semantic carrier is not a configuration-memory address
layout. A backend cannot infer a component mode from field ordinal or replace
the Fabric validator with a private route or instruction codec.

The ABI cannot merge fields, copy FU operation fields into a PE or FU codebook,
omit an owner because one Mapping leaves it inactive, or assign a field to a
Mapping-selected dynamic schema.

The ABI owns physical codes, destination slices, padding, and inactive bits.
Fabric owns semantic field need, the one joint behavior domain, its projector,
and semantic codec. Mapping owns the authoritative actor and refinement
selections; `ConfiguredHardwareProjection` carries the value mechanically
derived by the Fabric projector and is not another selection authority. Entry
order or code assignment cannot become a backend behavior key.

`destination_slices` may cross words or be non-contiguous. Every destination
bit has exactly one source, slices are in range and non-overlapping, and all
reserved or padding bits are zero. An unselected hardware field uses the
ABI-declared `inactive_value`, which must be an encodable member of the exact
Fabric relation domain. The disabled resource or topology contract proves that
the encoded active-domain value is unobservable; the value need not denote an
inert active behavior. The encoder may not invent a default. RTL providers
consume these fields and their codebooks; they cannot create an independent
exact-mode index or decoder authority.

The slices of one field cover every source bit exactly once. Fields and slices
are sorted by canonical Fabric reference bytes and destination position.
Encoding starts from an all-zero payload, substitutes `inactive_value` for an
omitted semantic field, and then applies the slices. Decoding first rejects a
nonzero reserved bit, reconstructs every encoded field, rejects a
FiniteCodebook pattern with no entry, and rejects a DirectBits value outside
the Fabric-owned direct domain. Re-encoding a decoded complete image must
return identical bytes.

## HardwareConfigurationImage

One image has this canonical header model:

```text
HardwareConfigurationImage {
  version
  configuration_abi_ref
  programming_unit_id
  source_mapping : SpatialMappingRef | SystemMappingRef
  payload_bit_count
  payload
}
```

TechMapping is not a physical configuration and cannot source an image. The
image does not duplicate Fabric, backend, or semantic `sw_configs`. Its source
Mapping and ABI references recover those facts exactly.

Canonical semantic bytes are:

```text
u32be(canonical_header_size)
canonical_header_json
u64be(payload_size)
payload_bytes
```

The JSON header uses canonical enum and reference spelling, canonical integers,
and stable field order. Payload bytes use the bit and byte order declared by
the ABI. Bits beyond `payload_bit_count` are zero and no trailing bytes are
allowed. Common ArtifactIdentity SHA-256 v1 covers the entire framed value.
The raw payload is not a second Artifact family.

Image finalization consumes the Mapping verifier's validated temporary
`ConfiguredHardwareProjection` specified by
[Fabric Reconfigurable Operations](spec-fabric-reconfigurable-op.md), encodes
it through the ABI, and verifies the resulting bytes. It does not reconstruct
actor or refinement selections, invoke a second semantic projector, or accept
simulator-produced values. That projection is derived data and is never
persisted as another authority. Images are bound to exact hardware occurrences;
equal bytes may share blob storage but do not permit implicit rebinding.

## Deployment

Deployment is one selected, executable system closure:

```text
Deployment {
  version
  system_mapping_ref
  host_program
  instruction_core_binary_refs[]
  hardware_bindings[] {
    hardware_implementation_ref
    runtime_platform_binding_ref
  }
  configuration_image_refs[]
  static_memory_images[]
  thread_dispatch_image
  spatial_launch_image?
  admission_image
}

DeploymentProgramEntryRef =
  (exact Deployment ArtifactIdentity, program_entry_ordinal)

DeploymentExternalInterfaceRef =
  (exact Deployment ArtifactIdentity, external_interface_ordinal)
```

The exact Canonical Dataflow Program and architecture-only Fabric are recovered
from `system_mapping_ref`. Exact Fabric, ConfigurationABI, Interconnect
Implementation, ImplementationPlatform target and corner facts, and
provider-owned external bindings are recovered from each selected
HardwareImplementation. Deployment does not duplicate those
references. Its verifier requires the recovered closures to agree exactly and
requires `hardware_bindings[]` to cover the complete selected system without a
foreign or unused implementation.

`host_program`, `instruction_core_binary_refs[]`, and
`static_memory_images[]` use the closed types in
`docs/spec-executable-closure.md`. Each InstructionCore selected by the exact
SystemMapping has exactly one compatible binary and entry. Each hardware
implementation has exactly one compatible RuntimePlatformBinding. Runtime may
choose a concrete installed device instance admitted by that binding, but it
cannot substitute another implementation, target, binary, Mapping, or
configuration.

The two Deployment-owned references above are valid only after finalization.
Their ordinals resolve into the inline HostProgramLeaf catalogs and cannot be
replaced by ABI symbols, file offsets, or runtime addresses.

`configuration_image_refs[]` is sorted by typed semantic keys. Duplicate,
foreign, missing, or incompatible references are rejected. The
configuration-image set equals the transitive closure mechanically required by
the complete Mapping and exact ABI programming units. Selected
HardwareImplementations must implement that ABI; they cannot change image
membership or content.

Equal payload bytes for several occurrence-qualified images may share blob
storage and may be installed by one provider multicast transaction. Every
image reference and programming binding remains present and independently
verified. Multicast is therefore an execution optimization over the exact
Deployment closure; it cannot merge Programming Units, replace an image,
change its source Mapping, or create a cross-core configuration identity.

A verified SystemMapping closure, joined only with the exact downstream leaves
named by each schema, mechanically derives `ThreadDispatchImage`,
`SpatialLaunchImage`, and `AdmissionImage`. `ThreadDispatchImage` and
`AdmissionImage` are present in every valid Deployment. `SpatialLaunchImage`
is present if and only if the exact imported SpatialMapping set is non-empty.

These payloads are immutable, typed, versioned children of Deployment. They
are not Mapping records, runtime-owned mutable state, or independent Artifacts
with their own ArtifactIdentity. Their stable keys and canonical ordering
derive from exact Mapping structural keys and relations.
Runtime-image children must not duplicate Mapping legality or target choices.
Every selected spatial programming unit still requires its exact
`HardwareConfigurationImage`.

### Runtime-Image Semantic Contract And Persistent Wire

The three children are versioned Deployment-local payloads, not independent
Artifact families. They use the closed descriptors
`loom.thread_dispatch_image 1.0`, `loom.spatial_launch_image 1.0`, and
`loom.admission_image 1.0` inside Deployment canonical JSON. They have no
independent digest or identity; their complete canonical bytes are covered by
the enclosing Deployment identity.

Every child binds one exact `source_system_mapping_ref`. That single
source is sufficient because its transitive lineage already owns exact
Dataflow, Fabric, and imported SpatialMapping identities. The persistent
schema uses Deployment canonical JSON and preserves the semantic keys,
relations, fields, and cardinalities below without introducing another target
selection.

`ThreadDispatchImage.payload` is:

```text
ThreadDispatchPayload {
  rows[] keyed by RootThreadLaunchRef {
    compiled_thread_execution_binding
    logical_parameter_schema
    explicit_dependencies[]
    thread_completion_destination
    target_cases[] keyed by AccCoreOccurrenceRef {
      instruction_core_entry_ref :
        (InstructionCoreBinaryRef, thread_entry_ordinal)
      memory_capability_requirements[]
      long_lived_activation_uses[]
    }
  }
}
```

`logical_parameter_schema` is exactly the Dataflow-owned root-launch parameter
inventory used by `EventLogicalProjection`: extents first in coordinate order,
then admitted `index` or signless-integer body operands in body-operand order.
It is a verified compiled copy for dispatch decoding, not an editable runtime
or Deployment-owned schema. The importer rederives it from the exact Dataflow
artifact and rejects disagreement.

There is exactly one row for every root thread launch in the complete
SystemMapping closure and no other row. The compiled binding preserves the
closed `BindingRelation<AccCoreOccurrenceRef>` semantics; it does not create a
second target-selection authority. Its finite unique range is exactly the
canonical `target_cases[]` key set, so a heterogeneous relation may select
different AccCores and executable entries for different concrete points. A
missing, extra, or duplicate case is invalid. `instruction_core_entry_ref` is
a typed reference into the exact executable closure; it cannot be replaced by
a source symbol, emitted symbol, address, or runtime handle.

For every target case, the referenced InstructionCoreBinary must bind the row's
exact `RootThreadLaunchRef` to the selected `thread_entry_ordinal`. Deployment
does not infer this correspondence from a shared callee definition and cannot
select an ordinal merely because it exists in the binary. The binary owns the
compiled-support relation; this image owns only the Mapping-specific target
selection. A missing relation row, wrong Dataflow owner, or mismatched ordinal
is invalid before runtime-image publication.

`SpatialLaunchImage.payload` is:

```text
SpatialLaunchPayload {
  rows[] keyed by GraphExecutionBindingKey {
    compiled_graph_execution_binding
    target_cases[] keyed by SpatialExecutionContextKey {
      required_configuration_image_refs[]
      value_boundary_bindings[]
      stream_boundary_bindings[]
      control_boundary_bindings[]
      memory_boundary_bindings[]
      graph_start_activation_set_ref
      result_destinations[]
      done_destination
    }
  }
}
```

There is exactly one row for each reachable static graph launch covered by a
Graph Execution Binding and no other row.
The row key is the Dataflow-owned `RootedGraphLaunchRef`; Deployment does not
reconstruct the root-plus-site tuple. Value, stream, control, memory, result,
and done entries are keyed by the closed Dataflow boundary, channel, and
logical-memory root/view or memory-exposure references rather than symbols or
local operation positions.
The finite unique range of the relational join between the parent Thread
Execution Binding and this compiled
`BindingRelation<SpatialMappingImportRef>` is
exactly the canonical `target_cases[]` key set. The key is structural, not a
new entity. It is exactly the Spatial variant of the `ExecutionContextKey`
owned by `docs/spec-mapping-identity.md`, encoded through the SystemMapping's
existing canonical SpatialMapping import table. It distinguishes the same
SpatialMapping instantiated on different AccCore occurrences. Each case
contains only material mechanically derived for that already selected pair.
`required_configuration_image_refs[]` is the exact sorted subset joined from
the Deployment configuration-image closure; it is an access index, not a
second configuration selection. The whole child is absent exactly when the
imported SpatialMapping set is empty.

`AdmissionImage.payload` is:

```text
AdmissionPayload {
  rows[] keyed by EventFamilyKey {
    contexts[] keyed by ExecutionContextKey {
      parameter_relation : BindingRelation<AdmissionCaseOrdinal>
      cases[] keyed by AdmissionCaseOrdinal {
        atomic_activation_set
        release_rules[] {
          activation_member_ref
          fabric_intrinsic_release : true
          causal_release? {
            all_of[] {
              alternatives[] : EventFamilyKey
              guaranteed_offset?
            }
          }
        }
        capacity_indices[]
      }
    }
  }
}
```

The `ExecutionContextKey` in this payload is the same key owned by
`docs/spec-mapping-identity.md`; Deployment does not define another context
tuple or context identity.

Each `EventFamilyKey` is exactly the complete Dataflow-owned closed union:

```text
EventFamilyKey =
    Transfer(Produced(CanonicalProducerTerminalRef)
           | Consumed(CanonicalSinkTerminalRef))
  | ActorTransition(ContextualActorTransitionEventRef)
```

The exact Dataflow program mechanically derives that key's
`EventLogicalProjection`; the projection is not copied into the row key or
stored as a second schema field. The keys denote static event families, never
dynamic event occurrences, static event IDs, concrete coordinate or parameter
values, or absolute time. In particular, a System memory or fence
`ResourceUse` retains its exact rooted actor-issue transition rather than
inventing a transfer terminal.

An imported Spatial endpoint trigger is indexed under every member of the
Dataflow-owned `RootedGraphEndpointEventProjection` for its exact rooted graph
launch. Those rows are alternatives for one original trigger, not independent
simultaneous acquisitions. Each original Spatial causal release point becomes
one `causal_all_of` member whose `alternatives[]` is that same nonempty
projection; the original Spatial `AllOf` therefore remains
`AllOf(AnyOf(point_0), AnyOf(point_1), ...)`. A direct System event point has a
singleton alternatives array. Empty alternatives, flattening alternatives
into the conjunction, or replacing a completion-frontier token with graph-wide
done is invalid.

The relation uses the same closed partition/lookup algebra as Mapping over the
typed slots in the derived projection and, for DynamicWork, the separately
owned stable-item projection. It selects a canonical child-local case. Its
finite unique range is exactly the case-key set. There is exactly one row and
context for every admission relation required by the verified closure and no
other one. Thread Dispatch and Spatial Launch may reference the same case.
`AdmissionCaseOrdinal` is assigned only after unique case payloads are sorted
by canonical semantic bytes; relations are then rewritten to that derived
ordinal. It is not an EntityId or an independent selection authority.

The Deployment canonical JSON writer renders each row key as the exact typed
`Transfer.Produced`, `Transfer.Consumed`, or `ActorTransition` variant plus its
terminal or contextual actor-transition reference, and renders each relation
variable as the exact `EventLogicalInputSlot` variant plus ordinal. It orders
rows and variables by the Dataflow-owned comparison wires. The importer
rederives the complete projection from the exact Dataflow artifact and rejects
disagreement. No projection digest, copied type, native index, symbol path,
JSON field order, or parallel binary Deployment schema becomes another
authority.

Contexts, target cases, and admission cases are sorted by canonical key
bytes. Every nested set is sorted and unique; every relation uses the canonical
Mapping relation representation owned by the Mapping schema. Authoring order,
runtime queue order, and derived dense-index assignment do not affect the
eventual bytes. Validation rederives all row domains, relation ranges, and
payload contents from the exact verified closure, then checks the eventual
descriptor, source, digest, presence, and cardinality and rejects disagreement
atomically. It never repairs a child or writes derived content back into
Mapping.

Runtime handles, allocations, paths, logs, and mutable admission state do not
enter Deployment semantic bytes. For each invocation, runtime establishes
authorization, lease, and isolation state for the exact preselected resources.
It cannot remap execution, service, route, tag, context, configuration, or any
other target choice. Runtime verifies the actual platform's Fabric,
implementation, and ABI identity, using hardware-reported identity when
available or an exact trusted `RuntimePlatformBinding`. A mismatch is rejection, not
remapping, compatible-hardware selection, or package repair.

## Finalization Dependency Graph

The complete Deployment has this mechanical derivation graph:

```text
exact Fabric
  -> finalize ConfigurationABI

exact Fabric + ConfigurationABI
  -> independently derive and finalize HardwareImplementation

verified SystemMapping closure
  + ConfigurationABI + selected HardwareImplementation
  -> verify their exact common Fabric and ABI closure
  -> consume the Mapping verifier's validated ConfiguredHardwareProjection
  -> encode and verify every required HardwareConfigurationImage

verified SystemMapping closure
  + required HardwareConfigurationImage set
  + HostProgramLeaf + exact InstructionCoreBinary set
  -> mechanically derive the required Deployment runtime-image children

verified executable leaves + selected HardwareImplementation set
  + exact RuntimePlatformBinding set
  + required HardwareConfigurationImage and runtime-image children
  + static logical-memory images
  -> finalize Deployment
```

ConfigurationABI and HardwareImplementation may be finalized without any
software Mapping. Validation of this branch begins only after those independent
branches and a verified SystemMapping closure meet. An implementation must not
impose a hidden sequence in which Mapping is required before ABI or RTL
generation. Deployment finalization owns complete closure and atomically
rejects a missing, ambiguous, foreign, incompatible, or unused binding.

Missing or duplicate fields, an unencodable semantic value, overlapping or
out-of-range slices, ABI/implementation mismatch, a wrong programming unit,
payload corruption, binary coverage failure, runtime-platform mismatch, or an
image set different from the required closure causes finalization to fail. No
later component reconciles it.

## Deployment Canonicalization And Publication

Deployment canonical semantic bytes are one canonical JSON object. Required
fields are always present. `spatial_launch_image` is omitted exactly when its
presence condition is false; no `null`, empty placeholder, or alternate child
encoding is permitted. Artifact references use the Common exact framing, blob
references use `BlobDigest`, integers use canonical decimal spelling, enums use
their one registered spelling, and arrays are sorted and deduplicated by their
complete typed semantic keys.

The root stores only direct references and inline leaves. Its transitive
artifact and blob closure is derived and validated, not copied into a manifest
field. Runtime-image children acquire no child digest. Package layout, producer
metadata, paths, timestamps, diagnostics, and report data do not enter bytes.

Finalization is failure-atomic: validate all referenced artifacts and blobs,
rederive binary and hardware coverage, build and verify the three runtime-image
children, serialize canonical JSON, independently import and reverify it, then
compute and publish the Common ArtifactIdentity. No partial Deployment object
is ever visible, and a failed attempt returns no successful root reference. A
post-insertion `artifact_store_io` may nevertheless leave the complete object
visible; retry and recovery follow the Common single-object store contract.

## Public Driver Contract

`loom-cc` and `loom-c++` retain ordinary compiler-driver output semantics.
`-o` never changes meaning because acceleration is enabled. Semantic compiler,
DSE, Mapping, Evaluation, and backend policy is selected only through the
public semantic profile selector owned by
[Resolved Configuration](spec-config-ssot.md#public-selection). This
specification does not repeat or alias that option spelling.

A final link or whole-program invocation requests a Deployment package only
through:

```text
--loom-deploy-output=<path>
```

The path and package choice are invocation bindings and do not affect
ResolvedConfig, Deployment, or upstream Artifact identities. `-E`, `-S`, and
`-c` retain their ordinary primary output and reject a Deployment request.
Requesting Deployment does not silently strengthen the selected profile; an
incomplete Mapping, ABI, implementation, or Deployment closure fails.

Drop-in separate compilation requires a frontend-owned relocatable accelerator
payload in or beside ordinary objects and collection of that payload at final
link. Compile-only output must not embed Mapping, Fabric,
ConfigurationABI, or HardwareConfigurationImage artifacts. The exact payload
wire schema, symbol resolution, config-view compatibility, and carrier-neutral
embedding contract belong to `docs/spec-compiler-part-1-source.md`. An object-
format provider must implement that closed payload contract or report typed
`Unsupported`; it cannot substitute a whole-program-only path while claiming
drop-in support.

## Package Projection

The first stable package projection is a content-addressed directory:

```text
<deploy-output>/
  root
  objects/<artifact-identity>
  blobs/<blob-digest>
```

`root` contains exactly one lowercase hexadecimal Deployment identity.
`objects/` contains the exact typed Artifact Store objects in the Deployment
execution closure. `blobs/` contains only referenced non-Artifact binary
payloads. The directory contains no missing or unreferenced entries and does
not duplicate an image payload as a second bitstream file.

Publication builds and verifies a temporary sibling tree, then atomically
publishes it. Paths, mtimes, inodes, creation order, compression, and a future
archive or executable-section projection do not affect Deployment identity.
There is no public `loom-deploy`, `loom-bitstream`, raw-image flag, implicit
sidecar, or alternate encoder in the first contract. Developer tools may call
the same owner library using exact identities.

## Verification Anchors

Tests protect only stable boundaries:

* harmless ABI field ordering does not change identity;
* invalid slices, foreign fields, invalid codebooks, and invented inactive
  values are rejected;
* a System-rooted programming unit distinguishes two occurrences importing the
  same Module and rejects a bare Module-local owner or field;
* `None` rejects an ABI field, a finite codebook has exact set equality with
  its Fabric behavior-key domain, and `DirectBits` matches the Fabric carrier
  width exactly;
* one Spatial PE with at least two FUs exposes its complete static activation
  and selector field schema in canonical order independent of authoring order;
  each field has exact finite codebook coverage and foreign endpoint rejection,
  activation has canonical disabled inactivity, and one selected Mapping emits
  only the chosen FU's selector and operation fields;
* fixed canonical byte vectors cover `Disabled`, one `Active` occurrence, one
  `Disconnected` selector, one selector carrying a PE transport endpoint, and
  output `Discard` with no payload; malformed or noncanonical bytes fail;
* one known vector is shared by the encoder and RTL/runtime decoder;
* image ABI, programming-unit, Mapping, padding, and payload mismatches fail;
* Deployment requires `ThreadDispatchImage` and `AdmissionImage`, and requires
  `SpatialLaunchImage` if and only if the exact imported SpatialMapping set is
  non-empty;
* Deployment derives runtime-image child stable keys and canonical ordering
  from exact Mapping structural keys and relations;
* admission rows use exactly one Dataflow-owned `EventFamilyKey` per static
  event alternative, endpoint alternatives preserve disjunction inside the
  original release conjunction, and projection-slot ordering, wire roundtrip,
  empty projection, and foreign or noncanonical slot rejection match the exact
  Dataflow owner;
* heterogeneous Thread and Graph binding ranges produce exact target-case
  tables without a singular duplicate selection field;
* Deployment accepts exactly its required configuration-image/runtime-image
  branch and atomically rejects missing, ambiguous, foreign, incompatible, or
  extra child bindings;
* complete and unique HostCore, InstructionCore binary, static memory,
  HardwareImplementation, and RuntimePlatformBinding coverage;
* runtime-image inline canonical bytes changing Deployment identity without
  acquiring a second child identity;
* recovery of exact Dataflow, Fabric, ABI, interconnect, and platform facts
  through owner references, with a duplicate disagreement rejected;
* runtime authorization, lease, and isolation state cannot change a
  Mapping-selected resource or target choice;
* package projection preserves Deployment identity across output paths and
  never exposes a partial tree; and
* public drivers preserve ordinary `-o` behavior and reject Deployment output
  in non-link modes.

Tests do not create a generic schema registry, one fixture per transport,
archive-format matrices, filesystem-metadata snapshots, or alternate encoders.
