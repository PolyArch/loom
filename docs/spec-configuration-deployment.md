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
physical bit positions. `ConfigurationABI` owns those positions once. An
encoder, RTL decoder, runtime loader, simulator, or backend must consume the
same ABI rather than duplicating it.

The Artifact DAG is acyclic. A `ConfigurationABI` references its exact Fabric.
A `HardwareImplementation` references the exact Fabric and ConfigurationABI it
implements. The ABI does not refer back to a final implementation.

An implementation-only backend recipe is selected by the exact hardware
candidate-generator binding per `FabricEntityRef`. That selection contributes
to `HardwareImplementation` lineage and identity, but only when every
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
loom.configuration_abi             1.0
loom.hardware_configuration_image  1.0
loom.deployment                    1.0
```

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
      fabric_config_field_ref
      semantic_encoding
      destination_slices[]
      inactive_value
    }
  }
}
```

A programming unit is the smallest physical unit for which a complete
configuration state can be independently installed and activated, such as one
SpatialCore context bank. It is not a Mapping entity, a runtime handle, or a
temporal-PE instruction slot.

The only persistent reference to a programming unit is:

```text
ProgrammingUnitRef =
  ArtifactReference<ConfigurationABI> + ABI-local unit_id
```

There is no global programming-unit registry or compatibility label. Each unit
owns an exact Fabric occurrence closure. The first programming model installs
a complete image and defines when that image becomes visible and active.
AXI, JTAG, MMIO, and custom transport mechanisms belong to the implementation
that realizes the ABI; the ABI does not contain a command-script language.

### Field Encoding

Composite semantic configuration is flattened into typed leaves during ABI
finalization. A leaf uses one of two atoms:

* `DirectBits`: encode one fixed-width bit vector directly.
* `FiniteCodebook`: map each allowed typed value to one unique physical code.

`destination_slices` may cross words or be non-contiguous. Every destination
bit has exactly one source, slices are in range and non-overlapping, and all
reserved or padding bits are zero. An unselected hardware field uses the
ABI-declared `inactive_value`; Fabric must prove that value functionally inert.
The encoder may not invent a default. RTL providers consume these fields and
their codebooks; they cannot create an independent exact-mode index or decoder
authority.

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

Image finalization reconstructs a temporary `ConfiguredHardwareProjection`
from exact Fabric and complete Mapping, encodes it through the ABI, and verifies
the resulting bytes. That projection is derived data and is never persisted as
another authority. Images are bound to exact hardware occurrences; equal bytes
may share blob storage but do not permit implicit rebinding.

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
Implementation, and ImplementationPlatform facts are recovered from each
selected HardwareImplementation. Deployment does not duplicate those
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
    target_cases[] keyed by GraphLaunchTargetKey {
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

GraphLaunchTargetKey = (AccCoreOccurrenceRef, SpatialMappingImportRef)
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
new entity. It distinguishes the same SpatialMapping instantiated on different
AccCore occurrences. Each case contains only material mechanically derived for
that already selected pair. `required_configuration_image_refs[]` is the exact
sorted subset joined from the Deployment configuration-image closure; it is an
access index, not a second configuration selection. The whole child is absent
exactly when the imported SpatialMapping set is empty.

`AdmissionImage.payload` is:

```text
AdmissionPayload {
  rows[] keyed by EventFamilyKey {
    contexts[] keyed by ExecutionContextKey {
      parameter_relation : BindingRelation<AdmissionCaseOrdinal>
      cases[] keyed by AdmissionCaseOrdinal {
        atomic_activation_set
        release_rules[]
        capacity_indices[]
      }
    }
  }
}
```

Each `EventFamilyKey` is exactly the Dataflow-owned
`Produced(CanonicalProducerTerminalRef)` or
`Consumed(CanonicalSinkTerminalRef)` structural event. The exact Dataflow
program mechanically derives that key's `EventLogicalProjection`; the
projection is not copied into the row key or stored as a second schema field.
The keys denote static event families, never dynamic event occurrences,
static event IDs, concrete coordinate or parameter values, or absolute time.
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
`Produced` or `Consumed` variant plus its terminal reference, and renders each
relation variable as the exact `EventLogicalInputSlot` variant plus ordinal.
It orders rows and variables by the Dataflow-owned comparison wires. The
importer rederives the complete projection from the exact Dataflow artifact and
rejects disagreement. No projection digest, copied type, native index, symbol
path, JSON field order, or parallel binary Deployment schema becomes another
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
  -> derive temporary ConfiguredHardwareProjection
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
DSE, Mapping, Evaluation, and backend policy is selected only through:

```text
--loom-accel-profile=<builtin-preset-or-config-path>
```

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
* one known vector is shared by the encoder and RTL/runtime decoder;
* image ABI, programming-unit, Mapping, padding, and payload mismatches fail;
* Deployment requires `ThreadDispatchImage` and `AdmissionImage`, and requires
  `SpatialLaunchImage` if and only if the exact imported SpatialMapping set is
  non-empty;
* Deployment derives runtime-image child stable keys and canonical ordering
  from exact Mapping structural keys and relations;
* admission rows use exactly one Dataflow-owned `EventFamilyKey` per static
  event, and projection-slot ordering, wire roundtrip, empty projection, and
  foreign or noncanonical slot rejection match the exact Dataflow owner;
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
