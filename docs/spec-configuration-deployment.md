# Configuration And Deployment

This document is the semantic owner for Loom hardware-configuration encoding,
configuration images, Deployment artifacts, and the first public Deployment
package projection. It expands the closed configuration and deployment
decisions without creating a new MLIR dialect or a generic artifact-schema
framework.

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
  -> exact executable dependency closure
```

Fabric and Mapping may expose or select semantic `sw_configs`; they do not own
physical bit positions. `ConfigurationABI` owns those positions once. An
encoder, RTL decoder, runtime loader, simulator, or backend must consume the
same ABI rather than duplicating it.

The Artifact DAG is acyclic. A `ConfigurationABI` references its exact Fabric.
A `HardwareImplementation` references the exact Fabric and ConfigurationABI it
implements. The ABI does not refer back to a final implementation.

SystemMapping binds the architecture-only Fabric and exact Transport
Architecture. Replacing one Interconnect or hardware implementation with
another verified implementation of the same architecture and
`ConfigurationABI` does not change SystemMapping or semantic configuration
selection. The selected implementation changes Deployment closure, but the
same semantic configuration encodes to the same
`HardwareConfigurationImage` identity under that unchanged ABI.

## Artifact Families

The three Artifact families have these fixed schema descriptors:

```text
loom.configuration_abi             1.0
loom.hardware_configuration_image  1.0
loom.deployment                    1.0
```

Each owner library defines one typed C++ model and one canonical serializer and
parser. `ConfigurationABI` and `Deployment` use canonical JSON semantic bytes.
`HardwareConfigurationImage` uses a canonical typed header followed by its raw
payload. Loom does not define MLIR mirrors, Protobuf alternatives, base64
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
The encoder may not invent a default.

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

`Deployment` is the selected executable dependency graph:

```text
Deployment {
  version
  dataflow_ref
  fabric_ref
  system_mapping_ref
  hardware_implementation_refs[]
  interconnect_implementation_and_refinement
  instruction_binary_bindings[]
  configuration_image_refs[]
  thread_dispatch_image
  spatial_launch_image?
  admission_image
  memory_image_refs[]
  platform_binding_refs[]
}
```

Sets are sorted by typed semantic keys. Duplicate, foreign, missing, or
incompatible references are rejected. The configuration-image set equals the
transitive closure mechanically required by the complete Mapping and exact ABI
programming units. Selected HardwareImplementations must implement that ABI;
they cannot change image membership or content.

`ThreadDispatchImage`, `SpatialLaunchImage`, and `AdmissionImage` are typed,
versioned Deployment payloads rather than independent Artifact families. An
InstructionCore-only Deployment may omit the spatial-launch payload and all
SpatialCore configuration images. Every selected spatial programming unit
requires its exact image.

Runtime handles, allocations, paths, logs, and mutable admission state do not
enter Deployment semantic bytes. Runtime verifies the actual platform's
Fabric, implementation, and ABI identity, using hardware-reported identity
when available or an exact trusted `PlatformBinding`. A mismatch is rejection,
not remapping, compatible-hardware selection, or package repair.

## Mechanical Dependency And Finalization

The only dependency and finalization graph is:

```text
exact Fabric
  -> finalize ConfigurationABI

exact Fabric + ConfigurationABI
  -> independently derive and finalize HardwareImplementation

complete Mapping + ConfigurationABI + selected HardwareImplementation
  -> verify their exact common Fabric and ABI closure
-> derive temporary ConfiguredHardwareProjection
-> encode and verify every required HardwareConfigurationImage
-> derive runtime images
-> finalize Deployment dependency graph
-> emit the selected package projection
```

ConfigurationABI and HardwareImplementation may be finalized without any
software Mapping. Deployment finalization begins only after those independent
branches and a complete Mapping meet. An implementation must not impose a
hidden sequence in which Mapping is required before ABI or RTL generation.

Missing or duplicate fields, an unencodable semantic value, overlapping or
out-of-range slices, ABI/implementation mismatch, a wrong programming unit,
payload corruption, or an image set different from the required closure is a
finalization failure. No later component reconciles these failures.

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
wire schema, symbol resolution, config-view compatibility, and object embedding
belong to the compiler frontend rather than this specification. A frontend
that lacks that complete contract must report separate-compilation acceleration
as unsupported; it cannot substitute a whole-program-only path while claiming
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
* Deployment accepts exactly its required closure and rejects missing or extra
  entries;
* package projection preserves Deployment identity across output paths and
  never exposes a partial tree; and
* public drivers preserve ordinary `-o` behavior and reject Deployment output
  in non-link modes.

Tests do not create a generic schema registry, one fixture per transport,
archive-format matrices, filesystem-metadata snapshots, or alternate encoders.
