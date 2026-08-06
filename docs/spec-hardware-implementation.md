# Hardware Implementation

This document defines the immutable hardware-state artifact produced by
Fabric-to-RTL and downstream ASIC or FPGA implementation flows. Fabric remains
the hardware semantic authority. A HardwareImplementation records concrete
implementation state that cannot be recovered from Fabric alone.

## Artifact Family

```text
loom.hardware_implementation 1.0
```

```text
HardwareImplementation {
  version
  fabric_ref
  configuration_abi_ref
  interconnect_implementation_refs[]
  representation
  implementation_platform_ref?
  payloads[]
  interfaces[]
  activity_points[]
  memory_macro_bindings[]
  external_implementation_bindings[]
}
```

The artifact does not store Mapping, workload, configuration images, QoR
metrics, tool logs, report paths, or pass/fail signoff booleans.

`implementation_platform_ref` is absent only for a target-independent RTL
representation. It is mandatory when the represented RTL is specialized to an
ASIC technology release or FPGA ordering code, and for GateNetlist, every ASIC
physical variant, every FPGA physical variant, and an FPGA image. Dependence on
DesignWare, ChipWare, another tool-bundled library, or an explicit user IP does
not by itself create a target manifest; that dependence is recorded by an
external implementation binding.

## Representation

```text
Representation =
    Rtl
  | GateNetlist
  | AsicPhysical { Placed | Routed | Extracted }
  | FpgaPhysical { Placed | Routed }
  | FpgaImage
```

The closed variant identifies the semantic implementation state represented by
the payload closure. It is not a linear mandatory pipeline: a selected flow
may omit forms it does not materialize. A generic stage string or bag of
optional format fields is forbidden.

## Semantic Closure And Derivation

A HardwareImplementation is a complete self-contained description of the
represented implementation state. Its canonical root contains no parent
implementation, generator binding, or derivation edge. A consumer can import
and validate the implementation from its own exact Artifact references and
payload closure without recovering the invocation that produced it.

`InvocationManifest` is the sole owner of generation history. Its canonical
`MechanicalDerivation` or `CandidateDecision` record binds the exact typed
input Artifacts, resolved candidate-generator binding, and output
HardwareImplementation. Several valid derivations that produce identical
canonical implementation state converge on one HardwareImplementation
identity while the manifest may retain every derivation edge.

A downstream transformation may reuse payload BlobDigests from an input
implementation, but its output must enumerate every payload and exact
dependency required to reconstruct or consume the new represented state. It
cannot rely on an implicit parent closure. A generation choice that remains
relevant to the represented state must be materialized by an existing
HardwareImplementation owner such as a payload, interface, platform reference,
memory-macro binding, or external implementation binding. If none of those
owners can express a required fact, the HardwareImplementation schema requires
a specific semantic field; the generator binding cannot serve as a catch-all.

Every persisted implementation change produces another immutable
HardwareImplementation. In-place mutation is forbidden. A change to
Fabric-visible function, latency, initiation interval, capacity, buffering,
progress, reset, or ConfigurationABI is not an implementation refinement: it
requires a new Fabric and remapping.

## Semantic Payload Roles

```text
PayloadRole =
    RtlSource
  | Netlist
  | PhysicalDatabase
  | Parasitics
  | LayoutStream
  | DeviceImage
  | GenerationConstraint
  | BlackBoxContract
```

Each payload records a closed role, canonical logical name, media type, and
BlobDigest. The artifact contains every payload required to reconstruct or
consume the represented implementation state. Backend logs, reports,
waveforms, temporary databases, and tool caches remain raw attempt bundles
unless they are one of these semantic implementation payloads.

Generated SDC or equivalent constraints use `GenerationConstraint`. Their
bytes are derived during generation from Fabric clock/reset/crossing facts, the
exact generator configuration, implementation interfaces, the target manifest
when present, and any exact external implementation contract that supplies a
required load or interface fact. The resulting payload is self-contained;
consuming it does not require the generator binding. It does not become a
timing authority or a separate constraint artifact.

## Interfaces And Activity Points

```text
ImplementationInterface {
  interface_key
  role: ImplementationInterfaceRole
  semantic_fabric_ref
  representation_locator
  device_pin_ref?
}

ActivityPoint {
  activity_point_id
  representation_locator
  semantic_fabric_ref?
}

ImplementationInterfaceRole =
    Data
  | Clock
  | Reset
  | Configuration
  | Memory
  | ExternalProtocol

RepresentationLocator {
  object_kind: RepresentationObjectKind
  canonical_name
}

RepresentationObjectKind =
    Module
  | Instance
  | Port
  | Net
  | Register
  | Memory
  | Cell
  | Pin
  | PhysicalObject
  | DeviceResource
```

The interface catalog binds Fabric-visible boundaries, clocks, resets,
configuration transports, memories, and external protocols to exact
implementation locators. The activity catalog is the sole implementation-
local source for RTL, netlist, physical, and FPGA activity references used by
simulation or Evaluation.

The enclosing representation gives every locator its representation-local
interpretation; a locator therefore does not repeat a representation tag.
`canonical_name` is the stable name within that exact represented state, not a
filesystem path, report path, tool query, or Fabric entity name. The closed
object kind prevents a port, net, cell, pin, physical object, and device
resource from becoming interchangeable strings. A locator kind incompatible
with the enclosing representation is invalid.

Locators do not alter Fabric or Mapping identity. `device_pin_ref` is valid
only for an FPGA representation with an exact FPGA target manifest. A missing
required interface or activity point makes the artifact incomplete.

## Memory And External Bindings

```text
ExternalDependencyIdentity =
    ExplicitFile {
      content_sha256
    }
  | ToolBundledResource {
      stable_provider_build_identity
      resource_key
    }

ExternalInputBinding {
  provider_input_slot_ref
  dependency_identity: ExternalDependencyIdentity
}

ExternalImplementationBinding {
  binding_id
  provider_contract_ref
  external_inputs: canonical nonempty catalog<ExternalInputBinding>
  fabric_resource_refs[]
  representation_locators[]
  black_box_contract_payload_ref?
}

MemoryMacroBinding {
  fabric_memory_ref
  external_implementation_binding_id
  representation_locator
}
```

An explicit-file identity is the SHA-256 fingerprint of exactly one ordinary
file selected through a provider-owned typed input slot. It is not a BlobDigest
claim and does not require Loom to copy the file into BlobStore. A tool-bundled
identity combines the stable provider build selected by the semantic binding
with one exact provider resource key. A display version alone is invalid when
the provider requires a stronger build identity.

The exact provider contract is the sole owner of input-slot identity, role,
cardinality, and compatibility. An external implementation binding records the
closed slot-to-dependency relation required by its represented implementation
state. A memory macro may therefore bind distinct logical, timing, physical,
and layout files without collapsing them into one directory or one digest.
A representation includes only the slots required to reconstruct or consume
that state; a later state closes its own required set rather than inheriting an
implicit earlier binding.

`external_implementation_bindings` cover vendor arithmetic libraries, FPGA
primitives or configured IP, fixed or generated memory macros, encrypted or
black-box user IP, and technology libraries instantiated by the represented
state. The provider contract owns each dependency's typed interpretation and
compatibility rules. The HardwareImplementation owner owns only the closed
slot-to-identity relation, exact Fabric relations, locators, and optional
black-box payload relation shown above. Paths, filenames as semantic roles,
and free-form property maps are forbidden.

`memory_macro_bindings` map exact Fabric memory occurrences to one compatible
external implementation binding and representation locator. Fabric owns the
required memory semantics. The provider contract owns the offered macro
contract and exact external view slots. ImplementationPlatform does not contain
a macro library or its files.

Synthesizable user source that is incorporated into the represented RTL is an
`RtlSource` payload rather than an external file dependency. An encrypted or
otherwise nonmaterialized implementation remains an external binding with an
exact `BlackBoxContract`. No binding permits a missing dependency to masquerade
as a complete implementation.

These bindings are downstream `HardwareImplementation` facts. They are not
Fabric `ImplementationInput` dependencies and cannot be used to make that
reserved-unavailable `loom.fabric 2.x` role legal. An Interconnect
Implementation remains self-contained apart from its exact RefinedSystem root;
provider-owned external implementation state is selected and validated here.

An implementation provider may report `Unsupported` when an otherwise valid
Fabric resource lacks an implementation. It cannot emit a substituted,
truncated, or placeholder implementation.

## Physical Design Boundaries

The first version supports one always-on power state. Power gating, isolation,
retention, DVFS, partial reconfiguration, DFT insertion, ATPG, fault injection,
and general reliability policy are explicitly deferred until they have an
independently observable contract.

Floorplan, placement, routing, and tool-control choices are typed generator
configuration and resulting implementation payloads. Loom does not define a
global floorplan or vendor-command DSL. FPGA uses the same immutable
implementation family as ASIC; the first version produces a static full-device
image and does not claim partial-reconfiguration support.

Timing, power, area, thermal, DRC, and other observations are
EvaluationEvidence over the exact HardwareImplementation. Negative slack or a
physical violation may be reported for a completed implementation; a tool
failure that produces no coherent represented state publishes no
HardwareImplementation.

Timing closure follows semantic ownership. Gate sizing, placement, or another
implementation-only choice creates another HardwareImplementation when it
changes materialized semantic state. `InvocationManifest` records the exact
input and generation decision. Selecting a Fabric-declared bypass, buffer, or
latency refinement changes Mapping and its configuration. Inserting state or
changing latency, initiation interval, or recurrence behavior outside such a
declared refinement creates a new Fabric candidate and requires remapping.
Central DSE composes the resulting candidates and EvaluationEvidence; an EDA
adapter cannot mutate any of those owners in place.

## Finalization

Finalization verifies exact Fabric, ConfigurationABI, interconnect,
the optional target manifest, payload roles, digests, interfaces, activity
points, memory-macro bindings, and external bindings. It resolves every
provider contract, validates each external dependency identity, verifies every
Fabric-resource and representation-locator relation, and rejects an external
module without its required black-box contract. It also verifies that the
represented state has no implicit dependency on a parent implementation or
generator invocation. Canonical ordering uses typed semantic keys. Filesystem
paths, mtimes, tool invocation order, generator search history, and reports do
not enter identity.

Canonical semantic bytes are one canonical JSON root containing exact Artifact
references and BlobDigest payload references. Closed variants and typed keys
use their registered canonical spelling; sets and catalogs are sorted and
deduplicated by complete semantic key. Backend-native manifests are payloads,
not an alternate HardwareImplementation root.

The artifact is published only after every required payload and binding is
available and independently re-readable. Partial manifests and path-based
success markers are invalid. Finalization independently reimports and verifies
the canonical root before atomic publication.

## Anchor Verification

Anchor tests cover:

* manifest edges for `Fabric + ConfigurationABI -> RTL H0` and `H0 -> H1`
  retaining exact typed inputs and resolved generator bindings;
* two distinct derivations with identical canonical implementation state
  converging on one HardwareImplementation identity;
* a recipe-only structural change changing HardwareImplementation identity
  exactly when it changes a materialized payload or binding, without changing
  Fabric or ConfigurationABI identity;
* rejection of a derived implementation that depends on an implicit parent
  payload;
* a Fabric-visible timing or capacity change being rejected as an
  implementation refinement;
* missing, duplicate, wrong-role, or corrupt payloads;
* required interface, activity-point, memory-macro, and configuration-
  transport coverage;
* explicit-file and tool-bundled external identities producing distinct
  bindings without importing their installation trees;
* a memory macro selecting one exact external binding and rejection of a
  platform-owned macro-file lookup;
* ASIC and FPGA representation variants under the same family; and
* completed adverse timing evidence versus a tool failure that publishes no
  HardwareImplementation.

Tests do not freeze vendor report text, Tcl formatting, database directory
layout, every EDA tool, or a large format-conversion matrix.
