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
  lineage
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

`implementation_platform_ref` is absent only for a technology-independent RTL
representation whose complete providers and black-box contracts require no
technology facts. It is mandatory for a technology-bound RTL variant,
GateNetlist, every ASIC physical variant, every FPGA physical variant, and an
FPGA image.

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

## Lineage

```text
ImplementationLineage =
    Initial {
      resolved_generator_binding_ref
    }
  | Derived {
      parent_implementation_ref
      resolved_generator_binding_ref
      relation = MechanicalDerivation | CandidateDecision
    }
```

An initial implementation is derived from exact Fabric and ConfigurationABI.
A child preserves exact parent identity and the generator binding that created
its new implementation state. A search choice uses `CandidateDecision`; a
deterministic format or tool transformation uses `MechanicalDerivation`.

Every persisted physical change creates a child. In-place mutation is
forbidden. A change to Fabric-visible function, latency, initiation interval,
capacity, buffering, progress, reset, or ConfigurationABI is not an
implementation refinement: it requires a new Fabric and remapping.

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
bytes are derived from Fabric clock/reset/crossing facts, the exact generator
configuration, implementation interfaces, and ImplementationPlatform. They do
not become a timing authority or a separate constraint artifact.

## Interfaces And Activity Points

```text
ImplementationInterface {
  interface_key
  role
  semantic_fabric_ref
  representation_locator
  device_pin_ref?
}

ActivityPoint {
  activity_point_id
  representation_locator
  semantic_fabric_ref?
}
```

The interface catalog binds Fabric-visible boundaries, clocks, resets,
configuration transports, memories, and external protocols to exact
implementation locators. The activity catalog is the sole implementation-
local source for RTL, netlist, physical, and FPGA activity references used by
simulation or Evaluation.

Locators are representation-local typed values. They do not alter Fabric or
Mapping identity. A missing required interface or activity point makes the
artifact incomplete.

## Memory And External Bindings

`memory_macro_bindings` map exact Fabric memory occurrences to typed
TechnologyMemoryMacro references from the ImplementationPlatform and exact
representation locators. `external_implementation_bindings` cover other
Fabric-declared black boxes or external IP through typed provider-owned
references. Paths and free-form property maps are forbidden.

These bindings are downstream `HardwareImplementation` facts. They are not
Fabric `ImplementationInput` dependencies and cannot be used to make that
reserved-unavailable `loom.fabric 1.0` role legal. An Interconnect
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
global floorplan or vendor-command DSL. FPGA uses the same artifact lineage as
ASIC; the first version produces a static full-device image and does not claim
partial-reconfiguration support.

Timing, power, area, thermal, DRC, and other observations are
EvaluationEvidence over the exact HardwareImplementation. Negative slack or a
physical violation may be reported for a completed implementation; a tool
failure that produces no coherent represented state publishes no child.

Timing closure follows semantic ownership. Gate sizing, placement, or another
implementation-only choice creates a HardwareImplementation child. Selecting a
Fabric-declared bypass, buffer, or latency refinement changes Mapping and its
configuration. Inserting state or changing latency, initiation interval, or
recurrence behavior outside such a declared refinement creates a new Fabric
candidate and requires remapping. Central DSE composes the resulting candidates
and EvaluationEvidence; an EDA adapter cannot mutate any of those owners in
place.

## Finalization

Finalization verifies exact Fabric, ConfigurationABI, interconnect,
ImplementationPlatform, parent lineage, generator binding, payload roles,
digests, interfaces, activity points, and external bindings. Canonical ordering
uses typed semantic keys. Filesystem paths, mtimes, tool invocation order, and
reports do not enter identity.

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

* `Fabric + ConfigurationABI -> RTL H0 -> derived H1` with exact parent
  lineage;
* a recipe-only structural change changing HardwareImplementation identity but
  not Fabric or ConfigurationABI identity;
* a Fabric-visible timing or capacity change being rejected as an
  implementation refinement;
* missing, duplicate, wrong-role, or corrupt payloads;
* required interface, activity-point, memory-macro, and configuration-
  transport coverage;
* ASIC and FPGA representation variants under the same family; and
* completed adverse timing evidence versus a tool failure that publishes no
  child.

Tests do not freeze vendor report text, Tcl formatting, database directory
layout, every EDA tool, or a large format-conversion matrix.
