# Hardware Implementation

This document defines the immutable hardware-state artifact produced by
Fabric-to-RTL and downstream ASIC or FPGA implementation flows. Fabric remains
the hardware semantic authority. A HardwareImplementation records concrete
implementation state that cannot be recovered from Fabric alone.

## Artifact Family

```text
loom.hardware_implementation 2.0
```

```text
HardwareImplementation {
  version
  fabric_ref
  configuration_abi_ref
  interconnect_implementation_refs[]
  representation_root
  implementation_platform_ref?
  interfaces[]
  activity_points[]
  memory_macro_bindings[]
  external_implementation_bindings[]
}
```

`fabric_ref` is an exact `loom.fabric 2.0` System root and
`configuration_abi_ref` is an exact `loom.configuration_abi 2.0` root bound to
that same System. Imported Module internals referenced by interfaces, activity,
configuration, memory, recipe, or external bindings use exact
occurrence-qualified physical targets. A bare Module root or unqualified
imported Module-local target cannot describe a complete implementation.

The artifact does not store Mapping, workload, configuration images, QoR
metrics, tool logs, report paths, or pass/fail signoff booleans.

`implementation_platform_ref` is absent only for a target-independent RTL
representation. It is mandatory when the represented RTL is specialized to an
ASIC technology release or FPGA ordering code, and for GateNetlist, every ASIC
physical variant, every FPGA physical variant, and an FPGA image. Dependence on
DesignWare, ChipWare, another tool-bundled library, or an explicit user IP does
not by itself create a target manifest; that dependence is recorded by an
external implementation binding.

## Representation Root

```text
ImplementationRepresentationRoot =
    Rtl {
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<Module>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | GateNetlist {
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<Module>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | AsicPhysical {
      stage: Placed | Routed | Extracted
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<PhysicalObject>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | FpgaPhysical {
      stage: Placed | Routed
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<DeviceResource>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | FpgaImage {
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<DeviceResource>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }

ImplementationPayload {
  role: PayloadRole
  canonical_logical_name: nonempty logical path
  blob_digest: BlobDigest
}

ImplementationPayloadRef =
  dense ordinal in canonical ImplementationPayload order
```

The stable root-variant tags are `Rtl = 0`, `GateNetlist = 1`,
`AsicPhysical = 2`, `FpgaPhysical = 3`, and `FpgaImage = 4`. ASIC stage tags
are `Placed = 0`, `Routed = 1`, and `Extracted = 2`; FPGA physical stage tags
are `Placed = 0` and `Routed = 1`. Payload-role and representation-object tags
follow their displayed declaration order below. Binary encoding uses `u32be`
for every tag and `u64be` length framing for variable byte strings and arrays.
Canonical JSON uses the exact displayed spellings and contains no aliases.

Payloads sort by `(role tag, canonical logical-name bytes, BlobDigest bytes)`.
The pair `(role, canonical_logical_name)` is unique. A logical name is a
normalized relative UTF-8 path with nonempty segments, `/` separators, and no
`.` or `..` segment; it is a namespace inside the represented state, never a
host path or an attempt output path. Dense payload refs are derived after this
ordering and a caller cannot author them.

One static typed representation-format registry owns how a payload closure is
interpreted:

```text
RepresentationFormatDescriptor {
  format_ref: RepresentationFormatDescriptorRef
  implementation_semantic_identity
  admitted root variant and stage set
  exact payload role, media-type, and cardinality contract
  canonical locator grammar
  index(logical payload bytes) -> owner-typed RepresentationIndex
  lookup(RepresentationIndex, RepresentationLocator)
    -> RepresentationObjectFacts
  unresolved_external_definitions(RepresentationIndex)
    -> canonical array<RepresentationLocator<Module | Cell>>
}

RepresentationObjectFacts {
  object_kind: RepresentationObjectKind
  signal_geometry?: {
    direction: Input | Output | Inout
    bit_width: positive uint64
  }
}
```

The initial registry identity is
`loom.hardware_representation_format`, version `1.0`. Its exact reference bytes
are `u64be(identity length) || identity bytes || u32be(major) || u32be(minor) ||
u32be(format kind)`. Existing format kinds retain their numeric meaning; an
incompatible indexer, object-fact, or locator contract requires a new
descriptor version.
A canonical JSON reference is exactly the object fields `registry`, `major`,
`minor`, and `kind` in that order, with the registry string above and canonical
unsigned integers.
A MIME string, filename suffix, tool name, or caller parser cannot replace this
reference.

Registry 1.0 owns these initial format kinds:

| Kind | Stable spelling | Admitted root | Payload contract |
| ---: | --- | --- | --- |
| 0 | `systemverilog_rtl` | `Rtl` | one or more `RtlSource`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| 1 | `structural_verilog_gate_netlist` | `GateNetlist` | one or more `Netlist`; zero or more `GenerationConstraint` and `BlackBoxContract` |

Both descriptors use the exact media-type spellings
`text/x-systemverilog; charset=utf-8` for `RtlSource`,
`text/x-verilog; charset=utf-8` for `Netlist`,
`application/x-sdc; charset=utf-8` for `GenerationConstraint`, and
`application/vnd.loom.black-box-contract` for `BlackBoxContract`. Text payloads
use LF line endings, contain no NUL byte, and cannot depend on an ambient
include path, command-line macro, or library search order. Source or netlist
units are compiled in canonical logical-name order. An unresolved external
definition is accepted only when the complete HardwareImplementation closes it
through an exact black-box contract and external implementation binding.

Their canonical locator grammar uses an unescaped HDL identifier
`[A-Za-z_][A-Za-z0-9_$]*` and a nonempty `.`-separated path of such identifiers.
A top `Module` locator is one identifier. Instance and contained-object
locators are top-rooted paths, and a `Port` or `Pin` appends the exact terminal
signal identifier. Escaped identifiers, ambient generate-name inference, and
filename-derived module names are outside these two descriptors. Another
grammar requires another exact format reference.

The `systemverilog_rtl` indexer parses and elaborates the complete source set
without ambient inputs. The `structural_verilog_gate_netlist` indexer also
rejects behavioral processes and timing controls. Both return the exact object
kind for every admitted locator, exact direction and bit width for every
`Port` or `Pin`, and the complete canonical unresolved external-definition
inventory.

Indexing reads payload bytes only through BlobStore and is pure: it cannot
execute a tool, inspect a workdir, or use a local path. The returned index is a
removable owner-typed value, not a persistent payload or second object catalog.
The HardwareImplementation finalizer uses the exact descriptor to resolve the
top and every stored locator, then independently derives expected interface
direction, width, and protocol facts from the exact Fabric and
ConfigurationABI and compares them with the indexed representation facts. The
finalizer also requires every indexed unresolved definition to be closed by an
exact black-box contract and external implementation binding, and rejects a
binding that closes no indexed definition. The format descriptor does not
redefine Fabric semantics, and the finalizer does not parse a format
independently.

A descriptor for an otherwise opaque vendor database must require a canonical
provider-produced index within its declared logical payload closure and return
the same typed object facts from that index. Until such a descriptor and
indexer exist, that representation is typed `Unsupported` rather than a blob
plus an unverified top claim. New format kinds are allocated only by this
registry; provider registration cannot assign a private meaning to an existing
kind.

This closed root, rather than a flat tag plus an independently interpreted
payload bag, is the sole representation authority. Its variant owns the root
locator, admitted payload roles, and required cardinalities:

| Variant | Complete allowed payload-role catalog |
| --- | --- |
| `Rtl` | one or more `RtlSource`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| `GateNetlist` | one or more `Netlist`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| `AsicPhysical::Placed` | one or more `PhysicalDatabase`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| `AsicPhysical::Routed` | one or more `PhysicalDatabase`; zero or more `LayoutStream`, `GenerationConstraint`, and `BlackBoxContract` |
| `AsicPhysical::Extracted` | one or more each of `PhysicalDatabase` and `Parasitics`; zero or more `LayoutStream`, `GenerationConstraint`, and `BlackBoxContract` |
| `FpgaPhysical` | one or more `PhysicalDatabase`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| `FpgaImage` | exactly one `DeviceImage` |

`GenerationConstraint` and `BlackBoxContract` payloads are present exactly
when required to reconstruct or consume that represented state. Any role or
cardinality outside the selected row is invalid. The selected format descriptor
may strengthen, but never relax, its row. Its derived index lets the
HardwareImplementation finalizer validate the root locator and every interface,
activity, external-binding, and memory-binding locator against the exact
logical payload closure. The root cannot be inferred from a filename, first
parsed module, tool default, or report.

The closed variant identifies the semantic implementation state represented
by its payload closure. It is not a linear mandatory pipeline: a selected flow
may omit forms it does not materialize. A generic stage string, flat payload
catalog outside this root, or bag of optional format fields is forbidden.

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

The displayed order fixes stable role tags `0` through `7`. Each payload
records a closed role, canonical logical name, and BlobDigest; the selected
representation-format descriptor owns its exact media type and parser. The
artifact contains every payload required to reconstruct or consume the
represented implementation state. Backend logs, reports, waveforms, temporary
databases, and tool caches remain raw attempt bundles unless they are one of
these semantic implementation payloads.

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

The displayed order fixes stable representation-object tags `0` through `9`.
Every locator encodes its `u32be` object-kind tag followed by the
`u64be`-length-framed canonical-name bytes. Locator arrays are sorted by these
canonical bytes and reject duplicates.

The interface catalog binds Fabric-visible boundaries, clocks, resets,
configuration transports, memories, and external protocols to exact
implementation locators. The activity catalog is the sole implementation-
local source for RTL, netlist, physical, and FPGA activity references used by
simulation or Evaluation.

The enclosing representation and exact format descriptor give every locator
its representation-local interpretation; a locator therefore does not repeat
a representation tag.
`canonical_name` is the stable name within that exact represented state, not a
filesystem path, report path, tool query, or Fabric entity name. The closed
object kind prevents a port, net, cell, pin, physical object, and device
resource from becoming interchangeable strings. A locator kind incompatible
with the enclosing representation is invalid.

Locators do not alter Fabric or Mapping identity. `device_pin_ref` is valid
only for an FPGA representation with an exact FPGA target manifest. A missing
required interface or activity point makes the artifact incomplete.

Interfaces sort by their complete canonical records and activity points sort
by `(representation_locator, optional semantic_fabric_ref)`. Both catalogs
reject duplicate records and duplicate locators whose roles would be
ambiguous. Their dense ordinals are derived only after sorting; no caller-
authored interface or activity ID enters identity.

The schema-2.0 owner-local reference catalog is:

```text
0  HardwareImplementationInterfaceRef
1  HardwareImplementationActivityPointRef
2  ExternalImplementationBindingRef
```

Each local payload is one `u64be` dense ordinal into the corresponding
canonical catalog. A complete cross-artifact reference uses the Common exact
HardwareImplementation Artifact identity plus this owner-local kind and
payload. Strict decoding rejects an unknown kind, out-of-range ordinal,
noncanonical catalog, or a target whose enclosing representation does not
admit its locator.

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
  provider_contract_ref
  external_inputs: canonical nonempty catalog<ExternalInputBinding>
  fabric_resource_refs: canonical set<FabricPhysicalOccurrenceOwnerRef>
  representation_locators[]
  black_box_contract_payload_ref?
}

ExternalImplementationBindingRef =
  dense owner-local ordinal in canonical binding-key order

MemoryMacroBinding {
  fabric_memory_ref: FabricPhysicalOccurrenceOwnerRef refined to memory
  external_implementation_binding_ref
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

An external binding has no caller-authored ID. Its canonical key is the exact
tuple of provider contract, external-input catalog, occurrence-qualified Fabric
relations, representation locators, and optional black-box payload reference.
Finalization sorts and deduplicates complete keys, then assigns dense
owner-local ordinals. `MemoryMacroBinding` and every other internal reference
use only that derived ordinal. Authoring order, a display label, and a stale or
sparse supplied number cannot become binding identity.

Within each binding, `fabric_resource_refs` and `representation_locators` are
canonical sorted-unique arrays. `black_box_contract_payload_ref`, when present,
must resolve to an `ImplementationPayloadRef` in the same root whose role is
exactly `BlackBoxContract`. A memory-macro locator must pass the same selected
representation-format index/lookup and HardwareImplementation finalizer
cross-check as every other locator.

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
the optional target manifest, the closed representation root, exact format
descriptor, payload roles, BlobStore bytes and digests, interfaces, activity
points, memory-macro bindings, and external bindings. It resolves every
provider contract, validates each external dependency identity, verifies every
Fabric-resource relation, builds the selected descriptor's pure index over the
exact logical payload closure, resolves and cross-checks the complete locator
set, and rejects an external module without its required black-box contract.
It also verifies that the represented state has no implicit dependency on a
parent implementation or generator invocation. Canonical ordering uses typed
semantic keys. Filesystem
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

For any downstream producer, the authoritative implementation bytes are only
the logical bytes returned by `BlobStore` for the `BlobDigest` references in
the exact input's `representation_root`. The producer rehashes those bytes
before materialization. A previous invocation directory, declared output path,
report path, vendor database path, or caller-supplied duplicate source is not a
production input. Explicit PDK, cell, macro, and user-IP dependencies retain
their existing external-binding identities; this rule does not import their
installation trees into BlobStore.

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
* stable root, stage, payload-role, and object-kind tags, payload logical-name
  normalization, dense local-reference round trips, and one known canonical
  root vector;
* variant-specific representation-root locator and payload cardinality, with a
  flat or inferred top rejected;
* missing representation-format providers, wrong-format payloads, locators
  absent from the exact logical representation, and an opaque database without
  its descriptor-required canonical index;
* required interface, activity-point, memory-macro, and configuration-
  transport coverage;
* explicit-file and tool-bundled external identities producing distinct
  bindings without importing their installation trees;
* authoring-order-independent dense external-binding references and rejection
  of a supplied, sparse, duplicate, or stale binding ordinal;
* a memory macro selecting one exact external binding and rejection of a
  platform-owned macro-file lookup;
* ASIC and FPGA representation variants under the same family; and
* downstream materialization accepting only BlobStore-verified bytes from the
  exact representation root and rejecting a substituted source or work path;
* completed adverse timing evidence versus a tool failure that publishes no
  HardwareImplementation.

Tests do not freeze vendor report text, Tcl formatting, database directory
layout, every EDA tool, or a large format-conversion matrix.
