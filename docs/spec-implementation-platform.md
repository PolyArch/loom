# Implementation Platform

This document defines immutable technology inputs used to create a
HardwareImplementation. It is distinct from Fabric architecture and from a
runtime binding to one installed device.

## Artifact Family

```text
loom.implementation_platform 1.0
```

The family owns the exact Common descriptor:

```text
ArtifactSchemaDescriptor {
  identity = "loom.implementation_platform"
  version = {1, 0}
}
```

The schema descriptor is declared once by the C++ owner and is reused by the
root codec, Artifact store, typed references, and every consumer. A caller may
not reconstruct it from a string or maintain a second version constant.
The family has one typed `ImplementationPlatform` C++ root, one canonical
serializer/parser pair, one read-only importer view, and one validator. The
parser rejects unknown fields and closed-variant values; successful parse is
not validation or publication.

```text
ImplementationPlatform {
  technology_corners: canonical nonempty catalog<TechnologyCorner>
  platform:
      AsicPlatform {
        technology_identity
        standard_cell_libraries[]
        rc_corners[]
        technology_memory_macros[]
        physical_rule_payloads[]
      }
    | FpgaPlatform {
        vendor
        family
        part
        package
        speed_grade
        primitive_library_payloads[]
        timing_payloads[]
        package_pins[]
      }
}

TechnologyCorner {
  corner_id: TechnologyCornerId
  model_inputs: canonical nonempty set<TechnologyCornerModelInput>
}

TechnologyCornerModelInput {
  role: TechnologyCornerModelInputRole
  payload: exact BlobDigest in this platform's immutable payload closure
}

TechnologyCornerModelInputRole =
    StandardCellTiming
  | MemoryTiming
  | IoTiming
  | FpgaTiming
```

All references and payloads are exact and content-addressed. A path, shell
module name, license server, workstation installation, or vendor environment
variable is an invocation binding and does not enter platform identity.

`TechnologyCorner` identifies one immutable process/timing-model selection.
Voltage, temperature, required clock, RC extraction choice, activity, and tool
effort are not corner fields; they remain their existing typed Evaluation or
implementation-flow inputs. More than one input may have the same role, for
example several standard-cell libraries, and the complete `(role, digest)` set
is the corner's semantic key. An ASIC corner contains at least one
`StandardCellTiming` input. An FPGA corner contains at least one `FpgaTiming`
input. A role incompatible with the platform variant is invalid.

## Technology Corner References

`ImplementationPlatform` is the sole owner of technology-corner local
identity. There is no sibling Technology Artifact family.

```text
TechnologyCornerId = unsigned 64-bit owner-local ordinal
TechnologyCornerRef = ArtifactReference<TechnologyCornerId>

ImplementationPlatformLocalReferenceKind {
  TechnologyCorner = 0
}
```

Finalization sorts corners by their complete semantic model-input key, rejects
duplicates, and assigns dense `TechnologyCornerId` values in `[0, N)`. IDs are
not author labels, payload ordinals, hashes, or reusable across platform
Artifacts. Namespace exhaustion fails before identity generation.

The family-owned existential-reference codec for local kind
`TechnologyCorner` is exactly `u64be(corner_id)`. Its payload is therefore
exactly eight bytes; canonical JSON carries those bytes as exactly sixteen
lowercase hexadecimal characters. Decoding any other length is invalid. The
owner validator requires the referenced Artifact to have the exact
`loom.implementation_platform 1.0` schema, independently validates its root,
and resolves the ID to exactly one catalog entry. Evaluation and EDA adapters
invoke this codec and validator; they never reinterpret the ID or erase an
arbitrary `ArtifactReference<T>` to a bare integer.

## ASIC Platform

An ASIC platform owns the exact cell-library, timing-corner, extraction-corner,
design-rule, and generated or fixed memory-macro contracts used by hardware
implementation. One typed memory-macro entry owns logical ports, widths,
depths, masks, clocks, latency, timing models, physical views, and payload
digests required by the selected provider.

HardwareImplementation binds a Fabric memory occurrence to one compatible
macro entry. It does not copy macro facts. Mapping and Fabric cannot infer a
macro from a filename or memory dimensions alone.

## FPGA Platform

An FPGA platform owns the exact part, package, speed grade, primitive library,
timing data, and package-pin catalog. DSP, block-memory, routing, and clocking
resources are implementation capabilities exposed by typed provider bindings,
not by parsing vendor report strings.

Application pin assignment, clock placement, and other design choices belong
to the resolved hardware generator configuration and resulting
HardwareImplementation payloads. The platform owns the legal device universe,
not one workload's selection.

## Runtime Boundary

An ImplementationPlatform is a design-time technology target. Runtime device
enumeration, installed-device identity, transport handles, leases, and actual
addresses are owned by the Runtime ABI and its `RuntimePlatformBinding`.
Deployment may require both an exact HardwareImplementation and an exact
RuntimePlatformBinding. It does not treat this design-time platform as proof
that a particular device instance is present.

## Finalization And Versioning

The platform root contains one closed variant, the technology-corner catalog,
and exact direct payload digests. Canonical ordering uses typed library,
corner-model-input, RC-corner, macro, primitive, and pin keys. Duplicate keys,
duplicate semantic technology corners, non-dense corner IDs, unresolved or
out-of-closure corner payloads, variant-incompatible corner roles,
inconsistent RC data, invalid macro port contracts, or an incomplete FPGA
identity fail finalization.

Canonical semantic bytes are canonical JSON with exact BlobDigest references.
The schema descriptor is supplied to Common framing and is not copied into the
root. Finalization independently reimports the root, validates every referenced
payload and typed catalog relation, verifies canonical corner ordering and ID
assignment, and publishes atomically. Vendor-native database paths and
installation manifests are not parallel platform roots.

The owner codec uses fixed root field order and registered enum spellings.
Corner model inputs sort by `(role discriminant, BlobDigest bytes)`; corners
sort by the complete framed model-input sequence before dense IDs are assigned.
All other sets and catalogs use their typed complete semantic keys. A decoder
must re-encode to exactly the supplied canonical bytes; permissive JSON parsing
cannot admit a second spelling of the same platform.

Schema versions follow the common `X.Y` rule. Updating a process, cell library,
memory compiler output, FPGA part, package, speed grade, or semantic payload
creates a new platform identity. Runtime environment or license changes do
not.

## Anchor Verification

Anchor tests cover:

* one ASIC platform with two distinct technology corners and one typed memory
  macro;
* one FPGA platform with one technology corner and exact
  part/package/speed-grade identity;
* the fixed `TechnologyCornerRef` eight-byte known vector and typed
  encode/decode/validate round-trip;
* rejection of a wrong Artifact schema, wrong local-reference kind,
  noncanonical payload length, out-of-range corner ID, and duplicate corner;
* memory-macro and primitive binding compatibility;
* payload corruption and duplicate-key rejection; and
* identical semantic inputs under different filesystem layouts producing the
  same identity.

Tests do not freeze vendor install paths, license configuration, shell module
names, every process corner, or every FPGA device.
