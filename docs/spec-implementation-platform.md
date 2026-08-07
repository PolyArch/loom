# Implementation Platform

This document defines the minimal immutable design target shared by hardware
generation and Evaluation. An ImplementationPlatform is a target manifest. It
is not a PDK archive, standard-cell catalog, IP repository, FPGA device
database, tool installation, board instance, or invocation environment.

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

The descriptor is declared once by the C++ owner and reused by the root codec,
Artifact store, typed references, and every consumer. A caller may not
reconstruct it from a string or maintain a second version constant.

```text
ImplementationPlatform {
  target:
      AsicTarget {
        technology_identity
        release_identity
      }
    | FpgaTarget {
        vendor: AmdXilinx | IntelAltera
        device_ordering_code
      }
  technology_corners: canonical nonempty catalog<TechnologyCorner>
}

TechnologyCorner {
  corner_id: TechnologyCornerId
  corner_key
}
```

`technology_identity`, `release_identity`, `device_ordering_code`, and
`corner_key` are nonempty, case-sensitive canonical ASCII identifiers. They
contain only letters, digits, `.`, `_`, `-`, `+`, `:`, and `/`, begin and end
with an alphanumeric character, and are never Unicode-normalized or
case-folded. The complete target variant and identifier bytes are semantic.
Display labels and local aliases are not.

The ASIC pair identifies one exact technology release admitted by a provider.
It does not assert that any particular logical, timing, physical, RC, rule, or
macro view is locally available. The FPGA ordering code is the single exact
part identity accepted by the selected vendor provider; family, package, and
speed-grade substrings are not copied into parallel semantic fields.

## Technology Corner References

`ImplementationPlatform` is the sole owner of technology-corner local
identity. There is no sibling Technology Artifact family and no
Evaluation-owned corner string.

```text
TechnologyCornerId = unsigned 64-bit owner-local ordinal
TechnologyCornerRef = ArtifactReference<TechnologyCornerId>

ImplementationPlatformLocalReferenceKind {
  TechnologyCorner = 0
}
```

Finalization sorts corners by `corner_key`, rejects duplicate keys, and assigns
dense `TechnologyCornerId` values in `[0, N)`. IDs are not author labels,
hashes, tool ordinals, or reusable across platform Artifacts. Namespace
exhaustion fails before identity generation.

The family-owned existential-reference codec for local kind
`TechnologyCorner` is exactly `u64be(corner_id)`. Its payload is exactly eight
bytes; canonical JSON carries those bytes as exactly sixteen lowercase
hexadecimal characters. Decoding any other length is invalid. The owner
validator requires the referenced Artifact to have the exact
`loom.implementation_platform 1.0` schema, independently validates its root,
and resolves the ID to exactly one catalog entry. Evaluation and EDA adapters
invoke this codec and validator; they never reinterpret the ID or erase an
arbitrary `ArtifactReference<T>` to a bare integer.

A corner key names one target-local process or vendor timing corner. Voltage,
temperature, required clock, activity, RC extraction choice, analysis mode,
tool effort, and tool-specific library spelling are not corner fields. They
remain typed Evaluation conditions or provider-owned generator/evaluator
configuration. A provider binding maps one exact `TechnologyCornerRef` to the
specific models it consumes without redefining the corner.

## Target-Independent RTL

An ImplementationPlatform is optional for a technology-independent RTL
HardwareImplementation. Portable RTL can be generated, linted, and simulated
without inventing an ASIC process or FPGA part.

A target-bound RTL variant references an exact ImplementationPlatform when its
materialized state depends on that target. Gate netlists, ASIC physical states,
FPGA physical states, and FPGA images always reference the exact target
manifest they implement. Tool or library dependence alone does not make a
portable RTL implementation target-bound; the exact external dependency is
recorded through its provider-owned HardwareImplementation binding.

## External Technology Inputs

PDK files, standard-cell libraries, macro views, rule decks, user IP, and
other external bytes are not fields or payloads of ImplementationPlatform.
Each candidate-generator or evaluator descriptor owns closed typed input slots
for the exact external resources it consumes. Its resolved semantic binding
identifies each selected input by role and either:

* the exact content fingerprint of an explicitly supplied ordinary file; or
* an exact provider semantic identity and resource key for content shipped as
  part of a verified tool release.

The common platform schema does not define a generic library-role bag. A
provider cannot claim an undeclared input, infer a role from a filename, or
substitute a nearby view. Explicit external files are fingerprinted
individually. Loom does not recursively scan, import, copy, or hash a PDK,
vendor SDK, IP tree, or tool installation merely because a configured path
exists.

Machine-local paths are resolved through the explicit local configuration and
frozen into the ignored ExternalToolInvocationBundle. They do not enter the
ImplementationPlatform, resolved semantic binding, HardwareImplementation
identity, Evaluation Request, or tracked source. The bundle verifies every
selected explicit-file fingerprint before invoking a tool.

DesignWare, ChipWare, and Vivado or Quartus device-database resources are
tool-bundled resources. Their semantic identity is the provider-owned stable
tool/build identity plus the exact resource or device key. Loom does not
re-import or hash the complete tool installation tree. A provider whose public
version string is insufficient must supply a stronger stable build probe; a
mutable installation cannot be legitimized by a display version alone.

## Macro And User-IP Boundary

Fabric owns the required memory behavior, capacity, timing, ports, masks,
clocking, and progress semantics. A selected fixed or generated memory macro
is a provider-owned external implementation binding in the resulting
HardwareImplementation. That binding maps the exact Fabric memory occurrence
to its representation locator and exact macro contract. Logical, timing,
physical, and layout files are supplied only to the flow stages that declare
those inputs.

Other user IP follows the same boundary. Synthesizable source that becomes
part of the represented RTL is an explicit generator input and an `RtlSource`
payload of the HardwareImplementation. Encrypted or black-box IP is represented
by an exact `BlackBoxContract` and provider-owned external implementation
binding; its local bytes remain invocation material. User IP never becomes an
ImplementationPlatform field.

## Flow Admission

A valid ImplementationPlatform is a valid target manifest, not a claim that
every EDA flow is available. Each generator or evaluator descriptor separately
declares:

* accepted target variants and target identifiers;
* required exact technology-corner relations;
* required explicit external-file input slots;
* required tool-bundled resource slots; and
* compatibility between those inputs and the selected implementation state.

Admission validates the exact platform and resolved provider binding before
bundle preparation. A valid target with missing libraries, macro views, rule
decks, provider resources, or tool support is `Unsupported` or `Unavailable`
for that flow without invalidating the target manifest. No adapter renames a
technology, changes an FPGA part, substitutes another corner, or treats a
synthesis-only input set as physical or signoff closure.

This section owns target compatibility only. Central plan admission owns input
readiness, each Artifact family owns strict import, the Candidate Generator or
Evaluation descriptor owns flow-specific compatibility and consumption, and
the local invocation layer owns executable, runtime, and frozen-file preflight
as specified by `docs/spec-external-tool-invocation.md`. No layer restates
their union as a second total-admission authority, and none may treat a valid
Platform as proof supplied by another.

Initial ASIC conformance covers these independently configured target
identities:

* SAED 5 nm;
* SAED 14 nm;
* Samsung 4 nm; and
* Intel 18A.

These names are coverage requirements, not builtin Artifact identities,
filesystem aliases, or promises that proprietary inputs are installed. Each
real target instance supplies an exact technology and release identity.

Initial FPGA conformance covers these exact vendor ordering codes:

| Vendor generation | HBM-oriented target | DSP-oriented target |
| --- | --- | --- |
| AMD Versal | `xcvh1782-lsva4737-3HP-e-S` | `xcvp1802-vsva5601-3HP-e-S` |
| AMD Virtex UltraScale+ | `xcvu47p-fsvh2892-3-e` | `xcvu13p-flga2577-3-e` |
| Intel/Altera Agilex 7 | `AGMF039R47A1E1VC` | `AGIA040R39A1E1VC` |
| Intel/Altera Stratix 10 | `1SM21BHN1F53E1VG` | `1SG280HN2F43E2VG` |

These are tool/device-database targets, not claims that a physical board is
installed. Board clocks, connectors, application pin assignments, and measured
execution require an exact HardwareImplementation or RuntimePlatformBinding
contract. Adding another ordering code extends provider coverage without
changing the platform schema.

## Runtime Boundary

An ImplementationPlatform is a design-time target. Runtime device enumeration,
installed-device identity, transport handles, leases, and actual addresses are
owned by the Runtime ABI and its `RuntimePlatformBinding`. Deployment may
require both an exact HardwareImplementation and an exact
RuntimePlatformBinding. It does not treat the design-time target as proof that
a particular device instance is present.

## Finalization And Versioning

The root contains one closed target variant and one canonical nonempty corner
catalog. Canonical semantic bytes are canonical JSON with fixed field order and
the registered target discriminants `asic` and `fpga`. FPGA vendor spellings
are `amd_xilinx` and `intel_altera`. Identifiers use their exact validated ASCII
bytes. Corners sort by `corner_key` before dense IDs are assigned.

Duplicate identifiers, duplicate corner keys, non-dense corner IDs, invalid
identifier bytes, an unknown target or vendor discriminant, or a corner
reference outside the exact platform fail finalization. The schema descriptor
is supplied to Common framing and is not copied into the root. Finalization
independently reimports the root, validates the complete typed relation,
requires decode/re-encode byte equality, and publishes atomically.

Schema versions follow the common `X.Y` rule. Changing the target variant,
technology or release identity, FPGA ordering code, or corner catalog creates a
new platform identity. Changing a local path, module, license, host, or bundle
location does not. Changing an explicit external input or provider tool build
changes the exact resolved provider binding and any implementation or Evidence
that materializes that dependency; it does not mutate the target manifest.

## Repository Boundary

Repository eligibility for direct EDA attempts and results derived from them
is owned by the [EDA Tooling](spec-eda-tooling.md) disclosure boundary. This
specification defines no additional exception.

Real platform Artifacts, resolved provider bindings, local input fingerprints,
and proprietary platform inputs remain in ignored or external local storage.
The public repository may contain schemas, deterministic generators, adapters,
and small authored synthetic fixtures. It never tracks proprietary PDK,
library, macro, user-IP, tool-database, bundle, implementation, report, or
Evidence content.

## Anchor Verification

Anchor tests cover:

* one ASIC target with exact technology/release identity and two distinct
  technology corners;
* one FPGA target for each vendor with an exact ordering code and no copied
  family, package, or speed-grade fields;
* deterministic corner ordering and dense ID assignment;
* the fixed `TechnologyCornerRef` eight-byte known vector and typed
  encode/decode/validate round-trip;
* rejection of a wrong Artifact schema, wrong local-reference kind,
  noncanonical payload length, out-of-range corner ID, duplicate corner, and
  malformed target identifier;
* portable RTL admission without a platform and rejection of a target-bound
  state without one;
* provider admission using one explicit-file fingerprint and one tool-bundled
  resource identity without copying either into the platform root; and
* identical semantic targets under different local paths producing the same
  platform identity.

Tests do not freeze vendor installation paths, licenses, PDK directory layouts,
tool database contents, every corner, or every FPGA device.
