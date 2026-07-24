# Implementation Platform

This document defines immutable technology inputs used to create a
HardwareImplementation. It is distinct from Fabric architecture and from a
runtime binding to one installed device.

## Artifact Family

```text
loom.implementation_platform 1.0
```

```text
ImplementationPlatform =
    AsicPlatform {
      technology_identity
      standard_cell_libraries[]
      timing_corners[]
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
```

All references and payloads are exact and content-addressed. A path, shell
module name, license server, workstation installation, or vendor environment
variable is an invocation binding and does not enter platform identity.

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

The platform root contains one closed variant and exact direct payload
digests. Canonical ordering uses typed library, corner, macro, primitive, and
pin keys. Duplicate keys, unresolved payloads, inconsistent corners, invalid
macro port contracts, or an incomplete FPGA identity fail finalization.

Canonical semantic bytes are canonical JSON with exact BlobDigest references.
Finalization independently reimports the root, validates every referenced
payload and typed catalog relation, and publishes atomically. Vendor-native
database paths and installation manifests are not parallel platform roots.

Schema versions follow the common `X.Y` rule. Updating a process, cell library,
memory compiler output, FPGA part, package, speed grade, or semantic payload
creates a new platform identity. Runtime environment or license changes do
not.

## Anchor Verification

Anchor tests cover:

* one ASIC platform with two corners and one typed memory macro;
* one FPGA platform with exact part/package/speed-grade identity;
* memory-macro and primitive binding compatibility;
* payload corruption and duplicate-key rejection; and
* identical semantic inputs under different filesystem layouts producing the
  same identity.

Tests do not freeze vendor install paths, license configuration, shell module
names, every process corner, or every FPGA device.
