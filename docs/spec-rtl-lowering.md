# RTL Lowering

This document specifies Loom RTL lowering from Fabric hardware
descriptions to synthesizable and simulatable SystemVerilog artifacts.

## Purpose

RTL lowering answers this question:

```text
Can this Fabric hardware description be emitted as concrete RTL that
preserves Fabric ADG structure and can be simulated or evaluated by
backend tools?
```

RTL lowering consumes:

* Fabric ADG;
* optional mapping artifact for workload-specific configuration or
  testbench generation;
* optional RTL lowering configuration;
* optional tool and library profile references specified in
  `docs/spec-eda-tooling.md`.

RTL lowering produces:

* SystemVerilog source files;
* package and interface files;
* top-level module or harness files;
* optional testbench files;
* optional constraints manifest;
* RTL manifest;
* diagnostics.

## Boundary With Fabric ADG

Fabric ADG is the hardware architecture source of truth. RTL lowering
must not invent hardware topology absent from Fabric ADG. It may lower typed
architecture concepts into concrete implementation structures only through
Fabric-owned implementation refinement. The exact system-level lowering
surface remains open until the typed `fabric.system` schema and its
Interconnect Implementation contracts are closed.

RTL lowering must preserve:

* exact Fabric artifact and typed entity identity;
* typed endpoint direction, payload, and service contracts;
* explicit Transport Architecture resources and directed connectivity;
* selected Interconnect Implementation refinements;
* external boundaries as intentional top-level hardware interfaces;
* clock, reset, power, address, coherence, consistency, and protection
  domains; and
* typed memory and service capabilities.

Fabric visualization metadata must not affect emitted RTL.

## Architecture RTL And Mapped RTL

Loom distinguishes architecture RTL from mapped-workload RTL.

Architecture RTL is generated from Fabric ADG alone. It represents the
hardware design and can be reused across workloads.

Mapped-workload RTL may additionally consume a mapping artifact. It may
emit configuration packages, initialization data, testbenches, or
workload-bound harnesses. A mapping artifact must not create new Fabric
resources, endpoints, connections, attachments, services, or refinements; it
only configures or exercises hardware already described by Fabric ADG.

Architecture RTL is the first architectural target because it validates
the reusable hardware description independent of any one workload.
Mapped-workload RTL remains part of the same target universe, and its
manifest schema must be defined alongside architecture RTL so later
workload-bound harnesses can be added without changing the manifest
identity model.

## SystemVerilog Structure

The generated RTL source set should use deterministic labels derived from
canonical Fabric artifact and entity identities. Emitted names are not
hardware identity.

Baseline RTL structure includes:

* one top-level module for a selected `fabric.system` or
  `fabric.module`;
* module definitions for reusable typed resources or SpatialCore templates;
* ports, interfaces, and adapters required by the selected Interconnect
  Implementation refinement;
* signals for typed directed endpoint connectivity;
* packages for type, parameter, and implementation definitions;
* optional harnesses for simulation and backend checks.

Protocol-specific interfaces or grouped signals may be emitted only when the
selected Interconnect Implementation refinement requires them. A protocol name
or generated interface does not replace the architecture-level endpoint,
transport, or service contract.

## Connectivity Lowering

Each typed directed connection and selected refinement lowers to explicit RTL
connectivity and implementation resources. Replication, arbitration,
adaptation, and temporal sharing may be emitted only when the Fabric
architecture and selected refinement require them. RTL lowering must not infer
those behaviors from fanout, naming, or topology shape.

If a required typed resource or implementation refinement lacks a supported
RTL realization, lowering emits a diagnostic instead of silently replacing
the behavior.

Clock-domain crossings must be implemented only when Fabric ADG contains a
legal typed crossing and selected implementation. Same-domain connectivity
must not acquire crossing logic implicitly.

## Clock, Reset, And Power

Clock and reset ports are emitted for the effective domains required by the
selected hardware root. Domain membership and crossing behavior come only from
typed Fabric facts. RTL lowering must not invent a default domain unless the
closed Fabric schema defines one.

Reset-domain and power-domain metadata may produce:

* reset ports;
* reset synchronizer hooks when explicitly represented;
* power intent hooks;
* clock or power gating hooks;
* backend constraint metadata.

Power intent generation may be partial, but missing support must be
reported explicitly. Silent omission of required domain behavior is not
allowed.

## Memory And Interconnect Implementation

RTL lowering must preserve typed memory and service capabilities, physical
address spaces, external memory boundaries, cache facts, coherence domains,
consistency guarantees, and the selected Interconnect Implementation at the
supported abstraction level.

The initial target may emit behavioral memory models or black-box
memory wrappers when concrete macro implementations are not selected.
The RTL manifest must identify behavioral models, black boxes, and
library-bound instances separately.

## Verification Hooks

RTL lowering should emit hooks that allow later tools to check:

* syntax and elaboration;
* basic connectivity;
* typed endpoint and service compatibility;
* Transport Architecture and implementation-refinement closure;
* clock-domain crossing coverage;
* reset connectivity;
* top-level external interface shape;
* optional testbench execution.

The RTL path may be checked against CGRA-sim behavior for mapped
workloads, but RTL lowering does not replace CGRA-sim and CGRA-sim does
not replace RTL validation.

## Activity Hooks

Generated RTL should provide an activity-name map from emitted hierarchy and
signals to canonical Fabric identities. RTL simulation may produce waveform
or switching activity artifacts. Those artifacts can be consumed by FPA
estimation through the contract in `docs/spec-fpa-estimation.md`.

CGRA-sim activity reports may also feed FPA estimation before RTL
activity is available. RTL activity and CGRA-sim activity are separate
evidence sources and must be labeled as such.

## RTL Manifest

An RTL manifest records:

* manifest schema version;
* manifest mode, either `architecture_rtl` or `mapped_workload_rtl`;
* source Fabric ADG identity;
* optional mapping artifact identity, required only for
  `mapped_workload_rtl`;
* lowering configuration;
* emitted source files;
* top-level module names;
* generated packages and interfaces;
* black-box modules;
* behavioral models;
* required tool capability classes;
* required library profile classes;
* constraints and activity hooks;
* diagnostics.

The manifest is the stable input to EDA tooling profiles.

`architecture_rtl` manifests must not pretend to have workload mapping
evidence. `mapped_workload_rtl` manifests must identify the mapping
artifact consumed for configuration, initialization, or harness
generation, and that mapping artifact must resolve against the same
Fabric ADG identity.

## Non-Goals

RTL lowering is not PnR. It does not choose software-to-hardware
mapping.

RTL lowering is not CGRA-sim. It emits hardware artifacts rather than
simulating mapped execution.

RTL lowering is not FPA estimation. It may emit artifacts consumed by
FPA tools, but it does not by itself produce final frequency, power, or
area reports.

RTL lowering is not a place-and-route backend. Physical implementation
may consume the RTL artifacts later through EDA tooling profiles.

## Acceptance Criteria

RTL lowering is complete at the target-spec level when:

* a selected Fabric hardware root can emit deterministic SystemVerilog
  sources and an RTL manifest;
* the RTL manifest declares `architecture_rtl` or
  `mapped_workload_rtl` mode and enforces the corresponding mapping
  artifact rule;
* typed Fabric connectivity lowers to explicit RTL connectivity without
  inventing replication or arbitration;
* external boundaries become intentional top-level interfaces;
* clock-domain crossing logic is emitted only for explicit crossing
  constructs;
* missing typed resource implementations or unsupported implementation
  refinements produce structured diagnostics;
* emitted RTL can be checked by at least one configured `rtl_lint`,
  `rtl_sim`, or `rtl_synth` profile;
* emitted manifests can feed FPA estimation through
  `docs/spec-fpa-estimation.md`.
