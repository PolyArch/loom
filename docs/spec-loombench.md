# LoomBench

This document specifies `loombench`, Loom's repository-owned benchmark
suite under `test/app`. `test/app/manifest.json` is the authoritative
inventory and currently lists 133 cases.

## Purpose

`loombench` answers this question:

```text
Can Loom evaluate architecture-aware compiler, mapping, simulation,
RTL/FPA, and DSE behavior on a stable benchmark family that is owned by
the project and independent of external CMSIS source trees?
```

`loombench` includes the self-contained drop-in corpus described in
`docs/spec-app-dropin-test-corpus.md`. CMSIS-DSP and CMSIS-NN are the
other two canonical suites. CMSIS tests
validate compatibility with public external APIs and source-tree
conventions. `loombench` can use Loom-owned harnesses, metadata, and
taxonomy tags as long as the benchmark semantics remain stable.

## Target Universe

The `loombench` target universe includes:

* Loom-owned kernels that cover graph, memory, control,
  streaming, spatial, temporal, or heterogeneous-system behaviors;
* benchmark metadata that maps every case into the unified workload
  taxonomy used by LoomBench, CMSIS, reports, and DSE;
* deterministic inputs and reference-oracle behavior;
* validation tiers from native or drop-in compile/run through RTL/FPA
  evidence;
* feature tags for graph shape, memory behavior, data type, control
  behavior, expected accelerator pressure, and required hardware
  capabilities.

Manifest tiers select the cases exercised by a compiler or hardware path.
A case that is not selected for a downstream tier remains part of the
canonical inventory and must not be reported as having passed that tier.

## Required Evidence

Each `loombench` case must provide:

* stable benchmark identity;
* source identity;
* deterministic input data;
* reference-oracle identity;
* workload taxonomy tags;
* tier support declarations;

When a simulator, mapping, RTL, or DSE tier is run, its report must identify
the selected workload, hardware candidate, runtime input, and metric
provenance relevant to that tier.

## Objective Verification

The target is verifiable when:

* a manifest or equivalent structured index can enumerate every
  `loombench` case;
* each case has deterministic inputs and reference-oracle behavior;
* all cases compile and run with both the baseline compiler and Loom's
  drop-in drivers;
* representative source-to-IR and hardware-aware paths fail on missing or
  invalid artifacts rather than passing an empty run;
* every passing simulator row includes functional output and memory-diff
  evidence when the selected tier requires it;
* DSE report bundles aggregate `loombench` cases without changing their
  suite identity.

## Unsupported Scope Policy

Diagnostics emitted for an attempted unsupported `loombench` tier must
satisfy the Unsupported Scope Policy in `docs/spec-loom-stack.md`.
Unselected tiers do not require parallel status records.

## Relationships To Other Contracts

`loombench` follows the global workload/evidence policy and
unsupported-scope policy in `docs/spec-loom-stack.md`. Current mapping
and simulation artifact formats are described in
`docs/spec-intermediate-artifacts.md`.

Compiler behavior still follows the source, raise, and dataflow specs.
Hardware behavior still follows the Fabric specs. `loombench` does not
create benchmark-specific compiler or hardware semantics.

## Canonical Identity

All status and report producers emit exactly one of these source-suite
identities: `loombench`, `cmsis-dsp`, or `cmsis-nn`. A `test/app` case
emits `suite=loombench` directly. Every emitted `(suite, case)` pair must
be unique.
