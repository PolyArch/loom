# LoomBench

This document specifies `loombench`, Loom's repository-owned long-term
benchmark family. `loombench` starts from kernels migrated from the
previous DSA-oriented corpus and may grow with new Loom-specific
benchmarks.

## Purpose

`loombench` answers this question:

```text
Can Loom evaluate architecture-aware compiler, mapping, simulation,
RTL/FPA, and DSE behavior on a stable benchmark family that is owned by
the project and independent of external CMSIS source trees?
```

`loombench` is distinct from the self-contained app drop-in corpus in
`docs/spec-app-dropin-test-corpus.md`. The app corpus validates ordinary
drop-in compiler behavior on small standalone C/C++ programs.
`loombench` validates accelerator-oriented benchmark coverage and DSE
comparability.

`loombench` is also distinct from CMSIS-DSP and CMSIS-NN. CMSIS tests
validate compatibility with public external APIs and source-tree
conventions. `loombench` can use Loom-owned harnesses, metadata, and
taxonomy tags as long as the benchmark semantics remain stable.

## Target Universe

The `loombench` target universe includes:

* kernels migrated from the prior DSA-oriented corpus;
* Loom-owned kernels added to cover missing graph, memory, control,
  streaming, spatial, temporal, or heterogeneous-system behaviors;
* benchmark metadata that maps every case into the unified workload
  taxonomy used by app, CMSIS, reports, and DSE;
* deterministic inputs and reference-oracle behavior;
* validation tiers from native or drop-in compile/run through RTL/FPA
  evidence;
* feature tags for graph shape, memory behavior, data type, control
  behavior, expected accelerator pressure, and required hardware
  capabilities.

Every accepted benchmark must end in one of these states for each
validation tier:

* `pass` with evidence artifacts;
* `fail` with diagnostics;
* `unsupported` with a structured unsupported-scope record;
* `blocked` with a missing external capability or unavailable profile
  record.

Silent exclusion, empty-run success, or missing-file success is invalid.

## Required Evidence

Each `loombench` case must provide:

* stable benchmark identity;
* source or generated workload artifact identity;
* input data identity;
* reference-oracle identity;
* workload taxonomy tags;
* tier support declarations;
* output artifacts for each passing tier;
* structured diagnostics for unsupported, failed, or blocked tiers.

For simulator and DSE tiers, reports must identify the selected hardware
candidate, mapping artifact, runtime input, feedback fidelity, and metric
provenance.

## Objective Verification

The target is verifiable when:

* a manifest or equivalent structured index can enumerate every
  `loombench` case;
* each case has deterministic inputs and reference-oracle behavior;
* runner outputs distinguish pass, fail, unsupported, and blocked;
* every passing simulator row includes functional output and memory-diff
  evidence when the selected tier requires it;
* DSE report bundles can aggregate `loombench` cases without treating
  them as app or CMSIS cases.

## Unsupported Scope Policy

Unsupported `loombench` tiers must name the benchmark, tier, missing
capability, selected profile, and diagnostic class. A benchmark may be
accepted before every tier is implemented only when the unsupported
tiers are explicit and auditable.

## Relationships To Other Contracts

`loombench` uses the validation ladder defined by
`docs/spec-app-dropin-test-corpus.md` and the report and artifact
contracts in `docs/spec-full-stack-reporting.md` and
`docs/spec-intermediate-artifacts.md`. It participates in DSE through
`docs/spec-dse-feedback.md`.

Compiler behavior still follows the source, raise, and dataflow specs.
Hardware behavior still follows the Fabric specs. `loombench` does not
create benchmark-specific compiler or hardware semantics.

## Current Implementation Notes

This section is non-normative. It records current repository facts for
orientation only and is not part of target acceptance.

The current repository has app and CMSIS runner infrastructure. The
dedicated `loombench` benchmark family is a target contract and still
needs manifest, runner, and report integration.
