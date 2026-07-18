# Full-Stack Traceability

This document specifies the target traceability contract for Loom's
complete source-to-report flow. The contract exists so the implementation
can be built and verified without relying on chat history, hidden state,
or broad prose descriptions.

## Purpose

Full-stack traceability answers this question:

```text
For every artifact transition from C/C++ input to hardware-aware
reports, which tool produced the artifact, which tool consumes it, which
schema validates it, and which diagnostic is emitted when the transition
is invalid?
```

Traceability is not a user interface and not a build-system detail. It
is the target artifact contract that lets compiler tests, simulators,
PnR, RTL lowering, EDA adapters, reports, and DSE agree on the same
evidence chain.

## Trace Edge Model

A trace edge records one transition between two artifact kinds. Every
edge has these required fields:

* edge id;
* producer component;
* consumer component;
* producer artifact kind;
* consumer artifact kind;
* public spec owner;
* schema or verifier name;
* validation command role;
* required input artifact identities;
* produced output artifact identities;
* negative diagnostic classes;
* minimal positive demonstrator requirement.

Optional fields are:

* profile requirement;
* runtime input requirement;
* allowed unsupported-scope condition;
* related report kind;
* DSE feedback use.

An optional field must be omitted when it is not meaningful. It must not
be filled with a placeholder value.

## Required Trace Path

The target Loom stack must define trace edges for this path:

```text
C/C++ or CMSIS source
  -> LLVM IR
  -> raised MLIR / SCF-shaped representation
  -> dataflow.thread / dataflow.graph
  -> DFG-sim report
  -> ADG Builder C++ hardware description
  -> fabric.system + fabric.module hardware IR
  -> PnR and independent mapping artifact
  -> CGRA-sim report
  -> runtime package / launch descriptor
  -> RTL manifest + SystemVerilog
  -> EDA tooling reports
  -> FPA report
  -> combined cycle/frequency/power/area feedback report
  -> DSE feedback into compiler and hardware choices
```

The path is a traceability requirement, not a requirement that every
ordinary compiler invocation runs every stage. Compatibility-mode builds
may stop after ordinary compiler output. Artifact, acceleration,
simulation, RTL, and estimation modes opt into later edges.

## Required Edge Set

### Source To LLVM IR

Producer: `loom-cc` or `loom-c++`.

Consumer: LLVM parser, raising pipeline, app and CMSIS test runners.

Spec owners: `docs/spec-cmsis-dropin-compiler.md`,
`docs/spec-app-dropin-test-corpus.md`, and
`docs/spec-compiler-part-1-source.md`.

Required diagnostics include ordinary compiler failure, unsupported
ordinary option, missing include path, target mismatch, invalid response
file, and artifact-output collision.

### LLVM IR To Raised MLIR

Producer: LLVM-to-MLIR raising pipeline.

Consumer: placement framework and dataflow lowering.

Spec owners: `docs/spec-compiler-part-2-scf.md` and related compiler
source specs.

Required diagnostics include unsupported LLVM construct, missing data
layout, unsupported intrinsic, unrepresentable memory effect, and
malformed raised artifact.

### Raised MLIR To Dataflow IR

Producer: compiler placement and dataflow lowering.

Consumer: DFG-sim, PnR, runtime artifact packaging, and comparison
tools.

Spec owners: `docs/spec-compiler-part-3-dfg.md`,
`docs/spec-compiler-part-3-placement-framework.md`,
`docs/spec-dataflow-part-1-streaming.md`, and
`docs/spec-dataflow-part-2-control.md`.

Required diagnostics include unsupported L1, L2, or L3 placement,
illegal thread/graph layering, unsupported dataflow primitive, invalid
memory-order dependency, and malformed graph or subgraph boundary.

### Dataflow IR To DFG-sim Report

Producer: DFG-sim.

Consumer: comparison, PnR cost model, DSE, full-stack reporting.

Spec owners: `docs/spec-sim-dfg.md` and
`docs/spec-full-stack-reporting.md`.

Required diagnostics include unsupported operation, invalid token
state, invalid memory model, missing runtime input, non-deterministic
configuration, and report schema failure.

### ADG Builder To Fabric IR

Producer: ADG Builder C++ API.

Consumer: Fabric verifier, PnR, CGRA-sim, RTL lowering, FPA
estimation.

Spec owners: `docs/spec-adg-builder.md`,
`docs/spec-fabric-system-adg.md`, and Fabric module specs.

Required diagnostics include duplicate symbol, invalid node kind,
invalid protocol schema, illegal channel direction, missing required
port, invalid one-to-one link, unsupported crossing, invalid coherence
domain, invalid memory model, and malformed visualization metadata.

### Fabric IR To Mapping Artifact

Producer: PnR.

Consumer: mapping verifier, CGRA-sim, runtime packaging, RTL
workload-harness generation, FPA estimation, DSE, visualization tools.

Spec owners: `docs/spec-pnr.md`,
`docs/spec-mapping-artifact.md`, and related
`docs/spec-mapping-*.md` files.

Required diagnostics include unresolved software reference, unresolved
hardware reference, artifact identity mismatch, incompatible placement,
illegal route, missing buffer, illegal schedule, invalid temporal tag,
illegal memory binding, and no legal candidate.

### Mapping Artifact To CGRA-sim Report

Producer: CGRA-sim.

Consumer: comparison, FPA estimation, DSE, full-stack reporting.

Spec owners: `docs/spec-sim-cgra.md`,
`docs/spec-sim-comparison.md`, and
`docs/spec-full-stack-reporting.md`.

Required diagnostics include stale mapping, missing route, missing
schedule, missing buffer depth, invalid memory binding, unsupported
hardware primitive, invalid runtime input, and report schema failure.

### Runtime Package And Launch Descriptor

Producer: compiler packaging or runtime package builder.

Consumer: runtime, simulator dispatch, hardware execution dispatch, and
full-stack reporting.

Spec owner: `docs/spec-runtime-abi.md`.

Required diagnostics include missing dataflow artifact, missing Fabric
artifact, missing mapping artifact, unsupported target profile, invalid
memory descriptor, missing fallback policy, stale package identity, and
unsupported launch mode.

### Fabric IR To RTL Manifest And SystemVerilog

Producer: RTL lowering.

Consumer: RTL lint, RTL simulation, synthesis, timing, power, FPA
estimation, and full-stack reporting.

Spec owners: `docs/spec-rtl-lowering.md` and
`docs/spec-eda-tooling.md`.

Required diagnostics include unsupported primitive lowering,
unsupported protocol feature, missing external-port implementation,
illegal implicit fanout, missing clock crossing implementation, missing
memory model, unsupported black box, and RTL manifest schema failure.

### RTL Manifest To EDA Reports

Producer: EDA tooling adapter.

Consumer: FPA estimation, DSE, full-stack reporting.

Spec owners: `docs/spec-eda-tooling.md` and
`docs/spec-fpa-estimation.md`.

Required diagnostics include no matching tool profile, activation
failure, missing library profile, backend execution failure, parser
failure, timing failure, incomplete activity evidence, and unsupported
capability.

### EDA And Activity Reports To FPA Report

Producer: FPA estimator.

Consumer: full-stack reporting and DSE.

Spec owners: `docs/spec-fpa-estimation.md` and
`docs/spec-full-stack-reporting.md`.

Required diagnostics include unsupported fidelity request, missing
activity source, incompatible activity source, missing timing evidence,
missing area evidence, missing power evidence, and stale input
identity.

### Reports To DSE Feedback

Producer: full-stack reporter and DSE controller.

Consumer: compiler placement, PnR search, ADG Builder candidate
generation, hardware profile selection.

Spec owner: `docs/spec-dse-feedback.md`.

Required diagnostics include incompatible report set, missing objective,
untrusted metric, unsupported feedback target, conflicting constraints,
and non-reproducible candidate identity.

## Artifact Identity

Every finalized artifact that can cross a tool boundary has an
`ArtifactIdentity` computed from exactly one fixed preimage:

```text
bytes("loom.artifact.identity.v1\0")
|| u32be(length(schema_identity))
|| bytes(schema_identity)
|| u32be(schema_version.major)
|| u32be(schema_version.minor)
|| u64be(length(canonical_semantic_bytes))
|| canonical_semantic_bytes
```

`ArtifactIdentity` is SHA-256 of this preimage. SHA-256, the domain tag,
the framing, and the 32-byte output width are fixed and not configurable.
The external spelling is exactly 64 lowercase hexadecimal characters.
Each artifact family owns its schema descriptor and canonical semantic
serialization; Common owns only framing and hashing.

The local Artifact Store stores and compares the exact full preimage under
the derived identity key. An identical preimage deduplicates, a different
valid preimage with the same identity is an identity collision, and an
invalid preimage or key mismatch is store corruption. Publication never
overwrites an existing key.

Validated reads take an expected schema descriptor and an
`ArtifactIdentity`. The object path is derived only from the identity. The
reader rejects missing objects, symbolic links, and non-regular files; parses
the fixed identity domain, schema descriptor, and canonical byte length;
recomputes SHA-256 against the path identity; and returns exactly the stored
canonical semantic bytes. A key-valid object with a different schema identity
or version is a schema mismatch. A malformed preimage or identity-key mismatch
is store corruption.

The store root is a caller-provisioned, durably established non-symlink
directory. `ArtifactStore` does not create the root or any containing
directory.

Logical names, producer and invocation data, configuration records,
timestamps, host paths, diagnostics, and parent lineage belong in
manifests, reports, or Evaluation Evidence. They are not identity inputs
unless the artifact family's canonical semantic bytes explicitly contain
a typed upstream artifact reference.

Current mapping and simulation artifact formats are described in
`docs/spec-intermediate-artifacts.md`. They do not replace the artifact
identities checked by this traceability contract.

## Validation

The traceability verifier checks that:

* every required edge is represented by a schema or verifier;
* every edge has a producer, consumer, artifact kinds, and diagnostic
  classes;
* referenced artifacts resolve;
* required artifact identities match exactly;
* unsupported-scope records are explicit;
* reports identify the artifacts they summarize;
* DSE feedback refers to immutable candidate artifacts.

The verifier must not accept an edge because a file name looks
conventional. The required facts must be present in the artifact
records.

## Acceptance Criteria

The full-stack traceability target is complete when:

* every edge in the required trace path has a public spec owner;
* every edge has a schema or verifier target;
* every edge has deterministic negative diagnostics;
* every edge has at least one demonstrator requirement;
* full-stack reports can cite the same artifact identities checked by
  tool verifiers;
* DSE feedback can refer to exact report and candidate identities
  without mutating source IR, Fabric IR, or mapping artifacts.
