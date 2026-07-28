# LoomBench

This document owns Loom's repository-maintained high-level-language benchmark
suite under `test/app`. `test/app/manifest.json` is the single membership
authority. Directory enumeration, fixed counts in documentation, runner lists,
and generated status files are not alternate inventories.

## Purpose

LoomBench tests two related product contracts:

* `loom-cc` and `loom-c++` can replace ordinary C and C++ compiler drivers for
  self-contained programs; and
* every case is eligible for Loom's compiler, Mapping, simulation, hardware,
  and Evaluation flows without gaining benchmark-specific semantics.

CMSIS-DSP and CMSIS-NN are the other two canonical source suites. Their
external-source-tree contract is owned by `spec-cmsis-dropin-compiler.md`.
LoomBench owns only repository cases; it does not duplicate CMSIS membership.

## Membership And Identity

The manifest schema is version `2.0`. Each case has one stable `case` name and
records its source files, language mode, compiler and link flags, expected
executables, deterministic oracle, selected compiler tiers, and nonempty
feature tags.

The accepted compiler tiers are:

```text
run
raise
dfg
```

`run` requires native and Loom drop-in build and execution. `raise` requires a
valid raised MLIR artifact. `dfg` requires a finalized canonical Dataflow
artifact. Selecting a tier asserts success; an unsupported diagnostic is a
failure for that selected tier, not a passing fixture.

These tiers select fast regression checkpoints; they do not classify a case's
maximum supported compiler depth. A complete-stage invocation can request the
same deeper checkpoint for every manifest member. Absence of that tier from a
case's fast selection does not waive the product contract or turn a failure at
that stage into success.

Every emitted source-suite identity is exactly one of `loombench`,
`cmsis-dsp`, or `cmsis-nn`. A repository case emits `suite=loombench` and its
manifest case name. The pair is unique. Counts are always derived from the
manifest or pinned external source tree and never become specification
constants.

## Case Contract

Each LoomBench case is a standalone source package under `test/app/<case>/`.
It provides:

* all sources required to build the case;
* a deterministic entry point and input;
* a reference or expected-output oracle that fails on mismatch; and
* the metadata referenced by its manifest row.

A case cannot depend on Loom libraries, private machine paths, network access,
wall-clock time, random devices, host-specific files, or mutable state outside
its case directory. Reference and candidate function naming is a harness
convention only and does not define compiler, Dataflow, or Mapping semantics.

Generated build products are not committed unless one small stable golden
artifact is necessary to test a semantic contract. Formatting or harness
changes cannot weaken the output oracle.

## Drop-In Execution

For every `run` case, ordinary `gcc` or `g++` first builds and executes the
program with the manifest flags. The same source and flags then run through
`loom-cc` or `loom-c++`. Both executions must satisfy the same observable
oracle.

Compatibility mode cannot fail merely because acceleration is unavailable.
Loom-specific artifact generation and acceleration remain explicit options.
When requested acceleration is unsupported, the selected Structured Program
Candidate keeps the work on an InstructionCore or compilation returns a typed
failure; runtime cannot silently fall back from an invalid Mapping.

## Compiler Artifacts

Selected `raise` and `dfg` cases exercise the real driver and in-process
compiler libraries:

```text
source
  -> LLVM IR
  -> initial Structured Program Candidate S0
  -> selected Structured Program Candidate Sn
  -> initial Canonical Dataflow Program D0
  -> selected Canonical Dataflow Program D*
```

Artifacts must parse, verify, and retain the case and source identities needed
for exact lineage. A `dfg` case cannot pass with residual unsupported control
or an unsupported-scope placeholder. The Canonical Dataflow Program may be
graph-free only under the complete linked-workload candidate and evidence
contract owned by `spec-loom-stack.md`; it must still preserve the complete
InstructionCore program and pass ordinary whole-program finalization. A
representative anchor whose contract selects Spatial execution must contain
the required nonempty graph.

The representative ten-kernel frontend anchor set is owned by
`spec-end-to-end-demonstrators.md`. Each anchor must resolve to a LoomBench
manifest member before the complete frontend gate can pass; the anchor list is
not a second inventory.

## Downstream Evaluation

PnR, DFG-sim, CGRA-sim, RTL execution, and EDA evaluation use ordinary exact
Artifacts, Evaluation Requests, and resolved policies. They do not require a
parallel LoomBench tier schema or one status record per case.

When such work is requested:

* Mapping follows `spec-mapping-artifact.md` and `spec-pnr.md`;
* DFG and CGRA execution follow `spec-sim-dfg.md` and
  `spec-sim-cgra.md`;
* normalized results follow `spec-dse-feedback.md` and
  `spec-evaluation-metrics.md`; and
* human-readable output remains a removable projection under
  `spec-intermediate-artifacts.md`.

An unselected downstream evaluation makes no pass claim. A selected evaluation
must produce its real owner artifacts or a typed honest outcome; inventory
membership, wrapper execution, or generated filenames cannot substitute for
evidence.

## Inventory Tooling

`test/corpus_inventory.py` is the shared derived inventory view. It validates
the LoomBench manifest and pinned CMSIS source inventories, preserves case and
source identities, and supports complete-suite or explicit-case selection. It
does not maintain its own membership list.

Runners discover cases from that structured view. Empty selections, duplicate
case/source identities, missing sources, malformed manifest records, stale
oracles, and missing requested artifacts fail explicitly. Output directories
are deterministic for a resolved invocation but are not semantic identities.

## Breadth

Feature tags cover the semantic breadth needed for compiler and architecture
work, including dense and sparse numeric kernels, graph and irregular access,
reductions and scans, stencils, signal processing, bit operations, control,
streaming, vector behavior, and neural-network kernels.

A smoke or tier selection is an execution choice and never a smaller canonical
suite or a weaker product boundary.
SPEC CPU 2026 is a separate external conformance corpus and does not alter
LoomBench or CMSIS membership.

## Anchor Tests

Stable tests cover manifest validity and uniqueness, deterministic inventory
derivation, native/drop-in oracle agreement, selected raise/DFG artifact
validity, rejection of empty work, and honest downstream evidence coupling.

The suite must not create one wrapper or golden IR file per transformation,
snapshot diagnostic text, duplicate membership counts, or preserve runner
implementation shape.
