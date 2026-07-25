# Full-Stack Architecture Rationale

Normative boundaries are owned by
[Loom Full-Stack Architecture](../spec-loom-stack.md) and
[Core Dialect Boundary](../spec-core-dialect-boundary.md).

## Why Loom Is A Full Stack

The project is intended to answer one connected question: how a high-level
program is transformed, mapped, executed, evaluated, configured, and realized
for a multi-core heterogeneous spatial accelerator. Independent pass wrappers,
reports, and demonstrations cannot answer that question if they do not share
semantic artifacts and exact identities.

This led to six semantic owners rather than a collection of stage scripts:
compiler frontend, hardware frontend, Mapping, simulation, hardware backend,
and central Evaluation/DSE. They can be grouped operationally into compilation,
evaluation/exploration, and hardware construction, but the three operational
groups do not erase the six ownership boundaries.

The decision rejects two extremes. Loom is not only a CGRA mapper fed by
hand-written Dataflow, because source-level transformations and ownership cuts
are central research questions. It is also not a monolithic tool that owns CPU,
NoC, EDA, and every external runtime implementation. Mature external systems
remain providers behind exact Loom-owned contracts.

## Why The Machine Model Uses AccCore

The target system must represent both spatial execution and the code that
cannot profitably or legally become a spatial graph. The distilled unit is:

```text
AccCore = InstructionCore + SpatialCore
```

`InstructionCore` was chosen instead of `ScalarCore` because the fallback may
be vector-capable, superscalar, or out-of-order. `TemporalCore` was rejected
because temporal already describes time-multiplexed resources inside a
SpatialCore. The name denotes a PC-driven instruction stream, not datapath
width or performance class.

The hardware-root split and the rejection of a separate System hardware
dialect are explained by
[Fabric And ADG Construction](fabric-and-adg.md). The full-stack machine model
depends on that one hardware authority rather than restating its ownership.

## Why The Inputs And Outputs Are Asymmetric

LLVM IR is the language-independent software boundary because C and C++ are
the initial drivers while other languages can participate only when their ABI
and runtime contracts are supported. Fabric MLIR is the hardware boundary
because both custom C++ ADG construction and builtin targets must converge on
one hardware truth before compiler, Mapping, simulation, or RTL generation.

The outputs are not one generic build bundle. Evaluation results, executable
Deployment/configuration, and HardwareImplementation answer different
questions and have different identity and failure semantics. Combining them
would make an analysis-only run look executable or make generated RTL imply a
successful workload mapping.

## Why Public And Developer Tools Differ

Only `loom-cc` and `loom-c++` are stable end-user compiler drivers. A prior
single `loom` binary was rejected because C and C++ driver compatibility is the
actual public surface. A public visualization binary was also rejected;
`--loom-viz-export` is a removable projection of the same in-process flow.

Developer binaries remain valuable. `loom-opt`, focused simulators, replay
tools, and a PnR driver let engineers exercise real shared libraries without
starting from source every time. They are not alternate products: they own CLI
parsing and presentation only. The public drivers compose the same libraries
in process and never shell out to stage tools as their architecture.

The earlier `loom` versus `loom-devel` package wording was only an analogy for
compiler users versus ADG-library users. Distribution and package naming are
deliberately not architectural commitments.

## Why Vertical Closure Comes Before Breadth

A single real path exposes ownership and integration defects that hundreds of
isolated fixtures can conceal. Loom therefore closes representative source,
hardware, Mapping, simulation, configuration, and implementation paths before
expanding breadth. The representative anchors intentionally include dense,
sparse, irregular, DSP, stencil, vector, reduction, and multi-stage streaming
behavior so that the core does not specialize around vector addition.

The anchor list is not a benchmark inventory. LoomBench membership is owned by
its manifest; CMSIS membership is owned by the pinned source trees; external
SPEC membership is owned by its harness. Snapshot counts are evidence, not
design constants. This distinction prevents a smoke subset or generated status
file from becoming a false support boundary.

## Why Suites Share One Compiler Contract

LoomBench, CMSIS-DSP, and CMSIS-NN differ in ownership, build flags, source
shape, and regression cost, not in compiler semantics. Assigning each suite a
different terminal stage would turn benchmark organization into a product
capability rule and could make the same conforming C function succeed or fail
solely because of its directory.

Fast tests may stop at stage checkpoints to localize regressions, and smoke
selections may keep routine execution affordable. Those are invocation choices.
The complete inventory remains eligible for every requested stage under the
same driver, Artifact, verifier, and typed-failure contracts. A graph-free
Canonical Dataflow result is legal only because the exact ownership decision
kept all work on stored-program cores, not because CMSIS or LoomBench received
a weaker contract.

## Why Several Features Are Explicitly Deferred

Deferred features are absent rather than represented by empty records and
dormant flags. The single-tenant runtime decision is owned by
[Runtime And Deployment](runtime-and-deployment.md). Generic channel sessions,
remote deployment, installation packaging, and several advanced hardware
concerns likewise reopen their owners only when a concrete behavior cannot be
composed from current primitives.

This avoids two recurring failure modes: designing a large framework before a
real consumer exists, and letting an unimplemented placeholder be mistaken for
supported functionality.
