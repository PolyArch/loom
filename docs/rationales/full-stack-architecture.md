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

## Why Host And Accelerator Domains Are A Partition

The intended system weakly couples a HostCore domain to a heterogeneous
AccCore cluster with its own NoC and accelerator memory. PCIe is a useful
analogy, but the same organization may be integrated through CXL, an SoC
interconnect, or a custom link. Physical packaging is not a semantic
distinction. Making the accelerator domain another persistent artifact would
duplicate the same occurrences, endpoints, services, and transport already
owned by `fabric.system`.

Loom therefore treats the two domains as a conceptual partition derived from
the system graph. Typed service endpoints express communication, Transport
Architecture owns its logical guarantees, and Interconnect Implementation
owns the concrete protocol. This preserves arbitrary topology and lets the
hardware and simulator bindings change protocol realization without changing
software or Mapping identity.

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
its manifest; CMSIS membership is owned by the independently invocable
translation units in the pinned source trees; external SPEC membership is
owned by its harness. Snapshot counts are evidence, not design constants. This
distinction prevents a smoke subset or generated status file from becoming a
false support boundary.

## Why Source And Workload Coverage Are Separate

LoomBench, CMSIS-DSP, and CMSIS-NN differ in ownership, build flags, source
shape, and regression cost, not in compiler semantics. Assigning each suite a
different terminal stage would turn benchmark organization into a product
capability rule and could make the same conforming C function succeed or fail
solely because of its directory.

An independently compiled translation unit is not necessarily a program. It
may contain only constants, expose code only under another target profile, or
call bodies that become visible only after archive selection and LTO. Requiring
every object to invent a program entry or a nonempty graph would change source
semantics. Conversely, accepting object compilation as proof of executable
acceleration would hide unresolved calls, unused archive members, and workload
behavior.

Loom therefore checks source coverage and linked workload coverage separately,
then requires a total relation between them. The source side protects drop-in
and separate-compilation behavior. The workload side supplies the exact link
closure, profile, inputs, and oracle needed by optimization, simulation, and
Mapping. Data-only units receive honest coverage through real consumers rather
than synthetic computation.

CMSIS already owns executable tests, call sequences, patterns, and expected
values. Replacing them with a Loom-authored operator catalog would create a
second workload and oracle authority. Treating each test vector as a top-level
workload would instead repeat final link and DSE for mere input-size changes;
treating a whole descriptor or suite as one workload would hide independent
operator semantics behind a giant wrapper.

The smallest stable middle ground is the typed operator protocol. Unity
wrappers and DSP descriptors mechanically project to ordered public-call
protocols. Equal protocols under one target profile share compilation and DSE,
while each upstream test contributes an ordered vector with an independent
oracle and execution limit. Stateful initialization and execution remain
together because splitting them would change semantics. Query and failure-path
protocols remain separate because merging them with compute would erase a real
callable distinction. Aggregate and individual builds are producer variants,
not new operators or a blanket multiplication rule. This granularity preserves
upstream ownership, keeps work modular, and prevents both giant-wrapper and
per-vector fragmentation.

The protocol boundary must also exclude the test harness itself. Starting
candidate discovery from a Unity or descriptor test method admits pattern
loaders, assertions, statistics, and error-comparison loops that happen to be
in the same static call closure but are not the operator being evaluated. A
minimal generated wrapper is useful only when it preserves an ordered
multi-call protocol such as initialization followed by execution. A single
public call needs no synthetic ownership authority. This is why the normative
[Canonical Source Inventory](../spec-cmsis-dropin-compiler.md#canonical-source-inventory)
allows an exact public symbol for one-call protocols, requires an atomic wrapper
for multi-call protocols, and rejects test-method fallback.

The primary semantic-alignment gate needs breadth across operator protocols,
not a Cartesian product of every profile, build alias, and input vector. It
therefore chooses one real producer, one applicable profile, and one
deterministic vector per typed operator identity. Additional profiles and
vectors remain useful extended coverage, but multiplying them into the primary
gate would spend most of its time repeating compilation and DSE rather than
exposing new compiler semantics.

Loom pins the Unity runtime selected upstream and lets upstream build metadata
select library sources and archive members. Staging generated files outside the
submodule keeps external sources immutable while preserving the real test
semantics. Exact SourceCoverageEdges separately prove that code and data owners
participate in at least one executable protocol, so a table translation unit
does not need fabricated computation.

Fast tests may stop at stage checkpoints to localize regressions, and smoke
selections may keep routine execution affordable. Those are invocation choices,
not alternate semantics. A graph-free Canonical Dataflow result is legal only
after the exact linked workload accounts for every legal Spatial candidate and
selects stored-program ownership with the existing compiler and Evaluation
evidence. Directory membership, an empty object, or a feature-disabled stub is
never that proof.

## Why Hardware Construction Closes First

The ADG Builder-to-Fabric path has no software prerequisite, while useful
Structured optimization needs target ABI facts, resource bounds, topology,
memory capability, and hardware-aware quality estimates. Closing one real
hardware substrate first therefore removes an upstream ambiguity instead of
hard-coding an abstract target into the compiler.

After an exact builtin Fabric exists, frontend and non-Mapping Evaluation must
advance together. The frontend owns legal candidate generation; Evaluation
owns comparable observations over software-only or software-plus-Fabric
subjects. Developing either side alone would force transformations to use
fixed heuristics or force models to evaluate candidate forms the compiler
cannot yet produce. Mapping remains a later fidelity boundary and is not
required to establish the complete pre-Mapping corpus path.

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
