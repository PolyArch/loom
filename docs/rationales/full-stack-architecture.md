# Full-Stack Architecture Rationale

Normative boundaries are owned by
[Loom Full-Stack Architecture](../spec-loom-stack.md),
[Core Dialect Boundary](../spec-core-dialect-boundary.md),
[End-To-End Conformance Anchors](../spec-end-to-end-demonstrators.md),
[LoomBench](../spec-loombench.md),
[CMSIS Drop-In Compiler](../spec-cmsis-dropin-compiler.md), and
[Real Application Portfolio](../spec-application-portfolio.md).

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

## Why The Architecture Is Heterogeneous And Tiled

The machine model distills a common structure from production and research
spatial accelerators rather than copying one product. AMD Versal combines
stored-program processing, programmable logic, tiled AI engines with local
memory, and a programmable NoC. Plasticine separates pipelined compute units
from banked scratchpad and address-generation units on a hierarchical spatial
interconnect. Tenstorrent exposes tile-local SRAM and explicit NoC data
movement, with data-movement processors distinct from cooperative compute
processors. Gemmini demonstrates why a generated accelerator must remain in a
complete SoC and software stack, while DSAGEN demonstrates that a small
composable set of spatial primitives can support hardware/software co-design.

Their shared lesson is not that Loom needs vendor-specific tile kinds. It is
that control, compute, local storage, data movement, and interconnect are
independent physical roles whose composition determines application quality.
`HostCore`, `InstructionCore`, `SpatialCore`, PE, memory, boundary, switch,
service, and transport already express those roles. New application-named
hardware objects would increase conceptual surface without adding an essential
distinction.

The same evidence rejects one generic size knob. Compute residency, routing
table depth, tag namespace, buffering, memory concurrency, and network shape
have different area, latency, throughput, and mapping effects. Coupling them
may be a convenient preset recipe, but exposing only the coupled value makes
Hardware DSE unable to repair the actual bottleneck. Versioned builtin schemas
therefore expose independent physical axes and may offer presets only as fully
resolved starting points.

Primary architecture references:

* AMD, *Versal Adaptive SoC AI Engine Architecture Manual* (AM009), 2026.
* AMD, *Versal Programmable Network on Chip Product Guide* (PG313), 2023.
* R. Prabhakar et al., *Plasticine: A Reconfigurable Architecture for
  Parallel Patterns*, ISCA 2017.
* Tenstorrent, *TT-Metalium Memory from a Kernel Developer's Perspective* and
  data-movement kernel documentation.
* H. Genc et al., *Gemmini: Enabling Systematic Deep-Learning Architecture
  Evaluation via Full-Stack Integration*, DAC 2021.
* J. Weng et al., *DSAGEN: Synthesizing Programmable Spatial Accelerators*,
  ISCA 2020.

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

## Why Invocation Diagnostics Use One Binding

Mapping search, runtime integration, generated RTL harnesses, and external-tool
caches need different diagnostic events, but they do not need different
verbosity concepts. Separate environment variables and parsers made one
invocation depend on subsystem-specific knowledge and allowed invalid values or
level ranges to diverge.

One Common-owned `LOOM_VERBOSE_LEVEL` therefore supplies the closed levels zero
through three. Each subsystem still owns its event vocabulary and may use only
the levels it needs. ResolvedConfig ownership was rejected because diagnostic
presentation cannot change semantic identity or reproducibility.
Implicit propagation and provider-authored verbosity were also rejected. An
external owner mechanically projects the Common-parsed value through the one
shared spelling, while cache normalization excludes that presentation-only
argument. This preserves one operator control without allowing host state to
become a second semantic execution input.

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

Target-profile identity and executable-cohort compatibility are deliberately
separate. Replacing an ARM MVE, DSP, or NEON row with a portable scalar body
would test a different program, while failing the whole inventory merely
because the selected System cohort is RISC-V would conflate profile coverage
with semantic execution. The distilled boundary is therefore a typed
profile/cohort incompatibility outcome before provider setup. It remains
separate from semantic pass and from ordinary missing-provider failure, so a
future exact Arm cohort can execute the same manifest row without changing its
identity.

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

Header-defined operators require one additional distinction without a new
authority. A header is not an independently compiled source row, but the
compiler already preserves its inline definition location through LLVM debug
provenance. The workload provider therefore identifies the one pinned file
that contains the typed public protocol body, and the gate intersects that
file with the selected graph's compiler-derived provenance. Treating the
generated caller as an alias for the header would allow unrelated wrapper work
to pass; accepting every included header would have the same defect. Requiring
the exact intersection preserves both source-inventory meaning and operator
ownership with no symbol blacklist or parallel provenance system.

The primary semantic-alignment gate needs breadth across operator protocols,
not a Cartesian product of every profile, build alias, and input vector. It
therefore chooses one real producer, one exact profile, and one
deterministic vector per typed operator identity. Additional profiles and
vectors remain useful extended coverage, but multiplying them into the primary
gate would spend most of its time repeating compilation and DSE rather than
exposing new compiler semantics. Under one executable cohort, only compatible
rows enter semantic execution; incompatible profile rows remain explicit
conformance outcomes rather than silently changing source paths.

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

## Why Real Applications Are A Separate Portfolio

The 892-row corpus is broad operator evidence, not a substitute for complete
applications. Its units intentionally isolate typed call protocols so failures
can be localized and compilation work can be shared across vectors. That
granularity does not exercise whole-program ownership, multi-stage data
movement, cross-graph scheduling, SystemMapping, deployment, or sustained
feedback between quality and compiler cost.

The initial ten frontend anchors exposed a concrete inventory conflict:
`vector_pack`, `stencil3d`, and `attention` were required as independent
source-to-Dataflow workflows, while the pinned LoomBench manifest contained no
matching operator identities. Treating `pack_bits`, a seven-point Jacobi
kernel, and standalone softmax as aliases would conflate different protocols.
Replacing those existing rows would discard valid corpus coverage. Keeping the
new workflows outside the manifest would create a second application
inventory. The selected closure therefore appends three ordinary LoomBench
operator rows and advances the representative gate from 889 to 892 without
removing or renaming any existing row. This preserves the manifest as the sole
membership authority and lets each workflow retain its own source, protocol,
and oracle.

The real-application portfolio therefore starts with five complementary
programs: bounded TinyML anomaly inference, a compact language-model kernel
harness, irregular PageRank, a Loom-owned heterogeneous multisensor Attention
pipeline, and regular contiguous-memory vecadd. A fixed starter set makes
vertical closure reviewable, while a general admission rule allows later
applications only when they contribute a new stack behavior. Treating
directory contents or a generated dashboard as membership would make ordinary
checkout state a semantic authority.

One thin manifest is the smallest sufficient repository owner. It selects
source/build entries, named inputs, independent oracles, and execution cadence;
all program, workload, Mapping, simulation, implementation, and Evidence facts
remain in their existing owners. An `ApplicationArtifact` would merely wrap
those references and create another identity without adding a semantic fact.
Similarly, one manifest with smoke, validation, and scale/EDA selections keeps
cadence vocabulary separate from membership and avoids three drifting
inventories. The current rows select only smoke because no independent
validation or scale/EDA workload-and-oracle row is yet admitted; retaining an
empty tier claim would defeat that separation.

Pinned external source revisions remain Gitlink facts rather than copied
manifest strings. Fixed model data follows the executable image contract and
runtime samples follow SimulationRuntimeInput. Large bytes can therefore stay
in a verified local cache without turning paths into identity. This split also
keeps the public repository free of restricted datasets and direct EDA
material while preserving exact reproducibility through digests.

Pairwise engine equality is deliberately conditional. It is powerful for a
deterministic exact workload, but approximate special math and legal software
nondeterminism can produce several correct observations. Requiring every
engine to satisfy the same independent typed oracle tests the application
contract directly and avoids making one implementation the golden authority.

## Why Workload Sets Define Optimization Scope

Application-specific, domain-specific, and general acceleration differ by the
workloads a design must serve, not by a switch in the optimizer. A scope enum
would copy selection already fixed by exact workload and runtime-input roots,
and its meaning would drift as manifests evolve. The exact selected set is
therefore sufficient: one application, one coherent domain subset, or a cross-
domain portfolio declared complete for the release naturally supports the
corresponding application-specific, domain-specific, or general claim.

Every member must retain its own correctness and acceleration gate. Allowing a
mean score to hide one unsupported or regressed workload would optimize the
benchmark aggregate rather than the declared scope. The same reasoning
requires every released AccCore occurrence to have at least one selected
SystemMapping user; otherwise unused hardware could improve an abstract
capacity score while contributing nothing to the selected applications.

## Why Hardware Optimization Covers The Complete System

A SpatialCore Module is the right detailed RTL and EDA unit, but it is not the
whole accelerator architecture. AccCore count and heterogeneity,
InstructionCore realization, transport topology, memory, and services control
software partitioning, cross-core communication, and total system quality.
Optimizing Modules in isolation would freeze those choices outside the search
and contradict Loom's multi-core machine model.

The final hardware candidate is therefore a complete `fabric.system`, with
Module rewrites retained as intermediate candidates. Detailed physical
implementation remains limited to SpatialCore modules because that is the
portable hardware boundary Loom currently owns. Analytic and learned models
plus gem5 cover the other System components. Keeping each observation's exact
model identity prevents those estimates from being mislabeled as synthesized
or measured hardware while still letting System-level choices participate in
joint optimization.

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
