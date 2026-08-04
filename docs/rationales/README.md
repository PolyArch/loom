# Loom Architecture Rationales

These documents preserve the reasoning behind Loom's current architecture.
They are intentionally non-normative. Exact types, fields, algorithms,
validation rules, error classifications, defaults, and conformance anchors are
owned only by the linked `spec-*.md` documents.

A rationale may name a rejected design to explain the current boundary. It
must not retain a second implementable version of that design. Historical
discussion order, implementation status, worker state, and temporary queues
are not architecture facts and are not reproduced here.

## Authority Map

| Rationale owner | Normative owners |
|---|---|
| [Documentation Authority](documentation-authority.md) | [Full-Stack Architecture](../spec-loom-stack.md) |
| [Full-Stack Architecture](full-stack-architecture.md) | [Full-Stack Architecture](../spec-loom-stack.md), [Core Dialect Boundary](../spec-core-dialect-boundary.md), [End-To-End Conformance Anchors](../spec-end-to-end-demonstrators.md), [LoomBench](../spec-loombench.md), [CMSIS Drop-In Compiler](../spec-cmsis-dropin-compiler.md) |
| [Engineering Methodology](engineering-methodology.md) | [Full-Stack Architecture](../spec-loom-stack.md), [Resolved Configuration](../spec-config-ssot.md), subsystem conformance-anchor sections |
| [Configuration And Artifact Identity](configuration-and-artifact-identity.md) | [Resolved Configuration](../spec-config-ssot.md), [Full-Stack Traceability](../spec-full-stack-traceability.md), [Intermediate Reports](../spec-intermediate-artifacts.md), [Visualization](../spec-mapping-visualization.md) |
| [Compiler Frontend](compiler-frontend.md) | [Source Integration](../spec-compiler-part-1-source.md), [Structured Compiler Frontend](../spec-compiler-part-2-scf.md), [SCF To DFG](../spec-compiler-part-3-dfg.md), [Memory Frontier Lowering](../spec-compiler-part-3-mem.md), [Logical Domains](../spec-compiler-part-4-partitioned-data.md), [Vector Semantics](../spec-dataflow-vectorization.md) |
| [Dataflow Execution](dataflow-execution.md) | [SCF To DFG](../spec-compiler-part-3-dfg.md), [Streaming And Channels](../spec-dataflow-part-1-streaming.md), [Control Operations](../spec-dataflow-part-2-control.md), [Memory Consistency](../spec-dataflow-memory-consistency.md) |
| [Fabric And ADG Construction](fabric-and-adg.md) | [ADG Builder](../spec-adg-builder.md), [Fabric Artifact](../spec-fabric-artifact.md), [Fabric Module](../spec-fabric-module.md), [Fabric System ADG](../spec-fabric-system-adg.md), [Fabric Identity](../spec-fabric-identity.md) |
| [Fabric Capabilities](fabric-capabilities.md) | [Hardware Sharing Groups](../spec-fabric-hw-share-group.md), [Reconfigurable Operations](../spec-fabric-reconfigurable-op.md), [Fabric FU](../spec-fabric-fu.md), [FU Synthesis](../spec-generalize-subgraphs-to-fu.md), [Fabric PE](../spec-fabric-pe.md), [Temporal PE](../spec-fabric-pe-temporal.md) |
| [Fabric Memory And Resources](fabric-memory-and-resources.md) | [Fabric Memory](../spec-fabric-mem.md), [Fabric Resource Contract](../spec-fabric-resource-contract.md), [Fabric Boundary](../spec-fabric-boundary.md), [Fabric FIFO](../spec-fabric-fifo.md), [Fabric Switch](../spec-fabric-switch.md), [Fabric Instantiate](../spec-fabric-instantiate.md) |
| [Mapping And PnR](mapping-and-pnr.md) | [Mapping Artifact](../spec-mapping-artifact.md), [Mapping Identity](../spec-mapping-identity.md), [Mapping Memory](../spec-mapping-memory.md), [Mapping Verification](../spec-mapping-verification.md), [TechMapping Generation](../spec-tech-mapping.md), [Place And Route](../spec-pnr.md) |
| [Evaluation And DSE](evaluation-and-dse.md) | [Evaluation And DSE](../spec-dse-feedback.md), [Evaluation Metrics](../spec-evaluation-metrics.md), [External Tool Invocation](../spec-external-tool-invocation.md), [FPA Evaluation](../spec-fpa-estimation.md) |
| [Simulation](simulation.md) | [Simulation Artifacts](../spec-simulation-artifacts.md), [DFG-sim](../spec-sim-dfg.md), [CGRA-sim](../spec-sim-cgra.md), [Simulation Comparison](../spec-sim-comparison.md) |
| [Runtime And Deployment](runtime-and-deployment.md) | [Executable Closure](../spec-executable-closure.md), [Configuration And Deployment](../spec-configuration-deployment.md), [Runtime ABI](../spec-runtime-abi.md), [Implementation Platform](../spec-implementation-platform.md) |
| [Hardware Backend](hardware-backend.md) | [RTL Lowering](../spec-rtl-lowering.md), [Hardware Implementation](../spec-hardware-implementation.md), [Implementation Platform](../spec-implementation-platform.md), [EDA Tooling](../spec-eda-tooling.md), [External Tool Invocation](../spec-external-tool-invocation.md), [FPA Evaluation](../spec-fpa-estimation.md) |

This table is navigation, not a second ownership registry. Each linked
specification states its own exact scope and points to narrower owners when
necessary.
