# Loom: A Full-Stack Compilation Framework for Domain-Specific CGRA Accelerators

## Slide Outline (20 slides)

### Part I: Framework and Infrastructure (Slides 1-10)
### Part II: Scheduling Algorithm for Multi-Core Heterogeneous CGRAs (Slides 10-15)
### Part III: Evaluation and Comparison (Slides 15-20)

---

## Part I: Framework and Infrastructure

### Slide 1: Title

**Loom: A Full-Stack Compilation Framework for Domain-Specific CGRA Accelerators**

- Subtitle: From Annotated C++ to Verified Multi-Core Heterogeneous Hardware
- Authors / Affiliation
- CDSC Presentation, 2026

---

### Slide 2: Motivation

**The Domain-Specific Accelerator Dilemma**

- DSAs deliver orders-of-magnitude efficiency, but building one end-to-end is painful:
  - RTL for hardware, custom mapper for scheduling, separate host integration, manual verification
  - Each layer uses a different representation; changes in one cascade to all others
- CGRAs sit at the sweet spot: reconfigurable like FPGAs, efficient like ASICs
  - But existing CGRA toolchains are architecture-locked, use ad-hoc IRs, and lack retargetability
- **Loom's thesis**: A single MLIR-based framework can compile annotated C++ to *any* CGRA topology, with unified scheduling, backend generation, and end-to-end verification

---

### Slide 3: Loom at a Glance

**Three Worlds, One Framework**

- Diagram: three-column architecture
  - **Software World**: C/C++ with pragmas -> LLVM IR -> Handshake/Dataflow MLIR
  - **Mapping Bridge**: Graph-to-graph place-and-route with tag-aware scheduling
  - **Hardware World**: Fabric MLIR -> SystemC (cycle-accurate) / SystemVerilog (synthesizable)
- Two custom MLIR dialects: **Dataflow** (software streaming semantics) and **Fabric** (hardware resources)
- Key property: hardware and software share the same MLIR type system and infrastructure
- 135+ application kernels validated end-to-end

---

### Slide 4: Pipeline Overview

**Five Stages from Source to Verified Silicon**

- Diagram: pipeline flow with clear A/B independence and C as bridge

```
Stage A: Frontend          Stage B: ADG
  C++ -> LLVM -> SCF         C++ ADGBuilder API
  -> Handshake/Dataflow      -> Fabric MLIR + SV/SystemC
         \                    /
          \                  /
       Stage C: Mapper Place-and-Route
         -> Placement + Routing + Temporal Assignment
         -> config_mem image
                |
       Stage D: Backend Realization
         -> Configured SystemC / SystemVerilog
                |
       Stage E: Co-Simulation & Verification
         -> ESI-based host-device cosim, CPU reference comparison
```

- A and B are independent; C bridges them; D and E close the loop

---

### Slide 5: Frontend and Dataflow Dialect

**From C++ Loops to Streaming Dataflow Graphs**

- Compilation chain: Clang -> LLVM IR -> SCF MLIR -> Handshake MLIR
- Optional pragmas as hints: `LOOM_ACCEL`, `LOOM_STREAM`, `LOOM_PARALLEL`, `LOOM_TARGET`
- Handshake dialect (from CIRCT) captures dataflow, but lacks loop iteration semantics
- Loom's **Dataflow dialect** adds four streaming primitives:
  - `dataflow.stream` -- index generator with runtime-configurable predicate (`<, <=, >, >=, !=`)
  - `dataflow.carry` -- loop-carried dependency (two-phase: init -> loop body)
  - `dataflow.invariant` -- loop-invariant broadcast (consume once, replay per iteration)
  - `dataflow.gate` -- stream alignment across region boundaries
- Tagged type system: `!dataflow.tagged<i32, i4>` packs tag + value for port multiplexing

---

### Slide 6: Fabric Dialect -- Hardware as MLIR IR

**A First-Class Hardware Representation**

- `fabric.module`: top-level container with graph-region semantics (cycles ok, combinational loops not)
- Core operations:
  - `fabric.pe` -- Compute / constant / load-store / dataflow PE (hardware params + runtime config)
  - `fabric.temporal_pe` -- Time-multiplexed PE with instruction memory and tag-based dispatch
  - `fabric.switch` / `fabric.temporal_sw` -- Static / temporal routing with connectivity matrix
  - `fabric.memory` / `fabric.extmemory` -- On-chip SRAM / external memory with LSQ
  - `fabric.fifo` -- Buffering element (bypassable, breaks combinational loops)
  - `fabric.add_tag` / `fabric.del_tag` / `fabric.map_tag` -- Tag boundary operations
- Key design: **structure vs. configuration separation**
  - Hardware params (latency, interval, port counts) are fixed at build time
  - Runtime config (tags, predicates, constants, route tables) populated by mapper

---

### Slide 7: ADG -- Programmatic Hardware Construction

**Defining Arbitrary CGRA Topologies in C++**

- ADGBuilder API -- single-source C++, no YAML/JSON/Python
  ```
  ADGBuilder builder("my_cgra");
  auto pe  = builder.newPE("alu").setLatency(1).addOp("arith.addi", "arith.muli");
  auto tpe = builder.newTemporalPE("shared_alu").setNumInstruction(8).setNumRegister(4);
  auto sw  = builder.newSwitch("router").setPortCount(4, 4);
  builder.buildMesh(4, 4, pe, sw, Topology::Mesh);
  ```
- Topologies: Mesh, Torus, DiagonalMesh, DiagonalTorus (or fully custom)
- Multi-backend export from single Fabric MLIR: SystemC model, SystemVerilog RTL, DOT visualization
- Clone-with-deduplication: structural equivalence detected at validation, shared templates reused

---

### Slide 8: Tagged Types and Temporal PE -- Key Innovations

**Tag-Driven Resource Sharing Without Explicit Mux Networks**

- **Tagged types**: `!dataflow.tagged<value_type, tag_type>` -- tag in high bits, value in low bits
  - Port multiplexing: multiple logical streams share one physical port
  - Instruction matching: temporal PE selects FU by input tag
  - Memory disambiguation: tagged LSQ matches transactions per tag
- **Temporal PE**: one physical datapath, N logical operations
  - Instruction memory stores `[tag | opcode | reg_read | reg_write | ...]`
  - Incoming token's tag matched against stored tags -> selects FU + operand routing
  - Internal register file (FIFO-based) for inter-instruction communication
  - Operand buffer: per-instruction or shared mode (per-tag virtual channels)
- **Temporal Switch**: tag-based route selection with round-robin arbitration across tags
- This is the architectural foundation for heterogeneous multi-core scheduling (Part II)

---

### Slide 9: Backend and Co-Simulation

**Closing the Loop: From IR to Verified Hardware**

- Two backends from identical Fabric MLIR:
  - **SystemC**: cycle-accurate or loosely-timed, TLM 2.0 config_mem access, fast iteration
  - **SystemVerilog**: synthesizable RTL, AXI-Lite config_mem, ASIC/FPGA ready
  - Functional parity guarantee between backends
- Unified config_mem: 32-bit words, per-module aligned, AXI-Lite/TLM access
  - CONFIG_WIDTH derived from hardware params (not settable)
  - Host writes once before execution; retains across soft-reset
- Co-simulation via ESI channels:
  - Config phase -> execution phase -> verification phase
  - Bit-exact comparison against CPU reference (tolerance for FP)
  - Backend binding: Verilator (open-source) or VCS (commercial)

---

### Slide 10: End-to-End Walkthrough -- AXPY on a Heterogeneous CGRA

**Tracing `Y[i] = a*X[i] + Y[i]` Through the Full Stack**

- **Frontend**: C++ kernel -> Handshake MLIR with `stream(i)`, `invariant(a)`, `carry` for reduction boundary, `load`, `mul`, `add`, `store`
- **ADG**: 2x2 heterogeneous mesh -- 2 spatial PEs (mul, add), 1 temporal PE (shared ALU), 2 load PEs, 1 store PE, 4 switches, 1 extmem
- **Mapper**: places stream -> dataflow PE, mul -> spatial PE, add -> spatial PE, loads/store -> LS PEs; routes through switches; assigns tags for memory port disambiguation
- **Backend**: generates SystemC model + config_mem image (switch routes, tag assignments, stream predicate)
- **Cosim**: stream 1024 elements, compare against CPU -- pass
- This walkthrough sets up the scheduling problem addressed in Part II

---

## Part II: Scheduling Algorithm for Multi-Core Heterogeneous CGRAs

### Slide 11: The Scheduling Problem -- Formal Definition

**Multi-Dimensional Mapping on Heterogeneous Resources**

- **Given**:
  - Software graph G_sw = (V_sw, E_sw) from Handshake/Dataflow MLIR
  - Hardware graph G_hw = (V_hw, E_hw) from Fabric MLIR with heterogeneous tiles
    - Spatial PEs (fixed-function), Temporal PEs (multi-function), Dataflow PEs (state machines)
    - Static switches, Temporal switches (tag-multiplexed)
    - Memories with typed ports and LSQ constraints
- **Find**: a mapping M = (placement, routing, temporal assignment) such that:
  - Every software op assigned to a compatible hardware tile
  - Every software edge routed through legal hardware paths
  - Every temporal resource (slot, tag, register) legally assigned
  - All hard constraints (C1-C6) satisfied
  - Cost function minimized
- **Dimensions of difficulty**: spatial placement x routing x temporal slot/tag/register allocation -- coupled, not separable

---

### Slide 12: Constraint Hierarchy

**Six Classes of Hard Constraints**

- **C1: Node compatibility** -- software op semantics match hardware tile capability
  - CandidateSet(swNode) derived from operation type, port types, interface category
  - Load/store specialization: TagOverwrite vs. TagTransparent modes
- **C2: Port and type compatibility** -- no implicit conversion at any boundary
  - Native/tagged category match; value type match; tag-width uniformity per interface
  - Explicit conversions only via `add_tag`, `del_tag`, `map_tag`, cast PEs
- **C3: Route legality** -- edges follow physical connectivity with correct directionality
  - Memory done-token wiring legality (lddone/stdone must reach control paths)
- **C4: Capacity** -- hardware resources not oversubscribed
  - Switch fan-in/fan-out, temporal PE slot/register limits, memory queue depths (ldCount, stCount, lsqDepth)
- **C5: Temporal** -- tag uniqueness, slot range, register index validity
  - No duplicate tags within temporal PE; register write requires tag=0
- **C6: Configuration encoding** -- emitted config must match Fabric op spec and error model
  - `constant_value`, `cont_cond_sel`, `output_tag`, compare predicates all correctly encoded

---

### Slide 13: Three-Phase Mapping Algorithm

**Placement -> Routing -> Temporal Assignment (with Repair)**

- **Phase 1: Candidate Construction + Placement**
  - Build CandidateSet per software node from C1 rules
  - Visit nodes in dependency order (topological)
  - Greedy selection with cost-based tie-breaking (placement pressure metric)
  - Capacity pre-check (C4) before committing each placement
  - Action primitive: `MapNode(swNode, hwNode)` -- atomic, rollback on failure

- **Phase 2: Routing**
  - Route each software edge between placed endpoints
  - Shortest legal path through switch network (C3 + C2 type checks at every hop)
  - Reserve switch ports; update capacity bookkeeping (C4)
  - Temporal switches: assign route-table slot + tag + enable mask
  - Action primitive: `MapEdge(swEdge, pathHint)` -- atomic

- **Phase 3: Temporal Assignment**
  - For each temporal PE: assign instruction slot, tag, opcode, register indices
  - For each temporal switch: assign route-table slots and tags
  - Verify C5 (no duplicate tags, register bounds, slot range)
  - Register allocation: assign register indices for inter-instruction data, respect FIFO depth

- **Repair Loop**: if conflict detected at any phase:
  - Escalation: reroute edges -> reassign nodes -> reassign temporal metadata -> restart
  - Bounded: `max_local_repairs`, `max_rollback_depth`, `max_global_restarts`

---

### Slide 14: Cost Model -- Multi-Objective Optimization

**Weighted-Sum Objective Over Five Metric Families**

`total_cost = w1*placement_pressure + w2*routing_cost + w3*temporal_cost + w4*throughput_proxy + w5*config_footprint`

- **Placement pressure**: `sum_tile (occ/cap)^2` -- penalizes concentration on limited tiles
- **Routing cost**: `sum_edge (hop_weight * usage)` -- penalizes long paths and congestion
  - Hop weight: direct < switch hop; congestion factor scales with shared-edge count
- **Temporal cost**: `alpha*slot_util + beta*reg_pressure + gamma*tag_pressure`
  - Slot utilization: how full is instruction memory
  - Register pressure: allocated / available registers
  - Tag pressure: used tag range / tag width capacity
- **Throughput/latency proxy**: `critical_path_est + ii_pressure + queue_pressure`
  - Critical path: sum of PE latencies along longest mapped path
  - II pressure: loop-carried dependency length (from dataflow.carry chains)
  - Queue pressure: load/store queue depth utilization
- **Config footprint**: `non_default_words / total_config_words` -- prefer compact configurations
- Named profiles: `balanced`, `throughput_first`, `area_power_first`, `deterministic_debug`
- Hard constraints are **never traded** for cost -- invalid mappings are hard-rejected

---

### Slide 15: Heterogeneous Scheduling Challenges and Solutions

**What Makes This Different from Classical Modulo Scheduling**

- **Challenge 1: Spatial + Temporal coexistence**
  - Same fabric has spatial PEs (1:1 mapping) and temporal PEs (N:1 time-multiplexed)
  - Scheduler must decide: dedicate a spatial PE or share a temporal PE slot?
  - Solution: unified candidate set includes both; cost model penalizes temporal pressure vs. spatial congestion

- **Challenge 2: Tag-driven scheduling (not slot-driven)**
  - Classical CGRA: VLIW-style, slot = time step in modulo schedule
  - Loom: tag = data-driven selector, execution order determined by token arrival
  - Solution: tag assignment is a constraint satisfaction problem over tag-width + uniqueness, not a time-slot allocation

- **Challenge 3: Coupled routing through temporal switches**
  - Different tags share physical switch ports but use different route-table entries
  - Routing and temporal assignment are coupled: route depends on tag, tag depends on placement
  - Solution: iterative refinement -- route with tentative tags, then fix tags, re-verify routes

- **Challenge 4: Memory port multiplexing via tags**
  - Multi-port memories use tags to identify logical ports; tag-matching at LSQ
  - Mapper must assign consistent tags across load/store PEs and their memory interfaces
  - Solution: tag propagation from memory port assignment backward through the datapath

- **Challenge 5: Register allocation in temporal PEs**
  - Limited register file shared across time-multiplexed instructions
  - FIFO-based registers: depth matters, not just count
  - Solution: graph-coloring-style allocation within temporal PE, respecting FIFO depth and single-writer constraint

---

## Part III: Evaluation and Comparison

### Slide 16: Experimental Setup

**Benchmarks, Architecture Configurations, and Methodology**

- **Benchmark suite**: 135+ kernels from diverse domains
  - Signal processing: conv1d, conv2d, FFT butterfly, autocorrelation, FIR
  - Linear algebra: axpy, dot product, gemv, matmul, Cholesky
  - Stencil: Jacobi, Gauss-Seidel, Laplacian, blur
  - Sorting/search: bitonic sort, binary search, merge step
  - ML: batchnorm, depthwise conv, softmax, ReLU
  - Reduction: cumsum, histogram, prefix sum
  - Sparse: delta encode/decode, compact, scatter/gather
- **Architecture configurations** (via ADG):
  - Spatial-only: 4x4 mesh, all spatial PEs
  - Temporal-only: 2x2 mesh, all temporal PEs (8 instruction slots each)
  - Heterogeneous: 4x4 mesh, mix of spatial + temporal PEs
  - Topology sweep: Mesh, Torus, DiagonalMesh
- **Metrics**: mapping success rate, PE utilization, routing congestion, config_mem size, cycle count (cosim)
- **Verification**: every mapped kernel runs through ESI cosim with CPU reference comparison

---

### Slide 17: Mapping Quality Results

**Placement, Routing, and Utilization Analysis**

- **Mapping success rate** by architecture:
  - Bar chart: spatial-only vs. temporal-only vs. heterogeneous vs. kernel complexity
  - Heterogeneous architecture maps the widest kernel set (temporal PEs absorb overflow)
  - Pure spatial fails on high-op-count kernels; pure temporal underutilizes simple kernels
- **PE utilization**:
  - Heatmap: per-tile utilization across 4x4 mesh for representative kernels
  - Temporal PE instruction slot fill rate: how many of N slots are used
  - Spatial PEs: near 100% when mapped, 0% otherwise (binary)
- **Routing congestion**:
  - Switch port utilization histogram
  - Temporal switch slot utilization vs. static switch
  - Impact of topology (Torus reduces max congestion by ~X% vs. Mesh)
- **Config memory footprint**:
  - Stacked bar: config_mem words breakdown by operation type (PE, switch, temporal PE, tag ops)
  - Temporal PEs dominate config_mem for complex kernels (instruction memory)

---

### Slide 18: Performance and Scalability

**Cycle-Accurate Cosimulation Results**

- **Cycle count comparison** (normalized to single-PE baseline):
  - Line chart across kernel complexity (number of software ops)
  - Spatial 4x4: best for simple kernels (no temporal overhead)
  - Heterogeneous 4x4: best for complex kernels (temporal PEs absorb more ops)
  - Temporal 2x2: competitive throughput despite fewer tiles (time-multiplexing)
- **Scalability study**:
  - Architecture sweep: 2x2, 3x3, 4x4, 6x6 mesh
  - Mapping time vs. architecture size (how fast is the mapper?)
  - Utilization vs. architecture size (diminishing returns at what point?)
- **Temporal PE efficiency**:
  - Operand buffer mode comparison: per-instruction vs. shared
  - Shared buffer reduces area but adds latency for high-slot-count configs
  - Register file utilization: how many inter-instruction registers are actually used?
- **End-to-end compilation time**:
  - Breakdown: frontend (Clang+MLIR) + mapper (P&R) + backend (codegen) + cosim
  - Mapper dominates for large kernels; frontend and codegen are fast

---

### Slide 19: Comparison with Related Work

**Positioning in the CGRA Compilation Landscape**

| Aspect | CGRA-ME | OpenCGRA | Pillars | Calyx/Filament | Loom |
|--------|---------|----------|---------|----------------|------|
| Input language | Restricted C | DFG | Custom | Custom IR | C/C++ with pragmas |
| Hardware description | Fixed XML | Fixed RTL | Chisel | Custom | ADGBuilder C++ API |
| IR foundation | Custom | Custom | FIRRTL | Custom IR | MLIR (Dataflow+Fabric) |
| Temporal sharing | Modulo schedule | Modulo schedule | No | Static schedule | Tag-based dispatch |
| Retargetable | Limited | No | Yes | Partial | Full (any topology) |
| Backend parity | No | RTL only | RTL only | RTL only | SystemC + RTL (verified parity) |
| Cosim framework | External | External | External | External | Integrated ESI |
| Scheduling model | ILP/heuristic | Simulated annealing | Manual | Compiler pass | Constraint-driven P&R |

- **vs. CGRA-ME/OpenCGRA**: Loom supports heterogeneous (spatial+temporal) tiles; tag-based scheduling vs. modulo scheduling
- **vs. Calyx/Filament**: Loom operates on C++ input directly; retargetable to arbitrary topologies via ADG
- **vs. HLS tools (Vitis/Catapult)**: Loom targets multi-core CGRA, not single-kernel pipeline; transparent config_mem vs. opaque bitstream

---

### Slide 20: Conclusion and Future Directions

**Summary**

- Loom is a **complete, MLIR-based compilation stack** for domain-specific CGRAs
  - C++ -> Handshake/Dataflow MLIR -> Mapper P&R -> Fabric MLIR -> SystemC/SystemVerilog
- **Tag-based scheduling** on heterogeneous fabrics: spatial PEs for throughput, temporal PEs for flexibility, temporal switches for resource-efficient routing
- **Constraint-driven mapping** with six hard-constraint classes and multi-objective cost optimization
- **Unified config_mem** provides transparent, atomic, per-module reconfiguration
- 135+ kernels validated end-to-end with bit-exact cosimulation

**Future Directions**

- ILP-based global placement for provably optimal solutions on small instances
- gem5 integration: full-system simulation with accelerator in the memory hierarchy
- Multi-kernel scheduling: time-share the CGRA across kernel sequences, partial config_mem update
- Physical design closure: ASIC synthesis flow, timing-driven placement feedback
- Auto-tiling and auto-parallelization for kernels exceeding single-fabric capacity

---

## Speaker Notes / Presentation Flow

**Part I: Framework (Slides 1-10, ~15 min)**
- Slides 1-2 (2 min): Title + motivation
- Slide 3 (2 min): Loom overview diagram
- Slide 4 (2 min): Pipeline stages -- the roadmap for the talk
- Slide 5 (2 min): Frontend + Dataflow dialect (compressed)
- Slide 6 (2 min): Fabric dialect (compressed)
- Slide 7 (1 min): ADG builder (brief)
- Slide 8 (2 min): Tagged types + temporal PE intro -- bridge to Part II
- Slide 9 (1 min): Backend + cosim (brief)
- Slide 10 (2 min): AXPY walkthrough -- concrete grounding before algorithm deep-dive

**Part II: Scheduling Algorithm (Slides 11-15, ~12 min)**
- Slide 11 (2 min): Formal problem definition
- Slide 12 (3 min): Constraint hierarchy C1-C6 -- this is the technical core
- Slide 13 (3 min): Three-phase algorithm with repair loop
- Slide 14 (2 min): Cost model and optimization objectives
- Slide 15 (2 min): Heterogeneous challenges -- what makes this different

**Part III: Evaluation (Slides 16-20, ~13 min)**
- Slide 16 (2 min): Experimental setup
- Slide 17 (3 min): Mapping quality results
- Slide 18 (3 min): Performance and scalability
- Slide 19 (3 min): Related work comparison table
- Slide 20 (2 min): Conclusion + future work

**Total: ~40 minutes + Q&A**
