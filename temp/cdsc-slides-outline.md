# Loom: A Compiler and Hardware Generator for Heterogeneous Reconfigurable Systems

## Slide Outline (12 slides + 3 backup)

### Part I: Why Graphs? The Framework (Slides 1-4, ~4 min)
### Part II: Mapping and Scheduling for Heterogeneous Fabrics (Slides 5-8, ~4 min)
### Part III: Results and Vision (Slides 9-12, ~4 min)

---

## Part I: Why Graphs? The Framework

### Slide 1: Title

**Loom: A Compiler and Hardware Generator for Heterogeneous Reconfigurable Systems**

- Subtitle: When Graphs and Streams Are First-Class Citizens
- Sihao Liu, Tony Nowatzki
- UCLA / CDSC Annual Review, Feb 2026

---

### Slide 2: The Representation Problem in Accelerator Design

**Why Existing Full-Stack Toolchains Break Down**

- Building a domain-specific accelerator requires a full stack: frontend, scheduling, hardware generation, verification
- Today's reality: each layer speaks a different language
  - HLS tools: C -> scheduled RTL (opaque, monolithic)
  - CGRA toolchains: ad-hoc DFG -> architecture-locked mapper -> fixed RTL
  - Verification: separate testbench, no shared semantics with compiler
- The root cause: **no shared representation** across the stack
  - A change in hardware topology forces rewriting the mapper
  - A change in the software IR breaks backend assumptions
  - Design-space exploration requires manual re-integration at every layer
- **Key observation**: both software kernels and hardware architectures are *graphs*
  - Software: dataflow graphs with streaming semantics
  - Hardware: resource graphs with connectivity and capacity constraints
  - Mapping: associating one graph onto another
- If we make graphs and dataflow streams **first-class citizens** in a unified IR, the entire accelerator compilation problem reduces to **graph transformations**

---

### Slide 3: The Graph-First Thesis

**Complexity Emerges from Connectivity, Not Fine-Tuning**

- Central claim: represent *everything* as native MLIR dialect operations on graphs
  - **Software graph** (DFG): Handshake + Dataflow dialect -- streaming ops, loop-carried deps, tagged types
  - **Hardware graph** (ADG): Fabric dialect -- PEs, switches, memories, temporal resources, connectivity
  - **Mapping**: graph-to-graph associative binding (placement + routing + temporal assignment)
- Three compilation problems become three classes of graph transformations:
  - *Software optimization / DSE*: graph mutation preserving execution semantics (dead-code elimination, constant folding, fork minimization -- all as graph rewrites)
  - *Hardware optimization / DSE*: topology graph transformation (tile replication, connectivity reshaping, temporal slot allocation -- all as graph rewrites)
  - *SW-HW mapping*: constrained graph-to-graph association (the scheduling problem)
- Why this matters:
  - Fully retargetable: any topology expressible as a Fabric graph is a valid compilation target
  - Unified type system: software ports and hardware ports share MLIR types -- mismatches caught at compile time
  - Composable: MLIR's pass infrastructure applies uniformly across both dialects
- No prior framework treats both software and hardware as native graph IRs in the same compiler infrastructure

---

### Slide 4: Loom Full-Stack Pipeline

**From C++ to Verified Multi-Core Hardware in One Framework**

- Diagram: two independent paths converging at the mapper

```
  Software Path (A)              Hardware Path (B)
  C++ with pragmas               ADGBuilder C++ API
  -> Clang -> LLVM IR            -> Fabric MLIR
  -> SCF -> Handshake            -> SystemC / SystemVerilog
  -> Dataflow MLIR (DFG)            (ADG)
          \                        /
           \                      /
        Mapper: Graph-to-Graph P&R (C)
           -> placement + routing
           -> temporal assignment
           -> config_mem bitstream
                    |
        Backend Realization (D)
           -> Configured SystemC / SV
                    |
        Verification (E)
           -> ESI cosim (standalone)
           -> gem5 full-system sim
```

- Two custom MLIR dialects: **Dataflow** (4 streaming primitives + tagged types) and **Fabric** (PEs, switches, memories, temporal resources)
- Tagged type system: `!dataflow.tagged<i32, i4>` -- tag-driven port multiplexing, instruction dispatch, memory disambiguation
- Dual backend from single Fabric MLIR: cycle-accurate SystemC + synthesizable SystemVerilog (verified parity)
- gem5-loom: CGRA device model in gem5 memory hierarchy for full-system evaluation
- 135+ kernels validated end-to-end with bit-exact cosimulation

---

## Part II: Mapping and Scheduling for Heterogeneous Fabrics

### Slide 5: The Heterogeneous Mapping Problem

**Why Mapping Onto Heterogeneous Reconfigurable Fabrics Is Hard**

- **Given**: DFG (software graph) + ADG (hardware graph) with heterogeneous tiles
  - Spatial PEs: fixed-function, 1:1 mapping, high throughput, low flexibility
  - Temporal PEs: time-multiplexed, N:1 mapping, flexible but resource-constrained (slots, registers)
  - Static switches (fixed routing) vs. temporal switches (multiplexed routing)
  - Heterogeneous memories: on-chip SRAM, external memory with LSQ
- **Find**: M = (placement, routing, resource assignment) satisfying all constraints, minimizing cost
- **Why this is fundamentally harder than classical CGRA mapping**:
  - Classical CGRA: homogeneous tiles, modulo scheduling over uniform time slots
  - Heterogeneous fabric: the scheduler must jointly decide -- dedicate a spatial PE *or* share a temporal PE slot? Each choice has cascading effects on routing and resource allocation
  - Placement, routing, and resource assignment are **coupled, not separable** -- a placement decision constrains which routes are legal, and routing choices constrain resource allocation
  - The search space grows combinatorially: (spatial vs. temporal placement) x (route alternatives) x (slot/register allocation)
- This is a **graph-to-graph mapping problem**: find a constrained association from the software DFG onto the hardware ADG

---

### Slide 6: Constraint-Driven Mapping Algorithm

**Three-Phase Place-and-Route with Iterative Repair**

- **Phase 1: Placement** -- assign each DFG node to a compatible ADG tile
  - Build candidate set per DFG node: operation semantics must match tile capability, port types must be compatible, capacity must not be exceeded
  - Topological-order traversal with cost-based tie-breaking (favor tiles near already-placed neighbors)
  - Key heterogeneous decision: spatial PE gives dedicated throughput; temporal PE shares resources but accommodates more operations per tile
- **Phase 2: Routing** -- route each DFG edge through legal hardware paths
  - BFS/Dijkstra on the ADG connectivity graph from source to destination tile
  - Legality checks at every hop: physical connectivity, type compatibility, directionality
  - Reserve switch ports; update capacity bookkeeping to prevent oversubscription
- **Phase 3: Resource Assignment** -- allocate temporal slots, registers, and configuration
  - For temporal PEs: assign instruction slot, opcode, register indices for inter-instruction data
  - For temporal switches: assign route-table entries
  - Graph-coloring-style register allocation within temporal PEs, respecting FIFO depth
- **Repair loop**: when a conflict is detected at any phase:
  - Escalation: reroute conflicting edges -> reassign conflicting nodes -> reassign resources -> bounded restart
  - Prevents getting stuck in local minima while keeping search tractable
- **Cost model**: multi-objective (placement pressure + routing congestion + resource utilization + throughput proxy)
- 6 classes of hard constraints (node/port/route/capacity/temporal/config) are **never relaxed**

---

### Slide 7: Enabling Heterogeneity -- Tagged Types and Resource Sharing

**The Mechanism That Makes Heterogeneous Mapping Possible**

- Core challenge: how do multiple logical operations share one physical tile and one physical interconnect?
- Loom's answer: **tagged types** (`!dataflow.tagged<value, tag>`) as a unified multiplexing mechanism
  - Port multiplexing: multiple logical streams share one physical port, demuxed by tag at the receiver
  - Instruction dispatch: temporal PE matches incoming tag -> selects operation + operand routing
  - Memory disambiguation: LSQ matches load/store transactions per tag
  - Route multiplexing: temporal switches use per-tag route-table entries on shared physical links
- This is not a separate scheduling dimension -- it is **integrated into the mapping algorithm**:
  - Placement decides spatial vs. temporal; if temporal, tag assignment follows from the placement
  - Routing through temporal switches naturally assigns route-table entries per tag
  - The mapper treats tag assignment as a consequence of placement/routing, not an independent phase
- Enables the key architectural tradeoff: spatial tiles for throughput-critical ops, temporal tiles for flexibility -- unified under one mapping framework

---

### Slide 8: Multi-Kernel Scheduling on Shared Fabric

**From Single-Kernel Mapping to Workload-Level Scheduling**

- Real workloads invoke multiple accelerated kernels in sequence (or concurrently on disjoint fabric regions)
- Each kernel is independently compiled to a DFG, but they share the same physical ADG
- **Config-mem switching**: each kernel produces a config_mem image; host writes new config between invocations
  - Unified config_mem format: 32-bit words, per-module aligned, AXI-Lite / TLM accessible
  - Fast reconfiguration: only overwrite changed modules (delta update)
- **Cross-kernel optimization opportunities**:
  - Kernels with complementary resource profiles can coexist on temporal PEs (kernel A uses slots 0-3, kernel B uses slots 4-7)
  - Disjoint spatial regions can execute different kernels concurrently
- **Full-system evaluation via gem5-loom**:
  - CGRA device model integrated into gem5 memory hierarchy
  - CPU drives kernel dispatch, config loading, DMA transfers, result verification
  - Captures realistic overheads: config write latency, cache effects, memory contention
  - Enables workload-level evaluation beyond isolated kernel benchmarks

---

## Part III: Results and Vision

### Slide 9: Experimental Setup

**135+ Kernels Across Diverse Domains**

- **Benchmark suite**:
  - Signal processing: conv1d, conv2d, FFT butterfly, FIR, autocorrelation
  - Linear algebra: axpy, dot product, gemv, matmul
  - Stencil: Jacobi, Gauss-Seidel, Laplacian, blur
  - ML: batchnorm, depthwise conv, softmax, ReLU
  - Sparse/irregular: delta encode/decode, scatter/gather, histogram
- **Architecture configurations** (all defined via ADGBuilder):
  - Spatial-only: 4x4 mesh, all spatial PEs
  - Temporal-only: 2x2 mesh, all temporal PEs (8 instruction slots each)
  - Heterogeneous: 4x4 mesh, mix of spatial + temporal PEs
  - Topology sweep: Mesh, Torus, DiagonalMesh
- **Methodology**:
  - Every mapped kernel verified via ESI cosim with CPU reference (bit-exact for integer, tolerance for FP)
  - Metrics: mapping success rate, PE utilization, routing congestion, cycle count, config_mem footprint

---

### Slide 10: Mapping Quality and Performance Highlights

**Heterogeneous Wins: The Best of Both Worlds**

- **Mapping success rate** (bar chart: spatial-only vs. temporal-only vs. heterogeneous):
  - Pure spatial fails on high-op-count kernels (PE exhaustion)
  - Pure temporal underutilizes simple kernels (unnecessary overhead)
  - Heterogeneous maps the widest kernel set -- temporal PEs absorb overflow from spatial
- **PE utilization** (heatmap across 4x4 mesh for representative kernels):
  - Spatial PEs: near 100% when mapped, 0% otherwise
  - Temporal PEs: instruction slot fill rate varies by kernel complexity
  - Heterogeneous achieves highest aggregate utilization
- **Cycle-accurate cosim results** (line chart, normalized):
  - Spatial 4x4: best throughput for simple kernels
  - Heterogeneous 4x4: best for complex kernels (more ops mapped per config)
  - Temporal 2x2: competitive despite fewer tiles (time-multiplexing)
- **Config memory footprint**: temporal PEs dominate for complex kernels (instruction memory); spatial kernels are compact

---

### Slide 11: Positioning in the Landscape

**What Loom Enables That Others Cannot**

- Comparison table (compact):

| | CGRA-ME | OpenCGRA | Calyx | Loom |
|---|---------|----------|-------|------|
| Input | Restricted C | DFG | Custom IR | C/C++ |
| HW desc | Fixed XML | Fixed RTL | Custom | ADGBuilder API |
| IR | Custom | Custom | Custom | MLIR native |
| Scheduling | Modulo (ILP) | Modulo (SA) | Static | Constraint-driven P&R |
| Heterogeneous | No | No | No | Spatial + Temporal |
| Retargetable | Limited | No | Partial | Any topology |
| Verification | External | External | External | Integrated ESI + gem5 |

- Key differentiators:
  - **Graph-native IR**: both SW and HW as MLIR dialects -- no ad-hoc intermediate formats
  - **Heterogeneous scheduling**: constraint-driven mapping across spatial + temporal tiles
  - **Full-system verification**: ESI cosim + gem5 integration (not just standalone kernel test)
- vs. HLS (Vitis/Catapult): Loom targets multi-core reconfigurable fabrics with transparent config; HLS produces opaque monolithic RTL

---

### Slide 12: Conclusion and Research Vision

**Graph-First Accelerator Compilation: A New Direction**

- **What we built**: Loom -- a complete MLIR-based stack from C++ to verified heterogeneous CGRA hardware
  - Two native MLIR dialects (Dataflow + Fabric) unify SW and HW representation
  - Constraint-driven mapping across heterogeneous spatial+temporal fabrics
  - 135+ kernels, end-to-end validated, dual-backend (SystemC + SystemVerilog)
  - gem5-loom for full-system accelerator simulation

- **The bigger thesis**: when graphs and streams are first-class citizens, complexity and functionality **emerge from connectivity**
  - We don't fine-tune individual hardware blocks or hand-optimize software mappings
  - We define the graph structure, and the compiler finds the right associations
  - This is a fundamentally different philosophy from both HLS and classical CGRA toolchains

- **Open research directions**:
  - ILP/SAT-based global placement for provably optimal mappings
  - Auto-tiling: partition large kernels across multiple fabric instances
  - Physical design closure: timing-driven feedback from synthesis into the mapper cost model
  - Cross-layer DSE: jointly explore software transformations and hardware topologies as coupled graph optimizations

---

## Backup Slides

### Backup 1: Dataflow and Fabric Dialect Details

**Dataflow Dialect Primitives**:
- `dataflow.stream` -- index generator with runtime-configurable predicate (`<, <=, >, >=, !=`)
- `dataflow.carry` -- loop-carried dependency (two-phase: init -> loop body)
- `dataflow.invariant` -- loop-invariant broadcast (consume once, replay per iteration)
- `dataflow.gate` -- stream alignment across region boundaries

**Fabric Dialect Operations**:
- `fabric.pe` -- Compute / constant / load-store / dataflow PE
- `fabric.temporal_pe` -- Time-multiplexed PE with instruction memory
- `fabric.switch` / `fabric.temporal_sw` -- Static / temporal routing
- `fabric.memory` / `fabric.extmemory` -- On-chip / external memory with LSQ
- `fabric.fifo` -- Buffering (bypassable, breaks combinational loops)
- `fabric.add_tag` / `fabric.del_tag` / `fabric.map_tag` -- Tag boundary ops

**Structure vs. Configuration separation**:
- Hardware params (latency, interval, port counts) fixed at build time
- Runtime config (tags, predicates, constants, route tables) populated by mapper via config_mem

---

### Backup 2: ADGBuilder API and Hardware Construction

```cpp
ADGBuilder builder("my_cgra");
auto pe  = builder.newPE("alu").setLatency(1).addOp("arith.addi", "arith.muli");
auto tpe = builder.newTemporalPE("shared").setNumInstruction(8).setNumRegister(4);
auto sw  = builder.newSwitch("router").setPortCount(4, 4);
builder.buildMesh(4, 4, pe, sw, Topology::Mesh);
builder.exportSV("output/sv");    // -> synthesizable SystemVerilog
builder.exportSysC("output/sysc"); // -> cycle-accurate SystemC
builder.exportDOT("output/viz");   // -> DOT visualization
```

- Single-source C++ API (no YAML/JSON/Python)
- Topologies: Mesh, Torus, DiagonalMesh, DiagonalTorus, or fully custom
- Clone-with-deduplication: structural equivalence detected, shared templates reused
- Multi-backend export from single Fabric MLIR

---

### Backup 3: Constraint Hierarchy (C1-C6)

- **C1: Node compatibility** -- software op semantics match hardware tile capability
- **C2: Port and type compatibility** -- native/tagged category match, value type match, tag-width uniformity
- **C3: Route legality** -- edges follow physical connectivity with correct directionality
- **C4: Capacity** -- hardware resources not oversubscribed (switch fan-in/out, temporal PE slots, memory queues)
- **C5: Temporal** -- tag uniqueness within temporal PE, slot range validity, register index bounds
- **C6: Configuration encoding** -- emitted config matches Fabric op spec (constant values, predicates, output tags)

All six classes are **hard constraints** -- never traded for cost optimization. Invalid mappings are unconditionally rejected.

---

## Speaker Notes / Presentation Flow

**Part I: Framework (Slides 1-4, ~4 min)**
- Slide 1 (15 sec): Title, quick intro
- Slide 2 (1.5 min): Motivation -- the representation fragmentation problem. Build to the key observation: everything is graphs. This is the "why" slide.
- Slide 3 (1.5 min): The thesis -- graph-first compilation. Three problems become three graph transformations. This is the intellectual core. Deliver with conviction: "no prior framework does this."
- Slide 4 (1 min): Pipeline overview -- quick walkthrough of the full stack. Don't dwell; this is orientation for the audience. Mention 135+ kernels and gem5 integration as credibility markers.

**Part II: Algorithm (Slides 5-8, ~4 min)**
- Slide 5 (1 min): The mapping problem -- why heterogeneous is fundamentally harder than homogeneous. Coupled dimensions, combinatorial search space. Set up the intellectual challenge.
- Slide 6 (1.5 min): The mapping algorithm -- three phases (placement, routing, resource assignment) + repair loop. This is the core algorithmic slide. Emphasize the heterogeneous placement decision (spatial vs. temporal) and that hard constraints are never relaxed.
- Slide 7 (30 sec): Tagged types as the enabling mechanism -- keep concise. One key diagram showing how tags enable multiplexing. Don't dwell; this supports the mapping story, not the other way around.
- Slide 8 (1 min): Multi-kernel scheduling -- the system-level view. Config-mem switching, gem5 integration. This bridges to evaluation.

**Part III: Results (Slides 9-12, ~4 min)**
- Slide 9 (30 sec): Setup -- quick mention of benchmark suite and configs.
- Slide 10 (1.5 min): Highlight results -- heterogeneous wins. Focus on 2-3 key charts. Don't enumerate all numbers.
- Slide 11 (1 min): Comparison table -- position Loom clearly. Emphasize the unique combination: graph-native IR + heterogeneous scheduling + integrated verification.
- Slide 12 (1 min): Conclusion + vision. End strong: "graphs and streams as first-class citizens -- complexity emerges from connectivity." Open research directions for poster discussion.

**Total: ~12 minutes + 3 min Q&A**

**Poster discussion hooks** (plant during talk for evening session):
- "Come to the poster to see a live demo of the full pipeline on AXPY"
- "Poster has detailed constraint hierarchy and cost model breakdown"
- "gem5 integration results with full-system workloads at the poster"
