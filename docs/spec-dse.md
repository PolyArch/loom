# FCC DFG-Domain and Four-Layer Exploration Specification

## Overview

FCC replaces Loom's pragma-led domain selection with a staged exploration model.
The exploration loop is intentionally structured so fast filters run before slow
mapping and simulation.

## Four Layers

### Layer 1: DFG-Domain and Loop Parameters

This layer decides what part of the SCF-level program should become the DFG.

The candidate space is defined by:

- a contiguous region that respects SCF hierarchy boundaries
- parameter choices such as unroll factor and parallel degree

For MVP cases, the implementation may select a single obvious region, but the
architecture must preserve the candidate abstraction.

### Layer 2: Tech-Mapping Co-Design

This layer matches DFG patterns against configurable `function_unit` DAGs.
It is FCC-specific because `fabric.static_mux` changes FU behavior.

Output:

- candidate FU matches
- selected FU internal configurations
- coverage and feasibility feedback to Layer 1

### Layer 3: Place and Route

This layer performs:

- placement
- routing
- congestion and legality checks
- PE mux/demux reconstruction
- switch route-table reconstruction

### Layer 4: Validation

This layer runs simulation or host-backed validation to confirm correctness and
collect performance data.

## Fast Rejection Contract

Layer 1 must support a lightweight feasibility check before full mapping.
The check may use resource estimates such as:

- arithmetic or control operation count
- memory port requirements
- widest operand width
- estimated FU demand

The exact heuristic may evolve, but the existence of a cheap pruning step is
part of the FCC architecture.

## Feedback Arcs

The exploration loop allows upward feedback:

- Layer 2 to Layer 1 when tech-mapping coverage is weak
- Layer 3 to Layer 2 or Layer 1 when routing or capacity fails
- Layer 4 to the lower layers when placement quality is poor or correctness
  fails under realistic execution

## Relationship to Other Specs

- Pipeline overview: [spec-fcc.md](./spec-fcc.md)
- Host integration: [spec-host-accel-interface.md](./spec-host-accel-interface.md)
- Mapper: [spec-mapper.md](./spec-mapper.md)
