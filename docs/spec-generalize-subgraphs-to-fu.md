# Generalize Subgraphs to FU

This document specifies the design of `loom-generalize-subgraphs-to-fu`, a
new MLIR pass that performs the **inverse** of the existing FU enumeration
pipeline. Given a set of `dataflow.subgraph` instances (the input "software
patterns"), the pass synthesizes a single `fabric.fu` (a "hardware
template") whose materialization, under
`loom-enumerate-fu-subgraphs`, is a **superset** of the input set.

The output `fabric.fu` represents a reconfigurable hardware block that can
be programmed (via `sw_configs`) to realize every input subgraph, plus
possibly additional configurations that fall out of the merged structure.

The canonical source for the inverse pipeline is:

* `lib/Fabric/Tech/EnumerateFuSubgraphsPass.cpp` (forward direction)
* `lib/Fabric/Tech/SubgraphEnumerator.cpp` (forward direction)
* `lib/Fabric/IR/FabricOps.cpp::hwShareGroups()` (sharing rules)
* `docs/spec-fabric-reconfigurable-op.md` (configuration axes)
* `docs/spec-fabric-hw-share-group.md` (sharing rules narrative)

## Background

Today the toolflow is:

```
dataflow.graph
   |  loom-partition-graph
   v
dataflow.subgraph  (many)
   |  loom-map-subgraph-to-fus
   v
matched against fabric.fu library  <-- library produced manually
   |  loom-enumerate-fu-subgraphs
   v
materialized dataflow.subgraph candidates per fabric.fu
```

Today the `fabric.fu` library is produced **manually**: an architect writes
each FU template by hand, listing what its `fabric.op`s can do, what
`fabric.mux`/`fabric.demux` ports look like, and what `hw_params` are
allowed. The proposed pass closes the loop in the opposite direction:

```
dataflow.subgraph (many)
   |  loom-generalize-subgraphs-to-fu     <-- this spec
   v
fabric.fu  (synthesized; covers every input subgraph)
```

The synthesized FU then re-enters the existing flow. Re-running
`loom-enumerate-fu-subgraphs` on the synthesized FU must produce a set
that contains every input subgraph (this is the correctness invariant,
verified at synthesis time).

## Goals and non-goals

### Goals

1. **Coverage correctness**: Every input subgraph in the same group is
   isomorphic to at least one materialization of the synthesized FU.
2. **Hardware-cost-minimality**: Among FUs that satisfy coverage, prefer
   the FU with the smallest weighted hardware cost (see CostModel below).
3. **Tiered scope**:
   * tier A: input subgraphs share an identical DAG topology; only the op
     identity at each node varies.
   * tier B: input subgraphs share a common skeleton with localized
     branch differences (extra/missing edges, fanout shape variation).
   * tier C: input subgraphs may have heterogeneous topology, including
     feedback edges (`dataflow.carry`, `dataflow.gate`,
     `dataflow.invariant`).
4. **Configurable strategy**: Four interchangeable algorithms (anchor,
   mcs, incremental, incremental_random) selectable by config; mirrors
   the existing `Partitioner/` plug-in family.
5. **Parallelism as a first-class concern**: cross-group, intra-strategy,
   and verification parallelism enabled by default with safe defaults.
6. **End-to-end self-verification**: Default-on coverage check using the
   existing `SubgraphEnumerator` + `SubgraphMatcher` (no separate
   coverage prover).

### Non-goals

* The pass does **not** decide partitioning of a `dataflow.graph` into
  subgraphs (`loom-partition-graph` already does that).
* The pass does **not** emit `fabric.fifo`. FIFOs are inserted later by
  scheduling/buffering passes outside this pass.
* The pass does **not** synthesize multiple FUs from a single input
  group. One group in, at most one FU out (or one failure marker).
* The pass does **not** perform schematic floorplan / area extraction;
  the CostModel is an analytic weighted-resource formula, not an EDA
  call.
* The pass does **not** rewrite the input subgraphs.

## Glossary

* **input subgraph**: a `dataflow.subgraph` operation contained in a
  `func.func` body of the input module. Represents one observed software
  pattern that the FU must support.
* **synth group**: a set of input subgraphs intended to be covered by one
  synthesized FU. Identified by a string-valued attribute
  `loom.synth_group` on the enclosing `func.func`. Subgraphs without the
  attribute belong to the implicit `"default"` group.
* **synthesized FU**: the output `fabric.fu` produced for one synth
  group, appended to the module under a fresh symbol name.
* **alignment**: a partial mapping between nodes/edges of two or more
  input subgraphs that identifies which positions correspond and may
  therefore be merged into the same `fabric.op` / port.
* **anchor**: a structurally distinguished node (typically the producer
  feeding `dataflow.yield`, or a `dataflow.carry` head) used as a fixed
  pivot for alignment.
* **share group**: a multi-member hardware-share group as defined by
  `hwShareGroups()` in `lib/Fabric/IR/FabricOps.cpp`. Two ops can occupy
  the same `fabric.op.op_list` only if they belong to the same share
  group **and** their data-path bit-widths match.

## End-to-end interface

### Pass

```
Pass:     loom-generalize-subgraphs-to-fu
Scope:    ModuleOp
Inputs:   any number of func.func, each containing exactly one
          dataflow.subgraph in its body, optionally annotated with
            loom.synth_group = "<group_name>"
          Functions without the attribute belong to the "default" group.
          Functions whose body shape violates the "exactly one subgraph"
          rule are rejected with loom.synth_failed = "invalid_input".
Output:   the same module, with one wrapper func.func (containing one
          fabric.fu) appended per group that synthesized successfully,
          and loom.synth_failed string attribute on input func.func of
          any group that failed.
Options:  config=<path>            -- YAML or TOML SynthConfig file;
                                       same option name as
                                       loom-partition-graph-into-subgraphs
                                       for consistency
          fail-as-error=<bool>     -- escalate warnings to errors
                                       (default: false)
          dump-stats=<bool>        -- print one-line per-group stats
                                       (strategy, cost, coverage,
                                        reason) as remarks; consumed by
                                       lit tests (default: false)
```

The pass is registered alongside the existing tech-mapping passes in the
`loom` driver (`tools/loom/`). No new top-level binary is added; the test
helper `tools/loom-synth-fu-dump/` (parallel in spirit to
`tools/loom-template-dump/`) exists only to print synthesized FUs in a
stable format for lit tests, and is not part of the production pipeline.

### Acceptance criteria for the pass

1. **Empty input**: a module with no `dataflow.subgraph` produces an
   unchanged module and a `remark` diagnostic per pass invocation.
2. **Single subgraph**: a module with one input subgraph produces a
   module that, when fed through `loom-enumerate-fu-subgraphs` followed
   by `loom-map-subgraph-to-fus`, matches the original subgraph against
   the synthesized FU successfully.
3. **Multi-group module**: subgraphs annotated with distinct
   `loom.synth_group` values produce one independent FU each. The output
   module is order-deterministic across runs (subgraph processing order
   is the lexical order of group names).
4. **Coverage invariant**: the synthesized FU, after enumeration, has at
   least one materialized candidate isomorphic to each input subgraph
   in its group. This invariant is verified by default (see
   CoverageVerifier).
5. **Failure isolation**: failure to synthesize one group does not
   prevent synthesis of other groups. Failed groups have their input
   `func.func`s annotated with `loom.synth_failed = "<reason>"` (see
   failure enumeration below).
6. **Idempotence on synthesized output**: the synthesized wrapper
   function symbol name for group `<g>` is `@fu_<sanitized(g)>` where
   `sanitized` replaces any character outside `[A-Za-z0-9_]` with `_`.
   The inner `fabric.fu` carries no name. Rerunning the pass on a
   module that already contains a top-level `func.func` symbol with
   that name is a no-op for that group, emitting a `remark`. Name
   collisions with non-synthesized functions of the same name are
   reported as `symbol_conflict` failures.
7. **Output IR validity**: every emitted wrapper function passes the
   MLIR verifier; the inner `fabric.fu` passes `FuOp::verify` (which
   restricts the body to `fabric.op` / `fabric.mux` / `fabric.demux`)
   and every nested `fabric.op` passes `OpOp::verify` (which
   transitively enforces `hwShareGroups()` rules). The pass invokes
   the verifier on the freshly built FU before splicing it into the
   module; a verifier failure is reported as `verifier_failed` with a
   diagnostic.
8. **Input validation**: each input `func.func` must contain exactly
   one `dataflow.subgraph` operation in its body. Functions with zero
   or with more than one subgraph are skipped, the function is
   annotated with `loom.synth_failed = "invalid_input"`, and a
   `warning` is emitted.

### IR conventions

#### Input

```mlir
// Group "alu_int_32"
func.func @pattern_addi_subi() attributes {loom.synth_group = "alu_int_32"} {
  %g = dataflow.subgraph (...) -> (...) {
    ...
    %y = arith.addi %a, %b : i32
    dataflow.yield %y : i32
  }
  return
}

func.func @pattern_subi() attributes {loom.synth_group = "alu_int_32"} {
  %g = dataflow.subgraph (...) -> (...) {
    ...
    %y = arith.subi %a, %b : i32
    dataflow.yield %y : i32
  }
  return
}

// Group default
func.func @pattern_loose_floor() {
  %g = dataflow.subgraph (...) -> (...) {
    ...
    %y = math.floor %x : f32
    dataflow.yield %y : f32
  }
  return
}
```

#### Output

`fabric.fu` is not a top-level symbol op; it is `IsolatedFromAbove`,
must be wrapped inside another op (typically a `func.func`), and its
body permits only `fabric.op`, `fabric.mux`, `fabric.demux`, and the
`fabric.yield` terminator. The pass therefore appends one wrapper
function per group, naming the function (not the inner `fabric.fu`)
after the group:

```mlir
// New: synthesized wrapper per group, appended to module. The function
// symbol name is the canonical FU identity used downstream by
// loom-map-subgraph-to-fus.
func.func @fu_alu_int_32_addi_subi(%a: !fabric.bits<32>,
                                   %b: !fabric.bits<32>)
                                  -> !fabric.bits<32> {
  %y = fabric.fu(%aa = %a : !fabric.bits<32>,
                 %bb = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %r = fabric.op [@arith.addi, @arith.subi] (%aa, %bb)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %r : !fabric.bits<32>
  }
  return %y : !fabric.bits<32>
}

// Failed groups have their input func.func annotated:
func.func @pattern_x() attributes {
  loom.synth_group = "loose",
  loom.synth_failed = "cross_share_group"
} { ... }
```

The synthesized FU never contains `fabric.fifo`. FIFO insertion is the
responsibility of downstream passes. The `fabric.fu` body conforms to
`FuOp::verify`: only `fabric.op`, `fabric.mux`, `fabric.demux` between
the entry block and `fabric.yield`. Any state-bearing dataflow op
(e.g. `dataflow.carry`, `dataflow.stream`, `dataflow.invariant`,
`dataflow.gate`) is realized as `fabric.op [@dataflow.<op>]` inside the
body.

#### Failure reasons (closed enumeration)

* `cross_share_group` -- tier A required to merge ops across distinct
  share groups, and `allow_intra_position_mux` was disabled.
* `topology_mismatch` -- tier B could not express a structural
  difference using `fabric.mux`/`fabric.demux`.
* `feedback_align_conflict` -- tier C found incompatible flow signatures
  on cyclic SCCs (e.g. `fabric.op[@dataflow.carry]` heads driven by
  incompatible `dataflow.stream` parameter sets).
* `timeout` -- a strategy exceeded its `timeout_sec` budget.
* `resource_exhausted` -- a strategy generated more candidates than its
  `candidate_cap`.
* `unsupported_op` -- an input subgraph contains a software op not
  supported by `fabric.op` (per `opSchemas()` in
  `lib/Fabric/IR/FabricOps.cpp`); for example `dataflow.load`,
  `dataflow.store`, `dataflow.graph`, `arith.constant`, `ub.poison`.
* `invalid_input` -- the enclosing `func.func` does not contain
  exactly one `dataflow.subgraph`, or a subgraph signature is
  ill-typed.
* `verifier_failed` -- the synthesized FU did not pass MLIR's verifier
  (`FuOp::verify` or a nested `OpOp::verify`). Indicates a compiler
  bug; the FU is dropped, no IR is appended.
* `symbol_conflict` -- the wrapper function symbol name
  `@fu_<sanitized(group)>` already exists in the module and does not
  correspond to a previous synthesizer run that we may safely skip.
* `config_parse_failed` -- the `--config=<path>` file failed to load.

These failure reasons are stored verbatim as the `loom.synth_failed`
attribute on the offending input function. Implementations must keep
this list in lockstep with a `SynthFailureReason` C++ enum to enable
exhaustive `switch` checks.

## Module architecture

### Directory layout

```
include/Fabric/Tech/Synthesizer/
  Synthesizer.h         -- abstract base + factory
  CostModel.h           -- weighted-resource hardware cost
  CoverageVerifier.h    -- enumerate + match check, default on
  Alignment.h           -- DAG correspondence utilities
  Anchor.h              -- strategy: anchor-driven BFS
  MCS.h                 -- strategy: maximum common subgraph
  Incremental.h         -- strategy: seed + incremental merge
  IncrementalRandom.h   -- strategy: incremental + random restarts
  Parallel.h            -- shared thread-pool primitives
  Hwsg.h                -- thin wrapper over public HwShareGroup API

lib/Fabric/Tech/Synthesizer/
  Synthesizer.cpp
  CostModel.cpp
  CoverageVerifier.cpp
  Alignment.cpp
  Anchor.cpp
  MCS.cpp
  Incremental.cpp
  IncrementalRandom.cpp
  Parallel.cpp
  Hwsg.cpp

lib/Fabric/Tech/GeneralizeSubgraphsToFuPass.cpp
include/Fabric/Tech/Passes.h          -- add registration + Options struct

include/Common/HwShareGroup.h         -- public API extracted from
                                          FabricOps.cpp (see below)
lib/Common/HwShareGroup.cpp

include/Common/SynthConfig.h          -- YAML/TOML schema mirror
lib/Common/SynthConfig.cpp

tools/loom-synth-fu-dump/             -- test-only helper binary
test/techmap/synth/                   -- lit tests, see Test plan
```

### HwShareGroup public API extraction

`hwShareGroups()` and `findShareGroup()` currently live as
file-`static` helpers in `lib/Fabric/IR/FabricOps.cpp`. They are
single-source for sharing decisions and `OpOp::verify` already
enforces them on hand-written FUs. Extract them to a public header
without changing semantics:

```cpp
// include/Common/HwShareGroup.h
namespace loom::common {

// Returns the canonical multi-member share group table. Singleton ops
// (any op not appearing in the table) are implicitly their own group.
llvm::ArrayRef<llvm::DenseSet<llvm::StringRef>> hwShareGroups();

// Returns the index of the multi-member share group containing `name`,
// or std::nullopt if `name` is a singleton (or unknown).
std::optional<size_t> findShareGroup(llvm::StringRef name);

// Returns true iff `a` and `b` belong to the same share group (or both
// are the same singleton).
bool sameShareGroup(llvm::StringRef a, llvm::StringRef b);

} // namespace loom::common
```

`FabricOps.cpp` rewrites its file-local helpers as one-liners forwarding
to `loom::common::*`. Behavior identical, no PDK YAML, no override
mechanism: there is one canonical table; future PDK overrides are out of
scope.

### SynthConfig schema

```yaml
synth:
  strategy: incremental_random      # anchor | mcs | incremental | incremental_random
  parallelism:
    cross_group: true               # parallel across loom.synth_group values
    workers: auto                   # std::thread::hardware_concurrency()
  coverage_verifier:
    parallel_match: true            # parallelize across input subgraphs
  fallback_chain: []                # optional list of strategies to try
                                    #   in order on failure
  cost:
    mux_penalty: 1.5                # multiplier on mux area
    demux_penalty: 1.5
    carry_penalty: 2.0              # register area weight
  anchor:
    allow_intra_position_mux: false # tier A behavior, see Q11
  incremental:
    input_order_heuristic: largest_first
                                    # largest_first | smallest_first
                                    # | random_seeded
                                    # ordering metric: total node count
                                    # in the dataflow.subgraph body
                                    # (ties broken by lexical func name)
    coverage_verify_each_attempt: true
                                    # if true, each candidate FU after
                                    # an extend_to_cover sub-attempt is
                                    # verified for full back-coverage of
                                    # all previously folded inputs
  incremental_random:
    restarts: 16
    seed: 42
    input_order_heuristic: random_seeded
  mcs:
    timeout_sec: 60
    branch_workers: 8
    candidate_cap: 1000000
  scc_full_unroll: false            # tier C: see Q13
  subgraph_share_recurse: false     # tier B: see Q12
```

The schema is loaded into a `SynthConfig` C++ struct that mirrors
`TechMapConfig` in `include/Common/Config.h`. The pass exposes it via
the standard MLIR pass-option mechanism as `config=<path>`, identical
to the existing `loom-partition-graph-into-subgraphs` convention. An
unsupplied or empty path uses built-in defaults equivalent to the YAML
above. A failure to parse the file is reported as
`config_parse_failed` and aborts the pass.

## Strategies

All strategies implement the same abstract interface:

```cpp
// include/Fabric/Tech/Synthesizer/Synthesizer.h
namespace loom::fabric::tech {

struct SynthInputs {
  llvm::StringRef groupName;
  // one entry per input subgraph in this group; ownership not transferred
  llvm::ArrayRef<dataflow::SubgraphOp> subgraphs;
  const SynthConfig &config;
  mlir::MLIRContext *context;
};

struct SynthResult {
  // success: ownership of a freshly built fabric.fu (detached, caller
  // inserts into the module). nullptr on failure.
  mlir::OwningOpRef<fabric::FuOp> fu;
  // empty on success; one of the closed failure-reason enums on failure.
  llvm::StringRef failureReason;
  // coverage report (populated by CoverageVerifier when enabled)
  CoverageReport coverage;
  // diagnostics emitted during synthesis (informational)
  llvm::SmallVector<std::string, 4> notes;
};

class Synthesizer {
public:
  virtual ~Synthesizer() = default;
  virtual SynthResult run(const SynthInputs &) = 0;
};

// Factory: looks up SynthConfig.strategy and constructs the right
// concrete subclass.
std::unique_ptr<Synthesizer>
makeSynthesizer(llvm::StringRef strategyName, const SynthConfig &);

} // namespace loom::fabric::tech
```

### Tier coverage matrix

| Strategy            | Tier A | Tier B | Tier C | Strength                              | Cost                                     |
|---------------------|:------:|:------:|:------:|---------------------------------------|------------------------------------------|
| anchor              |  yes   | partial|  no    | fast, deterministic                   | cannot align disjoint topologies         |
| mcs                 |  yes   |  yes   |  yes   | provably-best alignment               | exponential worst case                   |
| incremental         |  yes   |  yes   |  yes   | wall-time linear in N inputs (verify amortized via cache)  | order-sensitive                          |
| incremental_random  |  yes   |  yes   |  yes   | best cost via random restarts         | wall-time scales with restart count      |

`anchor` covers tier B in the restricted case where differences are
single-edge insertions/deletions handled by local mux/demux when
`allow_intra_position_mux=true`.

### Strategy: anchor (tier A by default)

#### Idea

When all input subgraphs are topology-isomorphic, the **list of values
yielded by `dataflow.yield`** gives an ordered set of fixed anchors
(one per yield operand). For each anchor index, the producer is a
**Source**, which is one of:

* `BodyOp(op, result_index)` -- a value produced by an op in the
  subgraph body (possibly multi-result, e.g. `dataflow.stream` /
  `dataflow.gate`).
* `BlockArg(arg_index)` -- a value live-in to the subgraph (one of the
  subgraph's external operands).
* `BackEdge(carry_op, result_index)` -- in graph-region bodies, a value
  produced by an op that is also a downstream consumer; identified by
  the matcher's `GraphView` SCC pre-pass.

Alignment proceeds in lock-step on the Source position across all
input subgraphs simultaneously. The same `Source` abstraction (and the
same SCC pre-pass) is borrowed from `SubgraphMatcher`'s `GraphView` so
that anchor / mcs / incremental share one source-resolution
implementation.

#### Pseudocode

```
function synthesize_anchor(inputs):
    sgs = inputs.subgraphs
    yield_arities = [yield_operand_count(sg) for sg in sgs]
    if not all_equal(yield_arities):
        return failure("topology_mismatch")

    // anchors_per_index[k] = [Source for sg_i's k-th yield operand]
    anchors_per_index = build_yield_anchors(sgs)

    fu = empty_wrapper_with_inputs(union_block_args(sgs))
    visited = {}      // dedup so DAG fanout is not double-emitted
    pending = anchors_per_index[*]    // queue all anchor sources

    while pending not empty:
        s0_peers = pending.pop()      // [Source_0, ..., Source_{N-1}]
        if s0_peers in visited:
            wire_existing(visited[s0_peers])
            continue
        kind = unify_kind(s0_peers)   // BodyOp | BlockArg | BackEdge
        if kind == BlockArg:
            wire_to_input_port(s0_peers)
            visited[s0_peers] = port
            continue
        if kind == BackEdge:
            // Reserve unrealized_conversion_cast placeholder; resolved
            // when the SCC head emits its result.
            place = reserve_back_edge_placeholder(s0_peers)
            visited[s0_peers] = place
            continue
        // kind == BodyOp
        op_set = {op_name(s.op) for s in s0_peers}
        bw_set = {bitwidth(s.op.result(s.result_index)) for s in s0_peers}
        if len(bw_set) != 1:
            return failure("topology_mismatch")
        if not all_same_arity(s0_peers):
            return failure("topology_mismatch")
        decision = decide_op_node(op_set, bw=single(bw_set), config)
            // decide_op_node options (ranked by CostModel):
            //   single_share_group => one fabric.op{op_list = op_set}
            //   cross_share_group  => if allow_intra_position_mux:
            //                            split into per-share-group
            //                            fabric.ops merged through a
            //                            local fabric.mux; pick the
            //                            cheapest legal layout
            //                         else:
            //                            return failure("cross_share_group")
        if decision is None: return failure("topology_mismatch")
        emit decision into fu                  // returns produced Value
        visited[s0_peers] = decision.value
        for each operand index i of s0_peers[0].op:
            child_peers = [operand_source(s.op, i) for s in s0_peers]
            pending.push(child_peers)

    resolve_back_edge_placeholders(fu, visited)
    finalize_yield(fu, anchors_per_index)
    return success(fu)
```

`decide_op_node` ranks every legal share-group layout (single-group +
optional discard / disconnect mux modes) by `CostModel::evaluate` on a
hypothetical sub-FU and returns the lowest-cost legal candidate. This
replaces "first success wins" everywhere in synthesis; ties are broken
by stable structural id.

#### Acceptance criteria (anchor)

1. Two `arith.addi`-shaped i32 subgraphs with identical topology produce
   one `fabric.op{op_list=[arith.addi]}` of width 32.
2. One `arith.addi` and one `arith.subi`, both i32, identical topology,
   produce one `fabric.op{op_list=[arith.addi, arith.subi]}` (same
   share group `arith.addi/subi`).
3. One `arith.addi` (i32) and one `arith.muli` (i32): different share
   groups. With `allow_intra_position_mux=false`, fail with
   `cross_share_group`. With `=true`, produce two parallel `fabric.op`s
   merged through an inserted `fabric.mux`.
4. Two i32 subgraphs and one i64 subgraph at the same topology position
   fail with `topology_mismatch` regardless of the mux flag (bit-width
   is part of the share-group identity per Q5).

### Strategy: mcs (all tiers)

#### Idea

Enumerate maximum common edge subgraphs (MCES) shared by every input
and use each candidate as a FU skeleton. Bypass each input's
edges/nodes outside the MCS through a per-input
`fabric.demux`/`fabric.mux` shell, with mode bits choosing which input
pattern is realized. Mux/demux modes follow the enumerator semantics:
`sel` for normal selection, `discard` for ports drained by hardware
(consumed without producing a software value), `disconnect` for ports
that block the configuration entirely; `decide_mux_modes` chooses the
cheapest mode tuple that legalizes a given input. The output is the
**legal candidate FU with the lowest `CostModel::evaluate` score**;
"maximum common subgraph" is the search structure, not a guarantee of
hardware-area optimality (which depends on mux/demux costs and
duplicated datapaths).

This problem is NP-hard. Mitigations:

* prune by share-group compatibility on candidate node mappings;
* prune by bit-width compatibility;
* split SCC heads first (Q13: flow-signature equivalence) so feedback
  alignment is decided before pure-DAG MCS work begins;
* parallel branch-and-bound across `branch_workers`;
* `timeout_sec` and `candidate_cap` are hard caps;
* stop early when a candidate matches CostModel lower bound.

#### Pseudocode (high-level)

```
function synthesize_mcs(inputs):
    sgs = inputs.subgraphs
    sccsets = [scc_decompose(sg) for sg in sgs]
    pre_align_sccs(sccsets)             // Q13: signature heuristic
                                        // or full unroll if scc_full_unroll
    seed_pairs = candidate_node_seeds(sgs)  // share-group + width compatible
    best = None
    parallel_for seed in seed_pairs:
        cand = branch_and_bound_extend(seed,
                                       prune=share_group_and_width,
                                       cap=config.mcs.candidate_cap,
                                       deadline=config.mcs.timeout_sec)
        if cand is timeout: continue
        fu = build_fu_from_mces(cand, sgs)
        if best is None or cost(fu) < cost(best):
            best = fu
    if best is None:
        return failure("timeout" or "topology_mismatch")
    return success(best)
```

#### Acceptance criteria (mcs)

1. On a tier-A workload (all inputs isomorphic), mcs produces a FU
   whose CostModel score is `<=` the anchor strategy's score on the
   same input.
2. On `(a+b)*c` and `(a+b)` mixed inputs, mcs identifies the shared
   `arith.addi` skeleton and bypasses the multiplication via a single
   `fabric.mux`.
3. On a 2-input workload reaching `candidate_cap`, mcs returns
   `resource_exhausted`.
4. Per `parallel_match=true`, coverage verification of the best
   candidate runs in parallel across input subgraphs.

### Strategy: incremental (all tiers)

#### Idea

Treat synthesis as a left-fold over inputs:

```
fu_0 = trivial FU built from input_0 (one fabric.op per node)
for i in 1..N:
    if coverage_verifier(fu_{i-1}, input_i):
        fu_i = fu_{i-1}                  // already covered, no-op
    else:
        fu_i = extend_to_cover(fu_{i-1}, input_i)
        verify fu_i covers all of input_0..i  // cheap with cache
return fu_N
```

`extend_to_cover` is the only routine that mutates the FU. It generates
candidate extensions of three kinds and ranks **all candidates** by
`CostModel::evaluate` on the resulting FU; it returns the lowest-cost
legal candidate. (First-success is **not** used: it can permanently
foreclose lower-cost future sharing.)

1. **op-list widen**: widen some `fabric.op.op_list` (within share
   group + width) so input_i's op identity at that position becomes a
   member.
2. **mux/demux insert**: insert a `fabric.mux` (output side) or
   `fabric.demux` (input side) at a diff site so input_i's branch
   becomes one configurable arm. The candidate generator enumerates
   `(sel, discard, disconnect)` mode tuples per port consistent with
   the enumerator's drain semantics:
     * a `sel`-mode arm is produced and consumed by software;
     * a `discard`-mode arm drains the producer's value in hardware
       without a software consumer (used for fanout where one branch
       is structurally unused in this configuration);
     * a `disconnect`-mode arm makes the configuration invalid
       (used to prune cross-configuration interference).
3. **structural extend (tier C only)**: graft a new sub-FU for the
   diff region, including `fabric.op[@dataflow.carry]` SCC bodies if
   needed; gated by tier detection (only attempted if the diff has a
   back-edge in the alignment).

#### Pseudocode

```
function synthesize_incremental(inputs):
    order = sort_inputs(inputs.subgraphs,
                        by=config.incremental.input_order_heuristic)
    fu = build_trivial_fu(order[0])
    covered = [order[0]]
    for sg in order[1:]:
        if coverage_verifier.is_covered(fu, sg):
            covered.append(sg)
            continue
        candidates = []
        candidates += widen_oplist_candidates(fu, sg, config)
        candidates += insert_mux_demux_candidates(fu, sg, config)
        if has_backedge_in_diff(fu, sg):
            candidates += structural_extend_candidates(fu, sg, config)
        legal = []
        for cand in candidates:
            if not verifier_passes(cand): continue
            if config.incremental.coverage_verify_each_attempt:
                if not all_covered(cand, covered + [sg]): continue
            legal.append(cand)
        if not legal:
            return failure(reason_from_attempts(candidates))
        fu = min(legal, key=cost_model.evaluate)
        covered.append(sg)
    return success(fu)
```

#### Acceptance criteria (incremental)

1. On a tier-A workload presented in already-isomorphic order, the
   resulting FU is identical to the anchor strategy's output.
2. On `(a+b)`, then `(a+b)*c`: after the first input the FU has one
   `arith.addi`. After the second input the FU has the addi feeding a
   `fabric.demux` whose two arms are (a) directly to yield, (b) feeding
   a multiplier whose other operand is a new input port.
3. On 100 randomly permuted tier-A inputs, the result is identical
   regardless of order (lemma: tier-A op-list widen commutes).
4. With `coverage_verify_each_attempt=false`, a final verify still catches
   any extension bug; the test demonstrates this by injecting a known
   buggy `extend_to_cover` and asserting that final verify fails.

### Strategy: incremental_random (all tiers)

#### Idea

Wrap `incremental` in a multi-restart driver. The order in which inputs
are folded matters for FU cost (different orders may produce
structurally different FUs even when both are correct). Run `restarts`
independent permutations in parallel and return the lowest-cost
successful FU.

#### Pseudocode

```
function synthesize_incremental_random(inputs):
    rng = seeded_rng(config.incremental_random.seed)
    permutations = [random_permutation(inputs, rng)
                    for _ in 0..config.incremental_random.restarts]
    parallel_for perm in permutations:
        result = synthesize_incremental(perm)
        if result.success: candidates.append(result.fu)
    if no candidates: return failure(merge_reasons())
    return success(min(candidates, key=cost))
```

#### Acceptance criteria (incremental_random)

1. Determinism: same `seed` produces the same set of permutations and
   thus the same chosen FU (modulo non-determinism in candidate cost
   ties, which are broken by stable input ordering).
2. Improvement: on a hand-crafted workload where order matters, the
   chosen FU has cost `<=` the cost of `incremental` with the default
   `largest_first` heuristic. This is a regression check, not a
   guaranteed bound.
3. Parallel speedup: with `restarts=16` and `workers=8`, wall time is
   no worse than `2 *` single-restart wall time + epsilon (validated
   via perf test).

## Sub-algorithms shared by strategies

### Alignment

`Alignment.h` is a thin facade over the existing
`SubgraphMatcher::GraphView` data model so that synthesis and matching
agree on what "the same Source position" means across DAG fanout,
multi-result ops, block arguments, commutative operands, graph-region
back-edges, and yield wiring. Synthesis re-uses GraphView's SCC
pre-pass and source descriptors verbatim; this avoids inventing a
parallel hashing/iteration order that would diverge from the matcher.

```cpp
// Source descriptor: how a value is produced inside a subgraph.
// Identical in semantics to SubgraphMatcher::GraphView::Source.
//   BodyOp:    op.result(resultIndex)
//   BlockArg:  the subgraph's argIndex-th block argument
//   BackEdge:  a graph-region back-edge into op (resolved by SCC
//              pre-pass; consumed by reserve/resolve placeholder
//              passes during emission)
struct Source {
  enum Kind { BodyOp, BlockArg, BackEdge } kind;
  mlir::Operation *op;     // BodyOp / BackEdge
  unsigned resultIndex;    // BodyOp / BackEdge
  unsigned argIndex;       // BlockArg
};

// A node signature collapses op identity, share-group id, bit-width,
// arity, and operand-source kinds (NOT operand identity). Two subgraph
// positions are alignable iff their signatures match AND their per-operand
// source kinds match positionally (commutative operands are normalized
// upstream by GraphView's canonicalization).
//
// `op` borrows from the MLIR registered-name interning pool, so it
// outlives any single pass invocation. NodeSignature is trivially
// copyable and safe to cache across thread boundaries.
struct NodeSignature {
  llvm::StringRef op;
  std::optional<size_t> shareGroup;
  unsigned bitwidth;
  unsigned arity;
  llvm::SmallVector<Source::Kind, 4> operandKinds;
  uint64_t structuralHash;   // stable, deterministic
};
NodeSignature signatureOf(Source);

// Yield anchors: the ordered list of Source descriptors corresponding
// to dataflow.yield's operands. This is the canonical entry point for
// anchor / mcs / incremental alignment.
llvm::SmallVector<Source, 4>
yieldAnchors(dataflow::SubgraphOp sg);
```

### CoverageVerifier

```cpp
// include/Fabric/Tech/Synthesizer/CoverageVerifier.h
namespace loom::fabric::tech {

struct CoverageReport {
  // For each input subgraph, the index of a materialized FU candidate
  // that matches it, or std::nullopt on miss.
  llvm::SmallVector<std::optional<size_t>, 8> matchIndex;
  bool allCovered() const;
};

class CoverageVerifier {
public:
  CoverageVerifier(const SynthConfig &);
  // Materializes `fu` by:
  //   1. Constructing a scratch ModuleOp owned by this verifier.
  //   2. Cloning the wrapper func.func + fabric.fu into the scratch
  //      module (so SubgraphEnumerator's append behavior does not
  //      pollute the user's module).
  //   3. Invoking SubgraphEnumerator::enumerateFuSubgraphs on the
  //      scratch module.
  //   4. Matching each input subgraph against the appended candidates
  //      with SubgraphMatcher (parallel per `parallel_match`).
  //   5. Discarding the scratch module deterministically before return.
  CoverageReport verify(fabric::FuOp fu,
                        llvm::ArrayRef<dataflow::SubgraphOp> inputs);
};

} // namespace loom::fabric::tech
```

The verifier is implemented in terms of the existing
`SubgraphEnumerator` and `SubgraphMatcher`. It does not embed a
hand-written coverage proof; the enumerator is treated as the
authoritative oracle for "what an FU can become."

For each input subgraph we evaluate isomorphism against materialized
candidates. Per `parallel_match=true`, we shard inputs across workers
and short-circuit per input on first match.

### SCC handling for tier C

`dataflow.carry` itself carries no `step_op` / `cont_cond` attributes;
its signature is just the carried-value type. Reduction-shaped
patterns therefore expose their stepping/continuation parameters via
the **`dataflow.stream`** op driving the carry's `cond` operand
(`step_op` and `cont_cond` are `dataflow.stream` attributes per
`include/Dataflow/IR/DataflowOps.td`).

The flow signature of a tier-C SCC head is the tuple

```
flow_signature(carry) = (
    carry_type,       // MLIR Type of the carried value
    upstream_stream_signature_or_none
        // present iff carry.cond is produced by a dataflow.stream:
        //   (index_type, step_op, cont_cond)
        // otherwise (e.g. cond comes from arith.cmpi or block-arg):
        //   (cond_source_kind, cond_source_op_name)
)
```

Two carry heads "match" iff their signatures are equal under structural
type equality plus string equality on the attributes / op names. For
N > 2 inputs, the heuristic builds an equivalence relation by
transitive closure of pairwise matches; if the closure produces a
partition with more than one class within a single input, that input
fails `feedback_align_conflict`.

```
function pre_align_sccs(sccsets):
    if not config.scc_full_unroll:
        // signature heuristic per Q13
        all_carries = collect_carry_heads_across(sccsets)
        classes = partition(all_carries,
                            equiv = signature_equality)
        for each input sg:
            heads_in_sg = carry_heads_of(sg)
            if any class C has more than one head from sg:
                return failure("feedback_align_conflict")
        return classes  // one class per merged "carry slot" in the FU
    else:
        // unroll once per SCC: treat each cycle as a path of length
        // equal to the longest cycle across inputs, materializing
        // unrealized_conversion_cast placeholders for back-edges, then
        // run alignment on the unrolled DAG. Re-fold post-alignment.
        return unroll_then_align(sccsets)
```

The unroll path mirrors the four-pass materialization scheme used by
`SubgraphEnumerator` for forward-direction graph-region bodies (see
recent commit history under `enumerator: graph-region body`).
Carry-heads are realized inside the synthesized FU as
`fabric.op [@dataflow.carry]`, never as bare `dataflow.carry`
(`FuOp::verify` rejects the latter).

### CostModel

```cpp
// include/Fabric/Tech/Synthesizer/CostModel.h
namespace loom::fabric::tech {

struct AreaWeights {
  double muxPenalty   = 1.5;
  double demuxPenalty = 1.5;
  double carryPenalty = 2.0;
};

class CostModel {
public:
  CostModel(const SynthConfig &);
  double evaluate(fabric::FuOp fu) const;
};

} // namespace loom::fabric::tech
```

Formula (`fabric.fu` body contains only `fabric.op` / `fabric.mux` /
`fabric.demux`; carry-shaped ops are `fabric.op[@dataflow.carry]`):

```
cost(fu) = sum_{op in fu.fabric.op (op_list[0] != @dataflow.carry)}
                baseArea(shareGroupOf(op_list[0]), bitwidthOf(op))
         + sum_{op in fu.fabric.op (op_list[0] == @dataflow.carry)}
                carry_penalty * bitwidthOf(op)
         + sum_{m  in fu.fabric.mux}
                mux_penalty   * portCount(m) * bw(m)
         + sum_{d  in fu.fabric.demux}
                demux_penalty * portCount(d) * bw(d)
```

`baseArea(group, bw)` is a built-in table keyed on share-group id with a
linear bit-width factor:

```
baseArea(group, bw) = baseUnit[group] * (bw / 32.0)
```

Initial `baseUnit` table (informative; tunable in code, not in user
config per Q15):

| Share group                                         | baseUnit |
|-----------------------------------------------------|----------|
| arith.addi/subi                                     | 1.0      |
| arith.andi/ori/xori                                 | 0.5      |
| arith.shli/shrsi/shrui                              | 1.5      |
| arith.minsi/maxsi, arith.minui/maxui                | 1.0      |
| arith.divsi/remsi, arith.divui/remui                | 8.0      |
| arith.addf/subf                                     | 4.0      |
| arith.divf/remf                                     | 12.0     |
| arith.minimumf/maximumf                             | 3.0      |
| arith.sitofp/uitofp, arith.fptosi/fptoui            | 3.0      |
| math.sin/cos, math.sinh/cosh, math.tanh/erf         | 16.0     |
| math.exp/exp2/expm1, math.log/log2/log10/log1p      | 12.0     |
| math.sqrt/rsqrt                                     | 8.0      |
| math.floor/ceil/round/trunc/roundeven               | 2.0      |
| singleton (any op not in the table)                 | 1.0      |

`evaluate` is the only function that ranks FUs across strategy
restarts, across MCS branch candidates, and as the regression metric
for perf tests.

### Acceptance criteria (CostModel)

1. The per-op base cost increases with bit-width linearly: an i64 addi
   has exactly 2x the base cost of an i32 addi.
2. A 2-port mux costs strictly less than a 4-port mux of the same
   width (under positive `mux_penalty`).
3. Adding a `fabric.op[@dataflow.carry]` to an FU strictly increases
   its cost (under positive `carry_penalty`).
4. CostModel is pure: same FU + same config gives identical cost
   across runs and threads.

## Parallelism plan

| Layer                  | Where                             | Default                                   |
|------------------------|-----------------------------------|-------------------------------------------|
| Cross-group            | pass top level                    | on (`workers=auto`)                       |
| Coverage verification  | `CoverageVerifier::verify`        | on (`parallel_match=true`)                |
| MCS branch search      | `MCS::run` branch-and-bound       | on (`branch_workers=8`)                   |
| Random restarts        | `IncrementalRandom::run`          | on (`restarts=16`, `workers=auto`)        |
| Cost evaluation        | candidate ranking                 | inline; cheap, not parallelized           |

Implementation primitive: a single shared thread-pool wrapper in
`Parallel.h` built atop `llvm::ThreadPool`. All long-running parallel
sections submit closures that capture by value to avoid lifetime issues
across MLIR `OwningOpRef`s.

**MLIR mutation is never parallel.** Each per-group worker builds its
candidate FU in a *thread-local scratch* `MLIRContext`/`OwningOpRef`
context and produces a detached `OwningOpRef<func::FuncOp>` (the
wrapper). The pass's main thread, after all workers complete, splices
the wrappers into the user's `ModuleOp` in **lexical group-name order**
(serial). The same rule applies to `CoverageVerifier`, which builds a
private scratch `ModuleOp` per call.

#### Determinism rules

Every emitted IR construct is canonicalized through a single
`Canonicalize.h` pass after synthesis and before splicing. The
canonicalization step normalizes:

* **`fabric.op.op_list`**: members sorted by string name (already a
  precondition for `OpOp::verify`'s share-group check, but
  re-asserted here).
* **`fabric.fu` operand and result port order**: stable structural id
  derived from the union of the input subgraphs' yield order +
  block-arg order.
* **Mux / demux arm order**: each arm carries a structural id derived
  from the lowest-id subgraph it originated in; arms sorted by id.
* **Wrapper symbol name**: `@fu_<sanitized(group)>` with the
  `[A-Za-z0-9_]` sanitization rule.
* **`hw_params` and `sw_configs` dictionaries**: keys sorted
  lexically; allowed-set arrays sorted (e.g. `predicate = ["eq", "ne"]`
  not `["ne", "eq"]`).
* **Candidate ranking ties**: broken by `(cost, structural_id)` where
  `structural_id` is a deterministic 64-bit hash of the canonical
  printed form.

No `DenseMap` / `DenseSet` iteration is permitted in the emission
path. Internal data structures may use them; conversion to ordered
form happens at every emission boundary.

## Failure handling

Per Q14 (best-effort + optional fallback chain):

```
function generalize_pass(module, config):
    valid, invalid = validate_input_funcs(module)
        // invalid: zero-or-many dataflow.subgraph in body, or other
        // schema violations -> annotated immediately with
        // loom.synth_failed = "invalid_input"; not enqueued for synth.
    annotate_invalid(invalid)
    groups = collect_groups(valid)              // by loom.synth_group
    sorted = sort(groups, by=name)              // determinism

    // Per-group workers run in parallel; each builds a detached
    // wrapper func.func in a scratch MLIRContext. Workers do NOT
    // mutate the user's module.
    results = parallel_map sorted:
        lambda group: run_with_fallback(group, config)

    // Splice serially in sorted order; this is the ONLY place that
    // mutates the user's module after input validation.
    for (group, result) in zip(sorted, results):
        if result.success:
            // Symbol-name precheck: detect collision before splice.
            if module_has_symbol(module, result.fu_name):
                if previously_synthesized_marker(module, result.fu_name):
                    emit_remark(group, "skipping idempotent re-synth")
                    continue
                annotate_failure(group, "symbol_conflict")
                emit_warning(group, "symbol_conflict")
                continue
            splice(module, result.wrapper)
            tag(result.wrapper, loom.synthesized_for = group.name)
        else:
            for sg in group.subgraphs:
                annotate(sg.parent_func,
                         loom.synth_failed = result.failureReason)
            emit_diagnostic(group, result.failureReason,
                            severity=config.fail_as_error ? Error : Warning)

    return success
```

`run_with_fallback`:

```
function run_with_fallback(group, config):
    primary = makeSynthesizer(config.strategy, config)
    result = primary.run(SynthInputs(group))
    if result.success: return result
    for fallback_strategy in config.fallback_chain:
        s = makeSynthesizer(fallback_strategy, config)
        r = s.run(SynthInputs(group))
        if r.success: return r
    return result   // most informative: primary's failure reason
```

All emitted failures are MLIR `Diagnostic`s: `warning` by default,
`error` when `--synth-fail-as-error` flag is passed.

## Examples

The following examples illustrate the **structural** behavior of the
pass. The MLIR text is a readable sketch; exact assembly format for
some ops (`dataflow.subgraph`, `fabric.demux`, `dataflow.carry`) is
defined by the dialects themselves and may differ in trivial syntax
from the snippets here. The pass output is what the dialect printers
emit verbatim.

### Tier A example (op-list widen, single share group)

#### Input

```mlir
// All three inputs share identical topology and bit-width; only the
// op identity at the single internal node varies. All belong to
// share group {arith.addi, arith.subi}.
func.func @p_addi() attributes {loom.synth_group = "alu_int_32_addi_subi"} {
  %g = dataflow.subgraph (%a, %b) -> (%y) {
    %y = arith.addi %a, %b : i32
    dataflow.yield %y : i32
  }
  return
}
func.func @p_subi() attributes {loom.synth_group = "alu_int_32_addi_subi"} {
  %g = dataflow.subgraph (%a, %b) -> (%y) {
    %y = arith.subi %a, %b : i32
    dataflow.yield %y : i32
  }
  return
}
```

#### Synthesized FU

```mlir
fabric.fu @fu_alu_int_32_addi_subi () -> () {
^bb0(%a : fabric.bits<32>, %b : fabric.bits<32>):
  %y = fabric.op {op_list = [@arith.addi, @arith.subi],
                  hw_params = [{}]}
       (%a, %b)
       : (fabric.bits<32>, fabric.bits<32>) -> fabric.bits<32>
  fabric.yield %y : fabric.bits<32>
}
```

#### Topology (before/after)

```
input_addi:        input_subi:        synth_fu:
  a   b              a   b              a   b
   \ /                \ /                \ /
   addi               subi               op{op_list=[addi,subi]}
    |                  |                  |
   yield              yield              yield
```

### Tier B example (mux/demux insert)

#### Input

```mlir
// Two inputs share an arith.addi prefix; one extends with arith.muli,
// one terminates immediately.
func.func @p_add_only() attributes {loom.synth_group = "tierB_demo"} {
  %g = dataflow.subgraph (%a, %b) -> (%y) {
    %t = arith.addi %a, %b : i32
    dataflow.yield %t : i32
  }
  return
}
func.func @p_add_then_mul() attributes {loom.synth_group = "tierB_demo"} {
  %g = dataflow.subgraph (%a, %b, %c) -> (%y) {
    %t = arith.addi %a, %b : i32
    %y = arith.muli %t, %c : i32
    dataflow.yield %y : i32
  }
  return
}
```

#### Synthesized FU

```mlir
fabric.fu @fu_tierB_demo () -> () {
^bb0(%a : fabric.bits<32>, %b : fabric.bits<32>, %c : fabric.bits<32>):
  %t = fabric.op {op_list = [@arith.addi], hw_params = [{}]}
       (%a, %b)
       : (fabric.bits<32>, fabric.bits<32>) -> fabric.bits<32>

  // demux selects between the two downstream branches.
  %t_to_yield, %t_to_mul = fabric.demux %t
       : fabric.bits<32> -> fabric.bits<32>, fabric.bits<32>

  %m = fabric.op {op_list = [@arith.muli], hw_params = [{}]}
       (%t_to_mul, %c)
       : (fabric.bits<32>, fabric.bits<32>) -> fabric.bits<32>

  // mux collapses the two arms back into one yield port.
  %y = fabric.mux %t_to_yield, %m
       : (fabric.bits<32>, fabric.bits<32>) -> fabric.bits<32>

  fabric.yield %y : fabric.bits<32>
}
```

Materialization of this FU produces:

* `demux.sel=0, mux.sel=0` -> `t = a + b; yield t` (matches `p_add_only`)
* `demux.sel=1, mux.sel=1` -> `t = a + b; m = t*c; yield m`
  (matches `p_add_then_mul`)

#### Topology

```
synth_fu (tier B):

   a  b
    \/
   addi
    |
  demux
   /  \
   |   muli <- c
   |    |
   +----+
        |
       mux
        |
       yield
```

### Tier C example (feedback alignment)

#### Input

```mlir
// Two reductive accumulators driven by identical streams
// (lb=0, ub=N, step=1, step_op="+=", cont_cond="<"); both feed a
// dataflow.carry whose carried value is then post-processed differently
// (arith.addi vs arith.xori). addi and xori are in different share
// groups, so the post-carry op cannot share a single fabric.op.
func.func @p_accum_addi() attributes {loom.synth_group = "accum"} {
  %g = dataflow.subgraph (%lb, %ub, %step, %init)
                          -> (%out) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %init, %nxt : i32
    %nxt = arith.addi %c, %idx : i32          // post-carry: addi
    dataflow.yield %c : i32
  }
  return
}
func.func @p_accum_xori() attributes {loom.synth_group = "accum"} {
  %g = dataflow.subgraph (%lb, %ub, %step, %init)
                          -> (%out) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %init, %nxt : i32
    %nxt = arith.xori %c, %idx : i32          // post-carry: xori
    dataflow.yield %c : i32
  }
  return
}
```

The flow-signature heuristic matches the two carries by
`(carry_type=i32, upstream_stream=(i32, "+=", "<"))`. The post-carry
diff (addi vs xori) is then handled as a tier-B mux insertion.

#### Synthesized FU (sketch)

```mlir
func.func @fu_accum(%lb: !fabric.bits<32>, %ub: !fabric.bits<32>,
                    %step: !fabric.bits<32>, %init: !fabric.bits<32>)
                   -> !fabric.bits<32> {
  %y = fabric.fu(%plb = %lb : !fabric.bits<32>,
                 %pub = %ub : !fabric.bits<32>,
                 %pstep = %step : !fabric.bits<32>,
                 %pinit = %init : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %idx, %rwc = fabric.op [@dataflow.stream] (%plb, %pub, %pstep)
                 {hw_params = [{step_op = ["+="], cont_cond = ["<"]}]}
                 : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<1>)
    %c = fabric.op [@dataflow.carry] (%rwc, %pinit, %nxt)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>            // back-edge from %nxt
    %a = fabric.op [@arith.addi] (%c, %idx)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %x = fabric.op [@arith.xori] (%c, %idx)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %nxt = fabric.mux %a, %x : !fabric.bits<32>
    fabric.yield %c : !fabric.bits<32>
  }
  return %y : !fabric.bits<32>
}
```

The carry's third operand `%nxt` is a graph-region back-edge from
`fabric.mux`; emission uses an `unrealized_conversion_cast` placeholder
during the build phases (mirroring `SubgraphEnumerator`'s four-pass
materialization), resolved before the verifier runs.

#### Topology

```
synth_fu (tier C, signature-heuristic):

  lb   ub   step   init
   \   |    /       |
   stream            |
   /     \          |
 idx      rwc       |
   \      |         |
    \    carry <----+----  back-edge (graph-region body)
     \   /  \
      \ /    \
   +---+---+  \
   |       |   \
  addi    xori  \
       \   /     \
        mux ------+ (feeds %nxt)
                 (yield = %c)
```

## Test plan

### Layout

```
test/techmap/synth/
  unit/
    anchor/
      single_share_group.mlir       # tier A: addi+subi share group
      cross_share_group_strict.mlir # tier A: cross group, must fail
      cross_share_group_with_mux.mlir
                                    # tier A: same input, with mux flag on
      bitwidth_mismatch.mlir
      multi_anchor.mlir             # multiple yield producers
    mcs/
      tier_a_equivalent_to_anchor.mlir
                                    # cost(mcs) <= cost(anchor)
      tier_b_add_then_mul.mlir
      tier_c_accumulator.mlir
      cap_resource_exhausted.mlir
      timeout.mlir
    incremental/
      tier_a_oplist_widen.mlir
      tier_b_diff_arm.mlir
      tier_c_signature_heuristic.mlir
      input_order_invariance.mlir   # tier-A regardless of order
    incremental_random/
      cost_improvement.mlir
      restart_determinism.mlir      # same seed, same FU
    coverage_verifier/
      basic.mlir
      injected_bug_caught.mlir      # assert verify catches synth bugs
    failure/
      cross_share_group.mlir
      topology_mismatch.mlir
      feedback_align_conflict.mlir
      timeout.mlir
      resource_exhausted.mlir
      unsupported_op.mlir           # dataflow.load in input
      invalid_input_zero.mlir       # func.func with 0 dataflow.subgraph
      invalid_input_many.mlir       # func.func with 2 dataflow.subgraphs
      verifier_failed.mlir          # FU build that intentionally
                                    # violates FuOp::verify; assert the
                                    # pass detects and aborts cleanly
      symbol_conflict.mlir
      config_parse_failed.mlir
    grouping/
      multi_group.mlir              # multiple loom.synth_group values
      default_group.mlir            # missing attr -> default group
      empty_module.mlir
      idempotent_resynth.mlir       # second pass run is a no-op
  integration/
    cross_strategy_equivalence.mlir
                                    # same input, all four strategies,
                                    # each output FU verifies coverage
    enumerate_then_match_round_trip.mlir
                                    # synth -> enumerate -> match must
                                    # find all original inputs
  perf/
    synth_n100_anchor.mlir
    synth_n100_incremental_random.mlir
    synth_n1000_incremental_random.mlir
    synth_n5000_incremental_random.mlir
                                    # mirrors test/techmap/perf/
                                    # synth_n5000 partition perf test
```

### Test conventions

* lit + FileCheck. Each `.mlir` runs
  `loom %s -loom-generalize-subgraphs-to-fu='config=<path>
  dump-stats=true' | FileCheck %s`. With `dump-stats=true` the pass
  emits one canonical line per group as a remark:

      synth-stat group=<name> strategy=<s> reason=<r> cost=<f>
                 covered=<n>/<m> nodes=<n_op>/<n_mux>/<n_demux>

  These lines are stable across runs (per the determinism rules) and
  are the primary FileCheck targets for cost/coverage/strategy/reason
  acceptance criteria. The synthesized FU IR is also printed for
  structural assertions.
* perf tests use `loom-synth-fu-dump` to print timing measurements;
  pass/fail criterion is wall-time below a per-test budget. Mirrors
  the existing `synth_n5000` partition perf test gating.
* Cross-strategy equivalence test: parameterized by strategy name; its
  invariant is `covered=<m>/<m>` (full coverage) rather than identical
  FU text (different strategies legitimately produce different FUs).

### Acceptance criteria for the test plan

1. Every closed failure-reason enum value has at least one lit test
   asserting both the diagnostic text and the `loom.synth_failed`
   attribute.
2. Every config knob in SynthConfig has at least one lit test that
   exercises it (on/off, or distinct values).
3. Every strategy has at least one perf test in `perf/`.
4. The integration test `enumerate_then_match_round_trip.mlir` is the
   end-to-end gate: it never xfails.

## Open questions / known limits

* `hw_params` policy: the synthesizer emits an **observed-value union**
  for every configurable axis required by the enumerator. Concretely:
    * `op_list` -- union of observed op names at the merged position
      (already constrained by share-group + width).
    * variadic `bitmask` (sync / mux / demux) -- union of the observed
      bitmask values, encoded as the explicit allowed set.
    * `predicate` (`arith.cmpi` / `arith.cmpf`) -- union of observed
      predicate strings.
    * `step_op` / `cont_cond` (`dataflow.stream`) -- union of observed
      attribute strings.
    * `const_hex_value` (`dataflow.constant`) -- union of observed
      constants encoded as hex strings.
  Empty `hw_params` (`[{}]`) is only valid when the inner op kind has
  no configurable attribute axis (e.g. `arith.addi`). This is a
  correctness requirement, not an optional tightening:
  `SubgraphEnumerator` only fans out attribute axes that appear in
  `hw_params`, so omission would prevent the synthesized FU from
  enumerating any matching candidate.
* The `baseArea` weight table is encoded in C++ source. PDK-specific
  override, or a YAML data file, is explicitly out of scope per Q16.
* `IncrementalRandom` cost ranking with ties: the current spec picks
  the lowest permutation index. This is deterministic but arbitrary;
  no semantic preference is implied.
* MCS over highly heterogeneous workloads (many small groups, no shared
  skeleton) will hit `timeout` or `resource_exhausted` before producing
  a useful result. Best practice in such cases is to refine
  `loom.synth_group` so each group is structurally cohesive.
* `dataflow.load` / `dataflow.store` are out of scope for this spec; they
  introduce memory-port reasoning that is orthogonal to op-graph
  alignment. The pass emits a `topology_mismatch` failure with a note
  instructing callers to remove load/store-bearing inputs from synth
  groups.

## References

* `lib/Fabric/Tech/EnumerateFuSubgraphsPass.cpp`
* `lib/Fabric/Tech/SubgraphEnumerator.cpp`
* `lib/Fabric/Tech/SubgraphMatcher.cpp`
* `lib/Fabric/Tech/Partitioner/` (architectural template for
  `Synthesizer/`)
* `lib/Fabric/IR/FabricOps.cpp::hwShareGroups()`
* `docs/spec-fabric-reconfigurable-op.md`
* `docs/spec-fabric-hw-share-group.md`
* `include/Common/Config.h` (template for `SynthConfig`)
* `test/techmap/unit/` and `test/techmap/perf/` (template for
  `test/techmap/synth/`)
