# Subgraph-to-FU Generalization

This document specifies `loom-generalize-subgraphs-to-fu`, the Fabric
technology pass that generalizes a group of software
`dataflow.subgraph` partition units into one reconfigurable `fabric.fu`
hardware template. Given a set of `dataflow.subgraph` instances from
`dataflow.graph` definitions, the pass synthesizes a single `fabric.fu`
whose materialization under `loom-enumerate-fu-subgraphs` is a
**superset** of the input set.

The output `fabric.fu` represents a reconfigurable hardware block that can
be programmed (via `sw_configs`) to realize every input subgraph, plus
possibly additional configurations that fall out of the merged structure.
The input `dataflow.subgraph` operations remain software partition units:
they do not carry schedule slots, temporal tags, PE identity, routes, or
resource-sharing decisions.

The pass follows the Loom RISC/Occam rule: it does not create a meta
hardware instruction or encode placement decisions. It composes the
atomic Fabric configuration primitives already defined by the Fabric
dialect: `fabric.op`, `fabric.mux`, `fabric.demux`, and
`fabric.yield`. The partitioning compiler owns which software regions
become subgraphs; this pass owns only the derivation of an FU template
that can realize those subgraphs.

The target semantics are defined together with:

* `docs/spec-fabric-reconfigurable-op.md` for `fabric.op`
  configuration axes.
* `docs/spec-fabric-hw-share-group.md` for legal hardware sharing.
* `docs/spec-core-dialect-boundary.md` for the boundary between
  software dataflow, hardware Fabric, and mapping artifacts.

## Normative Boundary

This document defines the target contract for
`loom-generalize-subgraphs-to-fu`. It is intentionally limited to the
semantic boundary, input/output contract, IR conventions, strategy
behavior, failure semantics, and objective verification requirements.
Implementation layout, concrete C++ class structure, worker mechanics,
and test directory organization are execution-plan details, not target
semantics.

If an execution-plan detail in this document conflicts with the Fabric,
dataflow, mapping, or global evidence specs, the owning spec wins and
this document must be updated.

## System Contract

```
dataflow.graph
   |  graph partitioning
   v
dataflow.subgraph  (many)
   |  loom-generalize-subgraphs-to-fu
   v
fabric.fu templates
   |  loom-enumerate-fu-subgraphs
   v
materialized dataflow.subgraph candidates per fabric.fu
   |  subgraph matching / mapping artifact construction
   v
software-to-hardware binding
```

Architect-authored FUs and synthesized FUs share the same Fabric
semantics. A synthesized FU is not a placement result and does not name
placed physical PEs, routes, time slots, or AccCore instances. Its
module/PE/FU symbol path is template identity, not a placement binding.
It is a reusable hardware template that may later be selected by
mapping, PnR, simulation, or RTL generation flows.

Re-running `loom-enumerate-fu-subgraphs` on the synthesized FU must
produce a set that contains every input subgraph in the synth group.
This is the correctness invariant and is verified at synthesis time.

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
   mcs, incremental, incremental_random) selectable by config and
   implementing one common synthesizer interface.
5. **Parallelism as a first-class concern**: cross-group, intra-strategy,
   and verification parallelism enabled by default with safe defaults.
6. **End-to-end self-verification**: Default-on coverage check using
   `SubgraphEnumerator` and `SubgraphMatcher` as the coverage oracle
   (no separate coverage prover).

### Non-goals

* The pass does **not** decide partitioning of a `dataflow.graph`
  definition's body into subgraphs (`loom-partition-graph` already
  does that).
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
  `dataflow.graph` definition body. Represents one observed software
  partition pattern that the FU must support.
* **synth group**: a set of input subgraphs intended to be covered by one
  synthesized FU. Identified by a string-valued attribute
  `loom.synth_group` on the `dataflow.subgraph` operation. Subgraphs
  without the attribute belong to the implicit `"default"` group.
* **synthesized FU**: the PE-local `fabric.fu` template produced for
  one synth group. Its stable identity is the symbol path
  `(owning fabric.module symbol, owning fabric.pe symbol,
  PE-local fabric.fu symbol)`.
* **alignment**: a partial mapping between nodes/edges of two or more
  input subgraphs that identifies which positions correspond and may
  therefore be merged into the same `fabric.op` / port.
* **anchor**: a structurally distinguished node (typically the producer
  feeding `dataflow.yield`, or a `dataflow.carry` head) used as a fixed
  pivot for alignment.
* **share group**: a multi-member hardware-share group as defined by
  `docs/spec-fabric-hw-share-group.md`. Two ops can occupy the same
  `fabric.op.op_list` only if they belong to the same share group
  **and** their data-path bit-widths match.

## End-to-end interface

### Pass

```
Pass:     loom-generalize-subgraphs-to-fu
Scope:    ModuleOp
Inputs:   any number of dataflow.subgraph ops inside dataflow.graph
          definitions, optionally annotated with
            loom.synth_group = "<group_name>"
          Subgraphs without the attribute belong to the "default" group.
          Subgraphs whose body violates the subgraph verifier contract
          are rejected with loom.synth_failed = "invalid_input".
Output:   the same module, with a legal Fabric template container for
          each group that synthesized successfully. The target shape is
          a synthesizer-owned fabric.module that contains one named
          fabric.pe per group; each PE contains one PE-local named
          fabric.fu template plus the PE wiring needed to instantiate
          that FU. Failed groups annotate input subgraphs with
          loom.synth_failed.
Options:  config=<path>            -- configuration file accepted by
                                       the pass config verifier;
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

The pass is part of the Fabric technology pass set exposed by the
`loom` driver. No new production top-level binary is added. A stable
dump helper may exist for tests, but it is not part of the production
pipeline.

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
   `dataflow.subgraph` ops annotated with
   `loom.synth_failed = "<reason>"` (see failure enumeration below).
6. **Idempotence on synthesized output**: the synthesized PE symbol name
   for group `<g>` is `@pe_<sanitized(g)>`, and the PE-local FU symbol
   name is `@fu_<sanitized(g)>`, where `sanitized` replaces any
   character outside `[A-Za-z0-9_]` with `_`. The synthesizer-owned
   `fabric.module` symbol is part of the identity and collision check.
   Rerunning the pass on a module that already contains a
   synthesizer-owned module/PE/FU symbol path for that group is a no-op
   for that group, emitting a `remark`. Name collisions with
   non-synthesized Fabric symbols at any path component are reported as
   `symbol_conflict` failures.
7. **Output IR validity**: every emitted `fabric.module`, `fabric.pe`,
   `fabric.fu`, and `fabric.instantiate` passes the MLIR verifier. The
   PE-local `fabric.fu` passes `FuOp::verify` (which restricts the body
   to `fabric.op` / `fabric.mux` / `fabric.demux`) and every nested
   `fabric.op` passes `OpOp::verify` (which transitively enforces
   `hwShareGroups()` rules). The pass invokes the verifier on the
   freshly built Fabric container before splicing it into the module; a
   verifier failure is reported as `verifier_failed` with a diagnostic.
8. **Input validation**: each input `dataflow.subgraph` must satisfy
   the subgraph verifier contract, including explicit boundary values,
   supported body operations, and memory exclusion. Invalid subgraphs
   are skipped, annotated with
   `loom.synth_failed = "invalid_input"`, and reported with a
   `warning`.

### IR conventions

#### Dataflow Input

```mlir
// Group "alu_int_32"
dataflow.graph @pattern_addi_subi(%ctrl : none, %a : i32, %b : i32)
    -> (none, i32) {
  %g = dataflow.subgraph (...) -> (...) {
    ...
    %y = arith.addi %a, %b : i32
    dataflow.yield %y : i32
  } {loom.synth_group = "alu_int_32"}
  dataflow.yield %ctrl, %g : none, i32
}

dataflow.graph @pattern_subi(%ctrl : none, %a : i32, %b : i32)
    -> (none, i32) {
  %g = dataflow.subgraph (...) -> (...) {
    ...
    %y = arith.subi %a, %b : i32
    dataflow.yield %y : i32
  } {loom.synth_group = "alu_int_32"}
  dataflow.yield %ctrl, %g : none, i32
}

// Group default
dataflow.graph @pattern_loose_floor(%ctrl : none, %x : f32)
    -> (none, f32) {
  %g = dataflow.subgraph (...) -> (...) {
    ...
    %y = math.floor %x : f32
    dataflow.yield %y : f32
  }
  dataflow.yield %ctrl, %g : none, f32
}
```

#### Output

`fabric.fu` is PE-internal. It is not a module-level tile, not a
top-level hardware target, and not wrapped in `func.func` for hardware
identity. The pass emits a legal Fabric shape: an owning `fabric.module`
contains a named `fabric.pe`; the PE contains a named `fabric.fu`
template and a `fabric.instantiate` that wires the PE ports to that FU.
Downstream FU matching refers to the generated PE/FU identity pair.

```mlir
// New: synthesized Fabric container for group "alu_int_32_addi_subi".
fabric.module @loom_synth_fus() -> () {
  fabric.pe @pe_alu_int_32_addi_subi [spatial]
      (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>) {
  ^bb0(%a : !fabric.bits<32>, %b : !fabric.bits<32>):
    fabric.fu @fu_alu_int_32_addi_subi
        (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>) {
    ^bb0(%aa : !fabric.bits<32>, %bb : !fabric.bits<32>):
      %r = fabric.op [@arith.addi, @arith.subi] (%aa, %bb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %r : !fabric.bits<32>
    }
    %y = fabric.instantiate @fu_alu_int_32_addi_subi(
           %a : !fabric.bits<32>,
           %b : !fabric.bits<32>)
         -> (!fabric.bits<32>)
    fabric.yield %y : !fabric.bits<32>
  }
  fabric.yield
}

// Failed groups annotate the input subgraphs:
dataflow.graph @pattern_x(%ctrl : none, %x : i32) -> (none, i32) {
  %g = dataflow.subgraph (...) -> (...) {
    ...
    dataflow.yield %x : i32
  } {loom.synth_group = "loose",
     loom.synth_failed = "cross_share_group"}
  dataflow.yield %ctrl, %g : none, i32
}
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
  supported by `fabric.op` according to
  `docs/spec-fabric-reconfigurable-op.md`; for example
  `dataflow.load`, `dataflow.store`, `dataflow.graph`,
  `arith.constant`, `ub.poison`.
* `invalid_input` -- an input `dataflow.subgraph` violates the
  subgraph verifier contract, such as an ill-typed boundary or an
  unsupported body operation.
* `verifier_failed` -- the synthesized FU did not pass MLIR's verifier
  (`FuOp::verify` or a nested `OpOp::verify`). Indicates a compiler
  bug; the FU is dropped, no IR is appended.
* `symbol_conflict` -- the generated `@pe_<sanitized(group)>` or
  `@fu_<sanitized(group)>` Fabric symbol, or the synthesizer-owned
  `fabric.module` symbol that would contain them, already exists in the
  target Fabric symbol table and does not correspond to a previous
  synthesizer run that may safely be skipped.
* `config_parse_failed` -- the `--config=<path>` file failed to load.

These failure reasons are stored verbatim as the `loom.synth_failed`
attribute on the offending input `dataflow.subgraph` operations.
Implementations must expose this closed set through their
failure-reason representation and enforce exhaustive handling.

## Implementation Boundary

The pass must share the canonical hardware-share-group source of truth
with `fabric.op` verification, FU enumeration, subgraph matching, and
mapping. It must not introduce a private sharing table or a
strategy-local override mechanism.

The pass option `config=<path>` selects strategy and policy knobs. The
target schema is defined by the accepted pass options and config
verifier, not by this spec's prose. A missing config uses deterministic
built-in defaults; a malformed config reports `config_parse_failed` and
does not mutate the user's Fabric IR.

## Strategies

All strategies share one semantic result contract:

* input: one synth group, the group's `dataflow.subgraph` operations,
  the active config/profile identity, and the canonical Fabric/dataflow
  semantics.
* success: one legal generated `fabric.pe` containing one PE-local
  `fabric.fu`, stable module/PE/FU identity, coverage evidence, cost
  evidence, and diagnostics.
* failure: one closed failure reason and diagnostics, with no partial
  Fabric IR appended for that group.

### Tier coverage matrix

| Strategy            | Tier A | Tier B | Tier C | Strength                              | Cost                                     |
|---------------------|:------:|:------:|:------:|---------------------------------------|------------------------------------------|
| anchor              |  yes   | partial|  no    | fast, deterministic                   | cannot align disjoint topologies         |
| mcs                 |  yes   |  yes   |  yes   | exact graph-native candidates under caps | exponential worst case                |
| incremental         |  yes   |  yes   |  yes   | wall-time linear in N inputs (verify amortized via cache)  | order-sensitive                          |
| incremental_random  |  yes   |  yes   |  yes   | best cost via random restarts         | wall-time scales with restart count      |

`anchor` covers tier B in the restricted case where differences are
single-edge insertions/deletions handled by local mux/demux when
`allow_intra_position_mux=true`.

### Strategy: anchor (tier A by default)

#### Anchor Strategy Idea

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

#### Anchor Strategy Pseudocode

```
function synthesize_anchor(inputs):
    sgs = inputs.subgraphs
    yield_arities = [yield_operand_count(sg) for sg in sgs]
    if not all_equal(yield_arities):
        return failure("topology_mismatch")

    // anchors_per_index[k] = [Source for sg_i's k-th yield operand]
    anchors_per_index = build_yield_anchors(sgs)

    pe = empty_pe_template_with_inputs(union_block_args(sgs))
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
            // Reserve a temporary back-edge placeholder; resolved
            // before accepted output is verified.
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
   fail with `topology_mismatch` regardless of the mux flag because
   bit-width is part of share-group identity.

### Strategy: mcs (all tiers)

#### MCS Strategy Idea

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
* enumerate disjoint shared-node tuple subsets in deterministic order;
* use branch-and-bound lower bounds from the private-op baseline minus
  already selected sharing and maximum remaining sharing;
* shard independent first-node mapping choices with `branch_workers`;
* classify graph-region SCC edges with Tarjan metadata during graph
  extraction;
* use `timeout_sec` and `candidate_cap` as hard caps;
* accept only candidates proven by the enumerator/matcher roundtrip.

MCS admits concrete local candidate families before falling back to an
explicit outer strategy chain: a lock-step candidate for isomorphic
inputs, exact graph-region MCES candidates, bounded graph-region MCES
candidates, and a shared-prefix candidate. The graph-region path
supports cycles, commutative operand normalization, non-positional
block-argument mappings, multi-yield outputs, and strict-superset
enumeration. `candidate_cap` limits admitted materialized candidates;
exact search may inspect additional estimated-cost MCES candidates so
unverified raw mappings do not consume the budget. The strategy enforces
`timeout_sec` around graph search, construction, and coverage
verification. A candidate is accepted only if `CoverageVerifier` can
enumerate software subgraphs covering every input. MCS must not invoke
incremental behavior internally; if a group needs incremental behavior,
it is requested through `fallback_chain`.

#### MCS Strategy Pseudocode

```
function synthesize_mcs(inputs):
    sgs = inputs.subgraphs
    best = []
    maybe_add(lockstep_candidate(sgs))
    graphs = build_graph_region_views(sgs)
    for cand in exact_branch_and_bound_mces(graphs,
                                            cap=config.mcs.candidate_cap,
                                            deadline=config.mcs.timeout_sec,
                                            workers=config.mcs.branch_workers):
        fu = build_fu_with_mux_demux_adapters(cand, graphs)
        if coverage_roundtrip_accepts(fu, sgs):
            best.append(fu)
    for cand in bounded_disjoint_tuple_subsets(graphs,
                                               cap=remaining_candidate_budget(best),
                                               deadline=config.mcs.timeout_sec):
        fu = build_fu_with_mux_demux_adapters(cand, graphs)
        if coverage_roundtrip_accepts(fu, sgs):
            best.append(fu)
    maybe_add(shared_prefix_candidate(sgs))
    if best.empty():
        return failure("timeout" or "topology_mismatch")
    return success(lowest_cost(best))
```

#### Acceptance criteria (mcs)

1. On a tier-A workload (all inputs isomorphic), mcs produces a FU
   whose CostModel score is `<=` the anchor strategy's score on the
   same input.
2. On `(a+b)*c` and `(a+b)` mixed inputs, mcs identifies the shared
   `arith.addi` skeleton and bypasses the multiplication via a single
   `fabric.mux`.
3. With `candidate_cap=1`, mcs still admits one verified graph-native
   candidate instead of spending the budget on unverified raw mappings.
4. Per `parallel_match=true`, coverage verification of the best
   candidate runs in parallel across input subgraphs.
5. Graph-region MCES candidates cover acyclic common-private-common
   inputs, cyclic carry inputs, block-argument permutations, multi-yield
   outputs, exact-cover enumeration, and strict-superset enumeration.
6. MCS has no hidden incremental fallback. If graph-native candidates
   cannot produce a legal FU, MCS reports its own failure and the outer
   `fallback_chain` decides whether to try another strategy.
7. `branch_workers` applies to exact graph-region MCES search and does
   not change the chosen deterministic result.

### Strategy: incremental (all tiers)

#### Incremental Strategy Idea

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
foreclose lower-cost later sharing candidates.)

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

#### Incremental Strategy Pseudocode

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

#### Incremental Random Strategy Idea

Wrap `incremental` in a multi-restart driver. The order in which inputs
are folded matters for FU cost (different orders may produce
structurally different FUs even when both are correct). Run `restarts`
independent permutations in parallel and return the lowest-cost
successful FU.

#### Incremental Random Strategy Pseudocode

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

The alignment component shares the `SubgraphMatcher::GraphView` data
semantics so that synthesis and matching agree on what "the same Source
position" means across DAG fanout, multi-result ops, block arguments,
commutative operands, graph-region back-edges, and yield wiring.
Synthesis re-uses GraphView's SCC pre-pass and source descriptors
verbatim; this avoids inventing a parallel hashing/iteration order that
would diverge from the matcher.

The shared Source model has three semantic cases:

* `BodyOp`: a value produced by an operation result inside the subgraph.
* `BlockArg`: a value supplied by the subgraph boundary.
* `BackEdge`: a graph-region back-edge into an operation, resolved by
  the same SCC pre-pass used by matching.

A node signature collapses op identity, share-group id, bit-width,
arity, operand-source kinds, and a stable structural hash. Two subgraph
positions are alignable iff their signatures match and their
per-operand source kinds match positionally after commutative
normalization. Yield anchors are the ordered Source descriptors for
`dataflow.yield` operands and are the canonical entry point for anchor,
mcs, and incremental alignment.

### CoverageVerifier

The verifier uses `SubgraphEnumerator` and `SubgraphMatcher`. It does
not embed a hand-written coverage proof; the enumerator is treated as
the authoritative oracle for "what an FU can become."

For each input subgraph the verifier records either the index of one
materialized FU candidate that matches it or a miss. The coverage report
is successful only when every input has a match. Per
`parallel_match=true`, input matching may be sharded across workers, but
parallel and single-worker execution must produce the same report.

### SCC handling for tier C

`dataflow.carry`, `dataflow.gate`, and `dataflow.invariant` carry no
`step_op` / `cont_cond` attributes themselves. Their signatures include
the state-head op name, carried/latch value type, and the cond source.
Reduction-shaped patterns therefore expose their stepping/continuation
parameters via the **`dataflow.stream`** op driving the state head's
`cond` operand (`step_op` and `cont_cond` are `dataflow.stream`
attributes per `docs/spec-dataflow-part-1-streaming.md`).

The flow signature of a tier-C SCC head is the tuple

```
flow_signature(state_head) = (
    op_name,          // dataflow.carry / dataflow.gate / dataflow.invariant
    data_type,        // MLIR Type of the carried/latch value
    upstream_stream_signature_or_none
        // present iff cond is produced by a dataflow.stream:
        //   (index_type, step_op, cont_cond)
        // otherwise (e.g. cond comes from arith.cmpi or block-arg):
        //   (cond_source_kind, cond_source_op_name)
)
```

Two state heads "match" iff their signatures are equal under structural
type equality plus string equality on the attributes / op names. For
N > 2 inputs, the heuristic builds an equivalence relation by
transitive closure of pairwise matches; if the closure produces a
partition with more than one class within a single input, that input
fails `feedback_align_conflict`.

```
function pre_align_sccs(sccsets):
    if not config.scc_full_unroll:
        // use the configured flow-signature heuristic
        all_heads = collect_state_heads_across(sccsets)
        classes = partition(all_heads,
                            equiv = signature_equality)
        for each input sg:
            heads_in_sg = state_heads_of(sg)
            if any class C has more than one head from sg:
                return failure("feedback_align_conflict")
        return classes  // one class per merged state slot in the FU
    else:
        // conservative full-unroll fallback: do not force incompatible
        // state signatures to merge. Instead, keep each incompatible
        // state path as a separate fabric.op slot, add muxes at feedback
        // and yield join points, and rely on coverage verification to
        // accept only candidates that enumerate back to every input.
        return mirror_with_fresh_state_slots(sccsets)
```

The unroll path mirrors the four-pass materialization scheme used by
`SubgraphEnumerator` for forward-direction graph-region bodies. State
heads are realized inside the synthesized FU as `fabric.op` operations
whose `op_list` names `dataflow.carry`, `dataflow.gate`, or
`dataflow.invariant`, never as bare dataflow ops (`FuOp::verify` rejects
the latter).

When `scc_full_unroll = true`, the configured semantics are the
conservative fresh-state fallback shown above rather than a cycle-length
Tarjan unroll. The knob therefore has observable semantics:
incompatible state signatures that the default heuristic reports as
`feedback_align_conflict` may still synthesize if keeping the state paths
separate produces a candidate that `CoverageVerifier` proves covers every
input. This is intentionally stricter than accepting arbitrary topology:
the enumerator and matcher remain the legality oracle.

### CostModel

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
configuration):

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
restarts, across MCS graph candidates, and as the regression metric
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

## Parallel Execution Requirements

Parallelism is an optimization, not a semantic axis. The pass may
parallelize independent synth groups, coverage matching, MCS branch
search, and random restarts, but the emitted IR and diagnostics must be
equivalent to single-worker execution for deterministic configurations.

**User MLIR mutation is serial.** Parallel workers may construct
detached candidate IR and reports, but the pass mutates the user's
module only at deterministic emission boundaries. Successful groups are
spliced in lexical group-name order.

### Determinism Rules

Every emitted IR construct is canonicalized through a single
canonicalization step after synthesis and before splicing. The
canonicalization step must normalize:

* **`fabric.op.op_list`**: members sorted by string name (already a
  precondition for `OpOp::verify`'s share-group check, but
  re-asserted here).
* **`fabric.fu` operand and result port order**: stable structural id
  derived from the union of the input subgraphs' yield order +
  block-arg order.
* **Mux / demux arm order**: each arm carries a structural id derived
  from the lowest-id subgraph it originated in; arms sorted by id.
* **Generated Fabric symbols**: `@pe_<sanitized(group)>` and
  PE-local `@fu_<sanitized(group)>` with the `[A-Za-z0-9_]`
  sanitization rule.
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

The pass uses best-effort synthesis with an optional fallback chain:

```
function generalize_pass(module, config):
    valid, invalid = validate_input_subgraphs(module)
    // invalid: verifier or schema violations -> annotated immediately
    // with loom.synth_failed = "invalid_input"; not enqueued for synth.
    annotate_invalid(invalid)
    groups = collect_groups(valid)              // by loom.synth_group
    sorted = sort(groups, by=name)              // determinism

    // Per-group workers may build detached candidate Fabric IR and
    // reports. Workers do NOT mutate the user's module.
    results = parallel_map sorted:
        lambda group: run_with_fallback(group, config)

    // Splice serially in sorted order; this is the ONLY place that
    // mutates the user's module after input validation.
    for (group, result) in zip(sorted, results):
        if result.success:
            // Symbol-name precheck: detect collision before splice.
            if fabric_symbol_conflicts(module,
                                       result.moduleName,
                                       result.peName,
                                       result.fuName):
                if previously_synthesized_marker(module,
                                                 result.moduleName,
                                                 result.peName,
                                                 result.fuName):
                    emit_remark(group, "skipping idempotent re-synth")
                    continue
                annotate_failure(group, "symbol_conflict")
                emit_warning(group, "symbol_conflict")
                continue
            splice_into_target_fabric_module(module, result)
            tag(result.pe, loom.synthesized_for = group.name)
        else:
            for sg in group.subgraphs:
                annotate(sg,
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
`error` when `fail-as-error=true` is passed.

## Examples

The following examples illustrate the **structural** behavior of the
pass. The MLIR text is a readable sketch; exact assembly format for
some ops (`dataflow.subgraph`, `fabric.demux`, `dataflow.carry`) is
defined by the dialects themselves and may differ in trivial syntax
from the snippets here. The pass output is what the dialect printers
emit verbatim.

### Tier A example (op-list widen, single share group)

#### Tier A Input

```mlir
// All three inputs share identical topology and bit-width; only the
// op identity at the single internal node varies. All belong to
// share group {arith.addi, arith.subi}.
dataflow.graph @p_addi(%ctrl : none, %a : i32, %b : i32) -> (none, i32) {
  %g = dataflow.subgraph (%a, %b) -> (%y) {
    %y = arith.addi %a, %b : i32
    dataflow.yield %y : i32
  } {loom.synth_group = "alu_int_32_addi_subi"}
  dataflow.yield %ctrl, %g : none, i32
}
dataflow.graph @p_subi(%ctrl : none, %a : i32, %b : i32) -> (none, i32) {
  %g = dataflow.subgraph (%a, %b) -> (%y) {
    %y = arith.subi %a, %b : i32
    dataflow.yield %y : i32
  } {loom.synth_group = "alu_int_32_addi_subi"}
  dataflow.yield %ctrl, %g : none, i32
}
```

#### Tier A Synthesized PE-local FU

```mlir
// Excerpt from the generated fabric.module container.
fabric.pe @pe_alu_int_32_addi_subi [spatial]
    (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>) {
^bb0(%a : !fabric.bits<32>, %b : !fabric.bits<32>):
  fabric.fu @fu_alu_int_32_addi_subi
      (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>) {
  ^bb0(%aa : !fabric.bits<32>, %bb : !fabric.bits<32>):
    %y = fabric.op {op_list = [@arith.addi, @arith.subi],
                    hw_params = [{}]}
         (%aa, %bb)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %y : !fabric.bits<32>
  }
  %y = fabric.instantiate @fu_alu_int_32_addi_subi(
         %a : !fabric.bits<32>,
         %b : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield %y : !fabric.bits<32>
}
```

#### Tier A Topology Before And After

```
input_addi:        input_subi:        synth_fu:
  a   b              a   b              a   b
   \ /                \ /                \ /
   addi               subi               op{op_list=[addi,subi]}
    |                  |                  |
   yield              yield              yield
```

### Tier B example (mux/demux insert)

#### Tier B Input

```mlir
// Two inputs share an arith.addi prefix; one extends with arith.muli,
// one terminates immediately.
dataflow.graph @p_add_only(%ctrl : none, %a : i32, %b : i32) -> (none, i32) {
  %g = dataflow.subgraph (%a, %b) -> (%y) {
    %t = arith.addi %a, %b : i32
    dataflow.yield %t : i32
  } {loom.synth_group = "tierB_demo"}
  dataflow.yield %ctrl, %g : none, i32
}
dataflow.graph @p_add_then_mul(%ctrl : none, %a : i32, %b : i32, %c : i32)
    -> (none, i32) {
  %g = dataflow.subgraph (%a, %b, %c) -> (%y) {
    %t = arith.addi %a, %b : i32
    %y = arith.muli %t, %c : i32
    dataflow.yield %y : i32
  } {loom.synth_group = "tierB_demo"}
  dataflow.yield %ctrl, %g : none, i32
}
```

#### Tier B Synthesized PE-local FU

```mlir
// Excerpt from the generated fabric.module container.
fabric.pe @pe_tierB_demo [spatial]
    (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> (!fabric.bits<32>) {
^bb0(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>):
  fabric.fu @fu_tierB_demo
      (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>) {
  ^bb0(%aa : !fabric.bits<32>, %bb : !fabric.bits<32>, %cc : !fabric.bits<32>):
    %t = fabric.op {op_list = [@arith.addi], hw_params = [{}]}
         (%aa, %bb)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>

    // demux selects between the two downstream branches.
    %t_to_yield, %t_to_mul = fabric.demux %t
         : !fabric.bits<32> -> !fabric.bits<32>, !fabric.bits<32>

    %m = fabric.op {op_list = [@arith.muli], hw_params = [{}]}
         (%t_to_mul, %cc)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>

    // mux collapses the two arms back into one yield port.
    %y = fabric.mux %t_to_yield, %m
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>

    fabric.yield %y : !fabric.bits<32>
  }
  %y = fabric.instantiate @fu_tierB_demo(
         %a : !fabric.bits<32>,
         %b : !fabric.bits<32>,
         %c : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield %y : !fabric.bits<32>
}
```

Materialization of this FU produces:

* `demux.sel=0, mux.sel=0` -> `t = a + b; yield t` (matches `p_add_only`)
* `demux.sel=1, mux.sel=1` -> `t = a + b; m = t*c; yield m`
  (matches `p_add_then_mul`)

#### Tier B Topology

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

#### Tier C Input

```mlir
// Two reductive accumulators driven by identical streams
// (lb=0, ub=N, step=1, step_op="+=", cont_cond="<"); both feed a
// dataflow.carry whose carried value is then post-processed differently
// (arith.addi vs arith.xori). addi and xori are in different share
// groups, so the post-carry op cannot share a single fabric.op.
dataflow.graph @p_accum_addi(%ctrl : none, %lb : i32, %ub : i32,
                             %step : i32, %init : i32) -> (none, i32) {
  %g = dataflow.subgraph (%lb, %ub, %step, %init)
                          -> (%out) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %init, %nxt : i32
    %nxt = arith.addi %c, %idx : i32          // post-carry: addi
    dataflow.yield %c : i32
  } {loom.synth_group = "accum"}
  dataflow.yield %ctrl, %g : none, i32
}
dataflow.graph @p_accum_xori(%ctrl : none, %lb : i32, %ub : i32,
                             %step : i32, %init : i32) -> (none, i32) {
  %g = dataflow.subgraph (%lb, %ub, %step, %init)
                          -> (%out) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %init, %nxt : i32
    %nxt = arith.xori %c, %idx : i32          // post-carry: xori
    dataflow.yield %c : i32
  } {loom.synth_group = "accum"}
  dataflow.yield %ctrl, %g : none, i32
}
```

The flow-signature heuristic matches the two carries by
`(carry_type=i32, upstream_stream=(i32, "+=", "<"))`. The post-carry
diff (addi vs xori) is then handled as a tier-B mux insertion.

#### Tier C Synthesized PE-local FU Sketch

```mlir
fabric.pe @pe_accum [spatial]
    (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> (!fabric.bits<32>) {
^bb0(%lb : !fabric.bits<32>, %ub : !fabric.bits<32>,
     %step : !fabric.bits<32>, %init : !fabric.bits<32>):
  fabric.fu @fu_accum
      (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>) {
  ^bb0(%plb : !fabric.bits<32>, %pub : !fabric.bits<32>,
       %pstep : !fabric.bits<32>, %pinit : !fabric.bits<32>):
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
  %y = fabric.instantiate @fu_accum(
         %lb : !fabric.bits<32>,
         %ub : !fabric.bits<32>,
         %step : !fabric.bits<32>,
         %init : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield %y : !fabric.bits<32>
}
```

The carry's third operand `%nxt` is a graph-region back-edge from
`fabric.mux`. Any temporary back-edge placeholder used during
construction must be resolved before the verifier runs and must not
survive in accepted output IR.

#### Tier C Topology

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

## Objective Verification

1. Every closed failure-reason enum value has at least one lit test
   asserting both the diagnostic text and the `loom.synth_failed`
   attribute.
2. Every accepted configuration knob has at least one lit test that
   exercises it (on/off, or distinct values).
3. Every strategy has positive evidence for at least one covered input
   group and negative evidence for at least one relevant failure mode.
4. Cross-strategy equivalence evidence asserts full coverage
   (`covered=<m>/<m>`) rather than byte-identical FU text, because
   different strategies may legitimately produce different legal FUs.
5. The synth -> enumerate -> match roundtrip is the end-to-end gate:
   it must find all original input subgraphs and must never be replaced
   by fake or stub coverage evidence.
6. Deterministic configurations are checked by repeated runs with the
   same seed, worker count, and input set.
7. Timeout and resource-exhaustion tests assert structured diagnostics
   and must not report pass when the operation merely failed to run.
8. Performance evidence uses explicit wall-time budgets and records the
   measured command/report; it is not a substitute for semantic
   coverage verification.

## Policy Choices and Limits

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
* The `baseArea` weight table is compiled into the pass. PDK-specific
  override, or an external data file override, is explicitly out of
  scope for this pass.
* `IncrementalRandom` cost ranking with ties picks the lowest
  permutation index. This is deterministic but arbitrary; no semantic
  preference is implied.
* MCS over highly heterogeneous workloads (many small groups, no shared
  skeleton) will hit `timeout` or `resource_exhausted` before producing
  a useful result. Best practice in such cases is to refine
  `loom.synth_group` so each group is structurally cohesive.
* `dataflow.load` / `dataflow.store` are out of scope for this spec; they
  introduce memory-port reasoning that is orthogonal to op-graph
  alignment. The pass emits an `unsupported_op` failure with a note
  instructing callers to remove load/store-bearing inputs from synth
  groups.

## Related Specifications

* `docs/spec-core-dialect-boundary.md`
* `docs/spec-fabric-reconfigurable-op.md`
* `docs/spec-fabric-hw-share-group.md`
* `docs/spec-compiler-part-3-dfg.md`
* `docs/spec-compiler-part-3-placement-framework.md`
