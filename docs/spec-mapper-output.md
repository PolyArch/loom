# FCC Mapper Output Specification

## Overview

A successful FCC mapping produces machine-readable, human-readable, and
visualization-oriented outputs.

## Output Families

Mapping outputs are mixed software-plus-hardware artifacts and therefore use
the naming family `<dfg>.<adg>.*`.

The main mapping outputs are:

- `<dfg>.<adg>.map.json`
- `<dfg>.<adg>.map.txt`
- `<dfg>.<adg>.viz.html`
- `<dfg>.<adg>.config.json`
- `<dfg>.<adg>.config.bin`
- `<dfg>.<adg>.config.h`

Additional config or simulation artifacts may be produced by later stages.

If periodic mapper snapshots are enabled, FCC also emits a sibling directory:

- `mapper-snapshots/`

Each snapshot is an expanded partial mapping checkpoint and currently produces:

- `<mixed>.snapshot-<seq>.<trigger>.mapper-<ordinal>.map.json`
- `<mixed>.snapshot-<seq>.<trigger>.mapper-<ordinal>.viz.html`

`<seq>` is a monotonically increasing per-run snapshot serial. `<trigger>`
identifies the mapper stage that requested the snapshot, and `<ordinal>` is
the mapper-side emission count attached to that trigger.

## Config JSON

`<dfg>.<adg>.config.json` is the authoritative structured summary of serialized
runtime configuration.

It must include:

- flattened `words`
- per-slice metadata with `word_offset`, `word_count`, and completeness
- container slices for `spatial_pe` and `temporal_pe`, not only primitive
  routing or memory nodes

For `spatial_pe`, the slice low bit is `spatial_pe_enable`, followed by opcode,
PE input mux controls, PE output demux controls, and the selected FU-internal
config payload.

For `temporal_pe`, the slice low region is instruction memory ordered by slot,
and the high region stores persistent per-function_unit internal config bits.

## JSON Mapping Report

The JSON report is the authoritative structured output for downstream tools.

### Core Sections

At minimum, the report should include:

- `seed`
- `techmap`
- `search`
- `node_mappings`
- `edge_routings`
- `port_table`
- `temporal_registers`
- `memory_regions`

### Required Semantics

`techmap` must identify the Layer-2 handoff artifact used by Layer 3.

In memory, the same Layer-2 handoff should also be consumable through explicit
`TechMapper::Plan` query helpers rather than only through implicit vector-index
conventions.

That in-memory query surface should include config-class compatibility and
support-class hard-capacity semantics, not only raw object lookup.

Mapper-side consumers should prefer those explicit `TechMapper::Plan` queries
over direct dependence on storage-layout details such as vector indexing or
DenseMap iteration order.

The same query surface should also expose quantitative support-class capacity
and config-class compatibility neighborhoods, not only boolean predicates.

At minimum it must preserve:

- a stable Layer-2 contract version and the selected-plan shape
- Layer-2 summary metrics such as coverage, selected-candidate count, fallback
  count, and migration-oracle status
- the Layer-2 summary should also expose whether the current selected plan is
  still blocked from mapper handoff by legacy-oracle gaps, legacy-only
  candidate-pool dependence, or selected legacy-fallback dependence, rather
  than forcing downstream tooling to infer that status indirectly from several
  counters
- when that handoff status is not ready, the report should also enumerate the
  concrete blocker set instead of only publishing one aggregate status string
- Layer-2 summary metrics should also expose how many overlap components were
  solved by exact search, CP-SAT, and greedy fallback, so non-greedy
  selection coverage does not depend on inferring backend choice from one free
  form diagnostic string
- Layer-2 summary metrics should also summarize candidate-status buckets, so
  downstream tools can tell whether the current search is mainly losing on
  overlap, temporal legality, hardware-capacity legality, or objective ranking
  without scanning the full candidate table
- when cached Layer-2 feedback reselection is used, the summary metrics should
  also report how many reselections happened, how many candidates were filtered
  by feedback, and how many penalty terms were applied
- `demand_driven_primary_plan` must become false as soon as the retained legacy
  migration path contributes any legacy-derived support into the active
  candidate pool, including mixed-origin candidates, even if the final
  selected subset does not end up choosing a pure legacy-only candidate
- the Layer-2 summary should distinguish how many FU instances needed retained
  legacy fallback from how many legacy-only candidates were actually injected
  into the active candidate pool, so handoff review does not collapse those
  two different migration signals into one ambiguous counter
- that legacy-only candidate count should be measured at the active
  candidate-summary level, not as a raw per-hardware match-hit total, so the
  metric stays stable across initial build and cached reselection paths
- mapper handoff readiness should key off legacy-derived contamination in the
  active candidate pool, not merely the number of FU instances that needed the
  retained legacy fallback path
- when cached Layer-2 reselection is used, those retained-legacy contamination
  metrics must be recomputed from the filtered active candidate pool rather
  than inherited from the seed plan or cleared by metric reset
- mixed-origin candidates and selected units also count as retained-legacy
  contamination for mapper handoff, because they still widen the Layer-2
  hardware-support domain using legacy-derived support even when the fused
  software pattern itself exists on the demand-driven path
- the Layer-2 summary should therefore expose both pure legacy-only counts and
  broader legacy-derived contamination counts for candidate pools and selected
  units
- the same distinction should also exist on the hardware-support-source side,
  so artifacts can tell whether legacy-derived support still survives only in
  pure legacy-only pools or also through mixed-origin candidates
- the handoff summary should also expose direct boolean answers for whether
  the selected subset, the active candidate pool, and the hardware-support
  source pool still use legacy-derived support, so reviewers do not need to
  reconstruct those states manually from multiple counters
- when cached Layer-2 feedback reselection is used, the report should also
  preserve the applied feedback request itself as structured data, including
  banned candidate/family/config-class ids, split requests, and penalty terms
- the feedback request should expose descriptor-rich mirrors of those ids, such
  as candidate family/config/component context and family/config-class keys, so
  review does not depend on a second join through the main candidate or class
  tables
- the feedback request should also distinguish unresolved references from
  successfully resolved ones, so invalid candidate/family/config-class ids do
  not masquerade as effective policy input
- support-class and config-class descriptors
- support-class descriptors must reveal whether hard capacity is enforced at
  Layer 2 or deferred because the class represents temporal reuse
- config-class descriptors should expose both id-based and key-based
  compatibility views, and temporal-incompatibility diagnostics should also
  carry descriptor keys instead of only raw class ids
- non-temporal config classes should not be overconstrained by that
  compatibility view: spatial config diversity is legal, while temporal
  same-instance reuse remains the case that needs explicit compatibility
- support-class and candidate hardware pools should expose both raw `hw_node`
  ids and human-readable `pe_name` sets, so artifact inspection does not depend
  on a second lookup through the ADG
- the human-readable report should also include support-class details,
  especially temporal/spatial kind, capacity, and whether hard capacity is
  enforced at Layer 2
- family-level baseline summaries including at least materialized-state count
  and match count
- a candidate table keyed by stable plan-local `candidate_id`, so every
  `candidate_id` referenced by selected-unit, per-node, component, fallback,
  or feedback-related artifacts can be resolved without reconstructing the
  aggregated match set offline
- each candidate-table entry should say whether it was selected into the
  current best plan, so downstream tools do not need to diff it against the
  selected-unit list just to recover that bit
- each candidate-table entry should also expose a stable status or rejection
  reason so a non-selected candidate can be explained without replaying the
  overlap solve offline
- when a candidate is selected, the candidate table should link directly to the
  resulting selected-unit id and contracted-node id, so consumers can move from
  candidate-space to contracted-plan-space without cross-reconstructing the
  selection
- overlap-component summaries that identify which software nodes and candidate
  ids belong to each coupled selection region, which candidate ids were
  selected there, whether the region contains temporal candidates, and which
  solver produced the result; the solver string should be explicit enough to
  distinguish `exact`, `cpsat`, and `greedy`
- when feedback reselection filters component-local candidates before the new
  solve, overlap-component summaries should also expose the filtered
  candidate-id subset so component-local policy changes are auditable
- overlap-component summaries should also expose enough score context to audit
  why the chosen subset won inside that region, such as the best candidate
  score and the total score of the selected subset
- when feedback reselection can rescore candidates, overlap-component
  summaries should preserve both the base and current score aggregates for that
  region, not only the latest post-feedback values
- overlap-component summaries should link directly to the selected-unit ids
  produced from that region, not only to candidate ids, so component-space and
  selected-plan-space can be joined without one extra reconstruction step
- a top-level `search` section that records staged lane narrowing
  (`techmap_feedback_attempts`,
  `techmap_feedback_accepted_reconfigurations`,
  `placement_seed_lane_count`, `successful_placement_seed_count`,
  `routed_lane_count`), local-repair attempts and successes, route-aware
  refinement passes, route-checkpoint rescore and restore counts,
  exact-neighborhood reroute attempts and accepted moves, coarse fallback move
  counts, accepted FIFO-bufferization toggles, and accepted outer joint-PnR
  rounds so Stage-4 and Stage-5 search behavior is observable without reading
  verbose logs

When migration-oracle or legacy-fallback machinery is enabled, family-level
baseline summaries should distinguish demand-driven structural-state
materialization from legacy-only materialization instead of collapsing them
into one count.

These materialization counts should be counted by unique materialized family
signature within the relevant path, not by the number of matched software
subgraphs that later reused that family.
- selected fused-unit summaries with stable plan-local identity
- contracted selected techmap-group nodes should carry that same stable
  selected-unit identity, so graph artifacts and summary tables can be joined
  directly without falling back to reverse lookup by contracted node id alone
- contracted fallback operation nodes should also carry per-node tech-map
  metadata such as candidate ids, support/config-class ids, and fallback
  reason, so graph-only inspection is not limited to fused groups
- contracted selected techmap-group nodes and contracted fallback operation
  nodes should also mirror support/config-class descriptor keys where those
  classes are already attached, so graph-only inspection can explain legality
  classes without reopening the top-level descriptor tables
- contracted selected techmap-group nodes and contracted fallback operation
  nodes should also mirror support-class legality semantics such as
  temporal/spatial kind and hard-capacity enforcement, so graph-only
  inspection does not lose the key distinction between spatial capacity and
  temporal reuse
- candidate support-class pools mirrored onto contracted nodes should carry the
  same legality semantics, not only support-class ids or keys, so graph-only
  inspection can distinguish which alternatives are temporal reuse candidates
  versus hard-capacity spatial pools
- graph-level support-class metadata should also include capacity where that
  concept exists, so graph-only inspection can see not only whether a class is
  hard-capacity constrained but also what the bound is
- config-class metadata mirrored onto contracted nodes should likewise carry
  temporal and reason semantics for both the chosen class and candidate pools,
  so graph-only inspection can audit temporal-sharing legality without
  reopening the top-level config-class table
- for chosen config classes, graph-level metadata should also expose the
  compatibility set in both id and key form, so graph-only inspection can see
  the legal temporal-sharing neighborhood directly
- contracted selected techmap-group nodes should also mirror the original and
  current Layer-2 selection score when feedback reselection can rescore
  candidates, so graph-only inspection retains the same score provenance as
  the JSON report
- selected fused-unit boundary semantics such as internal absorbed edges and
  external input or output bindings
- selected fused-unit score or score-breakdown information sufficient to
  explain Layer-2 selection decisions
- that score breakdown should expose not only fusion and boundary terms but
  also any explicit scarce-family pressure that biases Layer 2 away from
  support-poor FU families after hard legality is already enforced
- selected and non-selected candidate summaries may reflect feedback-adjusted
  scores rather than the original pre-feedback score, because cached
  reselection is allowed to modify the Layer-2 objective without reopening FU
  semantics
- when feedback can adjust the Layer-2 objective, candidate summaries and
  selected-unit summaries should preserve both the original score and the
  current score, so policy-driven rescoring is auditable without replaying the
  solve
- when feedback penalties come from multiple layers such as candidate id,
  family id, and config class, the report should preserve that penalty
  breakdown instead of collapsing everything into one opaque score delta
- candidate-table entries should expose the same score-breakdown semantics as
  selected-unit summaries, including feedback penalty terms, so selected and
  rejected candidates can be compared without replaying the objective offline
- selected-unit score breakdowns should include those feedback-penalty terms as
  first-class fields, not only the base structural heuristic terms
- selected fused-unit candidate pools for hardware node, support class, and
  config class choices
- selected fused-unit candidate pools should also mirror support/config-class
  descriptor semantics such as support kind, temporal bit, hard-capacity
  enforcement, capacity, and config-class reason or temporal bit, so the JSON
  summary does not lag behind the contracted-graph handoff contract
- selected fused-unit candidate pools and preferred bindings should also expose
  human-readable `pe_name` information alongside raw `hw_node` ids, so Layer-2
  review does not require a second ADG lookup just to interpret where a chosen
  or available binding sits spatially
- selected fused-unit summaries should also provide a structured
  per-candidate object view, not only parallel arrays, so downstream consumers
  do not need to zip hardware ids, support classes, config classes, and config
  fields back together by position
- selected fused-unit summaries should expose support-class and config-class
  descriptors alongside raw ids where a preferred choice or candidate pool is
  reported, so consumers can inspect Layer-2 legality semantics without
  rejoining those ids through the top-level class tables
- migration-path provenance showing whether a selected fused unit was produced
  by the demand-driven path or only survived because legacy fallback was used
- per-software-node tech-map summaries
- per-software-node tech-map summaries should carry the local candidate-id set,
  not only counts, so one software node can be traced back to the exact Layer-2
  candidates that covered it
- per-software-node and fallback-node summaries should carry support-class and
  config-class id sets, not only counts, so feedback or diagnostics can target
  the exact legality classes involved without reconstructing them from other
  tables
- per-software-node and fallback-node summaries should also expose the
  corresponding support-class and config-class descriptor keys, so common
  debugging and DSE inspection does not require a second join through the
  top-level class tables
- per-software-node and fallback-node summaries should also carry their
  contracted-node id when available, so software-node-space can be joined
  directly to the contracted graph without an external lookup table

When migration provenance is exported, reports should expose a stable categorical
field such as `origin_kind` rather than requiring downstream tools to infer the
state from multiple booleans.
- conservative fallback information for single-op coverage
- fallback diagnostics should identify the coupled selection component when that
  node belonged to one, so rejection reasons can be traced back to the exact
  overlap region without a second join through per-node summaries
- fallback diagnostics should also expose the local candidate-id set for that
  software node, so later feedback or debugging can target the exact Layer-2
  candidates that were available before fallback
- conservative fallback candidate summaries should expose support-class and
  config-class descriptors in addition to ids, so fallback-plan inspection can
  explain the same legality classes as the main selected-plan artifact
- fallback reasons may explicitly distinguish “no candidate existed” from “all
  previously known candidates were filtered out by feedback”, so the upward DSE
  loop can tell semantic absence from policy-driven exclusion
- feedback-driven candidate statuses should remain artifact-visible, so a
  candidate filtered by feedback is distinguishable from one rejected by the
  normal overlap or legality objective
- optional tech-map identities such as selection-component id or selected-unit
  id should serialize as `null` when absent, rather than leaking internal
  sentinel integers into the public output contract
- the conservative fallback graph artifact should carry per-node coverage and
  candidate-pool metadata, not only the top-level JSON summary
- the conservative fallback graph should mirror the same support/config-class
  legality semantics as the contracted selected graph as far as practical,
  including support-class kind/temporal/hard-capacity/capacity and config
  reason/temporal labels on candidate pools
- legacy-oracle miss samples should preserve structured family and hardware
  context, not only one opaque key string, so demand-driven regressions can be
  localized without reverse-parsing matcher keys

`node_mappings` must identify:

- software node id
- mapped hardware node id
- hardware resource name
- enclosing PE identity when applicable

`edge_routings` must identify:

- software edge id
- a route description whose step semantics are reconstructable
- whether the edge is routed through inter-component hardware or absorbed as an
  intra-FU edge by tech-mapping
- whether the edge is an internal temporal-register dependency rather than an
  inter-component route

`port_table` must identify:

- flat port id
- component kind
- component name
- local port index
- direction

`memory_regions` must identify:

- selected hardware memory node id and name
- hardware memory kind
- hardware `num_region`
- for each occupied region:
  - software memory node id
  - backing software memref argument index
  - selected bridge or tag lane
  - load/store counts
  - `elem_size_log2`
- exported `addr_offset_table`

`temporal_registers` must identify:

- temporal PE name
- software edge id
- allocated register index
- writer software and hardware node ids
- reader software and hardware node ids
- writer output index and reader input index

## Extended Visualization Payload

The current FCC `map.json` report also exposes component-local routing facts
needed by visualization and config inspection.

Current sections include:

- `switch_routes`
  - per switch, list of configured `input_index -> output_index` selections
- `pe_routes`
  - per PE, list of selected ingress and egress mux or demux bindings
- `fu_configs`
  - selected effective FU configuration per mapped hardware FU
  - software nodes absorbed into that FU
  - selected `mux` fields such as `sel`
  - configurable software-op fields such as:
    - `handshake.constant` literal value
    - `handshake.join` active-input bitmask `join_mask`
    - `arith.cmpi` / `arith.cmpf` predicate
    - `dataflow.stream` continuation condition
- `tag_configs`
  - per-tag-boundary runtime configuration
  - `fabric.add_tag` exports one constant `tag`
  - `fabric.map_tag` exports `table_size`, tag widths, and the full structured
    table entries `[valid, src_tag, dst_tag]`
- `fifo_configs`
  - per-FIFO runtime configuration
  - only present for bypassable FIFOs
  - exports whether the FIFO is currently bypassed
  - exports the physical FIFO `depth`
  - if mapper post-route FIFO bufferization accepts a timing cut on one routed
    FIFO, this section reflects the mapper-selected `bypassed` runtime value
- `timing`
  - mapper timing surrogate summary for the routed mapping
  - exports estimated critical-path delay, estimated clock period,
    estimated initiation interval, estimated throughput cost, recurrence
    pressure, critical-path edge ids, FIFO buffer counts, forced-buffered FIFO
    ids and depths, mapper-selected buffered FIFO ids and depths,
    bufferized edge ids, and per-recurrence-cycle summaries
- `temporal_registers`
  - explicit register-backed dependencies inside `temporal_pe`
  - the assigned register index per writer result

For memory-oriented visualization, `memory_regions` is not optional in
practice. It is the authoritative bridge between:

- software memref arguments
- software memory ops
- chosen hardware memory interfaces
- region-table configuration

These sections are part of the current report family because they remove
ambiguity from mapping-aware visualization and config inspection.

Current `edge_routings` entries also classify the software edge into one of:

- `routed`
- `unrouted`
- `intra_fu`
- `temporal_reg`

## Text Mapping Report

The text report is for human inspection. It should summarize:

- Layer-2 tech-map summary, including selected fused units and important
  diagnostics
- when feedback reselection is active, the applied feedback request should be
  summarized in human-readable form, not only as aggregate counts
- mapper timing summary, including critical-path, clock-period, II, throughput,
  FIFO-buffer, recurrence-cycle surrogates, critical-path edge ids, and FIFO
  depth reporting when available
- node placements
- edge routes
- PE utilization
- important diagnostics or omissions

The text report is informative, not the primary data interface.

## Visualization HTML

The visualization HTML is self-contained and embeds:

- ADG data
- DFG data
- mapping data
- renderer assets

The HTML is a consumer of mapping JSON semantics, not a separate source of
mapping truth.
