# FCC Mapper Configuration Specification

## Purpose

FCC mapper tuning is defined in two layers:

- a repository-tracked base YAML template
- a small promoted subset of standalone CLI flags

The goal is to keep the full mapper heuristic surface configurable without
forcing users to pass long command lines for every run.

## Authoritative Sources

The authoritative field inventory is the checked-in base template:

- [`configs/mapper/default.yaml`](../configs/mapper/default.yaml)

The authoritative in-memory schema is:

- [`include/fcc/Mapper/MapperOptions.h`](../include/fcc/Mapper/MapperOptions.h)

The YAML comments in `configs/mapper/default.yaml` are normative parameter
descriptions. This document defines merge semantics, ownership rules, and which
parameters are promoted to standalone CLI flags.

## Config Resolution Order

FCC computes effective mapper options in this order:

1. compile-time built-in `MapperOptions` defaults
2. mapper base YAML
3. explicit CLI overrides

Normative behavior:

- if `--mapper-base-config <path>` is provided, FCC loads that YAML file
- if `--mapper-base-config` is omitted, FCC automatically loads the
  repository-tracked default file at [`configs/mapper/default.yaml`](../configs/mapper/default.yaml)
- if a YAML value and an explicit CLI value differ, FCC emits a warning and
  uses the CLI value
- if the YAML file cannot be opened, parsed, or validated, FCC must fail
  before mapping starts

## YAML Shape

The base config file is a YAML document with this top-level shape:

    version: 1
    mapper:
      ...

`version` is required for forward compatibility. The current schema version is
`1`.

The `mapper` mapping contains:

- promoted top-level controls such as budget, seed, lane count, and CP-SAT
  enablement
- grouped sub-maps such as `refinement`, `lane`, `routing`, `congestion`,
  `timing`, `bufferization`, `tech_feedback`, `cpsat_tuning`, and
  `local_repair`

The `lane` map owns speculative-lane generation and staged narrowing. Its
fields control:

- automatic serial fallback and auto lane-cap behavior
- optional routing-stage beam width after placement/refinement lanes are scored
- seed spacing and restart spacing between lanes
- budget reservation for final polish on the selected lane
- minimum CP-SAT time floors for preferred global and boundary-repair lanes

The `congestion` map owns negotiated-routing feedback controls such as:

- saturation penalties
- history-increment caps
- historical-congestion decay
- routing-output history bump and decay
- early-termination window for non-improving negotiated iterations

The `timing` map owns mapper timing-proxy and timing-summary constants, including:

- recurrence-edge weight amplification used by placement and repair heuristics
- recurrence-node latency and interval penalties used when choosing hardware
  candidates for recurrence-critical software nodes
- surrogate combinational node delay used by mapper timing analysis
- surrogate routed-hop delay used by mapper timing analysis

The `bufferization` map owns the post-route FIFO-only timing-cut loop. Its
fields control:

- enablement
- the maximum number of accepted FIFO buffering toggles
- the maximum number of outer joint PnR and FIFO-bufferization rounds
- the minimum throughput-cost improvement required for acceptance
- the minimum clock-period improvement used as a tie-breaker when throughput
  cost remains unchanged

The `tech_feedback` map owns the Layer-3 to Layer-2 reconfiguration loop. Its
fields control:

- enablement
- the maximum number of techmap reselection retries within one mapper run
- the maximum number of temporal-conflict or routing-failure hotspots that may
  be translated into Layer-2 split or ban actions during one retry

The `local_repair` map owns the post-routing repair stack. Its fields control:

- global enablement so ablation and benchmark runs can disable local repair
- exact-neighborhood repair thresholds and deadlines
- focused hotspot and residual repair tuning
- CP-SAT escalation thresholds used inside local repair

The `relaxed_routing` map owns an optional negotiated-routing extension that
permits temporary overuse of non-tagged switch-owned routing outputs during one
iteration, followed by strict legalization before a checkpoint may be accepted.
Its fields control:

- enablement
- how many legalization passes are allowed after one relaxed iteration
- the additive penalty for reusing an already claimed routing output
- the multiplicative growth of that penalty as more logical sources compete
- the checkpoint-rejection cap for too many simultaneously overused outputs

The `refinement` map owns simulated-annealing placement tuning, including:

- geometric cooling (`initial_temperature`, `cooling_rate`)
- adaptive cooling enablement and acceptance-ratio windowing
- coarse route-checkpoint reroute/rescore enablement and the accepted-move
  batch size that triggers it
- route-aware accepted-move neighborhood reroute enablement and its
  neighborhood-size cap
- cold-search reheating and hot-search extra cooling multipliers
- plateau reheating thresholds and temperature clamps

The top-level `snapshot_interval_seconds` and `snapshot_interval_rounds`
controls own periodic mapper snapshot emission:

- both default to `-1` in the repository template
- `-1` means disabled
- only one of the two may be enabled at a time
- `snapshot_interval_seconds` is evaluated at mapper progress checkpoints, not
  by an asynchronous timer thread

## CLI Promotion Rule

All mapper heuristics and thresholds must be configurable through the base YAML.
Only a small high-frequency tuning subset may also receive dedicated CLI flags.

Parameters should be promoted to standalone CLI only when they are:

- used frequently in ad-hoc tuning loops
- stable enough to deserve short names
- broad enough to affect the outer mapper strategy rather than one narrow
  repair heuristic

Parameters should remain config-only when they are:

- narrowly scoped to one repair or endgame heuristic
- unlikely to be tuned on every command invocation
- part of a larger grouped strategy that is easier to reason about in YAML

Adaptive simulated-annealing cooling currently remains config-only. It is part
of the grouped `refinement` strategy and is expected to be tuned together with
the base SA temperature schedule rather than as an ad-hoc standalone flag.

Relaxed routing also remains config-only. It materially changes negotiated
routing semantics and is expected to be tuned together with the broader routing
strategy rather than as a frequently flipped standalone flag.

## Promoted CLI Surface

The current promoted mapper CLI flags are:

- `--mapper-base-config`
- `--mapper-budget`
- `--mapper-seed`
- `--mapper-lanes`
- `--mapper-snapshot-interval-seconds`
- `--mapper-snapshot-interval-rounds`
- `--mapper-interleaved-rounds`
- `--mapper-selective-ripup-passes`
- `--mapper-placement-move-radius`
- `--mapper-cpsat-global-node-limit`
- `--mapper-cpsat-neighborhood-node-limit`
- `--mapper-cpsat-time-limit`
- `--mapper-enable-cpsat`
- `--mapper-routing-heuristic-weight`
- `--mapper-negotiated-routing-passes`
- `--mapper-congestion-history-factor`
- `--mapper-congestion-history-scale`
- `--mapper-congestion-present-factor`
- `--mapper-congestion-placement-weight`

All other mapper knobs currently remain config-only and are expected to be set
through the base YAML.

## Validation

Mapper configuration must be validated before mapping starts.

Validation covers:

- required ranges such as positive time limits and positive candidate caps
- cross-field constraints such as max values not being smaller than min values
- schema version compatibility

Invalid configuration is a tool invocation error, not a mapper failure.

Validation also covers:

- `snapshot_interval_seconds` must be `-1` or `> 0`
- `snapshot_interval_rounds` must be `-1` or `> 0`
- the two snapshot modes must not both be enabled
- `refinement.target_acceptance_low` and
  `refinement.target_acceptance_high` must both be within `[0, 1]`, and the
  low threshold must not exceed the high threshold
- `refinement.cold_acceptance_reheat_multiplier`,
  `refinement.plateau_reheat_multiplier`, and
  `refinement.max_temperature_scale` must be `>= 1`
- `refinement.hot_acceptance_cooling_multiplier` must be within `[0, 1]`
- `refinement.min_temperature` and `refinement.adaptive_window` must be
  positive
- `timing.recurrence_edge_weight_multiplier`,
  `timing.combinational_node_delay`, and `timing.routing_hop_delay` must be
  positive
- `timing.recurrence_node_latency_weight` and
  `timing.recurrence_node_interval_weight` must be non-negative
- `bufferization.max_iterations` must be positive
- `bufferization.outer_joint_iterations` must be positive
- `bufferization.min_throughput_improvement` and
  `bufferization.clock_tie_break_improvement` must be non-negative
- `tech_feedback.max_retries` must be positive
- `tech_feedback.max_targets_per_retry` must be positive

## Related Documents

- [spec-cli.md](./spec-cli.md)
- [spec-mapper.md](./spec-mapper.md)
