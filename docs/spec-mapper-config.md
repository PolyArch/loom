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
  `cpsat_tuning`, and `local_repair`

The `congestion` map owns negotiated-routing feedback controls such as:

- saturation penalties
- history-increment caps
- historical-congestion decay
- routing-output history bump and decay
- early-termination window for non-improving negotiated iterations

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

## Related Documents

- [spec-cli.md](./spec-cli.md)
- [spec-mapper.md](./spec-mapper.md)
