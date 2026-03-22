# LOOM Trace and Performance Specification

## Overview

LOOM trace and performance output provides cycle-level visibility into mapped
accelerator execution for both standalone and gem5-backed runs.

## Goals

Trace and performance data should support:

- per-node execution visibility
- route and resource activity debugging
- cycle-level regression comparison
- post-run summary statistics for performance analysis
- visualization playback and highlighting

## Canonical Event Model

Trace events conceptually include:

- cycle
- hardware node id
- event kind
- event-specific payload fields

A minimal conceptual event family includes:

- node fire
- input stall
- output stall
- route use
- config write
- invocation start
- invocation done
- runtime error

## Route Visibility

Trace must be rich enough to connect activity back to mapped routes where
useful. This is especially important for visualization playback in mapping-on
mode.

## Summary Statistics

Performance summaries should cover at least:

- active cycles
- input stall cycles
- output stall cycles
- transferred token counts
- configuration write counts where relevant

Derived metrics such as utilization ratios may be computed from these fields.

## Output Artifacts

The planned trace-related artifact family includes:

- `.trace` for detailed event streams
- `.stat` for summary statistics

The concrete encoding may evolve, but the semantic field set should remain
stable enough for validation and visualization.

## Ordering Rules

Trace ordering must be deterministic under deterministic execution settings.

At minimum:

- events are ordered by cycle
- same-cycle event ordering must be stable
- invocation start precedes its execution events
- invocation completion follows the final functional activity of that invocation

## Relationship to Visualization

Visualization playback consumes trace semantics but does not redefine them.
If trace sampling or filtering is used, the resulting playback limits must be
documented explicitly.

## Related Documents

- [spec-viz.md](./spec-viz.md)
- [spec-simulation.md](./spec-simulation.md)
- [spec-validation.md](./spec-validation.md)
