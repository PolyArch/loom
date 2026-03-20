# FCC CLI Specification

## Overview

`fcc` currently supports five invocation families:

1. source compilation without mapping
2. source compilation plus ADG mapping
3. DFG-direct mapping
4. visualization-only export
5. mapped-case runtime replay

`-o <output_dir>` is required in all modes.

## Invocation Modes

### Source Compilation Only

```
fcc <sources> -o <output_dir> [-I<include_path> ...]
```

This mode runs the software frontend and lowering pipeline only. It emits
software-side IR and host code artifacts, but does not load an ADG, run the
mapper, or generate mixed mapping artifacts.

Representative outputs:

- `<dfg>.ll`
- `<dfg>.llvm.mlir`
- `<dfg>.cf.mlir`
- `<dfg>.scf.mlir`
- `<dfg>.dfg.mlir`
- `<dfg>_host.c`
- `fcc_accel.h`
- `fcc_accel.c`

### Source Compilation Plus Mapping

```
fcc <sources> --adg <fabric.mlir> -o <output_dir> [options]
```

This mode runs the full software frontend, lowers to DFG, loads and verifies
the ADG, runs place-and-route, emits mapping/configuration artifacts, writes a
runtime manifest, and optionally runs standalone simulation.

### DFG-Direct Mapping

```
fcc --dfg <dfg.mlir> --adg <fabric.mlir> -o <output_dir> [options]
```

This mode skips source compilation and maps a pre-built DFG onto a pre-built
ADG.

`--dfg` requires `--adg` outside visualization-only mode.

### Visualization Only

```
fcc --viz-only [--dfg <dfg.mlir>] [--adg <fabric.mlir>] -o <output_dir>
fcc --viz-only --map-json <map.json> [--dfg <dfg.mlir>] [--adg <fabric.mlir>] -o <output_dir>
```

This mode emits HTML visualization without running mapping.

Normative rules:

- `--viz-only` requires at least one of `--dfg` or `--adg`
- `--map-json` is meaningful only in visualization-only mode
- if `--map-json` is present, FCC overlays an existing mapping onto the
  visualization instead of regenerating one

### Runtime Replay

```
fcc --runtime-manifest <case.runtime.json> \
    --runtime-request <request.json> \
    --runtime-result <result.json> \
    -o <output_dir> \
    [--runtime-trace <trace_path>] \
    [--runtime-stat <stat_path>]
```

This mode bypasses compilation and mapping. FCC loads an existing mapped-case
runtime manifest, replays one invocation on the standalone simulator, and
writes result JSON plus trace/stat artifacts.

Normative rules:

- `--runtime-manifest` requires both `--runtime-request` and `--runtime-result`
- if `--runtime-trace` is omitted, FCC writes
  `<output_dir>/<case_name>.runtime.trace`
- if `--runtime-stat` is omitted, FCC writes
  `<output_dir>/<case_name>.runtime.stat`

## Option Reference

### General Input and Output

| Option | Meaning |
|--------|---------|
| `-o <dir>` | Output directory. Required in all modes. |
| `-I<path>` | Frontend include path forwarded to clang. Only meaningful in source-compilation modes. |
| `--adg <path>` | Path to input ADG in Fabric MLIR. |
| `--dfg <path>` | Path to pre-built DFG MLIR. |

### Visualization

| Option | Meaning |
|--------|---------|
| `--viz-only` | Emit visualization without running mapping. |
| `--map-json <path>` | Reuse an existing mapping JSON when regenerating visualization. |
| `--viz-layout <default\|neato>` | Visualization layout policy. |

`--viz-layout` semantics:

- `default`:
  - if the ADG provides an explicit sidecar layout, use it
  - otherwise use FCC's offline auto-layout path
- `neato`:
  - ignore explicit sidecar placement and force offline neato-style layout

### Standalone Simulation After Mapping

| Option | Meaning |
|--------|---------|
| `--simulate` | Run the standalone simulator after a successful mapping. |
| `--sim-bundle <path>` | Explicit simulation bundle JSON with concrete inputs and expected outputs. |
| `--sim-max-cycles <n>` | Simulation cycle budget. Default: `1000000`. |

Additional semantics:

- `--simulate` is effective only in modes that actually perform mapping
- if `--sim-bundle` is omitted, FCC synthesizes a standalone simulation setup
  from the mapped graph
- if `--sim-bundle` is present, FCC also emits a validation report comparing
  simulated outputs against bundle expectations

### Runtime Replay

| Option | Meaning |
|--------|---------|
| `--runtime-manifest <path>` | Path to mapped-case runtime manifest JSON. |
| `--runtime-request <path>` | Runtime invocation request JSON. |
| `--runtime-result <path>` | Output path for replay result JSON. |
| `--runtime-trace <path>` | Optional explicit trace output path. |
| `--runtime-stat <path>` | Optional explicit stat output path. |

### Mapper Controls

FCC now always resolves mapper settings from a base YAML template plus an
explicit CLI override layer.

Authoritative merge semantics and the full mapper parameter inventory are
defined in [spec-mapper-config.md](./spec-mapper-config.md).

The following promoted mapper controls are intentionally exposed by the CLI.

| Option | Meaning |
|--------|---------|
| `--mapper-base-config <path>` | Path to mapper base-config YAML. If omitted, FCC loads the repository-tracked default template at `configs/mapper/default.yaml`. |
| `--mapper-budget <seconds>` | Overall mapper time budget. Default: `60`. |
| `--mapper-seed <n>` | Deterministic random seed. Default: `0`. |
| `--mapper-lanes <n>` | Parallel multi-start lane count. `0` means auto-select. |
| `--mapper-snapshot-interval-seconds <x>` | Emit periodic mapper snapshots every `x` wall-clock seconds. `-1` disables time-based snapshots. |
| `--mapper-snapshot-interval-rounds <n>` | Emit periodic mapper snapshots every `n` mapper progress rounds. `-1` disables round-based snapshots. |
| `--mapper-interleaved-rounds <n>` | Number of interleaved place-route rounds. Default: `4`. |
| `--mapper-selective-ripup-passes <n>` | Failed-edge selective rip-up passes per routing round. Default: `3`. |
| `--mapper-placement-move-radius <n>` | Detailed placement move radius in Manhattan distance. `0` means unrestricted. Default: `3`. |
| `--mapper-cpsat-global-node-limit <n>` | Maximum neighborhood size for CP-SAT global placement. Default: `24`. |
| `--mapper-cpsat-neighborhood-node-limit <n>` | Maximum neighborhood size for CP-SAT local repair. Default: `8`. |
| `--mapper-cpsat-time-limit <seconds>` | Per-solve CP-SAT time limit. Default: `0.75`. |
| `--mapper-enable-cpsat=<bool>` | Enable OR-Tools CP-SAT placement refinement. Default: `true`. |
| `--mapper-routing-heuristic-weight <x>` | Weighted A* heuristic multiplier for routing. Default: `1.5`. |
| `--mapper-negotiated-routing-passes <n>` | Negotiated congestion routing iterations. `0` disables negotiated routing. Default: `12`. |
| `--mapper-congestion-history-factor <x>` | Negotiated-routing history increment. Default: `1.0`. |
| `--mapper-congestion-history-scale <x>` | Negotiated-routing history scaling per iteration. Default: `1.5`. |
| `--mapper-congestion-present-factor <x>` | Negotiated-routing present-demand weight. Default: `1.0`. |
| `--mapper-congestion-placement-weight <x>` | Placement penalty weight derived from congestion estimation. Default: `0.3`. |

Current implementation note:

- standalone CLI flags are only the promoted high-frequency tuning subset
- all other mapper thresholds and heuristic constants are config-only and must
  be set through the base YAML
- relaxed negotiated routing is currently config-only under
  `mapper.relaxed_routing`
- `--mapper-snapshot-interval-seconds` and
  `--mapper-snapshot-interval-rounds` are mutually exclusive; enabling both is
  a CLI error
- if neither snapshot flag is enabled, FCC does not emit periodic mapper
  snapshots

## Output Naming Rules

FCC derives three stems:

- software stem:
  - from the first source file in source-compilation modes
  - from `--dfg` when using DFG-direct mode
  - from `--runtime-manifest` with a trailing `.runtime` suffix removed in
    runtime replay mode
- ADG stem:
  - from `--adg`
  - if the input filename stem ends with `.fabric`, that trailing stem
    component is dropped
- mixed stem:
  - `<software>.<adg>` when both exist
  - otherwise whichever stem is available

Representative base paths:

- software base: `<output_dir>/<software>`
- hardware base: `<output_dir>/<adg_or_software>`
- mixed base: `<output_dir>/<mixed>`

## Output Artifacts by Mode

### Source Compilation

Source compilation emits software artifacts under the software base:

- `<software>.ll`
- `<software>.llvm.mlir`
- `<software>.cf.mlir`
- `<software>.scf.mlir`
- `<software>.dfg.mlir`
- `<software>_host.c`
- `fcc_accel.h`
- `fcc_accel.c`

### Mapping

Successful or failed mapping attempts emit mixed artifacts under the mixed base:

- `<mixed>.config.bin`
- `<mixed>.config.json`
- `<mixed>.config.h`
- `<mixed>.map.json`
- `<mixed>.map.txt`
- `<mixed>.viz.html`

If periodic mapper snapshots are enabled, FCC also emits:

- `<output_dir>/mapper-snapshots/<mixed>.snapshot-<seq>.<trigger>.mapper-<ordinal>.map.json`
- `<output_dir>/mapper-snapshots/<mixed>.snapshot-<seq>.<trigger>.mapper-<ordinal>.viz.html`

Snapshot emission is best-effort and captures the current best expanded mapper
checkpoint at the trigger point.

If mapping succeeds, FCC also emits:

- `<mixed>.runtime.json`

### Standalone Simulation

If `--simulate` is enabled and mapping succeeds, FCC emits:

- `<mixed>.sim.trace`
- `<mixed>.sim.stat`
- `<mixed>.sim.setup.json`
- `<mixed>.sim.result.json`

If `--sim-bundle` is provided and validation succeeds or fails, FCC also emits:

- `<mixed>.sim.report.json`

### Visualization Only

Visualization-only mode emits one HTML file:

- with both DFG and ADG:
  - `<mixed>.viz.html`
- with DFG only:
  - `<software>.viz.html`
- with ADG only:
  - `<adg>.viz.html`

### Runtime Replay

Runtime replay always writes the explicitly requested result path:

- `<runtime-result path>`

It also writes trace/stat artifacts either to explicitly requested paths or to
default paths under `-o`.

## Validation and Responsibility Boundaries

The CLI is responsible for:

- parsing mode-specific arguments
- creating the output directory
- deriving software, ADG, and mixed stems
- invoking ADG verification before flattening
- selecting the correct high-level pipeline
- forwarding mapper options into `Mapper::Options`

The CLI is not the authority for:

- Fabric IR legality beyond invoking verifier entry points
- mapper legality rules
- routing semantics
- configuration encoding semantics
- simulator microarchitectural semantics

Those behaviors are specified by the corresponding subsystem specifications.
