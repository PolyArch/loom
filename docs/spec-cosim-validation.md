# Loom Co-Simulation Validation Specification

## Overview

This document is the authoritative validation and acceptance specification for
Loom `cosim`. It defines required artifacts, end-to-end execution procedure,
pass/fail criteria, and regression test matrix.

Protocol/runtime/trace details are defined in their respective sub-specs.

## Validation Scope

Validation proves end-to-end behavior from host software to simulated
accelerator backend and back to host result checking.

A valid run demonstrates:

- successful configuration transfer of mapper-produced `config_mem`
- successful accelerator execution with ESI-based I/O
- successful output comparison against CPU reference
- optional trace/perf collection consistency

Validation tool priority follows [spec-adg-tools.md](./spec-adg-tools.md):
Synopsys VCS/Verdi is the primary execution/debug flow, and Verilator/GTKWave
is the secondary open-source fallback.

## Required Artifacts

A test case must provide:

- accelerator program interface (from Stage A output and manifest)
- hardware backend model (SystemC or RTL simulation target)
- mapper configuration image (`config_mem` words)
- host input vectors
- CPU reference implementation for expected outputs

Optional artifacts:

- trace/perf enable configuration
- replay metadata (seed/scheduling mode/backend options)

## Canonical End-to-End Procedure

The canonical `cosim` execution procedure is:

1. **Prepare backend**
   - build or launch SystemC/RTL simulation backend
   - ensure connection metadata is available (`host:port` or `cosim.cfg`)

2. **Connect host runtime**
   - create session and connect
   - fetch and validate manifest
   - verify required services/channels

3. **Load configuration**
   - upload mapper `config_mem` words in authoritative address order
   - optional readback verification

4. **Run invocation(s)**
   - send input payload(s)
   - start invocation
   - wait completion
   - receive output payload(s)

5. **Compare outputs**
   - compute CPU reference outputs
   - compare accelerator and CPU outputs with configured policy

6. **Collect observability data**
   - gather trace/perf artifacts when enabled
   - verify schema/ordering/counter consistency

7. **Report verdict**
   - pass only if all required checks succeed

## Pass/Fail Criteria

A test case is `PASS` only when all conditions hold:

- session lifecycle completed without fatal protocol/runtime error
- all required outputs were received and decoded successfully
- CPU comparison succeeded under selected compare policy
- if trace/perf enabled, telemetry satisfies schema and consistency rules

A test case is `FAIL` if any required condition fails.

## Compare Policies

Validation framework must support explicit compare policies per output:

- exact bitwise equality
- integer tolerance window
- floating-point epsilon/relative tolerance
- custom comparator function

Policy choice must be recorded in run report.

## Required Run Report Fields

Each invocation report must include at least:

- backend id and connection mode
- deterministic mode flag and seed (if applicable)
- epoch id and invocation id
- config image checksum
- cycle start/end if available
- compare policy id
- pass/fail verdict and mismatch summary

## Regression Matrix

A conforming validation suite must include these scenarios.

1. **Single invocation baseline**
   - one configuration epoch
   - one invocation
   - exact output match

2. **Repeated invocation same epoch**
   - one configuration epoch
   - multiple invocations
   - stable outputs across repetitions

3. **Reconfiguration across epochs**
   - two or more different config images
   - per-epoch invocation verification

4. **Concurrent host request submission**
   - multiple host worker threads submit requests
   - dispatcher preserves required ordering

5. **Trace/perf enabled run**
   - event and perf streams emitted
   - schema/ordering/counter checks pass

6. **Failure-path diagnostics**
   - inject at least one controlled failure (for example missing required
     channel) and verify error classification/reporting
7. **Secondary simulator parity**
   - execute a representative subset on Verilator fallback flow
   - verify functional verdict parity with primary VCS flow

## Determinism Check

For deterministic mode, run the same test case at least twice with identical
inputs and configuration.

Required expectation:

- identical functional outputs
- identical invocation verdicts
- identical ordering for deterministic report fields

Trace byte-for-byte identity is recommended but may be relaxed if timestamps are
known to include documented nondeterministic fields; such exceptions must be
explicitly declared.

## Recovery and Retry Rules

Validation harness must support safe retry behavior:

- failed invocation may be retried after clean reconnect and reload
- stale session resources must not be reused after fatal error
- retry count and reason must be logged in report

## Minimal Implementation Path for New Contributors

A practical minimal path:

1. implement baseline connect/manifest/config/invoke/compare loop
2. add multi-thread host submission with single dispatcher
3. add deterministic-mode replay checks
4. add trace/perf collection and checks
5. add failure-injection regression tests

This sequence is sufficient for undergraduate implementation with incremental
confidence.

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-architecture.md](./spec-cosim-architecture.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-trace.md](./spec-cosim-trace.md)
- [spec-adg-tools.md](./spec-adg-tools.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
