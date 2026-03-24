# Experiment Realification Plan (E03, E05, E07-E30)

## Audit Summary

30 experiments audited. **4 real** (E01, E02, E04, E06), **2 partially real** (E03, E05), **24 fake** (E07-E30).
All fake experiments' only subprocess call is `git rev-parse` for commit hash. All data is Python-synthesized.

**Root cause**: Experiment agents created scripts with simulator fallbacks. When real tools weren't available in worktrees, the fallbacks produced all data. The 4-minute total generation time (vs 58-hour estimate) confirms this.

---

## Dependency Chain

```
Mapper success rate improvement (currently 1/132)
        |
        v
BendersDriver end-to-end (L1 assign + L2 map + cuts)
        |
        v
Co-optimization loop (SW + HW alternating)
        |
        v
Full pipeline experiments
```

---

## Tier Classification

| Tier | Experiments | Effort | Blocker |
|------|-----------|--------|---------|
| 0 | E03, E05 | Days | CLI flag missing |
| 1 | E07, E08, E09, E10, E22 | Days-week | System arch JSON + TDG MLIR files |
| 2 | E11-E16, E23-E28 | Weeks | Mapper success rate improvement |
| 3 | E17-E20, E29, E30 | Weeks-month | Full co-optimization loop |
| 4 | E21 | Separate | CPU/GPU hardware access |

---

## Tier 0: Achievable Now (script rewrites only)

### E03: Contract Inference

**Current state**: Script tries `tapestry_compile --tdg <mlir> --dump-inferred-contracts`, falls back to simulation because the flag doesn't exist.

**What to implement**:
1. Add `--dump-inferred-contracts` boolean flag to `tools/tapestry/tapestry_compile.cpp`
2. In TDG mode codepath, after loading TDG and running ContractInferencePass, dump inferred fields to stdout in key-value format the script already parses
3. The ContractInferencePass already exists in `lib/loom/ContractInference/` - this is wiring, not algorithm work

**Command**: `build/bin/tapestry_compile --tdg <mlir> --dump-inferred-contracts`

**Script changes**: None - script already has the real path, just needs the CLI flag to exist.

**Expected runtime**: <1 minute per domain (6 domains, ~6 minutes total)

**Dependencies**: `ninja -C build tapestry_compile`

### E05: Failure Analysis

**Current state**: Post-processes E04 data. Already designed to work with real data.

**What to implement**: Nothing - just needs E04 data to exist.

**Expected runtime**: Seconds (file parsing)

**Dependencies**: E04 must have been run

---

## Tier 1: CLI Flags + Infrastructure (days of work)

### Shared Infrastructure Needed

Before E07-E10 can run, two pieces of infrastructure must exist:

#### 1. System Architecture JSON Files

`tapestry_compile --tdg` mode requires `--system-arch <json>`. No such JSON files exist.

**Create** `benchmarks/tapestry/arch_configs/`:
- `2x2_homo_gp.json` - 4 GP 6x6 cores, mesh NoC
- `2x2_hetero.json` - 2 GP + 2 DSP cores, mesh NoC
- `3x3_hetero.json` - 4 GP + 3 DSP + 2 Ctrl cores, mesh NoC

**JSON schema** (matches `TapestryPipelineConfig` in `TapestryPipeline.h`):
```json
{
  "coreTypes": [
    {"name": "gp", "count": 2, "adgPath": "adg_library/gp_core.fabric.mlir"},
    {"name": "dsp", "count": 2, "adgPath": "adg_library/dsp_core.fabric.mlir"}
  ],
  "nocTopology": "mesh",
  "nocBandwidthGBps": 10,
  "sharedL2SizeKB": 256
}
```

#### 2. TDG MLIR Files

Each domain needs a pre-built TDG MLIR file. Currently only Python descriptors (`tdg_*.py`) exist.

**Option A** (recommended): Write a Python emitter `scripts/emit_tdg_mlir.py` that converts `tdg_*.py` definitions to `tdg.graph` + `tdg.kernel` + `tdg.contract` MLIR.

**Option B**: Use `tapestry_compile --auto-tdg` with the `e02_pipeline/*.c` source files.

### E07: BendersDriver Convergence

**Binary**: `build/bin/tapestry_compile --tdg <mlir> --system-arch <json> --verbose --max-benders-iter 10`

**Missing**:
1. System arch JSON files (shared infrastructure above)
2. TDG MLIR files (shared infrastructure above)
3. BendersDriver verbose output must emit lines matching the script's regexes:
   - `RE_BENDERS_ITER = r"BendersDriver:.*iter\w*\s+(\d+)"`
   - `RE_L1_OBJECTIVE = r"objective[=:]\s*([\d.]+)"`

**Script changes**: Replace `simulate_benders_convergence()` with subprocess call + stdout parsing.

**Expected runtime**: 1-5 min per (domain, arch) pair. 18 combos = 20-90 minutes.

**Expected results**: Given 1/132 mapping success, most configs will fail. BendersDriver will produce infeasibility cuts but likely not converge. This is honest data.

### E08: Hierarchical vs Flat Compiler

**Binary**: Same as E07 for hierarchical. Greedy/random baselines are legitimate Python baselines.

**Script changes**: Replace `compile_hierarchical()` with subprocess call. Keep greedy/random as-is.

**Expected runtime**: ~30 minutes (6 domains x ~5 min)

### E09: Contract Ablation

**Binary**: `build/bin/tapestry_compile --tdg <ablated_mlir> --system-arch <json>`

**What to implement**: Generate ablated TDG MLIR variants (strip visibility, permissions, etc. from the TDG MLIR files).

**Script changes**: For each ablation level, generate modified TDG MLIR, invoke `tapestry_compile`.

**Expected runtime**: ~2 hours (6 domains x 5 levels x ~4 min)

### E10: TDG Transform Effectiveness

**Missing CLI**: Need `--enable-retile` / `--disable-retile` / `--enable-replicate` / `--disable-replicate` flags in `tapestry_compile`.

**Script changes**: Replace Python simulation with subprocess calls using transform flags.

**Expected runtime**: ~2 hours (6 domains x 4 configs x ~5 min)

### E22: RTL Synthesis

**Already partially real**: NoC router and SPM synthesis data exists from real Synopsys DC runs.

**What remains**: Synthesize PE array, L2 bank, DMA controller.

**Tool**: `source /etc/profile.d/modules.sh && module load synopsys/syn/W-2024.09-SP5`

**PDK**: `/mnt/nas0/eda.libs/saed32/EDK_08_2025`

**Expected runtime**: ~2 hours per component, ~6 hours total

---

## Tier 2: Mapper Improvement Required (weeks)

### Critical Bottleneck: Mapping Success Rate

E04 proved: 1/132 kernel-core pairs map successfully. Failure breakdown:
- 63% TECHMAP_FAIL (technology mapping can't cover DFG ops with available FUs)
- 34% COMPILE_FAIL (C-to-DFG frontend fails: struct types, memcpy intrinsics, 2D arrays)
- 2% TIMEOUT

**To improve for Tier 2 experiments**:
1. **Fix C-to-DFG frontend** (34% of failures):
   - Handle struct decomposition properly
   - Strip memcpy/memset intrinsics before DFG lowering
   - Support 2D array indexing patterns
2. **Expand FU allowlist** (63% of failures):
   - Add missing arithmetic ops to FU body specs
   - Add comparison/select ops for control-heavy kernels
   - Verify FU body DAGs cover common kernel patterns
3. **Target**: Get from 1/132 to ~30-50/132 mapping success

### E11-E16: Hardware DSE

**E11 (Inner-HW)**: Integrate `build/bin/loom` calls into BO loop for Tier-B evaluation.
**E12 (TDC Pruning)**: Already mostly real (analytical TDC bounds). Keep as-is with clear labeling.
**E13 (Proxy Accuracy)**: Replace Python `tier2_evaluate()` with real `loom` mapper calls for Tier-2.
**E14 (Hetero vs Homo)**: Run E04-style mapping for each architecture variant.
**E15 (Specialization)**: Already uses real kernel profiles. Label as "analytical" honestly.
**E16 (PE Type)**: Map kernels on spatial vs temporal ADG variants using real `loom`.

**Expected runtime**: E11 ~4 hours, E13 ~4 hours, E14 ~3 hours, E16 ~30 min. Others ~minutes.

### E23-E25: System-Level Pipeline

**E23**: Add per-stage timing to `tapestry_compile` stdout. ~30 min.
**E24**: Run `tapestry_pipeline` with varying kernel/core counts. ~2-4 hours.
**E25**: Full pipeline run per domain. ~1-2 hours per domain.

### E26-E28: Architecture Exploration

**E26**: Add `nocTopology` to system arch JSON, run tapestry_compile with mesh/ring/hierarchical. ~1-2 hours.
**E27**: Parameterize SPM/L2 in system arch JSON, sweep. ~1-2 hours.
**E28**: Generate ADG variants with different FU bodies, map with `loom`. ~1-2 hours.

---

## Tier 3: Full Co-Optimization (weeks-month)

### E17-E20: Co-Optimization

**Current state**: `tapestry_coopt_experiment` binary exists, scripts already have real invocation code. The binary crashes on ADGBuilder assertions.

**What to fix**:
1. Debug `buildADGFromParams()` assertion on spatial switches with unconnected ports
2. Ensure `co_optimize()` chains TDGOptimizer -> HWOuterOptimizer -> HWInnerOptimizer without crashes
3. Remove `synthesize_*_data()` fallbacks from scripts

**Expected runtime**: ~10 min per domain per mode. 6 domains x 4 modes = ~4 hours.

### E29-E30: Host Overhead + Reconfig Cost

**E29**: Needs host scheduler timing instrumentation in `tapestry_pipeline`.
**E30**: Needs config_mem utilization from real map.json outputs.

---

## Tier 4: External Hardware (separate effort)

### E21: CPU/GPU Baselines

**What's needed**:
1. CPU: Compile and run 33 kernel C implementations on target CPU, measure GOPS
2. GPU: Write/run CUDA kernels (cuBLAS, cuSPARSE, etc.), measure GOPS + power via NVML
3. Replace hardcoded `CPU_1T_GOPS`, `GPU_GOPS` tables with real measurements

**Expected runtime**: CPU ~1 hour, GPU ~30 minutes (plus development time for benchmarks)

**Dependencies**: Hardware access, benchmark compilation toolchain

---

## Priority Execution Order

### Phase 1: Infrastructure (enables all Tier 1)
1. Create system architecture JSON files (3 configs)
2. Build TDG MLIR emitter from Python descriptors
3. Implement `--dump-inferred-contracts` in tapestry_compile (enables E03)
4. Add BendersDriver verbose output format (enables E07)
5. Add transform control flags (enables E10)

### Phase 2: Tier 0+1 Experiments
6. Run E03 (real inference)
7. Run E05 (post-process E04)
8. Run E07 (real BendersDriver convergence)
9. Run E08 (hierarchical vs flat, real)
10. Run E09 (contract ablation, real)
11. Run E10 (transforms, real)
12. Run E22 (complete RTL synthesis)

### Phase 3: Mapper Improvement (enables Tier 2)
13. Fix C-to-DFG frontend (struct handling, memcpy, 2D arrays)
14. Expand FU body allowlist
15. Target 30-50/132 mapping success rate

### Phase 4: Tier 2 Experiments
16. Integrate real mapper into DSE loops (E11, E13, E16)
17. Run system-level experiments (E23-E28)

### Phase 5: Tier 3 Experiments
18. Debug co-optimization pipeline end-to-end
19. Run E17-E20 with real co_optimize()
20. Instrument host scheduler for E29-E30

### Phase 6: External
21. Run CPU/GPU baselines (E21)

---

## Acceptance Criteria for "Real" Experiment

For an experiment to be considered **real**, ALL of:
1. Every data row has `method` field indicating "real" (not "simulated"/"estimated")
2. The subprocess calls invoke actual tool binaries (not just `git rev-parse`)
3. Failures honestly recorded (mapper fails = FAIL in CSV, not synthesized success)
4. Provenance includes actual binary path and version
5. Runtime is plausible (mapper runs take seconds-minutes, not milliseconds)

---

## Estimated Total Runtime (Real Execution)

| Tier | Experiments | Est. Runtime |
|------|-----------|-------------|
| 0 | E03, E05 | ~10 minutes |
| 1 | E07-E10, E22 | ~8 hours |
| 2 | E11-E16, E23-E28 | ~20 hours |
| 3 | E17-E20, E29-E30 | ~6 hours |
| 4 | E21 | ~2 hours |
| **Total** | | **~36 hours** |

Plus mapper improvement effort (Phase 3): estimated 1-2 weeks of engineering.
