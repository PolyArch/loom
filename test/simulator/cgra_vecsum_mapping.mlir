// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecsum %t.dir/vecsum/main_func.dfg.mlir %t.dir/vecsum.dfg.report.json %t.dir/dfg-summary.csv
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecsum.dfg.report.json --mapping-artifact %t.dir/mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecsum.cgra.report.json
// RUN: FileCheck %s --check-prefix=CGRA < %t.dir/vecsum.cgra.report.json
// RUN: loom-sim-cycle-summary --dfg-report %t.dir/vecsum.dfg.report.json --cgra-report %t.dir/vecsum.cgra.report.json --output %t.dir/summary.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// MAPPING-DAG: "schedule": "spatial"
// MAPPING-DAG: "resource_kind": "fabric.mem.load"
// MAPPING-DAG: "status": "fail"
// MAPPING-DAG: "routed_edges": 0
// MAPPING-DAG: "unrouted_edges": 6
// MAPPING-DAG: "config_records": 0

// CGRA-DAG: "kind": "cgra_sim_report"
// CGRA-DAG: "workload": "vecsum"
// CGRA-DAG: "hardware_artifact": "
// CGRA-DAG: "mapping_id": "vecsum__g_t_vecsum_red_0_0__shared_reduction_adg"
// CGRA-DAG: "status": "blocked"
// CGRA-DAG: "fidelity_level": "mapping_constraint_estimate"
// CGRA-DAG: "operation_semantics_source": "loom.sim.operation_semantics.v1"
// CGRA-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// CGRA-DAG: "difference_classification": "unsupported_scope"
// CGRA-DAG: "dfg_cycles": 579
// CGRA-DAG: "modeled_lower_bound_cycles": 579
// CGRA-DAG: "hardware_bound_classification": "unsupported_scope"
// CGRA-DAG: "performance_delta_cycles": 0
// CGRA-DAG: "route_latency_cycles": 0
// CGRA-DAG: "route_segments": 0
// CGRA-DAG: "memory_latency_cycles": 0
// CGRA-DAG: "temporal_penalty_cycles": 0
// CGRA-DAG: "hardware_aware_cycles": 579
// CGRA-DAG: "config_records": 0
// CGRA-DAG: "cycle_breakdown"
// CGRA-DAG: "category": "route_latency"
// CGRA-DAG: "evidence": "mapping.route_segments"
// CGRA-DAG: "category": "memory_latency"
// CGRA-DAG: "evidence": "fabric.mem placement"
// CGRA-DAG: "category": "temporal_conflict"
// CGRA-DAG: "evidence": "placement schedule"
// CGRA-DAG: "unmodeled_constraints"
// CGRA-DAG: "explicit_fabric_route_paths"
// CGRA-DAG: "fifo_latency"
// CGRA-DAG: "cache_behavior"
// CGRA-DAG: "scratchpad_bank_conflicts"
// CGRA-DAG: "coherence_consistency"
// CGRA-DAG: "first_principles_checks"
// CGRA-DAG: "cgra_not_more_optimistic_than_dfg"
// CGRA-DAG: "delta_explained_by_modeled_constraints"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-NEXT: vecsum,579,,blocked
