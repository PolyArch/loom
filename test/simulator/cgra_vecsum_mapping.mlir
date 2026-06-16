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
// MAPPING-DAG: "status": "pass"
// MAPPING-DAG: "config_id": "loom.default"
// MAPPING-DAG: "config_fingerprint": "{{[0-9a-f]+}}"
// MAPPING-DAG: "component_config_view": "pnr.mapping.v1"
// MAPPING-DAG: "component_config_fingerprint": "{{[0-9a-f]+}}"
// MAPPING-DAG: "routed_edges": 6
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "config_records": 137

// CGRA-DAG: "kind": "cgra_sim_report"
// CGRA-DAG: "workload": "vecsum"
// CGRA-DAG: "hardware_artifact": "
// CGRA-DAG: "mapping_id": "vecsum__g_t_vecsum_red_0_0__shared_reduction_adg"
// CGRA-DAG: "status": "pass"
// CGRA-DAG: "config_id": "loom.default"
// CGRA-DAG: "config_fingerprint": "{{[0-9a-f]+}}"
// CGRA-DAG: "component_config_view": "cgra.sim.v1"
// CGRA-DAG: "component_config_fingerprint": "{{[0-9a-f]+}}"
// CGRA-DAG: "fidelity_level": "mapping_constraint_estimate"
// CGRA-DAG: "operation_semantics_source": "loom.sim.operation_semantics.v1"
// CGRA-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// CGRA-DAG: "difference_classification": "expected_hardware_constraint"
// CGRA-DAG: "dfg_cycles": 579
// CGRA-DAG: "modeled_lower_bound_cycles": 607
// CGRA-DAG: "hardware_bound_classification": "within_modeled_bounds"
// CGRA-DAG: "performance_delta_cycles": 28
// CGRA-DAG: "route_latency_cycles": 24
// CGRA-DAG: "route_segments": 24
// CGRA-DAG: "memory_latency_cycles": 4
// CGRA-DAG: "temporal_penalty_cycles": 0
// CGRA-DAG: "hardware_aware_cycles": 607
// CGRA-DAG: "config_records": 137
// CGRA-DAG: "functional_state_source": "carried_from_dfg_sim_report"
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
// SUMMARY-NEXT: vecsum,579,607,pass,"DFG-sim and CGRA-sim reports available; CGRA-sim includes mapping route, memory, and temporal penalties"
