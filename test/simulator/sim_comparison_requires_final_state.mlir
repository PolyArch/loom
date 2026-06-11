// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","runtime_input_identity":"toy::input","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","metric_definition":"optimistic_pipeline_latency_throughput_sum","optimistic_cycles":3,"final_memory_state":{}}' > %t.no_outputs.dfg.json
// RUN: echo '{"schema_version":1,"kind":"cgra_sim_report","workload":"toy","runtime_input_identity":"toy::input","status":"pass","metric_definition":"mapping_constraint_estimate","hardware_aware_cycles":3,"performance_delta_cycles":0,"functional_state_source":"carried_from_dfg_sim_report","final_memory_state":{}}' > %t.no_outputs.cgra.json
// RUN: not timeout 30s bash %S/run_sim_comparison_report.sh --dfg-report %t.no_outputs.dfg.json --cgra-report %t.no_outputs.cgra.json --output %t.no_outputs.report.json
// RUN: FileCheck %s --check-prefix=MISSING-OUTPUTS < %t.no_outputs.report.json

// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","runtime_input_identity":"toy::input","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","metric_definition":"optimistic_pipeline_latency_throughput_sum","optimistic_cycles":3,"final_outputs":[]}' > %t.no_memory.dfg.json
// RUN: echo '{"schema_version":1,"kind":"cgra_sim_report","workload":"toy","runtime_input_identity":"toy::input","status":"pass","metric_definition":"mapping_constraint_estimate","hardware_aware_cycles":3,"performance_delta_cycles":0,"functional_state_source":"carried_from_dfg_sim_report","final_outputs":[]}' > %t.no_memory.cgra.json
// RUN: not timeout 30s bash %S/run_sim_comparison_report.sh --dfg-report %t.no_memory.dfg.json --cgra-report %t.no_memory.cgra.json --output %t.no_memory.report.json
// RUN: FileCheck %s --check-prefix=MISSING-MEMORY < %t.no_memory.report.json

// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","runtime_input_identity":"toy::input","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","metric_definition":"optimistic_pipeline_latency_throughput_sum","optimistic_cycles":3,"final_outputs":["7"],"final_memory_state":{"visible":[]}}' > %t.blocked_cgra.dfg.json
// RUN: echo '{"schema_version":1,"kind":"cgra_sim_report","workload":"toy","runtime_input_identity":"toy::input","status":"blocked","metric_definition":"mapping_constraint_estimate","difference_classification":"unsupported_scope","hardware_aware_cycles":3,"performance_delta_cycles":0,"final_outputs":["7"],"final_memory_state":{"visible":[]},"diagnostics":["mapping artifact status fail blocks CGRA-sim"]}' > %t.blocked_cgra.cgra.json
// RUN: not timeout 30s bash %S/run_sim_comparison_report.sh --dfg-report %t.blocked_cgra.dfg.json --cgra-report %t.blocked_cgra.cgra.json --output %t.blocked_cgra.report.json
// RUN: FileCheck %s --check-prefix=BLOCKED-CGRA < %t.blocked_cgra.report.json

// MISSING-OUTPUTS-DAG: "functional_comparison_status": "blocked"
// MISSING-OUTPUTS-DAG: "memory_comparison_status": "pass"
// MISSING-OUTPUTS-DAG: "difference_classification": "unsupported_scope"
// MISSING-OUTPUTS-DAG: "status": "blocked"

// MISSING-MEMORY-DAG: "functional_comparison_status": "pass"
// MISSING-MEMORY-DAG: "memory_comparison_status": "blocked"
// MISSING-MEMORY-DAG: "difference_classification": "unsupported_scope"
// MISSING-MEMORY-DAG: "status": "blocked"

// BLOCKED-CGRA-DAG: "functional_comparison_status": "pass"
// BLOCKED-CGRA-DAG: "memory_comparison_status": "pass"
// BLOCKED-CGRA-DAG: "performance_comparison_status": "blocked"
// BLOCKED-CGRA-DAG: "difference_classification": "unsupported_scope"
// BLOCKED-CGRA-DAG: "status": "blocked"
