// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":3,"final_outputs":[],"final_memory_state":{}}' > %t.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","routed_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dfg.json --mapping-artifact %t.mapping.json --output %t.cgra.json
// RUN: FileCheck %s < %t.cgra.json

// CHECK-DAG: "difference_classification": "no_modeled_hardware_constraints"
// CHECK-DAG: "dfg_cycles": 3
// CHECK-DAG: "performance_delta_cycles": 0
// CHECK-DAG: "modeled_lower_bound_cycles": 3
// CHECK-DAG: "hardware_aware_cycles": 3
