// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":3,"final_outputs":[],"final_memory_state":{}}' > %t.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"fail","placed_records":2,"routed_edges":0,"unrouted_edges":1,"unplaced_records":0,"config_records":0,"placements":[{"software":"arith.addi#0","operation":"arith.addi","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#0","schedule":"spatial"},{"software":"arith.muli#0","operation":"arith.muli","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#1","schedule":"spatial"}],"routes":[],"config_bitstream":[],"diagnostics":["unrouted software edges lack Fabric ADG connectivity"]}' > %t.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dfg.json --mapping-artifact %t.mapping.json --output %t.cgra.json
// RUN: FileCheck %s < %t.cgra.json

// CHECK-DAG: "kind": "cgra_sim_report"
// CHECK-DAG: "workload": "toy"
// CHECK-DAG: "hardware": "toy_adg"
// CHECK-DAG: "mapping_id": "toy__toy_adg"
// CHECK-DAG: "status": "blocked"
// CHECK-DAG: "difference_classification": "unsupported_scope"
// CHECK-DAG: "hardware_bound_classification": "unsupported_scope"
// CHECK-DAG: "dfg_cycles": 3
// CHECK-DAG: "hardware_aware_cycles": 3
// CHECK-DAG: "routed_edges": 0
// CHECK-DAG: "route_segments": 0
// CHECK-DAG: mapping artifact status fail blocks CGRA-sim
